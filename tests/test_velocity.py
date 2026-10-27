"""Velocity task invariants, command boundaries, gradients and export contracts."""

import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path

import torch

from src.tasks.supermimic.config.rl_cfg import OnlineConfig
from src.tasks.supermimic.config.velocity_cfg import VelocityConfig
from src.tasks.supermimic.data.motion_library import MotionLibrary
from src.tasks.supermimic.data.replay_buffer import ReplayBuffer
from src.tasks.supermimic.mdp.state import observe, quat_apply, quat_mul, rotvec_to_quat
from src.tasks.supermimic.mdp.velocity import (
  measured_velocity,
  standing_command,
  velocity_failure,
  velocity_reward,
)
from src.tasks.supermimic.mdp.velocity_commands import VelocityCommands
from src.tasks.supermimic.models.kinematics import G1Kinematics, robot_metadata
from src.tasks.supermimic.models.policy import MotionActor
from src.tasks.supermimic.models.velocity_policy import (
  VelocityActor,
  VelocityCritic,
  VelocityEncoder,
)
from src.tasks.supermimic.models.world_model import WorldModel
from src.tasks.supermimic.rl.imagination import frozen_parameters
from src.tasks.supermimic.rl.velocity_evaluation import step_commands
from src.tasks.supermimic.rl.velocity_imagination import (
  VelocityKernels,
  imagine_velocity,
)
from src.tasks.supermimic.rl.velocity_inference import load_velocity_actor
from tests.supermimic_fixtures import make_motions


class VelocityTests(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    torch.set_num_threads(2)
    cls.metadata = robot_metadata()
    cls.kinematics = G1Kinematics(cls.metadata)
    cls.folder = tempfile.TemporaryDirectory()
    cls.manifest, _ = make_motions(Path(cls.folder.name), cls.metadata)
    cls.library = MotionLibrary(
      cls.manifest, cls.kinematics.joint_names, cls.kinematics.body_names
    )

  @classmethod
  def tearDownClass(cls):
    cls.folder.cleanup()

  def state(self, count=2):
    state = torch.zeros(count, 71)
    state[:, 2], state[:, 3] = 0.8, 1.0
    state[:, 7:36] = self.metadata["default_q"].float()
    return state

  def starts(self, commands):
    state = self.state()
    context = torch.zeros(2, 33)
    context[:, 0] = 1
    history = (
      observe(state, torch.zeros(2, 29), self.metadata["default_q"].float(), context)[
        :, None
      ]
      .expand(-1, 10, -1)
      .clone()
    )
    starts = commands.empty(2, "cpu")
    commands.resample(
      starts,
      torch.arange(2),
      torch.Generator().manual_seed(1),
      0,
      reference_only=True,
    )
    return starts | {"state": state, "history": history, "context": context}

  def models(self):
    actor = MotionActor(self.metadata["default_q"], (24, 16))
    encoder = VelocityEncoder()
    world = WorldModel(self.metadata, (24, 24), "analytic")
    critic = VelocityCritic(self.metadata["default_q"], hidden_dims=(16, 8))
    return actor, encoder, world, critic

  def test_heading_invariance_reward_and_critic(self):
    state = self.state()
    state[:, 36:42] = torch.tensor([0.4, 0.2, 0.1, 0.1, 0.2, 0.5])
    state[:, 3:7] = rotvec_to_quat(torch.tensor([[0.2, 0.3, 0.1]])).expand(2, -1)
    rotation = rotvec_to_quat(torch.tensor([[0.0, 0.0, 1.3]])).expand(2, -1)
    rotated = state.clone()
    rotated[:, :3] = quat_apply(rotation, state[:, :3]) + torch.tensor([4.0, -3.0, 0.0])
    rotated[:, 3:7] = quat_mul(rotation, state[:, 3:7])
    for sl in (slice(36, 39), slice(39, 42)):
      rotated[:, sl] = quat_apply(rotation, state[:, sl])
    command = torch.tensor([[0.5, 0.0, 0.5]]).expand(2, -1)
    action = torch.zeros(2, 29)
    cfg = VelocityConfig()

    def reward(s):
      return velocity_reward(
        s,
        self.kinematics(s),
        command,
        action,
        action,
        self.metadata["joint_limits"],
        self.metadata["default_q"],
        cfg,
      )[0]

    torch.testing.assert_close(measured_velocity(state), measured_velocity(rotated))
    torch.testing.assert_close(reward(state), reward(rotated))
    critic = VelocityCritic(self.metadata["default_q"])
    context = torch.zeros(2, 33)
    torch.testing.assert_close(
      critic.features(state, command, action, context),
      critic.features(rotated, command, action, context),
    )

  def test_reward_denominators_and_torso_forward_lean(self):
    state = self.state()
    action = torch.zeros(2, 29)
    command = torch.tensor([[0.5, 0.0, 0.5]]).expand(2, -1)
    _, terms = velocity_reward(
      state,
      self.kinematics(state),
      command,
      action,
      action,
      self.metadata["joint_limits"],
      self.metadata["default_q"],
      VelocityConfig(),
    )
    torch.testing.assert_close(
      terms["linear_tracking"],
      torch.full((2,), 0.04 * torch.exp(torch.tensor(-1.0)).item()),
    )
    torch.testing.assert_close(
      terms["yaw_tracking"],
      torch.full((2,), 0.04 * torch.exp(torch.tensor(-0.5)).item()),
    )
    state[:, 3:7] = rotvec_to_quat(torch.tensor([[0.0, 0.4, 0.0]])).expand(2, -1)
    _, terms = velocity_reward(
      state,
      self.kinematics(state),
      command,
      action,
      action,
      self.metadata["joint_limits"],
      self.metadata["default_q"],
      VelocityConfig(),
    )
    self.assertLess(float(terms["torso_sideways"].abs().max()), 1e-6)
    self.assertFalse(
      standing_command(torch.tensor([[0.0, 0.0, 1.0]]), VelocityConfig()).item()
    )

  def test_hard_and_soft_fall_thresholds(self):
    state = self.state()
    state[:, 2] = torch.tensor([0.34, 0.36])
    failure = velocity_failure(state, VelocityConfig())
    self.assertEqual(failure["height_failure"].tolist(), [True, False])
    self.assertGreater(float(failure["termination_probability"][0]), 0.5)
    self.assertLess(float(failure["termination_probability"][1]), 0.5)
    state[:, 2] = 0.8
    state[:, 3:7] = rotvec_to_quat(
      torch.deg2rad(torch.tensor([[69.0, 0.0, 0.0], [71.0, 0.0, 0.0]]))
    )
    self.assertEqual(
      velocity_failure(state, VelocityConfig())["tilt_failure"].tolist(),
      [False, True],
    )

  def test_command_sequence_matches_simulation_across_resampling_and_clip_end(self):
    commands = VelocityCommands(
      replace(VelocityConfig(), resample_steps=3), self.library
    )
    starts = commands.empty(2, "cpu")
    commands.resample(
      starts,
      torch.arange(2),
      torch.Generator().manual_seed(11),
      0,
      reference_only=True,
    )
    starts["reference_frame"][:] = self.library.lengths[starts["reference_motion"]] - 1
    commands._read_reference(starts)
    rng = torch.Generator().manual_seed(12)
    expected = commands.sequence(starts, 9, rng, 119998)
    actual = {k: v.clone() for k, v in starts.items()}
    rng.manual_seed(12)
    for step in range(9):
      commands.advance(actual, rng, 119999 + step)
      for key, value in actual.items():
        torch.testing.assert_close(value, expected[key][:, step + 1])
    # Fixture clips have constant, different velocities; padding cannot mix them.
    for offset, length, speed in zip(
      self.library.offsets, self.library.lengths, (0.05, 0.1)
    ):
      torch.testing.assert_close(
        commands.reference_commands[offset : offset + length, 0],
        torch.full((int(length),), speed),
      )

  def test_random_command_does_not_need_reference_and_curriculum_is_bounded(self):
    cfg = replace(VelocityConfig(), reference_probability=0.0, standing_probability=0.0)
    commands = VelocityCommands(cfg)
    state = commands.empty(100, "cpu")
    rng = torch.Generator().manual_seed(2)
    commands.resample(state, torch.arange(100), rng, cfg.curriculum_steps)
    self.assertTrue((state["reference_motion"] == -1).all())
    self.assertFalse(state["command_source"].any())
    self.assertTrue((state["command"][:, 2].abs() > 1.0).any())
    before = state["command"].clone()
    commands.advance(state, rng, cfg.curriculum_steps + 1)
    torch.testing.assert_close(state["command"], before)

  def test_velocity_gradients_freeze_decoder_world_critic_and_keep_bootstrap(self):
    torch.manual_seed(4)
    actor, encoder, world, critic = self.models()
    commands = VelocityCommands(VelocityConfig(), self.library)
    starts = self.starts(commands)
    snapshots = [
      {k: v.clone() for k, v in module.state_dict().items()}
      for module in (actor, world, critic)
    ]
    with frozen_parameters(actor, world, critic):
      trace = imagine_velocity(
        actor,
        encoder,
        world,
        critic,
        commands,
        starts,
        horizon=3,
        sensor_noise=False,
        generator=torch.Generator().manual_seed(5),
      )
      bootstrap_grad = torch.autograd.grad(
        trace["bootstrap_objective"],
        list(encoder.parameters()),
        retain_graph=True,
      )
      self.assertGreater(sum(float(g.abs().sum()) for g in bootstrap_grad), 0.0)
      (-trace["objective"] + 0.1 * trace["alignment_loss"]).backward()
    self.assertTrue(
      all(
        p.grad is None for module in (actor, world, critic) for p in module.parameters()
      )
    )
    self.assertGreater(
      sum(float(p.grad.abs().sum()) for p in encoder.parameters()), 0.0
    )
    for module, snapshot in zip((actor, world, critic), snapshots):
      for key, value in module.state_dict().items():
        torch.testing.assert_close(value, snapshot[key], rtol=0, atol=0)
    self.assertTrue(trace["states"].requires_grad)
    self.assertEqual(trace["history"].shape, (2, 10, 93))

  def test_random_rollouts_have_no_alignment_and_compiled_kernels_match(self):
    actor, encoder, world, critic = self.models()
    commands = VelocityCommands(VelocityConfig())
    starts = self.starts(commands)
    with frozen_parameters(actor, world, critic):
      eager = imagine_velocity(
        actor,
        encoder,
        world,
        critic,
        commands,
        starts,
        horizon=2,
        sensor_noise=False,
        generator=torch.Generator().manual_seed(5),
      )
      kernels = VelocityKernels(
        actor, encoder, world, commands.cfg, enabled=True, backend="eager"
      )
      compiled = imagine_velocity(
        actor,
        encoder,
        world,
        critic,
        commands,
        starts,
        horizon=2,
        sensor_noise=False,
        generator=torch.Generator().manual_seed(5),
        kernels=kernels,
      )
    self.assertEqual(float(eager["alignment_loss"].detach()), 0.0)
    torch.testing.assert_close(eager["objective"], compiled["objective"])

  def test_standalone_export_round_trip_without_motion_encoder(self):
    actor, encoder, _, _ = self.models()
    policy = VelocityActor(actor, encoder).eval()
    cfg = OnlineConfig(actor_dims=(24, 16), amp="none", compile=False)
    saved = {
      "kind": "supermimic_velocity_actor",
      "schema_version": 1,
      "observation_schema": "proprioceptive_93_v1",
      "history_length": 10,
      "config": asdict(cfg),
      "velocity_config": asdict(VelocityConfig()),
      "robot_metadata": self.metadata,
      "actor": policy.state_dict(),
    }
    path = Path(self.folder.name) / "velocity.pt"
    torch.save(saved, path)
    loaded, _ = load_velocity_actor(path)
    history, command = torch.randn(2, 10, 93), torch.randn(2, 3)
    torch.testing.assert_close(
      policy(history, command), loaded(history, command), rtol=0, atol=0
    )
    self.assertFalse(any(key.startswith("encoder.0") for key in saved["actor"]))
    self.assertNotIn("manifest_sha256", saved)
    with self.assertRaises(ValueError):
      VelocityActor(
        MotionActor(self.metadata["default_q"], (8,), motion_anchor_ori_b=True),
        encoder,
      )

  def test_velocity_replay_history_boundary_and_schema_roundtrip(self):
    buffer = ReplayBuffer(2, 20, task="velocity")
    commands = VelocityCommands(VelocityConfig())
    for step in range(14):
      state = self.state()
      obs = torch.full((2, 93), float(step))
      command_state = commands.empty(2, "cpu")
      transition = {
        "state": state,
        "next_state": state,
        "action": torch.zeros(2, 29),
        "previous_action": torch.zeros(2, 29),
        "observation": obs,
        "next_observation": obs + 1,
        "clean_observation": obs,
        "next_clean_observation": obs + 1,
        "context": torch.zeros(2, 33),
        "reward": torch.zeros(2),
        "episode_id": torch.full((2,), int(step >= 7)),
        "episode_step": torch.full((2,), step % 7),
        "terminated": torch.zeros(2, dtype=torch.bool),
        "time_out": torch.full((2,), step == 6),
      }
      buffer.append(
        transition | command_state | {"next_command": command_state["command"]}
      )
    loaded = ReplayBuffer.from_state_dict(buffer.state_dict())
    self.assertEqual(loaded.task, "velocity")
    self.assertNotIn("motion_id", loaded.data)
    starts = loaded.sample_starts(8)
    for history, episode in zip(starts["history"], starts["episode_id"]):
      self.assertTrue((history >= 7 if episode else history < 7).all())
    batch = loaded.sample_windows(8)
    self.assertTrue(torch.isfinite(batch["state"]).all())

  def test_full_grid_has_672_nonzero_conditions(self):
    grid = step_commands(True)
    self.assertEqual(len(grid), 672)
    self.assertTrue(all(any(v != 0 for v in command) for _, command in grid))


if __name__ == "__main__":
  unittest.main()
