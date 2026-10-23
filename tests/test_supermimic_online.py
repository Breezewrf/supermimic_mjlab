"""Feature, BPTT, boundary, optimizer-isolation and real online-loop tests."""

import copy
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch

from src.tasks.supermimic.config.rl_cfg import OnlineConfig
from src.tasks.supermimic.config.task_cfg import (
  TRACKED_BODIES,
  FoundationConfig,
)
from src.tasks.supermimic.data.motion_library import MotionLibrary
from src.tasks.supermimic.mdp.state import observe, quat_apply, quat_mul, rotvec_to_quat
from src.tasks.supermimic.models.kinematics import robot_metadata
from src.tasks.supermimic.models.policy import (
  MotionActor,
  MotionCritic,
  reference_features,
)
from src.tasks.supermimic.models.world_model import WorldModel
from src.tasks.supermimic.rl.imagination import (
  critic_loss,
  frozen_parameters,
  gae_targets,
  imagine,
)
from src.tasks.supermimic.rl.online_training import OnlineTrainer
from src.tasks.supermimic.rl.world_model_training import world_model_loss
from tests.supermimic_fixtures import make_motions

ROOT = Path(__file__).resolve().parents[1]
DEVICE = os.environ.get(
  "SUPERMIMIC_TEST_DEVICE", "cuda:0" if torch.cuda.is_available() else ""
)


def make_starts(library, default_q, count=2, *, frame_id=0):
  motion = torch.zeros(count, dtype=torch.long, device=library.device)
  frame = torch.full_like(motion, frame_id)
  root = library.root_state(motion, frame)
  ref = library.get_frame(motion, frame)
  state = torch.cat((root[:, :7], ref["joint_pos"], root[:, 7:], ref["joint_vel"]), -1)
  context = torch.zeros(count, 33, device=library.device)
  context[:, 0] = 0.8
  obs = observe(
    state, torch.zeros(count, 29, device=library.device), default_q.to(state), context
  )
  return {
    "state": state,
    "history": obs[:, None].repeat(1, 10, 1),
    "context": context,
    "motion_id": motion,
    "frame_id": frame,
    "episode_step": frame.clone(),
  }


class OnlineConfigTests(unittest.TestCase):
  def test_reference_dimension_requires_a_supported_layout(self):
    for width in (730, 2680):
      self.assertEqual(OnlineConfig(reference_dim=width).reference_dim, width)
    with self.assertRaisesRegex(ValueError, "reference_dim"):
      OnlineConfig(reference_dim=73)

  def test_wm_only_iters_accepts_zero_and_positive_but_rejects_negative(self):
    self.assertEqual(OnlineConfig().wm_only_iters, 0)
    self.assertEqual(OnlineConfig(wm_only_iters=500).wm_only_iters, 500)
    with self.assertRaisesRegex(ValueError, "wm_only_iters"):
      OnlineConfig(wm_only_iters=-1)


class ImaginationTests(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    torch.set_num_threads(4)
    cls.metadata = robot_metadata()
    folder = tempfile.TemporaryDirectory()
    cls.addClassCleanup(folder.cleanup)
    manifest, _ = make_motions(Path(folder.name), cls.metadata)
    cls.library = MotionLibrary(
      manifest,
      tuple(cls.metadata["joint_names"]),
      TRACKED_BODIES,
      clips=("fixture_walk",),
    )

  def models(self):
    actor = MotionActor(self.metadata["default_q"], (16, 16))
    critic = MotionCritic(self.metadata["default_q"], (16,))
    world = WorldModel(self.metadata, (16,))
    torch.nn.init.normal_(world.dynamics[-1].weight, std=0.0001)
    return actor, world, critic

  def test_features_are_se2_invariant_and_quaternion_sign_independent(self):
    starts = make_starts(self.library, self.metadata["default_q"])
    ref = self.library.get_future(starts["motion_id"], starts["frame_id"])
    actor, _, critic = self.models()
    rotation = rotvec_to_quat(torch.tensor([[0.0, 0.0, 0.7], [0.0, 0.0, -0.4]]))
    translation = torch.tensor([3.0, -2.0, 0.0])
    transformed = {key: value.clone() for key, value in ref.items()}
    transformed["body_pos_w"] = (
      quat_apply(rotation[:, None, None], ref["body_pos_w"]) + translation
    )
    transformed["body_quat_w"] = -quat_mul(rotation[:, None, None], ref["body_quat_w"])
    for key in ("body_lin_vel_w", "body_ang_vel_w"):
      transformed[key] = quat_apply(rotation[:, None, None], ref[key])
    torch.testing.assert_close(
      reference_features(ref, actor.default_q),
      reference_features(transformed, actor.default_q),
      atol=5e-6,
      rtol=2e-5,
    )
    state = starts["state"].clone()
    state[:, :3] = quat_apply(rotation, state[:, :3]) + translation
    state[:, 3:7] = -quat_mul(rotation, state[:, 3:7])
    for offset in (36, 39):
      state[:, offset : offset + 3] = quat_apply(
        rotation, state[:, offset : offset + 3]
      )
    torch.testing.assert_close(
      critic.features(starts["state"], ref, starts["context"]),
      critic.features(state, transformed, starts["context"]),
      atol=5e-6,
      rtol=2e-5,
    )
    torch.testing.assert_close(
      actor(starts["history"], ref),
      actor(starts["history"], transformed),
      atol=1e-6,
      rtol=1e-4,
    )

  def test_32_step_state_gradient_reaches_actor_with_frozen_world(self):
    actor, world, critic = self.models()
    starts = make_starts(self.library, actor.default_q)
    with frozen_parameters(world, critic):
      trajectory = imagine(
        actor, world, critic, self.library, starts, sensor_noise=False
      )
      # Use physical state alone: this cannot pass via the action regularizer.
      trajectory["states"][:, -1].square().mean().backward()
      self.assertTrue(
        all(p.grad is None for m in (world, critic) for p in m.parameters())
      )
    self.assertGreater(
      sum(float(p.grad.abs().sum()) for p in actor.parameters() if p.grad is not None),
      0.0,
    )
    self.assertTrue(torch.isfinite(trajectory["states"]).all())
    self.assertTrue(all(p.requires_grad for p in world.parameters()))
    torch.testing.assert_close(
      trajectory["history"][:, -1, -29:], trajectory["actions"][-1]
    )

  def test_action_feedback_through_observation_history_is_differentiable(self):
    actor, world, critic = self.models()
    with frozen_parameters(world, critic):
      trace = imagine(
        actor,
        world,
        critic,
        self.library,
        make_starts(self.library, actor.default_q),
        horizon=3,
        sensor_noise=False,
      )
      feedback = torch.autograd.grad(trace["actions"][-1].sum(), trace["actions"][0])[0]
    self.assertGreater(float(feedback.abs().sum()), 0.0)
    self.assertTrue(torch.isfinite(feedback).all())

  def test_imagination_rejects_any_start_without_full_reference_horizon(self):
    actor, world, critic = self.models()
    starts = make_starts(self.library, actor.default_q)
    # A mixed batch must reject even one insufficient start (only 3 successors).
    starts["frame_id"][1] = int(self.library.lengths[0]) - 4
    with self.assertRaisesRegex(ValueError, "full reference horizon"):
      imagine(actor, world, critic, self.library, starts, horizon=4, sensor_noise=False)

  def test_last_reference_frame_preserves_bootstrap_unless_physical_failure(self):
    actor, world, critic = self.models()
    horizon, gamma = 4, 0.9
    starts = make_starts(
      self.library,
      actor.default_q,
      frame_id=int(self.library.lengths[0]) - 1 - horizon,
    )
    with torch.no_grad():
      for parameter in critic.parameters():
        parameter.zero_()
      critic.value[-1].bias.fill_(5.0)
    original_heads = world.task_heads
    for failure in (0.0, 1.0):

      def constant_heads(*args, failure=failure):
        result = original_heads(*args)
        result["reward"] = torch.ones_like(result["reward"])
        result["termination_probability"] = torch.full_like(result["reward"], failure)
        return result

      with (
        self.subTest(physical_failure=failure),
        frozen_parameters(world, critic),
        patch.object(world, "task_heads", side_effect=constant_heads),
      ):
        trace = imagine(
          actor,
          world,
          critic,
          self.library,
          starts,
          horizon=horizon,
          gamma=gamma,
          sensor_noise=False,
        )
        self.assertTrue(trace["mask"].all())
        torch.testing.assert_close(
          trace["continuation"], torch.full((2, horizon), 1 - failure)
        )
        bootstrap = gamma**horizon * (1 - failure) ** horizon * 5
        expected_reward = sum((gamma * (1 - failure)) ** t for t in range(horizon))
        torch.testing.assert_close(
          trace["bootstrap_objective"], torch.tensor(bootstrap)
        )
        torch.testing.assert_close(
          trace["objective"], torch.tensor(expected_reward + bootstrap)
        )
        targets = gae_targets(
          trace["rewards"], trace["continuation"], trace["values"], gamma, lam=1.0
        )
        expected = torch.tensor(
          [
            sum((gamma * (1 - failure)) ** t for t in range(steps))
            + (gamma * (1 - failure)) ** steps * 5
            for steps in range(horizon, 0, -1)
          ]
        )
        torch.testing.assert_close(targets, expected.expand(2, -1))

  def test_collection_timeout_does_not_cut_imagination_or_gae_bootstrap(self):
    actor, world, critic = self.models()
    with torch.no_grad():
      for parameter in critic.parameters():
        parameter.zero_()
      critic.value[-1].bias.fill_(5.0)
    original_heads = world.task_heads

    def no_failure_or_reward(*args):
      result = original_heads(*args)
      result["reward"] = torch.zeros_like(result["reward"])
      result["termination_probability"] = torch.zeros_like(result["reward"])
      return result

    starts = make_starts(self.library, actor.default_q)
    traces = []
    with (
      frozen_parameters(world, critic),
      patch.object(world, "task_heads", side_effect=no_failure_or_reward),
    ):
      for episode_step in (0, 499):
        starts["episode_step"].fill_(episode_step)
        traces.append(
          imagine(actor, world, critic, self.library, starts, sensor_noise=False)
        )
    for trace in traces:
      self.assertTrue(trace["mask"].all())
      torch.testing.assert_close(trace["continuation"], torch.ones(2, 32))
      torch.testing.assert_close(trace["survival"], torch.ones(2))
      torch.testing.assert_close(trace["objective"], torch.tensor(0.99**32 * 5))
      target = gae_targets(
        trace["rewards"], trace["continuation"], trace["values"], lam=1.0
      )
      expected = 5 * 0.99 ** torch.arange(32, 0, -1).float()
      torch.testing.assert_close(target, expected.expand(2, -1))
    for key in ("states", "values", "history", "objective"):
      torch.testing.assert_close(traces[0][key], traces[1][key])

  def test_gae_uses_fixed_detached_targets_and_soft_continuation(self):
    rewards = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
    continuation = torch.tensor([[0.5, 0.0, 1.0]], requires_grad=True)
    values = torch.tensor([[0.1, 0.2, 0.3, 0.4]], requires_grad=True)
    actual = gae_targets(rewards, continuation, values, gamma=0.9, lam=1.0)
    torch.testing.assert_close(actual, torch.tensor([[1.9, 2.0, 3.36]]))
    self.assertFalse(actual.requires_grad)

  def test_terminal_bootstrap_keeps_its_state_gradient(self):
    actor, world, critic = self.models()
    starts = make_starts(
      self.library, actor.default_q, frame_id=int(self.library.lengths[0]) - 3
    )
    original_heads = world.task_heads

    def zero_reward(*args):
      result = original_heads(*args)
      result["reward"] = torch.zeros_like(result["reward"])
      result["termination_probability"] = torch.full_like(result["reward"], 0.5)
      return result

    with (
      frozen_parameters(world, critic),
      patch.object(world, "task_heads", side_effect=zero_reward),
    ):
      trace = imagine(
        actor,
        world,
        critic,
        self.library,
        starts,
        horizon=2,
        gamma=0.9,
        sensor_noise=False,
      )
      trace["objective"].backward()
    self.assertGreater(
      sum(float(p.grad.abs().sum()) for p in actor.parameters() if p.grad is not None),
      0.0,
    )
    self.assertTrue(all(p.grad is None for p in critic.parameters()))
    torch.testing.assert_close(
      trace["objective"].detach(), 0.9**2 * 0.5**2 * trace["values"][:, -1].mean()
    )

  def test_sensor_noise_is_added_only_to_the_new_observation(self):
    actor, world, critic = self.models()
    starts = make_starts(self.library, actor.default_q)
    with frozen_parameters(world, critic):
      trace = imagine(
        actor,
        world,
        critic,
        self.library,
        starts,
        horizon=1,
        generator=torch.Generator().manual_seed(5),
      )
    torch.testing.assert_close(trace["history"][:, :-1], starts["history"][:, 1:])
    clean = observe(
      trace["states"][:, -1], trace["actions"][-1], actor.default_q, starts["context"]
    )
    error = (trace["history"][:, -1] - clean).abs()
    amplitude = error.new_tensor(
      [0.01] * 29 + [0.5] * 29 + [0.2] * 3 + [0.05] * 3 + [0.0] * 29
    )
    self.assertTrue((error <= amplitude + 1e-6).all())
    self.assertGreater(float(error[:, :64].sum()), 0.0)

  def test_analytic_head_errors_are_diagnostics_in_online_loss(self):
    state = torch.zeros(2, 2, 71)
    state[..., 3] = 1
    pred = {
      "next_state": state,
      "clean_observation": torch.zeros(2, 2, 93),
      "reward": torch.zeros(2, 2),
      "termination_logit": torch.zeros(2, 2),
    }
    batch = {
      "next_state": state,
      "next_clean_observation": torch.ones(2, 2, 93),
      "reward": torch.ones(2, 2),
      "terminated": torch.zeros(2, 2, dtype=torch.bool),
      "mask": torch.ones(2, 2, dtype=torch.bool),
    }
    online, parts = world_model_loss(pred, batch, analytic_losses=False)
    torch.testing.assert_close(online, parts["state"] + parts["termination"])
    diagnostic, _ = world_model_loss(pred, batch)
    self.assertGreater(float(diagnostic), float(online) + 1.0)


class TimeReductionTests(unittest.TestCase):
  def test_world_model_sums_time_and_averages_windows_with_padding(self):
    state = torch.zeros(2, 3, 71)
    state[..., 3] = 1
    predicted_state = state.clone()
    predicted_state[..., :3] = 0.3
    logits = torch.zeros(2, 3, requires_grad=True)
    prediction = {
      "next_state": predicted_state,
      "clean_observation": torch.ones(2, 3, 93),
      "reward": torch.full((2, 3), 0.1),
      "termination_logit": logits,
    }
    batch = {
      "next_state": state,
      "next_clean_observation": torch.zeros(2, 3, 93),
      "reward": torch.zeros(2, 3),
      "terminated": torch.zeros(2, 3, dtype=torch.bool),
      "mask": torch.tensor([[True, True, True], [True, False, False]]),
    }
    # Invalid labels must be excluded before any arithmetic (including NaNs).
    for name in ("next_state", "next_clean_observation", "reward"):
      batch[name][~batch["mask"]] = float("nan")
    loss, parts = world_model_loss(prediction, batch)
    valid_steps_per_window = 2.0
    expected = {
      "state": torch.tensor(valid_steps_per_window / 6),
      "observation": valid_steps_per_window
      * torch.tensor([1 / 0.3**2, 1 / 5**2, 1 / 3.14**2, 1.0]).mean(),
      "reward": torch.tensor(valid_steps_per_window),
      "termination": valid_steps_per_window * torch.tensor(2.0).log(),
    }
    for key, value in expected.items():
      torch.testing.assert_close(parts[key], value)
    torch.testing.assert_close(loss, sum(expected.values()))
    loss.backward()
    torch.testing.assert_close(logits.grad, batch["mask"].float() * 0.25)
    # Duplicating time doubles the loss; duplicating the batch leaves it equal.
    for dim, factor in ((1, 2), (0, 1)):
      repeated_prediction = {
        key: torch.cat((value, value), dim=dim) for key, value in prediction.items()
      }
      repeated_batch = {
        key: torch.cat((value, value), dim=dim) for key, value in batch.items()
      }
      repeated, _ = world_model_loss(repeated_prediction, repeated_batch)
      torch.testing.assert_close(repeated, loss.detach() * factor)

  def test_critic_sums_time_and_averages_trajectories_with_padding(self):
    values = torch.tensor([[1.0, 2.0, 3.0], [4.0, 8.0, 9.0]], requires_grad=True)
    mask = torch.tensor([[True, True, True], [True, False, False]])
    targets = torch.zeros_like(values)
    targets[~mask] = float("nan")
    loss = critic_loss(values, targets, mask)
    # ((1² + 2² + 3²) + 4²) / 2 trajectories = 15, not / 4 valid steps.
    torch.testing.assert_close(loss, torch.tensor(15.0))
    loss.backward()
    torch.testing.assert_close(values.grad, values.detach() * mask)
    for dim, factor in ((1, 2), (0, 1)):
      repeated = critic_loss(
        torch.cat((values, values), dim=dim),
        torch.cat((targets, targets), dim=dim),
        torch.cat((mask, mask), dim=dim),
      )
      torch.testing.assert_close(repeated, loss.detach() * factor)


@unittest.skipUnless(
  DEVICE, "Set SUPERMIMIC_TEST_DEVICE=cpu or use CUDA for real online integration"
)
class OnlineIntegrationTests(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    from src.tasks.supermimic.env import SuperMimicEnv

    torch.set_num_threads(4)
    cls.metadata = robot_metadata()
    folder = tempfile.TemporaryDirectory()
    cls.addClassCleanup(folder.cleanup)
    manifest, _ = make_motions(Path(folder.name), cls.metadata)
    cls.env = SuperMimicEnv(
      FoundationConfig.training(manifest=manifest, num_envs=2, clips=("fixture_walk",)),
      device=DEVICE,
    )
    cls.config = OnlineConfig(
      num_envs=2,
      capacity=32,
      warmup_steps=12,
      collect_steps=2,
      wm_batches=1,
      wm_batch_size=4,
      policy_batch_size=4,
      horizon=4,
      critic_epochs=2,
      critic_minibatches=2,
      world_dims=(16,),
      actor_dims=(16,),
      critic_dims=(16,),
      clips=("fixture_walk",),
    )

  @classmethod
  def tearDownClass(cls):
    cls.env.close()

  def trainer(self, config=None):
    self.env.reset()
    trainer = OnlineTrainer(self.env, self.metadata, config or self.config)
    trainer.bootstrap()
    return trainer

  def test_wm_only_warmup_preserves_policy_and_resumes_at_phase_boundary(self):
    config = replace(self.config, wm_only_iters=3)
    trainer = self.trainer(config)
    actor = copy.deepcopy(trainer.actor.state_dict())
    critic = copy.deepcopy(trainer.critic.state_dict())
    world = [p.detach().clone() for p in trainer.world.parameters()]
    rows = [trainer.train_iteration() for _ in range(2)]
    self.assertEqual([row["collected_this_iteration"] for row in rows], [True, False])
    for row in rows:
      self.assertEqual(row["training_phase"], "wm_only")
      self.assertNotIn("actor_objective", row)
      self.assertNotIn("critic_loss", row)
      self.assertEqual(row["policy_and_critic_seconds"], 0)
    self.assertTrue(
      any(not torch.equal(a, b) for a, b in zip(world, trainer.world.parameters()))
    )
    self.assertFalse(trainer.actor_optimizer.state)
    self.assertFalse(trainer.critic_optimizer.state)
    saved = copy.deepcopy(trainer.state_dict())
    resumed = OnlineTrainer(self.env, self.metadata, config)
    resumed.load_state_dict(saved)
    last_warmup = resumed.train_iteration()
    self.assertEqual(last_warmup["iteration"], 3)
    self.assertEqual(last_warmup["training_phase"], "wm_only")
    self.assertTrue(last_warmup["collected_this_iteration"])
    for module, before in ((resumed.actor, actor), (resumed.critic, critic)):
      for name, value in module.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    self.assertEqual(resumed.actor_updates, 0)
    self.assertEqual(resumed.critic_updates, 0)
    self.assertEqual(resumed.world_updates, 3 * config.wm_batches)
    joint = resumed.train_iteration()
    self.assertEqual(joint["iteration"], 4)
    self.assertEqual(joint["training_phase"], "joint")
    self.assertFalse(joint["collected_this_iteration"])
    self.assertTrue(torch.isfinite(torch.tensor(joint["actor_objective"])))
    self.assertEqual(resumed.world_updates, 4 * config.wm_batches)
    self.assertEqual(resumed.actor_updates, 1)
    self.assertEqual(
      resumed.critic_updates, config.critic_epochs * config.critic_minibatches
    )
    self.assertTrue(
      any(
        not torch.equal(value, actor[name])
        for name, value in resumed.actor.state_dict().items()
      )
    )

  def test_optimizer_isolation_and_fresh_policy_collection(self):
    trainer = self.trainer()
    trainer.update_world()
    before = copy.deepcopy(trainer.world.state_dict())
    actor = [p.detach().clone() for p in trainer.actor.parameters()]
    critic = [p.detach().clone() for p in trainer.critic.parameters()]
    trainer.update_policy()
    for key, value in trainer.world.state_dict().items():
      torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    self.assertTrue(
      any(not torch.equal(a, b) for a, b in zip(actor, trainer.actor.parameters()))
    )
    self.assertTrue(
      any(not torch.equal(a, b) for a, b in zip(critic, trainer.critic.parameters()))
    )
    self.assertTrue(all(p.grad is None for p in trainer.world.parameters()))
    rows = [trainer.train_iteration() for _ in range(3)]
    self.assertEqual([r["collected_this_iteration"] for r in rows], [True, False, True])
    self.assertTrue(
      torch.isfinite(torch.tensor([r["actor_objective"] for r in rows])).all()
    )
    count = min(trainer.replay.steps, trainer.replay.capacity)
    self.assertGreater(int(trainer.replay.data["policy_version"][:count].max()), 0)

  def test_restore_replay_and_start_new_episode_ids(self):
    trainer = self.trainer()
    trainer.train_iteration()
    with tempfile.TemporaryDirectory() as folder:
      path = Path(folder) / "checkpoint.pt"
      torch.save(trainer.state_dict(), path)
      saved = torch.load(path, map_location="cpu", weights_only=True)
      with self.assertRaisesRegex(ValueError, "730-dimensional"):
        trainer.load_state_dict(saved | {"reference_schema": "full_body_268"})
      # Older online checkpoints have no warmup setting and must retain 0.
      del saved["config"]["wm_only_iters"]
      resumed = OnlineTrainer(self.env, self.metadata, self.config)
      resumed.load_state_dict(saved)
      self.assertEqual(resumed.iteration, trainer.iteration)
      count = min(resumed.replay.steps, resumed.replay.capacity)
      self.assertTrue(
        (
          self.env.episode_id > resumed.replay.data["episode_id"][:count].max(0).values
        ).all()
      )
      torch.testing.assert_close(
        trainer.world.input_mean, resumed.world.input_mean, rtol=0, atol=0
      )
      resumed.train_iteration()
      resumed.train_iteration()
      windows = resumed.replay.sample_windows(8, 8)
      for episode, mask in zip(windows["episode_id"], windows["mask"]):
        self.assertEqual(len(episode[mask].unique()), 1)

  def test_full_body_training_resumes_historical_checkpoint(self):
    config = replace(self.config, reference_dim=2680, amp="none", compile=False)
    trainer = self.trainer(config)
    trainer.train_iteration()
    saved = copy.deepcopy(trainer.state_dict())
    self.assertEqual(saved["reference_schema"], "full_body_268")
    self.assertEqual(trainer.critic.value[0].in_features, 2784)
    # b050140 checkpoints predate schema, execution and dimension metadata.
    del saved["reference_schema"]
    for key in ("reference_dim", "amp", "compile", "wm_only_iters"):
      del saved["config"][key]
    resumed = OnlineTrainer(self.env, self.metadata, config)
    resumed.load_state_dict(saved)
    for name in ("actor", "critic", "world"):
      for key, value in getattr(trainer, name).state_dict().items():
        torch.testing.assert_close(getattr(resumed, name).state_dict()[key], value)
    self.assertEqual(resumed.actor_updates, trainer.actor_updates)
    row = resumed.train_iteration()
    self.assertTrue(torch.isfinite(torch.tensor(row["actor_objective"])))
    self.assertEqual(resumed.iteration, 2)

  def test_capacity_overflow_rejects_corrupted_physical_transitions(self):
    counter = self.env.sim.data.ncollision
    original = counter.clone()
    try:
      counter.fill_(self.env.sim.wp_data.naconmax + 1)
      with self.assertRaisesRegex(RuntimeError, "capacity exceeded"):
        self.env.check_simulation_capacity()
    finally:
      counter.copy_(original)


class OnlineCliTests(unittest.TestCase):
  def test_cpu_train_resume_and_checkpoint_inference(self):
    self.run_cli_training(730)

  def test_full_body_cpu_train_resume_and_checkpoint_inference(self):
    self.run_cli_training(2680)

  def run_cli_training(self, reference_dim):
    with tempfile.TemporaryDirectory() as folder:
      output = Path(folder) / "training"
      manifest, _ = make_motions(Path(folder) / "motions", robot_metadata())
      prefix = [sys.executable]
      common = prefix + [
        "scripts/train_supermimic.py",
        "--manifest",
        str(manifest),
        "--device",
        "cpu",
        "--num-envs",
        "2",
        "--reference-dim",
        str(reference_dim),
        "--capacity",
        "32",
        "--warmup-steps",
        "8",
        "--learning-rate",
        "5e-5",
        "--wm-only-iters",
        "1",
        "--collect-steps",
        "1",
        "--world-dims",
        "16",
        "--actor-dims",
        "16",
        "--critic-dims",
        "16",
        "--wm-batches",
        "1",
        "--wm-batch-size",
        "2",
        "--policy-batch-size",
        "2",
        "--horizon",
        "2",
        "--critic-epochs",
        "1",
        "--critic-minibatches",
        "1",
        "--relative-orientation",
        "--termination-mode",
        "analytic",
        "--diagnostics-batch-size",
        "2",
        "--clips",
        "fixture_walk",
        "--evaluation-limit",
        "2",
        "--output",
        str(output),
        "--log-interval",
        "1",
        "--checkpoint-interval",
        "1",
      ]
      has_wandb = importlib.util.find_spec("wandb") is not None
      if has_wandb:
        common += ["--wandb", "--wandb-mode", "offline"]
      env = os.environ | {
        "MPLCONFIGDIR": str(Path(folder) / "matplotlib"),
        "XDG_CACHE_HOME": str(Path(folder) / "cache"),
        "WANDB_CONFIG_DIR": str(Path(folder) / "wandb-config"),
        "WANDB_CACHE_DIR": str(Path(folder) / "wandb-cache"),
      }
      previous_id = None
      for iteration in (1, 2):
        command = common + ["--iterations", str(iteration)]
        if iteration == 2:
          # Resume infers either layout without an explicit dimension argument.
          offset = command.index("--reference-dim")
          del command[offset : offset + 2]
          if reference_dim == 2680:
            historical = torch.load(
              output / "last.pt", map_location="cpu", weights_only=True
            )
            del historical["reference_schema"]
            for key in ("reference_dim", "amp", "compile"):
              del historical["config"][key]
            torch.save(historical, output / "last.pt")
          command += [
            "--resume",
            str(output / "last.pt"),
            "--full-checkpoint-interval",
            "2",
          ]
        result = subprocess.run(
          command,
          cwd=ROOT,
          env=env,
          capture_output=True,
          text=True,
          timeout=180,
          check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("WM loss", result.stdout)
        self.assertIn("WM-only warmup:", result.stdout)
        self.assertNotIn('"actor_objective":', result.stdout)
        saved = torch.load(output / "last.pt", map_location="cpu", weights_only=True)
        self.assertEqual(saved["config"]["wm_only_iters"], 1)
        self.assertEqual(saved["config"]["learning_rate"], 5e-5)
        self.assertEqual(saved["config"]["reference_dim"], reference_dim)
        self.assertEqual(saved["actor"]["encoder.0.weight"].shape[1], reference_dim)
        self.assertEqual(
          saved["critic"]["value.0.weight"].shape[1], reference_dim + 104
        )
        self.assertEqual(saved["actor_updates"], iteration - 1)
        self.assertEqual(saved["critic_updates"], iteration - 1)
        weights = torch.load(
          output / "policy_last.pt", map_location="cpu", weights_only=True
        )
        self.assertEqual(weights["kind"], "supermimic_actor")
        self.assertNotIn("replay", weights)
        self.assertNotIn("world", weights)
        self.assertEqual(weights["iteration"], iteration)
        for key, value in saved["actor"].items():
          torch.testing.assert_close(weights["actor"][key], value, rtol=0, atol=0)
        self.assertTrue((output / f"policy_iteration_{iteration:06d}.pt").exists())
        self.assertFalse((output / "iteration_000001.pt").exists())
        self.assertTrue((output / "policy_best.pt").exists())
        self.assertFalse((output / "best.pt").exists())
        if iteration == 2:
          self.assertTrue((output / "iteration_000002.pt").exists())
        rows = [
          json.loads(line)
          for line in (output / "metrics.jsonl").read_text().splitlines()
        ]
        self.assertEqual(rows[0]["training_phase"], "wm_only")
        self.assertNotIn("actor_objective", rows[0])
        if iteration == 2:
          self.assertEqual(rows[-1]["training_phase"], "joint")
          self.assertIn("actor_objective", rows[-1])
        if has_wandb:
          tracking = json.loads((output / "wandb_run.json").read_text())
          self.assertEqual(tracking["last_iteration"], iteration)
          self.assertEqual(tracking["mode"], "offline")
          self.assertNotEqual(tracking["id"], previous_id)
          previous_id = tracking["id"]
      self.assertIn("Base Environment", (output / "simulator.log").read_text())
      if has_wandb:
        self.assertEqual(
          len(list((output / "wandb").glob("offline-run-*/run-*.wandb"))), 2
        )
      checkpoint = torch.load(output / "last.pt", map_location="cpu", weights_only=True)
      self.assertEqual(checkpoint["iteration"], 2)
      self.assertEqual(checkpoint["world_updates"], 2)
      self.assertTrue(checkpoint["config"]["relative_orientation"])
      self.assertEqual(checkpoint["config"]["termination_mode"], "analytic")
      diagnostics = json.loads((output / "diagnostics.json").read_text())
      self.assertEqual(diagnostics["iteration"], 2)
      self.assertTrue(diagnostics["world_model"]["horizons"]["1"]["available"])
      self.assertFalse(diagnostics["world_model"]["horizons"]["32"]["available"])
      self.assertTrue(checkpoint["validation_windows"])
      evaluation = Path(folder) / "evaluation.json"
      result = subprocess.run(
        prefix
        + [
          "scripts/evaluate_supermimic.py",
          "--manifest",
          str(manifest),
          "--device",
          "cpu",
          "--checkpoint",
          str(output / "last.pt"),
          "--max-steps-per-clip",
          "2",
          "--output",
          str(evaluation),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
      )
      self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
      report = json.loads(evaluation.read_text())
      self.assertEqual(len(report["motions"]), 1)
      self.assertEqual(report["motions"][0]["motion"], "fixture_walk")
      self.assertTrue(report["motions"][0]["evaluation_limited"])
      result = subprocess.run(
        prefix
        + [
          "scripts/evaluate_supermimic.py",
          "--manifest",
          str(manifest),
          "--device",
          "cpu",
          "--checkpoint",
          str(output / "policy_last.pt"),
          "--max-steps-per-clip",
          "2",
          "--output",
          str(evaluation),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
      )
      self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
      lightweight_report = json.loads(evaluation.read_text())
      self.assertEqual(lightweight_report["motions"], report["motions"])


if __name__ == "__main__":
  unittest.main()
