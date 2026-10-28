"""Real-physics collection, staged updates and mixed-task checkpoint recovery."""

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch

from src.tasks.supermimic.config.rl_cfg import OnlineConfig
from src.tasks.supermimic.config.task_cfg import FoundationConfig
from src.tasks.supermimic.config.velocity_cfg import VelocityConfig
from src.tasks.supermimic.mdp.state import observe
from src.tasks.supermimic.mdp.velocity import velocity_reward
from src.tasks.supermimic.mdp.velocity_commands import VelocityCommands
from src.tasks.supermimic.models.kinematics import robot_metadata
from src.tasks.supermimic.rl.training_logger import simulator_output
from src.tasks.supermimic.rl.velocity_inference import load_velocity_actor
from src.tasks.supermimic.rl.velocity_training import VelocityTrainer
from src.tasks.supermimic.velocity_env import VelocityEnv
from tests.supermimic_fixtures import make_motions


class VelocityTrainingTests(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    torch.set_num_threads(2)
    cls.metadata = robot_metadata()
    cls.folder = tempfile.TemporaryDirectory()
    cls.root = Path(cls.folder.name)
    cls.manifest, cls.clips = make_motions(cls.root / "motions", cls.metadata)

  @classmethod
  def tearDownClass(cls):
    cls.folder.cleanup()

  def setUp(self):
    self.config = OnlineConfig(
      num_envs=4,
      capacity=64,
      warmup_steps=12,
      collect_steps=3,
      updates_per_collect=1,
      wm_batches=1,
      wm_batch_size=4,
      policy_batch_size=4,
      horizon=3,
      critic_epochs=1,
      critic_minibatches=1,
      world_dims=(16, 16),
      actor_dims=(16, 8),
      critic_dims=(16, 8),
      amp="none",
      compile=False,
      clips=tuple(self.clips),
      actor_input_normalization=True,
      critic_input_normalization=True,
      world_model_normalization=True,
      learning_rate=1e-5,
    )
    self.velocity = replace(
      VelocityConfig(), warmup_updates=1, resampling_time_range=(0.06, 0.06)
    )
    with (
      (self.root / "simulator.log").open("a") as sim_log,
      simulator_output(sim_log, verbose=False),
    ):
      self.env = VelocityEnv(
        FoundationConfig(manifest=self.manifest, clips=tuple(self.clips), num_envs=4),
        self.velocity,
        device="cpu",
        motion_envs=2,
        training=True,
      )
    self.addCleanup(self.env.close)

  def trainer(self):
    return VelocityTrainer(self.env, self.metadata, self.config, total_iterations=3)

  def test_initial_timeout_phase_preserves_real_episode_age_and_return(self):
    self.trainer()
    limits = self.env.episode_timeout_steps.clone()
    self.assertTrue(((limits >= 1) & (limits <= 500)).all())
    self.assertGreater(len(limits.unique()), 1)
    self.assertFalse(self.env.episode_length_buf.any())
    self.assertFalse(self.env.episode_discounted_return.any())
    self.env.episode_timeout_steps[:] = torch.tensor([400, 300, 1, 200])
    _, _, _, _, extras = self.env.step(torch.zeros(4, 29))
    transition = extras["velocity_transition"]
    self.assertEqual(int(transition["episode_step"][0]), 0)
    stats = extras["episode_statistics"]["velocity"]
    self.assertEqual((stats["count"], stats["steps_sum"], stats["time_out"]), (1, 1, 1))
    self.assertAlmostEqual(
      stats["metric_sums"]["reward_per_step"], float(transition["reward"][0])
    )
    self.assertEqual(int(self.env.episode_timeout_steps[2]), 500)
    self.assertEqual(int(self.env.episode_timeout_steps[3]), 200)
    self.assertAlmostEqual(
      float(self.env.episode_statistics.sums["velocity/reward_per_step"][3]),
      float(transition["reward"][1]),
    )

  def test_checkpoint_can_override_timing_but_not_other_configuration(self):
    trainer = self.trainer()
    trainer.bootstrap()
    path = self.root / "timing.pt"
    torch.save(trainer.state_dict(), path)
    saved = torch.load(path, weights_only=True)
    new_config = replace(self.velocity, resampling_time_range=(0.02, 0.08))
    self.env.velocity_config = new_config
    self.env.commands = VelocityCommands(new_config, self.env.library)
    resumed = self.trainer()
    with self.assertRaisesRegex(ValueError, "mismatch"):
      resumed.load_state_dict(saved)
    resumed.load_state_dict(saved, allow_timing_override=True)
    self.assertFalse(self.env.episode_length_buf.any())
    self.assertGreater(len(self.env.episode_timeout_steps.unique()), 1)
    # A new holding time uses the new scheduler; the current saved command was retained.
    self.env.command_state["command_remaining"].fill_(1)
    self.env.commands.advance(
      self.env.command_state, self.env.command_rng, self.env.control_steps
    )
    self.assertTrue((self.env.command_state["command_remaining"] <= 4).all())
    resumed.velocity_config = replace(new_config, alignment_weight=0.2)
    with self.assertRaisesRegex(ValueError, "mismatch"):
      resumed.load_state_dict(saved, allow_timing_override=True)

  def test_simulation_reward_matches_analytic_and_command_transition_order(self):
    trainer = self.trainer()
    before = {k: v.clone() for k, v in self.env.command_state.items()}
    self.env.command_state["command_remaining"].fill_(1)
    action = torch.zeros(4, 29)
    _, _, _, _, extras = self.env.step(action)
    transition = extras["velocity_transition"]
    torch.testing.assert_close(transition["command"], before["command"])
    torch.testing.assert_close(
      transition["next_command"], self.env.command_state["command"]
    )
    torch.testing.assert_close(
      transition["next_clean_observation"],
      observe(
        transition["next_state"],
        action[2:],
        trainer.actor.default_q,
        transition["context"],
      ),
    )
    reward, _ = velocity_reward(
      transition["next_state"],
      trainer.world.kinematics(transition["next_state"]),
      transition["command"],
      transition["action"],
      transition["previous_action"],
      trainer.world.kinematics.joint_limits,
      trainer.actor.default_q,
      self.velocity,
    )
    torch.testing.assert_close(reward, transition["reward"], atol=1e-6, rtol=1e-5)

  def test_timeout_preserves_pre_reset_transition_and_local_reset_history(self):
    self.env.episode_length_buf[2] = 499
    prior_ids = self.env.episode_id.clone()
    prior_history = self.env.history.data[3].clone()
    _, _, _, truncated, extras = self.env.step(torch.zeros(4, 29))
    transition = extras["velocity_transition"]
    self.assertTrue(truncated[2])
    self.assertTrue(transition["time_out"][0])
    self.assertFalse(transition["terminated"][0])
    self.assertEqual(int(transition["episode_id"][0]), int(prior_ids[2]))
    self.assertEqual(int(self.env.episode_id[2]), int(prior_ids[2]) + 1)
    torch.testing.assert_close(
      self.env.history.data[2], self.env.current_observation[2].expand(10, -1)
    )
    torch.testing.assert_close(self.env.history.data[3, :-1], prior_history[1:])

  def test_clip_end_changes_command_without_resetting_velocity_episode(self):
    self.env.command_state["command_source"].fill_(True)
    self.env.command_state["reference_motion"].zero_()
    self.env.command_state["reference_frame"][:] = self.env.library.lengths[0] - 1
    self.env.command_state["command_remaining"].fill_(100)
    self.env.commands._read_reference(self.env.command_state)
    episode = self.env.episode_id.clone()
    _, _, _, _, extras = self.env.step(torch.zeros(4, 29))
    torch.testing.assert_close(self.env.episode_id[2:], episode[2:])
    self.assertNotIn("motion_id", extras["velocity_transition"])
    self.assertTrue((self.env.command_state["reference_frame"] >= 0).all())

  def test_staged_training_freezes_motion_norm_then_updates_both_interfaces(self):
    trainer = self.trainer()
    trainer.bootstrap()
    actor_before = {k: v.clone() for k, v in trainer.actor.state_dict().items()}
    first = trainer.train_iteration()
    self.assertEqual(first["stage"], "warmup")
    self.assertEqual(trainer.warmup_completed, 1)
    for key, value in trainer.actor.state_dict().items():
      torch.testing.assert_close(value, actor_before[key], rtol=0, atol=0)
    second = trainer.train_iteration()
    self.assertEqual(second["stage"], "joint")
    self.assertEqual(trainer.actor_updates, 1)
    self.assertEqual(trainer.velocity_updates, 2)
    self.assertEqual(trainer.velocity_critic_updates, 1)
    self.assertGreater(int(trainer.actor.obs_norm.count), 0)
    self.assertTrue(
      all(
        torch.isfinite(torch.tensor(v)) for v in second.values() if isinstance(v, float)
      )
    )
    # Persist to break live storage aliases before changing trainer state.
    path = self.root / "resume.pt"
    torch.save(trainer.state_dict(), path)
    snapshot = torch.load(path, weights_only=True)
    command = {k: v.clone() for k, v in self.env.command_state.items()}
    trainer.train_iteration()
    trainer.load_state_dict(snapshot)
    self.assertEqual(trainer.iteration, 2)
    for key, value in command.items():
      torch.testing.assert_close(self.env.command_state[key], value)
    self.assertEqual(trainer.velocity_replay.task, "velocity")
    self.assertTrue(
      (
        self.env.episode_id[2:]
        > trainer.velocity_replay.data["episode_id"][: trainer.velocity_replay.steps]
        .max(0)
        .values
      ).all()
    )
    self.assertEqual(trainer.train_iteration()["stage"], "joint")
    export = self.root / "velocity.pt"
    torch.save(trainer.velocity_policy_state_dict(), export)
    actor, _ = load_velocity_actor(export)
    self.assertEqual(actor(torch.zeros(1, 10, 93), torch.zeros(1, 3)).shape, (1, 29))

  def test_inference_environment_does_not_open_manifest(self):
    with (
      (self.root / "inference.log").open("a") as sim_log,
      simulator_output(sim_log, verbose=False),
    ):
      inference = VelocityEnv(
        FoundationConfig(manifest=self.root / "does-not-exist.json", num_envs=1),
        self.velocity,
        device="cpu",
      )
    try:
      self.assertIsNone(inference.library)
      inference.set_command((0.5, 0.0, 0.2))
      inference.reset()
      torch.testing.assert_close(
        inference.policy_input()["command"], torch.tensor([[0.5, 0.0, 0.2]])
      )
      for _ in range(4):
        inference.step(torch.zeros(1, 29))
      torch.testing.assert_close(
        inference.policy_input()["command"], torch.tensor([[0.5, 0.0, 0.2]])
      )
    finally:
      inference.close()


if __name__ == "__main__":
  unittest.main()
