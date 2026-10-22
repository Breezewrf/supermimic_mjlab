"""Real MJWarp tests. CUDA by default; set SUPERMIMIC_TEST_DEVICE=cpu for CPU."""

import os
import unittest
from dataclasses import replace
from unittest.mock import patch

import torch

from src.tasks.supermimic.config.task_cfg import DEFAULT_MANIFEST, FoundationConfig
from src.tasks.supermimic.data.replay_buffer import ReplayBuffer

DEVICE = os.environ.get(
  "SUPERMIMIC_TEST_DEVICE", "cuda:0" if torch.cuda.is_available() else ""
)


@unittest.skipUnless(
  DEVICE and DEFAULT_MANIFEST.exists(),
  "CUDA/data unavailable; set SUPERMIMIC_TEST_DEVICE=cpu for CPU MJWarp",
)
class EnvironmentIntegrationTests(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    from src.tasks.supermimic.env import SuperMimicEnv

    cls.env = SuperMimicEnv(
      FoundationConfig.training(num_envs=2, clips=("walk1_subject1",)), device=DEVICE
    )

  @classmethod
  def tearDownClass(cls):
    cls.env.close()

  def setUp(self):
    self.env.foundation_config = replace(
      self.env.foundation_config, episode_length_s=10.0
    )
    self.env.reset(seed=3, options={"motion_id": 0, "frame_id": 0})
    self.zero = torch.zeros(2, 29, device=DEVICE)

  def test_real_collection_and_replay_interfaces(self):
    replay = ReplayBuffer(2, capacity=32, device=DEVICE)
    for _ in range(16):
      inputs, reward, _, _, extras = self.env.step(self.zero)
      self.assertTrue(torch.isfinite(reward).all())
      self.assertEqual(inputs["history"].shape, (2, 10, 93))
      replay.append(extras["transition"])
    windows = replay.sample_windows(4)
    self.assertEqual(windows["state"].shape, (4, 8, 71))
    starts = replay.sample_starts(4)
    self.assertEqual(starts["history"].shape, (4, 10, 93))

  def test_timeout_captures_pre_reset_state_and_resets_history(self):
    env = self.env
    env.foundation_config = replace(env.foundation_config, episode_length_s=0.02)
    episode = env.episode_id.clone()
    context = env.context.clone()
    captured = []
    original_reset = env.reset

    def capture_before_reset(**kwargs):
      captured.append(env.read_state().clone())
      return original_reset(**kwargs)

    with patch.object(env, "reset", side_effect=capture_before_reset):
      _, _, terminal, cutoff, extras = env.step(self.zero)
    t = extras["transition"]
    self.assertTrue(t["time_out"].all())
    self.assertTrue(cutoff.all())
    torch.testing.assert_close(env.episode_id, episode + 1)
    torch.testing.assert_close(t["episode_id"], episode)
    torch.testing.assert_close(t["context"], context)
    torch.testing.assert_close(t["next_state"], captured[0])
    self.assertTrue((env.episode_length_buf == 0).all())
    torch.testing.assert_close(env.history.data[:, 0], env.history.data[:, -1])
    self.assertTrue((t["previous_action"] == 0).all())
    self.assertTrue(torch.isfinite(t["next_state"]).all())
    frozen = t["next_state"].clone()
    env.step(self.zero)
    torch.testing.assert_close(t["next_state"], frozen)
    torch.testing.assert_close(terminal, t["terminated"])

  def test_physical_failure_and_motion_end_are_separate(self):
    env = self.env
    state = env.read_state()
    root = torch.cat((state[:, :7], state[:, 36:42]), dim=-1)
    root[:, :3] += env.scene.env_origins
    root[:, 2] += 0.6
    env.robot.write_root_state_to_sim(root)
    env.scene.write_data_to_sim()
    env.sim.forward()
    _, _, terminal, _, extras = env.step(self.zero)
    self.assertTrue(terminal.all())
    self.assertTrue(extras["metrics"]["height_failure"].all())
    self.assertFalse(extras["transition"]["motion_end"].any())
    env.reset(options={"motion_id": 0, "frame_id": int(env.library.lengths[0]) - 2})
    episode = env.episode_id.clone()
    _, _, _, cutoff, extras = env.step(self.zero)
    self.assertTrue(extras["transition"]["motion_end"].all())
    self.assertTrue(cutoff.all())
    torch.testing.assert_close(env.episode_id, episode + 1)

  def test_context_matches_model_and_partial_reset_is_isolated(self):
    env = self.env
    context = env.context.clone()
    torch.testing.assert_close(context[:, 1:30], env.robot.data.encoder_bias)
    self.assertTrue(((context[:, 0] >= 0.3) & (context[:, 0] <= 1.2)).all())
    self.assertTrue((context[:, 1:30].abs() <= 0.01).all())
    self.assertTrue(
      (
        context[:, 30:].abs() <= torch.tensor([0.025, 0.05, 0.05], device=DEVICE) + 1e-7
      ).all()
    )
    state_other = env.read_state()[1].clone()
    history_other = env.history.data[1].clone()
    env.reset(env_ids=[0], options={"motion_id": 0, "frame_id": 30})
    torch.testing.assert_close(env.context[1], context[1])
    torch.testing.assert_close(env.context[:, 0], context[:, 0], rtol=0, atol=0)
    torch.testing.assert_close(env.context[:, 30:], context[:, 30:], rtol=0, atol=0)
    torch.testing.assert_close(env.read_state()[1], state_other)
    torch.testing.assert_close(env.history.data[1], history_other)
    self.assertFalse(torch.equal(env.context[0, 1:30], context[0, 1:30]))

  def test_static_domain_parameters_survive_full_and_automatic_resets(self):
    env = self.env
    friction = env.sim.model.geom_friction.clone()
    com = env.sim.model.body_ipos.clone()
    static = env.context[:, [0, 30, 31, 32]].clone()
    bias = env.context[:, 1:30].clone()
    env.reset(options={"motion_id": 0, "frame_id": 10})
    self.assertFalse(torch.equal(env.context[:, 1:30], bias))
    env.foundation_config = replace(env.foundation_config, episode_length_s=0.02)
    env.step(self.zero)
    torch.testing.assert_close(env.context[:, [0, 30, 31, 32]], static, rtol=0, atol=0)
    torch.testing.assert_close(
      env.sim.model.geom_friction.clone(), friction, rtol=0, atol=0
    )
    torch.testing.assert_close(env.sim.model.body_ipos.clone(), com, rtol=0, atol=0)

  def test_startup_domain_samples_are_reproducible_for_a_new_environment(self):
    from src.tasks.supermimic.env import SuperMimicEnv

    other = SuperMimicEnv(
      FoundationConfig.training(num_envs=2, clips=("walk1_subject1",)), device=DEVICE
    )
    try:
      torch.testing.assert_close(
        other.context[:, [0, 30, 31, 32]],
        self.env.context[:, [0, 30, 31, 32]],
        rtol=0,
        atol=0,
      )
    finally:
      other.close()

  def test_training_reset_perturbs_state_but_preserves_joint_velocities(self):
    env = self.env
    state = env.read_state()
    reference = env.library.get_frame(env.motion_id, env.frame_id)
    root = env.library.root_state(env.motion_id, env.frame_id)
    self.assertTrue(env.foundation_config.reset_noise)
    self.assertGreater(float((state[:, :3] - root[:, :3]).abs().max()), 0.0)
    self.assertTrue(
      (
        (state[:, :3] - root[:, :3]).abs()
        <= state.new_tensor([0.05, 0.05, 0.01]) + 2e-6
      ).all()
    )
    self.assertTrue(
      (
        (state[:, 36:39] - root[:, 7:10]).abs()
        <= state.new_tensor([0.5, 0.5, 0.2]) + 2e-6
      ).all()
    )
    self.assertTrue(
      (
        (state[:, 39:42] - root[:, 10:13]).abs()
        <= state.new_tensor([0.52, 0.52, 0.78]) + 2e-6
      ).all()
    )
    torch.testing.assert_close(
      state[:, 42:], reference["joint_vel"], atol=2e-5, rtol=2e-5
    )

  def test_evaluation_reset_is_exact_and_context_is_deterministic(self):
    from src.tasks.supermimic.env import SuperMimicEnv

    env = SuperMimicEnv(
      FoundationConfig(num_envs=1, clips=("walk1_subject1",)), device=DEVICE
    )
    try:
      for frame in (0, 30):
        env.reset(options={"motion_id": 0, "frame_id": frame})
        state = env.read_state()
        root = env.library.root_state(env.motion_id, env.frame_id)
        reference = env.library.get_frame(env.motion_id, env.frame_id)
        torch.testing.assert_close(state[:, :7], root[:, :7])
        torch.testing.assert_close(state[:, 36:42], root[:, 7:13])
        torch.testing.assert_close(state[:, 7:36], reference["joint_pos"])
        torch.testing.assert_close(state[:, 42:], reference["joint_vel"])
        torch.testing.assert_close(env.context[:, 0], torch.ones(1, device=DEVICE))
        torch.testing.assert_close(
          env.context[:, 1:], torch.zeros(1, 32, device=DEVICE)
        )
        torch.testing.assert_close(env.current_observation, env.clean_observation)
    finally:
      env.close()

  def test_reference_kinematics_matches_npz(self):
    env = self.env
    for frame in (0, 50, 500):
      actual = env.replay_frame(0, frame)
      reference = env.library.get_frame(
        torch.zeros(2, dtype=torch.long, device=DEVICE),
        torch.full((2,), frame, dtype=torch.long, device=DEVICE),
      )
      for key in ("body_pos_w", "body_lin_vel_w", "body_ang_vel_w"):
        torch.testing.assert_close(actual[key], reference[key], atol=2e-4, rtol=2e-4)


if __name__ == "__main__":
  unittest.main()
