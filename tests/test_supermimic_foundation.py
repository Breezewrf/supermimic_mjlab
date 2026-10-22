"""CPU tests for reference, task, sampler and replay contracts."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.tasks.supermimic.config.task_cfg import (
  DEFAULT_MANIFEST,
  TRACKED_BODIES,
  FoundationConfig,
)
from src.tasks.supermimic.data.motion_library import MotionLibrary
from src.tasks.supermimic.data.replay_buffer import ReplayBuffer
from src.tasks.supermimic.mdp.reset_sampler import ResetSampler
from src.tasks.supermimic.mdp.rewards import tracking_reward
from src.tasks.supermimic.mdp.state import ObservationHistory, add_sensor_noise, observe
from src.tasks.supermimic.mdp.terminations import failure_terms
from src.tasks.supermimic.rl.evaluation import evaluate


def reference_fixture(directory):
  joints = tuple(f"joint_{i}" for i in range(29))
  bodies = ("pelvis", "foot")
  entries = []
  for clip, count in enumerate((3, 7)):
    stored_joints = joints[::-1]
    stored_bodies = bodies[::-1]
    values = np.broadcast_to(np.arange(29, dtype=np.float32)[::-1], (count, 29)).copy()
    positions = np.zeros((count, 2, 3), dtype=np.float32)
    positions[:, :, 0] = clip * 100 + np.arange(count)[:, None]
    positions[:, 1, 2] = 0.8
    quats = np.zeros((count, 2, 4), dtype=np.float32)
    quats[..., 0] = 1
    name = f"clip{clip}.npz"
    np.savez(
      directory / name,
      joint_pos=values,
      joint_vel=values * 0,
      body_pos_w=positions,
      body_quat_w=quats,
      body_lin_vel_w=positions * 0,
      body_ang_vel_w=positions * 0,
      joint_names=stored_joints,
      body_names=stored_bodies,
      fps=np.array([50]),
      quaternion_order="wxyz",
    )
    entries.append({"file": name, "frames": count})
  manifest = directory / "manifest.json"
  manifest.write_text(
    json.dumps(
      {"schema_version": 1, "fps": 50, "quaternion_order": "wxyz", "clips": entries}
    )
  )
  return manifest, joints, bodies


def transition(step, episode=0, episode_step=None, done=None, num_envs=1):
  scalar = torch.full((num_envs,), step, dtype=torch.long)
  state = scalar[:, None].float().expand(-1, 71).clone()
  observation = scalar[:, None].float().expand(-1, 93).clone()
  zeros = torch.zeros(num_envs, dtype=torch.bool)
  result = {
    "state": state,
    "next_state": state + 1,
    "action": torch.full((num_envs, 29), 0.1),
    "previous_action": torch.zeros(num_envs, 29),
    "observation": observation,
    "next_observation": observation + 1,
    "clean_observation": observation.clone(),
    "next_clean_observation": observation + 1,
    "context": torch.ones(num_envs, 33),
    "reward": torch.ones(num_envs),
    "motion_id": scalar * 0,
    "frame_id": scalar,
    "episode_id": torch.full_like(scalar, episode),
    "episode_step": torch.full_like(
      scalar, step if episode_step is None else episode_step
    ),
    "terminated": zeros.clone(),
    "time_out": zeros.clone(),
    "motion_end": zeros.clone(),
  }
  if done:
    result[done][:] = True
  return result


class LibraryTests(unittest.TestCase):
  def test_names_variable_lengths_and_future_padding(self):
    with tempfile.TemporaryDirectory() as folder:
      manifest, joints, bodies = reference_fixture(Path(folder))
      library = MotionLibrary(manifest, joints, bodies)
      frame = library.get_frame(torch.tensor([0, 1]), torch.tensor([2, 1]))
      torch.testing.assert_close(frame["joint_pos"][0], torch.arange(29).float())
      torch.testing.assert_close(frame["body_pos_w"][:, 0, 2], torch.tensor([0.8, 0.8]))
      future = library.get_future(torch.tensor([0, 1]), torch.tensor([1, 5]), 5)
      torch.testing.assert_close(
        future["body_pos_w"][:, :, 0, 0],
        torch.tensor([[1, 2, 2, 2, 2], [105, 106, 106, 106, 106.0]]),
      )
      with self.assertRaises(IndexError):
        library.get_frame(torch.tensor([0]), torch.tensor([3]))
      with self.assertRaises(ValueError):
        MotionLibrary(manifest, joints, bodies, clips=("missing",))
      self.assertEqual(
        len(MotionLibrary(manifest, joints, bodies, clips=("clip1",))), 1
      )

  def test_manifest_and_array_validation(self):
    with tempfile.TemporaryDirectory() as folder:
      manifest, joints, bodies = reference_fixture(Path(folder))
      metadata = json.loads(manifest.read_text())
      metadata["fps"] = 30
      manifest.write_text(json.dumps(metadata))
      with self.assertRaises(ValueError):
        MotionLibrary(manifest, joints, bodies)
      metadata["fps"] = 50
      metadata["clips"][0]["frames"] += 1
      manifest.write_text(json.dumps(metadata))
      with self.assertRaises(ValueError):
        MotionLibrary(manifest, joints, bodies)


class TaskTests(unittest.TestCase):
  def setUp(self):
    self.state = torch.zeros(2, 71)
    self.state[:, 2] = 0.8
    self.state[:, 3] = 1
    self.action = torch.zeros(2, 29)
    self.context = torch.zeros(2, 33)
    self.bodies = {
      "body_pos_w": self.state[:, None, :3].repeat(1, 2, 1),
      "body_quat_w": self.state[:, None, 3:7].repeat(1, 2, 1),
      "body_lin_vel_w": torch.zeros(2, 2, 3),
      "body_ang_vel_w": torch.zeros(2, 2, 3),
    }
    self.limits = torch.tensor([[-1.0, 1.0]]).repeat(29, 1)

  def test_observation_order_bias_noise_and_history(self):
    self.state[:, 7:36] = 0.2
    self.state[:, 42:71] = 0.3
    self.state[:, 39:42] = 0.4
    self.context[:, 1:30] = 0.01
    clean = observe(self.state, self.action, torch.zeros(29), self.context)
    self.assertEqual(clean.shape, (2, 93))
    torch.testing.assert_close(clean[:, :29], torch.full((2, 29), 0.21))
    torch.testing.assert_close(clean[:, 29:58], torch.full((2, 29), 0.3))
    torch.testing.assert_close(clean[:, 58:61], torch.full((2, 3), 0.4))
    torch.testing.assert_close(
      clean[:, 61:64], torch.tensor([[0, 0, -1.0]]).repeat(2, 1)
    )
    noise = add_sensor_noise(clean) - clean
    amplitude = torch.tensor(
      [0.01] * 29 + [0.5] * 29 + [0.2] * 3 + [0.05] * 3 + [0.0] * 29
    )
    self.assertTrue((noise.abs() <= amplitude + 1e-7).all())
    history = ObservationHistory(2, 10, "cpu")
    history.reset(torch.tensor([0, 1]), clean)
    history.append(clean + 1)
    history.reset(torch.tensor([0]), clean[:1] + 7)
    torch.testing.assert_close(history.data[0], (clean[0] + 7).expand(10, -1))
    torch.testing.assert_close(history.data[1, -1], clean[1] + 1)

  def test_reward_regularization_and_failures(self):
    reward, _ = tracking_reward(
      self.state, self.bodies, self.bodies, self.action, self.action, self.limits
    )
    torch.testing.assert_close(reward, torch.full((2,), 0.1))
    action = self.action + 1
    penalized, _ = tracking_reward(
      self.state, self.bodies, self.bodies, action, self.action, self.limits
    )
    torch.testing.assert_close(penalized, reward - 29 * 0.1 * 0.02)
    moved = self.state.clone()
    moved[0, 2] += 0.26
    moved[1, 3:7] = torch.tensor([0.0, 1.0, 0.0, 0.0])
    failed = failure_terms(moved, self.bodies)
    self.assertTrue(failed["height_failure"][0])
    self.assertTrue(failed["tilt_failure"][1])
    self.assertFalse(failed["height_failure"][1])

  def test_shared_functions_have_finite_gradients_and_no_mutation(self):
    state = self.state.clone().requires_grad_()
    action = self.action.clone().requires_grad_()
    original = state.detach().clone()
    bodies = self.bodies | {
      "body_pos_w": state[:, None, :3].expand(-1, 2, -1),
      "body_quat_w": state[:, None, 3:7].expand(-1, 2, -1),
    }
    reward, _ = tracking_reward(
      state, bodies, self.bodies, action, self.action, self.limits
    )
    obs = observe(state, action, torch.zeros(29), self.context)
    (reward.sum() + obs.sum()).backward()
    self.assertTrue(torch.isfinite(state.grad).all())
    self.assertTrue(torch.isfinite(action.grad).all())
    torch.testing.assert_close(state.detach(), original)


class SamplerTests(unittest.TestCase):
  def test_low_returns_get_more_practice_and_no_final_frame_resets(self):
    sampler = ResetSampler(torch.tensor([51, 101]), mode="return", seed=4)
    before = sampler.probabilities().clone()
    sampler.update_episode(0, torch.arange(50), torch.zeros(50))
    self.assertGreater(float(sampler.probabilities()[0]), float(before[0]))
    self.assertTrue((sampler.probabilities() >= 0.5 / 3).all())
    motion, frame = sampler.sample(1000)
    self.assertTrue((frame < sampler.lengths[motion] - 1).all())
    snapshot = sampler.state_dict()
    restored = ResetSampler(torch.tensor([51, 101]), mode="return")
    restored.load_state_dict(snapshot)
    a, b = sampler.sample(20), restored.sample(20)
    torch.testing.assert_close(a[0], b[0])
    torch.testing.assert_close(a[1], b[1])


class ReplayTests(unittest.TestCase):
  def test_terminal_windows_copy_and_history_padding(self):
    replay = ReplayBuffer(1, capacity=16)
    for step in range(3):
      item = transition(step, done="terminated" if step == 2 else None)
      replay.append(item)
      item["state"].fill_(1000)
    windows = replay.sample_windows(100, 8)
    for state, mask, terminal in zip(
      windows["state"], windows["mask"], windows["terminated"]
    ):
      valid = state[mask, 0]
      self.assertTrue((valid < 3).all())
      self.assertEqual(int(valid[-1]), 2)
      self.assertTrue(terminal[mask][-1])
      self.assertTrue((state[~mask] == 0).all())
    starts = replay.sample_starts(100, 10)
    for state, history in zip(starts["state"], starts["history"]):
      t = int(state[0])
      expected = torch.arange(t - 9, t + 1).clamp_min(0).float()
      torch.testing.assert_close(history[:, 0], expected)

  def test_ring_wrap_and_episode_boundaries(self):
    replay = ReplayBuffer(1, capacity=10)
    for step in range(25):
      replay.append(
        transition(
          step,
          episode=step // 5,
          episode_step=step % 5,
          done="motion_end" if step % 5 == 4 else None,
        )
      )
    windows = replay.sample_windows(200, 8)
    for state, episode, mask in zip(
      windows["state"], windows["episode_id"], windows["mask"]
    ):
      self.assertTrue((state[mask, 0] >= 15).all())
      self.assertTrue((episode[mask] == episode[0]).all())
      self.assertTrue((torch.diff(state[mask, 0]) == 1).all())
    restored = ReplayBuffer.from_state_dict(replay.state_dict())
    a, b = replay.sample_starts(10), restored.sample_starts(10)
    torch.testing.assert_close(a["history"], b["history"])

  def test_insufficient_windows_and_overwritten_history(self):
    replay = ReplayBuffer(1, capacity=10)
    replay.append(transition(0))
    with self.assertRaises(ValueError):
      replay.sample_windows(1, 8)
    for step in range(1, 15):
      replay.append(transition(step))
    starts = replay.sample_starts(30, 10)
    self.assertTrue((starts["state"][:, 0] == 14).all())
    torch.testing.assert_close(starts["history"][0, :, 0], torch.arange(5, 15).float())

  def test_all_end_reasons_stop_sequences_and_keep_terminal_next_state(self):
    for kind in ("terminated", "time_out", "motion_end"):
      replay = ReplayBuffer(1, capacity=10)
      replay.append(transition(0, done=kind))
      replay.append(transition(1, episode=1, episode_step=0, done="terminated"))
      windows = replay.sample_windows(20, 8)
      self.assertTrue((windows["mask"].sum(-1) == 1).all())
      torch.testing.assert_close(
        windows["next_state"][:, 0], windows["state"][:, 0] + 1
      )
      self.assertTrue(windows[kind][:, 0].any())


class EvaluationTests(unittest.TestCase):
  def test_normalization_and_debug_limit_are_distinct_from_failure(self):
    class FakeEnv:
      num_envs = 1
      step_dt = 0.02
      foundation_config = FoundationConfig()
      library = type("Library", (), {"names": ["example"], "lengths": [6]})()

      def reset(self, **kwargs):
        self.time = 0
        return {"history": torch.zeros(1, 10, 93), "future_reference": {}}, {}

      def step(self, action):
        self.time += 1
        end = torch.tensor([self.time == 5])
        return (
          {"history": torch.zeros(1, 10, 93), "future_reference": {}},
          torch.tensor([0.1]),
          torch.tensor([False]),
          end,
          {
            "metrics": {
              key: torch.zeros(1)
              for key in (
                "body_position_error",
                "body_orientation_error",
                "root_position_error",
                "root_orientation_error",
              )
            },
            "transition": {"time_out": torch.tensor([False])},
          },
        )

    full = evaluate(FakeEnv())
    self.assertEqual(full["successes"], 1)
    self.assertAlmostEqual(full["normalized_return"], 1)
    limited = evaluate(FakeEnv(), max_steps_per_clip=2)["motions"][0]
    self.assertTrue(limited["evaluation_limited"])
    self.assertEqual(limited["failure_reason"], "evaluation_limit")
    self.assertFalse(limited["success"])


@unittest.skipUnless(DEFAULT_MANIFEST.exists(), "Converted LAFAN dataset not present")
class RealDatasetTests(unittest.TestCase):
  def test_all_40_clips_validate_and_have_safe_futures(self):
    metadata = json.loads(DEFAULT_MANIFEST.read_text())
    library = MotionLibrary(
      DEFAULT_MANIFEST, tuple(metadata["joint_names"]), TRACKED_BODIES
    )
    self.assertEqual(len(library), 40)
    ids = torch.arange(len(library))
    future = library.get_future(ids, library.lengths - 1)
    self.assertEqual(future["joint_pos"].shape, (40, 10, 29))
    torch.testing.assert_close(future["joint_pos"][:, 0], future["joint_pos"][:, -1])


if __name__ == "__main__":
  unittest.main()
