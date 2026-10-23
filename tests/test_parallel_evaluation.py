"""Serial/parallel score parity, auto-reset masking and real simulator batching."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from src.tasks.supermimic.config.rl_cfg import OnlineConfig
from src.tasks.supermimic.config.task_cfg import FoundationConfig
from src.tasks.supermimic.rl.evaluation import (
  ERROR_METRICS,
  evaluate,
  evaluation_env_count,
)


class VectorEnv:
  """Return corrupt post-reset data to expose accidental scoring of later episodes."""

  device = "cpu"
  step_dt = 0.02
  library = SimpleNamespace(
    names=["short", "height", "tilt", "physical", "long"],
    lengths=torch.tensor([6, 11, 8, 5, 13]),
  )

  def __init__(self, width=1, mode="failure", poison=True):
    self.num_envs, self.mode, self.poison = width, mode, poison
    self.foundation_config = FoundationConfig(
      num_envs=width, terminate_on_failure=mode != "playback"
    )
    self.calls = 0

  def inputs(self):
    history = torch.zeros(self.num_envs, 2, 93)
    history[:, :, 0] = self.motion[:, None]
    history[:, :, 1] = self.time[:, None]
    return {"history": history, "future_reference": {}}

  def read_state(self):
    return torch.zeros(self.num_envs, 71)

  def reset(self, *, seed, options):
    assert seed == 0 and options["frame_id"] == 0
    self.motion = torch.as_tensor(options["motion_id"]).expand(self.num_envs).clone()
    self.time = torch.zeros(self.num_envs, dtype=torch.long)
    self.finished = torch.zeros(self.num_envs, dtype=torch.bool)
    return self.inputs(), {}

  def step(self, action):
    self.calls += 1
    self.time += 1
    height = (self.motion == 1) & (self.time >= 2)
    tilt = (self.motion == 2) & (self.time >= 3)
    physical = (self.motion == 3) & (self.time == 2)
    timeout = torch.zeros(self.num_envs, dtype=torch.bool)
    if self.mode in ("complete", "timeout"):
      height.zero_()
      tilt.zero_()
      physical.zero_()
    if self.mode == "timeout":
      timeout = (self.motion == 1) & (self.time == 2)
    terminal = (height | tilt | physical) & self.foundation_config.terminate_on_failure
    cutoff = timeout | (self.time == self.library.lengths[self.motion] - 1)
    reward = (self.motion + 1) * 0.01 + self.time * 0.0001 + action[:, 0] * 0.001
    errors = {
      key: (self.motion + 1) * (i + 1) * 0.1 + self.time * 0.01
      for i, key in enumerate(ERROR_METRICS)
    }
    if self.poison:
      reward[self.finished] = float("nan")
      for value in errors.values():
        value[self.finished] = float("nan")
      height |= self.finished
      tilt |= self.finished
    self.finished |= terminal | cutoff
    return (
      self.inputs(),
      reward,
      terminal,
      cutoff,
      {
        "metrics": {**errors, "height_failure": height, "tilt_failure": tilt},
        "transition": {"time_out": timeout, "next_state": self.read_state()},
      },
    )


def history_policy(history, reference):
  del reference
  return (history[:, -1, :1] * 0.1 + history[:, -1, 1:2] * 0.01).expand(-1, 29)


class ParallelEvaluationTests(unittest.TestCase):
  def test_scores_match_serial_for_failure_completion_timeout_and_playback(self):
    for mode in ("failure", "complete", "timeout", "playback"):
      for width in (2, 3, 5, 8):
        for limit in (None, 2):
          with self.subTest(mode=mode, width=width, limit=limit):
            serial = VectorEnv(mode=mode)
            expected = evaluate(serial, history_policy, limit)
            parallel = VectorEnv(width, mode=mode)
            rows = []
            actual = evaluate(parallel, history_policy, limit, on_motion=rows.append)
            self.assertEqual(actual.pop("evaluation_num_envs"), width)
            expected.pop("evaluation_num_envs")
            self.assertEqual(actual, expected)
            self.assertEqual(rows, actual["motions"])
            self.assertLess(parallel.calls, serial.calls)

  def test_nonfinite_reward_in_active_lane_is_rejected(self):
    env = VectorEnv(3)
    original = env.step

    def corrupt(action):
      result = original(action)
      result[1][0] = float("nan")
      return result

    env.step = corrupt
    with self.assertRaisesRegex(FloatingPointError, "Nonfinite"):
      evaluate(env)

  def test_environment_count_respects_manifest_selection_and_cap(self):
    with tempfile.TemporaryDirectory() as directory:
      manifest = Path(directory) / "manifest.json"
      manifest.write_text(
        json.dumps(
          {"clips": [{"file": "a.npz"}, {"file": "nested/b.npz"}, {"file": "c.npz"}]}
        )
      )
      self.assertEqual(evaluation_env_count(manifest), 3)
      self.assertEqual(evaluation_env_count(manifest, max_envs=2), 2)
      self.assertEqual(evaluation_env_count(manifest, ("a", "nested/b.npz")), 2)
      self.assertEqual(evaluation_env_count(manifest, ("a", "a.npz")), 1)
      with self.assertRaisesRegex(ValueError, "positive"):
        evaluation_env_count(manifest, max_envs=0)

  def test_training_wrapper_uses_parallel_pool_and_restores_actor_and_rng(self):
    spec = importlib.util.spec_from_file_location(
      "parallel_train", "scripts/train_supermimic.py"
    )
    train = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train)
    actor = torch.nn.Linear(2, 2).train()
    before = torch.get_rng_state().clone()
    environments = []

    def create(config, device):
      self.assertEqual(device, "cpu")
      env = VectorEnv(config.num_envs, mode="complete")
      env.close = lambda: environments.append("closed")
      environments.append(env)
      torch.rand(10)
      return env

    with (
      patch("src.tasks.supermimic.env.SuperMimicEnv", side_effect=create),
      patch.object(train, "evaluation_env_count", return_value=3) as count,
      patch.object(train, "evaluate", return_value={"normalized_return": 0.8}),
    ):
      report = train.evaluate_actor(
        actor, Path("manifest.json"), OnlineConfig(), "cpu", max_envs=3
      )
    self.assertEqual(report["normalized_return"], 0.8)
    self.assertEqual(environments[0].num_envs, 3)
    self.assertEqual(environments[1], "closed")
    count.assert_called_once_with(Path("manifest.json"), (), 3)
    self.assertTrue(actor.training)
    torch.testing.assert_close(torch.get_rng_state(), before)


class ParallelSimulatorTests(unittest.TestCase):
  def test_real_mujoco_serial_parallel_scores_match(self):
    from src.tasks.supermimic.env import SuperMimicEnv
    from src.tasks.supermimic.models.kinematics import robot_metadata
    from tests.supermimic_fixtures import make_motions

    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory() as directory:
      manifest, _ = make_motions(Path(directory), robot_metadata())
      # Run complete, short clips through real physics, including clip-end resets.
      metadata = json.loads(manifest.read_text())
      for index, entry in enumerate(metadata["clips"]):
        path = manifest.parent / entry["file"]
        with np.load(path, allow_pickle=False) as source:
          arrays = {key: source[key] for key in source.files}
        frames = entry["frames"] = 7 + 4 * index
        for key in (
          "joint_pos",
          "joint_vel",
          "body_pos_w",
          "body_quat_w",
          "body_lin_vel_w",
          "body_ang_vel_w",
        ):
          arrays[key] = arrays[key][:frames]
        np.savez(path, **arrays)
      manifest.write_text(json.dumps(metadata))
      reports = []
      for width in (1, 2):
        env = SuperMimicEnv(
          FoundationConfig(manifest=manifest, num_envs=width, episode_length_s=1e9),
          device="cpu",
        )
        try:
          reports.append(evaluate(env))
        finally:
          env.close()
      for serial, parallel in zip(reports[0]["motions"], reports[1]["motions"]):
        for key in (
          "steps",
          "expected_steps",
          "success",
          "failure_reason",
          "detected_failure_reasons",
        ):
          self.assertEqual(serial[key], parallel[key])
        self.assertAlmostEqual(
          serial["normalized_return"], parallel["normalized_return"], places=5
        )
        for key in ERROR_METRICS:
          self.assertAlmostEqual(
            serial["mean_errors"][key], parallel["mean_errors"][key], places=5
          )


if __name__ == "__main__":
  unittest.main()
