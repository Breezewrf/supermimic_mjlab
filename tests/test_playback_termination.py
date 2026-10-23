"""Playback survives physical failures while retaining clip boundaries and diagnostics."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch

from src.tasks.supermimic.config.task_cfg import FoundationConfig
from src.tasks.supermimic.models.kinematics import robot_metadata
from src.tasks.supermimic.rl.evaluation import evaluate, evaluation_summary
from tests.supermimic_fixtures import make_motions

ROOT = Path(__file__).resolve().parents[1]


class PlaybackTerminationTests(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    from src.tasks.supermimic.env import SuperMimicEnv

    torch.set_num_threads(2)
    folder = tempfile.TemporaryDirectory()
    cls.addClassCleanup(folder.cleanup)
    cls.directory = Path(folder.name)
    cls.manifest, names = make_motions(cls.directory / "motions", robot_metadata())
    cls.config = FoundationConfig(
      manifest=cls.manifest,
      clips=(names[0],),
      num_envs=1,
      episode_length_s=1e9,
    )
    cls.env = SuperMimicEnv(cls.config, device="cpu")
    cls.addClassCleanup(cls.env.close)

  def setUp(self):
    self.env.foundation_config = self.config
    self.env.reset(seed=0, options={"motion_id": 0, "frame_id": 0})
    self.zero = torch.zeros(1, 29)

  def displace_robot(self, *, tilt=False):
    env = self.env
    state = env.read_state()
    root = torch.cat((state[:, :7], state[:, 36:42]), -1)
    root[:, :3] += env.scene.env_origins
    root[:, 2] += 0.6
    if tilt:
      root[:, 3:7] = torch.tensor([0.0, 1.0, 0.0, 0.0])
    env.robot.write_root_state_to_sim(root)
    env.scene.write_data_to_sim()
    env.sim.forward()

  def test_training_and_evaluation_keep_failure_termination_by_default(self):
    self.assertTrue(FoundationConfig().terminate_on_failure)
    self.assertTrue(FoundationConfig.training().terminate_on_failure)
    self.displace_robot()
    episode = self.env.episode_id.clone()
    _, _, terminal, _, extras = self.env.step(self.zero)
    self.assertTrue(bool(extras["metrics"]["height_failure"][0]))
    self.assertTrue(bool(terminal[0]))
    torch.testing.assert_close(self.env.episode_id, episode + 1)

  def test_disabled_height_and_tilt_failures_do_not_reset_episodes(self):
    self.env.foundation_config = replace(self.config, terminate_on_failure=False)
    self.displace_robot(tilt=True)
    episode = self.env.episode_id.clone()
    for frame in (1, 2):
      _, _, terminal, cutoff, extras = self.env.step(self.zero)
      self.assertTrue(bool(extras["metrics"]["height_failure"][0]))
      self.assertTrue(bool(extras["metrics"]["tilt_failure"][0]))
      self.assertFalse(bool(terminal[0]))
      self.assertFalse(bool(cutoff[0]))
      self.assertFalse(bool(extras["transition"]["terminated"][0]))
      torch.testing.assert_close(self.env.episode_id, episode)
      self.assertEqual(int(self.env.frame_id[0]), frame)

  def test_clip_end_still_resets_after_a_failure(self):
    self.env.foundation_config = replace(self.config, terminate_on_failure=False)
    self.env.reset(
      options={"motion_id": 0, "frame_id": int(self.env.library.lengths[0]) - 2}
    )
    self.displace_robot()
    episode = self.env.episode_id.clone()
    _, _, terminal, cutoff, extras = self.env.step(self.zero)
    self.assertFalse(bool(terminal[0]))
    self.assertTrue(bool(cutoff[0]))
    self.assertTrue(bool(extras["transition"]["motion_end"][0]))
    torch.testing.assert_close(self.env.episode_id, episode + 1)

  def test_timeout_still_resets_when_failure_termination_is_disabled(self):
    self.env.foundation_config = replace(
      self.config, terminate_on_failure=False, episode_length_s=0.02
    )
    self.displace_robot()
    episode = self.env.episode_id.clone()
    _, _, terminal, cutoff, extras = self.env.step(self.zero)
    self.assertFalse(bool(terminal[0]))
    self.assertTrue(bool(cutoff[0]))
    self.assertTrue(bool(extras["transition"]["time_out"][0]))
    torch.testing.assert_close(self.env.episode_id, episode + 1)

  def test_full_playback_reports_failures_without_stopping_or_counting_success(self):
    self.env.foundation_config = replace(self.config, terminate_on_failure=False)

    def displace(env, clip, frame, state, reward):
      if frame == 0:
        self.displace_robot()

    report = evaluate(self.env, on_frame=displace)
    row = report["motions"][0]
    self.assertEqual(row["steps"], row["expected_steps"])
    self.assertTrue(report["evaluation_complete"])
    self.assertFalse(report["termination_enabled"])
    self.assertEqual(report["evaluation_mode"], "playback")
    self.assertEqual(report["successes"], 0)
    self.assertEqual(report["n_failed"], 1)
    self.assertFalse(row["terminated"])
    self.assertEqual(row["first_failure_step"], 1)
    self.assertIn("height_failure", row["detected_failure_reasons"])
    self.assertIn("PLAYBACK", evaluation_summary(report))

  def test_cli_defaults_and_playback_override(self):
    for script, options, termination, width in (
      ("play_supermimic.py", [], False, 1),
      ("play_supermimic.py", ["--termination"], True, 1),
      ("evaluate_supermimic.py", [], True, 2),
      ("evaluate_supermimic.py", ["--num-envs", "1"], True, 1),
    ):
      with self.subTest(script=script, options=options):
        output = self.directory / "cli.json"
        result = subprocess.run(
          [
            sys.executable,
            f"scripts/{script}",
            "--manifest",
            str(self.manifest),
            "--device",
            "cpu",
            "--viewer",
            "none",
            "--max-steps-per-clip",
            "2",
            "--policy",
            "src.tasks.supermimic.rl.evaluation:zero_policy",
            "--output",
            str(output),
            *options,
          ],
          cwd=ROOT,
          env=os.environ,
          capture_output=True,
          text=True,
          timeout=120,
          check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report = json.loads(output.read_text())
        self.assertEqual(report["termination_enabled"], termination)
        self.assertEqual(report["evaluation_num_envs"], width)
        self.assertEqual(
          report["evaluation_mode"], "standard" if termination else "playback"
        )


if __name__ == "__main__":
  unittest.main()
