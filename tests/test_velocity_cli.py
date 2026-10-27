"""Velocity CLI training and resume with the real offline W&B service."""

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from src.tasks.supermimic.models.kinematics import robot_metadata
from tests.supermimic_fixtures import make_motions


class VelocityOutputTests(unittest.TestCase):
  def test_new_training_rejects_existing_output_without_changing_files(self):
    with tempfile.TemporaryDirectory() as directory:
      output = Path(directory)
      originals = {
        "config.json": b'{"original_run": true}\n',
        "metrics.jsonl": b'{"iteration": 10}\n',
        "last.pt": b"original checkpoint",
        "wandb_run.json": b'{"id": "original"}\n',
      }
      for name, contents in originals.items():
        (output / name).write_bytes(contents)
      result = subprocess.run(
        [
          sys.executable,
          "scripts/train_velocity.py",
          "--from-scratch",
          "--device",
          "cpu",
          "--output",
          str(output),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
      )
      self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
      self.assertIn("Output exists", result.stderr)
      self.assertEqual(
        {path.name: path.read_bytes() for path in output.iterdir()}, originals
      )

  def test_resume_rejects_another_output_directory(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      source, destination = root / "source", root / "other_run"
      source.mkdir()
      destination.mkdir()
      checkpoint = source / "last.pt"
      checkpoint.write_bytes(b"original checkpoint")
      config = destination / "config.json"
      config.write_bytes(b"original config")
      result = subprocess.run(
        [
          sys.executable,
          "scripts/train_velocity.py",
          "--device",
          "cpu",
          "--resume",
          str(checkpoint),
          "--output",
          str(destination),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
      )
      self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
      self.assertIn("checkpoint's existing output directory", result.stderr)
      self.assertEqual(checkpoint.read_bytes(), b"original checkpoint")
      self.assertEqual(config.read_bytes(), b"original config")
      self.assertEqual(list(destination.iterdir()), [config])


@unittest.skipUnless(
  importlib.util.find_spec("wandb"), "optional W&B SDK not installed"
)
class VelocityCliTests(unittest.TestCase):
  def test_cpu_warmup_joint_resume_and_offline_tracking(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      manifest, _ = make_motions(root / "motions", robot_metadata())
      output = root / "run"
      common = [
        sys.executable,
        "scripts/train_velocity.py",
        "--manifest",
        str(manifest),
        "--device",
        "cpu",
        "--output",
        str(output),
        "--wandb",
        "--wandb-mode",
        "offline",
        "--log-interval",
        "1",
      ]
      first = common + [
        "--from-scratch",
        "--stage",
        "joint",
        "--warmup-updates",
        "1",
        "--num-envs",
        "4",
        "--capacity",
        "32",
        "--warmup-steps",
        "12",
        "--collect-steps",
        "3",
        "--updates-per-collect",
        "1",
        "--wm-batches",
        "1",
        "--wm-batch-size",
        "4",
        "--policy-batch-size",
        "4",
        "--horizon",
        "3",
        "--critic-epochs",
        "1",
        "--critic-minibatches",
        "1",
        "--world-dims",
        "16",
        "16",
        "--actor-dims",
        "16",
        "8",
        "--critic-dims",
        "16",
        "8",
        "--initial-evaluation-steps",
        "0",
        "--iterations",
        "2",
        "--clips",
        "fixture_walk",
        "fixture_run",
      ]
      result = subprocess.run(
        first, capture_output=True, text=True, timeout=120, check=False
      )
      self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
      self.assertIn("Stage: warmup", result.stdout)
      self.assertIn("Stage: joint", result.stdout)
      self.assertIn("W&B offline:", result.stdout)
      previous = json.loads((output / "wandb_run.json").read_text())
      checkpoint = torch.load(output / "last.pt", weights_only=True)
      self.assertEqual(checkpoint["wandb_run"]["id"], previous["id"])
      self.assertEqual(checkpoint["iteration"], 2)
      self.assertNotIn("resample_steps", checkpoint["velocity_config"])
      self.assertTrue(
        checkpoint["velocity_config"]["randomize_initial_episode_progress"]
      )
      # Omitting --output on resume must select the checkpoint's directory.
      resume_common = common.copy()
      output_index = resume_common.index("--output")
      del resume_common[output_index : output_index + 2]
      result = subprocess.run(
        resume_common + ["--resume", str(output / "last.pt"), "--iterations", "3"],
        capture_output=True,
        check=False,
        text=True,
        timeout=120,
      )
      self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
      current = json.loads((output / "wandb_run.json").read_text())
      self.assertEqual(current["last_iteration"], 3)
      self.assertNotEqual(current["id"], previous["id"])
      rows = [
        json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()
      ]
      self.assertEqual([row["iteration"] for row in rows], [1, 2, 3])
      self.assertEqual([row["stage"] for row in rows], ["warmup", "joint", "joint"])
      self.assertTrue(list((output / "wandb").glob("offline-run-*/*.wandb")))
      self.assertTrue((output / "velocity_policy_last.pt").exists())
      self.assertTrue(
        all(row["statistics_scope"] == "completed_episodes" for row in rows)
      )
      self.assertTrue(all(row["metric_schema_version"] == 2 for row in rows))
      # Current checkpoints support explicit timing overrides on resume.
      result = subprocess.run(
        resume_common
        + [
          "--resume",
          str(output / "last.pt"),
          "--iterations",
          "4",
          "--command-hold-seconds",
          "2",
          "4",
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
      )
      self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
      resumed = torch.load(output / "last.pt", weights_only=True)
      self.assertEqual(resumed["iteration"], 4)
      self.assertNotIn("resample_steps", resumed["velocity_config"])
      self.assertTrue(resumed["velocity_config"]["randomize_initial_episode_progress"])
      self.assertEqual(resumed["velocity_config"]["resampling_time_range"], (2.0, 4.0))
      self.assertIn("episode_rng", resumed)
