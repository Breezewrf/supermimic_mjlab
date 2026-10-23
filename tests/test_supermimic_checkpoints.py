"""Training and inference formats, export isolation and identical actor actions."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from src.tasks.supermimic.config.rl_cfg import OnlineConfig
from src.tasks.supermimic.data.world_model_data import file_sha256
from src.tasks.supermimic.models.policy import REFERENCE_SCHEMA, MotionActor
from src.tasks.supermimic.rl.checkpoints import (
  POLICY_KIND,
  export_policy,
  policy_checkpoint,
  save_checkpoint,
)
from src.tasks.supermimic.rl.inference import load_actor
from src.tasks.supermimic.rl.online_training import OnlineTrainer

ROOT = Path(__file__).resolve().parents[1]


class CheckpointTests(unittest.TestCase):
  def setUp(self):
    folder = tempfile.TemporaryDirectory()
    self.addCleanup(folder.cleanup)
    self.directory = Path(folder.name)
    self.manifest = self.directory / "manifest.json"
    self.manifest.write_text("{}")
    self.actor = MotionActor(torch.zeros(29), (8,))
    self.saved = {
      "kind": "supermimic_online",
      "schema_version": 1,
      "reference_schema": REFERENCE_SCHEMA,
      "manifest_sha256": file_sha256(self.manifest),
      "config": {"actor_dims": [8], "amp": "none"},
      "robot_metadata": {"default_q": self.actor.default_q, "signature": "fixture"},
      "simulation": {"physics_protocol_version": 2},
      "clips": ["fixture"],
      "actor": self.actor.state_dict(),
      "iteration": 7000,
      "environment_steps": 1000000,
      "replay": {"data": torch.zeros(1024 * 1024)},
      "world": {"unused": torch.ones(1)},
      "critic": {},
      "actor_optimizer": {},
      "rng": {},
    }

  def test_export_is_small_preserves_source_and_produces_identical_actions(self):
    source, output = self.directory / "last.pt", self.directory / "policy.pt"
    save_checkpoint(source, self.saved)
    original_stat = source.stat()
    exported = export_policy(source, output)
    self.assertEqual(exported["kind"], POLICY_KIND)
    self.assertEqual(exported["iteration"], 7000)
    self.assertFalse(
      set(exported) & {"replay", "world", "critic", "actor_optimizer", "rng"}
    )
    self.assertLess(output.stat().st_size, source.stat().st_size / 10)
    self.assertEqual(source.stat().st_mtime_ns, original_stat.st_mtime_ns)
    original, _, _ = load_actor(source, self.manifest, "cpu")
    lightweight, _, _ = load_actor(output, self.manifest, "cpu")
    history = torch.randn(2, 10, 93)
    reference = {
      "body_pos_w": torch.randn(2, 10, 1, 3),
      "body_quat_w": torch.tensor([1.0, 0, 0, 0]).expand(2, 10, 1, 4),
      "body_lin_vel_w": torch.randn(2, 10, 1, 3),
      "body_ang_vel_w": torch.randn(2, 10, 1, 3),
      "joint_pos": torch.randn(2, 10, 29),
      "joint_vel": torch.randn(2, 10, 29),
    }
    with torch.no_grad():
      torch.testing.assert_close(
        original(history, reference), lightweight(history, reference), rtol=0, atol=0
      )
    with self.assertRaisesRegex(ValueError, "differ"):
      export_policy(source, source)

  def test_export_clones_tensor_views_without_retaining_training_storage(self):
    shared = torch.arange(1024, dtype=torch.float32)
    self.saved["actor"] = {"view": shared[:2]}
    policy = policy_checkpoint(self.saved)
    view = policy["actor"]["view"]
    self.assertEqual(view.untyped_storage().nbytes(), 2 * view.element_size())
    view[0] = -100
    self.assertEqual(float(shared[0]), 0)

  def test_training_policy_export_never_serializes_replay(self):
    trainer = object.__new__(OnlineTrainer)
    trainer.config = OnlineConfig(actor_dims=(8,))
    trainer.metadata = self.saved["robot_metadata"]
    trainer.simulation_parameters = self.saved["simulation"]
    trainer.env = SimpleNamespace(library=SimpleNamespace(names=["fixture"]))
    trainer.actor, trainer.iteration, trainer.environment_steps = self.actor, 1, 10
    trainer.replay = MagicMock()
    trainer.replay.state_dict.side_effect = AssertionError("Replay was serialized")
    policy = trainer.policy_state_dict()
    self.assertEqual(policy["kind"], POLICY_KIND)
    self.assertNotIn("replay", policy)
    trainer.replay.state_dict.assert_not_called()

  def test_cli_export_and_inference_resume_rejection(self):
    source, output = self.directory / "last.pt", self.directory / "policy.pt"
    save_checkpoint(source, self.saved)
    result = subprocess.run(
      [
        sys.executable,
        "scripts/export_supermimic_policy.py",
        "--checkpoint",
        str(source),
        "--output",
        str(output),
      ],
      cwd=ROOT,
      capture_output=True,
      text=True,
      timeout=30,
      check=False,
    )
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    self.assertIn("MiB ->", result.stdout)
    result = subprocess.run(
      [
        sys.executable,
        "scripts/train_supermimic.py",
        "--device",
        "cpu",
        "--resume",
        str(output),
      ],
      cwd=ROOT,
      capture_output=True,
      text=True,
      timeout=30,
      check=False,
    )
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("policy weights cannot resume training", result.stderr)

  def test_missing_metadata_is_rejected(self):
    self.saved.pop("simulation")
    with self.assertRaisesRegex(ValueError, "simulation"):
      policy_checkpoint(self.saved)


if __name__ == "__main__":
  unittest.main()
