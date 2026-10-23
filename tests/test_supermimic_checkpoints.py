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
from src.tasks.supermimic.models.legacy_policy import (
  LEGACY_REFERENCE_SCHEMA,
  LegacyMotionActor,
)
from src.tasks.supermimic.models.policy import REFERENCE_SCHEMA, MotionActor
from src.tasks.supermimic.rl.checkpoints import (
  POLICY_KIND,
  export_policy,
  policy_checkpoint,
  save_checkpoint,
  training_config,
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
    # Reproduce historical training checkpoints: 2680 inputs, no schema field.
    self.saved.pop("reference_schema")
    self.saved["actor"]["encoder.0.weight"] = torch.randn(8, 2680)
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
    self.assertIn(f"Reference schema: {LEGACY_REFERENCE_SCHEMA}", result.stdout)
    self.assertIn("inferred", result.stdout)
    self.assertIn("select the legacy Actor automatically", result.stdout)
    exported = torch.load(output, map_location="cpu", weights_only=True)
    torch.testing.assert_close(
      exported["actor"]["encoder.0.weight"], self.saved["actor"]["encoder.0.weight"]
    )
    original, original_metadata, _ = load_actor(source, self.manifest, "cpu")
    loaded, _, _ = load_actor(output, self.manifest, "cpu")
    self.assertIsInstance(loaded, LegacyMotionActor)
    self.assertEqual(original_metadata["reference_schema"], LEGACY_REFERENCE_SCHEMA)
    self.assertEqual(loaded.encoder[0].in_features, 2680)
    reference = {
      "body_pos_w": torch.randn(2, 10, 14, 3),
      "body_quat_w": torch.tensor([1.0, 0, 0, 0]).expand(2, 10, 14, 4),
      "body_lin_vel_w": torch.randn(2, 10, 14, 3),
      "body_ang_vel_w": torch.randn(2, 10, 14, 3),
      "joint_pos": torch.randn(2, 10, 29),
      "joint_vel": torch.randn(2, 10, 29),
    }
    history = torch.randn(2, 10, 93)
    with torch.no_grad():
      torch.testing.assert_close(
        original(history, reference), loaded(history, reference), rtol=0, atol=0
      )
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

  def test_missing_current_schema_is_inferred_and_exported_policy_loads(self):
    self.saved.pop("reference_schema")
    source, output = self.directory / "last.pt", self.directory / "policy.pt"
    save_checkpoint(source, self.saved)
    exported = export_policy(source, output)
    self.assertEqual(exported["reference_schema"], REFERENCE_SCHEMA)
    self.assertTrue(exported["reference_schema_inferred"])
    self.assertNotIn("reference_schema", self.saved)
    loaded, _, _ = load_actor(output, self.manifest, "cpu")
    for key, value in self.actor.state_dict().items():
      torch.testing.assert_close(loaded.state_dict()[key], value, rtol=0, atol=0)
    repeated = policy_checkpoint(exported)
    self.assertTrue(repeated["reference_schema_inferred"])

  def test_missing_schema_with_unknown_encoder_shape_is_rejected(self):
    self.saved.pop("reference_schema")
    for actor in (
      {},
      {"encoder.0.weight": torch.zeros(8)},
      {"encoder.0.weight": torch.zeros(8, 1234)},
    ):
      with self.subTest(actor_keys=list(actor)):
        self.saved["actor"] = actor
        with self.assertRaisesRegex(ValueError, "Cannot infer reference_schema"):
          policy_checkpoint(self.saved)

  def test_existing_explicit_schema_is_preserved(self):
    self.saved["reference_schema"] = "custom_reference_v2"
    exported = policy_checkpoint(self.saved)
    self.assertEqual(exported["reference_schema"], "custom_reference_v2")
    self.assertNotIn("reference_schema_inferred", exported)

  def test_training_config_infers_historical_dimension_and_fp32_execution(self):
    del self.saved["reference_schema"]
    del self.saved["config"]["amp"]
    for width in (730, 2680):
      with self.subTest(width=width):
        self.saved["actor"]["encoder.0.weight"] = torch.zeros(8, width)
        config = training_config(self.saved)
        self.assertEqual(config["reference_dim"], width)
        self.assertEqual(config["amp"], "none")
        self.assertFalse(config["compile"])
        self.assertEqual(config["wm_only_iters"], 0)

  def test_training_config_rejects_mismatched_dimension_or_weights(self):
    self.saved["config"]["reference_dim"] = 2680
    with self.assertRaisesRegex(ValueError, "dimension and schema differ"):
      training_config(self.saved)
    self.saved["config"]["reference_dim"] = 730
    self.saved["actor"]["encoder.0.weight"] = torch.zeros(8, 2680)
    with self.assertRaisesRegex(
      ValueError, "input width and reference dimension differ"
    ):
      training_config(self.saved)


if __name__ == "__main__":
  unittest.main()
