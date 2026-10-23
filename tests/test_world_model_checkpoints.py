"""Portable world-model exports preserve predictions without replay or Actor data."""

import subprocess
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch

from src.tasks.supermimic.config.rl_cfg import OnlineConfig
from src.tasks.supermimic.models.kinematics import robot_metadata
from src.tasks.supermimic.models.world_model import WorldModel
from src.tasks.supermimic.rl.checkpoints import (
  WORLD_MODEL_KIND,
  WORLD_MODEL_REQUIRED_FIELDS,
  export_policy,
  export_world_model,
  save_checkpoint,
  world_model_checkpoint,
)

ROOT = Path(__file__).resolve().parents[1]


class WorldModelCheckpointTests(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    torch.set_num_threads(2)
    cls.metadata = robot_metadata()

  def setUp(self):
    folder = tempfile.TemporaryDirectory()
    self.addCleanup(folder.cleanup)
    self.directory = Path(folder.name)
    self.config = OnlineConfig(
      world_dims=(16, 8), relative_orientation=True, amp="none"
    )
    self.world = WorldModel(
      self.metadata, self.config.world_dims, relative_orientation=True
    ).eval()
    with torch.no_grad():
      self.world.input_mean.uniform_(-0.1, 0.1)
      self.world.input_std.uniform_(0.5, 1.5)
      self.world.correction_scale.uniform_(0.01, 0.1)
      self.world.dynamics[-1].weight.uniform_(-0.05, 0.05)
      self.world.dynamics[-1].bias.uniform_(-0.05, 0.05)
    self.saved = {
      "kind": "supermimic_online",
      "schema_version": 1,
      "config": asdict(self.config),
      "robot_metadata": self.metadata,
      "simulation": {"physics_protocol_version": 2, "control_dt": 0.02},
      "clips": ["fixture"],
      "manifest_sha256": "fixture-manifest-hash",
      "world": self.world.state_dict(),
      "iteration": 7000,
      "world_updates": 70000,
      "environment_steps": 1000000,
      "replay": {"data": torch.zeros(1024 * 1024)},
      "actor": {"unused": torch.ones(1)},
      "critic": {},
      "world_optimizer": {},
      "actor_optimizer": {},
      "critic_optimizer": {},
      "rng": {},
    }

  def restore(self, exported):
    config = exported["config"]
    world = WorldModel(
      exported["robot_metadata"],
      tuple(config["world_dims"]),
      config.get("termination_mode", "learned"),
      config.get("relative_orientation", False),
      config.get("amp", "none"),
    )
    world.load_state_dict(exported["world"])
    return world.eval()

  def test_export_is_small_and_preserves_source_and_exact_multistep_predictions(self):
    source, output = self.directory / "last.pt", self.directory / "world.pt"
    save_checkpoint(source, self.saved)
    original_stat = source.stat()
    with patch("torch.load", wraps=torch.load) as load:
      exported = export_world_model(source, output)
    load.assert_called_once_with(
      source, map_location="cpu", weights_only=True, mmap=True
    )
    self.assertEqual(exported["kind"], WORLD_MODEL_KIND)
    self.assertEqual(exported["iteration"], 7000)
    self.assertEqual(exported["world_updates"], 70000)
    self.assertEqual(exported["source_checkpoint"], str(source.resolve()))
    self.assertLess(output.stat().st_size, source.stat().st_size / 10)
    self.assertEqual(source.stat().st_mtime_ns, original_stat.st_mtime_ns)
    self.assertFalse(
      set(exported)
      & {
        "replay",
        "actor",
        "critic",
        "world_optimizer",
        "actor_optimizer",
        "critic_optimizer",
        "rng",
      }
    )
    reloaded = torch.load(output, map_location="cpu", weights_only=True)
    restored = self.restore(reloaded)
    for key, value in self.world.state_dict().items():
      torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)
    state = torch.zeros(3, 71)
    state[:, 2], state[:, 3] = 0.8, 1.0
    state[:, 7:36] = self.metadata["default_q"].float()
    context = torch.zeros(3, 33)
    context[:, 0] = 1.0
    reference = self.world.kinematics(state)
    action, previous = torch.randn(3, 29) * 0.1, torch.zeros(3, 29)
    with torch.no_grad():
      for _ in range(4):
        expected = self.world.predict(state, action, context, reference, previous)
        actual = restored.predict(state, action, context, reference, previous)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        state, previous = expected["next_state"], action

  def test_export_does_not_visit_actor_replay_or_optimizers_or_require_reference_schema(
    self,
  ):
    for key in ("actor", "replay", "world_optimizer"):
      self.saved[key] = MagicMock()
      self.saved[key].items.side_effect = AssertionError("Training storage was visited")
    exported = world_model_checkpoint(self.saved)
    self.assertNotIn("reference_schema", exported)
    self.restore(exported)
    del self.saved["actor"]
    self.assertEqual(world_model_checkpoint(self.saved)["kind"], WORLD_MODEL_KIND)

  def test_export_clones_normalization_views_and_metadata(self):
    shared = torch.arange(4096, dtype=torch.float32)
    self.saved["world"]["input_mean"] = shared[:130]
    exported = world_model_checkpoint(self.saved)
    view = exported["world"]["input_mean"]
    self.assertEqual(view.untyped_storage().nbytes(), 130 * view.element_size())
    view[0] = -100
    self.assertEqual(float(shared[0]), 0)
    exported["robot_metadata"]["default_q"].fill_(100)
    self.assertFalse(bool((self.metadata["default_q"] == 100).all()))

  def test_missing_metadata_and_unsupported_formats_are_rejected(self):
    for field in WORLD_MODEL_REQUIRED_FIELDS:
      saved = {key: value for key, value in self.saved.items() if key != field}
      with self.subTest(field=field), self.assertRaises(ValueError):
        world_model_checkpoint(saved)
    saved = {**self.saved, "config": {}}
    with self.assertRaisesRegex(ValueError, "world_dims"):
      world_model_checkpoint(saved)
    with self.assertRaisesRegex(ValueError, "Policy-only checkpoints"):
      world_model_checkpoint({"kind": "supermimic_actor", "schema_version": 1})
    with self.assertRaisesRegex(ValueError, "Expected a SuperMimic"):
      world_model_checkpoint({**self.saved, "kind": "unknown"})

  def test_world_artifact_can_be_reexported_but_cannot_be_exported_as_policy(self):
    source = self.directory / "world.pt"
    save_checkpoint(source, world_model_checkpoint(self.saved))
    with self.assertRaisesRegex(ValueError, "differ"):
      export_world_model(source, source)
    with self.assertRaises(ValueError):
      export_policy(source, self.directory / "policy.pt")
    output = self.directory / "copy.pt"
    self.restore(export_world_model(source, output))

  def test_cli_export_and_policy_only_error(self):
    source, output = self.directory / "last.pt", self.directory / "world.pt"
    save_checkpoint(source, self.saved)
    command = [
      sys.executable,
      "scripts/export_supermimic_policy.py",
      "--component",
      "world-model",
      "--checkpoint",
      str(source),
      "--output",
      str(output),
    ]
    result = subprocess.run(
      command, cwd=ROOT, capture_output=True, text=True, timeout=30, check=False
    )
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    self.assertIn("World-model weights:", result.stdout)
    self.assertIn("MiB ->", result.stdout)
    self.restore(torch.load(output, weights_only=True, map_location="cpu"))
    save_checkpoint(source, {"kind": "supermimic_actor", "schema_version": 1})
    output.unlink()
    result = subprocess.run(
      command, cwd=ROOT, capture_output=True, text=True, timeout=30, check=False
    )
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("Policy-only checkpoints do not contain a world model", result.stderr)
    self.assertFalse(output.exists())


if __name__ == "__main__":
  unittest.main()
