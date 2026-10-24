"""Training and inference formats, export isolation and identical actor actions."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

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
  evaluation_score,
  export_policy,
  policy_checkpoint,
  restore_best_inference,
  save_checkpoint,
  save_inference_pair,
  training_config,
  world_model_checkpoint,
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
      "config": {"actor_dims": [8], "world_dims": [8], "amp": "none"},
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
    trainer.world = torch.nn.Linear(2, 2)
    trainer.world_updates = 1
    policy = trainer.policy_state_dict()
    self.assertEqual(policy["kind"], POLICY_KIND)
    self.assertNotIn("replay", policy)
    world = trainer.world_model_state_dict()
    self.assertEqual(world["kind"], "supermimic_world_model")
    self.assertNotIn("replay", world)
    self.assertNotIn("actor", world)
    trainer.replay.state_dict.assert_not_called()

  def save_pair(self, iteration, successes, reward, tag="last"):
    report = {
      "iteration": iteration,
      "successes": successes,
      "normalized_return": reward,
      "motion_count": 40,
    }
    source = {
      **self.saved,
      "iteration": iteration,
      "evaluation": report,
      "best_evaluation": report if tag == "best" else None,
      "world": {"marker": torch.tensor(iteration)},
    }
    save_inference_pair(
      self.directory, policy_checkpoint(source), world_model_checkpoint(source), tag=tag
    )
    (self.directory / f"evaluation_{iteration:06d}.json").write_text(json.dumps(report))
    return report

  def test_best_prioritizes_successes_then_return_and_keeps_matching_world(self):
    self.save_pair(10000, 40, 0.8, "best")
    self.save_pair(11000, 40, 0.85)
    self.save_pair(15000, 39, 0.95)
    report, available = restore_best_inference(self.directory, 15000)
    self.assertEqual(report["iteration"], 11000)
    self.assertTrue(available)
    record = json.loads((self.directory / "best_checkpoint.json").read_text())
    for component in ("policy", "world_model"):
      best = torch.load(self.directory / f"{component}_best.pt", weights_only=True)
      historical = torch.load(self.directory / record[component], weights_only=True)
      latest = torch.load(self.directory / f"{component}_last.pt", weights_only=True)
      self.assertEqual(best["iteration"], 11000)
      self.assertEqual(historical["evaluation"], report)
      self.assertEqual(latest["iteration"], 15000)
      self.assertNotIn("replay", historical)
    world = torch.load(self.directory / "world_model_best.pt", weights_only=True)
    self.assertEqual(int(world["world"]["marker"]), 11000)

  def test_rollback_excludes_future_best_and_exact_ties_keep_earlier(self):
    first = self.save_pair(10000, 40, 0.8)
    self.save_pair(11000, 40, 0.8)
    future = self.save_pair(15000, 40, 0.9, "best")
    report, available = restore_best_inference(self.directory, 11000, future)
    self.assertEqual(report, first)
    self.assertTrue(available)

  def test_legacy_actor_only_best_does_not_borrow_latest_world(self):
    best = self.save_pair(10000, 40, 0.8)
    (self.directory / "world_model_iteration_010000.pt").unlink()
    self.save_pair(15000, 39, 0.95, "best")
    save_checkpoint(self.directory / "last.pt", {**self.saved, "iteration": 15000})
    report, available = restore_best_inference(self.directory, 15000)
    self.assertEqual(report, best)
    self.assertFalse(available)
    self.assertFalse((self.directory / "world_model_best.pt").exists())
    self.assertTrue((self.directory / "world_model_previous_best.pt").exists())
    record = json.loads((self.directory / "best_checkpoint.json").read_text())
    self.assertIsNone(record["world_model"])
    self.assertEqual(record["policy"], "policy_iteration_010000.pt")

  def test_resume_can_recover_pair_from_legacy_full_snapshot(self):
    report = {"iteration": 7000, "successes": 40, "normalized_return": 0.8}
    (self.directory / "evaluation_007000.json").write_text(json.dumps(report))
    save_checkpoint(self.directory / "iteration_007000.pt", self.saved)
    recovered, available = restore_best_inference(self.directory, 8000)
    self.assertEqual(recovered["iteration"], 7000)
    self.assertTrue(available)
    self.assertTrue((self.directory / "world_model_iteration_007000.pt").exists())

  def test_pair_rejects_different_iterations_before_writing(self):
    policy, world = policy_checkpoint(self.saved), world_model_checkpoint(self.saved)
    world["iteration"] += 1
    with self.assertRaisesRegex(ValueError, "differs in iteration"):
      save_inference_pair(self.directory, policy, world, tag="last")
    self.assertFalse(list(self.directory.glob("*.pt")))

  def test_incomplete_best_save_does_not_publish_new_best_record(self):
    self.save_pair(10000, 39, 0.8, "best")
    previous = (self.directory / "best_checkpoint.json").read_text()
    real_save = save_checkpoint

    def fail_world(path, data):
      if path.name == "world_model_iteration_015000.pt":
        raise OSError("disk full")
      real_save(path, data)

    with (
      patch("src.tasks.supermimic.rl.checkpoints.save_checkpoint", fail_world),
      self.assertRaisesRegex(OSError, "disk full"),
    ):
      self.save_pair(15000, 40, 0.9, "best")
    self.assertEqual((self.directory / "best_checkpoint.json").read_text(), previous)

  def test_nonfinite_evaluation_is_not_eligible_for_best(self):
    for reward in (float("nan"), float("inf"), -float("inf")):
      with self.subTest(reward=reward), self.assertRaisesRegex(ValueError, "finite"):
        evaluation_score({"successes": 40, "normalized_return": reward})

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

  def test_cli_resume_cannot_toggle_world_model_normalization(self):
    source = self.directory / "last.pt"
    for enabled in (False, True):
      saved = {
        **self.saved,
        "config": {**self.saved["config"], "world_model_normalization": enabled},
        "sampler": {"schema_version": 2},
        "evaluation_limit": None,
      }
      save_checkpoint(source, saved)
      flag = (
        "--no-world-model-normalization" if enabled else "--world-model-normalization"
      )
      result = subprocess.run(
        [
          sys.executable,
          "scripts/train_supermimic.py",
          "--device",
          "cpu",
          "--manifest",
          str(self.manifest),
          "--resume",
          str(source),
          flag,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
      )
      self.assertNotEqual(result.returncode, 0)
      self.assertIn("Resume cannot change --world-model-normalization", result.stderr)

  def test_cli_resume_cannot_reinterpret_world_prediction_targets(self):
    source = self.directory / "last.pt"
    for saved_mode, requested in (
      (None, "state_delta"),
      ("state_delta", "velocity_residual"),
    ):
      with self.subTest(saved_mode=saved_mode):
        saved = {
          **self.saved,
          "config": dict(self.saved["config"]),
          "sampler": {"schema_version": 2},
          "evaluation_limit": None,
        }
        if saved_mode is not None:
          saved["config"]["world_prediction"] = saved_mode
        save_checkpoint(source, saved)
        result = subprocess.run(
          [
            sys.executable,
            "scripts/train_supermimic.py",
            "--device",
            "cpu",
            "--manifest",
            str(self.manifest),
            "--resume",
            str(source),
            "--world-prediction",
            requested,
          ],
          cwd=ROOT,
          capture_output=True,
          text=True,
          timeout=30,
          check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Resume cannot change --world-prediction", result.stderr)

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
        self.assertEqual(config["world_prediction"], "velocity_residual")
        self.assertFalse(config["world_model_normalization"])

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
