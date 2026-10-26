"""Combined world-model ablation, physical decoding, gradients and portable stats."""

import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn

from src.tasks.supermimic.config.rl_cfg import OnlineConfig
from src.tasks.supermimic.data.world_model_data import file_sha256
from src.tasks.supermimic.models.kinematics import robot_metadata
from src.tasks.supermimic.models.world_model import (
  WorldModel,
  apply_state_delta,
  integrate,
)
from src.tasks.supermimic.rl.checkpoints import save_checkpoint, world_model_checkpoint
from src.tasks.supermimic.rl.inference import load_world_model
from tests.test_world_model import poses


class WorldModelNormalizationTests(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    torch.set_num_threads(2)
    cls.metadata = robot_metadata()

  def samples(self):
    state = poses(8)
    action, context = torch.randn(8, 29), torch.randn(8, 33)
    delta = torch.randn(8, 70) * 0.02 + 0.01
    return state, action, context, delta

  def test_one_switch_changes_hidden_layers_and_both_normalizers(self):
    original = WorldModel(self.metadata, (16,))
    enhanced = WorldModel(self.metadata, (16,), world_model_normalization=True)
    self.assertEqual(
      [type(m) for m in original.dynamics], [nn.Linear, nn.ELU, nn.Linear]
    )
    self.assertEqual(
      [type(m) for m in enhanced.dynamics],
      [nn.Linear, nn.LayerNorm, nn.SiLU, nn.Linear],
    )
    self.assertFalse(
      any(key.startswith(("in_norm.", "out_norm.")) for key in original.state_dict())
    )
    for name in ("in_norm", "out_norm"):
      self.assertIn(f"{name}.count", enhanced.state_dict())

  def test_running_statistics_and_zero_head_decode_mean_in_physical_units(self):
    state, action, context, delta = self.samples()
    for mode in ("state_delta", "velocity_residual"):
      with self.subTest(mode=mode):
        model = WorldModel(
          self.metadata, (16,), world_prediction=mode, world_model_normalization=True
        )
        apply = apply_state_delta if mode == "state_delta" else integrate
        target = apply(state, delta)
        for section in (slice(0, 3), slice(3, None)):
          model.update_normalization(
            state[section], action[section], context[section], target[section]
          )
        self.assertEqual(int(model.in_norm.count), 8)
        self.assertEqual(int(model.out_norm.count), 8)
        torch.testing.assert_close(model.in_norm.mean[68:97], action.double().mean(0))
        torch.testing.assert_close(
          model.out_norm.mean, delta.double().mean(0), atol=2e-7, rtol=2e-5
        )
        torch.testing.assert_close(
          model.out_norm.var,
          delta.double().var(0, unbiased=False),
          atol=2e-8,
          rtol=2e-5,
        )
        torch.testing.assert_close(
          model(state, action, context),
          apply(state, delta.mean(0).expand(8, -1)),
          atol=2e-7,
          rtol=2e-5,
        )

  def test_multistep_forward_keeps_statistics_fixed_and_input_gradients(self):
    model = WorldModel(self.metadata, (16,), world_model_normalization=True)
    state, action, context, delta = self.samples()
    model.update_normalization(state, action, context, apply_state_delta(state, delta))
    torch.nn.init.normal_(model.dynamics[-1].weight, std=0.01)
    before = {
      key: value.clone()
      for key, value in model.state_dict().items()
      if key.startswith(("in_norm.", "out_norm."))
    }
    state, action = state.requires_grad_(), action.requires_grad_()
    predicted = state
    for _ in range(8):
      predicted = model(predicted, action, context)
    predicted.square().mean().backward()
    self.assertTrue(torch.isfinite(state.grad).all())
    self.assertTrue(torch.isfinite(action.grad).all())
    self.assertGreater(float(action.grad.abs().max()), 0)
    for key, value in before.items():
      torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
    model.eval()
    model.update_normalization(
      state, action + 10, context, apply_state_delta(state, delta)
    )
    for key, value in before.items():
      torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)

  def test_export_load_preserves_eight_step_predictions_and_freezes_statistics(self):
    model = WorldModel(self.metadata, (16,), world_model_normalization=True)
    state, action, context, delta = self.samples()
    model.update_normalization(state, action, context, apply_state_delta(state, delta))
    torch.nn.init.normal_(model.dynamics[-1].weight, std=0.01)
    with tempfile.TemporaryDirectory() as folder:
      manifest, path = Path(folder) / "manifest.json", Path(folder) / "world.pt"
      manifest.write_text("{}")
      saved = {
        "schema_version": 1,
        "kind": "supermimic_online",
        "world": model.state_dict(),
        "config": asdict(
          OnlineConfig(world_dims=(16,), world_model_normalization=True)
        ),
        "robot_metadata": self.metadata,
        "manifest_sha256": file_sha256(manifest),
        "simulation": {},
        "clips": ["fixture"],
      }
      save_checkpoint(path, world_model_checkpoint(saved))
      loaded, _ = load_world_model(path, manifest, "cpu")
      self.assertTrue(loaded.world_model_normalization)
      actual, expected = state, state
      with torch.no_grad():
        for _ in range(8):
          actual, expected = (
            loaded(actual, action, context),
            model(expected, action, context),
          )
      torch.testing.assert_close(actual, expected, rtol=0, atol=0)
      loaded.update_normalization(
        state, action, context, apply_state_delta(state, delta)
      )
      self.assertEqual(int(loaded.out_norm.count), 8)


if __name__ == "__main__":
  unittest.main()
