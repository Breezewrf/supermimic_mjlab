"""Gradient clipping must distinguish norm overflow from invalid gradients."""

import copy
import unittest
from unittest.mock import patch

import torch

from src.tasks.supermimic.config.rl_cfg import OnlineConfig
from src.tasks.supermimic.rl.online_training import checked_step


class CheckedStepTests(unittest.TestCase):
  def test_finite_gradients_with_overflowing_fp32_norm_are_clipped(self):
    module = torch.nn.Linear(2, 1, bias=False)
    torch.nn.init.ones_(module.weight)
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    # Finite loss and gradients, but their squares overflow in FP32.
    inputs = torch.tensor([[1e20, 2e20]])
    before = module.weight.detach().clone()
    norm = checked_step(module(inputs).sum(), module, optimizer, element_clip=0.45)
    expected_norm = torch.linalg.vector_norm(inputs.double())
    self.assertAlmostEqual(norm / float(expected_norm), 1.0, places=6)
    expected_grad = (
      (inputs.double() / (expected_norm + 1e-6)).float().clamp(-0.45, 0.45)
    )
    torch.testing.assert_close(module.weight.grad, expected_grad)
    torch.testing.assert_close(module.weight, before - 0.1 * expected_grad)
    self.assertGreater(float(module.weight.grad.abs().sum()), 0.0)

  def test_ordinary_clipping_matches_pytorch_and_preserves_element_order(self):
    module = torch.nn.Linear(2, 1, bias=False)
    reference = copy.deepcopy(module)
    inputs = torch.tensor([[0.2, 4.0]])
    expected = reference(inputs).sum()
    expected.backward()
    expected_norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.0)
    reference.weight.grad.clamp_(-0.1, 0.1)
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    norm = checked_step(module(inputs).sum(), module, optimizer, element_clip=0.1)
    self.assertEqual(norm, float(expected_norm))
    torch.testing.assert_close(module.weight.grad, reference.weight.grad)

  def test_nonfinite_gradients_do_not_modify_weights_or_adam_state(self):
    for invalid in (float("nan"), float("inf")):
      with self.subTest(gradient=invalid):
        module = torch.nn.Linear(2, 1, bias=False)
        optimizer = torch.optim.Adam(module.parameters())
        checked_step(module(torch.ones(1, 2)).sum(), module, optimizer)
        optimizer.zero_grad(set_to_none=True)
        before = module.weight.detach().clone()
        state = copy.deepcopy(optimizer.state_dict())
        with (
          module.weight.register_hook(lambda gradient, value=invalid: gradient * value),
          self.assertRaisesRegex(FloatingPointError, "Linear: weight"),
        ):
          checked_step(module(torch.ones(1, 2)).sum(), module, optimizer)
        torch.testing.assert_close(module.weight, before, rtol=0, atol=0)
        self.assertIsNone(module.weight.grad)
        for index, saved in state["state"].items():
          for key, value in saved.items():
            torch.testing.assert_close(
              optimizer.state_dict()["state"][index][key], value
            )

  def test_nonfinite_loss_clears_gradients_without_stepping(self):
    module = torch.nn.Linear(2, 1, bias=False)
    optimizer = torch.optim.Adam(module.parameters())
    module.weight.grad = torch.ones_like(module.weight)
    with self.assertRaisesRegex(FloatingPointError, "training loss in Linear"):
      checked_step(module.weight.sum() * float("inf"), module, optimizer)
    self.assertIsNone(module.weight.grad)
    self.assertFalse(optimizer.state)

  def test_unrelated_runtime_errors_are_not_treated_as_overflow(self):
    module = torch.nn.Linear(2, 1, bias=False)
    optimizer = torch.optim.Adam(module.parameters())
    with (
      patch("torch.nn.utils.clip_grad_norm_", side_effect=RuntimeError("other error")),
      self.assertRaisesRegex(RuntimeError, "other error"),
    ):
      checked_step(module.weight.sum(), module, optimizer)
    self.assertFalse(optimizer.state)

  def test_learning_rate_must_be_positive_and_finite(self):
    for rate in (0.0, -1e-4, float("inf"), float("nan")):
      with self.subTest(rate=rate), self.assertRaises(ValueError):
        OnlineConfig(learning_rate=rate)
    self.assertEqual(OnlineConfig(learning_rate=5e-5).learning_rate, 5e-5)


if __name__ == "__main__":
  unittest.main()
