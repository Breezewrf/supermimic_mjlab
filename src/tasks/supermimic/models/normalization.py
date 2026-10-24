"""Explicitly updated running feature statistics, frozen during forward passes."""

import torch
from torch import Tensor, nn


class RunningNorm(nn.Module):
  """Per-feature cumulative moments, with FP64 statistics and FP32 inference.

  Update only from real collection data. Forward preserves input gradients and
  never changes statistics, including while the owning model is in train mode.
  """

  def __init__(self, dim: int, eps: float = 1e-2, max_count: int = 200_000_000):
    super().__init__()
    if dim < 1 or eps <= 0 or max_count < 1:
      raise ValueError("RunningNorm requires positive dim, eps and max_count")
    self.dim, self.eps, self.max_count = dim, eps, max_count
    self.register_buffer("mean", torch.zeros(dim, dtype=torch.float64))
    self.register_buffer("var", torch.ones(dim, dtype=torch.float64))
    self.register_buffer("std", torch.ones(dim))
    self.register_buffer("count", torch.zeros((), dtype=torch.long))

  @torch.no_grad()
  def update(self, samples: Tensor) -> None:
    if not self.training or self.count.item() >= self.max_count:
      return
    if samples.shape[-1] != self.dim:
      raise ValueError(f"RunningNorm expected {self.dim} features")
    samples = samples.detach().reshape(-1, self.dim).double()
    if not len(samples):
      return
    if not torch.isfinite(samples).all():
      raise FloatingPointError("Nonfinite RunningNorm samples; statistics unchanged")
    variance, mean = torch.var_mean(samples, dim=0, unbiased=False)
    count = len(samples)
    total = self.count.double() + count
    delta = mean - self.mean
    updated_mean = self.mean + delta * (count / total)
    updated_var = (
      self.var * self.count
      + variance * count
      + delta.square() * self.count * (count / total)
    ) / total
    self.mean.copy_(updated_mean)
    self.var.copy_(updated_var.clamp_min(0))
    self.std.copy_(self.var.sqrt())
    self.count.copy_(total)

  def forward(self, inputs: Tensor) -> Tensor:
    return (inputs - self.mean.to(inputs.dtype)) / (
      self.std.to(inputs.dtype) + self.eps
    )

  def denormalize(self, outputs: Tensor) -> Tensor:
    """Restore physical units without changing statistics or input gradients."""
    return outputs * (self.std.to(outputs.dtype) + self.eps) + self.mean.to(
      outputs.dtype
    )
