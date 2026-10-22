"""One-second-bin reset sampling with online discounted return estimates."""

import torch
from torch import Tensor


class ResetSampler:
  def __init__(
    self,
    clip_lengths: Tensor,
    mode: str = "uniform",
    gamma: float = 0.99,
    ema: float = 0.1,
    seed: int = 0,
  ):
    if mode not in ("uniform", "return") or not 0 < gamma < 1 or not 0 < ema <= 1:
      raise ValueError("Invalid sampler configuration")
    self.mode, self.gamma, self.ema = mode, gamma, ema
    self.device = clip_lengths.device
    self.lengths = clip_lengths
    # Only states with a subsequent reference frame can start an episode.
    counts = torch.div(clip_lengths - 2, 50, rounding_mode="floor") + 1
    self.bin_offsets = counts.cumsum(0) - counts
    self.bin_motion = torch.repeat_interleave(
      torch.arange(len(counts), device=self.device), counts
    )
    self.bin_start = torch.cat(
      [torch.arange(int(count), device=self.device) * 50 for count in counts]
    )
    self.bin_width = (clip_lengths[self.bin_motion] - 1 - self.bin_start).clamp(max=50)
    self.max_return = 5 * 0.02 / (1 - gamma)
    self.returns = torch.full_like(self.bin_start, self.max_return, dtype=torch.float)
    self.historical = torch.full_like(self.returns, 1 / len(self.returns))
    self.history_updates = 0
    self.generator = torch.Generator(device=self.device).manual_seed(seed)

  def probabilities(self) -> Tensor:
    uniform = torch.full_like(self.returns, 1 / len(self.returns))
    if self.mode == "uniform":
      return uniform
    hard = torch.softmax(-25 * self.returns / self.max_return, dim=0)
    return 0.5 * uniform + 0.4 * hard + 0.1 * self.historical

  def sample(self, count: int) -> tuple[Tensor, Tensor]:
    bins = torch.multinomial(
      self.probabilities(), count, replacement=True, generator=self.generator
    )
    fraction = torch.rand(count, device=self.device, generator=self.generator)
    frame = self.bin_start[bins] + (fraction * self.bin_width[bins]).long()
    return self.bin_motion[bins], frame

  def update_episode(self, motion_id: int, frames: Tensor, rewards: Tensor) -> None:
    if frames.numel() == 0:
      return
    if len(frames) != len(rewards) or not torch.isfinite(rewards).all():
      raise ValueError("Invalid episode rewards")
    if ((frames < 0) | (frames >= self.lengths[motion_id] - 1)).any():
      raise ValueError("Episode frame out of range")
    returns = torch.empty_like(rewards)
    value = rewards.new_zeros(())
    for index in range(len(rewards) - 1, -1, -1):
      value = rewards[index] + self.gamma * value
      returns[index] = value
    bins = self.bin_offsets[motion_id] + frames // 50
    unique, inverse = bins.unique(return_inverse=True)
    sums = torch.zeros(len(unique), device=self.device).scatter_add_(
      0, inverse, returns
    )
    counts = torch.bincount(inverse, minlength=len(unique))
    self.returns[unique] = torch.lerp(self.returns[unique], sums / counts, self.ema)
    current = torch.softmax(-25 * self.returns / self.max_return, dim=0)
    self.history_updates += 1
    self.historical += (current - self.historical) / self.history_updates

  def state_dict(self) -> dict:
    return {
      "returns": self.returns.clone(),
      "historical": self.historical.clone(),
      "history_updates": self.history_updates,
      "generator": self.generator.get_state(),
    }

  def load_state_dict(self, state: dict) -> None:
    for key in ("returns", "historical"):
      if state[key].shape != self.returns.shape:
        raise ValueError("Sampler checkpoint does not match library")
      getattr(self, key).copy_(state[key])
    self.history_updates = state["history_updates"]
    self.generator.set_state(state["generator"].cpu())
