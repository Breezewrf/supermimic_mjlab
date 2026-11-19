"""One-second reset bins, scored against the achievable episode return."""

import math

import torch
from torch import Tensor

from ..config.task_cfg import DEFAULT_RESET_SOFTMAX_CAP, STEP_DT


class ResetSampler:
  SCHEMA_VERSION = 3
  SUPPORTED_SCHEMAS = (2, SCHEMA_VERSION)

  def __init__(
    self,
    clip_lengths: Tensor,
    mode: str = "uniform",
    gamma: float = 0.99,
    ema: float = 0.1,
    seed: int = 0,
    *,
    softmax_cap: float = DEFAULT_RESET_SOFTMAX_CAP,
  ):
    if mode not in ("uniform", "return") or not 0 < gamma < 1 or not 0 < ema <= 1:
      raise ValueError("Invalid sampler configuration")
    if not math.isfinite(softmax_cap) or (softmax_cap != 0 and softmax_cap < 1):
      raise ValueError("Reset softmax cap must be zero or at least one")
    self.mode, self.gamma, self.ema = mode, gamma, ema
    self.softmax_cap = softmax_cap
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
    self.max_return = 5 * STEP_DT / (1 - gamma)
    self.returns = torch.full_like(self.bin_start, self.max_return, dtype=torch.float)
    self.visited = torch.zeros_like(self.bin_start, dtype=torch.bool)
    self.historical = torch.full_like(self.returns, 1 / len(self.returns))
    self.history_updates = 0
    self.generator = torch.Generator(device=self.device).manual_seed(seed)

  def probabilities(self) -> Tensor:
    uniform = torch.full_like(self.returns, 1 / len(self.returns))
    if self.mode == "uniform":
      return uniform
    hard = self._hard_probabilities()
    return 0.5 * uniform + 0.4 * hard + 0.1 * self.historical

  def _hard_probabilities(self) -> Tensor:
    hard = torch.softmax(-25 * self.returns / self.max_return, dim=0)
    if self.softmax_cap == 0 or self.softmax_cap >= hard.numel():
      return hard
    if self.softmax_cap == 1:
      return torch.full_like(hard, 1 / hard.numel())
    cap = self.softmax_cap / hard.numel()
    capped = hard.clamp_max(cap)
    # Redistribute in proportion to unused capacity in one vector pass.
    # The available capacity is (cap * N - 1) plus the removed mass.
    excess = (1 - capped.sum()).clamp_min(0)
    fill = excess / (self.softmax_cap - 1 + excess)
    return capped + (cap - capped) * fill

  def sample(self, count: int) -> tuple[Tensor, Tensor]:
    bins = torch.multinomial(
      self.probabilities(), count, replacement=True, generator=self.generator
    )
    fraction = torch.rand(count, device=self.device, generator=self.generator)
    frame = self.bin_start[bins] + (fraction * self.bin_width[bins]).long()
    return self.bin_motion[bins], frame

  def update_episode(
    self,
    motion_id: int,
    frames: Tensor,
    rewards: Tensor,
    *,
    max_episode_steps: int = 500,
  ) -> None:
    """Convenience wrapper; collection uses the vectorized update_episodes path."""
    if frames.numel() == 0:
      return
    if len(frames) != len(rewards) or not torch.isfinite(rewards).all():
      raise ValueError("Invalid episode rewards")
    if ((frames < 0) | (frames >= self.lengths[motion_id] - 1)).any():
      raise ValueError("Episode frame out of range")
    discount = self.gamma ** torch.arange(len(rewards), device=self.device)
    self.update_episodes(
      torch.tensor([motion_id], device=self.device),
      frames[:1],
      (rewards * discount).sum().reshape(1),
      torch.tensor([len(rewards)], device=self.device),
      max_episode_steps,
    )

  @torch.no_grad()
  def update_episodes(
    self,
    motion: Tensor,
    start_frame: Tensor,
    discounted_return: Tensor,
    steps: Tensor,
    max_episode_steps: int | Tensor,
  ) -> None:
    """Aggregate finished episodes before applying one EMA update per start bin.

    The denominator covers the shorter of the available reference and the
    episode time limit. Actual rewards stop at the last executed transition.
    The bounded score drives reset sampling.
    """
    if not motion.numel():
      return
    remaining = self.lengths[motion] - 1 - start_frame
    horizon = (
      torch.minimum(remaining, max_episode_steps)
      if isinstance(max_episode_steps, Tensor)
      else remaining.clamp_max(max_episode_steps)
    )
    if ((steps <= 0) | (steps > horizon) | (start_frame < 0)).any():
      raise ValueError("Invalid episode start or length")
    # Keep scores in return units; diagnostics divide by max_return for [0, 1].
    total_mass = -torch.expm1(horizon.float() * math.log(self.gamma))
    scores = discounted_return / total_mass
    if not torch.isfinite(scores).all():
      raise ValueError("Nonfinite episode return")
    bins = self.bin_offsets[motion] + start_frame // 50
    unique, inverse = bins.unique(return_inverse=True)
    sums = torch.zeros(len(unique), device=self.device).scatter_add_(0, inverse, scores)
    counts = torch.bincount(inverse, minlength=len(unique))
    means = sums / counts
    updated = torch.where(
      self.visited[unique], torch.lerp(self.returns[unique], means, self.ema), means
    )
    self.returns[unique] = updated.clamp(0, self.max_return)
    self.visited[unique] = True
    current = self._hard_probabilities()
    self.history_updates += 1
    self.historical += (current - self.historical) / self.history_updates

  def state_dict(self) -> dict:
    return {
      "schema_version": self.SCHEMA_VERSION,
      "returns": self.returns.clone(),
      "visited": self.visited.clone(),
      "historical": self.historical.clone(),
      "history_updates": self.history_updates,
      "generator": self.generator.get_state(),
    }

  def load_state_dict(self, state: dict) -> None:
    schema = state.get("schema_version")
    if schema not in self.SUPPORTED_SCHEMAS:
      raise ValueError("Legacy reset scores are incompatible; start a new training run")
    keys = ("returns", "historical") if schema == 2 else ("returns", "historical", "visited")
    if any(state[key].shape != self.returns.shape for key in keys):
      raise ValueError("Sampler checkpoint does not match library")
    if schema == 2:
      # Old scores used a different horizon and history can contain extreme mass.
      self.returns.fill_(self.max_return)
      self.visited.zero_()
      self.historical.fill_(1 / len(self.returns))
      self.history_updates = 0
    else:
      for key in keys:
        getattr(self, key).copy_(state[key])
      self.history_updates = state["history_updates"]
    self.generator.set_state(state["generator"].cpu())
