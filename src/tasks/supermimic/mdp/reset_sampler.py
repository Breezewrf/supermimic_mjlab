"""One-second-bin sampling, scored by length-normalized episode-start returns."""

import math

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

  def update_episode(
    self,
    motion_id: int,
    frames: Tensor,
    rewards: Tensor,
    *,
    terminated: bool = True,
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
      rewards.sum().reshape(1),
      torch.tensor([len(rewards)], device=self.device),
      torch.tensor([terminated], device=self.device),
    )

  @torch.no_grad()
  def update_episodes(
    self,
    motion: Tensor,
    start_frame: Tensor,
    discounted_return: Tensor,
    reward_sum: Tensor,
    steps: Tensor,
    terminated: Tensor,
  ) -> None:
    """Aggregate finished episodes before applying one EMA update per start bin.

    Normal cutoffs extrapolate the episode's mean reward over the remaining clip.
    Failures get no tail. Both are normalized by the *possible* remaining length,
    never by actual survival time, so early failures cannot earn a full score.
    These values drive reset sampling only; they are not critic targets.
    """
    if not motion.numel():
      return
    remaining = self.lengths[motion] - 1 - start_frame
    if ((steps <= 0) | (steps > remaining) | (start_frame < 0)).any():
      raise ValueError("Invalid episode start or length")
    # expm1 avoids cancellation for short tails and clips.
    log_gamma = math.log(self.gamma)
    tail_mass = -torch.expm1((remaining - steps).float() * log_gamma)
    total_mass = -torch.expm1(remaining.float() * log_gamma)
    tail = self.gamma**steps * (reward_sum / steps) * tail_mass / (1 - self.gamma)
    value = discounted_return + torch.where(terminated, 0.0, tail)
    scores = value / total_mass
    if not torch.isfinite(scores).all():
      raise ValueError("Nonfinite episode return")
    bins = self.bin_offsets[motion] + start_frame // 50
    unique, inverse = bins.unique(return_inverse=True)
    sums = torch.zeros(len(unique), device=self.device).scatter_add_(0, inverse, scores)
    counts = torch.bincount(inverse, minlength=len(unique))
    self.returns[unique] = torch.lerp(self.returns[unique], sums / counts, self.ema)
    current = torch.softmax(-25 * self.returns / self.max_return, dim=0)
    self.history_updates += 1
    self.historical += (current - self.historical) / self.history_updates

  def state_dict(self) -> dict:
    return {
      "schema_version": 2,
      "returns": self.returns.clone(),
      "historical": self.historical.clone(),
      "history_updates": self.history_updates,
      "generator": self.generator.get_state(),
    }

  def load_state_dict(self, state: dict) -> None:
    if state.get("schema_version") != 2:
      raise ValueError("Legacy reset scores are incompatible; start a new training run")
    for key in ("returns", "historical"):
      if state[key].shape != self.returns.shape:
        raise ValueError("Sampler checkpoint does not match library")
      getattr(self, key).copy_(state[key])
    self.history_updates = state["history_updates"]
    self.generator.set_state(state["generator"].cpu())
