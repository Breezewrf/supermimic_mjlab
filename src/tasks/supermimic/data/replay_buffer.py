"""Vectorized ring buffer with explicit episode boundaries and history recovery."""

import torch
from torch import Tensor

REQUIRED_FIELDS = {
  "state",
  "next_state",
  "action",
  "previous_action",
  "observation",
  "next_observation",
  "clean_observation",
  "next_clean_observation",
  "context",
  "reward",
  "motion_id",
  "frame_id",
  "episode_id",
  "episode_step",
  "terminated",
  "time_out",
  "motion_end",
}


class ReplayBuffer:
  def __init__(
    self,
    num_envs: int,
    capacity: int = 1000,
    device: str = "cpu",
    seed: int = 0,
  ):
    if num_envs < 1 or capacity < 1:
      raise ValueError("Replay environment count and capacity must be positive")
    self.num_envs, self.capacity, self.device = num_envs, capacity, device
    self.data: dict[str, Tensor] = {}
    self.steps = 0
    self.generator = torch.Generator(device=device).manual_seed(seed)

  def __len__(self):
    return min(self.steps, self.capacity) * self.num_envs

  def append(self, transition: dict[str, Tensor]) -> None:
    if not REQUIRED_FIELDS.issubset(transition):
      raise ValueError(
        f"Missing transition fields: {REQUIRED_FIELDS - transition.keys()}"
      )
    if self.data and transition.keys() != self.data.keys():
      raise ValueError("Replay schema cannot change after the first insertion")
    for name, value in transition.items():
      if value.shape[:1] != (self.num_envs,):
        raise ValueError(f"Wrong environment axis in {name}")
      if value.is_floating_point() and not torch.isfinite(value).all():
        raise ValueError(f"Nonfinite transition: {name}")
      if name not in self.data:
        self.data[name] = torch.empty(
          (self.capacity, *value.shape), dtype=value.dtype, device=self.device
        )
      if (
        value.shape != self.data[name].shape[1:] or value.dtype != self.data[name].dtype
      ):
        raise ValueError(f"Inconsistent replay shape/dtype: {name}")
    slot = self.steps % self.capacity
    # copy_ is essential: simulator tensors may be mutated on the next step.
    for name, value in transition.items():
      self.data[name][slot].copy_(value.detach())
    self.steps += 1

  def _candidates(self) -> tuple[Tensor, Tensor]:
    if not self.steps:
      raise ValueError("Replay buffer is empty")
    times = torch.arange(
      max(0, self.steps - self.capacity), self.steps, device=self.device
    )
    times = times[:, None].expand(-1, self.num_envs).flatten()
    envs = torch.arange(self.num_envs, device=self.device).repeat(
      min(self.steps, self.capacity)
    )
    return times, envs

  def _done(self, slots: Tensor, envs: Tensor) -> Tensor:
    return (
      self.data["terminated"][slots, envs]
      | self.data["time_out"][slots, envs]
      | self.data["motion_end"][slots, envs]
    )

  def _valid_windows(
    self, starts: Tensor, envs: Tensor, length: int, require_full: bool
  ) -> Tensor:
    times = starts[:, None] + torch.arange(length, device=self.device)
    slots, env = times % self.capacity, envs[:, None]
    exists = (times < self.steps) & (
      self.data["episode_id"][slots, env]
      == self.data["episode_id"][starts % self.capacity, envs, None]
    )
    done = self._done(slots, env)
    active = (
      torch.cat((torch.ones_like(done[:, :1]), ~done[:, :-1]), 1)
      .long()
      .cumprod(1)
      .bool()
    )
    return (exists & active).all(1) if require_full else (~active | exists).all(1)

  def _sample_indices(self, batch_size: int, valid) -> Tensor:
    """Uniform rejection sampling; enumerate only for exceptionally sparse data."""
    if not self.steps:
      raise ValueError("Replay buffer is empty")
    retained = min(self.steps, self.capacity)
    selected, remaining = [], batch_size
    for _ in range(16):
      flat = torch.randint(
        retained * self.num_envs,
        (max(64, 2 * remaining),),
        device=self.device,
        generator=self.generator,
      )
      starts = flat // self.num_envs + self.steps - retained
      envs = flat % self.num_envs
      accepted = torch.stack((starts, envs), -1)[valid(starts, envs)][:remaining]
      selected.append(accepted)
      remaining -= len(accepted)
      if not remaining:
        return torch.cat(selected)
    # Preserve correct behavior for tiny/sparse buffers and genuinely empty sets.
    starts, envs = self._candidates()
    choices = torch.stack((starts, envs), -1)[valid(starts, envs)]
    if not len(choices):
      raise ValueError("No replay starts satisfy the requested window/history")
    indices = torch.randint(
      len(choices), (remaining,), device=self.device, generator=self.generator
    )
    return torch.cat((*selected, choices[indices]))

  def window_indices(self, length: int = 8, require_full: bool = False) -> Tensor:
    """Return retained logical (step, env) starts, stable for an immutable snapshot."""
    if length < 1 or length > self.capacity:
      raise ValueError("Invalid window length")
    starts, envs = self._candidates()
    valid = self._valid_windows(starts, envs, length, require_full)
    return torch.stack((starts[valid], envs[valid]), dim=-1)

  def sample_windows(
    self, batch_size: int, length: int = 8, *, require_full: bool = False
  ) -> dict[str, Tensor]:
    if batch_size < 1 or not 1 <= length <= self.capacity:
      raise ValueError("Invalid batch size")
    indices = self._sample_indices(
      batch_size,
      lambda starts, envs: self._valid_windows(starts, envs, length, require_full),
    )
    return self.gather_windows(indices, length)

  def gather_windows(self, indices: Tensor, length: int = 8) -> dict[str, Tensor]:
    if (
      length < 1
      or length > self.capacity
      or indices.ndim != 2
      or indices.shape[1] != 2
      or indices.dtype != torch.long
    ):
      raise ValueError("Expected int64 (batch, 2) indices and a valid window length")
    starts, envs = indices.unbind(-1)
    if (
      (starts < max(0, self.steps - self.capacity))
      | (starts >= self.steps)
      | (envs < 0)
      | (envs >= self.num_envs)
    ).any():
      raise IndexError("Replay window indices are no longer retained")
    if not self._valid_windows(starts, envs, length, False).all():
      raise ValueError("Window crosses missing data or an unmarked episode boundary")
    batch_size = len(indices)
    times = starts[:, None] + torch.arange(length, device=self.device)
    slots, env = times % self.capacity, envs[:, None]
    done = self._done(slots, env)
    mask = (
      torch.cat(
        (
          torch.ones(batch_size, 1, dtype=torch.bool, device=self.device),
          ~done[:, :-1],
        ),
        dim=1,
      )
      .long()
      .cumprod(1)
      .bool()
    )
    result = {}
    for name, value in self.data.items():
      sampled = value[slots, env]
      expanded = mask.reshape(*mask.shape, *([1] * (sampled.ndim - 2)))
      result[name] = torch.where(expanded, sampled, torch.zeros_like(sampled))
    return result | {"mask": mask}

  def sample_starts(
    self, batch_size: int, history_length: int = 10
  ) -> dict[str, Tensor]:
    if batch_size < 1 or history_length < 1 or history_length > self.capacity:
      raise ValueError("Invalid batch size or history length")
    indices = self._sample_indices(
      batch_size,
      lambda starts, envs: self._history_indices(starts, envs, history_length)[1],
    )
    starts, envs = indices.unbind(-1)
    history_slots, _ = self._history_indices(starts, envs, history_length)
    slots = starts % self.capacity
    result = {name: value[slots, envs].clone() for name, value in self.data.items()}
    for key in ("observation", "clean_observation"):
      result["history" if key == "observation" else "clean_history"] = self.data[key][
        history_slots, envs[:, None]
      ].clone()
    return result

  def _history_indices(self, starts: Tensor, envs: Tensor, history_length: int):
    slots = starts % self.capacity
    episode = self.data["episode_id"][slots, envs]
    episode_start = starts - self.data["episode_step"][slots, envs]
    times = starts[:, None] - torch.arange(
      history_length - 1, -1, -1, device=self.device
    )
    times = times.maximum(episode_start[:, None])
    history_slots = times % self.capacity
    valid = (
      (times >= max(0, self.steps - self.capacity))
      & (self.data["episode_id"][history_slots, envs[:, None]] == episode[:, None])
    ).all(-1)
    return history_slots, valid

  def state_dict(self) -> dict:
    # Save only initialized slots. Contents of uninitialized storage never reach disk.
    count = min(self.steps, self.capacity)
    return {
      "num_envs": self.num_envs,
      "capacity": self.capacity,
      "steps": self.steps,
      "data": {key: value[:count].cpu().clone() for key, value in self.data.items()},
      "generator": self.generator.get_state().cpu(),
      "generator_seed": self.generator.initial_seed(),
      "generator_device_type": torch.device(self.device).type,
    }

  @classmethod
  def from_state_dict(cls, state: dict, device: str = "cpu", restore_rng: bool = True):
    buffer = cls(state["num_envs"], state["capacity"], device)
    buffer.steps = state["steps"]
    for name, value in state["data"].items():
      storage = torch.empty(
        (buffer.capacity, *value.shape[1:]), dtype=value.dtype, device=device
      )
      storage[: len(value)].copy_(value)
      buffer.data[name] = storage
    saved_rng = state["generator"].cpu()
    same_backend = (
      state.get("generator_device_type", torch.device(device).type)
      == torch.device(device).type
    )
    if (
      restore_rng
      and same_backend
      and saved_rng.numel() == buffer.generator.get_state().numel()
    ):
      buffer.generator.set_state(saved_rng)
    else:
      # CPU MT19937 and CUDA Philox states are not interchangeable. Data can
      # move freely; cross-backend sampling restarts deterministically.
      buffer.generator.manual_seed(state.get("generator_seed", 0))
    return buffer
