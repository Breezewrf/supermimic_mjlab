"""One command scheduler for simulation and imagined command sequences."""

import torch
from torch import Tensor
from torch.nn import functional as F

from ..config.task_cfg import STEP_DT
from ..config.velocity_cfg import VelocityConfig
from .state import quat_inverse_apply, quat_normalize, yaw_quat

COMMAND_FIELDS = (
  "command",
  "command_source",
  "command_remaining",
  "reference_motion",
  "reference_frame",
)


class VelocityCommands:
  def __init__(self, cfg: VelocityConfig, library=None):
    self.cfg, self.library = cfg, library
    self.reference_commands = None
    if library is not None:
      _, q, root_linear, root_angular = library.root_fields()
      commands = torch.empty((len(q), 3), device=q.device, dtype=torch.float32)
      # Convert only a batch of root fields at a time, including for large libraries.
      for start in range(0, len(q), 65536):
        frames = slice(start, start + 65536)
        rotation = q[frames]
        if rotation.dtype == torch.float16:
          rotation = quat_normalize(rotation.float())
        commands[frames, :2] = quat_inverse_apply(
          yaw_quat(rotation), root_linear[frames].float()
        )[:, :2]
        commands[frames, 2] = root_angular[frames, 2]
      for offset, length in zip(library.offsets.tolist(), library.lengths.tolist()):
        # Replicate endpoints; never blend samples from different clips.
        padded = F.pad(
          commands[offset : offset + length].T[None],
          (12, 12),
          mode="replicate",
        )
        commands[offset : offset + length] = F.avg_pool1d(padded, 25, stride=1)[0].T
      self.reference_commands = commands

  def empty(self, count: int, device) -> dict[str, Tensor]:
    return {
      "command": torch.zeros(count, 3, device=device),
      "command_source": torch.zeros(count, dtype=torch.bool, device=device),
      "command_remaining": torch.zeros(count, dtype=torch.long, device=device),
      "reference_motion": torch.full((count,), -1, dtype=torch.long, device=device),
      "reference_frame": torch.full((count,), -1, dtype=torch.long, device=device),
    }

  def _new_reference(self, state, ids, generator):
    if not len(ids):
      return
    if self.library is None:
      raise ValueError("Reference commands require a motion library")
    motion = torch.randint(
      len(self.library), (len(ids),), device=ids.device, generator=generator
    )
    frame = (
      torch.rand(len(ids), device=ids.device, generator=generator)
      * (self.library.lengths[motion] - 1)
    ).long()
    state["reference_motion"][ids] = motion
    state["reference_frame"][ids] = frame

  def _read_reference(self, state):
    ids = state["command_source"].nonzero().flatten()
    if len(ids):
      index = (
        self.library.offsets[state["reference_motion"][ids]]
        + state["reference_frame"][ids]
      )
      state["command"][ids] = self.reference_commands[index]

  @torch.no_grad()
  def resample(self, state, ids, generator, control_step: int, *, reference_only=False):
    if not len(ids):
      return
    cfg = self.cfg
    probability = (
      (1.0 if reference_only else cfg.reference_probability)
      if self.library is not None
      else 0.0
    )
    state["command_source"][ids] = (
      torch.rand(len(ids), device=ids.device, generator=generator) < probability
    )
    ranges = (
      cfg.final_ranges if control_step >= cfg.curriculum_steps else cfg.initial_ranges
    )
    bounds = state["command"].new_tensor(ranges)
    command = bounds[:, 0] + torch.rand(
      len(ids), 3, device=ids.device, generator=generator
    ) * (bounds[:, 1] - bounds[:, 0])
    if cfg.axis_command_probability and not reference_only:
      single_axis = (
        torch.rand(len(ids), device=ids.device, generator=generator)
        < cfg.axis_command_probability
      )
      axis = torch.randint(3, (len(ids), 1), device=ids.device, generator=generator)
      inactive_axes = axis != torch.arange(3, device=ids.device)
      command.masked_fill_(single_axis[:, None] & inactive_axes, 0)
    standing = (
      torch.rand(len(ids), device=ids.device, generator=generator)
      < cfg.standing_probability
    )
    if cfg.command_deadband:
      standing |= command.norm(dim=-1) <= cfg.command_deadband
    command[standing] = 0
    state["command"][ids] = command
    low, high = cfg.resampling_time_range
    seconds = low + (high - low) * torch.rand(
      len(ids), device=ids.device, generator=generator
    )
    state["command_remaining"][ids] = (seconds / STEP_DT).ceil().long().clamp_min(1)
    state["reference_motion"][ids] = -1
    state["reference_frame"][ids] = -1
    self._new_reference(state, ids[state["command_source"][ids]], generator)
    self._read_reference(state)

  @torch.no_grad()
  def advance(self, state, generator, control_step: int, *, reference_only=False):
    state["command_remaining"] -= 1
    ids = state["command_source"].nonzero().flatten()
    if len(ids):
      state["reference_frame"][ids] += 1
      ended = (
        state["reference_frame"][ids]
        >= self.library.lengths[state["reference_motion"][ids]]
      )
      self._new_reference(state, ids[ended], generator)
    resampled = (state["command_remaining"] <= 0).nonzero().flatten()
    self.resample(
      state,
      resampled,
      generator,
      control_step,
      reference_only=reference_only,
    )
    self._read_reference(state)
    return resampled

  @torch.no_grad()
  def sequence(self, starts, horizon, generator, control_step, *, reference_only=False):
    state = {key: starts[key].clone() for key in COMMAND_FIELDS}
    frames = {key: [value.clone()] for key, value in state.items()}
    for step in range(horizon):
      self.advance(
        state, generator, control_step + step + 1, reference_only=reference_only
      )
      for key, value in state.items():
        frames[key].append(value.clone())
    return {key: torch.stack(value, 1) for key, value in frames.items()}
