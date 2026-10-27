"""One command scheduler for simulation and imagined command sequences."""

import torch
from torch import Tensor
from torch.nn import functional as F

from ..config.velocity_cfg import VelocityConfig
from .state import quat_inverse_apply, yaw_quat

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
      q = library.data["body_quat_w"][:, 0]
      velocity = quat_inverse_apply(yaw_quat(q), library.data["body_lin_vel_w"][:, 0])[
        :, :2
      ]
      commands = torch.cat((velocity, library.data["body_ang_vel_w"][:, 0, 2:3]), -1)
      clips = []
      for offset, length in zip(library.offsets.tolist(), library.lengths.tolist()):
        # Replicate endpoints; never blend samples from different clips.
        padded = F.pad(
          commands[offset : offset + length].T[None],
          (12, 12),
          mode="replicate",
        )
        clips.append(F.avg_pool1d(padded, 25, stride=1)[0].T)
      self.reference_commands = torch.cat(clips)

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
    standing = (
      torch.rand(len(ids), device=ids.device, generator=generator)
      < cfg.standing_probability
    )
    command[standing] = 0
    state["command"][ids] = command
    state["command_remaining"][ids] = cfg.resample_steps
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
    self.resample(
      state,
      (state["command_remaining"] <= 0).nonzero().flatten(),
      generator,
      control_step,
      reference_only=reference_only,
    )
    self._read_reference(state)

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
