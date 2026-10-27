"""Manifest-backed, name-mapped reference motion library."""

import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from ..config.task_cfg import STEP_DT

MOTION_FIELDS = (
  "joint_pos",
  "joint_vel",
  "body_pos_w",
  "body_quat_w",
  "body_lin_vel_w",
  "body_ang_vel_w",
)


class MotionLibrary:
  def __init__(
    self,
    manifest: Path | str,
    joint_names: tuple[str, ...],
    body_names: tuple[str, ...],
    device: str = "cpu",
    clips: tuple[str, ...] = (),
  ):
    manifest = Path(manifest).resolve()
    metadata = json.loads(manifest.read_text())
    if metadata.get("schema_version") != 1:
      raise ValueError("Unsupported motion manifest schema")
    if (
      not np.isclose(metadata["fps"], 1 / STEP_DT)
      or metadata["quaternion_order"] != "wxyz"
    ):
      raise ValueError("SuperMimic requires 50 Hz, wxyz reference motions")
    if len(joint_names) not in (23, 29) or len(set(joint_names)) != len(joint_names):
      raise ValueError("Expected 23 or 29 unique joint names")
    if (
      not body_names
      or body_names[0] != "pelvis"
      or len(set(body_names)) != len(body_names)
    ):
      raise ValueError("Body names must be unique and start with pelvis")
    self.device = device
    self.joint_names, self.body_names = joint_names, body_names
    self.names: list[str] = []
    lengths, arrays = [], {key: [] for key in MOTION_FIELDS}
    requested = set(clips)
    matched = set()
    for entry in metadata["clips"]:
      relative = Path(entry["file"])
      if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Manifest clip paths must stay inside the motion directory")
      name = relative.with_suffix("").as_posix()
      selectors = {name, relative.as_posix()}
      if requested and not requested.intersection(selectors):
        continue
      matched.update(requested.intersection(selectors))
      if name in self.names:
        raise ValueError(f"Duplicate motion: {name}")
      with np.load(manifest.parent / relative, allow_pickle=False) as source:
        if source["quaternion_order"].item() != "wxyz" or not np.isclose(
          source["fps"].item(), 50
        ):
          raise ValueError(f"Invalid motion frequency/quaternion order: {name}")
        stored_joints, stored_bodies = (
          source["joint_names"].tolist(),
          source["body_names"].tolist(),
        )
        if len(set(stored_joints)) != len(stored_joints) or set(stored_joints) != set(
          joint_names
        ):
          raise ValueError(f"Joint names do not match robot: {name}")
        if len(set(stored_bodies)) != len(stored_bodies) or not set(
          body_names
        ).issubset(stored_bodies):
          raise ValueError(f"Body names do not match robot: {name}")
        joint_ids = [stored_joints.index(j) for j in joint_names]
        body_ids = [stored_bodies.index(b) for b in body_names]
        count = int(entry["frames"])
        if count < 2:
          raise ValueError(f"Clip must have at least two frames: {name}")
        for key in MOTION_FIELDS:
          raw = source[key]
          expected = (
            (count, len(stored_joints))
            if key.startswith("joint_")
            else (
              count,
              len(stored_bodies),
              4 if key == "body_quat_w" else 3,
            )
          )
          if raw.shape != expected or not np.isfinite(raw).all():
            raise ValueError(f"Invalid {key} array in {name}: {raw.shape}")
          if key == "body_quat_w" and not np.allclose(
            np.linalg.norm(raw, axis=-1), 1, atol=1e-5
          ):
            raise ValueError(f"Non-unit body quaternion in {name}")
          selected = raw[:, joint_ids if key.startswith("joint_") else body_ids].copy()
          arrays[key].append(torch.from_numpy(selected).float())
      lengths.append(count)
      self.names.append(name)
    if requested - matched:
      raise ValueError(f"Unknown clips: {sorted(requested - matched)}")
    if not lengths:
      raise ValueError("No reference motions selected")
    self.lengths = torch.tensor(lengths, dtype=torch.long, device=device)
    self.offsets = self.lengths.cumsum(0) - self.lengths
    self.data = {key: torch.cat(value).to(device) for key, value in arrays.items()}
    self.fps = 50.0

  def __len__(self):
    return len(self.names)

  def _indices(self, motion_id: Tensor, frame_id: Tensor) -> Tensor:
    if motion_id.dtype != torch.long or frame_id.dtype != torch.long:
      raise ValueError("Motion and frame indices must be int64 tensors")
    if ((motion_id < 0) | (motion_id >= len(self))).any():
      raise IndexError("Motion ID out of range")
    if ((frame_id < 0) | (frame_id >= self.lengths[motion_id])).any():
      raise IndexError("Frame ID out of clip range")
    return self.offsets[motion_id] + frame_id

  def get_frame(self, motion_id: Tensor, frame_id: Tensor) -> dict[str, Tensor]:
    indices = self._indices(motion_id, frame_id)
    return {key: value[indices] for key, value in self.data.items()}

  def get_future(
    self, motion_id: Tensor, frame_id: Tensor, horizon: int = 10
  ) -> dict[str, Tensor]:
    if horizon < 1:
      raise ValueError("Future horizon must be positive")
    self._indices(motion_id, frame_id)
    frames = frame_id[..., None] + torch.arange(horizon, device=self.device)
    frames = frames.minimum(self.lengths[motion_id][..., None] - 1)
    indices = self.offsets[motion_id][..., None] + frames
    return {key: value[indices] for key, value in self.data.items()}

  def root_state(self, motion_id: Tensor, frame_id: Tensor) -> Tensor:
    ref = self.get_frame(motion_id, frame_id)
    return torch.cat(
      (
        ref["body_pos_w"][..., 0, :],
        ref["body_quat_w"][..., 0, :],
        ref["body_lin_vel_w"][..., 0, :],
        ref["body_ang_vel_w"][..., 0, :],
      ),
      dim=-1,
    )
