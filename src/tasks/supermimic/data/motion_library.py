"""Name-mapped reference motions from a manifest, NPZ file, or directory."""

import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from ..config.task_cfg import STEP_DT
from .motion_sources import motion_entries, select_motion_entries

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
    manifest: Path | str | None,
    joint_names: tuple[str, ...],
    body_names: tuple[str, ...],
    device: str = "cpu",
    clips: tuple[str, ...] = (),
    *,
    motion: Path | None = None,
    motion_dir: Path | None = None,
    storage: str = "full",
    fk_cache_frames: int = 262144,
  ):
    if storage not in ("full", "compact") or fk_cache_frames < 0:
      raise ValueError("Invalid motion storage or FK cache capacity")
    self.storage = storage
    self.compact_state = None
    if manifest is not None:
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
    entries = select_motion_entries(
      motion_entries(manifest, motion=motion, motion_dir=motion_dir), clips
    )
    compact_cpu = None
    cursor = 0
    if storage == "compact":
      from ..models.kinematics import G1Kinematics, robot_metadata

      robot = "g1_23dof" if len(joint_names) == 23 else "g1_29dof"
      metadata = robot_metadata(robot)
      if tuple(metadata["joint_names"]) != joint_names:
        raise ValueError("Compact FK requires robot joint order")
      self.kinematics = G1Kinematics(metadata, body_names).to(device).eval()
      total = 0
      for entry in entries:
        if entry.frames is not None:
          total += int(entry.frames)
        else:
          with np.load(entry.path, allow_pickle=False) as source:
            total += source["joint_pos"].shape[0]
      compact_cpu = torch.empty((total, 13 + 2 * len(joint_names)))
    for entry in entries:
      name = entry.name
      with np.load(entry.path, allow_pickle=False) as source:
        required = set(MOTION_FIELDS) | {
          "fps",
          "quaternion_order",
          "joint_names",
          "body_names",
        }
        missing = required - set(source.files)
        if missing:
          raise ValueError(
            f"Motion {name} lacks required fields: {', '.join(sorted(missing))}"
          )
        if source["joint_pos"].ndim != 2:
          raise ValueError(f"Invalid joint_pos array in {name}")
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
        if manifest is not None and entry.frames is None:
          raise ValueError(f"Manifest lacks frame count: {name}")
        count = (
          int(entry.frames)
          if entry.frames is not None
          else source["joint_pos"].shape[0]
        )
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
          if compact_cpu is None:
            selected = raw[
              :, joint_ids if key.startswith("joint_") else body_ids
            ].copy()
            arrays[key].append(torch.from_numpy(selected).float())
          else:
            joints = len(joint_names)
            columns = {
              "body_pos_w": slice(0, 3),
              "body_quat_w": slice(3, 7),
              "joint_pos": slice(7, 7 + joints),
              "body_lin_vel_w": slice(7 + joints, 10 + joints),
              "body_ang_vel_w": slice(10 + joints, 13 + joints),
              "joint_vel": slice(13 + joints, 13 + 2 * joints),
            }
            selected = (
              raw[:, joint_ids]
              if key.startswith("joint_")
              else raw[:, stored_bodies.index("pelvis")]
            )
            compact_cpu[cursor : cursor + count, columns[key]] = torch.from_numpy(
              selected.copy()
            )
      cursor += count
      lengths.append(count)
      self.names.append(name)
    self.lengths = torch.tensor(lengths, dtype=torch.long, device=device)
    self.offsets = self.lengths.cumsum(0) - self.lengths
    if compact_cpu is None:
      self.data = {key: torch.cat(value).to(device) for key, value in arrays.items()}
      data_bytes = sum(x.numel() * x.element_size() for x in self.data.values())
    else:
      self.compact_state = compact_cpu.to(device)
      self.data = {}  # Full-body tensors must never be materialized for the whole library.
      self.fk_cache = ReferenceFKCache(
        self.kinematics, min(fk_cache_frames, cursor), device
      )
      data_bytes = self.compact_state.numel() * self.compact_state.element_size()
    print(
      f"MotionLibrary: {len(lengths)} clips, {cursor} frames, {storage}, "
      f"{data_bytes / 2**30:.3f} GiB arrays; FK cache "
      f"{self.fk_cache.capacity if compact_cpu is not None else 0} frames"
    )
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
    return self._frames(indices)

  def get_future(
    self, motion_id: Tensor, frame_id: Tensor, horizon: int = 10
  ) -> dict[str, Tensor]:
    if horizon < 1:
      raise ValueError("Future horizon must be positive")
    self._indices(motion_id, frame_id)
    frames = frame_id[..., None] + torch.arange(horizon, device=self.device)
    frames = frames.minimum(self.lengths[motion_id][..., None] - 1)
    indices = self.offsets[motion_id][..., None] + frames
    return self._frames(indices)

  @torch.no_grad()
  def _frames(self, indices: Tensor) -> dict[str, Tensor]:
    if self.compact_state is None:
      return {key: value[indices] for key, value in self.data.items()}
    state = self.compact_state[indices]
    joints = len(self.joint_names)
    return {
      "joint_pos": state[..., 7 : 7 + joints],
      "joint_vel": state[..., 13 + joints :],
      **self.fk_cache.get(indices, self.compact_state),
    }

  def root_fields(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Whole-library root views without reconstructing body states."""
    if self.compact_state is None:
      return tuple(self.data[key][:, 0] for key in MOTION_FIELDS[2:])
    state, joints = self.compact_state, len(self.joint_names)
    return (
      state[:, :3],
      state[:, 3:7],
      state[:, 7 + joints : 10 + joints],
      state[:, 10 + joints : 13 + joints],
    )

  def root_state(self, motion_id: Tensor, frame_id: Tensor) -> Tensor:
    indices = self._indices(motion_id, frame_id)
    return torch.cat([value[indices] for value in self.root_fields()], dim=-1)


class ReferenceFKCache:
  """Bounded device cache; collisions replace entries, never alter results."""

  def __init__(self, kinematics, capacity: int, device):
    self.kinematics, self.capacity = kinematics, capacity
    self.tags = torch.full((capacity,), -1, dtype=torch.long, device=device)
    self.values = torch.empty((capacity, len(kinematics.body_names), 13), device=device)

  @torch.no_grad()
  def get(self, indices: Tensor, states: Tensor) -> dict[str, Tensor]:
    unique, inverse = indices.flatten().unique(return_inverse=True)
    result = states.new_empty((len(unique), len(self.kinematics.body_names), 13))
    if self.capacity:
      slots = unique % self.capacity
      hit = self.tags[slots] == unique
      result[hit] = self.values[slots[hit]]
    else:
      hit = torch.zeros_like(unique, dtype=torch.bool)
    missing = (~hit).nonzero().flatten()
    for chunk in missing.split(8192):
      if not chunk.numel():
        continue
      bodies = self.kinematics(states[unique[chunk]])
      result[chunk] = torch.cat([bodies[key] for key in MOTION_FIELDS[2:]], -1)
    if self.capacity and missing.numel():
      # Pick one deterministic winner per slot, with no duplicate-index writes.
      slots = unique[missing] % self.capacity
      used, group = slots.unique(return_inverse=True)
      winners = torch.full_like(used, len(unique))
      winners.scatter_reduce_(0, group, missing, reduce="amin", include_self=True)
      self.values[used] = result[winners]
      self.tags[used] = unique[winners]
    result = result[inverse].reshape(
      *indices.shape, len(self.kinematics.body_names), 13
    )
    return dict(zip(MOTION_FIELDS[2:], result.split((3, 4, 3, 3), dim=-1)))
