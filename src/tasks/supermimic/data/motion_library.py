"""Name-mapped reference motions from a manifest, NPZ file, or directory."""

import json
import multiprocessing
import sys
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm

from ..config.task_cfg import STEP_DT
from ..mdp.state import quat_normalize
from .motion_sources import MotionEntry, motion_entries, select_motion_entries

MOTION_FIELDS = (
  "joint_pos",
  "joint_vel",
  "body_pos_w",
  "body_quat_w",
  "body_lin_vel_w",
  "body_ang_vel_w",
)


def _read_clip(
  entry: MotionEntry, *, joint_names, body_names, compact, require_frame_count
):
  selected = {}
  joints = len(joint_names)
  # Offsets within the half-precision portion, after the three FP32 coordinates.
  columns = {
    "body_quat_w": slice(0, 4),
    "joint_pos": slice(4, 4 + joints),
    "body_lin_vel_w": slice(4 + joints, 7 + joints),
    "body_ang_vel_w": slice(7 + joints, 10 + joints),
    "joint_vel": slice(10 + joints, 10 + 2 * joints),
  }
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
    joint_pos = source["joint_pos"]
    if joint_pos.ndim != 2:
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
    if len(set(stored_bodies)) != len(stored_bodies) or not set(body_names).issubset(
      stored_bodies
    ):
      raise ValueError(f"Body names do not match robot: {name}")
    joint_ids = [stored_joints.index(j) for j in joint_names]
    body_ids = [stored_bodies.index(b) for b in body_names]
    if require_frame_count and entry.frames is None:
      raise ValueError(f"Manifest lacks frame count: {name}")
    count = int(entry.frames) if entry.frames is not None else joint_pos.shape[0]
    if count < 2:
      raise ValueError(f"Clip must have at least two frames: {name}")
    if compact:
      rest = np.empty((count, 10 + 2 * joints), dtype=np.float16)
    for key in MOTION_FIELDS:
      raw = joint_pos if key == "joint_pos" else source[key]
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
      if not compact:
        indices = joint_ids if key.startswith("joint_") else body_ids
        selected[key] = raw[:, indices].astype(np.float32, order="C", copy=False)
      elif key == "body_pos_w":
        root_pos = np.ascontiguousarray(raw[:, body_ids[0]], dtype=np.float32)
      else:
        values = raw[:, joint_ids] if key.startswith("joint_") else raw[:, body_ids[0]]
        with np.errstate(over="ignore"):
          target = rest[:, columns[key]]
          target[:] = values
        if not np.isfinite(target).all():
          raise ValueError(f"Motion {name}: {key} exceeds FP16 storage range")
  return count, (root_pos, rest) if compact else selected


def _read_batch(read_clip, entries):
  return [read_clip(entry) for entry in entries]


def _read_clips(entries, read_clip, workers):
  if workers == 1 or len(entries) == 1:
    yield from map(read_clip, entries)
    return

  batch_size = min(16, (len(entries) + workers - 1) // workers)
  batches = (entries[i : i + batch_size] for i in range(0, len(entries), batch_size))
  # Spawn avoids forking the parent's CUDA runtime. Bound pending results and
  # consume in manifest order so clip IDs remain stable.
  with ProcessPoolExecutor(
    max_workers=workers, mp_context=multiprocessing.get_context("spawn")
  ) as pool:
    pending = deque()
    for _ in range(2 * workers):
      batch = next(batches, None)
      if batch is None:
        break
      pending.append(pool.submit(_read_batch, read_clip, batch))
    while pending:
      loaded = pending.popleft().result()
      batch = next(batches, None)
      if batch is not None:
        pending.append(pool.submit(_read_batch, read_clip, batch))
      yield from loaded


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
    load_workers: int = 1,
    show_progress: bool = False,
  ):
    if storage not in ("full", "compact") or fk_cache_frames < 0:
      raise ValueError("Invalid motion storage or FK cache capacity")
    if load_workers < 1:
      raise ValueError("Motion loading needs at least one worker")
    self.storage = storage
    self.compact_root_pos = None
    self.compact_rest = None
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
      compact_cpu = (
        torch.empty((total, 3), dtype=torch.float32),
        torch.empty((total, 10 + 2 * len(joint_names)), dtype=torch.float16),
      )
    read_clip = partial(
      _read_clip,
      joint_names=joint_names,
      body_names=body_names,
      compact=storage == "compact",
      require_frame_count=manifest is not None,
    )
    with tqdm(
      total=len(entries),
      desc=f"Loading motions ({storage}, {min(load_workers, len(entries))} worker(s))",
      unit="clip",
      file=sys.stderr,
      mininterval=2,
      disable=not show_progress,
    ) as progress:
      loaded = _read_clips(entries, read_clip, min(load_workers, len(entries)))
      for entry, (count, data) in zip(entries, loaded, strict=True):
        if compact_cpu is None:
          for key, value in data.items():
            arrays[key].append(torch.from_numpy(value))
        else:
          for target, values in zip(compact_cpu, data, strict=True):
            target[cursor : cursor + count].copy_(torch.from_numpy(values))
        cursor += count
        lengths.append(count)
        self.names.append(entry.name)
        progress.update()
    if show_progress:
      print(
        f"Finalizing {cursor:,} motion frames on {device}...",
        file=sys.stderr,
        flush=True,
      )
    self.lengths = torch.tensor(lengths, dtype=torch.long, device=device)
    self.offsets = self.lengths.cumsum(0) - self.lengths
    if compact_cpu is None:
      self.data = {key: torch.cat(value).to(device) for key, value in arrays.items()}
      data_bytes = sum(x.numel() * x.element_size() for x in self.data.values())
    else:
      self.compact_root_pos, self.compact_rest = (
        part.to(device) for part in compact_cpu
      )
      self.data = {}  # Full-body tensors must never be materialized for the whole library.
      self.fk_cache = ReferenceFKCache(
        self.kinematics, min(fk_cache_frames, cursor), device
      )
      data_bytes = sum(part.numel() * part.element_size() for part in compact_cpu)
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
    if self.compact_rest is None:
      return {key: value[indices] for key, value in self.data.items()}
    rest = self.compact_rest[indices]
    joints = len(self.joint_names)
    return {
      "joint_pos": rest[..., 4 : 4 + joints].float(),
      "joint_vel": rest[..., 10 + joints :].float(),
      **self.fk_cache.get(indices, self.compact_root_pos, self.compact_rest),
    }

  def root_fields(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Whole-library root views in storage precision, without FK or copies."""
    if self.compact_rest is None:
      return tuple(self.data[key][:, 0] for key in MOTION_FIELDS[2:])
    rest, joints = self.compact_rest, len(self.joint_names)
    return (
      self.compact_root_pos,
      rest[:, :4],
      rest[:, 4 + joints : 7 + joints],
      rest[:, 7 + joints : 10 + joints],
    )

  def root_state(self, motion_id: Tensor, frame_id: Tensor) -> Tensor:
    indices = self._indices(motion_id, frame_id)
    pos, quat, linear, angular = (
      value[indices].float() for value in self.root_fields()
    )
    if self.compact_rest is not None:
      quat = quat_normalize(quat)
    return torch.cat((pos, quat, linear, angular), dim=-1)


class ReferenceFKCache:
  """Bounded device cache; collisions replace entries, never alter results."""

  def __init__(self, kinematics, capacity: int, device):
    self.kinematics, self.capacity = kinematics, capacity
    self.tags = torch.full((capacity,), -1, dtype=torch.long, device=device)
    self.values = torch.empty(
      (capacity, len(kinematics.body_names), 13), device=device, dtype=torch.float32
    )

  @torch.no_grad()
  def get(self, indices: Tensor, root_pos: Tensor, rest: Tensor) -> dict[str, Tensor]:
    unique, inverse = indices.flatten().unique(return_inverse=True)
    result = self.values.new_empty((len(unique), len(self.kinematics.body_names), 13))
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
      frames = unique[chunk]
      state = torch.cat((root_pos[frames], rest[frames].float()), dim=-1)
      bodies = self.kinematics(state)
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
