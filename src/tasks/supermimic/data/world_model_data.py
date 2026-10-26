"""Immutable replay corpus with clip-stratified, episode-disjoint validation."""

import hashlib
from collections import defaultdict
from pathlib import Path

import torch
from torch import Tensor

from .replay_buffer import REQUIRED_FIELDS, ReplayBuffer


def file_sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


class ReplayCorpus:
  def __init__(
    self,
    paths: list[Path],
    clip_names: list[str],
    seed: int = 0,
    validation_fraction: float = 0.2,
    strict: bool = True,
  ):
    if not 0 < validation_fraction < 1:
      raise ValueError("Validation fraction must be between zero and one")
    self.clip_names = list(clip_names)
    self.buffers, self.sources, self.snapshots = [], [], []
    self.episodes = []
    seen = set()
    schemas = None
    for path in map(Path, paths):
      digest = file_sha256(path)
      if digest in seen:
        continue
      seen.add(digest)
      saved = torch.load(path, map_location="cpu", weights_only=True)
      buffer = ReplayBuffer.from_state_dict(saved["replay"], restore_rng=False)
      if not len(buffer) or not REQUIRED_FIELDS.issubset(buffer.data):
        raise ValueError(f"Empty or incomplete replay: {path}")
      source_clips = saved["clips"]
      if len(source_clips) != len(set(source_clips)) or not set(source_clips).issubset(
        clip_names
      ):
        raise ValueError(f"Replay clip names do not match manifest: {path}")
      count = min(buffer.steps, buffer.capacity)
      data = {name: value[:count] for name, value in buffer.data.items()}
      observation_shape = data["observation"].shape[2:]
      if observation_shape not in ((93,), (99,), (105,)):
        raise ValueError(f"Wrong replay shape for observation in {path}")
      expected = {
        "state": (71,),
        "next_state": (71,),
        "action": (29,),
        "previous_action": (29,),
        "context": (33,),
        "observation": observation_shape,
        "next_observation": observation_shape,
        "clean_observation": observation_shape,
        "next_clean_observation": observation_shape,
      }
      for name in REQUIRED_FIELDS:
        value = data[name]
        if value.shape[2:] != expected.get(name, ()):
          raise ValueError(f"Wrong replay shape for {name} in {path}")
        if value.is_floating_point() and not torch.isfinite(value).all():
          raise ValueError(f"Nonfinite replay data: {name}")
      for name in ("state", "next_state"):
        if not torch.allclose(
          data[name][..., 3:7].norm(dim=-1),
          torch.ones(count, buffer.num_envs),
          atol=1e-4,
        ):
          raise ValueError(f"Invalid root quaternion in {path}: {name}")
      for name in ("terminated", "time_out", "motion_end"):
        if data[name].dtype != torch.bool:
          raise ValueError(f"Expected boolean {name}")
      for name in ("motion_id", "frame_id", "episode_id", "episode_step"):
        if data[name].dtype != torch.long or (data[name] < 0).any():
          raise ValueError(f"Expected nonnegative int64 {name}")
      schema = {
        name: (value.shape[2:], value.dtype)
        for name, value in data.items()
        if name in REQUIRED_FIELDS
      }
      if schemas is not None and schema != schemas:
        raise ValueError("Replay schemas or dtypes differ between sources")
      schemas = schema
      mapping = torch.tensor([clip_names.index(name) for name in source_clips])
      if (data["motion_id"] >= len(mapping)).any():
        raise ValueError(f"Motion ID outside source clip list: {path}")
      data["motion_id"].copy_(mapping[data["motion_id"]])
      source = len(self.buffers)
      self.buffers.append(buffer)
      self.snapshots.append(buffer.steps)
      self.sources.append({"sha256": digest, "path": str(path.resolve())})
      for env in range(buffer.num_envs):
        for episode in data["episode_id"][:, env].unique().tolist():
          mask = data["episode_id"][:, env] == episode
          motions = data["motion_id"][:, env][mask].unique()
          if len(motions) != 1:
            raise ValueError("One replay episode contains multiple reference clips")
          context = data["context"][:, env][mask]
          if not torch.allclose(context, context[:1].expand_as(context), atol=1e-6):
            raise ValueError("Simulation context changes inside an episode")
          self.episodes.append(
            {
              "source": source,
              "environment": env,
              "episode": episode,
              "motion": int(motions[0]),
              "transitions": int(mask.sum()),
            }
          )
    if not self.buffers:
      raise ValueError("No replay data supplied")
    groups = defaultdict(list)
    for index, row in enumerate(self.episodes):
      groups[row["motion"]].append(index)
    missing = [name for index, name in enumerate(clip_names) if len(groups[index]) < 2]
    if strict and missing:
      raise ValueError(
        f"Formal validation needs at least two episodes per clip; collect more for: {missing}"
      )
    rng = torch.Generator().manual_seed(seed)
    for motion in sorted(groups):
      indices = groups[motion]
      shuffled = torch.randperm(len(indices), generator=rng).tolist()
      validation_count = min(
        len(indices) - 1, max(1, round(len(indices) * validation_fraction))
      )
      held_out = {indices[i] for i in shuffled[:validation_count]}
      for index in indices:
        self.episodes[index]["split"] = "validation" if index in held_out else "train"
    if not any(row["split"] == "validation" for row in self.episodes):
      raise ValueError("At least two episodes of one clip are needed for validation")
    self.validation_masks = [
      torch.zeros(min(b.steps, b.capacity), b.num_envs, dtype=torch.bool)
      for b in self.buffers
    ]
    for row in self.episodes:
      if row["split"] == "validation":
        mask = self.validation_masks[row["source"]]
        data = self.buffers[row["source"]].data
        mask[:, row["environment"]] |= (
          data["episode_id"][: len(mask), row["environment"]] == row["episode"]
        )
    self._index_cache = {}

  def _check_immutable(self):
    if [b.steps for b in self.buffers] != self.snapshots:
      raise ValueError("Replay corpus was modified after its episode split")

  def indices(self, length: int, split: str, require_full: bool = False) -> Tensor:
    self._check_immutable()
    if split not in ("train", "validation"):
      raise ValueError("Unknown split")
    key = (length, split, require_full)
    if key not in self._index_cache:
      tables = []
      for source, buffer in enumerate(self.buffers):
        if length > buffer.capacity:
          continue
        indices = buffer.window_indices(length, require_full)
        held_out = self.validation_masks[source][
          indices[:, 0] % buffer.capacity, indices[:, 1]
        ]
        indices = indices[held_out if split == "validation" else ~held_out]
        tables.append(
          torch.cat(
            (
              torch.full((len(indices), 1), source, dtype=torch.long),
              indices,
            ),
            dim=-1,
          )
        )
      self._index_cache[key] = (
        torch.cat(tables) if tables else torch.empty(0, 3, dtype=torch.long)
      )
    return self._index_cache[key]

  def batch(
    self, indices: Tensor, length: int, device: str = "cpu"
  ) -> dict[str, Tensor]:
    self._check_immutable()
    if indices.ndim != 2 or indices.shape[1] != 3 or not len(indices):
      raise ValueError("Expected nonempty (batch, 3) corpus indices")
    result = {}
    for source in indices[:, 0].unique().tolist():
      positions = (indices[:, 0] == source).nonzero().flatten()
      batch = self.buffers[source].gather_windows(indices[positions, 1:], length)
      for name in (*sorted(REQUIRED_FIELDS), "mask"):
        value = batch[name]
        if name not in result:
          result[name] = torch.empty(
            (len(indices), *value.shape[1:]), dtype=value.dtype
          )
        result[name][positions] = value
    return {name: value.to(device) for name, value in result.items()}

  def training_transitions(self) -> dict[str, Tensor]:
    self._check_immutable()
    return {
      name: torch.cat(
        [
          buffer.data[name][: len(mask)][~mask]
          for buffer, mask in zip(self.buffers, self.validation_masks)
        ]
      )
      for name in ("state", "next_state", "action", "context")
    }

  def split_description(self) -> dict:
    return {
      "sources": [row["sha256"] for row in self.sources],
      "clip_names": self.clip_names,
      "episodes": self.episodes,
    }
