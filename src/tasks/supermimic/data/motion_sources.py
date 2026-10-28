"""Resolve explicit motion inputs to a stable in-memory index."""

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MotionEntry:
  name: str
  path: Path
  frames: int | None = None


def motion_entries(
  manifest: Path | str | None = None, *,
  motion: Path | str | None = None, motion_dir: Path | str | None = None,
) -> list[MotionEntry]:
  if sum(value is not None for value in (manifest, motion, motion_dir)) != 1:
    raise ValueError("Select exactly one of manifest, motion, or motion_dir")
  if manifest is not None:
    manifest = Path(manifest).resolve()
    entries = []
    for row in json.loads(manifest.read_text())["clips"]:
      relative = Path(row["file"])
      if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Manifest clip paths must stay inside the motion directory")
      entries.append(MotionEntry(
        relative.with_suffix("").as_posix(), manifest.parent / relative,
        row.get("frames"),
      ))
    return entries
  if motion is not None:
    path = Path(motion).resolve()
    if not path.is_file() or path.suffix != ".npz":
      raise ValueError(f"Expected an existing NPZ motion file: {path}")
    return [MotionEntry(path.stem, path)]
  directory = Path(motion_dir).resolve()
  if not directory.is_dir():
    raise ValueError(f"Motion directory does not exist: {directory}")
  paths = sorted(
    (p for p in directory.rglob("*.npz") if p.is_file()),
    key=lambda p: p.relative_to(directory).as_posix(),
  )
  if not paths:
    raise ValueError(f"No NPZ motions found in {directory}")
  return [MotionEntry(p.relative_to(directory).with_suffix("").as_posix(), p) for p in paths]


def select_motion_entries(
  entries: list[MotionEntry], clips: tuple[str, ...] = (),
) -> list[MotionEntry]:
  requested, matched, selected = set(clips), set(), []
  names = set()
  for entry in entries:
    selectors = {entry.name, entry.name + ".npz"}
    if requested and not requested.intersection(selectors):
      continue
    if entry.name in names:
      raise ValueError(f"Duplicate motion: {entry.name}")
    names.add(entry.name)
    matched.update(requested.intersection(selectors))
    selected.append(entry)
  if requested - matched:
    raise ValueError(f"Unknown clips: {sorted(requested - matched)}")
  if not selected:
    raise ValueError("No reference motions selected")
  return selected
