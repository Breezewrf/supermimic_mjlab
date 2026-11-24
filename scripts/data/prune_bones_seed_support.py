"""Remove the reviewed external-support motions from a converted BONES-SEED dataset."""

import argparse
import json
import os
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from shutil import copymode
from tempfile import NamedTemporaryFile

from bones_seed_support import SUPPORT_KEYWORDS


def name_keyword(path):
  stem = Path(path).stem.lower()
  return next((keyword for keyword in SUPPORT_KEYWORDS if keyword in stem), None)


def decision_updates(report_path):
  if not report_path.is_file():
    return []

  latest = {}
  with report_path.open(encoding="utf-8") as report:
    for line in report:
      row = json.loads(line)
      if keyword := name_keyword(row["source"]):
        latest[row["source"]] = (row, keyword)
  return [
    dict(row, status="filtered", reason=["name"], keyword=keyword, output=None)
    for row, keyword in latest.values()
    if row["status"] == "keep"
  ]


@contextmanager
def atomic_replacement(path):
  temporary = None
  try:
    with NamedTemporaryFile(
      mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
      temporary = Path(stream.name)
      yield stream
      stream.flush()
      os.fsync(stream.fileno())
    copymode(path, temporary)
    temporary.replace(path)
  finally:
    if temporary is not None:
      temporary.unlink(missing_ok=True)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "dataset", type=Path, help="Directory containing manifest.json and g1/npz"
  )
  parser.add_argument(
    "--dry-run", action="store_true", help="Show counts without changing files"
  )
  args = parser.parse_args()
  root = args.dataset.resolve()
  manifest_path = root / "manifest.json"
  if not manifest_path.is_file() or not (root / "g1/npz").is_dir():
    parser.error(f"Expected manifest.json and g1/npz in {root}")

  manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
  clips = manifest["clips"]
  manifest["clips"] = [clip for clip in clips if not name_keyword(clip["file"])]
  removed_entries = len(clips) - len(manifest["clips"])
  npz_files = {
    path: keyword
    for path in (root / "g1/npz").rglob("*.npz")
    if (path.is_file() or path.is_symlink()) and (keyword := name_keyword(path))
  }
  report_path = root / "filter_decisions.jsonl"
  updates = decision_updates(report_path)
  for keyword, count in sorted(Counter(npz_files.values()).items()):
    print(f"{keyword}: {count} NPZ files")
  summary = (
    f"{len(npz_files)} NPZ files, {removed_entries} manifest entries, "
    f"{len(updates)} filter decisions"
  )
  if args.dry_run:
    print(f"Would prune {summary}.")
    return

  if updates:
    with (
      atomic_replacement(report_path) as output,
      report_path.open(encoding="utf-8") as original,
    ):
      for line in original:
        output.write(line.rstrip("\n") + "\n")
      for row in updates:
        output.write(json.dumps(row, ensure_ascii=False) + "\n")
  if removed_entries:
    with atomic_replacement(manifest_path) as output:
      json.dump(manifest, output, ensure_ascii=False, indent=2)
      output.write("\n")
  for path in npz_files:
    path.unlink(missing_ok=True)
  print(f"Pruned {summary}.")


if __name__ == "__main__":
  main()
