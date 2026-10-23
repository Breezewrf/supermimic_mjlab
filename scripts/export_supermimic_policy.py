"""Export actor-only weights from a training checkpoint without loading replay into RAM."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.tasks.supermimic.models.legacy_policy import LEGACY_REFERENCE_SCHEMA
from src.tasks.supermimic.models.policy import REFERENCE_SCHEMA
from src.tasks.supermimic.rl.checkpoints import export_policy


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  try:
    policy = export_policy(args.checkpoint, args.output)
  except ValueError as error:
    parser.error(str(error))
  print(
    f"Policy weights: {args.output} (iteration {policy.get('iteration', 'unknown')})"
  )
  inferred = (
    " (inferred from Actor encoder input width)"
    if policy.get("reference_schema_inferred")
    else ""
  )
  print(f"Reference schema: {policy['reference_schema']}{inferred}")
  if policy["reference_schema"] == LEGACY_REFERENCE_SCHEMA:
    print(
      "Legacy full-body features preserved; evaluation and playback select the legacy Actor automatically."
    )
  elif policy["reference_schema"] != REFERENCE_SCHEMA:
    print(
      "Legacy policy exported without converting its reference features. "
      "Evaluation and playback require the matching code version."
    )
  print(
    f"Size: {args.checkpoint.stat().st_size / 2**20:.2f} MiB -> "
    f"{args.output.stat().st_size / 2**20:.2f} MiB"
  )


if __name__ == "__main__":
  main()
