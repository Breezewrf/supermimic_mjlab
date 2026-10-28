"""Export policy or world-model weights without loading replay into RAM."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.tasks.supermimic.models.legacy_policy import LEGACY_REFERENCE_SCHEMA
from src.tasks.supermimic.models.policy import REFERENCE_SCHEMAS
from src.tasks.supermimic.rl.checkpoints import export_policy, export_world_model


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument(
    "--component",
    choices=("policy", "world-model"),
    default="policy",
    help="Weights to export (default: policy; world-model requires world weights)",
  )
  args = parser.parse_args()
  try:
    export = export_policy if args.component == "policy" else export_world_model
    exported = export(args.checkpoint, args.output)
  except ValueError as error:
    parser.error(str(error))
  label = "Policy" if args.component == "policy" else "World-model"
  print(
    f"{label} weights: {args.output} (iteration {exported.get('iteration', 'unknown')})"
  )
  if args.component == "policy":
    print_policy_schema(exported)
  else:
    print(f"World-model hidden dimensions: {tuple(exported['config']['world_dims'])}")
    print("Dynamics, normalization buffers, kinematics and termination head preserved.")
  print(
    f"Size: {args.checkpoint.stat().st_size / 2**20:.2f} MiB -> "
    f"{args.output.stat().st_size / 2**20:.2f} MiB"
  )


def print_policy_schema(policy: dict):
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
  elif policy["reference_schema"] not in REFERENCE_SCHEMAS.values():
    print(
      "Legacy policy exported without converting its reference features. "
      "Evaluation and playback require the matching code version."
    )


if __name__ == "__main__":
  main()
