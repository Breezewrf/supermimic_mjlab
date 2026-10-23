"""Separate resumable training snapshots from portable actor-only artifacts."""

from pathlib import Path

import torch

TRAINING_KIND = "supermimic_online"
POLICY_KIND = "supermimic_actor"
POLICY_REQUIRED_FIELDS = (
  "schema_version",
  "reference_schema",
  "config",
  "robot_metadata",
  "simulation",
  "clips",
  "actor",
  "manifest_sha256",
)
POLICY_OPTIONAL_FIELDS = (
  "iteration",
  "environment_steps",
  "runtime",
  "best_return",
  "evaluation_limit",
  "source_checkpoint",
)


def _cpu_copy(value):
  if isinstance(value, torch.Tensor):
    # Clone also breaks storage aliases: a small view must not serialize a large
    # training buffer that happens to share its underlying storage.
    return value.detach().cpu().clone()
  if isinstance(value, dict):
    return {key: _cpu_copy(item) for key, item in value.items()}
  if isinstance(value, list):
    return [_cpu_copy(item) for item in value]
  if isinstance(value, tuple):
    return tuple(_cpu_copy(item) for item in value)
  return value


def policy_checkpoint(saved: dict) -> dict:
  """Keep only actor inference data, without visiting replay/optimizer tensors."""
  if (
    saved.get("kind") not in (TRAINING_KIND, POLICY_KIND)
    or saved.get("schema_version") != 1
  ):
    raise ValueError("Expected a SuperMimic training checkpoint or policy weights")
  missing = set(POLICY_REQUIRED_FIELDS) - saved.keys()
  if missing:
    raise ValueError(f"Checkpoint lacks policy metadata: {', '.join(sorted(missing))}")
  return {
    "kind": POLICY_KIND,
    **{
      key: _cpu_copy(saved[key])
      for key in (*POLICY_REQUIRED_FIELDS, *POLICY_OPTIONAL_FIELDS)
      if key in saved
    },
  }


def save_checkpoint(path: Path, data: dict):
  """Replace a checkpoint only after its new file has been fully written."""
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_suffix(path.suffix + ".tmp")
  torch.save(data, temporary)
  temporary.replace(path)


def export_policy(checkpoint: Path, output: Path) -> dict:
  if checkpoint.resolve() == output.resolve():
    raise ValueError("Policy output must differ from the source training checkpoint")
  saved = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
  policy = policy_checkpoint(saved)
  policy["source_checkpoint"] = str(checkpoint.resolve())
  save_checkpoint(output, policy)
  return policy
