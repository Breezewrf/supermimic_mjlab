"""Separate resumable training snapshots from portable inference artifacts."""

from pathlib import Path

import torch

from ..models.legacy_policy import LEGACY_GOAL_DIM, LEGACY_REFERENCE_SCHEMA
from ..models.policy import GOAL_DIM, REFERENCE_SCHEMA

TRAINING_KIND = "supermimic_online"
POLICY_KIND = "supermimic_actor"
WORLD_MODEL_KIND = "supermimic_world_model"
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
  "reference_schema_inferred",
)
WORLD_MODEL_REQUIRED_FIELDS = (
  "schema_version",
  "config",
  "robot_metadata",
  "simulation",
  "clips",
  "world",
  "manifest_sha256",
)
WORLD_MODEL_OPTIONAL_FIELDS = (
  *POLICY_OPTIONAL_FIELDS,
  "reference_schema",
  "world_updates",
)


def _infer_reference_schema(actor: dict) -> str:
  """Recognize the two historical Actor input layouts without changing weights."""
  weight = actor.get("encoder.0.weight")
  if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
    raise ValueError(
      "Cannot infer reference_schema: expected a 2D Actor encoder.0.weight"
    )
  schemas = {GOAL_DIM: REFERENCE_SCHEMA, LEGACY_GOAL_DIM: LEGACY_REFERENCE_SCHEMA}
  width = weight.shape[1]
  if width not in schemas:
    raise ValueError(
      f"Cannot infer reference_schema from Actor encoder input width {width}; "
      f"recognized widths are {GOAL_DIM} and 2680. Use the checkpoint's matching code version."
    )
  return schemas[width]


def resolve_reference_schema(saved: dict) -> str:
  schema = saved.get("reference_schema")
  return schema if schema is not None else _infer_reference_schema(saved["actor"])


def training_config(saved: dict) -> dict:
  """Fill only historically absent settings, preserving the saved execution mode."""
  schema = resolve_reference_schema(saved)
  dimensions = {REFERENCE_SCHEMA: GOAL_DIM, LEGACY_REFERENCE_SCHEMA: LEGACY_GOAL_DIM}
  if schema not in dimensions:
    raise ValueError(f"Unsupported training reference schema {schema!r}")
  config = {
    "wm_only_iters": 0,
    "compile": False,
    "amp": "none",
    "reference_dim": dimensions[schema],
    "actor_input_normalization": False,
    "critic_input_normalization": False,
    **saved["config"],
  }
  if config["reference_dim"] != dimensions[schema]:
    raise ValueError("Checkpoint reference dimension and schema differ")
  if saved["actor"]["encoder.0.weight"].shape[1] != config["reference_dim"]:
    raise ValueError("Checkpoint Actor input width and reference dimension differ")
  return config


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
  missing = (set(POLICY_REQUIRED_FIELDS) - {"reference_schema"}) - saved.keys()
  if missing:
    raise ValueError(f"Checkpoint lacks policy metadata: {', '.join(sorted(missing))}")
  policy = {
    "kind": POLICY_KIND,
    **{
      key: _cpu_copy(saved[key])
      for key in (*POLICY_REQUIRED_FIELDS, *POLICY_OPTIONAL_FIELDS)
      if key in saved
    },
  }
  if saved.get("reference_schema") is None:
    policy["reference_schema"] = resolve_reference_schema(saved)
    policy["reference_schema_inferred"] = True
  return policy


def world_model_checkpoint(saved: dict) -> dict:
  """Keep dynamics, normalization, task heads and reconstruction metadata only."""
  if saved.get("kind") == POLICY_KIND:
    raise ValueError(
      "Policy-only checkpoints do not contain a world model; use a full training checkpoint"
    )
  if (
    saved.get("kind") not in (TRAINING_KIND, WORLD_MODEL_KIND)
    or saved.get("schema_version") != 1
  ):
    raise ValueError("Expected a SuperMimic training or world-model checkpoint")
  missing = set(WORLD_MODEL_REQUIRED_FIELDS) - saved.keys()
  if missing:
    raise ValueError(
      f"Checkpoint lacks world-model metadata: {', '.join(sorted(missing))}"
    )
  if "world_dims" not in saved["config"]:
    raise ValueError("Checkpoint lacks world-model configuration: world_dims")
  # The world model does not require Actor weights or a reference encoder schema.
  # Copy only selected fields so replay/optimizer storage is never visited.
  return {
    "kind": WORLD_MODEL_KIND,
    **{
      key: _cpu_copy(saved[key])
      for key in (*WORLD_MODEL_REQUIRED_FIELDS, *WORLD_MODEL_OPTIONAL_FIELDS)
      if key in saved
    },
  }


def save_checkpoint(path: Path, data: dict):
  """Replace a checkpoint only after its new file has been fully written."""
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_suffix(path.suffix + ".tmp")
  torch.save(data, temporary)
  temporary.replace(path)


def _export_inference(checkpoint: Path, output: Path, select, label: str) -> dict:
  if checkpoint.resolve() == output.resolve():
    raise ValueError(f"{label} output must differ from the source checkpoint")
  saved = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
  exported = select(saved)
  exported["source_checkpoint"] = str(checkpoint.resolve())
  save_checkpoint(output, exported)
  return exported


def export_policy(checkpoint: Path, output: Path) -> dict:
  return _export_inference(checkpoint, output, policy_checkpoint, "Policy")


def export_world_model(checkpoint: Path, output: Path) -> dict:
  return _export_inference(checkpoint, output, world_model_checkpoint, "World-model")
