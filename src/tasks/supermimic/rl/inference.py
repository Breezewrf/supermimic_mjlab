"""Shared checkpoint validation for evaluation and live policy playback."""

from pathlib import Path

import torch

from ..models.legacy_policy import LEGACY_REFERENCE_SCHEMA, LegacyMotionActor
from ..models.policy import REFERENCE_SCHEMAS, MotionActor
from ..models.world_model import WorldModel
from .checkpoints import (
  POLICY_KIND,
  TRAINING_KIND,
  WORLD_MODEL_KIND,
  WORLD_MODEL_REQUIRED_FIELDS,
  resolve_reference_schema,
)


def load_actor(
  checkpoint: Path, manifest: Path | None, device: str, clips: tuple[str, ...] = ()
) -> tuple[MotionActor, dict, tuple[str, ...]]:
  """Load weights; MotionLibrary validates the independently selected motion source."""
  # Training checkpoints include a large replay buffer. Map storage lazily so
  # inference only needs to read actor weights and validation metadata.
  saved = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
  if (
    saved.get("kind") not in (TRAINING_KIND, POLICY_KIND)
    or saved.get("schema_version") != 1
  ):
    raise ValueError(
      "Expected SuperMimic policy weights or an online training checkpoint"
    )
  schema = resolve_reference_schema(saved)
  if schema not in REFERENCE_SCHEMAS.values():
    raise ValueError(
      f"Unsupported reference schema {schema!r}; use its matching code version"
    )
  if saved.get("reference_schema") is None:
    saved = {**saved, "reference_schema": schema, "reference_schema_inferred": True}
  actor_class = LegacyMotionActor if schema == LEGACY_REFERENCE_SCHEMA else MotionActor
  reference_options = {} if actor_class is LegacyMotionActor else {"reference_dim": {s: w for w, s in REFERENCE_SCHEMAS.items()}[schema]}
  policy = actor_class(
    saved["robot_metadata"]["default_q"],
    tuple(saved["config"]["actor_dims"]),
    saved["config"].get("relative_orientation", False),
    saved["config"].get("amp", "none"),
    **reference_options,
    input_normalization=saved["config"].get("actor_input_normalization", False),
    motion_anchor_ori_b=saved["config"].get("motion_anchor_ori_b", False),
  )
  policy.load_state_dict(saved["actor"])
  # An empty selection means all clips in the current manifest, including new data.
  return policy.to(device).eval(), saved, clips


def load_world_model(
  checkpoint: Path, manifest: Path | None, device: str, clips: tuple[str, ...] = ()
) -> tuple[WorldModel, dict]:
  """Load exported world weights or a lazily mapped full training snapshot."""
  saved = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
  if (
    saved.get("kind") not in (TRAINING_KIND, WORLD_MODEL_KIND)
    or saved.get("schema_version") != 1
  ):
    raise ValueError(
      "Expected world-model weights or a full training checkpoint; policy-only files contain no world model"
    )
  missing = set(WORLD_MODEL_REQUIRED_FIELDS) - saved.keys()
  if missing:
    raise ValueError(
      f"Checkpoint lacks world-model metadata: {', '.join(sorted(missing))}"
    )
  config = saved["config"]
  if "world_dims" not in config:
    raise ValueError("Checkpoint lacks world-model configuration: world_dims")
  model = WorldModel(
    saved["robot_metadata"],
    tuple(config["world_dims"]),
    config.get("termination_mode", "learned"),
    config.get("relative_orientation", False),
    config.get("amp", "none"),
    world_prediction=config.get("world_prediction", "velocity_residual"),
    world_model_normalization=config.get("world_model_normalization", False),
    motion_anchor_ori_b=config.get("motion_anchor_ori_b", False),
  )
  try:
    model.load_state_dict(saved["world"])
  except RuntimeError as error:
    raise ValueError(
      f"World-model weights do not match its configuration: {error}"
    ) from error
  return model.to(device).eval(), saved


def validate_environment(env, saved: dict) -> None:
  from ..models.kinematics import robot_metadata, validate_robot_metadata
  from .online_training import simulation_parameters

  validate_robot_metadata(
    saved["robot_metadata"], robot_metadata(env.foundation_config.robot)
  )
  if saved.get("simulation") != simulation_parameters(env):
    raise ValueError("Checkpoint simulation configuration does not match")
