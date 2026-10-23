"""Shared checkpoint validation for evaluation and live policy playback."""

from pathlib import Path

import torch

from ..data.world_model_data import file_sha256
from ..models.legacy_policy import LEGACY_REFERENCE_SCHEMA, LegacyMotionActor
from ..models.policy import REFERENCE_SCHEMA, MotionActor
from ..models.world_model import WorldModel
from .checkpoints import (
  POLICY_KIND,
  TRAINING_KIND,
  WORLD_MODEL_KIND,
  WORLD_MODEL_REQUIRED_FIELDS,
  resolve_reference_schema,
)


def load_actor(
  checkpoint: Path, manifest: Path, device: str, clips: tuple[str, ...] = ()
) -> tuple[MotionActor, dict, tuple[str, ...]]:
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
  if schema not in (REFERENCE_SCHEMA, LEGACY_REFERENCE_SCHEMA):
    raise ValueError(
      f"Unsupported reference schema {schema!r}; use its matching code version"
    )
  if saved.get("reference_schema") is None:
    saved = {**saved, "reference_schema": schema, "reference_schema_inferred": True}
  if saved["manifest_sha256"] != file_sha256(manifest):
    raise ValueError("Checkpoint manifest does not match")
  if clips and not set(clips).issubset(saved["clips"]):
    raise ValueError("Evaluation clips must be present in the checkpoint")
  actor_class = LegacyMotionActor if schema == LEGACY_REFERENCE_SCHEMA else MotionActor
  policy = actor_class(
    saved["robot_metadata"]["default_q"],
    tuple(saved["config"]["actor_dims"]),
    saved["config"].get("relative_orientation", False),
    saved["config"].get("amp", "none"),
    input_normalization=saved["config"].get("actor_input_normalization", False),
  )
  policy.load_state_dict(saved["actor"])
  return policy.to(device).eval(), saved, clips or tuple(saved["clips"])


def load_world_model(
  checkpoint: Path, manifest: Path, device: str, clips: tuple[str, ...] = ()
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
  if saved["manifest_sha256"] != file_sha256(manifest):
    raise ValueError("World-model checkpoint manifest does not match")
  if clips and not set(clips).issubset(saved["clips"]):
    raise ValueError("Playback clips must be present in the world-model checkpoint")
  config = saved["config"]
  if "world_dims" not in config:
    raise ValueError("Checkpoint lacks world-model configuration: world_dims")
  model = WorldModel(
    saved["robot_metadata"],
    tuple(config["world_dims"]),
    config.get("termination_mode", "learned"),
    config.get("relative_orientation", False),
    config.get("amp", "none"),
  )
  try:
    model.load_state_dict(saved["world"])
  except RuntimeError as error:
    raise ValueError(
      f"World-model weights do not match its configuration: {error}"
    ) from error
  return model.to(device).eval(), saved


def validate_environment(env, saved: dict) -> None:
  from ..models.kinematics import robot_metadata
  from .online_training import simulation_parameters

  if saved["robot_metadata"]["signature"] != robot_metadata()["signature"]:
    raise ValueError("Checkpoint robot model does not match the current G1")
  if saved.get("simulation") != simulation_parameters(env):
    raise ValueError("Checkpoint simulation configuration does not match")
