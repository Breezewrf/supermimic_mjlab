"""Velocity exports load without a manifest, reference encoder or motion library."""

from pathlib import Path

import torch

from ..config.morphology import RobotLayout
from ..config.velocity_cfg import VelocityConfig
from ..models.policy import MotionActor
from ..models.velocity_policy import VelocityActor, VelocityEncoder


def load_velocity_actor(checkpoint: Path, device="cpu"):
  saved = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
  if (
    saved.get("kind") != "supermimic_velocity_actor" or saved.get("schema_version") != 1
  ):
    raise ValueError("Expected an exported SuperMimic velocity actor")
  layout = RobotLayout(len(saved["robot_metadata"]["joint_names"]))
  if (
    saved.get("observation_schema") != f"proprioceptive_{layout.observation_dim}_v1"
    or saved.get("history_length") != 10
  ):
    raise ValueError("Unsupported velocity observation schema")
  cfg = saved["config"]
  if cfg.get("relative_orientation", False) or cfg.get("motion_anchor_ori_b", False):
    raise ValueError("Velocity inference cannot depend on a motion orientation")
  velocity = VelocityConfig(**saved["velocity_config"])
  motion = MotionActor(
    saved["robot_metadata"]["default_q"],
    tuple(cfg["actor_dims"]),
    amp=cfg.get("amp", "none"),
    reference_dim=cfg["reference_dim"],
    input_normalization=cfg["actor_input_normalization"],
  )
  actor = VelocityActor(
    motion,
    VelocityEncoder(
      velocity.command_scale, cfg.get("amp", "none"), layout.observation_dim
    ),
  )
  actor.load_state_dict(saved["actor"])
  if not torch.equal(
    actor.encoder.command_scale, torch.tensor(velocity.command_scale).float()
  ):
    raise ValueError("Velocity command scaling metadata/weights disagree")
  return actor.to(device).eval(), saved
