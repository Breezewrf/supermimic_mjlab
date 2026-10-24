"""Backward-compatible names for the original full-body policy architecture."""

from torch import Tensor

from .policy import (
  LEGACY_GOAL_DIM,
  LEGACY_REFERENCE_SCHEMA,
  MotionActor,
  legacy_reference_features,
)

__all__ = [
  "LEGACY_GOAL_DIM",
  "LEGACY_REFERENCE_SCHEMA",
  "LegacyMotionActor",
  "legacy_reference_features",
]


class LegacyMotionActor(MotionActor):
  def __init__(
    self,
    default_q: Tensor,
    hidden_dims: tuple[int, ...] = (2048, 1024, 512, 256),
    relative_orientation: bool = False,
    amp: str = "none",
    input_normalization: bool = False,
    motion_anchor_ori_b: bool = False,
  ):
    super().__init__(
      default_q,
      hidden_dims,
      relative_orientation,
      amp,
      reference_dim=LEGACY_GOAL_DIM,
      input_normalization=input_normalization,
      motion_anchor_ori_b=motion_anchor_ori_b,
    )
