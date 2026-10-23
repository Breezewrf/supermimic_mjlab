"""Inference adapter for the original 268-feature full-body motion encoder.

Retains the feature construction from the pre-730-input policy implementation.
It does not convert weights or change the current training architecture.
"""

import torch
from torch import Tensor, nn

from ..mdp.state import quat_apply, quat_mul, quat_normalize
from .policy import MotionActor, reference_anchor

LEGACY_REFERENCE_SCHEMA = "full_body_268"
LEGACY_GOAL_DIM = 2680


def legacy_reference_features(
  reference: dict[str, Tensor], default_q: Tensor
) -> Tensor:
  origin, inverse_yaw = reference_anchor(reference)
  rotation = inverse_yaw[..., None, None, :]
  position = quat_apply(rotation, reference["body_pos_w"] - origin[..., None, None, :])
  quaternion = quat_normalize(quat_mul(rotation, reference["body_quat_w"]))
  orientation = torch.cat(
    (
      quat_apply(quaternion, quaternion.new_tensor([1, 0, 0])),
      quat_apply(quaternion, quaternion.new_tensor([0, 1, 0])),
    ),
    -1,
  )
  features = torch.cat(
    (
      (reference["joint_pos"] - default_q) / 0.3,
      reference["joint_vel"] / 5,
      (position / 0.3).flatten(-2),
      orientation.flatten(-2),
      quat_apply(rotation, reference["body_lin_vel_w"]).flatten(-2),
      (quat_apply(rotation, reference["body_ang_vel_w"]) / 3.14).flatten(-2),
    ),
    -1,
  )
  if features.shape[-2:] != (10, 268):
    raise ValueError(
      "Legacy Actor needs ten reference frames and fourteen tracked bodies"
    )
  return features.flatten(-2)


class LegacyMotionActor(MotionActor):
  def __init__(
    self,
    default_q: Tensor,
    hidden_dims: tuple[int, ...] = (2048, 1024, 512, 256),
    relative_orientation: bool = False,
    amp: str = "none",
  ):
    super().__init__(default_q, hidden_dims, relative_orientation, amp)
    self.encoder[0] = nn.Linear(LEGACY_GOAL_DIM, hidden_dims[0])

  def forward(self, history: Tensor, future_reference: dict[str, Tensor]) -> Tensor:
    return self.forward_features(
      history, legacy_reference_features(future_reference, self.default_q)
    )
