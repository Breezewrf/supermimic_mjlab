"""Differentiable BeyondMimic tracking rewards shared by simulation and imagination."""

import torch
from torch import Tensor

from ..config.task_cfg import STEP_DT
from .state import (
  quat_apply,
  quat_conjugate,
  quat_mul,
  rotation_error_squared,
  yaw_quat,
)


def tracking_reward(
  state: Tensor,
  bodies: dict[str, Tensor],
  reference: dict[str, Tensor],
  action: Tensor,
  previous_action: Tensor,
  joint_limits: Tensor,
) -> tuple[Tensor, dict[str, Tensor]]:
  """BeyondMimic heading-aligned body targets plus absolute pelvis targets.

  All body dictionaries are ordered with pelvis first and exclude env origins.
  Regularization is applied to executed normalized actions, before PD scaling.
  """
  root_pos, root_quat = state[..., :3], state[..., 3:7]
  ref_pos, ref_quat = reference["body_pos_w"], reference["body_quat_w"]
  delta_yaw = yaw_quat(quat_mul(root_quat, quat_conjugate(ref_quat[..., 0, :])))
  anchor = torch.cat((root_pos[..., :2], ref_pos[..., 0, 2:3]), dim=-1)
  desired_pos = anchor[..., None, :] + quat_apply(
    delta_yaw[..., None, :], ref_pos - ref_pos[..., :1, :]
  )
  desired_quat = quat_mul(delta_yaw[..., None, :].expand_as(ref_quat), ref_quat)
  pos_error = (desired_pos - bodies["body_pos_w"]).square().sum(-1).mean(-1)
  ori_error = rotation_error_squared(desired_quat, bodies["body_quat_w"]).mean(-1)
  lin_error = (
    (reference["body_lin_vel_w"] - bodies["body_lin_vel_w"]).square().sum(-1).mean(-1)
  )
  ang_error = (
    (reference["body_ang_vel_w"] - bodies["body_ang_vel_w"]).square().sum(-1).mean(-1)
  )
  root_pos_error = (ref_pos[..., 0, :] - root_pos).square().sum(-1)
  root_ori_error = rotation_error_squared(ref_quat[..., 0, :], root_quat)
  q = state[..., 7:36]
  limit_cost = (
    (q - joint_limits[..., 1]).clamp_min(0) + (joint_limits[..., 0] - q).clamp_min(0)
  ).sum(-1)
  terms = {
    "body_position": torch.exp(-pos_error / 0.3**2),
    "body_orientation": torch.exp(-ori_error / 0.4**2),
    "body_linear_velocity": torch.exp(-lin_error),
    "body_angular_velocity": torch.exp(-ang_error / 3.14**2),
    "root_position": 0.5 * torch.exp(-root_pos_error / 0.3**2),
    "root_orientation": 0.5 * torch.exp(-root_ori_error / 0.4**2),
    "action_smoothness": -0.1 * (action - previous_action).square().sum(-1),
    "joint_limit": -2 * limit_cost,
  }
  terms = {name: value * STEP_DT for name, value in terms.items()}
  metrics = {
    "body_position_error": pos_error.sqrt(),
    "body_orientation_error": ori_error.sqrt(),
    "root_position_error": root_pos_error.sqrt(),
    "root_orientation_error": root_ori_error.sqrt(),
  }
  return torch.stack(list(terms.values())).sum(0), terms | metrics
