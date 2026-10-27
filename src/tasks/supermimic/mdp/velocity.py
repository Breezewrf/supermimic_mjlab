"""Analytic velocity task signals shared by real and imagined trajectories."""

import math

import torch
from torch import Tensor

from ..config.task_cfg import MOTION_ANCHOR_INDEX, STEP_DT
from ..config.velocity_cfg import VelocityConfig
from .state import quat_apply, quat_inverse_apply, yaw_quat


def measured_velocity(state: Tensor) -> Tensor:
  planar = quat_inverse_apply(yaw_quat(state[..., 3:7]), state[..., 36:39])[..., :2]
  return torch.cat((planar, state[..., 41:42]), -1)


def standing_command(command: Tensor, cfg: VelocityConfig) -> Tensor:
  return (command[..., :2].norm(dim=-1) <= cfg.standing_linear_threshold) & (
    command[..., 2].abs() <= cfg.standing_angular_threshold
  )


def velocity_reward(
  state: Tensor,
  bodies: dict[str, Tensor],
  command: Tensor,
  action: Tensor,
  previous_action: Tensor,
  joint_limits: Tensor,
  default_q: Tensor,
  cfg: VelocityConfig,
) -> tuple[Tensor, dict[str, Tensor]]:
  actual = measured_velocity(state)
  error = actual - command
  heading = yaw_quat(state[..., 3:7])
  torso = bodies["body_quat_w"][..., MOTION_ANCHOR_INDEX, :]
  torso_z = quat_inverse_apply(
    heading, quat_apply(torso, state.new_tensor([0.0, 0.0, 1.0]))
  )
  # sin(sideways lean) is the lateral component of the torso's unit z-axis.
  sideways = torso_z[..., 1].square()
  torso_omega = quat_inverse_apply(
    heading, bodies["body_ang_vel_w"][..., MOTION_ANCHOR_INDEX, :]
  )
  q = state[..., 7:36]
  limit = (
    (q - joint_limits[..., 1]).clamp_min(0) + (joint_limits[..., 0] - q).clamp_min(0)
  ).sum(-1)
  terms = {
    "linear_tracking": 2 * torch.exp(-error[..., :2].square().sum(-1) / 0.5**2),
    "yaw_tracking": 2 * torch.exp(-error[..., 2].square() / 0.5),
    "action_smoothness": -0.3 * (action - previous_action).square().sum(-1),
    "joint_limit": -2 * limit,
    "torso_sideways": -4 * sideways,
    "torso_angular_velocity": -0.05 * torso_omega[..., :2].square().sum(-1),
    "stand_still": cfg.stand_still_weight
    * (q - default_q).square().sum(-1)
    * standing_command(command, cfg),
  }
  terms = {key: value * STEP_DT for key, value in terms.items()}
  return torch.stack(list(terms.values())).sum(0), terms | {
    "linear_velocity_error": error[..., :2].norm(dim=-1),
    "yaw_rate_error": error[..., 2].abs(),
    "action_change": (action - previous_action).square().sum(-1),
  }


def velocity_failure(state: Tensor, cfg: VelocityConfig) -> dict[str, Tensor]:
  gravity_z = quat_inverse_apply(state[..., 3:7], state.new_tensor([0.0, 0.0, -1.0]))[
    ..., 2
  ]
  tilt = 1 + gravity_z
  threshold = 1 - math.cos(math.radians(cfg.tilt_degrees))
  height_probability = torch.sigmoid(
    (cfg.min_height - state[..., 2]) / cfg.height_softness
  )
  tilt_probability = torch.sigmoid((tilt - threshold) / cfg.tilt_softness)
  return {
    "height_failure": state[..., 2] < cfg.min_height,
    "tilt_failure": tilt > threshold,
    "termination_probability": 1 - (1 - height_probability) * (1 - tilt_probability),
  }
