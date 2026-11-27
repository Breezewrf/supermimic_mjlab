"""Analytic velocity task signals shared by real and imagined trajectories."""

import math

import torch
from torch import Tensor

from ..config.morphology import state_layout
from ..config.task_cfg import MOTION_ANCHOR_INDEX, STEP_DT
from ..config.velocity_cfg import VelocityConfig
from .state import quat_apply, quat_inverse_apply, yaw_quat

# MotionBrick locomotion_vx_0_vy_0, frame 1617 (32.34 s), native 29DoF order.
STAND_STILL_JOINT_POS = (
  -0.1303912103176117,  # left_hip_pitch_joint
  0.02017482928931713,  # left_hip_roll_joint
  -0.006220028270035982,  # left_hip_yaw_joint
  0.22090697288513184,  # left_knee_joint
  -0.16351665556430817,  # left_ankle_pitch_joint
  0.022092321887612343,  # left_ankle_roll_joint
  -0.014323798939585686,  # right_hip_pitch_joint
  -0.07477139681577682,  # right_hip_roll_joint
  -0.402044415473938,  # right_hip_yaw_joint
  0.11557622998952866,  # right_knee_joint
  -0.1522573083639145,  # right_ankle_pitch_joint
  0.016696227714419365,  # right_ankle_roll_joint
  -0.015644552186131477,  # waist_yaw_joint
  0.04290251433849335,  # waist_roll_joint
  0.0016707987524569035,  # waist_pitch_joint
  -0.06607262045145035,  # left_shoulder_pitch_joint
  0.34128251671791077,  # left_shoulder_roll_joint
  -0.7624660730361938,  # left_shoulder_yaw_joint
  1.0942018032073975,  # left_elbow_joint
  0.2753640115261078,  # left_wrist_roll_joint
  0.0016952131409198046,  # left_wrist_pitch_joint
  -0.11200642585754395,  # left_wrist_yaw_joint
  0.14030113816261292,  # right_shoulder_pitch_joint
  -0.30800795555114746,  # right_shoulder_roll_joint
  0.3396586775779724,  # right_shoulder_yaw_joint
  1.1322705745697021,  # right_elbow_joint
  -0.5219122171401978,  # right_wrist_roll_joint
  -0.06335048377513885,  # right_wrist_pitch_joint
  0.014771653339266777,  # right_wrist_yaw_joint
)


def measured_velocity(state: Tensor) -> Tensor:
  planar = quat_inverse_apply(
    yaw_quat(state[..., 3:7]), state[..., state_layout(state).v]
  )[..., :2]
  return torch.cat((planar, state[..., state_layout(state).w][..., 2:3]), -1)


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
  self_collision_cost: Tensor | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
  standing_q = default_q
  if cfg.stand_still_reference == "motionbrick":
    target = STAND_STILL_JOINT_POS
    if default_q.numel() == 23:
      # 23DoF omits waist roll/pitch and wrist pitch/yaw on both arms.
      target = tuple(
        q for i, q in enumerate(target) if i not in (13, 14, 20, 21, 27, 28)
      )
    standing_q = state.new_tensor(target)
  actual = measured_velocity(state)
  error = actual - command
  linear_error = error[..., :2].square().sum(-1)
  yaw_error = error[..., 2].square()
  if cfg.tracking_vertical_weight or cfg.tracking_angular_xy_weight:
    layout = state_layout(state)
    orientation = state[..., 3:7]
    vertical = quat_inverse_apply(orientation, state[..., layout.v])[..., 2]
    angular_xy = quat_inverse_apply(orientation, state[..., layout.w])[..., :2]
    linear_error = linear_error + cfg.tracking_vertical_weight * vertical.square()
    yaw_error = yaw_error + cfg.tracking_angular_xy_weight * angular_xy.square().sum(-1)
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
  q = state[..., state_layout(state).q]
  limit = (
    (q - joint_limits[..., 1]).clamp_min(0) + (joint_limits[..., 0] - q).clamp_min(0)
  ).sum(-1)
  terms = {
    "linear_tracking": 2 * torch.exp(-linear_error / 0.5**2),
    "yaw_tracking": 2 * torch.exp(-yaw_error / 0.5),
    "action_smoothness": -0.3 * (action - previous_action).square().sum(-1),
    "joint_limit": -2 * limit,
    "torso_sideways": -4 * sideways,
    "torso_angular_velocity": -0.05 * torso_omega[..., :2].square().sum(-1),
    "stand_still": cfg.stand_still_weight
    * (q - standing_q).square().sum(-1)
    * standing_command(command, cfg),
  }
  if self_collision_cost is not None:
    terms["self_collisions"] = cfg.self_collision_weight * self_collision_cost
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
