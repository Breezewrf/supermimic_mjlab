"""Differentiable G1 link poses and velocities; MuJoCo is only a metadata oracle."""

import hashlib
import json

import torch
from torch import Tensor, nn

from ..config.morphology import robot_layout, state_layout, tracked_bodies
from ..config.task_cfg import MOTION_ANCHOR_BODY, MOTION_ANCHOR_INDEX
from ..mdp.state import (
  motion_anchor_ori_b,
  observe,
  quat_apply,
  quat_mul,
  quat_normalize,
  rotvec_to_quat,
)


def robot_metadata(robot: str = "g1_29dof") -> dict:
  import mujoco
  import numpy as np
  from mjlab.entity import Entity

  from ..config.robot_cfg import get_supermimic_robot_cfg

  cfg = (
    get_supermimic_robot_cfg()
    if robot == "g1_29dof"
    else get_supermimic_robot_cfg(robot)
  )
  layout = robot_layout(robot)
  robot = Entity(cfg)
  model = robot.spec.compile()
  if (
    model.nq != 7 + layout.num_joints
    or model.nv != 6 + layout.num_joints
    or model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE
  ):
    raise ValueError("Expected floating-base G1 matching the selected variant")
  joint_ids = [model.joint(name).id for name in robot.joint_names]
  body_ids = [model.body(name).id for name in robot.body_names]
  if body_ids[0] != int(model.jnt_bodyid[0]) or body_ids != list(range(1, model.nbody)):
    raise ValueError("Expected pelvis-first, topologically ordered G1 body tree")
  joint_map = {joint: index for index, joint in enumerate(joint_ids)}
  joints = []
  for body in body_ids:
    count, start = int(model.body_jntnum[body]), int(model.body_jntadr[body])
    if body == body_ids[0] or count == 0:
      joints.append(-1)
    elif count == 1 and model.jnt_type[start] == mujoco.mjtJoint.mjJNT_HINGE:
      joints.append(joint_map[start])
    else:
      raise ValueError("Only fixed bodies or one hinge per non-root body are supported")
  ranges = model.jnt_range[joint_ids]
  center, half = ranges.mean(-1), (ranges[:, 1] - ranges[:, 0]) / 2
  factor = cfg.articulation.soft_joint_pos_limit_factor
  limits = np.stack((center - factor * half, center + factor * half), axis=-1)
  qadr = model.jnt_qposadr[joint_ids]
  default_q = model.key_qpos[0, qadr] if model.nkey else model.qpos0[qadr]
  metadata = {
    "joint_names": list(robot.joint_names),
    "body_names": list(robot.body_names),
    "parents": [int(model.body_parentid[b]) - 1 for b in body_ids],
    "body_joints": joints,
    "body_pos": torch.from_numpy(model.body_pos[body_ids].copy()),
    "body_quat": torch.from_numpy(model.body_quat[body_ids].copy()),
    "joint_axis": torch.from_numpy(model.jnt_axis[joint_ids].copy()),
    "joint_anchor": torch.from_numpy(model.jnt_pos[joint_ids].copy()),
    "joint_reference": torch.from_numpy(model.qpos0[qadr].copy()),
    "default_q": torch.from_numpy(default_q.copy()),
    "joint_limits": torch.from_numpy(limits.copy()),
  }
  digest = hashlib.sha256()
  # Record physical parameters and the kinematic tree for provenance only.
  # Compatibility is checked structurally, without requiring an identical hash.
  for name in (
    "body_mass",
    "body_inertia",
    "body_ipos",
    "body_iquat",
    "dof_armature",
    "dof_damping",
    "dof_frictionloss",
    "actuator_trntype",
    "actuator_trnid",
    "actuator_gear",
    "actuator_gaintype",
    "actuator_gainprm",
    "actuator_biastype",
    "actuator_biasprm",
    "actuator_ctrllimited",
    "actuator_ctrlrange",
    "actuator_forcelimited",
    "actuator_forcerange",
    "geom_type",
    "geom_dataid",
    "geom_bodyid",
    "geom_pos",
    "geom_quat",
    "geom_size",
    "geom_friction",
    "geom_contype",
    "geom_conaffinity",
    "geom_condim",
    "geom_priority",
    "mesh_vertadr",
    "mesh_vertnum",
    "mesh_vert",
    "mesh_faceadr",
    "mesh_facenum",
    "mesh_face",
  ):
    digest.update(name.encode())
    digest.update(getattr(model, name).tobytes())
  for key, value in sorted(metadata.items()):
    digest.update(key.encode())
    digest.update(
      value.numpy().tobytes()
      if isinstance(value, Tensor)
      else json.dumps(value).encode()
    )
  metadata["signature"] = digest.hexdigest()
  return metadata


def validate_robot_metadata(saved: dict, current: dict) -> None:
  """Require compatible tensor ordering and topology, independent of physical SHA."""
  for key in ("joint_names", "body_names", "parents", "body_joints"):
    if key not in saved or tuple(saved[key]) != tuple(current[key]):
      raise ValueError(f"Checkpoint robot model has incompatible {key}")


class G1Kinematics(nn.Module):
  def __init__(self, metadata: dict, body_names: tuple[str, ...] | None = None):
    super().__init__()
    self.layout = robot_layout(
      "g1_23dof" if len(metadata["joint_names"]) == 23 else "g1_29dof"
    )
    if body_names is None:
      body_names = tracked_bodies(self.layout.robot)
    self.parents = tuple(metadata["parents"])
    self.body_joints = tuple(metadata["body_joints"])
    self.body_names = tuple(body_names)
    self.joint_names = tuple(metadata["joint_names"])
    self.signature = metadata.get("signature")
    anchor = metadata["body_names"].index(MOTION_ANCHOR_BODY)
    chain = []
    while anchor > 0:
      chain.append(anchor)
      anchor = self.parents[anchor]
    self.anchor_chain = tuple(reversed(chain))
    self.register_buffer(
      "selected",
      torch.tensor([metadata["body_names"].index(name) for name in body_names]),
    )
    for name in (
      "body_pos",
      "body_quat",
      "joint_axis",
      "joint_anchor",
      "joint_reference",
      "default_q",
      "joint_limits",
    ):
      self.register_buffer(name, metadata[name].detach().clone().float())

  def anchor_quaternion(self, state: Tensor) -> Tensor:
    """Differentiable torso orientation without computing unrelated link poses."""
    quaternion = quat_normalize(state[..., 3:7])
    for body in self.anchor_chain:
      quaternion = quat_mul(quaternion, self.body_quat[body].expand_as(quaternion))
      joint = self.body_joints[body]
      if joint >= 0:
        angle = state[..., 7 + joint] - self.joint_reference[joint]
        quaternion = quat_mul(
          quaternion,
          rotvec_to_quat(angle[..., None] * self.joint_axis[joint]),
        )
    return quaternion

  def observation(
    self,
    state: Tensor,
    action: Tensor,
    context: Tensor,
    reference_quats: Tensor,
    *,
    relative_orientation: bool = False,
    motion_anchor_orientation: bool = False,
    bodies: dict[str, Tensor] | None = None,
  ) -> Tensor:
    anchor_orientation = None
    if motion_anchor_orientation:
      quaternion = (
        self.anchor_quaternion(state)
        if bodies is None
        else bodies["body_quat_w"][..., MOTION_ANCHOR_INDEX, :]
      )
      anchor_orientation = motion_anchor_ori_b(
        quaternion, reference_quats[..., MOTION_ANCHOR_INDEX, :]
      )
    return observe(
      state,
      action,
      self.default_q,
      context,
      reference_quats[..., 0, :] if relative_orientation else None,
      anchor_orientation,
    )

  def forward(self, state: Tensor) -> dict[str, Tensor]:
    if state.shape[-1] != self.layout.state_dim:
      raise ValueError(f"Expected a {self.layout.state_dim}-dimensional physical state")
    pos, quat = [state[..., :3]], [quat_normalize(state[..., 3:7])]
    linear, angular = (
      [state[..., state_layout(state).v]],
      [state[..., state_layout(state).w]],
    )
    for body in range(1, len(self.parents)):
      parent, joint = self.parents[body], self.body_joints[body]
      if parent < 0 or parent >= body:
        raise ValueError("Body parents must precede their children")
      offset = quat_apply(quat[parent], self.body_pos[body])
      base_pos = pos[parent] + offset
      base_quat = quat_mul(quat[parent], self.body_quat[body].expand_as(quat[parent]))
      if joint < 0:
        child_pos, child_quat = base_pos, base_quat
        child_angular = angular[parent]
        child_linear = linear[parent] + torch.linalg.cross(angular[parent], offset)
      else:
        anchor_offset = quat_apply(base_quat, self.joint_anchor[joint])
        anchor = base_pos + anchor_offset
        axis = quat_apply(base_quat, self.joint_axis[joint])
        angle = state[..., 7 + joint] - self.joint_reference[joint]
        rotation = rotvec_to_quat(angle[..., None] * self.joint_axis[joint])
        child_quat = quat_mul(base_quat, rotation)
        rotated_anchor = quat_apply(child_quat, self.joint_anchor[joint])
        child_pos = anchor - rotated_anchor
        child_angular = (
          angular[parent] + axis * state[..., 13 + self.layout.num_joints + joint, None]
        )
        anchor_linear = linear[parent] + torch.linalg.cross(
          angular[parent], anchor - pos[parent]
        )
        child_linear = anchor_linear - torch.linalg.cross(child_angular, rotated_anchor)
      pos.append(child_pos)
      quat.append(child_quat)
      linear.append(child_linear)
      angular.append(child_angular)
    return {
      "body_pos_w": torch.stack(pos, -2)[..., self.selected, :],
      "body_quat_w": torch.stack(quat, -2)[..., self.selected, :],
      "body_lin_vel_w": torch.stack(linear, -2)[..., self.selected, :],
      "body_ang_vel_w": torch.stack(angular, -2)[..., self.selected, :],
    }
