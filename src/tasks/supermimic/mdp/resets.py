"""Reference-relative initial-state perturbations used only in training."""

import torch
from torch import Tensor

from .state import quat_mul, quat_normalize, rotvec_to_quat

RESET_NOISE = {
  "position_m": (0.05, 0.05, 0.01),
  "rotation_vector_rad": (0.1, 0.1, 0.2),
  "linear_velocity_m_s": (0.5, 0.5, 0.2),
  "angular_velocity_rad_s": (0.52, 0.52, 0.78),
  "joint_position_rad": 0.1,
}


def reference_reset_state(
  root: Tensor,
  joint_pos: Tensor,
  joint_vel: Tensor,
  joint_limits: Tensor,
  *,
  perturb: bool,
  generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
  """Perturb world-frame root state [xyz, wxyz, v, omega]; preserve reference qdot.

  The orientation perturbation is a world-frame rotation vector, left-multiplied
  onto the reference rotation (exp(rv) @ R_ref), not independent Euler angles.
  """
  root, joint_pos, joint_vel = root.clone(), joint_pos.clone(), joint_vel.clone()
  if perturb:

    def uniform_like(value, half_width):
      noise = torch.rand(
        value.shape, device=value.device, dtype=value.dtype, generator=generator
      )
      return (2 * noise - 1) * value.new_tensor(half_width)

    root[..., :3] += uniform_like(root[..., :3], RESET_NOISE["position_m"])
    rotation = uniform_like(root[..., :3], RESET_NOISE["rotation_vector_rad"])
    root[..., 3:7] = quat_normalize(quat_mul(rotvec_to_quat(rotation), root[..., 3:7]))
    root[..., 7:10] += uniform_like(root[..., 7:10], RESET_NOISE["linear_velocity_m_s"])
    root[..., 10:13] += uniform_like(
      root[..., 10:13], RESET_NOISE["angular_velocity_rad_s"]
    )
    joint_pos += uniform_like(joint_pos, RESET_NOISE["joint_position_rad"])
  joint_pos = joint_pos.clamp(joint_limits[..., 0], joint_limits[..., 1])
  return root, joint_pos, joint_vel
