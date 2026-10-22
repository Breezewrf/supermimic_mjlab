"""Explicit state layout and differentiable quaternion/observation operations.

State = [root xyz (0:3), wxyz (3:7), q (7:36), world v (36:39),
world omega (39:42), qdot (42:71)]. Root xyz excludes the environment origin.
Context = [foot friction, encoder bias (1:30), torso COM offset (30:33)].
"""

import torch
from torch import Tensor


def quat_conjugate(q: Tensor) -> Tensor:
  return torch.cat((q[..., :1], -q[..., 1:]), dim=-1)


def quat_mul(a: Tensor, b: Tensor) -> Tensor:
  aw, av, bw, bv = a[..., :1], a[..., 1:], b[..., :1], b[..., 1:]
  return torch.cat(
    (
      aw * bw - (av * bv).sum(-1, keepdim=True),
      aw * bv + bw * av + torch.linalg.cross(av, bv),
    ),
    dim=-1,
  )


def quat_apply(q: Tensor, v: Tensor) -> Tensor:
  # Broadcasting also supports a single vector and batches of quaternions.
  axis, v = torch.broadcast_tensors(q[..., 1:], v)
  cross = 2 * torch.linalg.cross(axis, v)
  return v + q[..., :1] * cross + torch.linalg.cross(axis, cross)


def quat_inverse_apply(q: Tensor, v: Tensor) -> Tensor:
  return quat_apply(quat_conjugate(q), v)


def quat_normalize(q: Tensor) -> Tensor:
  return q / torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp_min(1e-12)


def rotvec_to_quat(vector: Tensor) -> Tensor:
  angle = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
  # sinc avoids division by zero and has the correct derivative at the origin.
  return torch.cat(
    (torch.cos(angle / 2), 0.5 * torch.sinc(angle / (2 * torch.pi)) * vector),
    dim=-1,
  )


def quat_to_rotvec(q: Tensor) -> Tensor:
  q = quat_normalize(q)
  q = torch.where(q[..., :1] < 0, -q, q)
  vector = q[..., 1:]
  sine = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
  angle = 2 * torch.atan2(sine, q[..., :1].clamp_min(1e-12))
  scale = torch.where(sine > 1e-7, angle / sine.clamp_min(1e-12), 2 + sine.square() / 3)
  return scale * vector


def yaw_quat(q: Tensor) -> Tensor:
  w, x, y, z = q.unbind(-1)
  yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
  zero = torch.zeros_like(yaw)
  return torch.stack((torch.cos(yaw / 2), zero, zero, torch.sin(yaw / 2)), dim=-1)


def rotation_error_squared(a: Tensor, b: Tensor) -> Tensor:
  delta = quat_mul(a, quat_conjugate(b))
  sine = torch.linalg.vector_norm(delta[..., 1:], dim=-1)
  angle = 2 * torch.atan2(sine, delta[..., 0].abs().clamp_min(1e-8))
  return angle.square()


def observe(
  state: Tensor,
  previous_action: Tensor,
  default_q: Tensor,
  context: Tensor,
  reference_quat: Tensor | None = None,
) -> Tensor:
  gravity = state.new_tensor([0, 0, -1])
  relative = []
  if reference_quat is not None:
    rotation = quat_normalize(quat_mul(quat_conjugate(state[..., 3:7]), reference_quat))
    relative = [
      quat_apply(rotation, state.new_tensor([1, 0, 0])),
      quat_apply(rotation, state.new_tensor([0, 1, 0])),
    ]
  return torch.cat(
    (
      state[..., 7:36] - default_q + context[..., 1:30],
      state[..., 42:71],
      quat_inverse_apply(state[..., 3:7], state[..., 39:42]),
      quat_inverse_apply(state[..., 3:7], gravity),
      *relative,
      # Keep previous actions last in both the 93- and 99-dimensional layouts.
      previous_action,
    ),
    dim=-1,
  )


def add_sensor_noise(
  observation: Tensor, generator: torch.Generator | None = None
) -> Tensor:
  amplitude = observation.new_tensor(
    [0.01] * 29
    + [0.5] * 29
    + [0.2] * 3
    + [0.05] * 3
    + [0.05] * (observation.shape[-1] - 93)
    + [0.0] * 29
  )
  noise = torch.rand(
    observation.shape,
    device=observation.device,
    dtype=observation.dtype,
    generator=generator,
  )
  return observation + (2 * noise - 1) * amplitude


class ObservationHistory:
  def __init__(
    self, num_envs: int, length: int, device: str, observation_dim: int = 93
  ):
    self.data = torch.zeros(num_envs, length, observation_dim, device=device)

  def reset(self, env_ids: Tensor, initial: Tensor) -> None:
    self.data[env_ids] = initial[:, None, :]

  def append(self, observation: Tensor) -> None:
    self.data = torch.cat((self.data[:, 1:], observation[:, None]), dim=1)
