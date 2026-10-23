"""Motion-conditioned actor and privileged critic with shared feature conventions."""

from itertools import pairwise

import torch
from torch import Tensor, nn

from ..mdp.state import (
  quat_apply,
  quat_conjugate,
  quat_inverse_apply,
  quat_mul,
  quat_normalize,
  yaw_quat,
)
from .execution import run_mlp

REFERENCE_DIM = 73
GOAL_DIM = 10 * REFERENCE_DIM
CRITIC_DIM = 71 + GOAL_DIM + 33
REFERENCE_SCHEMA = "root_joint_73_v1"


def mlp(input_dim: int, hidden: tuple[int, ...], output_dim: int) -> nn.Sequential:
  layers = []
  for left, right in pairwise((input_dim, *hidden)):
    layers.extend((nn.Linear(left, right), nn.ELU()))
  layers.append(nn.Linear(hidden[-1], output_dim))
  return nn.Sequential(*layers)


def reference_anchor(reference: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
  root = reference["body_pos_w"][..., 0, 0, :]
  origin = torch.cat((root[..., :2], torch.zeros_like(root[..., 2:3])), -1)
  inverse_yaw = quat_conjugate(yaw_quat(reference["body_quat_w"][..., 0, 0, :]))
  return origin, inverse_yaw


def reference_features(reference: dict[str, Tensor], default_q: Tensor) -> Tensor:
  """Root displacement/6D rotation/local velocities and joints; 73 per frame."""
  root_pos = reference["body_pos_w"][..., 0, :]
  root_quat = reference["body_quat_w"][..., 0, :]
  inverse_yaw = quat_conjugate(yaw_quat(root_quat[..., 0, :]))
  rotation = inverse_yaw[..., None, :]
  position = quat_apply(rotation, root_pos - root_pos[..., :1, :])
  quaternion = quat_normalize(quat_mul(rotation, root_quat))
  # First two rotation-matrix columns avoid quaternion sign discontinuities.
  orientation = torch.cat(
    (
      quat_apply(quaternion, quaternion.new_tensor([1, 0, 0])),
      quat_apply(quaternion, quaternion.new_tensor([0, 1, 0])),
    ),
    -1,
  )
  parts = (
    position,
    orientation,
    quat_inverse_apply(root_quat, reference["body_lin_vel_w"][..., 0, :]),
    quat_inverse_apply(root_quat, reference["body_ang_vel_w"][..., 0, :]),
    reference["joint_pos"],
    reference["joint_vel"] * 0.1,
  )
  result = torch.cat(parts, -1)
  if result.shape[-2:] != (10, REFERENCE_DIM):
    raise ValueError("Expected ten reference frames with 73 root/joint features")
  return result.flatten(-2)


class MotionActor(nn.Module):
  def __init__(
    self,
    default_q: Tensor,
    hidden_dims: tuple[int, ...] = (2048, 1024, 512, 256),
    relative_orientation: bool = False,
    amp: str = "none",
  ):
    super().__init__()
    self.hidden_dims = tuple(hidden_dims)
    self.relative_orientation = relative_orientation
    self.amp = amp
    self.observation_dim = 99 if relative_orientation else 93
    self.register_buffer("default_q", default_q.detach().clone().float())
    self.register_buffer(
      "observation_scale",
      torch.tensor(
        [0.3] * 29 + [5.0] * 29 + [3.14] * 3 + [1.0] * (self.observation_dim - 61)
      ),
    )
    self.encoder = mlp(GOAL_DIM, self.hidden_dims, 32)
    self.decoder = mlp(32 + 10 * self.observation_dim, self.hidden_dims, 29)
    nn.init.normal_(self.decoder[-1].weight, std=0.001)
    nn.init.zeros_(self.decoder[-1].bias)

  def forward_features(self, history: Tensor, goal: Tensor) -> Tensor:
    if history.shape[-2:] != (10, self.observation_dim):
      raise ValueError(
        f"Actor needs ten {self.observation_dim}-dimensional observations"
      )
    latent = run_mlp(self.encoder, goal, self.amp).tanh()
    return run_mlp(
      self.decoder,
      torch.cat((latent, (history / self.observation_scale).flatten(-2)), -1),
      self.amp,
    )

  def forward(self, history: Tensor, future_reference: dict[str, Tensor]) -> Tensor:
    return self.forward_features(
      history, reference_features(future_reference, self.default_q)
    )


class MotionCritic(nn.Module):
  def __init__(
    self,
    default_q: Tensor,
    hidden_dims: tuple[int, ...] = (256, 128),
    amp: str = "none",
  ):
    super().__init__()
    self.hidden_dims = tuple(hidden_dims)
    self.amp = amp
    self.register_buffer("default_q", default_q.detach().clone().float())
    self.register_buffer(
      "context_scale", torch.tensor([1.0] + [0.01] * 29 + [0.025, 0.05, 0.05])
    )
    self.value = mlp(CRITIC_DIM, self.hidden_dims, 1)

  def features(
    self,
    state: Tensor,
    reference: dict[str, Tensor],
    context: Tensor,
    goal: Tensor | None = None,
  ) -> Tensor:
    origin, inverse_yaw = reference_anchor(reference)
    quaternion = quat_normalize(quat_mul(inverse_yaw, state[..., 3:7]))
    quaternion = torch.where(quaternion[..., :1] < 0, -quaternion, quaternion)
    physical = torch.cat(
      (
        quat_apply(inverse_yaw, state[..., :3] - origin) / 0.3,
        quaternion,
        (state[..., 7:36] - self.default_q) / 0.3,
        quat_apply(inverse_yaw, state[..., 36:39]),
        quat_apply(inverse_yaw, state[..., 39:42]) / 3.14,
        state[..., 42:71] / 5,
      ),
      -1,
    )
    if goal is None:
      goal = reference_features(reference, self.default_q)
    return torch.cat((physical, goal, context / self.context_scale), -1)

  def forward(
    self, state: Tensor, reference: dict[str, Tensor], context: Tensor
  ) -> Tensor:
    return self.forward_features(self.features(state, reference, context)).squeeze(-1)

  def forward_features(self, features: Tensor) -> Tensor:
    return run_mlp(self.value, features, self.amp)
