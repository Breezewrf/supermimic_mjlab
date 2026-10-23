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
from .normalization import RunningNorm

REFERENCE_DIM = 73
GOAL_DIM = 10 * REFERENCE_DIM
CRITIC_DIM = 71 + GOAL_DIM + 33
REFERENCE_SCHEMA = "root_joint_73_v1"
LEGACY_REFERENCE_SCHEMA = "full_body_268"
LEGACY_GOAL_DIM = 2680


def schema_for_dimension(reference_dim: int) -> str:
  if reference_dim == GOAL_DIM:
    return REFERENCE_SCHEMA
  if reference_dim == LEGACY_GOAL_DIM:
    return LEGACY_REFERENCE_SCHEMA
  raise ValueError("reference_dim must be 730 or 2680")


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


def legacy_reference_features(
  reference: dict[str, Tensor], default_q: Tensor
) -> Tensor:
  """Original full-body features, retaining absolute reference height."""
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
    raise ValueError("Full-body reference needs ten frames and fourteen tracked bodies")
  return features.flatten(-2)


def reference_feature_function(reference_dim: int):
  schema_for_dimension(reference_dim)
  return reference_features if reference_dim == GOAL_DIM else legacy_reference_features


class MotionActor(nn.Module):
  def __init__(
    self,
    default_q: Tensor,
    hidden_dims: tuple[int, ...] = (2048, 1024, 512, 256),
    relative_orientation: bool = False,
    amp: str = "none",
    reference_dim: int = GOAL_DIM,
    input_normalization: bool = False,
  ):
    super().__init__()
    self.hidden_dims = tuple(hidden_dims)
    self.relative_orientation = relative_orientation
    self.amp = amp
    self.reference_dim = reference_dim
    self.reference_schema = schema_for_dimension(reference_dim)
    self.reference_features = reference_feature_function(reference_dim)
    self.input_normalization = input_normalization
    self.observation_dim = 99 if relative_orientation else 93
    self.obs_norm = (
      RunningNorm(self.observation_dim) if input_normalization else nn.Identity()
    )
    self.goal_norm = (
      RunningNorm(reference_dim) if input_normalization else nn.Identity()
    )
    self.register_buffer("default_q", default_q.detach().clone().float())
    self.register_buffer(
      "observation_scale",
      torch.tensor(
        [0.3] * 29 + [5.0] * 29 + [3.14] * 3 + [1.0] * (self.observation_dim - 61)
      ),
    )
    self.encoder = mlp(reference_dim, self.hidden_dims, 32)
    self.decoder = mlp(32 + 10 * self.observation_dim, self.hidden_dims, 29)
    nn.init.normal_(self.decoder[-1].weight, std=0.001)
    nn.init.zeros_(self.decoder[-1].bias)

  def forward_features(self, history: Tensor, goal: Tensor) -> Tensor:
    if history.shape[-2:] != (10, self.observation_dim):
      raise ValueError(
        f"Actor needs ten {self.observation_dim}-dimensional observations"
      )
    latent = run_mlp(self.encoder, self.goal_norm(goal), self.amp).tanh()
    observation = (
      self.obs_norm(history)
      if self.input_normalization
      else history / self.observation_scale
    )
    return run_mlp(
      self.decoder,
      torch.cat((latent, observation.flatten(-2)), -1),
      self.amp,
    )

  def forward(self, history: Tensor, future_reference: dict[str, Tensor]) -> Tensor:
    return self.forward_features(
      history, self.reference_features(future_reference, self.default_q)
    )

  @torch.no_grad()
  def update_normalization(self, observation: Tensor, goal: Tensor) -> None:
    if self.input_normalization:
      self.obs_norm.update(observation)
      self.goal_norm.update(goal)


class MotionCritic(nn.Module):
  def __init__(
    self,
    default_q: Tensor,
    hidden_dims: tuple[int, ...] = (256, 128),
    amp: str = "none",
    reference_dim: int = GOAL_DIM,
    input_normalization: bool = False,
  ):
    super().__init__()
    self.hidden_dims = tuple(hidden_dims)
    self.amp = amp
    self.reference_dim = reference_dim
    self.reference_schema = schema_for_dimension(reference_dim)
    self.reference_features = reference_feature_function(reference_dim)
    self.input_normalization = input_normalization
    self.input_norm = (
      RunningNorm(71 + reference_dim + 33) if input_normalization else nn.Identity()
    )
    self.register_buffer("default_q", default_q.detach().clone().float())
    self.register_buffer(
      "context_scale", torch.tensor([1.0] + [0.01] * 29 + [0.025, 0.05, 0.05])
    )
    self.value = mlp(71 + reference_dim + 33, self.hidden_dims, 1)

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
      goal = self.reference_features(reference, self.default_q)
    return torch.cat((physical, goal, context / self.context_scale), -1)

  def forward(
    self, state: Tensor, reference: dict[str, Tensor], context: Tensor
  ) -> Tensor:
    return self.forward_features(self.features(state, reference, context)).squeeze(-1)

  def forward_features(self, features: Tensor) -> Tensor:
    return run_mlp(self.value, self.input_norm(features), self.amp)

  @torch.no_grad()
  def update_normalization(self, features: Tensor) -> None:
    if self.input_normalization:
      self.input_norm.update(features)
