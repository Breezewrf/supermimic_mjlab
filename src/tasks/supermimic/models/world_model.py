"""SE(2)-equivariant residual dynamics and differentiable analytical task heads."""

from itertools import pairwise

import torch
from torch import Tensor, nn

from ..config.task_cfg import STEP_DT
from ..mdp.rewards import tracking_reward
from ..mdp.state import (
  observe,
  quat_apply,
  quat_conjugate,
  quat_inverse_apply,
  quat_mul,
  quat_normalize,
  quat_to_rotvec,
  rotvec_to_quat,
)
from ..mdp.terminations import failure_terms
from .kinematics import G1Kinematics

GROUPS = (3, 3, 29, 3, 3, 29)
GROUP_NAMES = (
  "root_position",
  "root_orientation",
  "joint_position",
  "root_linear_velocity",
  "root_angular_velocity",
  "joint_velocity",
)
STATE_SCALES = (0.3, 0.4, 0.3, 1.0, 3.14, 5.0)


def encode_state(state: Tensor) -> Tensor:
  q = quat_normalize(state[..., 3:7])
  return torch.cat(
    (
      state[..., 2:3],
      quat_inverse_apply(q, state.new_tensor([0, 0, -1])),
      state[..., 7:36],
      quat_inverse_apply(q, state[..., 36:39]),
      quat_inverse_apply(q, state[..., 39:42]),
      state[..., 42:71],
    ),
    dim=-1,
  )


def integrate(state: Tensor, correction: Tensor | None = None) -> Tensor:
  """Constant-velocity integration plus 70 physical-unit residual corrections."""
  if correction is None:
    correction = state.new_zeros((*state.shape[:-1], 70))
  dp, dr, dq, dv, dw, dqd = correction.split(GROUPS, dim=-1)
  q = quat_normalize(state[..., 3:7])
  angular_body = quat_inverse_apply(q, state[..., 39:42])
  next_q = quat_normalize(quat_mul(q, rotvec_to_quat(angular_body * STEP_DT + dr)))
  return torch.cat(
    (
      state[..., :3] + state[..., 36:39] * STEP_DT + quat_apply(q, dp),
      next_q,
      state[..., 7:36] + state[..., 42:71] * STEP_DT + dq,
      state[..., 36:39] + quat_apply(q, dv),
      state[..., 39:42] + quat_apply(q, dw),
      state[..., 42:71] + dqd,
    ),
    dim=-1,
  )


def correction_targets(state: Tensor, next_state: Tensor) -> Tensor:
  q = quat_normalize(state[..., 3:7])
  rotation = quat_to_rotvec(quat_mul(quat_conjugate(q), next_state[..., 3:7]))
  return torch.cat(
    (
      quat_inverse_apply(
        q, next_state[..., :3] - state[..., :3] - state[..., 36:39] * STEP_DT
      ),
      rotation - quat_inverse_apply(q, state[..., 39:42]) * STEP_DT,
      next_state[..., 7:36] - state[..., 7:36] - state[..., 42:71] * STEP_DT,
      quat_inverse_apply(q, next_state[..., 36:39] - state[..., 36:39]),
      quat_inverse_apply(q, next_state[..., 39:42] - state[..., 39:42]),
      next_state[..., 42:71] - state[..., 42:71],
    ),
    dim=-1,
  )


def state_errors(prediction: Tensor, target: Tensor) -> tuple[Tensor, ...]:
  return (
    prediction[..., :3] - target[..., :3],
    quat_to_rotvec(quat_mul(quat_conjugate(target[..., 3:7]), prediction[..., 3:7])),
    prediction[..., 7:36] - target[..., 7:36],
    prediction[..., 36:39] - target[..., 36:39],
    prediction[..., 39:42] - target[..., 39:42],
    prediction[..., 42:71] - target[..., 42:71],
  )


def normalized_state_error(prediction: Tensor, target: Tensor) -> Tensor:
  return torch.stack(
    [
      (error / scale).square().mean(-1)
      for error, scale in zip(state_errors(prediction, target), STATE_SCALES)
    ],
    dim=-1,
  ).mean(-1)


class WorldModel(nn.Module):
  def __init__(self, metadata: dict, hidden_dims: tuple[int, ...] = (2048, 2048, 2048)):
    super().__init__()
    if not hidden_dims or any(width < 1 for width in hidden_dims):
      raise ValueError("Hidden widths must be positive")
    self.hidden_dims = tuple(hidden_dims)
    self.kinematics = G1Kinematics(metadata)
    widths = (130, *hidden_dims, 70)
    layers = []
    for index, (left, right) in enumerate(pairwise(widths)):
      layers.append(nn.Linear(left, right))
      if index < len(widths) - 2:
        layers.append(nn.ELU())
    self.dynamics = nn.Sequential(*layers)
    nn.init.zeros_(self.dynamics[-1].weight)
    nn.init.zeros_(self.dynamics[-1].bias)
    self.termination_head = nn.Sequential(
      nn.Linear(2, 64), nn.ELU(), nn.Linear(64, 64), nn.ELU(), nn.Linear(64, 1)
    )
    self.register_buffer("input_mean", torch.zeros(130))
    self.register_buffer("input_std", torch.ones(130))
    self.register_buffer("correction_scale", torch.ones(70))

  def features(self, state: Tensor, action: Tensor, context: Tensor) -> Tensor:
    return torch.cat((encode_state(state), action, context), dim=-1)

  @torch.no_grad()
  def fit_normalization(
    self, state: Tensor, action: Tensor, context: Tensor, next_state: Tensor
  ) -> None:
    features = self.features(state, action, context)
    self.input_mean.copy_(features.mean(0))
    self.input_std.copy_(features.std(0, unbiased=False).clamp_min(1e-3))
    targets = correction_targets(state, next_state)
    floor = targets.new_tensor([1e-3] * 35 + [1e-2] * 35)
    self.correction_scale.copy_(targets.std(0, unbiased=False).maximum(floor))

  def forward(self, state: Tensor, action: Tensor, context: Tensor) -> Tensor:
    features = (
      self.features(state, action, context) - self.input_mean
    ) / self.input_std
    return integrate(state, self.dynamics(features) * self.correction_scale)

  def task_heads(
    self,
    next_state: Tensor,
    action: Tensor,
    context: Tensor,
    reference_next: dict[str, Tensor],
    previous_action: Tensor,
  ) -> dict[str, Tensor | dict]:
    bodies = self.kinematics(next_state)
    reward, _ = tracking_reward(
      next_state,
      bodies,
      reference_next,
      action,
      previous_action,
      self.kinematics.joint_limits,
    )
    errors = failure_terms(next_state, reference_next)
    features = torch.stack(
      (errors["height_error"] / 0.25, errors["tilt_error"] / 0.8), dim=-1
    )
    logits = self.termination_head(features).squeeze(-1)
    return {
      "next_state": next_state,
      "bodies": bodies,
      "reward": reward,
      "clean_observation": observe(
        next_state, action, self.kinematics.default_q, context
      ),
      "termination_logit": logits,
      "termination_probability": logits.sigmoid(),
    }

  def predict(
    self,
    state: Tensor,
    action: Tensor,
    context: Tensor,
    reference_next: dict[str, Tensor],
    previous_action: Tensor,
  ) -> dict:
    return self.task_heads(
      self(state, action, context),
      action,
      context,
      reference_next,
      previous_action,
    )
