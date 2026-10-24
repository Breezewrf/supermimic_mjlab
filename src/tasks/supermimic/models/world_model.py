"""SE(2)-equivariant state-increment dynamics and analytical task heads."""

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
from ..mdp.terminations import failure_terms, soft_termination
from .execution import run_mlp
from .kinematics import G1Kinematics
from .normalization import RunningNorm

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
WORLD_PREDICTIONS = ("state_delta", "velocity_residual")


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


def apply_state_delta(state: Tensor, delta: Tensor) -> Tensor:
  """Apply complete one-step increments; no velocity-based extrapolation."""
  dp, dr, dq, dv, dw, dqd = delta.split(GROUPS, dim=-1)
  q = quat_normalize(state[..., 3:7])
  return torch.cat(
    (
      state[..., :3] + quat_apply(q, dp),
      quat_normalize(quat_mul(q, rotvec_to_quat(dr))),
      state[..., 7:36] + dq,
      state[..., 36:39] + quat_apply(q, dv),
      state[..., 39:42] + quat_apply(q, dw),
      state[..., 42:71] + dqd,
    ),
    dim=-1,
  )


def state_delta_targets(state: Tensor, next_state: Tensor) -> Tensor:
  """Complete one-step state differences in the current body frame."""
  q = quat_normalize(state[..., 3:7])
  return torch.cat(
    (
      quat_inverse_apply(q, next_state[..., :3] - state[..., :3]),
      quat_to_rotvec(quat_mul(quat_conjugate(q), next_state[..., 3:7])),
      next_state[..., 7:36] - state[..., 7:36],
      quat_inverse_apply(q, next_state[..., 36:39] - state[..., 36:39]),
      quat_inverse_apply(q, next_state[..., 39:42] - state[..., 39:42]),
      next_state[..., 42:71] - state[..., 42:71],
    ),
    dim=-1,
  )


def integrate(state: Tensor, correction: Tensor | None = None) -> Tensor:
  """Legacy residual dynamics; also supplies the constant-velocity diagnostic baseline."""
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
  def __init__(
    self,
    metadata: dict,
    hidden_dims: tuple[int, ...] = (2048, 2048, 2048),
    termination_mode: str = "learned",
    relative_orientation: bool = False,
    amp: str = "none",
    world_prediction: str = "state_delta",
    world_model_normalization: bool = False,
  ):
    super().__init__()
    if not hidden_dims or any(width < 1 for width in hidden_dims):
      raise ValueError("Hidden widths must be positive")
    self.hidden_dims = tuple(hidden_dims)
    if world_prediction not in WORLD_PREDICTIONS:
      raise ValueError(f"world_prediction must be one of {WORLD_PREDICTIONS}")
    self.world_prediction = world_prediction
    self.world_model_normalization = world_model_normalization
    if termination_mode not in ("learned", "analytic"):
      raise ValueError("termination_mode must be learned or analytic")
    self.termination_mode = termination_mode
    self.relative_orientation = relative_orientation
    self.amp = amp
    self.kinematics = G1Kinematics(metadata)
    widths = (130, *hidden_dims, 70)
    layers = []
    for index, (left, right) in enumerate(pairwise(widths)):
      layers.append(nn.Linear(left, right))
      if index < len(widths) - 2:
        if world_model_normalization:
          layers.extend((nn.LayerNorm(right), nn.SiLU()))
        else:
          layers.append(nn.ELU())
    self.dynamics = nn.Sequential(*layers)
    nn.init.zeros_(self.dynamics[-1].weight)
    nn.init.zeros_(self.dynamics[-1].bias)
    self.termination_head = nn.Sequential(
      nn.Linear(2, 64), nn.ELU(), nn.Linear(64, 64), nn.ELU(), nn.Linear(64, 1)
    )
    self.register_buffer("input_mean", torch.zeros(130))
    self.register_buffer("input_std", torch.ones(130))
    # Retain the historical buffer name so legacy weights load without conversion.
    self.register_buffer("correction_scale", torch.ones(70))
    self.in_norm = RunningNorm(130) if world_model_normalization else nn.Identity()
    self.out_norm = (
      RunningNorm(70, eps=1e-4) if world_model_normalization else nn.Identity()
    )

  def features(self, state: Tensor, action: Tensor, context: Tensor) -> Tensor:
    return torch.cat((encode_state(state), action, context), dim=-1)

  def targets(self, state: Tensor, next_state: Tensor) -> Tensor:
    return (
      state_delta_targets(state, next_state)
      if self.world_prediction == "state_delta"
      else correction_targets(state, next_state)
    )

  @torch.no_grad()
  def update_normalization(
    self, state: Tensor, action: Tensor, context: Tensor, next_state: Tensor
  ) -> None:
    """Count each newly collected real transition once; never update in forward."""
    if self.world_model_normalization:
      self.in_norm.update(self.features(state, action, context))
      self.out_norm.update(self.targets(state, next_state))

  @torch.no_grad()
  def fit_normalization(
    self, state: Tensor, action: Tensor, context: Tensor, next_state: Tensor
  ) -> None:
    if self.world_model_normalization:
      # Offline data are fixed: fit once, without recounting replay minibatches.
      self.update_normalization(state, action, context, next_state)
      return
    features = self.features(state, action, context)
    self.input_mean.copy_(features.mean(0))
    self.input_std.copy_(features.std(0, unbiased=False).clamp_min(1e-3))
    targets = self.targets(state, next_state)
    floor = targets.new_tensor([1e-3] * 35 + [1e-2] * 35)
    self.correction_scale.copy_(targets.std(0, unbiased=False).maximum(floor))

  def forward(self, state: Tensor, action: Tensor, context: Tensor) -> Tensor:
    features = self.features(state, action, context)
    if self.world_model_normalization:
      features = self.in_norm(features)
      delta = self.out_norm.denormalize(run_mlp(self.dynamics, features, self.amp))
    else:
      features = (features - self.input_mean) / self.input_std
      delta = run_mlp(self.dynamics, features, self.amp) * self.correction_scale
    if self.world_prediction == "state_delta":
      return apply_state_delta(state, delta)
    return integrate(state, delta)

  def termination(
    self, next_state: Tensor, reference_next: dict[str, Tensor]
  ) -> dict[str, Tensor]:
    if self.termination_mode == "analytic":
      probability = soft_termination(next_state, reference_next)
      return {"termination_probability": probability}
    errors = failure_terms(next_state, reference_next)
    features = torch.stack(
      (errors["height_error"] / 0.25, errors["tilt_error"] / 0.8), dim=-1
    )
    logits = run_mlp(self.termination_head, features, self.amp).squeeze(-1)
    return {
      "termination_logit": logits,
      "termination_probability": logits.sigmoid(),
    }

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
    return {
      "next_state": next_state,
      "bodies": bodies,
      "reward": reward,
      "clean_observation": observe(
        next_state,
        action,
        self.kinematics.default_q,
        context,
        reference_next["body_quat_w"][..., 0, :] if self.relative_orientation else None,
      ),
      **self.termination(next_state, reference_next),
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
