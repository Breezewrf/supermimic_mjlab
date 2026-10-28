"""Reference-free velocity actor and heading-invariant privileged critic."""

import torch
from torch import Tensor, nn

from ..config.morphology import RobotLayout, state_layout
from ..mdp.state import (
  quat_apply,
  quat_conjugate,
  quat_inverse_apply,
  quat_mul,
  yaw_quat,
)
from .execution import run_mlp
from .normalization import RunningNorm
from .policy import MotionActor, mlp


class VelocityEncoder(nn.Module):
  def __init__(self, command_scale=(2.0, 1.32, 3.35), amp="none", observation_dim=93):
    super().__init__()
    self.amp = amp
    self.register_buffer("command_scale", torch.tensor(command_scale).float())
    self.observation_dim = observation_dim
    self.network = mlp(10 * observation_dim + 3, (256, 256), 32)

  def forward(self, normalized_history: Tensor, command: Tensor) -> Tensor:
    features = torch.cat(
      (command / self.command_scale, normalized_history.flatten(-2)), -1
    )
    return run_mlp(self.network, features, self.amp).tanh()


class VelocityActor(nn.Module):
  """Standalone export: no motion encoder or library is needed for inference."""

  def __init__(self, motion: MotionActor, encoder: VelocityEncoder):
    super().__init__()
    self.observation_dim = RobotLayout(motion.default_q.numel()).observation_dim
    if (
      motion.observation_dim != self.observation_dim
      or encoder.observation_dim != self.observation_dim
    ):
      raise ValueError(
        "Velocity decoding requires matching proprioceptive observations"
      )
    self.decoder = motion.decoder
    self.obs_norm = motion.obs_norm
    self.encoder = encoder
    self.amp = motion.amp
    self.input_normalization = motion.input_normalization
    self.register_buffer("observation_scale", motion.observation_scale.detach().clone())
    self.register_buffer("default_q", motion.default_q.detach().clone())

  def forward(self, history: Tensor, command: Tensor) -> Tensor:
    if history.shape[-2:] != (10, self.observation_dim) or command.shape != (
      *history.shape[:-2],
      3,
    ):
      raise ValueError("Expected ten proprioceptive history frames and command (...,3)")
    observations = (
      self.obs_norm(history)
      if self.input_normalization
      else history / self.observation_scale
    )
    latent = self.encoder(observations, command)
    return run_mlp(
      self.decoder, torch.cat((latent, observations.flatten(-2)), -1), self.amp
    )


class VelocityCritic(nn.Module):
  def __init__(
    self,
    default_q: Tensor,
    command_scale=(2.0, 1.32, 3.35),
    hidden_dims=(256, 128),
    amp="none",
  ):
    super().__init__()
    self.amp = amp
    layout = RobotLayout(default_q.numel())
    self.register_buffer("default_q", default_q.detach().clone().float())
    self.register_buffer("command_scale", torch.tensor(command_scale).float())
    self.register_buffer(
      "context_scale",
      torch.tensor([1.0] + [0.01] * layout.num_joints + [0.025, 0.05, 0.05]),
    )
    # height + 6D rotation + q + v + omega + qdot + command + action + context
    self.input_norm = RunningNorm(4 * layout.num_joints + 20)
    self.value = mlp(4 * layout.num_joints + 20, tuple(hidden_dims), 1)

  def features(
    self, state: Tensor, command: Tensor, previous_action: Tensor, context: Tensor
  ) -> Tensor:
    heading = yaw_quat(state[..., 3:7])
    rotation = quat_mul(quat_conjugate(heading), state[..., 3:7])
    orientation = torch.cat(
      (
        quat_apply(rotation, state.new_tensor([1.0, 0.0, 0.0])),
        quat_apply(rotation, state.new_tensor([0.0, 1.0, 0.0])),
      ),
      -1,
    )
    return torch.cat(
      (
        state[..., 2:3],
        orientation,
        (state[..., state_layout(state).q] - self.default_q) / 0.3,
        quat_inverse_apply(heading, state[..., state_layout(state).v]),
        quat_inverse_apply(heading, state[..., state_layout(state).w]) / 3.14,
        state[..., state_layout(state).qd] / 5,
        command / self.command_scale,
        previous_action,
        context / self.context_scale,
      ),
      -1,
    )

  def forward_features(self, features: Tensor) -> Tensor:
    return run_mlp(self.value, self.input_norm(features), self.amp).squeeze(-1)

  def forward(self, state, command, previous_action, context):
    return self.forward_features(
      self.features(state, command, previous_action, context)
    )
