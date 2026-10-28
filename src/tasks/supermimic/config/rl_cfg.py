"""Online world-model and actor-critic training configuration."""

import math
from dataclasses import dataclass

from .morphology import robot_layout


@dataclass(frozen=True)
class OnlineConfig:
  seed: int = 0
  num_envs: int = 64
  capacity: int = 1000
  warmup_steps: int = 32
  wm_only_iters: int = 0
  collect_steps: int = 5
  updates_per_collect: int = 2
  wm_batches: int = 10
  wm_batch_size: int = 256
  policy_batch_size: int = 64
  horizon: int = 32
  critic_epochs: int = 5
  critic_minibatches: int = 4
  learning_rate: float = 2e-4
  gamma: float = 0.99
  gae_lambda: float = 0.95
  action_noise: float = 0.1
  episode_length_s: float = 10.0
  world_dims: tuple[int, ...] = (2048, 2048, 2048)
  actor_dims: tuple[int, ...] = (2048, 1024, 512, 256)
  critic_dims: tuple[int, ...] = (256, 128)
  clips: tuple[str, ...] = ()
  termination_mode: str = "learned"
  relative_orientation: bool = False
  motion_anchor_ori_b: bool = False
  diagnostics_batch_size: int = 64
  compile: bool = True
  amp: str = "bf16"
  reference_dim: int | None = None
  actor_input_normalization: bool = False
  critic_input_normalization: bool = False
  world_prediction: str = "state_delta"
  world_model_normalization: bool = False
  robot: str = "g1_29dof"

  def __post_init__(self):
    if self.world_prediction not in ("state_delta", "velocity_residual"):
      raise ValueError("world_prediction must be state_delta or velocity_residual")
    layout = robot_layout(self.robot)
    if self.reference_dim is None:
      object.__setattr__(self, "reference_dim", layout.reference_dim)
    if self.reference_dim not in (
      layout.reference_dim,
      layout.full_body_reference_dim,
    ):
      raise ValueError("reference_dim does not match the selected robot")
    counts = (
      self.num_envs,
      self.collect_steps,
      self.updates_per_collect,
      self.wm_batches,
      self.wm_batch_size,
      self.policy_batch_size,
      self.horizon,
      self.critic_epochs,
      self.critic_minibatches,
      self.diagnostics_batch_size,
    )
    if min(counts) < 1 or self.capacity < 10 or self.warmup_steps < 8:
      raise ValueError("Positive counts, capacity >= 10 and warmup_steps >= 8 required")
    if self.wm_only_iters < 0:
      raise ValueError("wm_only_iters must be nonnegative")
    if self.critic_minibatches > self.policy_batch_size:
      raise ValueError("Critic minibatches must not exceed the rollout batch size")
    if (
      not 0 < self.gamma < 1
      or not 0 <= self.gae_lambda <= 1
      or not math.isfinite(self.learning_rate)
      or self.learning_rate <= 0
      or self.action_noise < 0
      or self.episode_length_s <= 0
    ):
      raise ValueError("Invalid optimizer, noise or return settings")
    if self.termination_mode not in ("learned", "analytic"):
      raise ValueError("termination_mode must be learned or analytic")
    if self.amp not in ("none", "bf16"):
      raise ValueError("amp must be none or bf16")
    for widths in (self.world_dims, self.actor_dims, self.critic_dims):
      if not widths or min(widths) < 1:
        raise ValueError("Network hidden widths must be positive")
