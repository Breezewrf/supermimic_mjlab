"""Online collect / world-model fit / imagined actor-critic updates."""

import math
import random
import time
from dataclasses import asdict

import numpy as np
import torch

from ..config.rl_cfg import OnlineConfig
from ..data.replay_buffer import ReplayBuffer
from ..mdp.resets import RESET_NOISE
from ..models.policy import MotionActor, MotionCritic, schema_for_dimension
from ..models.world_model import WorldModel
from .checkpoints import resolve_reference_schema, training_config
from .diagnostics import replay_diagnostics, sampler_diagnostics
from .execution import TrainingKernels
from .imagination import frozen_parameters, imagine
from .world_model_training import rollout


def rng_state() -> dict:
  numpy = np.random.get_state()
  return {
    "python": random.getstate(),
    "numpy": (numpy[0], numpy[1].tolist(), *numpy[2:]),
    "torch": torch.get_rng_state(),
    "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
  }


def restore_rng(state: dict) -> None:
  random.setstate(state["python"])
  numpy = state["numpy"]
  np.random.set_state((numpy[0], np.asarray(numpy[1], dtype=np.uint32), *numpy[2:]))
  torch.set_rng_state(state["torch"].cpu())
  if state["cuda"] and torch.cuda.is_available():
    torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def simulation_parameters(env) -> dict:
  return {
    "physics_protocol_version": 2,
    "self_collisions": False,
    "training_reset_noise": dict(RESET_NOISE),
    "randomization_schedule": {
      "foot_friction": "startup",
      "torso_com": "startup",
      "encoder_bias": "reset",
    },
    "nconmax": env.cfg.sim.nconmax,
    "njmax": env.cfg.sim.njmax,
    "physics_dt": env.physics_dt,
    "control_dt": env.step_dt,
    "solver_iterations": env.cfg.sim.mujoco.iterations,
  }


def checked_step(
  loss, module, optimizer, *, element_clip: float | None = None
) -> float:
  if not torch.isfinite(loss):
    optimizer.zero_grad(set_to_none=True)
    raise FloatingPointError(f"Nonfinite training loss in {type(module).__name__}")
  loss.backward()
  parameters = [p for p in module.parameters() if p.grad is not None]
  try:
    norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
  except RuntimeError as error:
    if "non-finite" not in str(error):
      raise
    # A finite FP32 gradient can have an overflowing L2 norm (e.g. 1e20).
    # clip_grad_norm_ raises before modifying gradients in that case. Recompute
    # only this exceptional path in FP64, without changing the clipping order.
    invalid = [
      name
      for name, parameter in module.named_parameters()
      if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
    ]
    if invalid:
      optimizer.zero_grad(set_to_none=True)
      raise FloatingPointError(
        f"Nonfinite gradients in {type(module).__name__}: {', '.join(invalid)}. "
        "Optimizer step was not applied. Use --detect-anomaly to locate the "
        "backward operation; check world-model rollout and critic divergence."
      ) from None
    with torch.no_grad():
      norm = torch.linalg.vector_norm(
        torch.stack(
          [torch.linalg.vector_norm(p.grad, dtype=torch.float64) for p in parameters]
        )
      )
      if not torch.isfinite(norm):
        optimizer.zero_grad(set_to_none=True)
        raise FloatingPointError(
          f"Nonfinite FP64 gradient norm in {type(module).__name__}"
        ) from None
      scale = (1.0 / (norm + 1e-6)).clamp(max=1.0)
      for parameter in parameters:
        parameter.grad.mul_(scale.to(parameter.grad.dtype))
  if element_clip is not None:
    for parameter in module.parameters():
      if parameter.grad is not None:
        parameter.grad.clamp_(-element_clip, element_clip)
  optimizer.step()
  return float(norm)


class OnlineTrainer:
  def __init__(self, env, metadata: dict, config: OnlineConfig):
    if env.num_envs != config.num_envs:
      raise ValueError("Environment count differs from online configuration")
    self.env, self.metadata, self.config = env, metadata, config
    self.device = str(env.device)
    if (
      self.device.startswith("cuda")
      and config.amp == "bf16"
      and not torch.cuda.is_bf16_supported()
    ):
      raise ValueError("This GPU does not support BF16; use --amp none")
    if env.foundation_config.relative_orientation != config.relative_orientation:
      raise ValueError("Environment and policy observation settings differ")
    self.world = WorldModel(
      metadata,
      config.world_dims,
      config.termination_mode,
      config.relative_orientation,
      config.amp,
    ).to(self.device)
    self.actor = MotionActor(
      metadata["default_q"],
      config.actor_dims,
      config.relative_orientation,
      config.amp,
      reference_dim=config.reference_dim,
    ).to(self.device)
    self.critic = MotionCritic(
      metadata["default_q"],
      config.critic_dims,
      config.amp,
      reference_dim=config.reference_dim,
    ).to(self.device)
    self.kernels = TrainingKernels(
      self.actor,
      self.world,
      self.critic,
      enabled=config.compile and self.device.startswith("cuda"),
    )
    self.world_optimizer = torch.optim.Adam(
      self.world.parameters(), lr=config.learning_rate
    )
    self.actor_optimizer = torch.optim.Adam(
      self.actor.parameters(), lr=config.learning_rate, betas=(0.9, 0.95)
    )
    self.critic_optimizer = torch.optim.Adam(
      self.critic.parameters(), lr=config.learning_rate
    )
    self.replay = ReplayBuffer(
      config.num_envs, config.capacity, self.device, config.seed + 10
    )
    self.noise_rng = torch.Generator(device=self.device).manual_seed(config.seed + 11)
    self.imagination_rng = torch.Generator(device=self.device).manual_seed(
      config.seed + 12
    )
    self.critic_rng = torch.Generator(device=self.device).manual_seed(config.seed + 13)
    self.iteration, self.environment_steps = 0, 0
    self.world_updates, self.actor_updates, self.critic_updates = 0, 0, 0
    self.initialized = False
    self.last_collection = {}
    self.validation_windows = {}
    self.phase = "initialization"
    self.simulation_parameters = simulation_parameters(env)

  @torch.no_grad()
  def collect(self, steps: int) -> dict:
    self.phase = "collection"
    started = time.perf_counter()
    rewards = torch.zeros((), device=self.device)
    failures, endings = rewards.long(), rewards.long()
    terms, episode_steps, episode_returns = {}, rewards.clone(), rewards.clone()
    timeouts, motion_ends = rewards.long(), rewards.long()
    for _ in range(steps):
      inputs = self.env.policy_input()
      action = self.actor(inputs["history"], inputs["future_reference"])
      noise = torch.randn(action.shape, device=self.device, generator=self.noise_rng)
      _, reward, terminated, truncated, extras = self.env.step(
        action + self.config.action_noise * noise
      )
      transition = dict(extras["transition"])
      transition["policy_version"] = torch.full_like(
        transition["episode_id"], self.actor_updates
      )
      self.replay.append(transition)
      rewards += reward.mean()
      failures += terminated.sum()
      endings += (terminated | truncated).sum()
      episodes = extras["episodes"]
      episode_steps += episodes["steps"].sum()
      episode_returns += episodes["return"].sum()
      # End reasons are mutually exclusive here: physical failure has priority.
      motion_ends += (episodes["motion_end"] & ~episodes["terminated"]).sum()
      timeouts += (
        episodes["time_out"] & ~episodes["motion_end"] & ~episodes["terminated"]
      ).sum()
      for key, value in extras["metrics"].items():
        if value.dtype != torch.bool:
          terms[key] = terms.get(key, 0.0) + value.mean() / steps
      self.environment_steps += self.config.num_envs
    self.last_collection = {
      "real_reward_per_step": float(rewards / steps),
      "physical_failures": int(failures),
      "episodes_ended": int(endings),
      "collected_transitions": steps * self.config.num_envs,
      "collection_seconds": time.perf_counter() - started,
      "episode_steps_sum": float(episode_steps),
      "episode_returns_sum": float(episode_returns),
      "normal_timeouts": int(timeouts),
      "normal_motion_ends": int(motion_ends),
      **{"real_" + key: float(value) for key, value in terms.items()},
    }
    return self.last_collection

  def bootstrap(self):
    if self.initialized:
      raise ValueError("Trainer already initialized")
    self.collect(self.config.warmup_steps)
    if not len(self.replay.window_indices(8)):
      raise ValueError("Warmup produced no eight-step windows")
    count = min(self.replay.steps, self.replay.capacity)
    transitions = {
      name: self.replay.data[name][:count].flatten(0, 1)
      for name in ("state", "action", "context", "next_state")
    }
    self.world.fit_normalization(**transitions)
    self.initialized = True

  def update_world(self) -> dict:
    self.phase = "world_model"
    totals, counts = {}, {}
    self.world.train()
    for index in range(self.config.wm_batches):
      batch = self.replay.sample_windows(self.config.wm_batch_size, 8)
      self.world_optimizer.zero_grad(set_to_none=True)
      # Full analytic heads are needed only for diagnostics, once per update.
      prediction = rollout(
        self.world,
        batch,
        self.env.library,
        diagnostics=index == self.config.wm_batches - 1,
        dynamics=self.kernels.world_rollout,
      )
      loss, parts = self.kernels.world_loss(prediction, batch)
      norm = checked_step(loss, self.world, self.world_optimizer)
      self.world_updates += 1
      for name, value in {
        "loss": float(loss.detach()),
        "gradient_norm": norm,
        **{key: float(value.detach()) for key, value in parts.items()},
      }.items():
        totals[name] = totals.get(name, 0.0) + value
        counts[name] = counts.get(name, 0) + 1
    self.world_optimizer.zero_grad(set_to_none=True)
    return {"wm_" + key: value / counts[key] for key, value in totals.items()}

  def update_policy(self) -> dict:
    self.phase = "actor"
    cfg = self.config
    starts = self.replay.sample_starts(
      cfg.policy_batch_size,
      10,
      horizon=cfg.horizon,
      clip_lengths=self.env.library.lengths,
    )
    self.actor_optimizer.zero_grad(set_to_none=True)
    self.critic_optimizer.zero_grad(set_to_none=True)
    with frozen_parameters(self.world, self.critic):
      trajectory = imagine(
        self.actor,
        self.world,
        self.critic,
        self.env.library,
        starts,
        horizon=cfg.horizon,
        gamma=cfg.gamma,
        generator=self.imagination_rng,
        kernels=self.kernels,
      )
      objective = float(trajectory["objective"].detach())
      reward_objective = float(trajectory["reward_objective"].detach())
      bootstrap_objective = float(trajectory["bootstrap_objective"].detach())
      actor_norm = checked_step(
        -trajectory["objective"],
        self.actor,
        self.actor_optimizer,
        element_clip=0.1,
      )
    self.actor_updates += 1
    # Targets and critic data come from this one rollout, before any critic fit.
    targets = self.kernels.gae(
      trajectory["rewards"],
      trajectory["continuation"],
      trajectory["values"],
      cfg.gamma,
      cfg.gae_lambda,
    )
    mask = trajectory["mask"]
    inputs = trajectory["critic_features"]
    survival = float(trajectory["survival"].detach().mean())
    reward = float(trajectory["rewards"].detach()[mask].mean())
    # Release the 32-step actor graph before the twenty small critic updates.
    critic_source = {
      key: trajectory[key].detach() for key in ("rewards", "continuation", "values")
    }
    del trajectory
    self.phase = "critic"
    losses, norms = [], []
    for epoch in range(cfg.critic_epochs):
      # Shuffle complete trajectories so the time sum in Eq. (3) is preserved.
      order = torch.randperm(
        len(targets), device=self.device, generator=self.critic_rng
      )
      for minibatch, indices in enumerate(order.tensor_split(cfg.critic_minibatches)):
        self.critic_optimizer.zero_grad(set_to_none=True)
        predictions = self.kernels.critic_value(inputs[indices]).squeeze(-1)
        loss = self.kernels.critic_loss(
          predictions,
          targets[indices],
          mask[indices],
        )
        try:
          norms.append(checked_step(loss, self.critic, self.critic_optimizer))
        except FloatingPointError:
          # This is a diagnostic artifact, not a resumable training checkpoint:
          # the actor and earlier critic minibatches may already have updated.
          tensors = {
            "inputs": inputs[indices],
            "targets": targets[indices],
            "mask": mask[indices],
            "predictions": predictions,
            "loss": loss,
            **{key: value[indices] for key, value in critic_source.items()},
          }
          self.failure_context = {
            "kind": "supermimic_critic_failure",
            "iteration": self.iteration,
            "epoch": epoch,
            "minibatch": minibatch,
            "config": asdict(cfg),
            "critic": {
              key: value.detach().cpu().clone()
              for key, value in self.critic.state_dict().items()
            },
            "tensors": {
              key: value.detach().cpu().clone() for key, value in tensors.items()
            },
          }
          raise
        losses.append(float(loss.detach()))
        self.critic_updates += 1
    self.critic_optimizer.zero_grad(set_to_none=True)
    return {
      "actor_objective": objective,
      "actor_reward_objective": reward_objective,
      "actor_bootstrap_objective": bootstrap_objective,
      "actor_gradient_norm": actor_norm,
      "imagined_reward": reward,
      "survival_at_horizon": survival,
      "critic_loss": sum(losses) / len(losses),
      "critic_gradient_norm": max(norms),
    }

  def train_iteration(self) -> dict:
    if not self.initialized:
      raise ValueError("Call bootstrap or restore a checkpoint first")
    start = time.perf_counter()
    collected = self.iteration % self.config.updates_per_collect == 0
    if collected:
      self.collect(self.config.collect_steps)
    # Paper schedule: cosine to 0.03x over 250k updates, then constant.
    fraction = min(self.iteration / 250_000, 1.0)
    lr = self.config.learning_rate * (
      0.03 + 0.97 * (1 + math.cos(math.pi * fraction)) / 2
    )
    for optimizer in (
      self.world_optimizer,
      self.actor_optimizer,
      self.critic_optimizer,
    ):
      for group in optimizer.param_groups:
        group["lr"] = lr
    world_start = time.perf_counter()
    world_metrics = self.update_world()
    world_seconds = time.perf_counter() - world_start
    wm_only = self.iteration < self.config.wm_only_iters
    policy_metrics, policy_seconds = {}, 0.0
    if not wm_only:
      policy_start = time.perf_counter()
      policy_metrics = self.update_policy()
      policy_seconds = time.perf_counter() - policy_start
    self.iteration += 1
    elapsed = time.perf_counter() - start
    record = {
      "iteration": self.iteration,
      "training_phase": "wm_only" if wm_only else "joint",
      "environment_steps": self.environment_steps,
      "world_updates": self.world_updates,
      "actor_updates": self.actor_updates,
      "critic_updates": self.critic_updates,
      "replay_transitions": len(self.replay),
      "learning_rate": lr,
      "seconds": elapsed,
      "world_model_seconds": world_seconds,
      "policy_and_critic_seconds": policy_seconds,
      "collected_this_iteration": collected,
      **self.last_collection,
      **world_metrics,
      **policy_metrics,
    }
    record["effective_transitions_per_second"] = (
      self.last_collection["collected_transitions"] / elapsed if collected else 0.0
    )
    record["collection_transitions_per_second"] = (
      self.last_collection["collected_transitions"]
      / self.last_collection["collection_seconds"]
      if collected
      else 0.0
    )
    self.phase = "idle"
    if self.device.startswith("cuda"):
      record["peak_gpu_memory_gib"] = (
        torch.cuda.max_memory_allocated(self.device) / 2**30
      )
    return record

  def diagnostics(self):
    return {
      "iteration": self.iteration,
      "world_model": replay_diagnostics(
        self.world,
        self.replay,
        self.env.library,
        self.config.diagnostics_batch_size,
        self.validation_windows,
        self.iteration,
      ),
      "reset_sampler": sampler_diagnostics(self.env.sampler, self.env.library),
    }

  def policy_state_dict(self) -> dict:
    """Actor and simulator metadata only; never serialize replay for inference."""
    return {
      "schema_version": 1,
      "kind": "supermimic_actor",
      "reference_schema": schema_for_dimension(self.config.reference_dim),
      "config": asdict(self.config),
      "robot_metadata": self.metadata,
      "simulation": self.simulation_parameters,
      "clips": self.env.library.names,
      "actor": self.actor.state_dict(),
      "iteration": self.iteration,
      "environment_steps": self.environment_steps,
    }

  def state_dict(self) -> dict:
    return {
      **self.policy_state_dict(),
      "kind": "supermimic_online",
      "device_type": torch.device(self.device).type,
      "world": self.world.state_dict(),
      "critic": self.critic.state_dict(),
      "world_optimizer": self.world_optimizer.state_dict(),
      "actor_optimizer": self.actor_optimizer.state_dict(),
      "critic_optimizer": self.critic_optimizer.state_dict(),
      "replay": self.replay.state_dict(),
      "sampler": self.env.sampler.state_dict(),
      "validation_windows": self.validation_windows,
      "world_updates": self.world_updates,
      "actor_updates": self.actor_updates,
      "critic_updates": self.critic_updates,
      "last_collection": self.last_collection,
      "initialized": self.initialized,
      "rng": rng_state(),
      "noise_rng": self.noise_rng.get_state().cpu(),
      "imagination_rng": self.imagination_rng.get_state().cpu(),
      "critic_rng": self.critic_rng.get_state().cpu(),
      "env_rng": self.env.generator.get_state().cpu(),
    }

  def load_state_dict(self, saved: dict):
    if saved.get("schema_version") != 1 or saved.get("kind") != "supermimic_online":
      raise ValueError("Not an online SuperMimic checkpoint")
    if resolve_reference_schema(saved) != schema_for_dimension(
      self.config.reference_dim
    ):
      raise ValueError(
        f"Checkpoint reference features differ from the {self.config.reference_dim}-dimensional policy; "
        "use the checkpoint's reference dimension to resume"
      )
    saved_config = training_config(saved)
    if saved_config != asdict(self.config) or saved["clips"] != self.env.library.names:
      raise ValueError("Online checkpoint configuration or motion library mismatch")
    if saved["robot_metadata"]["signature"] != self.metadata["signature"]:
      raise ValueError("Online checkpoint robot signature mismatch")
    if saved.get("simulation") != self.simulation_parameters:
      raise ValueError("Online checkpoint simulation configuration mismatch")
    if saved["device_type"] != torch.device(self.device).type:
      raise ValueError(
        "Training resume requires the same RNG backend; actor inference can change device"
      )
    for name in (
      "world",
      "actor",
      "critic",
      "world_optimizer",
      "actor_optimizer",
      "critic_optimizer",
    ):
      getattr(self, name).load_state_dict(saved[name])
    self.replay = ReplayBuffer.from_state_dict(saved["replay"], self.device)
    self.env.sampler.load_state_dict(saved["sampler"])
    self.validation_windows = saved.get("validation_windows", {})
    for name in (
      "iteration",
      "environment_steps",
      "world_updates",
      "actor_updates",
      "critic_updates",
      "last_collection",
      "initialized",
    ):
      setattr(self, name, saved[name])
    restore_rng(saved["rng"])
    for name in ("noise_rng", "imagination_rng", "critic_rng"):
      getattr(self, name).set_state(saved[name].cpu())
    self.env.generator.set_state(saved["env_rng"].cpu())
    # A resumed simulator starts fresh episodes; it is not a physics snapshot.
    self.env.reset()
    count = min(self.replay.steps, self.replay.capacity)
    self.env.episode_id.copy_(self.replay.data["episode_id"][:count].max(0).values + 1)
