"""Staged velocity adaptation and joint motion/velocity analytic policy training."""

import math
import time
from dataclasses import asdict

import torch

from ..data.episode_statistics import EpisodeSummary
from ..data.replay_buffer import ReplayBuffer
from ..mdp.velocity import measured_velocity
from ..models.velocity_policy import VelocityActor, VelocityCritic, VelocityEncoder
from ..models.world_model import normalized_state_error
from .imagination import critic_loss, frozen_parameters, gae_targets, imagine
from .online_training import OnlineTrainer, checked_step, restore_rng
from .velocity_imagination import VelocityKernels, imagine_velocity
from .world_model_training import rollout, world_model_loss


class VelocityTrainer(OnlineTrainer):
  def __init__(self, env, metadata, config, *, total_iterations=1000, stage="joint"):
    if stage not in ("motion", "warmup", "joint"):
      raise ValueError("Unknown velocity training stage")
    if env.motion_count != config.num_envs // 2 or config.num_envs % 2:
      raise ValueError(
        "Joint collection requires an even environment count split equally"
      )
    if config.policy_batch_size < 2 or config.wm_batch_size < 2:
      raise ValueError("Mixed batches require at least two samples")
    if config.critic_minibatches > config.policy_batch_size // 2:
      raise ValueError("Too many critic minibatches for each interface")
    super().__init__(env, metadata, config, total_iterations=total_iterations)
    self.stage = stage
    self.velocity_config = env.velocity_config
    self.velocity_encoder = VelocityEncoder(
      self.velocity_config.command_scale, config.amp
    ).to(self.device)
    self.velocity_critic = VelocityCritic(
      metadata["default_q"],
      self.velocity_config.command_scale,
      config.critic_dims,
      config.amp,
    ).to(self.device)
    self.velocity_optimizer = torch.optim.Adam(
      self.velocity_encoder.parameters(),
      lr=config.learning_rate,
      betas=(0.9, 0.95),
    )
    self.velocity_critic_optimizer = torch.optim.Adam(
      self.velocity_critic.parameters(), lr=config.learning_rate
    )
    self.replay = ReplayBuffer(
      env.motion_count, config.capacity, self.device, config.seed + 10
    )
    self.velocity_replay = ReplayBuffer(
      len(env.velocity_ids),
      config.capacity,
      self.device,
      config.seed + 20,
      task="velocity",
    )
    self.velocity_kernels = VelocityKernels(
      self.actor,
      self.velocity_encoder,
      self.world,
      self.velocity_config,
      enabled=config.compile and self.device.startswith("cuda"),
    )
    self.velocity_updates = 0
    self.warmup_completed = 0
    self.velocity_critic_updates = 0
    self.world_pretrained = False
    self.env.reference_only = self.training_stage != "joint"
    self.env.reset()
    self.env.randomize_initial_episode_progress()

  @property
  def training_stage(self):
    if self.stage == "motion":
      return "motion"
    if (
      self.stage == "warmup"
      or self.warmup_completed < self.velocity_config.warmup_updates
    ):
      return "warmup"
    return "joint"

  @torch.no_grad()
  def collect(self, steps):
    started = time.perf_counter()
    episodes = EpisodeSummary()
    command_resamples = 0
    self.env.reference_only = self.training_stage != "joint"
    m, v = self.env.motion_ids, self.env.velocity_ids
    for _ in range(steps):
      inputs = self.env.policy_input()
      goal = self.actor.reference_features(
        inputs["future_reference"], self.actor.default_q
      )
      if self.training_stage != "warmup" and self.actor.input_normalization:
        observations = (
          inputs["history"][:, -1]
          if self.training_stage == "joint"
          else inputs["history"][m, -1]
        )
        self.actor.obs_norm.update(observations)
        self.actor.goal_norm.update(goal)
      physical = self.env.read_state()
      self.critic.update_normalization(
        self.critic.features(
          physical[m], inputs["future_reference"], self.env.context[m], goal
        )
      )
      self.velocity_critic.input_norm.update(
        self.velocity_critic.features(
          physical[v],
          inputs["command"],
          self.env.previous_action[v],
          self.env.context[v],
        )
      )
      action = torch.empty(self.env.num_envs, 29, device=self.device)
      action[m] = self.actor.forward_features(inputs["history"][m], goal)
      # During motion-only adaptation, collect velocity-side dynamics with the
      # motion teacher too. Velocity optimization starts in the next stage/run.
      if self.training_stage == "motion":
        source = self.env.command_state
        reference = self.env.library.get_future(
          source["reference_motion"], source["reference_frame"], 10
        )
        action[v] = self.actor(inputs["history"][v], reference)
      else:
        latent = self.velocity_encoder(
          self.actor.normalize_history(inputs["history"][v]),
          inputs["command"],
        )
        action[v] = self.actor.decode(inputs["history"][v], latent)
      noise = torch.randn(action.shape, device=self.device, generator=self.noise_rng)
      _, _, _, _, extras = self.env.step(action + self.config.action_noise * noise)
      self.replay.append(extras["motion_transition"])
      self.velocity_replay.append(extras["velocity_transition"])
      if self.config.world_model_normalization:
        self.world.update_normalization(
          **{
            key: extras["transition"][key]
            for key in ("state", "action", "context", "next_state")
          }
        )
      episodes.add(extras["episode_statistics"])
      command_resamples += extras["command_resamples"]
      self.environment_steps += self.env.num_envs
    summary = episodes.summary()
    # Keep historical reward/error names, now sampled only from finished episodes.
    aliases = {
      f"{task}/{key}": value / row["count"]
      for task, row in summary.items()
      for key, value in row["metric_mean_sums"].items()
    }
    self.last_collection = {
      **aliases,
      "statistics_scope": "completed_episodes",
      "episode_statistics": summary,
      "command_resamples": command_resamples,
      "collection_seconds": time.perf_counter() - started,
      "collected_transitions": steps * self.env.num_envs,
    }
    return self.last_collection

  def bootstrap(self):
    if self.initialized:
      raise ValueError("Trainer is already initialized")
    self.collect(self.config.warmup_steps)
    for replay in (self.replay, self.velocity_replay):
      if not len(replay.window_indices(8)):
        raise ValueError("Bootstrap needs eight-step windows for both tasks")
    if not self.config.world_model_normalization and not self.world_pretrained:
      transitions = {
        key: torch.cat(
          [
            replay.data[key][: min(replay.steps, replay.capacity)].flatten(0, 1)
            for replay in (self.replay, self.velocity_replay)
          ]
        )
        for key in ("state", "action", "context", "next_state")
      }
      self.world.fit_normalization(**transitions)
    self.initialized = True

  @torch.no_grad()
  def velocity_diagnostics(self, batch_size=64):
    """Logged-action endpoint errors; this is an in-replay diagnostic, not holdout."""
    rng = self.velocity_replay.generator.get_state()
    report = {
      "scope": "velocity replay, logged-action prediction; not held-out",
      "horizons": {},
    }
    try:
      for horizon in (1, 8, 32):
        try:
          batch = self.velocity_replay.sample_windows(
            batch_size, horizon, require_full=True
          )
        except ValueError as error:
          report["horizons"][str(horizon)] = {"available": False, "reason": str(error)}
          continue
        predicted = self.kernels.world_rollout(
          batch["state"][:, 0], batch["action"], batch["context"], batch["mask"]
        )
        target = batch["next_state"][:, -1]
        error = measured_velocity(predicted[:, -1]) - measured_velocity(target)
        report["horizons"][str(horizon)] = {
          "available": True,
          "samples": batch_size,
          "normalized_state_mse": float(
            normalized_state_error(predicted[:, -1], target).mean()
          ),
          "linear_velocity_rmse": float(error[:, :2].square().mean().sqrt()),
          "yaw_rate_rmse": float(error[:, 2].square().mean().sqrt()),
        }
    finally:
      self.velocity_replay.generator.set_state(rng)
    return report

  def update_world(self):
    cfg = self.config
    totals = {}
    for _ in range(cfg.wm_batches):
      motion = self.replay.sample_windows(cfg.wm_batch_size // 2, 8)
      velocity = self.velocity_replay.sample_windows(
        cfg.wm_batch_size - cfg.wm_batch_size // 2, 8
      )
      self.world_optimizer.zero_grad(set_to_none=True)
      motion_prediction = rollout(
        self.world,
        motion,
        self.env.library,
        diagnostics=False,
        dynamics=self.kernels.world_rollout,
      )
      motion_loss, _ = world_model_loss(
        motion_prediction, motion, analytic_losses=False
      )
      velocity_prediction = {
        "next_state": self.kernels.world_rollout(
          velocity["state"][:, 0],
          velocity["action"],
          velocity["context"],
          velocity["mask"],
        )
      }
      # A motion termination head must never receive fall-only velocity labels.
      velocity_loss, _ = world_model_loss(
        velocity_prediction, velocity, analytic_losses=False
      )
      loss = (
        motion_loss * len(motion["state"]) + velocity_loss * len(velocity["state"])
      ) / cfg.wm_batch_size
      norm = checked_step(loss, self.world, self.world_optimizer)
      for key, value in {
        "wm_loss": float(loss.detach()),
        "wm_motion_loss": float(motion_loss.detach()),
        "wm_velocity_loss": float(velocity_loss.detach()),
        "wm_gradient_norm": norm,
      }.items():
        totals[key] = totals.get(key, 0.0) + value / cfg.wm_batches
      self.world_updates += 1
    self.world_optimizer.zero_grad(set_to_none=True)
    return totals

  def _fit_critic(self, trace, critic, optimizer):
    cfg = self.config
    targets = gae_targets(
      trace["rewards"],
      trace["continuation"],
      trace["values"],
      cfg.gamma,
      cfg.gae_lambda,
    )
    features, mask = trace["critic_features"].detach(), trace["mask"]
    losses = []
    for _ in range(cfg.critic_epochs):
      order = torch.randperm(
        len(targets), device=self.device, generator=self.critic_rng
      )
      for ids in order.tensor_split(cfg.critic_minibatches):
        optimizer.zero_grad(set_to_none=True)
        value = critic.forward_features(features[ids])
        if value.ndim == 3:
          value = value.squeeze(-1)
        loss = critic_loss(value, targets[ids], mask[ids])
        checked_step(loss, critic, optimizer)
        losses.append(float(loss.detach()))
    optimizer.zero_grad(set_to_none=True)
    return sum(losses) / len(losses), len(losses)

  def update_motion(self):
    cfg = self.config
    starts = self.replay.sample_starts(
      cfg.policy_batch_size // 2,
      10,
      horizon=cfg.horizon,
      clip_lengths=self.env.library.lengths,
    )
    self.actor_optimizer.zero_grad(set_to_none=True)
    self.critic_optimizer.zero_grad(set_to_none=True)
    with frozen_parameters(self.world, self.critic):
      trace = imagine(
        self.actor,
        self.world,
        self.critic,
        self.env.library,
        starts,
        horizon=cfg.horizon,
        gamma=cfg.gamma,
        generator=self.imagination_rng,
        sensor_noise=self.env.foundation_config.sensor_noise,
        kernels=self.kernels,
      )
      norm = checked_step(
        -trace["objective"], self.actor, self.actor_optimizer, element_clip=0.1
      )
    objective = float(trace["objective"].detach())
    critic_trace = {
      key: trace[key].detach()
      for key in ("rewards", "continuation", "values", "critic_features", "mask")
    }
    del trace
    loss, count = self._fit_critic(critic_trace, self.critic, self.critic_optimizer)
    self.actor_optimizer.zero_grad(set_to_none=True)
    self.actor_updates += 1
    self.critic_updates += count
    return {
      "motion/objective": objective,
      "motion/actor_gradient_norm": norm,
      "motion/critic_loss": loss,
    }

  def update_velocity(self):
    cfg = self.config
    starts = self.velocity_replay.sample_starts(
      cfg.policy_batch_size - cfg.policy_batch_size // 2, 10
    )
    self.velocity_optimizer.zero_grad(set_to_none=True)
    self.velocity_critic_optimizer.zero_grad(set_to_none=True)
    with frozen_parameters(self.actor, self.world, self.velocity_critic):
      if self.training_stage == "warmup":
        selected = starts["command_source"]
        if not selected.any():
          # This can occur when explicitly restarting warmup from a joint run.
          return {"velocity/warmup_skipped": 1.0}
        history = starts["history"][selected]
        with torch.no_grad():
          reference = self.env.library.get_future(
            starts["reference_motion"][selected],
            starts["reference_frame"][selected],
            10,
          )
          target = self.actor.encode_motion(
            self.actor.reference_features(reference, self.actor.default_q)
          )
        latent = self.velocity_encoder(
          self.actor.normalize_history(history), starts["command"][selected]
        )
        loss = (latent - target).square().mean()
        norm = checked_step(
          loss,
          self.velocity_encoder,
          self.velocity_optimizer,
          element_clip=0.1,
        )
        self.warmup_completed += 1
        self.velocity_updates += 1
        return {
          "velocity/alignment_loss": float(loss.detach()),
          "velocity/actor_gradient_norm": norm,
        }
      trace = imagine_velocity(
        self.actor,
        self.velocity_encoder,
        self.world,
        self.velocity_critic,
        self.env.commands,
        starts,
        horizon=cfg.horizon,
        gamma=cfg.gamma,
        generator=self.imagination_rng,
        sensor_noise=self.env.foundation_config.sensor_noise,
        control_step=self.env.control_steps,
        kernels=self.velocity_kernels,
      )
      loss = (
        -trace["objective"]
        + self.velocity_config.alignment_weight * trace["alignment_loss"]
      )
      norm = checked_step(
        loss, self.velocity_encoder, self.velocity_optimizer, element_clip=0.1
      )
    metrics = {
      "velocity/" + key: float(trace[key].detach())
      for key in (
        "objective",
        "alignment_loss",
        "reward_objective",
        "bootstrap_objective",
      )
    }
    metrics["velocity/survival"] = float(trace["survival"].detach().mean())
    metrics["velocity/actor_gradient_norm"] = norm
    critic_trace = {
      key: trace[key].detach()
      for key in ("rewards", "continuation", "values", "critic_features", "mask")
    }
    del trace
    value_loss, count = self._fit_critic(
      critic_trace, self.velocity_critic, self.velocity_critic_optimizer
    )
    self.velocity_critic_updates += count
    self.velocity_updates += 1
    metrics["velocity/critic_loss"] = value_loss
    return metrics

  def train_iteration(self):
    if not self.initialized:
      raise ValueError("Bootstrap or resume before training")
    started = time.perf_counter()
    stage = self.training_stage
    collected = self.iteration % self.config.updates_per_collect == 0
    if collected:
      self.collect(self.config.collect_steps)
    fraction = min(self.iteration / max(self.total_iterations - 1, 1), 1.0)
    lr = self.config.learning_rate * (
      0.03 + 0.97 * (1 + math.cos(math.pi * fraction)) / 2
    )
    for name in (
      "world_optimizer",
      "actor_optimizer",
      "critic_optimizer",
      "velocity_optimizer",
      "velocity_critic_optimizer",
    ):
      for group in getattr(self, name).param_groups:
        group["lr"] = lr
    metrics = self.update_world()
    if self.iteration >= self.config.wm_only_iters:
      if stage in ("motion", "joint"):
        metrics.update(self.update_motion())
      if stage in ("warmup", "joint"):
        metrics.update(self.update_velocity())
    self.iteration += 1
    return {
      "iteration": self.iteration,
      "stage": stage,
      "collected_this_iteration": collected,
      "learning_rate": lr,
      "environment_steps": self.environment_steps,
      "control_steps": self.env.control_steps,
      "warmup_completed": self.warmup_completed,
      "seconds": time.perf_counter() - started,
      **self.last_collection,
      **metrics,
    }

  def velocity_policy_state_dict(self):
    return {
      "kind": "supermimic_velocity_actor",
      "schema_version": 1,
      "observation_schema": "proprioceptive_93_v1",
      "history_length": 10,
      "config": asdict(self.config),
      "velocity_config": asdict(self.velocity_config),
      "robot_metadata": self.metadata,
      "simulation": self.simulation_parameters,
      "iteration": self.iteration,
      "velocity_updates": self.velocity_updates,
      "actor": VelocityActor(self.actor, self.velocity_encoder).state_dict(),
    }

  def state_dict(self):
    result = super().state_dict()
    result.update(
      {
        "kind": "supermimic_velocity_online",
        "stage": self.stage,
        "velocity_config": asdict(self.velocity_config),
        "control_steps": self.env.control_steps,
        "warmup_completed": self.warmup_completed,
        "velocity_updates": self.velocity_updates,
        "velocity_critic_updates": self.velocity_critic_updates,
        "velocity_replay": self.velocity_replay.state_dict(),
        "command_state": {
          k: v.cpu().clone() for k, v in self.env.command_state.items()
        },
        "command_rng": self.env.command_rng.get_state().cpu(),
        "episode_rng": self.env.episode_rng.get_state().cpu(),
      }
    )
    for name in (
      "velocity_encoder",
      "velocity_critic",
      "velocity_optimizer",
      "velocity_critic_optimizer",
    ):
      result[name] = getattr(self, name).state_dict()
    return result

  def load_state_dict(self, saved, *, allow_timing_override=False):
    if saved.get("kind") != "supermimic_velocity_online":
      raise ValueError("Expected a velocity training checkpoint")
    saved_velocity = saved["velocity_config"]
    current_velocity = asdict(self.velocity_config)
    timing = {
      "resampling_time_range",
      "randomize_initial_episode_progress",
    }
    if saved_velocity.keys() != current_velocity.keys():
      raise ValueError("Velocity checkpoint/configuration mismatch")
    differences = {
      key for key in current_velocity if current_velocity[key] != saved_velocity[key]
    }
    if differences and (not allow_timing_override or not differences <= timing):
      raise ValueError("Velocity checkpoint/configuration mismatch")
    if saved["stage"] != self.stage:
      raise ValueError(
        "Resume preserves stage; use initialization for a new training stage"
      )
    # Base resume expects one replay row per environment; mixed replay only has
    # the motion half, so restore the common pieces explicitly here.
    if (
      saved["config"] != asdict(self.config) or saved["clips"] != self.env.library.names
    ):
      raise ValueError("Training configuration or motion library mismatch")
    if (
      saved["robot_metadata"]["signature"] != self.metadata["signature"]
      or saved["simulation"] != self.simulation_parameters
    ):
      raise ValueError("Robot or simulation mismatch")
    if saved["device_type"] != torch.device(self.device).type:
      raise ValueError("Training resume requires the same RNG backend")
    for name in (
      "actor",
      "world",
      "critic",
      "world_optimizer",
      "actor_optimizer",
      "critic_optimizer",
      "velocity_encoder",
      "velocity_critic",
      "velocity_optimizer",
      "velocity_critic_optimizer",
    ):
      getattr(self, name).load_state_dict(saved[name])
    self.replay = ReplayBuffer.from_state_dict(saved["replay"], self.device)
    self.velocity_replay = ReplayBuffer.from_state_dict(
      saved["velocity_replay"], self.device
    )
    self.env.sampler.load_state_dict(saved["sampler"])
    for name in (
      "iteration",
      "environment_steps",
      "world_updates",
      "actor_updates",
      "critic_updates",
      "last_collection",
      "initialized",
      "warmup_completed",
      "velocity_updates",
      "velocity_critic_updates",
    ):
      setattr(self, name, saved[name])
    self.env.control_steps = saved["control_steps"]
    self.env.reference_only = self.training_stage != "joint"
    self.env.generator.set_state(saved["env_rng"].cpu())
    self.env.episode_rng.set_state(saved["episode_rng"].cpu())
    self.env.reset()
    self.env.randomize_initial_episode_progress()
    for ids, buffer in (
      (self.env.motion_ids, self.replay),
      (self.env.velocity_ids, self.velocity_replay),
    ):
      count = min(buffer.steps, buffer.capacity)
      self.env.episode_id[ids] = buffer.data["episode_id"][:count].max(0).values + 1
    for key, value in saved["command_state"].items():
      self.env.command_state[key].copy_(value)
    self.env.command_rng.set_state(saved["command_rng"].cpu())
    restore_rng(saved["rng"])
    for name in ("noise_rng", "imagination_rng", "critic_rng"):
      getattr(self, name).set_state(saved[name].cpu())
