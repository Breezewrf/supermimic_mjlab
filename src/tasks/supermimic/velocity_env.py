"""Mixed motion/velocity collection, or standalone reference-free simulation."""

import math

import torch
from mjlab.envs import ManagerBasedRlEnv

from .config.task_cfg import FoundationConfig
from .config.velocity_cfg import VelocityConfig
from .env import SuperMimicEnv
from .mdp.resets import reference_reset_state
from .mdp.rewards import tracking_reward
from .mdp.terminations import failure_terms
from .mdp.velocity import velocity_failure, velocity_reward
from .mdp.velocity_commands import VelocityCommands


class VelocityEnv(SuperMimicEnv):
  def __init__(
    self,
    config: FoundationConfig,
    velocity: VelocityConfig,
    device="cuda:0",
    *,
    motion_envs=0,
    training=False,
  ):
    if (
      config.observation_dim != 93
      or config.history_length != 10
      or config.future_length != 10
    ):
      raise ValueError("Velocity requires ten 93-dimensional observations")
    if not 0 <= motion_envs < config.num_envs or (motion_envs and not training):
      raise ValueError("Invalid mixed environment allocation")
    self.velocity_config = velocity
    self.motion_count, self.training_tasks = motion_envs, training
    self.reference_only = False
    self.command_override = None
    super().__init__(config, device)

  def _load_library(self, config, device):
    return super()._load_library(config, device) if self.training_tasks else None

  def _initialize_task(self):
    self.motion_ids = torch.arange(self.motion_count, device=self.device)
    self.velocity_ids = torch.arange(
      self.motion_count, self.num_envs, device=self.device
    )
    self.commands = VelocityCommands(self.velocity_config, self.library)
    self.command_state = self.commands.empty(len(self.velocity_ids), self.device)
    self.command_rng = torch.Generator(device=self.device).manual_seed(
      self.foundation_config.seed + 101
    )
    # Counts actual collection steps, unaffected by episode resets.
    self.control_steps = 0

  def set_command(self, command):
    """Persistent external command for reference-free evaluation/playback."""
    if self.training_tasks:
      raise ValueError("External commands are only accepted in inference mode")
    command = torch.as_tensor(command, device=self.device, dtype=torch.float32)
    command = command.expand(len(self.velocity_ids), 3).clone()
    if not torch.isfinite(command).all():
      raise ValueError("Command must be finite")
    self.command_override = command
    self.command_state["command"].copy_(command)

  def policy_input(self):
    result = {
      "history": self.history.data.clone(),
      "command": self.command_state["command"].clone(),
    }
    if self.motion_count:
      result["future_reference"] = self.library.get_future(
        self.motion_id[self.motion_ids],
        self.frame_id[self.motion_ids],
        10,
      )
    return result

  def reset(self, *, seed=None, env_ids=None, options=None):
    if seed is not None:
      self.seed(seed)
      self.generator.manual_seed(seed)
      self.command_rng.manual_seed(seed + 101)
      if self.sampler is not None:
        self.sampler.generator.manual_seed(seed)
    ids = (
      torch.arange(self.num_envs, device=self.device)
      if env_ids is None
      else torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
    )
    options = options or {}
    ManagerBasedRlEnv._reset_idx(self, ids)
    count = len(ids)
    # Default root state is in asset coordinates before environment translation.
    root = self.robot.data.default_root_state[ids].clone()
    q = self.robot.data.default_joint_pos[ids].clone()
    qd = torch.zeros_like(q)
    if self.library is not None:
      motion, frame = self.sampler.sample(count)
      if "motion_id" in options:
        motion[:] = options["motion_id"]
        frame[:] = options.get("frame_id", 0)
      reference = self.library.get_frame(motion, frame)
      use_reference = ids < self.motion_count
      if self.training_tasks and not options.get("standing", False):
        use_reference = use_reference | (
          torch.rand(count, device=self.device, generator=self.generator)
          >= self.velocity_config.standing_reset_probability
        )
      reference_root = self.library.root_state(motion, frame)
      root = torch.where(use_reference[:, None], reference_root, root)
      q = torch.where(use_reference[:, None], reference["joint_pos"], q)
      qd = torch.where(use_reference[:, None], reference["joint_vel"], qd)
      self.motion_id[ids], self.frame_id[ids] = motion, frame
      self.episode_start_frame[ids] = frame
    self.previous_action[ids] = 0
    self.episode_id[ids] += 1
    self.episode_reward_sum[ids] = 0
    self.episode_discounted_return[ids] = 0
    root, q, qd = reference_reset_state(
      root,
      q,
      qd,
      self.robot.data.joint_pos_limits[ids],
      perturb=self.foundation_config.reset_noise,
      generator=self.generator,
    )
    self._apply_reset_state(ids, root, q, qd)
    velocity = ids[ids >= self.motion_count] - self.motion_count
    self.commands.resample(
      self.command_state,
      velocity,
      self.command_rng,
      self.control_steps,
      reference_only=self.reference_only,
    )
    if self.command_override is not None:
      self.command_state["command"].copy_(self.command_override)
    clean, noisy = self._observations(self.read_state())
    self.clean_observation[ids], self.current_observation[ids] = (
      clean[ids],
      noisy[ids],
    )
    self.history.reset(ids, noisy[ids])
    self.clean_history.reset(ids, clean[ids])
    self.obs_buf = self.policy_input()
    return self.obs_buf, self.extras

  @torch.no_grad()
  def step(self, action):
    if action.shape != (self.num_envs, 29) or not torch.isfinite(action).all():
      raise ValueError("Expected finite (num_envs,29) actions")
    action = action.to(self.device).clone()
    if self.foundation_config.action_clip is not None:
      action.clamp_(
        -self.foundation_config.action_clip, self.foundation_config.action_clip
      )
    transition = {
      "state": self.read_state().clone(),
      "action": action,
      "previous_action": self.previous_action.clone(),
      "observation": self.current_observation.clone(),
      "clean_observation": self.clean_observation.clone(),
      "context": self.context.clone(),
      "episode_id": self.episode_id.clone(),
      "episode_step": self.episode_length_buf.clone(),
    }
    velocity_transition = {
      key: value.clone() for key, value in self.command_state.items()
    }
    motion_transition = {
      "motion_id": self.motion_id[self.motion_ids].clone(),
      "frame_id": self.frame_id[self.motion_ids].clone(),
    }
    next_state = self._step_physics(action)
    bodies = self.read_bodies()
    m, v = self.motion_ids, self.velocity_ids
    reward = torch.zeros(self.num_envs, device=self.device)
    terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    motion_end = torch.zeros_like(terminated)
    metrics = {}
    if self.motion_count:
      next_frame = (self.frame_id[m] + 1).minimum(
        self.library.lengths[self.motion_id[m]] - 1
      )
      reference = self.library.get_frame(self.motion_id[m], next_frame)
      reward[m], motion_metrics = tracking_reward(
        next_state[m],
        {k: b[m] for k, b in bodies.items()},
        reference,
        action[m],
        self.previous_action[m],
        self.robot.data.soft_joint_pos_limits[m],
      )
      failure = failure_terms(next_state[m], reference)
      terminated[m] = failure["height_failure"] | failure["tilt_failure"]
      motion_end[m] = next_frame == self.library.lengths[self.motion_id[m]] - 1
      self.frame_id[m] = next_frame
      motion_transition["motion_end"] = motion_end[m].clone()
      metrics.update({"motion/" + k: b for k, b in motion_metrics.items()})
    reward[v], velocity_metrics = velocity_reward(
      next_state[v],
      {k: b[v] for k, b in bodies.items()},
      velocity_transition["command"],
      action[v],
      self.previous_action[v],
      self.robot.data.soft_joint_pos_limits[v],
      self.robot.data.default_joint_pos[v],
      self.velocity_config,
    )
    failure = velocity_failure(next_state[v], self.velocity_config)
    physical_failure = failure["height_failure"] | failure["tilt_failure"]
    terminated[v] = physical_failure
    metrics.update({"velocity/" + k: b for k, b in velocity_metrics.items()})
    metrics["velocity/physical_failure"] = physical_failure
    if not self.foundation_config.terminate_on_failure:
      terminated.zero_()
    self.episode_length_buf += 1
    self.common_step_counter += 1
    self.control_steps += 1
    time_out = self.episode_length_buf >= math.ceil(
      self.foundation_config.episode_length_s / self.step_dt
    )
    if self.command_override is None:
      self.commands.advance(
        self.command_state,
        self.command_rng,
        self.control_steps,
        reference_only=self.reference_only,
      )
    velocity_transition["next_command"] = self.command_state["command"].clone()
    self.previous_action.copy_(action)
    clean, noisy = self._observations(next_state)
    transition.update(
      {
        "next_state": next_state.clone(),
        "next_observation": noisy.clone(),
        "next_clean_observation": clean.clone(),
        "reward": reward.clone(),
        "terminated": terminated.clone(),
        "time_out": time_out.clone(),
      }
    )
    velocity_transition.update({k: b[v].clone() for k, b in transition.items()})
    motion_transition.update({k: b[m].clone() for k, b in transition.items()})
    self.current_observation.copy_(noisy)
    self.clean_observation.copy_(clean)
    self.history.append(noisy)
    self.clean_history.append(clean)
    self.episode_reward_sum += reward
    self.episode_discounted_return += 0.99 ** transition["episode_step"] * reward
    done = terminated | time_out | motion_end
    reset_ids = done.nonzero().flatten()
    if self.sampler is not None and self.foundation_config.sampling == "return":
      reset_motion = reset_ids[reset_ids < self.motion_count]
      if len(reset_motion):
        self.sampler.update_episodes(
          self.motion_id[reset_motion],
          self.episode_start_frame[reset_motion],
          self.episode_discounted_return[reset_motion],
          self.episode_reward_sum[reset_motion],
          self.episode_length_buf[reset_motion],
          terminated[reset_motion],
        )
    self.reset_buf = done.clone()
    self.reset_terminated = terminated.clone()
    self.reset_time_outs = (time_out | motion_end).clone()
    if len(reset_ids):
      self.reset(env_ids=reset_ids)
    self.reward_buf = reward
    self.obs_buf = self.policy_input()
    self.extras = {
      "motion_transition": motion_transition,
      "velocity_transition": velocity_transition,
      "transition": transition,
      "metrics": metrics,
    }
    return self.obs_buf, reward, terminated, time_out | motion_end, self.extras
