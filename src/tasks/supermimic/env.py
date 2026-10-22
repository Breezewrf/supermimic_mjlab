"""mjlab collection environment with synchronized, pre-reset transitions."""

import math

import torch
from mjlab.envs import ManagerBasedRlEnv
from torch import Tensor

from .config.task_cfg import TRACKED_BODIES, FoundationConfig
from .config.robot_cfg import FOOT_COLLISION_EXPR
from .data.motion_library import MotionLibrary
from .env_cfg import make_env_cfg
from .mdp.reset_sampler import ResetSampler
from .mdp.resets import reference_reset_state
from .mdp.rewards import tracking_reward
from .mdp.state import ObservationHistory, add_sensor_noise, observe
from .mdp.terminations import failure_terms


class SuperMimicEnv(ManagerBasedRlEnv):
  def __init__(
    self, config: FoundationConfig, device: str = "cuda:0", render_mode=None
  ):
    self.foundation_config = config
    super().__init__(make_env_cfg(config), device=device, render_mode=render_mode)
    self.robot = self.scene["robot"]
    self.library = MotionLibrary(
      config.manifest,
      tuple(self.robot.joint_names),
      TRACKED_BODIES,
      device=device,
      clips=config.clips,
    )
    if tuple(self.action_manager.get_term("joint_pos").target_names) != tuple(
      self.robot.joint_names
    ):
      raise ValueError("PD action order must match G1 joint/state order")
    self.body_ids = torch.tensor(
      [self.robot.body_names.index(b) for b in TRACKED_BODIES], device=device
    )
    foot_ids, _ = self.robot.find_geoms(FOOT_COLLISION_EXPR)
    if not foot_ids:
      raise ValueError("G1 foot collision geometry not found")
    self.foot_geom_id = int(self.robot.indexing.geom_ids[foot_ids[0]])
    self.torso_id = int(
      self.robot.indexing.body_ids[self.robot.body_names.index("torso_link")]
    )
    self.sampler = ResetSampler(
      self.library.lengths, mode=config.sampling, seed=config.seed
    )
    self.generator = torch.Generator(device=device).manual_seed(config.seed)
    self.motion_id = torch.zeros(self.num_envs, dtype=torch.long, device=device)
    self.frame_id = torch.zeros_like(self.motion_id)
    self.episode_id = torch.full_like(self.motion_id, -1)
    self.previous_action = torch.zeros(self.num_envs, 29, device=device)
    self.context = torch.zeros(self.num_envs, 33, device=device)
    self.history = ObservationHistory(self.num_envs, config.history_length, device)
    self.clean_history = ObservationHistory(
      self.num_envs, config.history_length, device
    )
    self.current_observation = torch.zeros(self.num_envs, 93, device=device)
    self.clean_observation = torch.zeros_like(self.current_observation)
    self._episode_rewards = (
      torch.zeros(
        math.ceil(config.episode_length_s / self.step_dt), self.num_envs, device=device
      )
      if config.sampling == "return"
      else None
    )
    self.reset()

  def read_state(self) -> Tensor:
    data = self.robot.data
    return torch.cat(
      (
        data.root_link_pos_w - self.scene.env_origins,
        data.root_link_quat_w,
        data.joint_pos,
        data.root_link_lin_vel_w,
        data.root_link_ang_vel_w,
        data.joint_vel,
      ),
      dim=-1,
    )

  def check_simulation_capacity(self) -> None:
    data, storage = self.sim.data, self.sim.wp_data
    overflow = (
      (data.ncollision > storage.naconmax)
      | (data.nacon > storage.naconmax)
      | (data.nefc > storage.njmax).any()
    )
    if bool(overflow.any()):
      raise RuntimeError(
        "MuJoCo Warp collision/constraint capacity exceeded; increase nconmax/njmax before collecting data"
      )

  def read_bodies(self) -> dict[str, Tensor]:
    data = self.robot.data
    return {
      "body_pos_w": data.body_link_pos_w[:, self.body_ids]
      - self.scene.env_origins[:, None],
      "body_quat_w": data.body_link_quat_w[:, self.body_ids],
      "body_lin_vel_w": data.body_link_lin_vel_w[:, self.body_ids],
      "body_ang_vel_w": data.body_link_ang_vel_w[:, self.body_ids],
    }

  def _observations(self, state: Tensor) -> tuple[Tensor, Tensor]:
    clean = observe(
      state, self.previous_action, self.robot.data.default_joint_pos, self.context
    )
    noisy = (
      add_sensor_noise(clean, self.generator)
      if self.foundation_config.sensor_noise
      else clean.clone()
    )
    return clean, noisy

  def policy_input(self) -> dict:
    return {
      "history": self.history.data.clone(),
      "future_reference": self.library.get_future(
        self.motion_id, self.frame_id, self.foundation_config.future_length
      ),
    }

  def replay_frame(self, motion_id: int, frame_id: int) -> dict[str, Tensor]:
    """Kinematic preview only: write a reference pose, without physics stepping."""
    motions = torch.full_like(self.motion_id, motion_id)
    frames = torch.full_like(self.frame_id, frame_id)
    reference = self.library.get_frame(motions, frames)
    root = self.library.root_state(motions, frames)
    root[:, :3] += self.scene.env_origins
    self.robot.write_joint_state_to_sim(reference["joint_pos"], reference["joint_vel"])
    self.robot.write_root_state_to_sim(root)
    self.scene.write_data_to_sim()
    self.sim.forward()
    return self.read_bodies()

  def reset(self, *, seed=None, env_ids=None, options=None):
    """Optional options={'motion_id': tensor, 'frame_id': tensor} fixes starts."""
    if seed is not None:
      self.seed(seed)
      self.generator.manual_seed(seed)
      self.sampler.generator.manual_seed(seed)
    if env_ids is None:
      env_ids = torch.arange(self.num_envs, device=self.device)
    else:
      env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
    options = options or {}
    if "motion_id" in options:
      motion = torch.as_tensor(
        options["motion_id"], dtype=torch.long, device=self.device
      ).expand(len(env_ids))
      frame = torch.as_tensor(
        options.get("frame_id", 0), dtype=torch.long, device=self.device
      ).expand(len(env_ids))
      self.library.get_frame(motion, frame)  # validate before changing simulation
      if (frame >= self.library.lengths[motion] - 1).any():
        raise ValueError("Reset needs a reference frame with a successor")
    else:
      motion, frame = self.sampler.sample(len(env_ids))
    super()._reset_idx(env_ids)
    self.motion_id[env_ids], self.frame_id[env_ids] = motion, frame
    self.episode_id[env_ids] += 1
    self.previous_action[env_ids] = 0
    reference = self.library.get_frame(motion, frame)
    root = self.library.root_state(motion, frame)
    root, joint_pos, joint_vel = reference_reset_state(
      root,
      reference["joint_pos"],
      reference["joint_vel"],
      self.robot.data.joint_pos_limits[env_ids],
      perturb=self.foundation_config.reset_noise,
      generator=self.generator,
    )
    root[:, :3] += self.scene.env_origins[env_ids]
    self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    self.robot.write_root_state_to_sim(root, env_ids=env_ids)
    self.robot.clear_state(env_ids)
    self.robot.set_joint_position_target(joint_pos, env_ids=env_ids)
    model = self.sim.model
    self.context[env_ids, 0] = model.geom_friction[env_ids, self.foot_geom_id, 0]
    self.context[env_ids, 1:30] = self.robot.data.encoder_bias[env_ids]
    original_com = torch.as_tensor(
      self.sim.mj_model.body_ipos[self.torso_id].copy(),
      dtype=self.context.dtype,
      device=self.device,
    )
    self.context[env_ids, 30:33] = (
      model.body_ipos[env_ids, self.torso_id] - original_com
    )
    self.scene.write_data_to_sim()
    self.sim.forward()
    self.sim.sense()
    self.check_simulation_capacity()
    clean, noisy = self._observations(self.read_state())
    self.clean_observation[env_ids], self.current_observation[env_ids] = (
      clean[env_ids],
      noisy[env_ids],
    )
    self.history.reset(env_ids, noisy[env_ids])
    self.clean_history.reset(env_ids, clean[env_ids])
    self.obs_buf = self.policy_input()
    return self.obs_buf, self.extras

  @torch.no_grad()
  def step(self, action: Tensor):
    if action.shape != (self.num_envs, 29) or not torch.isfinite(action).all():
      raise ValueError("Expected finite (num_envs, 29) normalized actions")
    action = action.to(self.device).clone()
    if self.foundation_config.action_clip is not None:
      action.clamp_(
        -self.foundation_config.action_clip, self.foundation_config.action_clip
      )
    transition = {
      "state": self.read_state().clone(),
      "action": action.clone(),
      "previous_action": self.previous_action.clone(),
      "observation": self.current_observation.clone(),
      "clean_observation": self.clean_observation.clone(),
      "context": self.context.clone(),
      "motion_id": self.motion_id.clone(),
      "frame_id": self.frame_id.clone(),
      "episode_id": self.episode_id.clone(),
      "episode_step": self.episode_length_buf.clone(),
    }
    self.action_manager.process_action(action)
    for _ in range(self.cfg.decimation):
      self._sim_step_counter += 1
      self.action_manager.apply_action()
      self.scene.write_data_to_sim()
      self.sim.step()
      self.check_simulation_capacity()
      self.scene.update(dt=self.physics_dt)
    # Refresh before reading any derived state or calculating task signals.
    self.sim.forward()
    self.sim.sense()
    self.check_simulation_capacity()
    next_state = self.read_state()
    next_frame = (self.frame_id + 1).minimum(self.library.lengths[self.motion_id] - 1)
    reference = self.library.get_frame(self.motion_id, next_frame)
    reward, metrics = tracking_reward(
      next_state,
      self.read_bodies(),
      reference,
      action,
      self.previous_action,
      self.robot.data.soft_joint_pos_limits,
    )
    failure = failure_terms(next_state, reference)
    terminated = failure["height_failure"] | failure["tilt_failure"]
    self.episode_length_buf += 1
    self.common_step_counter += 1
    time_out = self.episode_length_buf >= math.ceil(
      self.foundation_config.episode_length_s / self.step_dt
    )
    motion_end = next_frame == self.library.lengths[self.motion_id] - 1
    done = terminated | time_out | motion_end
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
        "motion_end": motion_end.clone(),
      }
    )
    self.clean_observation.copy_(clean)
    self.current_observation.copy_(noisy)
    self.history.append(noisy)
    self.clean_history.append(clean)
    if self._episode_rewards is not None:
      ids = torch.arange(self.num_envs, device=self.device)
      self._episode_rewards[transition["episode_step"], ids] = reward
      for env_id in done.nonzero().flatten().tolist():
        count = int(self.episode_length_buf[env_id])
        frames = torch.arange(
          int(self.frame_id[env_id]) - count + 1,
          int(self.frame_id[env_id]) + 1,
          device=self.device,
        )
        self.sampler.update_episode(
          int(self.motion_id[env_id]), frames, self._episode_rewards[:count, env_id]
        )
    self.frame_id.copy_(next_frame)
    self.reset_buf = done.clone()
    self.reset_terminated = terminated.clone()
    self.reset_time_outs = (time_out | motion_end).clone()
    reset_ids = done.nonzero(as_tuple=False).flatten()
    if reset_ids.numel():
      self.reset(env_ids=reset_ids)
    self.reward_buf = reward
    self.obs_buf = self.policy_input()
    # Base reset mutates manager buffers: return owned task flags and transition.
    self.extras = {
      "transition": transition,
      "metrics": {k: v.clone() for k, v in (metrics | failure).items()},
    }
    return self.obs_buf, reward, terminated, time_out | motion_end, self.extras
