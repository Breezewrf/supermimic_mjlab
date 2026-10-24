"""World-model playback: future policy previews and same-time dynamics comparisons."""

import math

import torch

from ..models.execution import run_mlp
from ..models.world_model import state_errors

ERROR_NAMES = (
  "root_position_m",
  "root_orientation_rad",
  "joint_position_rmse_rad",
  "body_position_mpjpe_m",
)


class WorldModelPlayback:
  """Read-only observer: reset to physics every H steps and at each clip start.

  Actions come from the real policy, not a second policy in imagination. Only the
  initial state is corrected within each prediction window. Frame callbacks use
  the pre-reset physical successor, so final-frame errors compare matching times.
  """

  def __init__(self, env, model, horizon=32):
    if env.num_envs != 1 or horizon < 1:
      raise ValueError(
        "World-model playback requires one environment and a positive horizon"
      )
    self.env, self.model, self.horizon = env, model, horizon
    self.begin_evaluation()

  def begin_evaluation(self):
    self.state = self.actual = self.context = None
    self.age = 0
    self.metrics = None
    self.diverged = False
    self.clip_stats = {}

  @torch.no_grad()
  def predict(self, action):
    if self.actual is None:
      raise RuntimeError(
        "World-model playback must receive the initial frame before an action"
      )
    if self.age >= self.horizon:
      self.state = self.actual.clone()
      self.context = self.env.context.detach().clone()
      self.age = 0
      self.diverged = False
    self.age += 1
    if self.diverged:
      return
    executed = action.detach()
    clip = self.env.foundation_config.action_clip
    if clip is not None:
      executed = executed.clamp(-clip, clip)
    predicted = self.model(self.state, executed, self.context)
    self.diverged = not bool(torch.isfinite(predicted).all())
    self.state = None if self.diverged else predicted.detach()

  def policy(self, actor):
    def observed(history, reference):
      action = actor(history, reference)
      self.predict(action)
      return action

    return observed

  @torch.no_grad()
  def on_frame(self, env, clip, frame, state, reward_sum):
    del reward_sum
    self.actual = state.detach().clone()
    if frame == 0:
      self.state = self.actual.clone()
      self.context = env.context.detach().clone()
      self.age = 0
      self.diverged = False
      self.metrics = None
      self.clip_stats[clip] = {
        "motion": env.library.names[clip],
        "samples": 0,
        "nonfinite_steps": 0,
        "sums": [0.0] * len(ERROR_NAMES),
      }
      return
    stats = self.clip_stats[clip]
    if self.diverged:
      stats["nonfinite_steps"] += 1
      self.metrics = None
      return
    errors = state_errors(self.state, self.actual)
    predicted_bodies = self.model.kinematics(self.state)["body_pos_w"]
    actual_bodies = self.model.kinematics(self.actual)["body_pos_w"]
    values = (
      torch.stack(
        (
          errors[0].norm(dim=-1).mean(),
          errors[1].norm(dim=-1).mean(),
          errors[2].square().mean().sqrt(),
          (predicted_bodies - actual_bodies).norm(dim=-1).mean(),
        )
      )
      .cpu()
      .tolist()
    )
    if not all(math.isfinite(value) for value in values):
      self.diverged, self.state, self.metrics = True, None, None
      stats["nonfinite_steps"] += 1
      return
    self.metrics = dict(zip(ERROR_NAMES, values))
    stats["samples"] += 1
    stats["sums"] = [a + b for a, b in zip(stats["sums"], values)]

  def status(self):
    if self.diverged:
      return f"DIVERGED at {self.age}/{self.horizon}; waiting for resync"
    if self.metrics is None:
      return f"synced to physics; horizon {self.horizon} steps"
    return (
      f"{self.age}/{self.horizon} steps | root {self.metrics['root_position_m']:.3f} m"
      f" | joints {self.metrics['joint_position_rmse_rad']:.3f} rad"
      f" | bodies {self.metrics['body_position_mpjpe_m']:.3f} m"
    )

  def description(self):
    return f"same-time comparison; actual actions; resync every {self.horizon} steps"

  def report(self):
    rows = []
    for stats in self.clip_stats.values():
      count = stats["samples"]
      rows.append(
        {
          "motion": stats["motion"],
          "samples": count,
          "nonfinite_steps": stats["nonfinite_steps"],
          "mean_errors": {
            key: value / count if count else None
            for key, value in zip(ERROR_NAMES, stats["sums"])
          },
        }
      )
    return {
      "mode": "compare",
      "horizon": self.horizon,
      "control_dt": self.env.step_dt,
      "actions": "executed simulator actions; not a separate imagined policy",
      "resync": "clip start and every horizon steps; no intermediate state correction",
      "external_forces_modeled": False,
      "motions": rows,
    }


class WorldModelFuturePreview(WorldModelPlayback):
  """Receding-horizon policy rollouts, displayed H steps ahead of real physics."""

  def __init__(self, env, model, actor, horizon=32, refresh=4):
    if refresh < 1:
      raise ValueError("Preview refresh interval must be positive")
    self.actor, self.refresh = actor, refresh
    super().__init__(env, model, horizon)

  def begin_evaluation(self):
    super().begin_evaluation()
    self.history = self.previous_action = self.trajectory = None
    self.cache = None
    self.anchor = self.target_frame = self.frame = 0
    self.pending = {}

  def policy(self, actor):
    def observed(history, reference):
      action = actor(history, reference)
      self.previous_action = self._executed(action).detach().clone()
      return action

    return observed

  def _executed(self, action):
    limit = self.env.foundation_config.action_clip
    return action if limit is None else action.clamp(-limit, limit)

  @torch.no_grad()
  def _rollout(self, clip, frame, steps):
    motion = torch.tensor([clip], device=self.env.device, dtype=torch.long)
    frames = torch.tensor([frame], device=self.env.device, dtype=torch.long)
    future = self.env.library.get_future(motion, frames, steps + 10)
    windows = {
      key: value.unfold(1, 10, 1).movedim(-1, 2) for key, value in future.items()
    }
    goals = self.actor.reference_features(windows, self.actor.default_q)
    # References do not depend on imagined states: encode the whole window once.
    latents = run_mlp(self.actor.encoder, goals, self.actor.amp).tanh()
    state, history = self.actual.clone(), self.history.clone()
    states = [state]
    for step in range(steps):
      action = self._executed(
        run_mlp(
          self.actor.decoder,
          torch.cat(
            (latents[:, step], (history / self.actor.observation_scale).flatten(-2)), -1
          ),
          self.actor.amp,
        )
      )
      state = self.model(state, action, self.context)
      observation = self.model.kinematics.observation(
        state,
        action,
        self.context,
        future["body_quat_w"][:, step + 1],
        relative_orientation=self.actor.relative_orientation,
        motion_anchor_orientation=self.actor.motion_anchor_ori_b,
      )
      history = torch.cat((history[:, 1:], observation[:, None]), 1)
      states.append(state)
    return torch.stack(states)

  @torch.no_grad()
  def on_frame(self, env, clip, frame, state, reward_sum):
    if frame == 0:
      super().on_frame(env, clip, frame, state, reward_sum)
      self.history = env.policy_input()["history"].detach().clone()
      self.cache, self.pending = None, {}
      self.clip_stats[clip]["forecast_updates"] = 0
      self.clip_stats[clip]["nonfinite_forecast_updates"] = 0
    else:
      self.actual = state.detach().clone()
      motion = torch.tensor([clip], device=env.device, dtype=torch.long)
      frames = torch.tensor([frame], device=env.device, dtype=torch.long)
      reference = env.library.get_frame(motion, frames)
      observation = self.model.kinematics.observation(
        self.actual,
        self.previous_action,
        self.context,
        reference["body_quat_w"],
        relative_orientation=self.actor.relative_orientation,
        motion_anchor_orientation=self.actor.motion_anchor_ori_b,
      )
      self.history = torch.cat((self.history[:, 1:], observation[:, None]), 1)
      # Only compare a forecast when the matching physical timestamp arrives.
      self.state = self.pending.pop(frame, None)
      self.metrics, self.diverged = None, False
      if self.state is not None:
        super().on_frame(env, clip, frame, state, reward_sum)
    self.frame = frame
    remaining = int(env.library.lengths[clip]) - 1 - frame
    self.target_frame = frame + min(self.horizon, remaining)
    if remaining == 0:
      self.state, self.trajectory = None, None
      self.age = 0
      return
    if self.cache is None or frame - self.anchor >= self.refresh:
      # Cache enough states to keep a fixed lead while advancing between refreshes.
      self.anchor = frame
      self.cache = self._rollout(
        clip, frame, min(self.horizon + self.refresh - 1, remaining)
      )
      self.clip_stats[clip]["forecast_updates"] += 1
      if not bool(torch.isfinite(self.cache).all()):
        self.clip_stats[clip]["nonfinite_forecast_updates"] += 1
    self.age = self.target_frame - frame
    self.state = self.cache[self.target_frame - self.anchor]
    self.trajectory = self.cache[
      frame - self.anchor : self.target_frame - self.anchor + 1
    ]
    self.diverged = not bool(torch.isfinite(self.trajectory).all())
    if self.diverged:
      self.state, self.trajectory = None, None
    else:
      # Keep the first (longest-lead) forecast if several previews reach clip end.
      self.pending.setdefault(self.target_frame, self.state.detach().clone())

  def description(self):
    return (
      f"future preview +{self.horizon * self.env.step_dt:.2f}s; "
      f"policy in world model; refresh every {self.refresh} steps"
    )

  def status(self):
    if self.diverged:
      return "FUTURE prediction diverged; retrying at next refresh"
    if self.age == 0:
      return "end of clip; no future frames"
    result = f"FUTURE +{self.age * self.env.step_dt:.2f}s ({self.age} steps)"
    if self.metrics is not None:
      result += f" | verified root error {self.metrics['root_position_m']:.3f} m"
    return result

  def report(self):
    result = super().report()
    result.update(
      mode="future",
      refresh_steps=self.refresh,
      max_rollout_steps=self.horizon + self.refresh - 1,
      actions="policy evaluated on imagined observations and future references",
      resync="new rollout from current real state/history at each refresh",
      error_alignment="displayed forecast vs physical state at the same target frame; longest lead at clip end",
    )
    for row, stats in zip(result["motions"], self.clip_stats.values()):
      for key in ("forecast_updates", "nonfinite_forecast_updates"):
        row[key] = stats[key]
    return result
