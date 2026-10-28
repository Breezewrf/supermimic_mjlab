"""Standing-start velocity step responses, including zero-command drift."""

import math

import torch

from ..mdp.velocity import measured_velocity


def step_commands(full=False):
  if not full:
    return [
      ("basic", command)
      for command in (
        (0.0, 0.0, 0.0),
        (0.5, 0.0, 0.0),
        (-0.5, 0.0, 0.0),
        (0.0, 0.5, 0.0),
        (0.0, -0.5, 0.0),
        (0.0, 0.0, 1.0),
        (0.0, 0.0, -1.0),
        (0.5, 0.0, 0.5),
        (0.5, 0.5, -0.5),
      )
    ]
  # Include zero exactly despite asymmetric forward/backward limits. Each plane
  # has 15*15 - 1 nonzero points; repeated axes remain distinct plane tests.
  ranges = ((-1.5, 2.0), (-1.32, 1.32), (-3.35, 3.35))
  axes = [
    torch.cat((torch.linspace(low, 0.0, 8)[:-1], torch.linspace(0.0, high, 8))).tolist()
    for low, high in ranges
  ]
  result = []
  for name, left, right in (("vx_vy", 0, 1), ("vx_wz", 0, 2), ("vy_wz", 1, 2)):
    for x in axes[left]:
      for y in axes[right]:
        if x == 0 and y == 0:
          continue
        command = [0.0, 0.0, 0.0]
        command[left], command[right] = x, y
        result.append((name, tuple(command)))
  return result


@torch.no_grad()
def evaluate_velocity(
  env, actor, *, full=False, trials=4, duration=5.0, steady_seconds=2.0
):
  if trials < 1 or not 0 < steady_seconds <= duration:
    raise ValueError("Invalid step-response duration/trials")
  if env.library is not None or env.motion_count:
    raise ValueError("Velocity evaluation must be independent of a motion library")
  if (
    env.foundation_config.randomization
    or env.foundation_config.sensor_noise
    or env.foundation_config.reset_noise
  ):
    raise ValueError("Step tests disable training randomization/noise")
  commands = step_commands(full)
  # Standing is evaluated separately from the 672 nonzero grid conditions.
  if full:
    commands.append(("standing", (0.0, 0.0, 0.0)))
  tasks = [
    (index, trial, cmd)
    for index, (_, cmd) in enumerate(commands)
    for trial in range(trials)
  ]
  rows = [
    {"plane": plane, "command": list(command), "trials": []}
    for plane, command in commands
  ]
  steps = math.ceil(duration / env.step_dt)
  tail = min(steps, math.ceil(steady_seconds / env.step_dt))
  for start in range(0, len(tasks), env.num_envs):
    selected = tasks[start : start + env.num_envs]
    cmd = torch.zeros(env.num_envs, 3, device=env.device)
    cmd[: len(selected)] = cmd.new_tensor([task[2] for task in selected])
    env.set_command(cmd)
    env.reset(options={"standing": True})
    initial = env.read_state().clone()
    fallen = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    error_sum = torch.zeros_like(cmd)
    change = torch.zeros(env.num_envs, device=env.device)
    for step in range(steps):
      action = actor(env.history.data, cmd)
      _, _, _, _, extras = env.step(action)
      transition = extras["velocity_transition"]
      fallen |= extras["metrics"]["velocity/physical_failure"]
      if step >= steps - tail:
        error_sum += measured_velocity(transition["next_state"]) - cmd
      change += (transition["action"] - transition["previous_action"]).square().sum(-1)
    mean_error = error_sum / tail
    final_state = extras["velocity_transition"]["next_state"]
    for index, (command_index, trial, _) in enumerate(selected):
      fail = bool(fallen[index])
      norm = float(cmd[index].norm())
      error = mean_error[index]
      rows[command_index]["trials"].append(
        {
          "trial": trial,
          "fell": fail,
          "mean_velocity_error": None if fail else error.tolist(),
          "relative_steady_error": None
          if fail or norm == 0
          else float(error.norm()) / norm,
          "linear_steady_error": None if fail else float(error[:2].norm()),
          "yaw_steady_error": None if fail else float(error[2].abs()),
          "xy_displacement_m": None
          if fail
          else float((final_state[index, :2] - initial[index, :2]).norm()),
          "mean_action_change_squared": None if fail else float(change[index]) / steps,
        }
      )
  for row in rows:
    successful = [trial for trial in row["trials"] if not trial["fell"]]
    row["fall_rate"] = 1 - len(successful) / trials
    relative = [
      trial["relative_steady_error"]
      for trial in successful
      if trial["relative_steady_error"] is not None
    ]
    row["median_relative_steady_error"] = (
      float(torch.tensor(relative).quantile(0.5)) if relative else None
    )
  return {
    "mode": "full_672_plus_standing" if full else "basic",
    "duration_s": duration,
    "steady_window_s": steady_seconds,
    "trials_per_command": trials,
    "trial_variation": "Identical deterministic standing starts; repeats measure reproducibility, not robustness",
    "relative_error_definition": "norm(mean([heading_vx_error, heading_vy_error, world_wz_error])) / norm(command)",
    "fall_rate": sum(row["fall_rate"] for row in rows) / len(rows),
    "commands": rows,
  }
