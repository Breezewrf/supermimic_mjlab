"""Deterministic full-clip evaluation with a small policy callback interface."""

from collections.abc import Callable

import torch
from torch import Tensor

Policy = Callable[[Tensor, dict[str, Tensor]], Tensor]


def zero_policy(history: Tensor, future_reference: dict[str, Tensor]) -> Tensor:
  del future_reference
  return history.new_zeros((len(history), 29))


@torch.no_grad()
def evaluate(
  env, policy: Policy = zero_policy, max_steps_per_clip: int | None = None
) -> dict:
  if env.num_envs != 1:
    raise ValueError("Fixed-clip evaluation requires exactly one environment")
  cfg = env.foundation_config
  if cfg.randomization or cfg.sensor_noise or cfg.reset_noise:
    raise ValueError("Evaluation must disable randomization, reset and sensor noise")
  if max_steps_per_clip is not None and max_steps_per_clip < 1:
    raise ValueError("Evaluation step limit must be positive")
  rows = []
  for clip, name in enumerate(env.library.names):
    inputs, _ = env.reset(seed=cfg.seed, options={"motion_id": clip, "frame_id": 0})
    expected = int(env.library.lengths[clip]) - 1
    horizon = min(expected, max_steps_per_clip) if max_steps_per_clip else expected
    reward_sum, error_sums, reason = 0.0, {}, None
    terminated, truncated = False, False
    for step in range(horizon):
      action = policy(inputs["history"], inputs["future_reference"])
      inputs, reward, terminal, cutoff, extras = env.step(action)
      reward_sum += float(reward[0])
      for key in (
        "body_position_error",
        "body_orientation_error",
        "root_position_error",
        "root_orientation_error",
      ):
        error_sums[key] = error_sums.get(key, 0.0) + float(extras["metrics"][key][0])
      terminated, truncated = bool(terminal[0]), bool(cutoff[0])
      if terminated:
        reasons = [
          key
          for key in ("height_failure", "tilt_failure")
          if bool(extras["metrics"][key][0])
        ]
        reason = "+".join(reasons)
      elif bool(extras["transition"]["time_out"][0]):
        reason = "time_limit"
      if terminated or truncated:
        break
    completed_steps = step + 1
    limited = completed_steps < expected and not (terminated or truncated)
    if limited:
      reason = "evaluation_limit"
    rows.append(
      {
        "motion": name,
        "steps": completed_steps,
        "expected_steps": expected,
        "completion": completed_steps / expected,
        "success": completed_steps == expected and not terminated,
        "failure_reason": reason,
        "evaluation_limited": limited,
        "return": reward_sum,
        "normalized_return": reward_sum / (5 * env.step_dt * expected),
        "mean_errors": {
          key: value / completed_steps for key, value in error_sums.items()
        },
      }
    )
  return {
    "seed": cfg.seed,
    "motions": rows,
    "successes": sum(row["success"] for row in rows),
    "normalized_return": sum(row["normalized_return"] for row in rows) / len(rows),
    "protocol": "first frame; full clip unless explicitly limited; no reset/sensor noise or domain randomization; self collisions disabled",
  }
