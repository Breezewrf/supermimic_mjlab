"""Deterministic full-clip evaluation with a small policy callback interface."""

import csv
import json
from collections.abc import Callable
from pathlib import Path

import torch
from torch import Tensor

Policy = Callable[[Tensor, dict[str, Tensor]], Tensor]
FrameCallback = Callable[[object, int, int, Tensor, float], None]


class EvaluationInterrupted(Exception):
  """A display requested stopping before all motions could be evaluated."""


def save_report(report: dict, output: Path, csv_path: Path | None = None):
  """Atomically write the full report and optionally append a comparison row."""
  output.parent.mkdir(parents=True, exist_ok=True)
  temporary = output.with_suffix(output.suffix + ".tmp")
  temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
  temporary.replace(output)
  if csv_path:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    columns = (
      "evaluated_at",
      "policy",
      "iteration",
      "seed",
      "normalized_return",
      "normalized_return_framew",
      "successes",
      "n_failed",
      "mean_completion",
      "evaluation_complete",
      "clips",
      "failed",
    )
    header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="") as stream:
      writer = csv.DictWriter(stream, fieldnames=columns)
      if header:
        writer.writeheader()
      row = {key: report.get(key) for key in columns}
      row["clips"] = json.dumps([motion["motion"] for motion in report["motions"]])
      row["failed"] = json.dumps(report["failed"])
      writer.writerow(row)


def evaluation_summary(report: dict) -> str:
  """Use the same score label for training, headless evaluation and playback."""
  rows = report["motions"]
  failures = sum(row.get("terminated", False) for row in rows)
  limited = any(row["evaluation_limited"] for row in rows)
  incomplete = limited or any(row.get("failure_reason") == "time_limit" for row in rows)
  return (
    f"Evaluation return {report['normalized_return']:.3f}"
    f" | success {report['successes']}/{len(rows)}"
    f" | failed {failures}/{len(rows)}"
    f" | completion {sum(row['completion'] for row in rows) / len(rows):.1%}"
    + (" | INCOMPLETE (debug limit or timeout)" if incomplete else "")
  )


def zero_policy(history: Tensor, future_reference: dict[str, Tensor]) -> Tensor:
  del future_reference
  return history.new_zeros((len(history), 29))


@torch.no_grad()
def evaluate(
  env,
  policy: Policy = zero_policy,
  max_steps_per_clip: int | None = None,
  *,
  on_frame: FrameCallback | None = None,
  on_motion: Callable[[dict], None] | None = None,
) -> dict:
  if env.num_envs != 1:
    raise ValueError("Fixed-clip evaluation requires exactly one environment")
  cfg = env.foundation_config
  if cfg.randomization or cfg.sensor_noise or cfg.reset_noise:
    raise ValueError("Evaluation must disable randomization, reset and sensor noise")
  if max_steps_per_clip is not None and max_steps_per_clip < 1:
    raise ValueError("Evaluation step limit must be positive")
  if not len(env.library.names):
    raise ValueError("Evaluation requires at least one motion")
  rows = []
  for clip, name in enumerate(env.library.names):
    inputs, _ = env.reset(seed=cfg.seed, options={"motion_id": clip, "frame_id": 0})
    expected = int(env.library.lengths[clip]) - 1
    if expected < 1:
      raise ValueError(f"Motion {name} needs at least two frames")
    horizon = min(expected, max_steps_per_clip) if max_steps_per_clip else expected
    reward_sum, error_sums, reason = 0.0, {}, None
    terminated, truncated = False, False
    if on_frame:
      on_frame(env, clip, 0, env.read_state(), reward_sum)
    for step in range(horizon):
      action = policy(inputs["history"], inputs["future_reference"])
      inputs, reward, terminal, cutoff, extras = env.step(action)
      if not torch.isfinite(reward).all():
        raise FloatingPointError(f"Nonfinite evaluation reward in {name}")
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
        reason = "+".join(reasons) or "physical_failure"
      elif bool(extras["transition"]["time_out"][0]):
        reason = "time_limit"
      # step() automatically resets finished environments. Always display the
      # pre-reset successor, including the final falling/completion pose.
      if on_frame:
        on_frame(env, clip, step + 1, extras["transition"]["next_state"], reward_sum)
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
        "terminated": terminated,
        "failure_reason": reason,
        "evaluation_limited": limited,
        "return": reward_sum,
        "normalized_return": reward_sum / (5 * env.step_dt * expected),
        "mean_errors": {
          key: value / completed_steps for key, value in error_sums.items()
        },
      }
    )
    if on_motion:
      on_motion(rows[-1])
  failed = [row["motion"] for row in rows if row["terminated"]]
  successes = sum(row["success"] for row in rows)
  complete = not any(
    row["evaluation_limited"] or row["failure_reason"] == "time_limit" for row in rows
  )
  return {
    "seed": cfg.seed,
    "motions": rows,
    "successes": successes,
    "success_rate": successes / len(rows),
    "n_failed": len(failed),
    "failed": failed,
    "mean_completion": sum(row["completion"] for row in rows) / len(rows),
    "evaluation_complete": complete,
    "normalized_return": sum(row["normalized_return"] for row in rows) / len(rows),
    "normalized_return_framew": sum(row["return"] for row in rows)
    / (5 * env.step_dt * sum(row["expected_steps"] for row in rows)),
    "protocol": "first frame; full clip unless explicitly limited; deterministic policy; friction 1.0; no reset/sensor/action noise or domain randomization; self collisions disabled",
    "normalization": "per-clip undiscounted reward / (5 * control_dt * (frames - 1)); arithmetic mean over clips; no reward after failure",
  }
