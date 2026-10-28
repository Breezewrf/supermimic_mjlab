"""Deterministic full-clip evaluation with a small policy callback interface."""

import csv
import json
from collections.abc import Callable
from pathlib import Path

import torch
from torch import Tensor

from ..data.motion_sources import motion_entries, select_motion_entries

Policy = Callable[[Tensor, dict[str, Tensor]], Tensor]
FrameCallback = Callable[[object, int, int, Tensor, float], None]
ERROR_METRICS = (
  "body_position_error",
  "body_orientation_error",
  "root_position_error",
  "root_orientation_error",
)
FAILURE_METRICS = ("height_failure", "tilt_failure", "physical_failure")


def evaluation_env_count(
  manifest: Path | None, clips: tuple[str, ...] = (), max_envs: int = 64,
  *, motion: Path | None = None, motion_dir: Path | None = None,
) -> int:
  """Size the pool from the selected source, without loading motion arrays."""
  if max_envs < 1:
    raise ValueError("Evaluation environment count must be positive")
  entries = select_motion_entries(
    motion_entries(manifest, motion=motion, motion_dir=motion_dir), clips
  )
  return min(max_envs, len(entries))


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
      "evaluation_mode",
      "mouse_perturbation_steps",
      "termination_enabled",
    )
    header = not csv_path.exists() or csv_path.stat().st_size == 0
    if not header:
      with csv_path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        previous_columns = reader.fieldnames
        if previous_columns != list(columns):
          if not previous_columns or set(previous_columns) - set(columns):
            raise ValueError("Evaluation CSV has incompatible columns")
          # Preserve older score files while adding the interaction provenance.
          temporary_csv = csv_path.with_suffix(csv_path.suffix + ".tmp")
          with temporary_csv.open("w", newline="") as upgraded:
            writer = csv.DictWriter(upgraded, fieldnames=columns)
            writer.writeheader()
            for row in reader:
              row.setdefault("evaluation_mode", "standard")
              row.setdefault("mouse_perturbation_steps", 0)
              row.setdefault("termination_enabled", True)
              writer.writerow(row)
      if previous_columns != list(columns):
        temporary_csv.replace(csv_path)
    with csv_path.open("a", newline="") as stream:
      writer = csv.DictWriter(stream, fieldnames=columns)
      if header:
        writer.writeheader()
      row = {key: report.get(key) for key in columns}
      row["clips"] = json.dumps([motion["motion"] for motion in report["motions"]])
      row["failed"] = json.dumps(report["failed"])
      row["evaluation_mode"] = report.get("evaluation_mode", "standard")
      row["mouse_perturbation_steps"] = report.get("mouse_perturbation_steps", 0)
      row["termination_enabled"] = report.get("termination_enabled", True)
      writer.writerow(row)


def evaluation_summary(report: dict) -> str:
  """Use the same score label for training, headless evaluation and playback."""
  rows = report["motions"]
  failures = sum(
    row.get("failure_detected", row.get("terminated", False)) for row in rows
  )
  limited = any(row["evaluation_limited"] for row in rows)
  incomplete = limited or any(row.get("failure_reason") == "time_limit" for row in rows)
  return (
    f"Evaluation return {report['normalized_return']:.3f}"
    f" | success {report['successes']}/{len(rows)}"
    f" | failed {failures}/{len(rows)}"
    f" | completion {sum(row['completion'] for row in rows) / len(rows):.1%}"
    + (" | INCOMPLETE (debug limit or timeout)" if incomplete else "")
    + (
      " | PLAYBACK (failure termination disabled)"
      if report.get("termination_enabled") is False
      else ""
    )
    + (
      " | INTERACTIVE (mouse forces)"
      if report.get("evaluation_mode") == "interactive"
      else ""
    )
  )


def zero_policy(history: Tensor, future_reference: dict[str, Tensor]) -> Tensor:
  joints = future_reference.get("joint_pos")
  return history.new_zeros((len(history), 29 if joints is None else joints.shape[-1]))


@torch.no_grad()
def evaluate(
  env,
  policy: Policy = zero_policy,
  max_steps_per_clip: int | None = None,
  *,
  on_frame: FrameCallback | None = None,
  on_motion: Callable[[dict], None] | None = None,
  before_step: Callable[[object], None] | None = None,
) -> dict:
  if env.num_envs < 1:
    raise ValueError("Evaluation requires at least one environment")
  if env.num_envs > 1 and (on_frame is not None or before_step is not None):
    raise ValueError("Frame and interaction callbacks require exactly one environment")
  cfg = env.foundation_config
  if cfg.randomization or cfg.sensor_noise or cfg.reset_noise:
    raise ValueError("Evaluation must disable randomization, reset and sensor noise")
  if max_steps_per_clip is not None and max_steps_per_clip < 1:
    raise ValueError("Evaluation step limit must be positive")
  if not len(env.library.names):
    raise ValueError("Evaluation requires at least one motion")
  if env.num_envs > 1:
    return _evaluation_report(
      env, _evaluate_parallel(env, policy, max_steps_per_clip, on_motion)
    )
  rows = []
  for clip, name in enumerate(env.library.names):
    inputs, _ = env.reset(seed=cfg.seed, options={"motion_id": clip, "frame_id": 0})
    expected = int(env.library.lengths[clip]) - 1
    if expected < 1:
      raise ValueError(f"Motion {name} needs at least two frames")
    horizon = min(expected, max_steps_per_clip) if max_steps_per_clip else expected
    reward_sum, error_sums, reason = 0.0, {}, None
    terminated, truncated = False, False
    detected_failures = set()
    first_failure_step = None
    if on_frame:
      on_frame(env, clip, 0, env.read_state(), reward_sum)
    for step in range(horizon):
      action = policy(inputs["history"], inputs["future_reference"])
      if before_step:
        before_step(env)
      inputs, reward, terminal, cutoff, extras = env.step(action)
      if not torch.isfinite(reward).all():
        raise FloatingPointError(f"Nonfinite evaluation reward in {name}")
      reward_sum += float(reward[0])
      for key in ERROR_METRICS:
        error_sums[key] = error_sums.get(key, 0.0) + float(extras["metrics"][key][0])
      terminated, truncated = bool(terminal[0]), bool(cutoff[0])
      reasons = [
        key
        for key in ("height_failure", "tilt_failure")
        if key in extras["metrics"] and bool(extras["metrics"][key][0])
      ]
      if terminated and not reasons:
        reasons = ["physical_failure"]
      if reasons and first_failure_step is None:
        first_failure_step = step + 1
      detected_failures.update(reasons)
      if terminated:
        reason = "+".join(reasons)
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
        "success": completed_steps == expected and not detected_failures,
        "terminated": terminated,
        "failure_detected": bool(detected_failures),
        "first_failure_step": first_failure_step,
        "detected_failure_reasons": sorted(detected_failures),
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
  return _evaluation_report(env, rows)


def _evaluate_parallel(env, policy, max_steps_per_clip, on_motion) -> list[dict]:
  """Run fixed batches; mask each lane after its first finished episode.

  The environment may auto-reset finished lanes while others are still running.
  Their later rewards, errors and failures never contribute to the original clip.
  Accumulators stay on the simulation device until the batch ends. Float64 sums
  preserve the serial evaluator's Python-float reward accumulation.
  """
  device, width = env.device, env.num_envs
  lengths = torch.as_tensor(env.library.lengths, device=device)
  expected_steps = (lengths - 1).cpu().tolist()
  if min(expected_steps) < 1:
    raise ValueError("Every motion needs at least two frames")
  rows = []
  for start in range(0, len(env.library.names), width):
    count = min(width, len(env.library.names) - start)
    # Fill the final batch's unused lanes with a valid clip, but never score them.
    motion_ids = torch.arange(start, start + width, device=device).clamp_max(
      len(env.library.names) - 1
    )
    inputs, _ = env.reset(
      seed=env.foundation_config.seed,
      options={"motion_id": motion_ids, "frame_id": 0},
    )
    horizons = lengths[motion_ids] - 1
    if max_steps_per_clip is not None:
      horizons = horizons.clamp_max(max_steps_per_clip)
    active = torch.arange(width, device=device) < count
    reward_sum = torch.zeros(width, dtype=torch.float64, device=device)
    error_sums = torch.zeros(
      width, len(ERROR_METRICS), dtype=torch.float64, device=device
    )
    steps = torch.zeros(width, dtype=torch.long, device=device)
    first_failure = torch.zeros_like(steps)
    terminated = torch.zeros_like(active)
    truncated = torch.zeros_like(active)
    timed_out = torch.zeros_like(active)
    failures = torch.zeros(width, len(FAILURE_METRICS), dtype=torch.bool, device=device)
    invalid_reward = torch.zeros_like(active)
    for step in range(int(horizons[:count].max())):
      action = policy(inputs["history"], inputs["future_reference"])
      action = torch.where(active[:, None], action, 0.0)
      inputs, reward, terminal, cutoff, extras = env.step(action)
      metrics = extras["metrics"]
      invalid_reward |= active & ~torch.isfinite(reward)
      reward_sum += torch.where(active, reward, 0.0).double()
      errors = torch.stack([metrics[key] for key in ERROR_METRICS], -1)
      error_sums += torch.where(active[:, None], errors, 0.0).double()
      steps += active.long()
      height = metrics.get("height_failure", torch.zeros_like(active))
      tilt = metrics.get("tilt_failure", torch.zeros_like(active))
      detected = torch.stack((height, tilt, terminal & ~(height | tilt)), -1)
      failed_now = active & detected.any(-1)
      first_failure = torch.where(
        failed_now & (first_failure == 0), step + 1, first_failure
      )
      failures |= active[:, None] & detected
      terminated |= active & terminal
      truncated |= active & cutoff
      timed_out |= active & extras["transition"]["time_out"]
      active &= ~(terminal | cutoff | (step + 1 >= horizons))
      if not bool(active.any()):
        break
    if bool(invalid_reward.any()):
      raise FloatingPointError("Nonfinite evaluation reward in parallel batch")
    # One small device-to-host copy per batch, rather than one per scalar/step.
    results = (
      torch.cat(
        (
          reward_sum[:, None],
          steps[:, None],
          terminated[:, None],
          truncated[:, None],
          timed_out[:, None],
          first_failure[:, None],
          failures,
          error_sums,
        ),
        -1,
      )[:count]
      .cpu()
      .tolist()
    )
    for slot, result in enumerate(results):
      total, completed, terminal, cutoff, timeout, first, *values = result
      completed, terminal, cutoff = int(completed), bool(terminal), bool(cutoff)
      reasons = [key for key, failed in zip(FAILURE_METRICS, values[:3]) if failed]
      expected = expected_steps[start + slot]
      limited = completed < expected and not (terminal or cutoff)
      reason = (
        "+".join(reasons)
        if terminal
        else "time_limit"
        if timeout
        else "evaluation_limit"
        if limited
        else None
      )
      row = {
        "motion": env.library.names[start + slot],
        "steps": completed,
        "expected_steps": expected,
        "completion": completed / expected,
        "success": completed == expected and not reasons,
        "terminated": terminal,
        "failure_detected": bool(reasons),
        "first_failure_step": int(first) if first else None,
        "detected_failure_reasons": sorted(reasons),
        "failure_reason": reason,
        "evaluation_limited": limited,
        "return": total,
        "normalized_return": total / (5 * env.step_dt * expected),
        "mean_errors": {
          key: value / completed for key, value in zip(ERROR_METRICS, values[3:])
        },
      }
      rows.append(row)
      if on_motion:
        on_motion(row)
  return rows


def _evaluation_report(env, rows: list[dict]) -> dict:
  cfg = env.foundation_config
  termination_enabled = cfg.terminate_on_failure
  failed = [row["motion"] for row in rows if row["failure_detected"]]
  successes = sum(row["success"] for row in rows)
  complete = not any(
    row["evaluation_limited"] or row["failure_reason"] == "time_limit" for row in rows
  )
  return {
    "seed": cfg.seed,
    "evaluation_num_envs": env.num_envs,
    "termination_enabled": termination_enabled,
    "evaluation_mode": "standard" if termination_enabled else "playback",
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
    "protocol": "first frame; full clip unless explicitly limited; deterministic policy; friction 1.0; no reset/sensor/action noise or domain randomization; self collisions disabled; "
    + (
      "height/tilt failure termination enabled"
      if termination_enabled
      else "height/tilt failures diagnostic only; termination disabled for playback"
    ),
    "normalization": "per-clip undiscounted reward / (5 * control_dt * (frames - 1)); arithmetic mean over clips; "
    + (
      "no reward after failure"
      if termination_enabled
      else "playback reward continues after failure thresholds"
    ),
  }
