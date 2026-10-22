"""Repeatable logged-action model checks and reset-distribution inspection."""

import torch

from ..models.world_model import (
  GROUP_NAMES,
  normalized_state_error,
  state_errors,
)
from .world_model_training import rollout


@torch.no_grad()
def replay_diagnostics(world, replay, library, batch_size, fixed_windows, iteration):
  """Keep copied windows across ring overwrites and checkpoints.

  These are regression snapshots of training replay, not held-out generalization
  data. Sampling never advances the training replay RNG. Unavailable horizons
  are retried at the next evaluation, without scoring padding as a prediction.
  """
  rng = replay.generator.get_state()
  was_training = world.training
  report = {
    "scope": "fixed training-replay snapshots; logged actions; no held-out claim",
    "horizons": {},
  }
  try:
    replay.generator.manual_seed(17001)
    world.eval()
    for horizon in (1, 8, 32):
      key = str(horizon)
      if key not in fixed_windows and min(replay.steps, replay.capacity) >= horizon:
        try:
          batch = replay.sample_windows(batch_size, horizon, require_full=True)
        except ValueError:
          pass
        else:
          fixed_windows[key] = {
            "iteration": iteration,
            "batch": {
              name: value.detach().cpu().clone() for name, value in batch.items()
            },
          }
      if key not in fixed_windows:
        report["horizons"][key] = {"available": False, "windows": 0}
        continue
      snapshot = fixed_windows[key]
      batch = {
        name: value.to(replay.device) for name, value in snapshot["batch"].items()
      }
      motion_ids = batch["motion_id"][:, 0].unique().tolist()
      row = {
        "available": True,
        "windows": len(batch["state"]),
        "snapshot_iteration": snapshot["iteration"],
        "covered_clips": [library.names[i] for i in motion_ids],
        "missing_clips": [
          name for i, name in enumerate(library.names) if i not in motion_ids
        ],
      }
      for name, prior in (("model", None), ("constant_velocity", 0.0)):
        prediction = rollout(world, batch, library, prior)
        finite = all(bool(torch.isfinite(value).all()) for value in prediction.values())
        metrics = {"finite": finite}
        if finite:
          final, target = (
            prediction["next_state"][:, -1],
            batch["next_state"][:, -1],
          )
          metrics.update(
            {
              group + "_rmse": float(error.square().mean().sqrt())
              for group, error in zip(GROUP_NAMES, state_errors(final, target))
            }
          )
          metrics.update(
            normalized_state_mse=float(normalized_state_error(final, target).mean()),
            reward_mae=float((prediction["reward"] - batch["reward"]).abs().mean()),
            predicted_failure_probability=float(
              prediction["termination_probability"].mean()
            ),
            physical_failure_fraction=float(batch["terminated"].float().mean()),
            termination_brier=float(
              (prediction["termination_probability"] - batch["terminated"].float())
              .square()
              .mean()
            ),
          )
        row[name] = metrics
      report["horizons"][key] = row
  finally:
    replay.generator.set_state(rng)
    world.train(was_training)
  return report


@torch.no_grad()
def sampler_diagnostics(sampler, library, top_k=10):
  probabilities = sampler.probabilities()
  values, indices = probabilities.topk(min(top_k, len(probabilities)))
  return {
    "bins": len(probabilities),
    "entropy": float(-(probabilities * probabilities.log()).sum()),
    "top": [
      {
        "motion": library.names[int(sampler.bin_motion[index])],
        "start_seconds": int(sampler.bin_start[index]) / library.fps,
        "probability": float(probability),
        "normalized_return_estimate": float(
          sampler.returns[index] / sampler.max_return
        ),
      }
      for probability, index in zip(values, indices)
    ],
  }
