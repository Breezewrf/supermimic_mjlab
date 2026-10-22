"""Autoregressive supervised fitting and fixed-window validation utilities."""

import torch
from torch import Tensor
from torch.nn import functional as F

from ..models.world_model import (
  GROUP_NAMES,
  integrate,
  normalized_state_error,
  state_errors,
)


def rollout(
  model, batch: dict[str, Tensor], library, baseline_prior: float | None = None
) -> dict[str, Tensor]:
  reference = library.get_frame(batch["motion_id"], batch["frame_id"] + 1)
  state = batch["state"][:, 0]
  outputs = {
    key: []
    for key in (
      "next_state",
      "reward",
      "clean_observation",
      "termination_logit",
      "termination_probability",
      "body_pos_w",
    )
  }
  for step in range(batch["mask"].shape[1]):
    action, context = batch["action"][:, step], batch["context"][:, step]
    next_state = (
      model(state, action, context) if baseline_prior is None else integrate(state)
    )
    # Frozen valid poses are used after termination; padded zero quaternions are
    # never fed into the dynamics or kinematics networks.
    state = torch.where(batch["mask"][:, step, None], next_state, state)
    predicted = model.task_heads(
      state,
      action,
      context,
      {key: value[:, step] for key, value in reference.items()},
      batch["previous_action"][:, step],
    )
    for key, values in outputs.items():
      value = predicted["bodies"][key] if key == "body_pos_w" else predicted[key]
      if baseline_prior is not None and key in (
        "termination_logit",
        "termination_probability",
      ):
        probability = value.new_full(value.shape, baseline_prior).clamp(1e-6, 1 - 1e-6)
        value = probability.logit() if key == "termination_logit" else probability
      values.append(value)
  return {key: torch.stack(values, dim=1) for key, values in outputs.items()}


def world_model_loss(
  prediction: dict[str, Tensor],
  batch: dict[str, Tensor],
  *,
  analytic_losses: bool = True,
) -> tuple[Tensor, dict]:
  """Equation (1): sum valid time steps, then average over replay windows."""
  mask = batch["mask"]
  batch_size = mask.shape[0]
  state = (
    normalized_state_error(
      prediction["next_state"][mask], batch["next_state"][mask]
    ).sum()
    / batch_size
  )
  observation_error = (
    prediction["clean_observation"][mask][:, :64]
    - batch["next_clean_observation"][mask][:, :64]
  )
  observation = torch.stack(
    [
      (value / scale).square().mean(-1).sum() / batch_size
      for value, scale in zip(
        observation_error.split((29, 29, 3, 3), dim=-1), (0.3, 5.0, 3.14, 1.0)
      )
    ]
  ).mean()
  reward = (
    (prediction["reward"][mask] - batch["reward"][mask]) / 0.1
  ).square().sum() / batch_size
  termination = (
    F.binary_cross_entropy_with_logits(
      prediction["termination_logit"][mask],
      batch["terminated"][mask].float(),
      reduction="sum",
    )
    / batch_size
  )
  parts = {
    "state": state,
    "observation": observation,
    "reward": reward,
    "termination": termination,
  }
  total = sum(parts.values()) if analytic_losses else state + termination
  return total, parts


def fixed_validation_indices(
  corpus, max_windows: int = 256, seed: int = 0
) -> dict[str, Tensor]:
  if max_windows < len(corpus.clip_names):
    raise ValueError(
      "Validation window limit must be at least the number of selected clips"
    )
  rng = torch.Generator().manual_seed(seed + 1000)
  result = {}
  for horizon in (1, 8, 32):
    table = corpus.indices(horizon, "validation", require_full=True)
    if len(table) <= max_windows:
      result[str(horizon)] = table.clone()
      continue
    # Include every clip with an eligible window, then fill randomly without
    # replacement. This avoids reporting only the easiest/longest motions.
    motion = corpus.batch(table, 1)["motion_id"][:, 0]
    selected = []
    for clip in motion.unique().tolist():
      candidates = (motion == clip).nonzero().flatten()
      selected.append(
        candidates[torch.randint(len(candidates), (), generator=rng)].item()
      )
    available = torch.ones(len(table), dtype=torch.bool)
    available[selected] = False
    remaining = available.nonzero().flatten()
    extras = remaining[
      torch.randperm(len(remaining), generator=rng)[: max_windows - len(selected)]
    ]
    result[str(horizon)] = table[torch.cat((torch.tensor(selected), extras))].clone()
  return result


def _termination_metrics(probability: Tensor, target: Tensor) -> dict:
  target = target.bool()
  selected = probability >= 0.5
  tp = int((selected & target).sum())
  predicted_positive, positive = int(selected.sum()), int(target.sum())
  return {
    "bce": float(
      F.binary_cross_entropy(probability.clamp(1e-6, 1 - 1e-6), target.float())
    ),
    "brier": float((probability - target.float()).square().mean()),
    "precision": tp / predicted_positive if predicted_positive else None,
    "recall": tp / positive if positive else None,
    "positive_fraction": float(target.float().mean()),
    "labels": len(target),
  }


def _aggregate_metrics(metrics: dict[str, Tensor], labels: Tensor) -> dict:
  result = {
    key: float(value.mean()) for key, value in metrics.items() if key != "probability"
  }
  for group in GROUP_NAMES:
    result[group + "_rmse"] = result.pop(group + "_mse") ** 0.5
  result["observation_rmse"] = result.pop("observation_mse") ** 0.5
  result["termination"] = _termination_metrics(
    metrics["probability"].flatten(), labels.flatten()
  )
  return result


@torch.no_grad()
def evaluate_world_model(
  model,
  corpus,
  library,
  fixed_indices: dict[str, Tensor],
  device: str,
  batch_size: int = 64,
) -> dict:
  training_labels = torch.cat(
    [
      buffer.data["terminated"][: len(mask)][~mask]
      for buffer, mask in zip(corpus.buffers, corpus.validation_masks)
    ]
  )
  prior = float(training_labels.float().mean())
  report = {
    "horizons": {},
    "clip_names": corpus.clip_names,
    "baseline": "constant linear/angular/joint velocity; train-set failure prior",
    "metric_definition": "endpoint state and link error; task errors across all rollout steps",
    "scope": "logged-action prediction on the collected controller distribution",
  }
  was_training = model.training
  model.eval()
  for text_horizon, indices in fixed_indices.items():
    horizon = int(text_horizon)
    if not len(indices):
      report["horizons"][text_horizon] = {"available": False, "windows": 0}
      continue
    collected = {name: [] for name in ("model", "baseline")}
    motions, targets = [], []
    finite, quaternion_error = True, 0.0
    for start in range(0, len(indices), batch_size):
      batch = corpus.batch(indices[start : start + batch_size], horizon, device)
      motions.append(batch["motion_id"][:, 0].cpu())
      targets.append(batch["terminated"].flatten().cpu())
      expected = model.kinematics(batch["next_state"][:, -1])["body_pos_w"]
      for name, probability in (("model", None), ("baseline", prior)):
        predicted = rollout(model, batch, library, probability)
        for value in predicted.values():
          finite &= bool(torch.isfinite(value).all())
        if not finite:
          break
        final = predicted["next_state"][:, -1]
        quaternion_error = max(
          quaternion_error,
          float((predicted["next_state"][..., 3:7].norm(dim=-1) - 1).abs().max()),
        )
        metrics = {
          "normalized_state_mse": normalized_state_error(
            final, batch["next_state"][:, -1]
          ),
          "link_position_mpjpe_m": (predicted["body_pos_w"][:, -1] - expected)
          .norm(dim=-1)
          .mean(-1),
          "reward_mae": (predicted["reward"] - batch["reward"]).abs().mean(-1),
          "observation_mse": (
            predicted["clean_observation"][..., :64]
            - batch["next_clean_observation"][..., :64]
          )
          .square()
          .mean((-1, -2)),
          "probability": predicted["termination_probability"],
        }
        for group, error in zip(
          GROUP_NAMES, state_errors(final, batch["next_state"][:, -1])
        ):
          metrics[group + "_mse"] = error.square().mean(-1)
        collected[name].append({key: value.cpu() for key, value in metrics.items()})
      if not finite:
        break
    result = {
      "available": True,
      "windows": len(indices),
      "finite": finite,
      "max_quaternion_norm_error": quaternion_error,
    }
    if finite:
      motion = torch.cat(motions)
      labels = torch.cat(targets).reshape(-1, horizon)
      result["covered_clips"] = [
        clip for i, clip in enumerate(corpus.clip_names) if (motion == i).any()
      ]
      result["missing_clips"] = [
        clip for clip in corpus.clip_names if clip not in result["covered_clips"]
      ]
      for name, batches in collected.items():
        merged = {
          key: torch.cat([batch[key] for batch in batches]) for key in batches[0]
        }
        result[name] = _aggregate_metrics(merged, labels)
        result[name]["per_clip"] = {
          clip: {
            "windows": int((motion == i).sum()),
            **_aggregate_metrics(
              {key: value[motion == i] for key, value in merged.items()},
              labels[motion == i],
            ),
          }
          for i, clip in enumerate(corpus.clip_names)
          if (motion == i).any()
        }
    report["horizons"][text_horizon] = result
  checks = {}
  for horizon in ("1", "8"):
    row = report["horizons"].get(horizon, {})
    checks[f"beats_baseline_{horizon}"] = bool(
      row.get("finite")
      and row["model"]["normalized_state_mse"] < row["baseline"]["normalized_state_mse"]
    )
  row = report["horizons"].get("32", {})
  checks["stable_32"] = bool(
    row.get("finite") and row["max_quaternion_norm_error"] < 1e-4
  )
  checks["full_clip_coverage"] = all(
    len(report["horizons"].get(str(h), {}).get("covered_clips", []))
    == len(corpus.clip_names)
    for h in (1, 8, 32)
  )
  report["qualification"] = {
    "checks": checks,
    "candidate_for_bptt": all(checks.values()),
  }
  model.train(was_training)
  return report
