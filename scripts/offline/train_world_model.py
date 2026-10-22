"""Fit/evaluate a differentiable world model on episode-disjoint replay snapshots."""

import argparse
import importlib.metadata
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.tasks.supermimic.config.task_cfg import DEFAULT_MANIFEST
from src.tasks.supermimic.data.motion_library import MotionLibrary
from src.tasks.supermimic.data.world_model_data import ReplayCorpus, file_sha256
from src.tasks.supermimic.models.kinematics import robot_metadata
from src.tasks.supermimic.models.world_model import WorldModel
from src.tasks.supermimic.rl.world_model_training import (
  evaluate_world_model,
  fixed_validation_indices,
  rollout,
  world_model_loss,
)


def write_json(path: Path, value: dict):
  path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def save_checkpoint(path: Path, payload: dict):
  temporary = path.with_suffix(".tmp")
  torch.save(payload, temporary)
  temporary.replace(path)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--mode", choices=("train", "evaluate"), default="train")
  parser.add_argument("--replay", nargs="+", type=Path, required=True)
  parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument(
    "--output", type=Path, default=Path("outputs/supermimic/world_model")
  )
  parser.add_argument("--hidden-dims", nargs="+", type=int, default=[2048, 2048, 2048])
  parser.add_argument(
    "--steps",
    type=int,
    default=5000,
    help="Total optimizer steps, including steps before resume",
  )
  parser.add_argument("--batch-size", type=int, default=256)
  parser.add_argument("--validation-windows", type=int, default=256)
  parser.add_argument("--validation-batch-size", type=int, default=64)
  parser.add_argument("--eval-interval", type=int, default=100)
  parser.add_argument("--checkpoint-interval", type=int, default=500)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--cpu-threads", type=int, default=4)
  parser.add_argument("--resume", type=Path)
  parser.add_argument("--checkpoint", type=Path, help="Checkpoint for --mode evaluate")
  parser.add_argument(
    "--smoke",
    action="store_true",
    help="Allow a subset of clips; never qualify this run for BPTT",
  )
  args = parser.parse_args()
  if (
    min(
      args.steps,
      args.batch_size,
      args.validation_windows,
      args.validation_batch_size,
      args.eval_interval,
      args.checkpoint_interval,
      args.cpu_threads,
      *args.hidden_dims,
    )
    < 1
  ):
    parser.error("Counts and network widths must be positive")
  if args.device.startswith("cuda") and not torch.cuda.is_available():
    parser.error("CUDA is not visible to this Python process")
  if args.device == "cpu":
    torch.set_num_threads(args.cpu_threads)
  torch.backends.cuda.matmul.allow_tf32 = True
  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  if args.mode == "evaluate" and not args.checkpoint:
    parser.error("--mode evaluate needs --checkpoint")
  if args.mode == "evaluate" and args.resume:
    parser.error("--resume is only for training")
  if args.mode == "train" and args.checkpoint:
    parser.error("--checkpoint is only for evaluation; use --resume for training")
  if args.output.exists() and not args.resume:
    parser.error("Output exists; select a new output directory or resume last.pt")
  if args.resume and args.output.resolve() != args.resume.resolve().parent:
    parser.error("Resume into the checkpoint's existing output directory")
  manifest = json.loads(args.manifest.read_text())
  all_names = [
    Path(row["file"]).with_suffix("").as_posix() for row in manifest["clips"]
  ]
  selected = all_names
  if args.smoke:
    supplied = set()
    for path in args.replay:
      supplied.update(torch.load(path, map_location="cpu", weights_only=True)["clips"])
    if not supplied.issubset(all_names):
      parser.error("Replay contains clips not present in the manifest")
    selected = [name for name in all_names if name in supplied]
  checkpoint_path = args.resume or args.checkpoint
  checkpoint = (
    torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint_path
    else None
  )
  split_seed = checkpoint["config"]["seed"] if checkpoint else args.seed
  corpus = ReplayCorpus(args.replay, selected, seed=split_seed, strict=not args.smoke)
  metadata = robot_metadata()
  manifest_hash = file_sha256(args.manifest)
  if checkpoint:
    if checkpoint.get("schema_version") != 1:
      raise ValueError("Unsupported world-model checkpoint schema")
    if (
      checkpoint["split"] != corpus.split_description()
      or checkpoint["manifest_sha256"] != manifest_hash
    ):
      raise ValueError("Checkpoint data, manifest, or episode split does not match")
    if checkpoint["robot_metadata"]["signature"] != metadata["signature"]:
      raise ValueError("Checkpoint robot model does not match the current G1")
    if checkpoint["config"]["smoke"] != args.smoke:
      raise ValueError("Keep the original checkpoint's --smoke setting")
  hidden = tuple(checkpoint["hidden_dims"] if checkpoint else args.hidden_dims)
  model = WorldModel(metadata, hidden).to(args.device)
  library = MotionLibrary(
    args.manifest,
    model.kinematics.joint_names,
    model.kinematics.body_names,
    device=args.device,
    clips=tuple(selected),
  )
  # Reject invalid reference indices before any optimization or output creation.
  for buffer, held_out in zip(corpus.buffers, corpus.validation_masks):
    motions = buffer.data["motion_id"][: len(held_out)]
    frames = buffer.data["frame_id"][: len(held_out)]
    if (frames >= library.lengths.cpu()[motions] - 1).any():
      raise ValueError(
        "Replay frame has no successor in the selected reference library"
      )
  optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
  sample_rng = torch.Generator().manual_seed(split_seed + 1)
  step, best_loss = 0, float("inf")
  if checkpoint:
    model.load_state_dict(checkpoint["model"])
    fixed = checkpoint["validation_indices"]
    if args.resume:
      if args.batch_size != checkpoint["config"]["batch_size"]:
        parser.error("Resuming requires the original batch size")
      optimizer.load_state_dict(checkpoint["optimizer"])
      step, best_loss = checkpoint["step"], checkpoint["best_loss"]
      sample_rng.set_state(checkpoint["sample_rng"])
      torch.set_rng_state(checkpoint["torch_rng"])
      if args.device.startswith("cuda") and checkpoint["cuda_rng"]:
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
  else:
    transitions = {
      key: value.to(args.device) for key, value in corpus.training_transitions().items()
    }
    model.fit_normalization(**transitions)
    del transitions
    fixed = fixed_validation_indices(corpus, args.validation_windows, split_seed)
  training_indices = corpus.indices(8, "train")
  if not len(training_indices):
    raise ValueError("No valid eight-step training windows")
  args.output.mkdir(parents=True, exist_ok=bool(args.resume))
  config = vars(args).copy()
  config["seed"] = split_seed
  runtime = {
    name: importlib.metadata.version(name)
    for name in ("torch", "mujoco", "mujoco-warp", "warp-lang")
  }
  write_json(
    args.output
    / ("evaluation_config.json" if args.mode == "evaluate" else "config.json"),
    {
      "arguments": {
        key: str(value)
        if isinstance(value, Path)
        else [str(v) for v in value]
        if key == "replay"
        else value
        for key, value in config.items()
      },
      "runtime": runtime,
      "robot_signature": metadata["signature"],
    },
  )
  write_json(args.output / "split.json", corpus.split_description())

  def validate():
    report = evaluate_world_model(
      model, corpus, library, fixed, args.device, args.validation_batch_size
    )
    report["step"] = step
    report["qualification"]["checks"]["formal_dataset"] = not args.smoke
    report["qualification"]["candidate_for_bptt"] = all(
      report["qualification"]["checks"].values()
    )
    write_json(args.output / "validation.json", report)
    return report

  def payload():
    return {
      "schema_version": 1,
      "model": model.state_dict(),
      "optimizer": optimizer.state_dict(),
      "step": step,
      "best_loss": best_loss,
      "hidden_dims": list(hidden),
      "robot_metadata": metadata,
      "manifest_sha256": manifest_hash,
      "split": corpus.split_description(),
      "validation_indices": fixed,
      "sample_rng": sample_rng.get_state(),
      "torch_rng": torch.get_rng_state(),
      "cuda_rng": torch.cuda.get_rng_state_all()
      if args.device.startswith("cuda")
      else [],
      "config": {
        "seed": split_seed,
        "batch_size": args.batch_size,
        "smoke": args.smoke,
      },
      "runtime": runtime,
    }

  if args.mode == "evaluate":
    step = checkpoint["step"]
    print(json.dumps(validate(), indent=2, allow_nan=False))
    return
  if args.steps < step:
    parser.error("--steps must be at least the checkpoint optimizer step")
  log_path = args.output / "metrics.jsonl"
  if args.resume and log_path.exists():
    records = [
      json.loads(line) for line in log_path.read_text().splitlines() if line.strip()
    ]
    log_path.write_text(
      "".join(json.dumps(row) + "\n" for row in records if row["step"] <= step)
    )
  initial = validate()
  if not args.resume:
    row = initial["horizons"]["8"]
    if row.get("finite"):
      best_loss = row["model"]["normalized_state_mse"]
    save_checkpoint(args.output / "best.pt", payload())
    save_checkpoint(args.output / "last.pt", payload())
  with log_path.open("a") as log:
    while step < args.steps:
      chosen = torch.randint(
        len(training_indices), (args.batch_size,), generator=sample_rng
      )
      batch = corpus.batch(training_indices[chosen], 8, args.device)
      model.train()
      optimizer.zero_grad(set_to_none=True)
      prediction = rollout(model, batch, library)
      loss, parts = world_model_loss(prediction, batch)
      if not torch.isfinite(loss):
        raise FloatingPointError(f"Nonfinite world-model loss at step {step + 1}")
      loss.backward()
      norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), 1.0, error_if_nonfinite=True
      )
      optimizer.step()
      step += 1
      record = {
        "step": step,
        "loss": float(loss.detach()),
        "gradient_norm": float(norm),
        **{f"loss_{key}": float(value.detach()) for key, value in parts.items()},
      }
      if step % args.eval_interval == 0 or step == args.steps:
        report = validate()
        row = report["horizons"]["8"]
        if row.get("finite"):
          score = row["model"]["normalized_state_mse"]
          record["validation_state_mse_8"] = score
          if score < best_loss:
            best_loss = score
            save_checkpoint(args.output / "best.pt", payload())
        print(json.dumps(record), flush=True)
      log.write(json.dumps(record, allow_nan=False) + "\n")
      log.flush()
      if step % args.checkpoint_interval == 0 or step == args.steps:
        snapshot = payload()
        save_checkpoint(args.output / "last.pt", snapshot)
        if step % args.checkpoint_interval == 0:
          save_checkpoint(args.output / f"step_{step:06d}.pt", snapshot)


if __name__ == "__main__":
  main()
