"""Train SuperMimic online from scratch using simulator data and imagined BPTT."""

import argparse
import importlib.metadata
import importlib.util
import json
import sys
import traceback
from dataclasses import asdict, fields
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.tasks.supermimic.config.rl_cfg import OnlineConfig
from src.tasks.supermimic.config.task_cfg import (
  DEFAULT_MANIFEST,
  FoundationConfig,
)
from src.tasks.supermimic.data.world_model_data import file_sha256
from src.tasks.supermimic.models.kinematics import robot_metadata
from src.tasks.supermimic.rl.evaluation import evaluate
from src.tasks.supermimic.rl.online_training import (
  OnlineTrainer,
  restore_rng,
  rng_state,
)
from src.tasks.supermimic.rl.training_logger import (
  ConsoleReporter,
  MetricWindow,
  WandbReporter,
  simulator_output,
)


def write_json(path: Path, data: dict):
  path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def save_checkpoint(path: Path, data: dict):
  temporary = path.with_suffix(".tmp")
  torch.save(data, temporary)
  temporary.replace(path)


def evaluate_actor(actor, manifest, config, device, limit=None):
  from src.tasks.supermimic.env import SuperMimicEnv

  saved_rng, training = rng_state(), actor.training
  env = None
  try:
    actor.eval()
    env = SuperMimicEnv(
      FoundationConfig(
        manifest=manifest,
        clips=config.clips,
        num_envs=1,
        seed=config.seed,
        episode_length_s=1e9,
        relative_orientation=config.relative_orientation,
      ),
      device=device,
    )
    return evaluate(env, actor, limit)
  finally:
    if env is not None:
      env.close()
    actor.train(training)
    restore_rng(saved_rng)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--output", type=Path)
  parser.add_argument("--resume", type=Path)
  parser.add_argument(
    "--iterations",
    type=int,
    default=1000,
    help="Total online update rounds, including saved rounds",
  )
  parser.add_argument("--eval-interval", type=int, default=100)
  parser.add_argument("--checkpoint-interval", type=int, default=100)
  parser.add_argument("--log-interval", type=int, default=10)
  parser.add_argument("--log-format", choices=("table", "json"), default="table")
  parser.add_argument(
    "--verbose-sim",
    action="store_true",
    help="Show simulator initialization and kernel output in the terminal",
  )
  parser.add_argument(
    "--wandb",
    action="store_true",
    help="Track metrics and evaluation charts with Weights & Biases",
  )
  parser.add_argument(
    "--wandb-project",
    help="W&B project; default: supermimic, or the saved project on resume",
  )
  parser.add_argument("--wandb-entity", help="W&B user or team")
  parser.add_argument(
    "--wandb-name", help="W&B run name; default: output directory name"
  )
  parser.add_argument("--wandb-mode", choices=("online", "offline"), default="online")
  parser.add_argument("--termination-mode", choices=("learned", "analytic"))
  parser.add_argument(
    "--relative-orientation",
    action=argparse.BooleanOptionalAction,
    default=None,
  )
  parser.add_argument(
    "--evaluation-limit",
    type=int,
    help="Debug only: explicitly label limited evaluations",
  )
  parser.add_argument("--cpu-threads", type=int, default=4)
  parser.add_argument(
    "--detect-anomaly",
    action="store_true",
    help="Diagnose backward failures; slower",
  )
  for name in (
    "num_envs",
    "capacity",
    "warmup_steps",
    "collect_steps",
    "updates_per_collect",
    "wm_batches",
    "wm_batch_size",
    "policy_batch_size",
    "horizon",
    "critic_epochs",
    "critic_minibatches",
    "diagnostics_batch_size",
    "seed",
  ):
    parser.add_argument("--" + name.replace("_", "-"), type=int)
  for name in ("world_dims", "actor_dims", "critic_dims"):
    parser.add_argument("--" + name.replace("_", "-"), type=int, nargs="+")
  parser.add_argument("--clips", nargs="+")
  args = parser.parse_args()
  if args.wandb and importlib.util.find_spec("wandb") is None:
    parser.error(
      "W&B is optional. Install logging support with: uv sync --extra logging"
    )
  if min(
    args.iterations,
    args.eval_interval,
    args.checkpoint_interval,
    args.log_interval,
    args.cpu_threads,
  ) < 1 or (args.evaluation_limit is not None and args.evaluation_limit < 1):
    parser.error("Counts must be positive")
  if args.device.startswith("cuda") and not torch.cuda.is_available():
    parser.error("CUDA is not visible to this Python process")
  torch.set_num_threads(args.cpu_threads)
  torch.backends.cuda.matmul.allow_tf32 = True
  torch.autograd.set_detect_anomaly(args.detect_anomaly)
  saved = (
    torch.load(args.resume, map_location="cpu", weights_only=True)
    if args.resume
    else None
  )
  if saved and (
    saved.get("schema_version") != 1 or saved.get("kind") != "supermimic_online"
  ):
    parser.error("--resume requires an online SuperMimic checkpoint")
  if saved:
    if saved.get("sampler", {}).get("schema_version") != 2:
      parser.error(
        "Legacy reset scores are incompatible with this fix; start a new run. Old actors remain usable for evaluation."
      )
    if args.evaluation_limit is None:
      args.evaluation_limit = saved["evaluation_limit"]
    elif args.evaluation_limit != saved["evaluation_limit"]:
      parser.error(
        "Resume requires the original evaluation limit for comparable best checkpoints"
      )
  config_values = dict(saved["config"]) if saved else asdict(OnlineConfig())
  for field in fields(OnlineConfig):
    value = getattr(args, field.name, None)
    if value is not None:
      config_values[field.name] = (
        tuple(value) if field.name.endswith("_dims") or field.name == "clips" else value
      )
  try:
    config = OnlineConfig(**config_values)
  except ValueError as error:
    parser.error(str(error))
  manifest_hash = file_sha256(args.manifest)
  if saved and (
    saved["config"] != asdict(config) or saved["manifest_sha256"] != manifest_hash
  ):
    parser.error("Resume requires the original training configuration and manifest")
  if saved and args.iterations < saved["iteration"]:
    parser.error("--iterations cannot be below the saved iteration")
  output = args.output or (
    args.resume.resolve().parent if args.resume else Path("outputs/supermimic/online")
  )
  if args.resume:
    if output.resolve() != args.resume.resolve().parent:
      parser.error("Resume into the checkpoint's existing output directory")
  elif output.exists():
    parser.error("Output exists; choose a new directory or use --resume")
  output.mkdir(parents=True, exist_ok=bool(saved))
  console = ConsoleReporter(args.iterations, args.log_format)
  console.message(f"Initializing simulation; details: {output / 'simulator.log'}")
  sim_log = (output / "simulator.log").open("a", buffering=1)
  env = trainer = tracking = None
  completed = False
  try:
    with simulator_output(sim_log, args.verbose_sim):
      from src.tasks.supermimic.env import SuperMimicEnv

      env = SuperMimicEnv(
        FoundationConfig.training(
          manifest=args.manifest,
          num_envs=config.num_envs,
          clips=config.clips,
          seed=config.seed,
          episode_length_s=config.episode_length_s,
          relative_orientation=config.relative_orientation,
        ),
        device=args.device,
      )
      trainer = OnlineTrainer(env, robot_metadata(), config)
      if saved:
        trainer.load_state_dict(saved)
      else:
        trainer.bootstrap()
    runtime = {
      name: importlib.metadata.version(name)
      for name in ("torch", "mujoco", "mujoco-warp", "warp-lang")
    }
    run_config = {
      "config": asdict(config),
      "device": args.device,
      "manifest": str(args.manifest.resolve()),
      "manifest_sha256": manifest_hash,
      "runtime": runtime,
      "robot_signature": trainer.metadata["signature"],
      "simulation": trainer.simulation_parameters,
      "evaluation_limit": args.evaluation_limit,
      "resume": str(args.resume) if args.resume else None,
    }
    write_json(output / "config.json", run_config)
    console.start(
      asdict(config), args.device, len(env.library), output, trainer.iteration
    )
    if args.wandb:
      state = rng_state()
      try:
        tracking = WandbReporter(
          output,
          run_config,
          trainer.iteration,
          project=args.wandb_project,
          entity=args.wandb_entity,
          name=args.wandb_name,
          mode=args.wandb_mode,
          console=console,
        )
      finally:
        restore_rng(state)
    best_return = saved["best_return"] if saved else -float("inf")

    def checkpoint():
      return {
        **trainer.state_dict(),
        "manifest_sha256": manifest_hash,
        "runtime": runtime,
        "best_return": best_return,
        "evaluation_limit": args.evaluation_limit,
        "wandb_run": dict(tracking.state) if tracking else None,
      }

    def validate():
      nonlocal best_return
      console.message(f"Evaluating policy at iteration {trainer.iteration:,}...")
      with simulator_output(sim_log, args.verbose_sim):
        report = evaluate_actor(
          trainer.actor,
          args.manifest,
          config,
          args.device,
          args.evaluation_limit,
        )
      report["iteration"] = trainer.iteration
      report["training_environment_steps"] = trainer.environment_steps
      report["evaluation_limit"] = args.evaluation_limit
      write_json(output / f"evaluation_{trainer.iteration:06d}.json", report)
      write_json(output / "evaluation.json", report)
      diagnostics = trainer.diagnostics()
      write_json(output / f"diagnostics_{trainer.iteration:06d}.json", diagnostics)
      write_json(output / "diagnostics.json", diagnostics)
      if report["normalized_return"] > best_return:
        best_return = report["normalized_return"]
        save_checkpoint(output / "best.pt", checkpoint())
      console.evaluation(report)
      if tracking:
        tracking.evaluation(report)
        tracking.diagnostics(diagnostics)

    log_path = output / "metrics.jsonl"
    if saved and log_path.exists():
      rows = [
        json.loads(line) for line in log_path.read_text().splitlines() if line.strip()
      ]
      log_path.write_text(
        "".join(
          json.dumps(row) + "\n"
          for row in rows
          if row["iteration"] <= trainer.iteration
        )
      )
    if not saved:
      validate()
      save_checkpoint(output / "last.pt", checkpoint())
    if args.device.startswith("cuda"):
      torch.cuda.reset_peak_memory_stats(args.device)
    window = MetricWindow()
    with log_path.open("a") as log:
      while trainer.iteration < args.iterations:
        with simulator_output(sim_log, args.verbose_sim):
          record = trainer.train_iteration()
        log.write(json.dumps(record, allow_nan=False) + "\n")
        log.flush()
        window.add(record)
        if tracking:
          tracking.training(record)
        if (
          trainer.iteration % args.log_interval == 0
          or trainer.iteration == args.iterations
        ):
          summary = window.summary()
          console.training(record if args.log_format == "json" else summary)
          if tracking:
            tracking.window(summary)
          window = MetricWindow()
        if (
          trainer.iteration % args.eval_interval == 0
          or trainer.iteration == args.iterations
        ):
          validate()
        if (
          trainer.iteration % args.checkpoint_interval == 0
          or trainer.iteration == args.iterations
        ):
          snapshot = checkpoint()
          save_checkpoint(output / "last.pt", snapshot)
          if trainer.iteration % args.checkpoint_interval == 0:
            save_checkpoint(output / f"iteration_{trainer.iteration:06d}.pt", snapshot)
          console.message(
            f"Checkpoint saved at iteration {trainer.iteration:,}: {output / 'last.pt'}"
          )
    completed = True
  except Exception as error:
    if output.exists():
      write_json(
        output / "failure.json",
        {
          "iteration": trainer.iteration if trainer else None,
          "phase": trainer.phase if trainer else "initialization",
          "error": str(error),
          "traceback": traceback.format_exc(),
        },
      )
    raise
  finally:
    try:
      if tracking:
        tracking.close(exit_code=0 if completed else 1)
    finally:
      try:
        if env is not None:
          env.close()
      finally:
        sim_log.close()


if __name__ == "__main__":
  main()
