"""Train a proprioceptive SuperMimic velocity interface with a shared decoder."""

import argparse
import json
import random
import sys
from dataclasses import asdict, fields, replace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tasks.supermimic.config.cli import add_robot_argument, resolve_robot_arguments
from src.tasks.supermimic.config.morphology import robot_layout
from src.tasks.supermimic.config.rl_cfg import OnlineConfig
from src.tasks.supermimic.config.task_cfg import FoundationConfig
from src.tasks.supermimic.config.velocity_cfg import VelocityConfig
from src.tasks.supermimic.data.world_model_data import file_sha256
from src.tasks.supermimic.models.kinematics import robot_metadata
from src.tasks.supermimic.rl.checkpoints import save_checkpoint, training_config
from src.tasks.supermimic.rl.inference import validate_environment
from src.tasks.supermimic.rl.training_logger import simulator_output
from src.tasks.supermimic.rl.training_runner import (
  TrainingRuntime,
  add_logging_arguments,
  validate_logging_arguments,
  write_json,
)
from src.tasks.supermimic.rl.velocity_logger import (
  VelocityConsoleReporter,
  VelocityMetricWindow,
  VelocityWandbReporter,
)
from src.tasks.supermimic.rl.velocity_training import VelocityTrainer
from src.tasks.supermimic.velocity_env import VelocityEnv


def parser():
  p = argparse.ArgumentParser(description=__doc__)
  add_robot_argument(p)
  p.add_argument("--manifest", type=Path)
  p.add_argument(
    "--actor-checkpoint",
    type=Path,
    default=Path("outputs/policy_iteration_010000.pt"),
  )
  p.add_argument(
    "--world-checkpoint",
    type=Path,
    default=Path("checkpoints/best/world_model_best.pt"),
  )
  p.add_argument("--from-scratch", action="store_true")
  p.add_argument("--resume", type=Path)
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--output", type=Path)
  p.add_argument("--stage", choices=("motion", "warmup", "joint"), default="joint")
  p.add_argument("--iterations", type=int, default=100000)
  p.add_argument("--warmup-updates", type=int, default=500)
  p.add_argument(
    "--command-hold-seconds",
    type=float,
    nargs=2,
    metavar=("MIN", "MAX"),
    help="Independent command holding-time range (new runs: 3 8); may override on resume",
  )
  p.add_argument(
    "--randomize-initial-episode-progress",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Stagger first episode timeouts (default: enabled, including resume)",
  )
  p.add_argument("--num-envs", type=int, default=4096)
  p.add_argument("--capacity", type=int, default=1000)
  p.add_argument("--warmup-steps", type=int, default=32)
  p.add_argument("--collect-steps", type=int, default=5)
  p.add_argument("--updates-per-collect", type=int, default=2)
  p.add_argument("--wm-batches", type=int, default=10)
  p.add_argument("--wm-batch-size", type=int, default=2048)
  p.add_argument("--policy-batch-size", type=int, default=4096)
  p.add_argument("--horizon", type=int, default=32)
  p.add_argument("--critic-epochs", type=int, default=5)
  p.add_argument("--critic-minibatches", type=int, default=4)
  p.add_argument("--learning-rate", type=float, default=2e-4)
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--clips", nargs="*", default=())
  p.add_argument(
    "--actor-dims",
    nargs="+",
    type=int,
    default=(2048, 1024, 512, 256),
    help="Used only with --from-scratch",
  )
  p.add_argument(
    "--world-dims",
    nargs="+",
    type=int,
    default=(2048, 2048, 2048),
    help="Used only with --from-scratch",
  )
  p.add_argument("--critic-dims", nargs="+", type=int, default=(256, 128))
  p.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False)
  p.add_argument("--checkpoint-interval", type=int, default=1000)
  add_logging_arguments(p)
  p.add_argument(
    "--initial-evaluation-steps",
    type=int,
    default=250,
    help="Per-clip motion baseline check; 0 skips it",
  )
  p.add_argument("--stand-still-weight", type=float, default=0.0)
  return p


def main():
  argument_parser = parser()
  args = argument_parser.parse_args()
  validate_logging_arguments(argument_parser, args)
  if (
    min(args.iterations, args.checkpoint_interval, args.log_interval) < 1
    or args.initial_evaluation_steps < 0
  ):
    raise ValueError("Invalid iteration/evaluation intervals")
  args.output = args.output or (
    args.resume.resolve().parent if args.resume else Path("outputs/velocity/run")
  )
  if args.resume:
    if args.output.resolve() != args.resume.resolve().parent:
      argument_parser.error("Resume into the checkpoint's existing output directory")
  elif args.output.exists():
    argument_parser.error("Output exists; choose a new directory or use --resume")
  resolve_robot_arguments(args, argument_parser)
  if args.device.startswith("cuda") and not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable in this process")
  torch.manual_seed(args.seed)
  np.random.seed(args.seed)
  random.seed(args.seed)
  torch.set_num_threads(4)
  saved = (
    torch.load(args.resume, map_location="cpu", weights_only=True, mmap=True)
    if args.resume
    else None
  )
  actor_saved = world_saved = None
  if saved is not None:
    if saved.get("kind") != "supermimic_velocity_online":
      raise ValueError("Expected a velocity training checkpoint")
    if args.iterations < saved["iteration"]:
      argument_parser.error("--iterations cannot be below the saved iteration")
    cfg = OnlineConfig(**saved["config"])
    velocity = VelocityConfig(**saved["velocity_config"])
    stage = saved["stage"]
    if saved["manifest_sha256"] != file_sha256(args.manifest):
      raise ValueError("Resume manifest mismatch")
  else:
    names = {f.name for f in fields(OnlineConfig)}
    values = {key: value for key, value in vars(args).items() if key in names}
    for key in ("clips", "actor_dims", "world_dims", "critic_dims"):
      values[key] = tuple(values[key])
    values.update(
      amp="none",
      reference_dim=robot_layout(args.robot).full_body_reference_dim,
      actor_input_normalization=True,
      critic_input_normalization=True,
      world_model_normalization=True,
    )
    if not args.from_scratch:
      actor_saved = torch.load(
        args.actor_checkpoint, map_location="cpu", weights_only=True, mmap=True
      )
      actor_cfg = training_config(actor_saved)
      if (
        actor_cfg["relative_orientation"]
        or actor_cfg["motion_anchor_ori_b"]
        or actor_saved["actor"]["decoder.0.weight"].shape[1]
        != 32 + 10 * robot_layout(args.robot).observation_dim
      ):
        raise ValueError(
          "Initialization requires a matching robot and proprioceptive decoder"
        )
      if actor_saved["manifest_sha256"] != file_sha256(args.manifest):
        raise ValueError("Actor initialization manifest mismatch")
      for key in ("actor_dims", "reference_dim", "actor_input_normalization"):
        values[key] = actor_cfg[key]
      world_saved = torch.load(
        args.world_checkpoint, map_location="cpu", weights_only=True, mmap=True
      )
      for key, default in (
        ("world_dims", (2048, 2048, 2048)),
        ("world_prediction", "velocity_residual"),
        ("world_model_normalization", False),
        ("termination_mode", "learned"),
      ):
        values[key] = world_saved["config"].get(key, default)
    if not values["clips"]:
      entries = json.loads(args.manifest.read_text())["clips"]
      values["clips"] = tuple(
        Path(e["file"]).stem
        for e in entries
        if Path(e["file"]).stem.startswith(("walk", "run", "sprint"))
      )
      if not values["clips"]:
        raise ValueError("No locomotion clips found; supply --clips explicitly")
    if actor_saved and not set(values["clips"]).issubset(actor_saved["clips"]):
      raise ValueError("Initialization clips must be in the motion checkpoint")
    cfg = OnlineConfig(**values)
    velocity = VelocityConfig(
      warmup_updates=args.warmup_updates,
      stand_still_weight=args.stand_still_weight,
    )
    stage = args.stage
  timing_override = (
    args.command_hold_seconds is not None
    or velocity.randomize_initial_episode_progress
    != args.randomize_initial_episode_progress
  )
  if args.command_hold_seconds is not None:
    velocity = replace(
      velocity,
      resampling_time_range=tuple(args.command_hold_seconds),
    )
  velocity = replace(
    velocity,
    randomize_initial_episode_progress=args.randomize_initial_episode_progress,
  )
  if cfg.relative_orientation or cfg.motion_anchor_ori_b:
    raise ValueError("Velocity training requires proprioceptive observations")
  run = {
    "config": asdict(cfg),
    "velocity_config": asdict(velocity),
    "stage": stage,
    "total_iterations": args.iterations,
    "manifest": str(args.manifest.resolve()),
    "manifest_sha256": file_sha256(args.manifest),
    "timing_override_on_resume": bool(args.resume and timing_override),
  }
  args.output.mkdir(parents=True, exist_ok=bool(saved))
  write_json(args.output / "config.json", run)
  console = VelocityConsoleReporter(args.iterations, args.log_format)
  console.message(f"Initializing simulation; details: {args.output / 'simulator.log'}")
  metadata = robot_metadata(args.robot)
  with TrainingRuntime(args, args.output, console) as runtime:
    with simulator_output(runtime.sim_log, verbose=args.verbose_sim):
      env = runtime.env = VelocityEnv(
        FoundationConfig.training(
          robot=args.robot,
          manifest=args.manifest,
          clips=cfg.clips,
          num_envs=cfg.num_envs,
          seed=cfg.seed,
          episode_length_s=cfg.episode_length_s,
        ),
        velocity,
        args.device,
        motion_envs=cfg.num_envs // 2,
        training=True,
      )
      trainer = runtime.trainer = VelocityTrainer(
        env, metadata, cfg, total_iterations=args.iterations, stage=stage
      )
      if saved:
        trainer.load_state_dict(saved, allow_timing_override=timing_override)
      else:
        if actor_saved:
          validate_environment(env, actor_saved)
          trainer.actor.load_state_dict(actor_saved["actor"])
        if world_saved:
          validate_environment(env, world_saved)
          trainer.world.load_state_dict(world_saved["world"])
          trainer.world_pretrained = True
        if args.initial_evaluation_steps:
          from train_supermimic import evaluate_actor

          report = evaluate_actor(
            trainer.actor,
            args.manifest,
            cfg,
            args.device,
            args.initial_evaluation_steps,
          )
          write_json(args.output / "initial_motion_evaluation.json", report)
        trainer.bootstrap()

    console.start(
      asdict(cfg) | {"wm_only_iters": 0},
      args.device,
      len(env.library.names),
      args.output,
      trainer.iteration,
      table_note="Table: interval means | Fail%: failure events per transition",
    )
    console.message(
      f"Velocity: 10 x {env.foundation_config.observation_dim} observations | shared decoder | stage={stage} | "
      f"latent warmup={velocity.warmup_updates:,} updates"
    )
    console.message(
      "Reward/s: completed episode sum / configured seconds (higher) | Error: episode mean in physical units (lower) | "
      "Loss: model/critic/alignment fitting (lower) | Objective: imagined return (higher)"
    )
    holding = f"{velocity.resampling_time_range[0]:g}-{velocity.resampling_time_range[1]:g} s independently per environment"
    console.message(
      f"Commands: {holding} | random initial episode progress="
      f"{velocity.randomize_initial_episode_progress}\n"
      "Statistics: completed episodes only; rewards normalized by configured episode seconds"
    )

    tracking = runtime.start_tracking(run, reporter=VelocityWandbReporter)

    def save(historical=False):
      full = trainer.state_dict() | {
        "manifest_sha256": run["manifest_sha256"],
        "wandb_run": dict(tracking.state) if tracking else None,
      }
      save_checkpoint(args.output / "last.pt", full)
      save_checkpoint(
        args.output / "velocity_policy_last.pt",
        trainer.velocity_policy_state_dict(),
      )
      save_checkpoint(
        args.output / "motion_policy_last.pt",
        trainer.policy_state_dict() | {"manifest_sha256": run["manifest_sha256"]},
      )
      save_checkpoint(
        args.output / "world_model_last.pt",
        trainer.world_model_state_dict() | {"manifest_sha256": run["manifest_sha256"]},
      )

    starting_iteration = trainer.iteration
    runtime.train(
      VelocityMetricWindow,
      checkpoint=save,
      phase=lambda: trainer.training_stage,
      log_first=True,
    )
    if trainer.iteration == starting_iteration:
      save()
    diagnostics = trainer.velocity_diagnostics()
    write_json(args.output / "velocity_diagnostics.json", diagnostics)
    if tracking:
      tracking.velocity_diagnostics(diagnostics, trainer.iteration)
    console.message(
      f"Saved velocity policy and resumable training checkpoint to {args.output}"
    )


if __name__ == "__main__":
  main()
