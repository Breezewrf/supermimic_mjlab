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

from src.tasks.supermimic.config.cli import (
  add_reset_sampling_arguments,
  add_robot_argument,
  resolve_robot_arguments,
)
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
  add_reset_sampling_arguments(p)
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
  p.add_argument(
    "--freeze-world-model",
    action=argparse.BooleanOptionalAction,
    default=None,
    help="Keep pretrained world weights and normalization fixed (new runs: disabled)",
  )
  p.add_argument("--from-scratch", action="store_true")
  p.add_argument(
    "--self-collisions",
    action=argparse.BooleanOptionalAction,
    default=None,
    help="Enable G1 self contacts and Unitree exclusions (new runs: enabled)",
  )
  p.add_argument("--resume", type=Path)
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--output", type=Path)
  p.add_argument(
    "--stage",
    choices=("motion", "warmup", "joint"),
    help="Training stage (new runs: joint)",
  )
  p.add_argument("--iterations", type=int, default=100000)
  p.add_argument(
    "--velocity-config", type=Path, help="JSON velocity settings for a new run"
  )
  p.add_argument(
    "--warmup-updates",
    type=int,
    help="Latent updates before joint training (default: 500)",
  )
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
  p.add_argument(
    "--stand-still-weight", type=float, help="Stand-still pose penalty (default: 0)"
  )
  p.add_argument(
    "--stand-still-reference",
    choices=("motionbrick", "unitree"),
    help="Stand-still target (new runs: motionbrick; resume: saved setting)",
  )
  p.add_argument(
    "--self-collision-weight",
    type=float,
    help="Self-collision penalty weight (default: -1; must be nonpositive)",
  )
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
    if args.velocity_config is not None:
      argument_parser.error("--resume uses the saved velocity config")
    if args.output.resolve() != args.resume.resolve().parent:
      argument_parser.error("Resume into the checkpoint's existing output directory")
  elif args.output.exists():
    argument_parser.error("Output exists; choose a new directory or use --resume")
  if not args.resume and args.from_scratch and args.freeze_world_model:
    argument_parser.error("--freeze-world-model requires a pretrained world checkpoint")
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
    restored = asdict(OnlineConfig()) | saved["config"]
    for key in ("motion_storage", "motion_fk_cache_frames", "reset_softmax_cap"):
      if getattr(args, key) is not None and getattr(args, key) != restored[key]:
        argument_parser.error(f"Resume requires original {key}")
    cfg = OnlineConfig(**restored)
    if args.self_collisions is not None and args.self_collisions != cfg.self_collisions:
      argument_parser.error("Resume requires the saved self-collision setting")
    if args.self_collision_weight is not None and args.self_collision_weight != saved[
      "velocity_config"
    ].get("self_collision_weight", -1.0):
      argument_parser.error("Resume requires the saved self-collision weight")
    velocity = VelocityConfig(**saved["velocity_config"])
    if (
      args.stand_still_reference is not None
      and args.stand_still_reference != velocity.stand_still_reference
    ):
      argument_parser.error("Resume requires the saved stand-still reference")
    stage = saved["stage"]
    freeze_world_model = saved.get("freeze_world_model", False)
    if args.stage is not None and args.stage != stage:
      argument_parser.error("Resume requires the saved training stage")
    if (
      args.freeze_world_model is not None
      and args.freeze_world_model != freeze_world_model
    ):
      argument_parser.error("Resume requires the saved world model freeze setting")
  else:
    names = {f.name for f in fields(OnlineConfig)}
    values = {
      key: value
      for key, value in vars(args).items()
      if key in names and value is not None
    }
    for key in ("clips", "actor_dims", "world_dims", "critic_dims"):
      values[key] = tuple(values[key])
    values.update(
      amp="none",
      reference_dim=robot_layout(args.robot).full_body_reference_dim,
      actor_input_normalization=True,
      critic_input_normalization=True,
      world_model_normalization=True,
      self_collisions=True if args.self_collisions is None else args.self_collisions,
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
    cfg = OnlineConfig(**values)
    try:
      settings = (
        json.loads(args.velocity_config.read_text()) if args.velocity_config else {}
      )
      if not isinstance(settings, dict):
        raise TypeError("Expected a JSON object")
      settings.setdefault("stand_still_reference", "motionbrick")
      for key in (
        "warmup_updates",
        "stand_still_weight",
        "self_collision_weight",
        "stand_still_reference",
      ):
        if getattr(args, key) is not None:
          settings[key] = getattr(args, key)
      velocity = VelocityConfig(**settings)
    except (OSError, ValueError, TypeError) as error:
      argument_parser.error(f"Invalid velocity configuration: {error}")
    stage = args.stage or "joint"
    freeze_world_model = bool(args.freeze_world_model)
    if (
      freeze_world_model
      and cfg.self_collisions
      and world_saved is not None
      and not world_saved.get("simulation", {}).get("self_collisions", False)
    ):
      argument_parser.error(
        "Enabling self-collisions from a non-collision world checkpoint requires "
        "online adaptation (--no-freeze-world-model)"
      )
  randomize_progress = args.randomize_initial_episode_progress
  if randomize_progress is None:
    randomize_progress = (
      saved is not None or velocity.randomize_initial_episode_progress
    )
  timing_override = (
    args.command_hold_seconds is not None
    or velocity.randomize_initial_episode_progress != randomize_progress
  )
  if args.command_hold_seconds is not None:
    velocity = replace(
      velocity,
      resampling_time_range=tuple(args.command_hold_seconds),
    )
  velocity = replace(
    velocity,
    randomize_initial_episode_progress=randomize_progress,
  )
  if cfg.relative_orientation or cfg.motion_anchor_ori_b:
    raise ValueError("Velocity training requires proprioceptive observations")
  run = {
    "config": asdict(cfg),
    "velocity_config": asdict(velocity),
    "stage": stage,
    "freeze_world_model": freeze_world_model,
    "total_iterations": args.iterations,
    "manifest": str(args.manifest.resolve()),
    "manifest_sha256": file_sha256(args.manifest),
    "timing_override_on_resume": bool(args.resume and timing_override),
    "tensorboard": args.tensorboard,
    "initialization_collision_transfer": {
      key: bool(
        checkpoint is not None
        and cfg.self_collisions
        and not checkpoint.get("simulation", {}).get("self_collisions", False)
      )
      for key, checkpoint in (("actor", actor_saved), ("world", world_saved))
    }
    if saved is None
    else saved.get("initialization_collision_transfer", {}),
  }
  args.output.mkdir(parents=True, exist_ok=bool(saved))
  write_json(args.output / "config.json", run)
  console = VelocityConsoleReporter(args.iterations, args.log_format)
  console.message(f"Initializing simulation; details: {args.output / 'simulator.log'}")
  metadata = robot_metadata(args.robot, self_collisions=cfg.self_collisions)
  initial_motion_report = None
  with TrainingRuntime(args, args.output, console) as runtime:
    with simulator_output(runtime.sim_log, verbose=args.verbose_sim):
      env = runtime.env = VelocityEnv(
        FoundationConfig.training(
          robot=args.robot,
          self_collisions=cfg.self_collisions,
          manifest=args.manifest,
          motion_storage=cfg.motion_storage,
          motion_fk_cache_frames=cfg.motion_fk_cache_frames,
          clips=cfg.clips,
          num_envs=cfg.num_envs,
          seed=cfg.seed,
          episode_length_s=cfg.episode_length_s,
          reset_softmax_cap=cfg.reset_softmax_cap,
        ),
        velocity,
        args.device,
        motion_envs=cfg.num_envs // 2,
        training=True,
      )
      trainer = runtime.trainer = VelocityTrainer(
        env,
        metadata,
        cfg,
        total_iterations=args.iterations,
        stage=stage,
        freeze_world_model=freeze_world_model,
      )
      trainer.initialization_collision_transfer = run[
        "initialization_collision_transfer"
      ]
      if saved:
        trainer.load_state_dict(saved, allow_timing_override=timing_override)
      else:
        if actor_saved:
          validate_environment(env, actor_saved, allow_collision_transfer=True)
          trainer.actor.load_state_dict(actor_saved["actor"])
        if world_saved:
          transfer = validate_environment(
            env, world_saved, allow_collision_transfer=True
          )
          if transfer:
            incompatible = trainer.world.load_state_dict(
              world_saved["world"], strict=False
            )
            expected = {
              key
              for key in trainer.world.state_dict()
              if key.startswith("collision_head.")
            }
            if (
              set(incompatible.missing_keys) != expected or incompatible.unexpected_keys
            ):
              raise ValueError(
                "Pretrained world weights differ beyond the collision head"
              )
          else:
            trainer.world.load_state_dict(world_saved["world"])
          trainer.world_pretrained = True
        if args.initial_evaluation_steps:
          from train_supermimic import evaluate_actor

          initial_motion_report = evaluate_actor(
            trainer.actor,
            args.manifest,
            cfg,
            args.device,
            args.initial_evaluation_steps,
            library=env.library,
          )
          write_json(
            args.output / "initial_motion_evaluation.json", initial_motion_report
          )
        trainer.bootstrap()

    console.start(
      asdict(cfg) | {"wm_only_iters": 0},
      args.device,
      len(env.library.names),
      args.output,
      trainer.iteration,
      table_note="Table: interval means | Fail%: failure events per transition",
    )
    world_updates = "frozen" if freeze_world_model else "online"
    console.message(
      f"Self collisions: {'ON' if cfg.self_collisions else 'OFF'} | "
      f"penalty weight={velocity.self_collision_weight}"
    )
    console.message(
      f"Velocity: 10 x {env.foundation_config.observation_dim} observations | shared decoder | stage={stage} | "
      f"latent warmup={velocity.warmup_updates:,} updates | world model={world_updates}"
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
    if runtime.tensorboard and initial_motion_report is not None:
      runtime.tensorboard.evaluation(initial_motion_report, trainer.iteration)

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
