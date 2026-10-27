"""Train SuperMimic online from scratch using simulator data and imagined BPTT."""

import argparse
import importlib.metadata
import sys
import time
from dataclasses import asdict, fields
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.tasks.supermimic.config.cli import add_robot_argument, resolve_robot_arguments
from src.tasks.supermimic.config.rl_cfg import OnlineConfig
from src.tasks.supermimic.config.task_cfg import (
  FoundationConfig,
)
from src.tasks.supermimic.data.world_model_data import file_sha256
from src.tasks.supermimic.models.kinematics import robot_metadata
from src.tasks.supermimic.models.policy import schema_for_dimension
from src.tasks.supermimic.rl.checkpoints import (
  evaluation_score,
  evaluation_summary,
  policy_checkpoint,
  restore_best_inference,
  save_checkpoint,
  save_inference_pair,
  training_config,
  world_model_checkpoint,
)
from src.tasks.supermimic.rl.evaluation import evaluate, evaluation_env_count
from src.tasks.supermimic.rl.online_training import (
  OnlineTrainer,
  restore_rng,
  rng_state,
)
from src.tasks.supermimic.rl.training_logger import (
  ConsoleReporter,
  MetricWindow,
  simulator_output,
)
from src.tasks.supermimic.rl.training_runner import (
  TrainingRuntime,
  add_logging_arguments,
  validate_logging_arguments,
  write_json,
)


def evaluate_actor(actor, manifest, config, device, limit=None, max_envs=64):
  from src.tasks.supermimic.env import SuperMimicEnv

  saved_rng, training = rng_state(), actor.training
  env = None
  try:
    actor.eval()
    env = SuperMimicEnv(
      FoundationConfig(
        robot=config.robot,
        manifest=manifest,
        clips=config.clips,
        num_envs=evaluation_env_count(manifest, config.clips, max_envs),
        seed=config.seed,
        episode_length_s=1e9,
        relative_orientation=config.relative_orientation,
        motion_anchor_ori_b=config.motion_anchor_ori_b,
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
  add_robot_argument(parser)
  parser.add_argument("--manifest", type=Path)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--output", type=Path)
  parser.add_argument("--resume", type=Path)
  parser.add_argument(
    "--iterations",
    type=int,
    default=1000,
    help="Total update rounds and cosine LR decay horizon, including WM-only warmup and saved rounds",
  )
  parser.add_argument("--eval-interval", type=int, default=100)
  parser.add_argument(
    "--eval-num-envs",
    type=int,
    default=64,
    help="Maximum parallel evaluation environments (capped by selected clips; 1: serial)",
  )
  parser.add_argument(
    "--checkpoint-interval",
    type=int,
    default=100,
    help="Save last.pt for resume and retain paired policy/world-model weights every N rounds",
  )
  parser.add_argument(
    "--full-checkpoint-interval",
    type=int,
    default=0,
    help="Additionally keep historical full iteration_NNNNNN.pt snapshots (0: disabled)",
  )
  add_logging_arguments(parser)
  parser.add_argument("--termination-mode", choices=("learned", "analytic"))
  parser.add_argument("--amp", choices=("bf16", "none"))
  parser.add_argument(
    "--world-model-normalization",
    action=argparse.BooleanOptionalAction,
    default=None,
    help="Enable SiLU + LayerNorm and running world-model input/output normalization",
  )
  parser.add_argument(
    "--world-prediction",
    choices=("state_delta", "velocity_residual"),
    help="World-model targets (new runs: state_delta; resume: saved mode)",
  )
  for name in ("actor-input-normalization", "critic-input-normalization"):
    parser.add_argument(
      "--" + name,
      action=argparse.BooleanOptionalAction,
      default=None,
      help="Enable RunningNorm for the selected network's inputs (default: disabled)",
    )
  parser.add_argument(
    "--reference-dim",
    type=int,
    choices=(610, 730, 2560, 2680),
    help="Ten-frame reference width: compact 610/730 (23/29DoF, default) or full-body 2560/2680",
  )
  parser.add_argument(
    "--learning-rate",
    type=float,
    help="Initial Adam learning rate for all three optimizers (default: 2e-4)",
  )
  parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=None)
  parser.add_argument(
    "--wm-only-iters",
    type=int,
    help="Initial rounds that collect and train only the world model (default: 0)",
  )
  parser.add_argument(
    "--relative-orientation",
    action=argparse.BooleanOptionalAction,
    default=None,
  )
  parser.add_argument(
    "--motion-anchor-ori-b",
    action=argparse.BooleanOptionalAction,
    default=None,
    help="Observe reference torso orientation in the robot torso frame (default: disabled)",
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
  resolve_robot_arguments(args, parser)
  validate_logging_arguments(parser, args)
  if min(
    args.iterations,
    args.eval_interval,
    args.eval_num_envs,
    args.checkpoint_interval,
    args.log_interval,
    args.cpu_threads,
  ) < 1 or (args.evaluation_limit is not None and args.evaluation_limit < 1):
    parser.error("Counts must be positive")
  if args.full_checkpoint_interval < 0:
    parser.error("--full-checkpoint-interval must be nonnegative")
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
    parser.error(
      "--resume requires a full online training checkpoint (last.pt); policy weights cannot resume training"
    )
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
  try:
    saved_config = training_config(saved) if saved else None
  except ValueError as error:
    parser.error(str(error))
  config_values = (
    dict(saved_config) if saved else asdict(OnlineConfig(robot=args.robot))
  )
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
  if saved and any(
    getattr(config, name) != saved_config[name]
    for name in ("actor_input_normalization", "critic_input_normalization")
  ):
    parser.error(
      "Resume cannot change input normalization; start a new run to enable or disable RunningNorm"
    )
  if saved and config.reference_dim != saved_config["reference_dim"]:
    parser.error(
      "Resume cannot change --reference-dim; use a new output directory for a new architecture"
    )
  if saved and config.world_prediction != saved_config["world_prediction"]:
    parser.error(
      "Resume cannot change --world-prediction; start a new run for state_delta training"
    )
  if (
    saved
    and config.world_model_normalization != saved_config["world_model_normalization"]
  ):
    parser.error(
      "Resume cannot change --world-model-normalization; start a new run for this ablation"
    )
  if saved and (
    saved_config != asdict(config) or saved["manifest_sha256"] != manifest_hash
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
  with TrainingRuntime(args, output, console) as runtime:
    sim_log = runtime.sim_log
    with simulator_output(sim_log, args.verbose_sim):
      from src.tasks.supermimic.env import SuperMimicEnv

      env = runtime.env = SuperMimicEnv(
        FoundationConfig.training(
          robot=args.robot,
          manifest=args.manifest,
          num_envs=config.num_envs,
          clips=config.clips,
          seed=config.seed,
          episode_length_s=config.episode_length_s,
          relative_orientation=config.relative_orientation,
          motion_anchor_ori_b=config.motion_anchor_ori_b,
        ),
        device=args.device,
      )
      trainer = runtime.trainer = OnlineTrainer(
        env,
        robot_metadata(args.robot),
        config,
        total_iterations=args.iterations,
      )
      if saved:
        trainer.load_state_dict(saved)
      else:
        trainer.bootstrap()
    runtime_versions = {
      name: importlib.metadata.version(name)
      for name in ("torch", "mujoco", "mujoco-warp", "warp-lang")
    }
    run_config = {
      "config": asdict(config),
      "total_iterations": args.iterations,
      "device": args.device,
      "manifest": str(args.manifest.resolve()),
      "manifest_sha256": manifest_hash,
      "runtime": runtime_versions,
      "robot_signature": trainer.metadata["signature"],
      "reference_schema": schema_for_dimension(config.reference_dim),
      "execution": {
        "compiled": trainer.kernels.enabled,
        "mlp_precision": config.amp if args.device.startswith("cuda") else "fp32",
        "physics_precision": "fp32",
      },
      "simulation": trainer.simulation_parameters,
      "evaluation_limit": args.evaluation_limit,
      "evaluation_max_envs": args.eval_num_envs,
      "full_checkpoint_interval": args.full_checkpoint_interval,
      "best_checkpoint_selection": "successes_then_normalized_return",
      "resume": str(args.resume) if args.resume else None,
    }
    write_json(output / "config.json", run_config)
    console.start(
      asdict(config), args.device, len(env.library), output, trainer.iteration
    )
    console.message(f"World-model prediction: {config.world_prediction}")
    if config.world_model_normalization:
      console.message(
        "World model: SiLU + LayerNorm | running input/output statistics from real collection"
      )
    tracking = runtime.start_tracking(run_config)
    best_evaluation = None
    current_evaluation = saved.get("evaluation") if saved else None
    if saved:
      best_evaluation, best_world_available = restore_best_inference(
        output, trainer.iteration, saved.get("best_evaluation")
      )
      if best_evaluation is not None:
        console.message(
          f"Best retained policy: iteration {best_evaluation['iteration']:,}, "
          f"success {best_evaluation['successes']}/{best_evaluation['motion_count']} "
          "(success count first, then return)"
        )
        if not best_world_available:
          console.message(
            "Historical best world model is unavailable; see best_checkpoint.json. "
            "Future best checkpoints will save both components."
          )

    def checkpoint_metadata():
      return {
        "manifest_sha256": manifest_hash,
        "runtime": runtime_versions,
        "best_return": (
          best_evaluation["normalized_return"] if best_evaluation else -float("inf")
        ),
        "best_evaluation": best_evaluation,
        "evaluation": (
          current_evaluation
          if current_evaluation and current_evaluation["iteration"] == trainer.iteration
          else None
        ),
        "evaluation_limit": args.evaluation_limit,
      }

    def checkpoint():
      return {
        **trainer.state_dict(),
        **checkpoint_metadata(),
        "wandb_run": dict(tracking.state) if tracking else None,
      }

    def save_weights(tag):
      metadata = checkpoint_metadata()
      save_inference_pair(
        output,
        policy_checkpoint({**trainer.policy_state_dict(), **metadata}),
        world_model_checkpoint({**trainer.world_model_state_dict(), **metadata}),
        tag=tag,
      )

    def validate():
      nonlocal best_evaluation, current_evaluation
      console.message(f"Evaluating policy at iteration {trainer.iteration:,}...")
      evaluation_started = time.perf_counter()
      with simulator_output(sim_log, args.verbose_sim):
        report = evaluate_actor(
          trainer.actor,
          args.manifest,
          config,
          args.device,
          args.evaluation_limit,
          max_envs=args.eval_num_envs,
        )
      report["evaluation_wall_seconds"] = time.perf_counter() - evaluation_started
      report["iteration"] = trainer.iteration
      report["training_environment_steps"] = trainer.environment_steps
      report["evaluation_limit"] = args.evaluation_limit
      write_json(output / f"evaluation_{trainer.iteration:06d}.json", report)
      write_json(output / "evaluation.json", report)
      diagnostics = trainer.diagnostics()
      write_json(output / f"diagnostics_{trainer.iteration:06d}.json", diagnostics)
      write_json(output / "diagnostics.json", diagnostics)
      current_evaluation = evaluation_summary(report)
      if best_evaluation is None or evaluation_score(report) > evaluation_score(
        best_evaluation
      ):
        best_evaluation = current_evaluation
        save_weights("best")
      console.evaluation(report)
      console.message(
        f"Evaluation: {report['evaluation_num_envs']} environments, "
        f"{report['evaluation_wall_seconds']:.1f}s including initialization"
      )
      if tracking:
        tracking.evaluation(report)
        tracking.diagnostics(diagnostics)

    if not saved:
      validate()
      save_checkpoint(output / "last.pt", checkpoint())
    elif best_evaluation is None:
      # A checkpoint copied to a new directory may have no retained best weights.
      validate()
    save_weights("last")
    if args.device.startswith("cuda"):
      torch.cuda.reset_peak_memory_stats(args.device)

    def save(historical):
      snapshot = checkpoint()
      save_checkpoint(output / "last.pt", snapshot)
      if historical:
        save_checkpoint(output / f"iteration_{trainer.iteration:06d}.pt", snapshot)
      save_weights("last")

    runtime.train(MetricWindow, checkpoint=save, evaluate=validate)


if __name__ == "__main__":
  main()
