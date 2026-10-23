"""Evaluate a policy callback with fixed starts and no observation/physics noise."""

import argparse
import importlib
import math
import sys
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.tasks.supermimic.config.task_cfg import (
  DEFAULT_MANIFEST,
  FoundationConfig,
)
from src.tasks.supermimic.rl.evaluation import (
  EvaluationInterrupted,
  evaluate,
  evaluation_env_count,
  evaluation_summary,
  save_report,
  zero_policy,
)
from src.tasks.supermimic.rl.inference import (
  load_actor,
  load_world_model,
  validate_environment,
)


def print_motion(row: dict):
  status = (
    "success"
    if row["success"]
    else row["failure_reason"]
    or (
      "completed with failure thresholds crossed"
      if row.get("failure_detected")
      else "incomplete"
    )
  )
  print(
    f"  {row['motion']}: return {row['normalized_return']:.3f}"
    f" | {row['steps']}/{row['expected_steps']} steps | {status}",
    flush=True,
  )


def main(*, playback=False):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
  parser.add_argument("--clips", nargs="*", default=[])
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument(
    "--num-envs",
    type=int,
    help="Maximum parallel environments (headless: 64; play/viewer/video/world-model: 1)",
  )
  parser.add_argument(
    "--policy",
    help="Python module:function accepting (history, future_reference); default: zero actions",
  )
  parser.add_argument(
    "--checkpoint",
    type=Path,
    help="Actor-only policy weights or a full online training checkpoint",
  )
  parser.add_argument(
    "--world-model",
    "--world-model-checkpoint",
    type=Path,
    help="Show world-model predictions from exported world weights or a full checkpoint",
  )
  parser.add_argument(
    "--world-model-horizon",
    type=int,
    default=32,
    help="Future lead or comparison resync horizon in steps (default: 32)",
  )
  parser.add_argument(
    "--world-model-mode",
    choices=("future", "compare"),
    default="future",
    help="future: policy rollout ahead of physics (default); compare: same-time dynamics errors",
  )
  parser.add_argument(
    "--world-model-refresh",
    type=int,
    default=4,
    help="Recompute future previews every N control steps; cached trajectory advances between updates",
  )
  parser.add_argument(
    "--max-steps-per-clip",
    type=int,
    help="Debug only: mark unfinished clips as evaluation_limited",
  )
  parser.add_argument(
    "--output", type=Path, default=Path("outputs/supermimic/evaluation.json")
  )
  parser.add_argument("--csv", type=Path, help="Append scores to a comparison CSV")
  parser.add_argument(
    "--viewer",
    choices=("none", "native"),
    default="native" if playback else "none",
    help="Open a real-time MuJoCo window with the reference ghost",
  )
  parser.add_argument("--speed", type=float, default=1.0, help="Viewer playback speed")
  parser.add_argument(
    "--perturbations",
    action=argparse.BooleanOptionalAction,
    default=None,
    help="Allow mouse forces in the native viewer (play: on; evaluation: off)",
  )
  parser.add_argument(
    "--termination",
    action=argparse.BooleanOptionalAction,
    default=None,
    help="Reset on height/tilt failure (play: off; evaluation: on)",
  )
  parser.add_argument("--no-ghost", action="store_true")
  parser.add_argument(
    "--global-ghost",
    action="store_true",
    help="Show absolute reference XY/heading instead of tracking-reward alignment",
  )
  parser.add_argument(
    "--loop", action="store_true", help="Repeat the selected motions in the viewer"
  )
  parser.add_argument(
    "--video",
    type=Path,
    help="Export evaluation as MP4 (requires ffmpeg; headless: MUJOCO_GL=egl)",
  )
  args = parser.parse_args()
  visual = playback or args.viewer != "none" or args.video is not None
  single_env = visual or args.world_model is not None
  max_envs = args.num_envs if args.num_envs is not None else (1 if single_env else 64)
  if max_envs < 1:
    parser.error("--num-envs must be positive")
  if single_env and max_envs != 1:
    parser.error(
      "World-model comparison, playback, viewer and video require --num-envs 1"
    )
  if args.world_model_horizon < 1:
    parser.error("--world-model-horizon must be positive")
  if args.world_model_refresh < 1:
    parser.error("--world-model-refresh must be positive")
  if args.world_model and args.world_model_mode == "future" and not args.checkpoint:
    parser.error(
      "Future preview requires --checkpoint; custom --policy callbacks can use --world-model-mode compare"
    )
  termination = args.termination if args.termination is not None else not playback
  perturbations = (
    args.perturbations
    if args.perturbations is not None
    else playback and args.viewer == "native"
  )
  if perturbations and args.viewer != "native":
    parser.error("--perturbations requires --viewer native")
  if args.device.startswith("cuda") and not torch.cuda.is_available():
    parser.error("CUDA is not visible to this Python process")
  if args.policy and args.checkpoint:
    parser.error("Select either --policy or --checkpoint")
  if playback and not (args.policy or args.checkpoint):
    parser.error("Playback requires --checkpoint or --policy")
  if not math.isfinite(args.speed) or args.speed <= 0:
    parser.error("--speed must be positive and finite")
  if args.loop and (args.viewer == "none" or args.video):
    parser.error("--loop requires --viewer native and cannot be combined with --video")
  if args.max_steps_per_clip is not None and args.max_steps_per_clip < 1:
    parser.error("--max-steps-per-clip must be positive")
  if args.video and args.video.suffix.lower() != ".mp4":
    parser.error("--video must end in .mp4")
  paths = [p.resolve() for p in (args.output, args.csv, args.video) if p is not None]
  if len(set(paths)) != len(paths):
    parser.error("Report, CSV and video paths must be different")

  policy = zero_policy
  saved = None
  if args.policy:
    try:
      module, function = args.policy.split(":", 1)
    except ValueError:
      parser.error("--policy must be module:function")
    policy = getattr(importlib.import_module(module), function)
    if isinstance(policy, torch.nn.Module):
      policy = policy.to(args.device).eval()
  if args.checkpoint:
    try:
      policy, saved, clips = load_actor(
        args.checkpoint, args.manifest, args.device, tuple(args.clips)
      )
    except ValueError as error:
      parser.error(str(error))
    args.clips = list(clips)
  world = world_saved = None
  if args.world_model:
    try:
      world, world_saved = load_world_model(
        args.world_model, args.manifest, args.device, tuple(args.clips)
      )
    except ValueError as error:
      parser.error(str(error))
    if not args.clips:
      args.clips = list(world_saved["clips"])
  from src.tasks.supermimic.env import SuperMimicEnv

  config = FoundationConfig(
    manifest=args.manifest,
    clips=tuple(args.clips),
    num_envs=evaluation_env_count(args.manifest, tuple(args.clips), max_envs),
    seed=args.seed,
    episode_length_s=1e9,
    terminate_on_failure=termination,
    relative_orientation=bool(
      saved and saved["config"].get("relative_orientation", False)
    ),
  )
  env = SuperMimicEnv(config, device=args.device)
  displays = []
  live_viewer = None
  comparison = None
  try:
    if saved:
      validate_environment(env, saved)
    if world_saved:
      validate_environment(env, world_saved)
      from src.tasks.supermimic.rl.world_model_playback import (
        WorldModelFuturePreview,
        WorldModelPlayback,
      )

      if args.world_model_mode == "future":
        comparison = WorldModelFuturePreview(
          env, world, policy, args.world_model_horizon, args.world_model_refresh
        )
      else:
        comparison = WorldModelPlayback(env, world, args.world_model_horizon)
    observed_policy = comparison.policy(policy) if comparison is not None else policy
    if args.viewer == "native" or args.video:
      from src.tasks.supermimic.rl.visualization import LiveViewer, VideoRecorder

      visual_options = {
        "ghost": not args.no_ghost,
        "global_ghost": args.global_ghost,
        "world_model": comparison,
      }
      if args.viewer == "native":
        live_viewer = LiveViewer(
          env, speed=args.speed, perturbations=perturbations, **visual_options
        )
        displays.append(live_viewer)
        print(
          "MuJoCo: solid = policy; blue ghost = reference. Space: pause; Right: step; G: ghost.",
          flush=True,
        )
        if comparison is not None:
          print(
            "Orange ghost = world-model prediction; W: toggle prediction.", flush=True
          )
        if perturbations:
          print(
            "Mouse forces ON: double-click a robot body, then hold Ctrl and drag. Release to stop.",
            flush=True,
          )
      if args.video:
        displays.append(VideoRecorder(env, args.video, **visual_options))

    def on_frame(*frame):
      if comparison is not None:
        comparison.on_frame(*frame)
      for display in displays:
        display(*frame)

    print(
      f"Policy: {args.checkpoint or args.policy or 'zero_policy (baseline)'}",
      flush=True,
    )
    if saved:
      print(f"Reference schema: {saved['reference_schema']}", flush=True)
    if world_saved:
      print(
        f"World model: {args.world_model} (iteration {world_saved.get('iteration', 'unknown')}); "
        f"{comparison.description()}. External forces are not model inputs.",
        flush=True,
      )
    if not termination:
      print(
        "Failure termination OFF: height/tilt failures do not reset playback; clips still end normally.",
        flush=True,
      )
    while True:
      if live_viewer:
        live_viewer.begin_evaluation()
      if comparison is not None:
        comparison.begin_evaluation()
      try:
        report = evaluate(
          env,
          observed_policy,
          args.max_steps_per_clip,
          on_frame=on_frame if displays or comparison is not None else None,
          on_motion=print_motion,
          before_step=live_viewer.before_step if live_viewer else None,
        )
      except EvaluationInterrupted:
        print("Viewer closed; the interrupted evaluation was not saved.", flush=True)
        break
      report.update(
        policy=str(args.checkpoint)
        if args.checkpoint
        else args.policy or "zero_policy",
        iteration=saved.get("iteration") if saved else None,
        training_environment_steps=saved.get("environment_steps") if saved else None,
        evaluated_at=datetime.now(UTC).isoformat(),
        evaluation_limit=args.max_steps_per_clip,
        reference_schema=saved.get("reference_schema") if saved else None,
        reference_schema_inferred=bool(
          saved and saved.get("reference_schema_inferred")
        ),
      )
      perturbed_steps = (
        live_viewer.mouse.applied_steps if live_viewer and live_viewer.mouse else 0
      )
      report["mouse_perturbations_enabled"] = bool(perturbations)
      report["mouse_perturbation_steps"] = perturbed_steps
      if perturbed_steps:
        report["evaluation_mode"] = "interactive"
      if perturbed_steps:
        report["protocol"] += "; interactive mouse forces applied"
      if comparison is not None:
        report["world_model_comparison"] = {
          "checkpoint": str(args.world_model),
          "iteration": world_saved.get("iteration"),
          **comparison.report(),
        }
      save_report(report, args.output, args.csv)
      print(evaluation_summary(report), flush=True)
      print(f"Report: {args.output}", flush=True)
      if not args.loop:
        break
  finally:
    with ExitStack() as cleanup:
      cleanup.callback(env.close)
      for display in displays:
        cleanup.callback(display.close)


if __name__ == "__main__":
  main()
