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
  evaluation_summary,
  save_report,
  zero_policy,
)
from src.tasks.supermimic.rl.inference import load_actor, validate_environment


def print_motion(row: dict):
  status = "success" if row["success"] else row["failure_reason"] or "incomplete"
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
    "--policy",
    help="Python module:function accepting (history, future_reference); default: zero actions",
  )
  parser.add_argument(
    "--checkpoint",
    type=Path,
    help="Actor-only policy weights or a full online training checkpoint",
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
  from src.tasks.supermimic.env import SuperMimicEnv

  config = FoundationConfig(
    manifest=args.manifest,
    clips=tuple(args.clips),
    num_envs=1,
    seed=args.seed,
    episode_length_s=1e9,
    relative_orientation=bool(
      saved and saved["config"].get("relative_orientation", False)
    ),
  )
  env = SuperMimicEnv(config, device=args.device)
  displays = []
  try:
    if saved:
      validate_environment(env, saved)
    if args.viewer == "native" or args.video:
      from src.tasks.supermimic.rl.visualization import LiveViewer, VideoRecorder

      visual_options = {"ghost": not args.no_ghost, "global_ghost": args.global_ghost}
      if args.viewer == "native":
        displays.append(LiveViewer(env, speed=args.speed, **visual_options))
        print(
          "MuJoCo: solid = policy; blue ghost = reference. Space: pause; Right: step; G: ghost.",
          flush=True,
        )
      if args.video:
        displays.append(VideoRecorder(env, args.video, **visual_options))

    def on_frame(*frame):
      for display in displays:
        display(*frame)

    print(
      f"Policy: {args.checkpoint or args.policy or 'zero_policy (baseline)'}",
      flush=True,
    )
    if saved:
      print(f"Reference schema: {saved['reference_schema']}", flush=True)
    while True:
      try:
        report = evaluate(
          env,
          policy,
          args.max_steps_per_clip,
          on_frame=on_frame if displays else None,
          on_motion=print_motion,
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
