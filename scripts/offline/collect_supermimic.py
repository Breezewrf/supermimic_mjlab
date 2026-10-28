"""Collect model-ready transitions, or check kinematic reference replay.

Example: python scripts/offline/collect_supermimic.py --steps 128
"""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import torch

# Executing a script directly must not depend on editable package installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.tasks.supermimic.config.cli import add_robot_argument, resolve_robot_arguments
from src.tasks.supermimic.config.task_cfg import FoundationConfig
from src.tasks.supermimic.data.replay_buffer import ReplayBuffer


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  add_robot_argument(parser)
  parser.add_argument("--manifest", type=Path)
  parser.add_argument("--clips", nargs="*", default=[])
  parser.add_argument("--num-envs", type=int, default=4)
  parser.add_argument("--steps", type=int, default=128)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--capacity", type=int, default=1000)
  parser.add_argument(
    "--output", type=Path, default=Path("outputs/supermimic/foundation")
  )
  parser.add_argument(
    "--training",
    action="store_true",
    help="Enable return sampling, sensor noise and domain randomization",
  )
  parser.add_argument("--action-noise", type=float, default=0.1)
  parser.add_argument("--mode", choices=("collect", "replay"), default="collect")
  args = parser.parse_args()
  resolve_robot_arguments(args, parser)
  if args.steps < 1 or args.capacity < 10 or args.action_noise < 0:
    parser.error("steps must be positive, capacity >= 10, action-noise >= 0")
  if args.device.startswith("cuda") and not torch.cuda.is_available():
    parser.error(
      "CUDA is not visible to this Python process; check its GPU/device access"
    )
  from src.tasks.supermimic.env import SuperMimicEnv

  factory = FoundationConfig.training if args.training else FoundationConfig
  config = factory(
    robot=args.robot,
    manifest=args.manifest,
    clips=tuple(args.clips),
    num_envs=args.num_envs,
    seed=args.seed,
  )
  if args.output.exists():
    parser.error(f"Output exists: {args.output}; choose a new output directory")
  env = SuperMimicEnv(config, device=args.device)
  try:
    summary = {
      "config": asdict(config),
      "mode": args.mode,
      "device": args.device,
      "clips": env.library.names,
    }
    if args.mode == "replay":
      largest = 0.0
      for clip in range(len(env.library)):
        for frame in range(min(args.steps, int(env.library.lengths[clip]))):
          actual = env.replay_frame(clip, frame)
          reference = env.library.get_frame(
            torch.tensor([clip], device=env.device),
            torch.tensor([frame], device=env.device),
          )
          largest = max(
            largest,
            float((actual["body_pos_w"] - reference["body_pos_w"]).abs().max()),
          )
      summary["max_body_position_disagreement_m"] = largest
      if largest > 1e-4:
        raise RuntimeError(f"Reference/model kinematics disagree by {largest:g} metres")
      args.output.mkdir(parents=True)
    else:
      replay = ReplayBuffer(
        env.num_envs, args.capacity, device=args.device, seed=args.seed
      )
      action_generator = torch.Generator(device=args.device).manual_seed(args.seed + 1)
      failures = cutoffs = 0
      for _ in range(args.steps):
        action = (
          torch.randn(
            (env.num_envs, config.layout.num_joints),
            device=env.device,
            generator=action_generator,
          )
          * args.action_noise
        )
        _, _, terminated, truncated, extras = env.step(action)
        replay.append(extras["transition"])
        failures += int(terminated.sum())
        cutoffs += int(truncated.sum())
      # Fail early if the exported replay cannot supply both future interfaces.
      windows = replay.sample_windows(4, 8)
      starts = replay.sample_starts(4, 10)
      summary.update(
        {
          "transitions": len(replay),
          "physical_failures": failures,
          "cutoffs": cutoffs,
          "window_shape": list(windows["state"].shape),
          "valid_window_steps": int(windows["mask"].sum()),
          "history_shape": list(starts["history"].shape),
        }
      )
      args.output.mkdir(parents=True)
      torch.save(
        {
          "replay": replay.state_dict(),
          "sampler": env.sampler.state_dict(),
          "clips": env.library.names,
        },
        args.output / "replay.pt",
      )
    (args.output / "summary.json").write_text(
      json.dumps(summary, default=str, indent=2) + "\n"
    )
    print(json.dumps(summary, default=str, indent=2))
  finally:
    env.close()


if __name__ == "__main__":
  main()
