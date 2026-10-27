"""Evaluate velocity exports from standing without loading any reference motion."""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tasks.supermimic.config.task_cfg import FoundationConfig
from src.tasks.supermimic.config.velocity_cfg import VelocityConfig
from src.tasks.supermimic.rl.inference import validate_environment
from src.tasks.supermimic.rl.training_logger import simulator_output
from src.tasks.supermimic.rl.velocity_evaluation import evaluate_velocity
from src.tasks.supermimic.rl.velocity_inference import load_velocity_actor
from src.tasks.supermimic.velocity_env import VelocityEnv


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--num-envs", type=int, default=64)
  parser.add_argument("--trials", type=int, default=4)
  parser.add_argument("--duration", type=float, default=5.0)
  parser.add_argument("--steady-seconds", type=float, default=2.0)
  parser.add_argument("--full-grid", action="store_true")
  parser.add_argument(
    "--output", type=Path, default=Path("outputs/velocity/evaluation.json")
  )
  args = parser.parse_args()
  torch.set_num_threads(4)
  actor, saved = load_velocity_actor(args.checkpoint, args.device)
  args.output.parent.mkdir(parents=True, exist_ok=True)
  env = None
  try:
    with (
      args.output.with_suffix(".simulator.log").open("w") as sim_log,
      simulator_output(sim_log, verbose=False),
    ):
      env = VelocityEnv(
        FoundationConfig(num_envs=args.num_envs, episode_length_s=1e9),
        VelocityConfig(**saved["velocity_config"]),
        args.device,
      )
      validate_environment(env, saved)
      report = evaluate_velocity(
        env,
        actor,
        full=args.full_grid,
        trials=args.trials,
        duration=args.duration,
        steady_seconds=args.steady_seconds,
      )
    report.update(
      checkpoint=str(args.checkpoint.resolve()), iteration=saved["iteration"]
    )
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Velocity fall rate: {report['fall_rate']:.1%}; report: {args.output}")
  finally:
    if env is not None:
      env.close()


if __name__ == "__main__":
  main()
