"""Evaluate a policy callback with fixed starts and no observation/physics noise."""

import argparse
import importlib
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.tasks.supermimic.config.task_cfg import (
  DEFAULT_MANIFEST,
  FoundationConfig,
)
from src.tasks.supermimic.data.world_model_data import file_sha256
from src.tasks.supermimic.models.policy import MotionActor
from src.tasks.supermimic.rl.evaluation import evaluate, zero_policy


def main():
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
    "--checkpoint", type=Path, help="Online SuperMimic actor checkpoint"
  )
  parser.add_argument(
    "--max-steps-per-clip",
    type=int,
    help="Debug only: mark unfinished clips as evaluation_limited",
  )
  parser.add_argument(
    "--output", type=Path, default=Path("outputs/supermimic/evaluation.json")
  )
  args = parser.parse_args()
  if args.device.startswith("cuda") and not torch.cuda.is_available():
    parser.error("CUDA is not visible to this Python process")
  if args.policy and args.checkpoint:
    parser.error("Select either --policy or --checkpoint")
  from src.tasks.supermimic.env import SuperMimicEnv

  policy = zero_policy
  if args.policy:
    module, function = args.policy.split(":", 1)
    policy = getattr(importlib.import_module(module), function)
  if args.checkpoint:
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if saved.get("kind") != "supermimic_online" or saved.get("schema_version") != 1:
      parser.error("Expected an online SuperMimic checkpoint")
    if saved["manifest_sha256"] != file_sha256(args.manifest):
      parser.error("Checkpoint manifest does not match")
    if args.clips and not set(args.clips).issubset(saved["clips"]):
      parser.error("Evaluation clips must be present in the checkpoint")
    if not args.clips:
      args.clips = saved["clips"]
    policy = MotionActor(
      saved["robot_metadata"]["default_q"],
      tuple(saved["config"]["actor_dims"]),
      saved["config"].get("relative_orientation", False),
    )
    policy.load_state_dict(saved["actor"])
    policy = policy.to(args.device).eval()
  config = FoundationConfig(
    manifest=args.manifest,
    clips=tuple(args.clips),
    num_envs=1,
    seed=args.seed,
    episode_length_s=1e9,
    relative_orientation=bool(
      args.checkpoint and saved["config"].get("relative_orientation", False)
    ),
  )
  env = SuperMimicEnv(config, device=args.device)
  try:
    if args.checkpoint:
      from src.tasks.supermimic.models.kinematics import robot_metadata
      from src.tasks.supermimic.rl.online_training import simulation_parameters

      if saved["robot_metadata"]["signature"] != robot_metadata()["signature"]:
        raise ValueError("Checkpoint robot model does not match the current G1")
      if saved.get("simulation") != simulation_parameters(env):
        raise ValueError("Checkpoint simulation configuration does not match")
    report = evaluate(env, policy, args.max_steps_per_clip)
    report["policy"] = (
      str(args.checkpoint) if args.checkpoint else args.policy or "zero_policy"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
  finally:
    env.close()


if __name__ == "__main__":
  main()
