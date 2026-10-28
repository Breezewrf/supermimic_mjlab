"""Resolve robot selection and variant-specific defaults at script boundaries."""

from .morphology import ROBOTS, robot_layout


def add_robot_argument(parser):
  parser.add_argument(
    "--robot",
    choices=ROBOTS,
    help="Robot variant (default: checkpoint variant, otherwise g1_29dof)",
  )


def resolve_robot_arguments(args, parser, *, default_manifest=True):
  import torch

  checkpoint = next(
    (
      getattr(args, key, None)
      for key in ("resume", "checkpoint", "world_model")
      if getattr(args, key, None)
    ),
    None,
  )
  if checkpoint is None and hasattr(args, "from_scratch") and not args.from_scratch:
    checkpoint = args.actor_checkpoint
  saved_robot = None
  if checkpoint is not None and checkpoint.is_file():
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    saved_robot = saved.get("config", {}).get("robot", "g1_29dof")
  if args.robot is not None and saved_robot is not None and args.robot != saved_robot:
    parser.error("--robot does not match checkpoint robot")
  args.robot = args.robot or saved_robot or "g1_29dof"
  if default_manifest and hasattr(args, "manifest") and args.manifest is None:
    args.manifest = robot_layout(args.robot).manifest
