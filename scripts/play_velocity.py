"""Reference-free velocity playback. WASD: translation; Q/E: turn; space: stop.

Mouse forces: double-click a body, then Ctrl-drag to perturb it.
Fall resets are disabled by default; --termination enables them.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from queue import SimpleQueue

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tasks.supermimic.config.task_cfg import FoundationConfig
from src.tasks.supermimic.config.velocity_cfg import VelocityConfig
from src.tasks.supermimic.mdp.velocity import measured_velocity
from src.tasks.supermimic.rl.inference import validate_environment
from src.tasks.supermimic.rl.velocity_inference import load_velocity_actor
from src.tasks.supermimic.velocity_env import VelocityEnv


@torch.no_grad()
def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--command", type=float, nargs=3, default=(0.0, 0.0, 0.0))
  parser.add_argument("--headless", action="store_true")
  parser.add_argument(
    "--termination",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Reset after falling (default: disabled during playback)",
  )
  parser.add_argument(
    "--perturbations",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Apply native viewer mouse forces to physics (default: enabled)",
  )
  parser.add_argument(
    "--steps",
    type=int,
    help="Control-step limit (default: until window closes; headless: 1000)",
  )
  args = parser.parse_args()
  if args.steps is not None and args.steps < 1:
    raise ValueError("steps must be positive")
  step_limit = (
    args.steps if args.steps is not None else (1000 if args.headless else None)
  )
  torch.set_num_threads(4)
  actor, saved = load_velocity_actor(args.checkpoint, args.device)
  env = viewer = passive = mouse = None
  try:
    env = VelocityEnv(
      FoundationConfig(
        num_envs=1, episode_length_s=1e9, terminate_on_failure=args.termination
      ),
      VelocityConfig(**saved["velocity_config"]),
      args.device,
    )
    validate_environment(env, saved)
    command = torch.tensor(args.command, device=args.device)
    env.set_command(command)
    keys = SimpleQueue()
    if not args.headless:
      from src.tasks.supermimic.rl.visualization import (
        EvaluationScene,
        MousePerturbations,
        PassiveViewer,
      )

      scene = EvaluationScene(env, ghost=False)
      scene.update(env, 0, 0, env.read_state())
      if args.perturbations:
        mouse = MousePerturbations(env, scene.model, scene.data)
      passive = PassiveViewer(scene.model, scene.data, key_callback=keys.put)
      viewer = passive.handle
      with viewer.lock():
        viewer.cam.distance = scene.camera.distance
        viewer.cam.azimuth = scene.camera.azimuth
        viewer.cam.elevation = scene.camera.elevation
      print(__doc__, flush=True)
      print(f"Mouse forces: {'ON' if mouse is not None else 'OFF'}", flush=True)
    failures = 0
    was_fallen = False
    step = 0
    while step_limit is None or step < step_limit:
      start = time.perf_counter()
      if viewer is not None and not viewer.is_running():
        break
      while not keys.empty():
        key = keys.get()
        changes = {
          87: (0, 0.1),
          83: (0, -0.1),
          65: (1, 0.1),
          68: (1, -0.1),
          81: (2, 0.2),
          69: (2, -0.2),
        }
        if key == 32:
          command.zero_()
        elif key in changes:
          axis, delta = changes[key]
          command[axis] += delta
        limits = command.new_tensor(saved["velocity_config"]["final_ranges"])
        command = command.clamp(limits[:, 0], limits[:, 1])
        env.set_command(command)
        print("command", command.tolist(), flush=True)
      action = actor(env.history.data, env.command_state["command"])
      if mouse is not None:
        with viewer.lock():
          mouse.apply(viewer.perturb)
      _, _, terminated, _, extras = env.step(action)
      step += 1
      fallen = bool(extras["metrics"]["velocity/physical_failure"].any())
      reset = bool(terminated.any())
      failures += int(fallen and not was_fallen)
      was_fallen = fallen and not reset
      if viewer is not None:
        if reset and mouse is not None:
          with viewer.lock():
            mouse.clear()
            # A drag target from before the fall must not pull the reset pose.
            viewer.perturb.active = 0
            viewer.perturb.active2 = 0
        scene.update(
          env,
          0,
          step,
          env.read_state() if reset else extras["transition"]["next_state"],
        )
        with viewer.lock():
          viewer.cam.lookat[:] = scene.camera.lookat
        viewer.sync()
        time.sleep(max(0.0, env.step_dt - (time.perf_counter() - start)))
    print(
      json.dumps(
        {
          "steps": step,
          "falls": failures,
          "command": command.tolist(),
          "measured_velocity": measured_velocity(env.read_state())[0].tolist(),
          "motion_library_loaded": env.library is not None,
          "failure_termination_enabled": args.termination,
          "mouse_perturbations_enabled": mouse is not None,
          "mouse_perturbed_steps": mouse.applied_steps if mouse is not None else 0,
        }
      )
    )
  finally:
    try:
      try:
        if mouse is not None:
          mouse.clear()
      finally:
        if passive is not None:
          passive.close()
    finally:
      if env is not None:
        env.close()


if __name__ == "__main__":
  main()
