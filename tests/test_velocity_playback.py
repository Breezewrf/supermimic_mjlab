"""Velocity playback forwards mouse forces to real physics and clears drags."""

import importlib.util
import io
import json
import sys
import unittest
from contextlib import nullcontext, redirect_stdout
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import mujoco
import numpy as np
import torch

from src.tasks.supermimic.config.velocity_cfg import VelocityConfig
from src.tasks.supermimic.velocity_env import VelocityEnv

spec = importlib.util.spec_from_file_location(
  "velocity_playback_cli",
  Path(__file__).resolve().parents[1] / "scripts/play_velocity.py",
)
play_velocity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(play_velocity)


class VelocityPlaybackTests(unittest.TestCase):
  def run_playback(
    self,
    *,
    disabled=False,
    fallen=False,
    termination=False,
    steps=3,
    close_after=None,
    headless=False,
  ):
    forces = []
    environments = []
    episodes = []
    output = io.StringIO()
    handle = MagicMock()
    handle.cam = SimpleNamespace(lookat=np.zeros(3))
    handle.lock.side_effect = nullcontext
    handle.is_running.return_value = close_after != 0

    def create(config, velocity, device):
      env = VelocityEnv(config, velocity, device)
      environments.append(env)
      original = env.step

      def step(action):
        forces.append(env.sim.data.qfrc_applied.clone())
        episodes.append(int(env.episode_id[0]))
        if fallen and len(forces) == 1:
          # Rotate the real robot 90 degrees to exercise fall-only termination.
          state = env.read_state()
          root = torch.cat((state[:, :7], state[:, 36:42]), -1)
          root[:, :3] += env.scene.env_origins
          root[:, 3:7] = root.new_tensor([2**-0.5, 2**-0.5, 0, 0])
          env.robot.write_root_state_to_sim(root)
          env.sim.forward()
          env.sim.sense()
        return original(action)

      env.step = step
      return env

    def launch(model, data, **kwargs):
      del kwargs
      scene = mujoco.MjvScene(model, maxgeom=500)
      camera = mujoco.MjvCamera()
      mujoco.mjv_defaultFreeCamera(model, camera)
      mujoco.mjv_updateScene(
        model, data, mujoco.MjvOption(), None, camera, mujoco.mjtCatBit.mjCAT_ALL, scene
      )
      perturb = mujoco.MjvPerturb()
      perturb.select = int(environments[0].robot.indexing.body_ids[0])
      mujoco.mjv_initPerturb(model, data, scene, perturb)
      perturb.active = mujoco.mjtPertBit.mjPERT_TRANSLATE
      perturb.refpos[0] += 0.01
      perturb.refselpos[0] += 0.01
      handle.perturb = perturb

      def sync():
        if not fallen:
          perturb.active = 0
        if close_after is not None and len(forces) >= close_after:
          handle.is_running.return_value = False

      handle.sync.side_effect = sync
      return handle

    args = [
      "play_velocity.py",
      "--checkpoint",
      str(Path("unused.pt")),
      "--device",
      "cpu",
    ]
    if steps is not None:
      args += ["--steps", str(steps)]
    if headless:
      args.append("--headless")
    if termination:
      args.append("--termination")
    if disabled:
      args.append("--no-perturbations")
    actor = lambda history, command: torch.zeros(len(history), 29)
    with (
      patch.object(sys, "argv", args),
      patch.object(
        play_velocity,
        "load_velocity_actor",
        return_value=(actor, {"velocity_config": asdict(VelocityConfig())}),
      ),
      patch.object(play_velocity, "validate_environment"),
      patch.object(play_velocity, "VelocityEnv", side_effect=create),
      patch("mujoco.viewer.launch_passive", side_effect=launch),
      patch.object(play_velocity.time, "sleep"),
      redirect_stdout(output),
    ):
      play_velocity.main()
    report = json.loads(
      next(
        line
        for line in reversed(output.getvalue().splitlines())
        if line.startswith("{")
      )
    )
    self.assertFalse(bool(environments[0].sim.data.qfrc_applied.any()))
    self.assertEqual(
      environments[0].foundation_config.terminate_on_failure, termination
    )
    report["episode_ids"] = episodes
    if headless:
      handle.close.assert_not_called()
    else:
      handle.close.assert_called_once()
    return forces, report, handle

  def test_default_viewer_runs_past_1000_steps_until_window_closes(self):
    forces, report, _ = self.run_playback(steps=None, close_after=1002)
    self.assertEqual(len(forces), 1002)
    self.assertEqual(report["steps"], 1002)

  def test_window_closed_before_first_step_reports_zero_steps(self):
    forces, report, _ = self.run_playback(steps=None, close_after=0)
    self.assertEqual(forces, [])
    self.assertEqual(report["steps"], 0)

  def test_headless_respects_explicit_step_limit(self):
    forces, report, _ = self.run_playback(steps=3, headless=True)
    self.assertEqual(len(forces), 3)
    self.assertEqual(report["steps"], 3)
    self.assertFalse(report["mouse_perturbations_enabled"])

  def test_default_drag_reaches_physics_and_release_clears_force(self):
    forces, report, _ = self.run_playback()
    self.assertTrue(bool(forces[0].any()))
    self.assertTrue(all(not bool(force.any()) for force in forces[1:]))
    self.assertTrue(report["mouse_perturbations_enabled"])
    self.assertEqual(report["mouse_perturbed_steps"], 1)
    self.assertFalse(report["motion_library_loaded"])

  def test_no_perturbations_keeps_external_forces_disabled(self):
    forces, report, _ = self.run_playback(disabled=True)
    self.assertTrue(all(not bool(force.any()) for force in forces))
    self.assertFalse(report["mouse_perturbations_enabled"])
    self.assertEqual(report["mouse_perturbed_steps"], 0)

  def test_default_playback_continues_after_fall_without_reset(self):
    forces, report, _ = self.run_playback(fallen=True)
    self.assertTrue(all(bool(force.any()) for force in forces))
    self.assertFalse(report["failure_termination_enabled"])
    self.assertEqual(report["falls"], 1)
    self.assertEqual(len(set(report["episode_ids"])), 1)

  def test_fall_reset_releases_drag_target_before_next_physics_step(self):
    forces, report, handle = self.run_playback(fallen=True, termination=True)
    self.assertTrue(bool(forces[0].any()))
    self.assertTrue(all(not bool(force.any()) for force in forces[1:]))
    self.assertEqual(handle.perturb.active, 0)
    self.assertEqual(handle.perturb.active2, 0)
    self.assertEqual(report["falls"], 1)
