"""Evaluation score semantics, terminal-frame rendering and checkpoint checks."""

import csv
import json
import tempfile
import threading
import unittest
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import mujoco
import numpy as np
import torch

from src.tasks.supermimic.config.task_cfg import FoundationConfig
from src.tasks.supermimic.data.world_model_data import file_sha256
from src.tasks.supermimic.models.policy import REFERENCE_SCHEMA, MotionActor
from src.tasks.supermimic.rl.evaluation import (
  EvaluationInterrupted,
  evaluate,
  evaluation_summary,
  save_report,
)
from src.tasks.supermimic.rl.inference import load_actor
from src.tasks.supermimic.rl.visualization import (
  EvaluationScene,
  LiveViewer,
  ViewerClosed,
)


class FakeEnv:
  num_envs = 1
  step_dt = 0.02
  foundation_config = FoundationConfig(num_envs=1)
  library = SimpleNamespace(names=["short", "long"], lengths=[6, 11])

  def __init__(self, stop="failure"):
    self.stop = stop

  def reset(self, *, seed, options):
    assert seed == self.foundation_config.seed
    assert options["frame_id"] == 0
    self.clip, self.time = options["motion_id"], 0
    self.state = torch.zeros(1, 71)
    return self.inputs(), {}

  def inputs(self):
    return {"history": torch.zeros(1, 10, 93), "future_reference": {}}

  def read_state(self):
    return self.state

  def step(self, action):
    assert not torch.is_grad_enabled()
    self.time += 1
    special = self.clip == 1 and self.time == 2
    failed = special and self.stop == "failure"
    timeout = special and self.stop == "timeout"
    end = self.time == self.library.lengths[self.clip] - 1
    next_state = torch.full((1, 71), float(self.time))
    # Reproduce auto-reset behavior: read_state() is already a new episode.
    self.state = torch.full((1, 71), -999.0) if failed or timeout or end else next_state
    return (
      self.inputs(),
      torch.tensor([0.1 if self.clip == 0 else 0.05]),
      torch.tensor([failed]),
      torch.tensor([timeout or end]),
      {
        "transition": {"time_out": torch.tensor([timeout]), "next_state": next_state},
        "metrics": {
          "height_failure": torch.tensor([failed]),
          "tilt_failure": torch.tensor([False]),
          **{
            key: torch.tensor([0.2])
            for key in (
              "body_position_error",
              "body_orientation_error",
              "root_position_error",
              "root_orientation_error",
            )
          },
        },
      },
    )


class EvaluationProtocolTests(unittest.TestCase):
  def test_clip_mean_frame_weighting_and_full_length_failure_penalty(self):
    report = evaluate(FakeEnv())
    self.assertAlmostEqual(report["normalized_return"], 0.55)
    self.assertAlmostEqual(report["normalized_return_framew"], 0.4)
    self.assertEqual(report["successes"], 1)
    self.assertEqual(report["n_failed"], 1)
    self.assertEqual(report["failed"], ["long"])
    self.assertTrue(report["evaluation_complete"])
    self.assertAlmostEqual(report["motions"][1]["normalized_return"], 0.1)
    self.assertAlmostEqual(report["motions"][1]["completion"], 0.2)
    self.assertIn("Evaluation return 0.550", evaluation_summary(report))
    self.assertNotIn("INCOMPLETE", evaluation_summary(report))

  def test_debug_limit_and_timeout_are_incomplete_without_physical_failures(self):
    for report in (
      evaluate(FakeEnv(), max_steps_per_clip=1),
      evaluate(FakeEnv("timeout")),
    ):
      self.assertFalse(report["evaluation_complete"])
      self.assertEqual(report["n_failed"], 0)
      self.assertIn("INCOMPLETE", evaluation_summary(report))

  def test_observer_gets_start_and_final_pre_reset_states_without_affecting_score(self):
    frames, rows = [], []

    def capture(env, clip, frame, state, reward):
      self.assertEqual(float(state[0, 0]), frame)
      if clip == 1 and frame == 2:
        self.assertEqual(float(env.read_state()[0, 0]), -999)
      frames.append((clip, frame))

    rendered = evaluate(FakeEnv(), on_frame=capture, on_motion=rows.append)
    self.assertEqual(rendered, evaluate(FakeEnv()))
    self.assertEqual(rows, rendered["motions"])
    self.assertEqual(
      frames, [(0, frame) for frame in range(6)] + [(1, frame) for frame in range(3)]
    )

  def test_window_close_interrupts_instead_of_reporting_partial_score(self):
    def close(*args):
      raise EvaluationInterrupted()

    with self.assertRaises(EvaluationInterrupted):
      evaluate(FakeEnv(), on_frame=close)

  def test_invalid_protocol_is_rejected(self):
    for option in ("randomization", "sensor_noise", "reset_noise"):
      env = FakeEnv()
      env.foundation_config = replace(env.foundation_config, **{option: True})
      with self.assertRaisesRegex(ValueError, "disable"):
        evaluate(env)
    env = FakeEnv()
    env.num_envs = 2
    with self.assertRaisesRegex(ValueError, "one environment"):
      evaluate(env, on_frame=lambda *args: None)
    with self.assertRaisesRegex(ValueError, "one environment"):
      evaluate(env, before_step=lambda env: None)
    with self.assertRaisesRegex(ValueError, "positive"):
      evaluate(FakeEnv(), max_steps_per_clip=0)
    env = FakeEnv()
    env.library = SimpleNamespace(names=[], lengths=[])
    with self.assertRaisesRegex(ValueError, "at least one"):
      evaluate(env)

  def test_json_and_csv_preserve_protocol_and_motion_selection(self):
    report = evaluate(FakeEnv()) | {"policy": "checkpoint.pt", "iteration": 123}
    with tempfile.TemporaryDirectory() as folder:
      output, scores = Path(folder) / "eval.json", Path(folder) / "eval.csv"
      save_report(report, output, scores)
      save_report(report, output, scores)
      self.assertEqual(json.loads(output.read_text()), report)
      with scores.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
      self.assertEqual(len(rows), 2)
      self.assertEqual(rows[0]["iteration"], "123")
      self.assertEqual(json.loads(rows[0]["clips"]), ["short", "long"])
      self.assertEqual(json.loads(rows[0]["failed"]), ["long"])

  def test_existing_csv_is_upgraded_without_losing_interaction_provenance(self):
    report = evaluate(FakeEnv())
    with tempfile.TemporaryDirectory() as folder:
      output, scores = Path(folder) / "eval.json", Path(folder) / "eval.csv"
      with scores.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["policy", "normalized_return"])
        writer.writeheader()
        writer.writerow({"policy": "old.pt", "normalized_return": "0.8"})
      interactive = report | {
        "evaluation_mode": "interactive",
        "mouse_perturbation_steps": 3,
      }
      save_report(interactive, output, scores)
      with scores.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
      self.assertEqual(rows[0]["policy"], "old.pt")
      self.assertEqual(rows[0]["evaluation_mode"], "standard")
      self.assertEqual(rows[1]["evaluation_mode"], "interactive")
      self.assertEqual(rows[1]["mouse_perturbation_steps"], "3")
      self.assertIn("INTERACTIVE", evaluation_summary(interactive))

  def test_before_step_runs_after_initial_render_and_once_per_transition(self):
    frames, steps = [], []
    env = FakeEnv()
    evaluate(
      env,
      on_frame=lambda e, c, f, s, r: frames.append((c, f)),
      before_step=lambda e: steps.append((e.clip, e.time)),
    )
    self.assertEqual(steps, [(0, frame) for frame in range(5)] + [(1, 0), (1, 1)])
    self.assertEqual(len(frames), len(steps) + 2)


class CheckpointInferenceTests(unittest.TestCase):
  def test_actor_loads_in_eval_mode_and_rejects_incompatible_data(self):
    with tempfile.TemporaryDirectory() as folder:
      manifest, checkpoint = Path(folder) / "manifest.json", Path(folder) / "actor.pt"
      manifest.write_text("{}")
      actor = MotionActor(torch.zeros(29), (8,), relative_orientation=True)
      saved = {
        "kind": "supermimic_online",
        "schema_version": 1,
        "reference_schema": REFERENCE_SCHEMA,
        "manifest_sha256": file_sha256(manifest),
        "clips": ["short", "long"],
        "robot_metadata": {"default_q": actor.default_q},
        "config": {"actor_dims": [8], "relative_orientation": True},
        "actor": actor.state_dict(),
      }
      torch.save(saved, checkpoint)
      loaded, _, clips = load_actor(checkpoint, manifest, "cpu")
      self.assertFalse(loaded.training)
      self.assertTrue(loaded.relative_orientation)
      self.assertEqual(clips, ("short", "long"))
      for key, value in actor.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], value)
      self.assertEqual(load_actor(checkpoint, manifest, "cpu", ("long",))[2], ("long",))
      with self.assertRaisesRegex(ValueError, "present"):
        load_actor(checkpoint, manifest, "cpu", ("unknown",))
      manifest.write_text('{"changed": true}')
      with self.assertRaisesRegex(ValueError, "manifest"):
        load_actor(checkpoint, manifest, "cpu")
      saved.pop("reference_schema")
      manifest.write_text("{}")
      torch.save(saved, checkpoint)
      _, metadata, _ = load_actor(checkpoint, manifest, "cpu")
      self.assertTrue(metadata["reference_schema_inferred"])
      self.assertEqual(metadata["reference_schema"], REFERENCE_SCHEMA)
      saved["reference_schema"] = "unknown_reference"
      torch.save(saved, checkpoint)
      with self.assertRaisesRegex(ValueError, "Unsupported reference schema"):
        load_actor(checkpoint, manifest, "cpu")


class RenderSceneTests(unittest.TestCase):
  def test_joint_mapping_reference_alignment_and_physics_isolation(self):
    joints = "".join(
      f'<body pos="0 0 {i * 0.02}"><joint name="j{i}"/><geom size="0.01"/></body>'
      for i in range(29)
    )
    model = mujoco.MjModel.from_xml_string(
      '<mujoco><worldbody><geom type="plane" size="2 2 .1"/>'
      '<body name="pelvis"><freejoint/><geom size=".1"/>'
      + joints
      + "</body></worldbody></mujoco>"
    )
    order = torch.arange(35, 6, -1)
    reference = {
      "body_pos_w": torch.tensor([[[10.0, 20.0, 0.8]]]),
      "body_quat_w": torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]),
      "joint_pos": torch.linspace(-0.2, 0.2, 29).unsqueeze(0),
    }
    env = SimpleNamespace(
      sim=SimpleNamespace(mj_model=model),
      device="cpu",
      step_dt=0.02,
      robot=SimpleNamespace(
        indexing=SimpleNamespace(
          free_joint_q_adr=torch.arange(7),
          joint_q_adr=order,
          free_joint_v_adr=torch.arange(6),
          joint_v_adr=torch.arange(34, 5, -1),
        )
      ),
      scene=SimpleNamespace(env_origins=torch.zeros(1, 3)),
      library=SimpleNamespace(get_frame=lambda *_: reference),
    )
    state = torch.zeros(1, 71)
    state[0, :3] = torch.tensor([1.0, 2.0, 0.7])
    state[0, 3:7] = torch.tensor([2**-0.5, 0, 0, 2**-0.5])
    state[0, 7:36] = torch.linspace(0.1, 0.3, 29)
    state[0, 36:42] = torch.tensor([0.3, 0.2, 0.1, 1.0, 0.0, 0.0])
    state[0, 42:] = torch.linspace(-0.2, 0.2, 29)
    original_qpos, original_colors = model.qpos0.copy(), model.geom_rgba.copy()
    for global_ghost in (False, True):
      scene = EvaluationScene(env, global_ghost=global_ghost)
      scene.update(env, 0, 1, state)
      np.testing.assert_allclose(scene.data.qpos[order], state[0, 7:36])
      np.testing.assert_allclose(scene.data.qvel[:3], state[0, 36:39])
      np.testing.assert_allclose(scene.data.qvel[3:6], [0, -1, 0], atol=1e-6)
      np.testing.assert_allclose(scene.data.qvel[np.arange(34, 5, -1)], state[0, 42:])
      np.testing.assert_allclose(
        scene.reference_data.qpos[order], reference["joint_pos"][0]
      )
      target_xy = [10, 20] if global_ghost else [1, 2]
      np.testing.assert_allclose(scene.reference_data.qpos[:3], [*target_xy, 0.8])
      target_quat = reference["body_quat_w"][0, 0] if global_ghost else state[0, 3:7]
      np.testing.assert_allclose(scene.reference_data.qpos[3:7], target_quat, atol=1e-6)
      render_scene = mujoco.MjvScene(scene.model, maxgeom=1000)
      scene.add_ghost(render_scene)
      self.assertGreater(render_scene.ngeom, 0)
      for geom in render_scene.geoms[: render_scene.ngeom]:
        np.testing.assert_allclose(geom.rgba, [0.25, 0.6, 1.0, 0.3], atol=1e-6)
      np.testing.assert_array_equal(model.qpos0, original_qpos)
      np.testing.assert_array_equal(model.geom_rgba, original_colors)


class ViewerControlTests(unittest.TestCase):
  def test_pause_step_ghost_and_close_without_a_display(self):
    camera = SimpleNamespace(
      lookat=np.zeros(3),
      distance=3.5,
      azimuth=135,
      elevation=-15,
    )
    scene = SimpleNamespace(
      model=MagicMock(),
      data=MagicMock(),
      camera=camera,
      ghost=True,
      update=MagicMock(),
      add_ghost=MagicMock(),
    )
    handle = MagicMock()
    handle.cam = SimpleNamespace(lookat=np.zeros(3))
    handle.user_scn = SimpleNamespace(ngeom=0)
    handle.lock.side_effect = nullcontext
    handle.is_running.return_value = True
    closed = threading.Event()
    render_thread = threading.Thread(target=lambda: closed.wait(5), daemon=True)

    def launch(*args, **kwargs):
      render_thread.start()
      return handle

    handle.close.side_effect = closed.set
    with (
      patch(
        "src.tasks.supermimic.rl.visualization.EvaluationScene", return_value=scene
      ),
      patch("mujoco.viewer.launch_passive", side_effect=launch),
      patch("src.tasks.supermimic.rl.visualization.time.sleep"),
    ):
      viewer = LiveViewer(FakeEnv(), speed=0.5)
      frame = (FakeEnv(), 0, 0, torch.zeros(1, 71), 0.0)
      viewer.commands.put(32)
      viewer.commands.put(262)
      viewer.commands.put(ord("G"))
      viewer(*frame)
      self.assertTrue(viewer.paused)
      self.assertFalse(scene.ghost)
      scene.update.assert_called_once()
      handle.sync.assert_called_once()
      viewer.commands.put(32)
      viewer(*frame)
      self.assertFalse(viewer.paused)
      handle.is_running.return_value = False
      with self.assertRaises(ViewerClosed):
        viewer(*frame)
      viewer.close()
      handle.close.assert_called_once()
      self.assertFalse(render_thread.is_alive())


if __name__ == "__main__":
  unittest.main()
