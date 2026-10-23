"""Native mouse forces reach physics and clear on release without touching poses."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import mujoco
import numpy as np
import torch

from src.tasks.supermimic.rl.visualization import LiveViewer, MousePerturbations


class MousePerturbationTests(unittest.TestCase):
  def setUp(self):
    self.model = mujoco.MjModel.from_xml_string(
      '<mujoco><option gravity="0 0 0"/><worldbody>'
      '<body name="robot" pos="0 0 1"><freejoint/><geom size=".1" mass="1"/>'
      '<body name="link" pos=".4 0 0"><joint axis="0 0 1"/>'
      '<geom size=".05" mass=".2"/></body></body></worldbody></mujoco>'
    )
    self.data = mujoco.MjData(self.model)
    mujoco.mj_forward(self.model, self.data)
    self.scene = mujoco.MjvScene(self.model, maxgeom=100)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(self.model, camera)
    mujoco.mjv_updateScene(
      self.model,
      self.data,
      mujoco.MjvOption(),
      None,
      camera,
      mujoco.mjtCatBit.mjCAT_ALL,
      self.scene,
    )
    self.env = SimpleNamespace(
      robot=SimpleNamespace(indexing=SimpleNamespace(body_ids=torch.tensor([1, 2]))),
      sim=SimpleNamespace(
        data=SimpleNamespace(
          qfrc_applied=torch.zeros(1, self.model.nv),
          xfrc_applied=torch.ones(1, self.model.nbody, 6),
        )
      ),
    )
    self.mouse = MousePerturbations(self.env, self.model, self.data)

  def perturb(self, body=1):
    perturb = mujoco.MjvPerturb()
    perturb.select = body
    mujoco.mjv_initPerturb(self.model, self.data, self.scene, perturb)
    perturb.active = mujoco.mjtPertBit.mjPERT_TRANSLATE
    perturb.refpos[0] += 0.1
    perturb.refselpos[0] += 0.1
    return perturb

  def test_drag_changes_physical_velocity_without_writing_poses_or_cartesian_forces(
    self,
  ):
    original_pose = self.data.qpos.copy()
    self.mouse.apply(self.perturb())
    qfrc = self.env.sim.data.qfrc_applied[0].numpy()
    self.assertGreater(qfrc[0], 0)
    physical = mujoco.MjData(self.model)
    physical.qfrc_applied[:] = qfrc
    mujoco.mj_step(self.model, physical)
    self.assertGreater(physical.qvel[0], 0)
    self.assertEqual(self.mouse.applied_steps, 1)
    np.testing.assert_array_equal(self.data.qpos, original_pose)
    torch.testing.assert_close(
      self.env.sim.data.xfrc_applied, torch.ones(1, self.model.nbody, 6)
    )

  def test_release_selection_change_and_clear_remove_stale_forces(self):
    perturb = self.perturb()
    self.mouse.apply(perturb)
    perturb.active = 0
    self.mouse.apply(perturb)
    self.assertFalse(bool(self.env.sim.data.qfrc_applied.any()))
    self.assertFalse(bool(self.data.xfrc_applied.any()))
    self.assertEqual(self.mouse.applied_steps, 1)
    self.mouse.apply(self.perturb(body=2))
    np.testing.assert_array_equal(self.data.xfrc_applied[1], 0)
    self.assertTrue(bool(self.env.sim.data.qfrc_applied.any()))
    self.mouse.clear()
    self.assertFalse(bool(self.env.sim.data.qfrc_applied.any()))

  def test_world_selection_does_not_apply_force(self):
    self.mouse.apply(self.perturb())
    perturb = mujoco.MjvPerturb()
    perturb.active = mujoco.mjtPertBit.mjPERT_TRANSLATE
    perturb.select = 0
    self.mouse.apply(perturb)
    self.assertFalse(bool(self.env.sim.data.qfrc_applied.any()))
    self.assertEqual(self.mouse.applied_steps, 1)

  def test_rotation_drag_applies_torque(self):
    perturb = self.perturb()
    perturb.refpos[:] = self.data.xipos[1]
    perturb.active = mujoco.mjtPertBit.mjPERT_ROTATE
    perturb.refquat[:] = [np.cos(0.1), 0, 0, np.sin(0.1)]
    self.mouse.apply(perturb)
    self.assertGreater(
      np.linalg.norm(self.env.sim.data.qfrc_applied[0, 3:6].numpy()), 0
    )

  def test_viewer_sync_is_read_only_until_step_and_clears_drag_between_motions(self):
    camera = SimpleNamespace(
      lookat=np.zeros(3), distance=3.5, azimuth=135, elevation=-15
    )
    scene = SimpleNamespace(
      model=self.model,
      data=self.data,
      camera=camera,
      update=MagicMock(),
      add_ghost=MagicMock(),
    )
    self.env.library = SimpleNamespace(names=["one", "two"], lengths=[10, 10])
    self.env.step_dt = 0.02
    handle = MagicMock()
    handle.perturb = self.perturb()
    handle.cam = SimpleNamespace(lookat=np.zeros(3))
    handle.lock.side_effect = nullcontext
    handle.is_running.return_value = True
    with (
      patch(
        "src.tasks.supermimic.rl.visualization.EvaluationScene", return_value=scene
      ),
      patch("mujoco.viewer.launch_passive", return_value=handle),
      patch("src.tasks.supermimic.rl.visualization.time.sleep"),
    ):
      viewer = LiveViewer(self.env, perturbations=True)
      viewer.begin_evaluation()
      viewer(self.env, 0, 0, torch.zeros(1, 71), 0)
      self.assertEqual(handle.perturb.active, 0)
      handle.perturb.active = mujoco.mjtPertBit.mjPERT_TRANSLATE
      viewer.commands.put(32)
      viewer.commands.put(262)
      viewer(self.env, 0, 1, torch.zeros(1, 71), 0)
      self.assertTrue(viewer.paused)
      self.assertEqual(viewer.mouse.applied_steps, 0)
      self.assertFalse(bool(self.env.sim.data.qfrc_applied.any()))
      viewer.before_step(self.env)
      self.assertEqual(viewer.mouse.applied_steps, 1)
      self.assertTrue(bool(self.env.sim.data.qfrc_applied.any()))
      viewer.commands.put(262)
      viewer(self.env, 1, 0, torch.zeros(1, 71), 0)
      self.assertEqual(handle.perturb.active, 0)
      self.assertFalse(bool(self.env.sim.data.qfrc_applied.any()))
      handle.perturb.active = mujoco.mjtPertBit.mjPERT_TRANSLATE
      viewer.before_step(self.env)
      viewer.close()
      self.assertFalse(bool(self.env.sim.data.qfrc_applied.any()))


if __name__ == "__main__":
  unittest.main()
