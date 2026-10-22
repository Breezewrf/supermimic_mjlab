"""Physical contact filtering and reference reset perturbation regressions."""

import re
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import mujoco
import numpy as np
import torch

from src.tasks.supermimic.config.robot_cfg import (
  FOOT_COLLISION_EXPR,
  G1_XML,
  get_supermimic_robot_cfg,
)
from src.tasks.supermimic.config.task_cfg import FoundationConfig
from src.tasks.supermimic.env_cfg import make_env_cfg
from src.tasks.supermimic.mdp.resets import RESET_NOISE, reference_reset_state
from src.tasks.supermimic.mdp.state import (
  quat_conjugate,
  quat_mul,
  quat_to_rotvec,
  rotvec_to_quat,
)
from src.tasks.supermimic.models.kinematics import robot_metadata
from src.tasks.supermimic.rl.evaluation import evaluate


class SimulationConfigTests(unittest.TestCase):
  def test_collision_filter_retains_all_body_ground_contacts_without_self_contacts(
    self,
  ):
    from mjlab.asset_zoo.robots import get_g1_robot_cfg
    from mjlab.entity import Entity

    model = Entity(get_supermimic_robot_cfg()).spec.compile()
    selected = np.array(
      [i for i in range(model.ngeom) if model.geom(i).name.endswith("_collision")]
    )
    self.assertGreater(len(np.unique(model.geom_bodyid[selected])), 10)
    contype, affinity = model.geom_contype[selected], model.geom_conaffinity[selected]
    self.assertFalse(
      (
        (contype[:, None] & affinity[None, :]) | (contype[None, :] & affinity[:, None])
      ).any()
    )
    # MuJoCo's ground default contype=1 still collides with every selected body.
    self.assertTrue(((1 & affinity) != 0).all())
    self.assertEqual(model.npair, 0)
    # Metadata must fingerprint the same physical collision configuration.
    signature = robot_metadata()["signature"]
    with patch(
      "src.tasks.supermimic.config.robot_cfg.get_supermimic_robot_cfg",
      side_effect=get_g1_robot_cfg,
    ):
      self.assertNotEqual(signature, robot_metadata()["signature"])

  def test_vendored_shapes_and_training_actuators(self):
    from mjlab.entity import Entity

    robot = Entity(get_supermimic_robot_cfg())
    model = robot.spec.compile()
    source = mujoco.MjModel.from_xml_path(str(G1_XML))
    selected = [
      i for i in range(model.ngeom) if model.geom(i).name.endswith("_collision")
    ]
    original = [
      i
      for i in range(source.ngeom)
      if source.geom_bodyid[i]
      and (source.geom_contype[i] or source.geom_conaffinity[i])
    ]
    for field in ("geom_type", "geom_bodyid", "geom_pos", "geom_quat", "geom_size"):
      np.testing.assert_array_equal(
        getattr(model, field)[selected], getattr(source, field)[original]
      )
    np.testing.assert_array_equal(model.mesh_vert, source.mesh_vert)
    np.testing.assert_array_equal(model.mesh_face, source.mesh_face)
    self.assertEqual((model.nq, model.nv, model.nu), (36, 35, 29))
    self.assertEqual(len(set(robot.actuator_names)), 29)
    self.assertTrue((model.actuator_biasprm[:, 1] < 0).all())  # Position PD.
    self.assertFalse((model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE).any())
    foot = [
      i for i in selected if re.fullmatch(FOOT_COLLISION_EXPR, model.geom(i).name)
    ]
    self.assertEqual(len(foot), 8)
    self.assertTrue((model.geom_type[foot] == mujoco.mjtGeom.mjGEOM_SPHERE).all())
    np.testing.assert_array_equal(model.geom_size[foot, 0], np.full(8, 0.005))
    # Visual meshes remain excluded even though other mesh geoms now collide.
    visual = [i for i in range(model.ngeom) if model.geom(i).name.endswith("_visual")]
    self.assertFalse(model.geom_contype[visual].any())
    self.assertFalse(model.geom_conaffinity[visual].any())

  def test_randomization_lifecycle_and_evaluation_ranges(self):
    for config in (FoundationConfig.training(), FoundationConfig()):
      events = make_env_cfg(config).events
      self.assertEqual(events["foot_friction"].mode, "startup")
      self.assertEqual(events["torso_com"].mode, "startup")
      self.assertEqual(events["encoder_bias"].mode, "reset")
    config = make_env_cfg(FoundationConfig())
    self.assertEqual(config.events["foot_friction"].params["ranges"], (1.0, 1.0))
    self.assertEqual(config.events["encoder_bias"].params["bias_range"], (0.0, 0.0))
    self.assertTrue(
      all(
        bounds == (0.0, 0.0)
        for bounds in config.events["torso_com"].params["ranges"].values()
      )
    )
    self.assertTrue(FoundationConfig.training().reset_noise)
    self.assertFalse(FoundationConfig().reset_noise)
    self.assertFalse(FoundationConfig.training(reset_noise=False).reset_noise)

  def test_evaluation_rejects_reset_perturbations(self):
    env = SimpleNamespace(
      num_envs=1, foundation_config=FoundationConfig(reset_noise=True)
    )
    with self.assertRaisesRegex(ValueError, "reset"):
      evaluate(env)


class ResetPerturbationTests(unittest.TestCase):
  def setUp(self):
    self.root = torch.zeros(1024, 13)
    self.root[:, :3] = torch.tensor([1.0, 2.0, 0.8])
    self.root[:, 3:7] = rotvec_to_quat(torch.tensor([[0.4, -0.3, 0.6]]))
    self.root[:, 7:] = 0.2
    self.q = torch.ones(1024, 29) * 0.95
    self.qd = torch.arange(29).float().expand(1024, -1)
    self.limits = torch.tensor([-1.0, 1.0]).expand(1024, 29, -1)

  def sample(self, perturb=True):
    return reference_reset_state(
      self.root,
      self.q,
      self.qd,
      self.limits,
      perturb=perturb,
      generator=torch.Generator().manual_seed(3),
    )

  def test_distributions_units_limits_and_world_frame_rotation(self):
    root, q, qd = self.sample()
    for section, key in (
      (slice(0, 3), "position_m"),
      (slice(7, 10), "linear_velocity_m_s"),
      (slice(10, 13), "angular_velocity_rad_s"),
    ):
      delta = root[:, section] - self.root[:, section]
      bounds = root.new_tensor(RESET_NOISE[key])
      self.assertTrue((delta.abs() <= bounds + 1e-6).all())
      self.assertTrue((delta.abs().amax(0) >= 0.9 * bounds).all())
    # Left multiplication must recover the sampled world-frame rotation vector.
    rotation = quat_to_rotvec(quat_mul(root[:, 3:7], quat_conjugate(self.root[:, 3:7])))
    bounds = root.new_tensor(RESET_NOISE["rotation_vector_rad"])
    self.assertTrue((rotation.abs() <= bounds + 1e-6).all())
    self.assertTrue((rotation.abs().amax(0) >= 0.9 * bounds).all())
    torch.testing.assert_close(root[:, 3:7].norm(dim=-1), torch.ones(1024))
    self.assertTrue((q.abs() <= 1).all())
    self.assertTrue((q == 1).any())
    self.assertLessEqual(float((q - self.q).abs().max()), 0.100001)
    torch.testing.assert_close(qd, self.qd)
    torch.testing.assert_close(self.q, torch.full_like(self.q, 0.95))
    for a, b in zip((root, q, qd), self.sample()):
      torch.testing.assert_close(a, b, rtol=0, atol=0)

  def test_noise_free_reset_preserves_reference_and_does_not_consume_rng(self):
    generator = torch.Generator().manual_seed(7)
    state = generator.get_state().clone()
    result = reference_reset_state(
      self.root, self.q, self.qd, self.limits, perturb=False, generator=generator
    )
    for expected, actual in zip((self.root, self.q, self.qd), result):
      torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    torch.testing.assert_close(generator.get_state(), state)


if __name__ == "__main__":
  unittest.main()
