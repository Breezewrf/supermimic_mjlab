"""Numerical checks for the motion converter; run with unittest discovery."""

import sys
import tempfile
import unittest
from pathlib import Path

import mujoco
import numpy as np

# Avoid unrelated installed packages also named "scripts" (e.g. ROS).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/data"))
from convert_lafan_to_npz import (
  CSV_JOINT_NAMES,
  load_robot,
  motion_to_arrays,
  resample_motion,
  save_npz,
)


class ResamplingTests(unittest.TestCase):
  def test_endpoints_translation_and_quaternion_sign(self):
    raw = np.zeros((4, 36))
    raw[:, 0] = np.arange(4) / 30
    raw[:, 6] = [1, -1, 1, -1]  # xyzw; identical orientation
    converted = resample_motion(raw, 30, 50)
    self.assertEqual(converted.shape, (6, 36))
    np.testing.assert_allclose(converted[:, 0], np.arange(6) / 50)
    np.testing.assert_allclose(converted[:, 3], 1)
    np.testing.assert_allclose(converted[:, 4:7], 0)

  def test_slerp_and_no_extrapolation(self):
    raw = np.zeros((3, 36))
    angle = np.array([0, 0.4, 0.8])
    raw[:, 5] = np.sin(angle / 2)
    raw[:, 6] = np.cos(angle / 2)
    converted = resample_motion(raw, 30, 50)
    self.assertEqual(len(converted), 4)  # last sample is .06, not .0667
    expected = np.arange(4) / 50 * 12
    np.testing.assert_allclose(converted[:, 3], np.cos(expected / 2))
    np.testing.assert_allclose(converted[:, 6], np.sin(expected / 2))

  def test_invalid_inputs(self):
    valid = np.zeros((3, 36))
    valid[:, 6] = 1
    for raw in (valid[:1], valid[:, :-1], valid * np.nan, np.zeros_like(valid)):
      with self.subTest(shape=raw.shape), self.assertRaises(ValueError):
        resample_motion(raw, 30, 50)


class KinematicsTests(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.model, cls.joints, cls.bodies, _ = load_robot()

  def test_rotating_translating_base_link_velocities(self):
    fps = 100.0
    time = np.arange(5) / fps
    motion = np.zeros((5, 36))
    velocity = np.array([0.3, -0.2, 0.1])
    motion[:, :3] = np.array([0, 0, 0.8]) + time[:, None] * velocity
    tilt = np.array([np.cos(0.3), np.sin(0.3), 0, 0])
    for index, t in enumerate(time):
      yaw = np.array([np.cos(t / 2), 0, 0, np.sin(t / 2)])
      mujoco.mju_mulQuat(motion[index, 3:7], yaw, tilt)
    arrays = motion_to_arrays(self.model, self.joints, self.bodies, motion, fps)
    omega = np.array([0, 0, 1.0])
    expected = velocity + np.cross(omega, arrays["body_pos_w"] - motion[:, None, :3])
    np.testing.assert_allclose(arrays["body_lin_vel_w"], expected, atol=2e-6)
    np.testing.assert_allclose(
      arrays["body_ang_vel_w"], np.broadcast_to(omega, expected.shape), atol=2e-6
    )
    np.testing.assert_allclose(arrays["joint_vel"], 0, atol=1e-8)
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / "motion.npz"
      save_npz(path, arrays)
      with np.load(path, allow_pickle=False) as saved:
        self.assertEqual(saved["body_names"][0], "pelvis")
        self.assertEqual(saved["quaternion_order"].item(), "wxyz")
        np.testing.assert_array_equal(saved["joint_pos"], arrays["joint_pos"])

  def test_joint_name_mapping_and_link_velocity_derivative(self):
    fps = 1000.0
    time = np.arange(5) / fps
    motion = np.zeros((5, 36))
    motion[:, 2:4] = [0.8, 1]
    source_index = CSV_JOINT_NAMES.index("left_hip_pitch_joint")
    motion[:, 7 + source_index] = 0.2 + time * 0.5
    # Deliberately change export ordering to check mapping by name, not index.
    joints = self.joints[::-1]
    arrays = motion_to_arrays(self.model, joints, self.bodies, motion, fps)
    target_index = joints.index("left_hip_pitch_joint")
    np.testing.assert_allclose(arrays["joint_pos"][:, target_index], 0.2 + time * 0.5)
    np.testing.assert_allclose(arrays["joint_vel"][:, target_index], 0.5)
    numerical = (arrays["body_pos_w"][3] - arrays["body_pos_w"][1]) * fps / 2
    np.testing.assert_allclose(arrays["body_lin_vel_w"][2], numerical, atol=4e-5)


if __name__ == "__main__":
  unittest.main()
