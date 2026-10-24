"""BeyondMimic torso-anchor conventions, differentiability and compatibility."""

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from src.tasks.supermimic.config.task_cfg import (
  MOTION_ANCHOR_INDEX,
  TRACKED_BODIES,
  FoundationConfig,
)
from src.tasks.supermimic.data.motion_library import MotionLibrary
from src.tasks.supermimic.data.world_model_data import file_sha256
from src.tasks.supermimic.mdp.state import (
  add_sensor_noise,
  motion_anchor_ori_b,
  quat_mul,
  rotvec_to_quat,
)
from src.tasks.supermimic.models.kinematics import G1Kinematics, robot_metadata
from src.tasks.supermimic.models.legacy_policy import LegacyMotionActor
from src.tasks.supermimic.models.policy import MotionActor, MotionCritic
from src.tasks.supermimic.models.world_model import WorldModel
from src.tasks.supermimic.rl.execution import TrainingKernels
from src.tasks.supermimic.rl.imagination import frozen_parameters, imagine
from src.tasks.supermimic.rl.inference import load_actor, load_world_model
from tests.supermimic_fixtures import make_motions


class MotionAnchorObservationTests(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    torch.set_num_threads(4)
    cls.metadata = robot_metadata()
    folder = tempfile.TemporaryDirectory()
    cls.addClassCleanup(folder.cleanup)
    cls.manifest, cls.clips = make_motions(Path(folder.name), cls.metadata)
    cls.library = MotionLibrary(
      cls.manifest, tuple(cls.metadata["joint_names"]), TRACKED_BODIES
    )
    cls.kinematics = G1Kinematics(cls.metadata)

  def state(self):
    state = torch.zeros(2, 71)
    state[:, 2:4] = torch.tensor([0.8, 1.0])
    state[:, 7:36] = self.metadata["default_q"].float()
    return state

  def test_upstream_matrix_order_and_common_world_rotation_invariance(self):
    identity = torch.tensor([[1.0, 0, 0, 0]])
    yaw = rotvec_to_quat(torch.tensor([[0.0, 0, torch.pi / 2]]))
    torch.testing.assert_close(
      motion_anchor_ori_b(identity, identity),
      torch.tensor([[1.0, 0, 0, 1, 0, 0]]),
    )
    torch.testing.assert_close(
      motion_anchor_ori_b(identity, yaw),
      torch.tensor([[0.0, -1, 1, 0, 0, 0]]),
    )
    robot = rotvec_to_quat(torch.tensor([[0.2, -0.4, 0.3], [-0.1, 0.3, 0.5]]))
    motion = rotvec_to_quat(torch.tensor([[0.5, 0.1, -0.2], [0.2, -0.3, -0.6]]))
    world = rotvec_to_quat(torch.tensor([[0.3, 0.2, -0.7]]))
    expected = motion_anchor_ori_b(robot, motion)
    torch.testing.assert_close(
      motion_anchor_ori_b(quat_mul(world, robot), quat_mul(world, motion)),
      expected,
    )
    torch.testing.assert_close(motion_anchor_ori_b(-robot, motion), expected)

  def test_anchor_uses_waist_joints_and_matches_full_forward_kinematics(self):
    state = self.state()
    waist = self.metadata["joint_names"].index("waist_yaw_joint")
    state[1, 7 + waist] += 0.6
    state.requires_grad_()
    anchor = self.kinematics.anchor_quaternion(state)
    full = self.kinematics(state)["body_quat_w"][:, MOTION_ANCHOR_INDEX]
    torch.testing.assert_close(anchor, full)
    feature = motion_anchor_ori_b(anchor, full.detach()[:1].expand_as(full))
    self.assertGreater(float((feature[0] - feature[1]).detach().abs().max()), 0.5)
    feature[:, 1].sum().backward()
    self.assertTrue(torch.isfinite(state.grad).all())
    self.assertGreater(float(state.grad[:, 7 + waist].abs().min()), 0.5)

  def test_simulator_observation_uses_torso_and_preserves_action_noise_layout(self):
    from src.tasks.supermimic.env import SuperMimicEnv

    state = self.state()
    # Torso matches its reference while pelvis deliberately has a yaw error.
    bodies = self.kinematics(state)
    reference = {key: value.clone() for key, value in bodies.items()}
    reference["body_quat_w"][:, 0] = rotvec_to_quat(torch.tensor([[0.0, 0, 0.7]]))
    action = torch.randn(2, 29)
    for relative in (False, True):
      config = FoundationConfig(relative_orientation=relative, motion_anchor_ori_b=True)
      env = SimpleNamespace(
        foundation_config=config,
        previous_action=action,
        context=torch.zeros(2, 33),
        body_ids=torch.arange(len(TRACKED_BODIES)),
        robot=SimpleNamespace(
          data=SimpleNamespace(
            default_joint_pos=self.metadata["default_q"].float(),
            body_link_quat_w=bodies["body_quat_w"],
          )
        ),
      )
      clean, _ = SuperMimicEnv._observations(env, state, reference)
      start = 64 + 6 * relative
      self.assertEqual(clean.shape, (2, config.observation_dim))
      torch.testing.assert_close(
        clean[:, start : start + 6],
        torch.tensor([[1.0, 0, 0, 1, 0, 0]]).expand(2, -1),
      )
      noisy = add_sensor_noise(clean, torch.Generator().manual_seed(42))
      self.assertLessEqual(
        float((noisy[:, start : start + 6] - clean[:, start : start + 6]).abs().max()),
        0.050001,
      )
      torch.testing.assert_close(noisy[:, -29:], action)

  def test_imagination_successor_reference_and_compiled_actor_gradients(self):
    state = self.state()
    # Vary torso targets independently of pelvis to catch root/torso mixups.
    self.library.data["body_quat_w"][:, MOTION_ANCHOR_INDEX] = rotvec_to_quat(
      torch.stack(
        (
          torch.zeros(len(self.library.data["body_quat_w"])),
          torch.zeros(len(self.library.data["body_quat_w"])),
          torch.arange(len(self.library.data["body_quat_w"])) * 0.07,
        ),
        -1,
      )
    )
    for relative in (False, True):
      actor = MotionActor(
        self.metadata["default_q"], (16,), relative, motion_anchor_ori_b=True
      )
      world = WorldModel(
        self.metadata, (16,), "analytic", relative, motion_anchor_ori_b=True
      )
      critic = MotionCritic(self.metadata["default_q"], (16,))
      context = torch.zeros(2, 33)
      motion, frame = torch.zeros(2, dtype=torch.long), torch.tensor([0, 2])
      ref = self.library.get_frame(motion, frame)
      observation = world.kinematics.observation(
        state,
        torch.zeros(2, 29),
        context,
        ref["body_quat_w"],
        relative_orientation=relative,
        motion_anchor_orientation=True,
      )
      starts = {
        "state": state,
        "context": context,
        "motion_id": motion,
        "frame_id": frame,
        "history": observation[:, None].repeat(1, 10, 1),
      }
      kernels = TrainingKernels(actor, world, critic, enabled=True, backend="eager")
      with frozen_parameters(world, critic):
        eager = imagine(
          actor,
          world,
          critic,
          self.library,
          starts,
          horizon=3,
          sensor_noise=False,
        )
        compiled = imagine(
          actor,
          world,
          critic,
          self.library,
          starts,
          horizon=3,
          sensor_noise=False,
          kernels=kernels,
        )
        gradients = torch.autograd.grad(eager["objective"], tuple(actor.parameters()))
        compiled_gradients = torch.autograd.grad(
          compiled["objective"], tuple(actor.parameters())
        )
      torch.testing.assert_close(compiled["history"], eager["history"])
      for expected, actual in zip(gradients, compiled_gradients):
        self.assertTrue(torch.isfinite(actual).all())
        torch.testing.assert_close(actual, expected)
      self.assertGreater(sum(float(g.abs().sum()) for g in gradients), 0)
      final_ref = self.library.get_frame(motion, frame + 3)
      bodies = world.kinematics(eager["states"][:, -1])
      expected = motion_anchor_ori_b(
        bodies["body_quat_w"][:, MOTION_ANCHOR_INDEX],
        final_ref["body_quat_w"][:, MOTION_ANCHOR_INDEX],
      )
      start = 64 + 6 * relative
      torch.testing.assert_close(eager["history"][:, -1, start : start + 6], expected)
      heads = world.task_heads(
        eager["states"][:, -1],
        eager["actions"][-1],
        context,
        final_ref,
        torch.zeros(2, 29),
      )
      torch.testing.assert_close(heads["clean_observation"], eager["history"][:, -1])

  def test_new_and_legacy_checkpoint_observation_settings(self):
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / "checkpoint.pt"
      for actor_class in (MotionActor, LegacyMotionActor):
        for enabled in (False, True):
          actor = actor_class(
            self.metadata["default_q"], (8,), motion_anchor_ori_b=enabled
          )
          world = WorldModel(self.metadata, (8,), motion_anchor_ori_b=enabled)
          config = {
            "actor_dims": [8],
            "world_dims": [8],
            "world_prediction": "state_delta",
          }
          if enabled:
            config["motion_anchor_ori_b"] = True
          torch.save(
            {
              "kind": "supermimic_online",
              "schema_version": 1,
              "reference_schema": actor.reference_schema,
              "config": config,
              "robot_metadata": self.metadata,
              "simulation": {},
              "clips": self.clips,
              "manifest_sha256": file_sha256(self.manifest),
              "actor": actor.state_dict(),
              "world": world.state_dict(),
            },
            path,
          )
          loaded, _, _ = load_actor(path, self.manifest, "cpu")
          loaded_world, _ = load_world_model(path, self.manifest, "cpu")
          self.assertEqual(loaded.observation_dim, 99 if enabled else 93)
          self.assertEqual(loaded_world.motion_anchor_ori_b, enabled)
          history = torch.randn(2, 10, loaded.observation_dim)
          goal = torch.randn(2, actor.reference_dim)
          torch.testing.assert_close(
            loaded.forward_features(history, goal),
            actor.forward_features(history, goal),
          )

  @unittest.skipUnless(
    os.environ.get("SUPERMIMIC_TEST_DEVICE"), "Set simulation device"
  )
  def test_real_simulator_reset_and_successor_match_differentiable_observation(self):
    from src.tasks.supermimic.env import SuperMimicEnv

    env = SuperMimicEnv(
      FoundationConfig(manifest=self.manifest, num_envs=2, motion_anchor_ori_b=True),
      device=os.environ["SUPERMIMIC_TEST_DEVICE"],
    )
    try:
      model = WorldModel(self.metadata, (8,), motion_anchor_ori_b=True).to(env.device)
      for _ in range(2):
        state = env.read_state()
        ref = env.library.get_frame(env.motion_id, env.frame_id)
        expected = model.kinematics.observation(
          state,
          env.previous_action,
          env.context,
          ref["body_quat_w"],
          motion_anchor_orientation=True,
        )
        torch.testing.assert_close(
          env.clean_observation, expected, atol=2e-6, rtol=2e-5
        )
        self.assertEqual(env.policy_input()["history"].shape, (2, 10, 99))
        env.step(torch.zeros(2, 29, device=env.device))
    finally:
      env.close()


if __name__ == "__main__":
  unittest.main()
