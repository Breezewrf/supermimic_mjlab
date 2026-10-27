"""Variant integration: CSV projection, differentiable dynamics and CPU training."""

import argparse
import hashlib
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/data"))
from convert_lafan_to_npz import (
  CSV_JOINT_NAMES,
  load_robot,
  motion_to_arrays,
)

from src.tasks.supermimic.config.cli import add_robot_argument, resolve_robot_arguments
from src.tasks.supermimic.config.morphology import RobotLayout, robot_layout
from src.tasks.supermimic.config.rl_cfg import OnlineConfig
from src.tasks.supermimic.config.task_cfg import FoundationConfig
from src.tasks.supermimic.config.velocity_cfg import VelocityConfig
from src.tasks.supermimic.env import SuperMimicEnv
from src.tasks.supermimic.mdp.state import add_sensor_noise, observe, rotvec_to_quat
from src.tasks.supermimic.models.kinematics import G1Kinematics, robot_metadata
from src.tasks.supermimic.models.policy import MotionActor, MotionCritic
from src.tasks.supermimic.models.world_model import (
  WorldModel,
  apply_state_delta,
  state_delta_targets,
)
from src.tasks.supermimic.rl.checkpoints import training_config
from src.tasks.supermimic.rl.execution import TrainingKernels
from src.tasks.supermimic.rl.imagination import imagination_step
from src.tasks.supermimic.rl.inference import (
  load_actor,
  load_world_model,
  validate_environment,
)
from src.tasks.supermimic.rl.online_training import OnlineTrainer
from src.tasks.supermimic.rl.training_logger import simulator_output
from src.tasks.supermimic.rl.velocity_inference import load_velocity_actor
from src.tasks.supermimic.rl.velocity_training import VelocityTrainer
from src.tasks.supermimic.velocity_env import VelocityEnv
from tests.supermimic_fixtures import make_motions


class RobotVariantTests(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    torch.set_num_threads(2)

  def test_csv_projection_matches_fk_and_velocities_for_both_robots(self):
    for robot in ("g1_29dof", "g1_23dof"):
      with self.subTest(robot=robot):
        layout = robot_layout(robot)
        model, joints, bodies, _ = load_robot(robot=robot)
        self.assertEqual(
          (model.nq, model.nv, model.nu),
          (7 + layout.num_joints, 6 + layout.num_joints, layout.num_joints),
        )
        # Exercise distinct joint values, including the six omitted joints.
        motion = np.zeros((5, 36))
        motion[:, 2] = 0.8
        motion[:, 3:7] = rotvec_to_quat(torch.tensor([0.1, 0.2, 0.3])).numpy()
        motion[:, 7:] = np.arange(29)[None] * 0.01 + np.arange(5)[:, None] * 0.001
        arrays = motion_to_arrays(model, joints, bodies, motion, 1000.0)
        for i, name in enumerate(joints):
          np.testing.assert_allclose(
            arrays["joint_pos"][:, i],
            motion[:, 7 + CSV_JOINT_NAMES.index(name)],
            atol=1e-7,
          )
        metadata = robot_metadata(robot)
        state = torch.zeros(5, layout.state_dim)
        state[:, :7] = torch.from_numpy(motion[:, :7])
        state[:, layout.q] = torch.from_numpy(arrays["joint_pos"])
        state[:, layout.qd] = torch.from_numpy(arrays["joint_vel"])
        actual = G1Kinematics(metadata, bodies)(state)
        for field, value in actual.items():
          np.testing.assert_allclose(value.detach().numpy(), arrays[field], atol=2e-6)
        with self.assertRaisesRegex(ValueError, "selected robot"):
          load_robot(layout.xml, "g1_23dof" if robot == "g1_29dof" else "g1_29dof")

  def test_observation_options_and_world_gradients(self):
    for n in (23, 29):
      layout = RobotLayout(n)
      metadata = robot_metadata(layout.robot)
      state = torch.randn(2, layout.state_dim) * 0.02
      state[:, 3:7] = rotvec_to_quat(torch.randn(2, 3) * 0.1)
      action = torch.randn(2, n, requires_grad=True)
      context = torch.zeros(2, layout.context_dim)
      default_q = metadata["default_q"].float()
      for relative in (False, True):
        for anchor in (False, True):
          config = FoundationConfig(
            robot=layout.robot,
            relative_orientation=relative,
            motion_anchor_ori_b=anchor,
          )
          clean = observe(
            state,
            action,
            default_q,
            context,
            state[:, 3:7] if relative else None,
            torch.zeros(2, 6) if anchor else None,
          )
          self.assertEqual(clean.shape[-1], config.observation_dim)
          noisy = add_sensor_noise(clean, num_joints=n)
          torch.testing.assert_close(noisy[:, -n:], action)
          actor = MotionActor(
            default_q, (8,), relative_orientation=relative, motion_anchor_ori_b=anchor
          )
          self.assertEqual(
            actor.decode(clean[:, None].expand(-1, 10, -1), torch.zeros(2, 32)).shape,
            (2, n),
          )
      delta = torch.randn(2, layout.state_dim - 1) * 0.01
      next_state = apply_state_delta(state, delta)
      torch.testing.assert_close(
        state_delta_targets(state, next_state), delta, atol=1e-6, rtol=1e-5
      )
      for mode in ("state_delta", "velocity_residual"):
        for normalization in (False, True):
          world = WorldModel(
            metadata,
            (8,),
            world_prediction=mode,
            world_model_normalization=normalization,
          )
          world.fit_normalization(state, action.detach(), context, next_state)
          with torch.no_grad():
            world.dynamics[-1].weight.normal_(std=0.01)
          result = world(state, action, context)
          self.assertEqual(result.shape, state.shape)
          grad = torch.autograd.grad(result.square().sum(), action, retain_graph=True)[
            0
          ]
          self.assertTrue(torch.isfinite(grad).all())
          self.assertGreater(float(grad.abs().sum()), 0)

  def test_cpu_training_resume_exports_and_cross_robot_rejection(self):
    robot = "g1_23dof"
    metadata = robot_metadata(robot)
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      manifest, names = make_motions(root / "motions", metadata)
      config = OnlineConfig(
        robot=robot,
        num_envs=4,
        capacity=32,
        warmup_steps=12,
        collect_steps=2,
        updates_per_collect=1,
        wm_batches=1,
        wm_batch_size=4,
        policy_batch_size=4,
        horizon=2,
        critic_epochs=1,
        critic_minibatches=1,
        world_dims=(8,),
        actor_dims=(8,),
        critic_dims=(8,),
        amp="none",
        compile=False,
        clips=tuple(names),
        world_model_normalization=True,
      )
      for velocity in (False, True):
        with (
          self.subTest(velocity=velocity),
          (root / "sim.log").open("a") as log,
          simulator_output(log, False),
        ):
          foundation = FoundationConfig(
            robot=robot, manifest=manifest, num_envs=4, clips=tuple(names)
          )
          env = (
            VelocityEnv(
              foundation,
              replace(VelocityConfig(), warmup_updates=0),
              device="cpu",
              motion_envs=2,
              training=True,
            )
            if velocity
            else SuperMimicEnv(foundation, device="cpu")
          )
          try:
            trainer = (
              VelocityTrainer(env, metadata, config, total_iterations=3)
              if velocity
              else OnlineTrainer(env, metadata, config, total_iterations=3)
            )
            trainer.bootstrap()
            record = trainer.train_iteration()
            self.assertEqual(record["iteration"], 1)
            saved = trainer.state_dict()
            trainer.load_state_dict(saved)
            policy = (
              trainer.velocity_policy_state_dict()
              if velocity
              else trainer.policy_state_dict()
            ) | {"manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()}
            path = root / "policy.pt"
            torch.save(policy, path)
            if velocity:
              loaded, export = load_velocity_actor(path)
              inputs = env.policy_input()
              self.assertEqual(
                loaded(inputs["history"][env.velocity_ids], inputs["command"]).shape,
                (2, 23),
              )
            else:
              loaded, export, _ = load_actor(path, manifest, "cpu")
              inputs = env.policy_input()
              self.assertEqual(loaded(**inputs).shape, (4, 23))
              self.assertEqual(training_config(export)["robot"], robot)
            validate_environment(env, export)
            world = trainer.world_model_state_dict() | {
              "manifest_sha256": policy["manifest_sha256"]
            }
            torch.save(world, root / "world.pt")
            loaded_world, _ = load_world_model(root / "world.pt", manifest, "cpu")
            self.assertEqual(
              loaded_world(env.read_state(), torch.zeros(4, 23), env.context).shape,
              (4, 59),
            )
            invalid = {**export, "robot_metadata": robot_metadata("g1_29dof")}
            with self.assertRaisesRegex(ValueError, "robot model"):
              validate_environment(env, invalid)
            parser = argparse.ArgumentParser()
            add_robot_argument(parser)
            parser.add_argument("--checkpoint", type=Path)
            args = parser.parse_args(["--checkpoint", str(path)])
            resolve_robot_arguments(args, parser)
            self.assertEqual(args.robot, robot)
          finally:
            env.close()

  @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
  def test_cuda_training_both_variants_and_interfaces(self):
    for robot in ("g1_23dof", "g1_29dof"):
      metadata = robot_metadata(robot)
      with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        manifest, names = make_motions(root / "motions", metadata)
        config = OnlineConfig(
          robot=robot,
          num_envs=4,
          capacity=32,
          warmup_steps=12,
          collect_steps=2,
          updates_per_collect=1,
          wm_batches=1,
          wm_batch_size=4,
          policy_batch_size=4,
          horizon=2,
          critic_epochs=1,
          critic_minibatches=1,
          world_dims=(16,),
          actor_dims=(16,),
          critic_dims=(16,),
          amp="bf16",
          compile=False,
          clips=tuple(names),
          world_model_normalization=True,
        )
        for velocity in (False, True):
          with (
            self.subTest(robot=robot, velocity=velocity),
            (root / "sim.log").open("a") as log,
            simulator_output(log, False),
          ):
            foundation = FoundationConfig.training(
              robot=robot, manifest=manifest, num_envs=4, clips=tuple(names)
            )
            env = (
              VelocityEnv(
                foundation,
                replace(VelocityConfig(), warmup_updates=0),
                device="cuda:0",
                motion_envs=2,
                training=True,
              )
              if velocity
              else SuperMimicEnv(foundation, device="cuda:0")
            )
            try:
              trainer = (
                VelocityTrainer(env, metadata, config, total_iterations=3)
                if velocity
                else OnlineTrainer(env, metadata, config, total_iterations=3)
              )
              trainer.bootstrap()
              record = trainer.train_iteration()
              self.assertEqual(record["iteration"], 1)
              self.assertTrue(torch.isfinite(env.read_state()).all())
            finally:
              env.close()

  def test_full_body_reference_and_compiled_tensor_functions(self):
    metadata = robot_metadata("g1_23dof")
    layout = RobotLayout(23)
    with tempfile.TemporaryDirectory() as directory:
      manifest, _ = make_motions(Path(directory), metadata)
      from src.tasks.supermimic.data.motion_library import MotionLibrary

      kinematics = G1Kinematics(metadata)
      library = MotionLibrary(manifest, kinematics.joint_names, kinematics.body_names)
      reference = library.get_future(torch.tensor([0]), torch.tensor([0]), 10)
      state = torch.zeros(1, layout.state_dim)
      state[:, 3] = 1
      state[:, layout.q] = metadata["default_q"].float()
      action, context = torch.zeros(1, 23), torch.zeros(1, 27)
      for width in (610, 2560):
        actor = MotionActor(metadata["default_q"], (8,), reference_dim=width)
        critic = MotionCritic(metadata["default_q"], (8,), reference_dim=width)
        world = WorldModel(metadata, (8,))
        kernels = TrainingKernels(actor, world, critic, enabled=True, backend="eager")
        goal = kernels.goals(reference, actor.default_q)
        self.assertEqual(goal.shape, (1, width))
        history = observe(state, action, actor.default_q, context)[:, None].expand(
          -1, 10, -1
        )
        args = (state, history, goal, context, reference["body_quat_w"][:, 0])
        eager = imagination_step(actor, world, *args)
        compiled = kernels.imag_step(*args)
        for expected, actual in zip(eager, compiled):
          torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
  unittest.main()
