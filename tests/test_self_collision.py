"""Excluded contact physics, measured costs, imagined gradients and recovery."""

import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np
import torch

from src.tasks.supermimic.config.robot_cfg import (
  COLLISION_SUBSTEPS,
  SELF_COLLISION_FORCE_THRESHOLD,
  get_supermimic_robot_cfg,
  self_collision_excludes,
)
from src.tasks.supermimic.config.task_cfg import STEP_DT
from src.tasks.supermimic.config.velocity_cfg import VelocityConfig
from src.tasks.supermimic.mdp.velocity import velocity_reward
from src.tasks.supermimic.models.kinematics import robot_metadata
from src.tasks.supermimic.models.world_model import WorldModel
from src.tasks.supermimic.rl.imagination import frozen_parameters
from src.tasks.supermimic.rl.inference import load_world_model, validate_environment
from src.tasks.supermimic.rl.velocity_imagination import imagine_velocity
from tests import test_velocity_training as velocity_tests


class CollisionGeometryTests(unittest.TestCase):
  def test_variants_enable_contacts_and_exclude_only_named_pairs(self):
    from mjlab.entity import Entity

    for variant in ("g1_29dof", "g1_23dof"):
      with self.subTest(variant=variant):
        model = Entity(
          get_supermimic_robot_cfg(variant, self_collisions=True)
        ).spec.compile()
        excluded = {
          frozenset((e.bodyname1, e.bodyname2))
          for e in get_supermimic_robot_cfg(variant, self_collisions=True)
          .spec_fn()
          .excludes
        }
        self.assertEqual(
          excluded, {frozenset(p) for p in self_collision_excludes(variant)}
        )
        for geom in range(model.ngeom):
          name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom)
          if name.endswith("_collision"):
            self.assertEqual(
              (model.geom_contype[geom], model.geom_conaffinity[geom]), (1, 1)
            )
          else:
            self.assertEqual(
              (model.geom_contype[geom], model.geom_conaffinity[geom]), (0, 0)
            )
        # These assets have real self contacts in legal joint configurations.
        # Find one reproducibly, and check excluded pairs never reach narrowphase.
        rng = np.random.default_rng(19)
        data = mujoco.MjData(model)
        found = False
        for _ in range(100):
          data.qpos[7:] = rng.uniform(model.jnt_range[1:, 0], model.jnt_range[1:, 1])
          mujoco.mj_forward(model, data)
          for contact in data.contact:
            pair = frozenset(
              mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g])
              for g in contact.geom
            )
            self.assertNotIn(pair, excluded)
            found = True
          if found:
            break
        self.assertTrue(found, "Enabled self contacts never produced a collision")


class CollisionTrainingTests(velocity_tests.VelocityTrainingTests):
  """Run inherited collection/recovery checks with collision physics as well."""

  self_collisions = True

  # The original equality check predates the optional contact reward argument.
  def test_simulation_reward_matches_analytic_and_command_transition_order(self):
    trainer = self.trainer()
    _, _, _, _, extras = self.env.step(torch.zeros(4, 29))
    transition = extras["velocity_transition"]
    forces = self.env.scene["self_collision"].data.force_history
    measured = (
      (forces.norm(dim=-1) > SELF_COLLISION_FORCE_THRESHOLD).any(1).sum(-1).float()
    )
    torch.testing.assert_close(extras["transition"]["self_collision_cost"], measured)
    self.assertFalse(
      self.env.sim.mj_model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_SENSOR
    )
    reward, terms = velocity_reward(
      transition["next_state"],
      trainer.world.kinematics(transition["next_state"]),
      transition["command"],
      transition["action"],
      transition["previous_action"],
      trainer.world.kinematics.joint_limits,
      trainer.actor.default_q,
      self.velocity,
      self_collision_cost=transition["self_collision_cost"],
    )
    torch.testing.assert_close(reward, transition["reward"], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(
      terms["self_collisions"],
      measured[2:] * STEP_DT * self.velocity.self_collision_weight,
    )

  def test_cost_head_receives_labels_and_penalty_reaches_encoder(self):
    trainer = self.trainer()
    trainer.bootstrap()
    before = {
      k: v.clone() for k, v in trainer.world.collision_head.state_dict().items()
    }
    metrics = trainer.train_iteration()
    self.assertGreater(metrics["wm_collision"], 0)
    self.assertTrue(
      any(
        not torch.equal(v, before[k])
        for k, v in trainer.world.collision_head.state_dict().items()
      )
    )
    starts = trainer.velocity_replay.sample_starts(2, 10)
    kwargs = {"horizon": 3, "sensor_noise": False}
    with frozen_parameters(trainer.actor, trainer.world, trainer.velocity_critic):
      rng = torch.Generator().manual_seed(4)
      trace = imagine_velocity(
        trainer.actor,
        trainer.velocity_encoder,
        trainer.world,
        trainer.velocity_critic,
        self.env.commands,
        starts,
        generator=rng,
        **kwargs,
      )
      commands = self.env.commands
      commands.cfg = replace(self.velocity, self_collision_weight=0.0)
      try:
        without = imagine_velocity(
          trainer.actor,
          trainer.velocity_encoder,
          trainer.world,
          trainer.velocity_critic,
          commands,
          starts,
          generator=torch.Generator().manual_seed(4),
          **kwargs,
        )
      finally:
        commands.cfg = self.velocity
      torch.testing.assert_close(trace["states"], without["states"])
      difference = without["rewards"] - trace["rewards"]
      self.assertTrue((difference > 0).all())
      difference.sum().backward()
      gradient = sum(
        float(p.grad.abs().sum())
        for p in trainer.velocity_encoder.parameters()
        if p.grad is not None
      )
      self.assertGreater(gradient, 0)
      self.assertTrue(all(p.grad is None for p in trainer.world.parameters()))

  def test_colliding_pose_produces_nonzero_physics_penalty(self):
    rng = np.random.default_rng(19)
    model = self.env.sim.mj_model
    joint_ids = [model.joint(f"robot/{name}").id for name in self.env.robot.joint_names]
    bounds = model.jnt_range[joint_ids]
    q = torch.tensor(rng.uniform(bounds[:, 0], bounds[:, 1]), dtype=torch.float32)
    ids = torch.arange(self.env.num_envs)
    root = self.env.robot.data.default_root_state.clone()
    root[:, 2] = 1.0
    self.env._apply_reset_state(ids, root, q.expand(4, -1), torch.zeros(4, 29))
    _, _, _, _, extras = self.env.step(torch.zeros(4, 29))
    cost = extras["transition"]["self_collision_cost"]
    self.assertTrue((cost > 0).any(), "Self contact never reached the force threshold")
    self.assertTrue((cost <= COLLISION_SUBSTEPS).all())
    self.assertTrue((extras["metrics"]["velocity/self_collisions"] < 0).any())

  def test_physics_transfer_is_only_allowed_for_initialization(self):
    trainer = self.trainer()
    old = dict(trainer.velocity_policy_state_dict())
    old["simulation"] = {
      k: v for k, v in old["simulation"].items() if not k.startswith("self_collision")
    } | {"self_collisions": False}
    with self.assertRaisesRegex(ValueError, "simulation configuration"):
      validate_environment(self.env, old)
    self.assertTrue(validate_environment(self.env, old, allow_collision_transfer=True))
    old["simulation"]["physics_dt"] *= 2
    with self.assertRaisesRegex(ValueError, "simulation configuration"):
      validate_environment(self.env, old, allow_collision_transfer=True)

  def test_collision_world_export_and_strict_resume(self):
    from src.tasks.supermimic.rl.checkpoints import world_model_checkpoint

    trainer = self.trainer()
    trainer.bootstrap()
    trainer.train_iteration()
    saved = trainer.state_dict()
    with tempfile.TemporaryDirectory() as folder:
      path = Path(folder) / "world.pt"
      # The regular exporter accepts the common training kind.
      saved["kind"] = "supermimic_online"
      torch.save(world_model_checkpoint(saved), path)
      world, _ = load_world_model(path, None, "cpu")
      self.assertTrue(world.self_collisions)
      for key, value in trainer.world.state_dict().items():
        torch.testing.assert_close(world.state_dict()[key], value)


class CollisionLossTests(unittest.TestCase):
  def test_substep_cost_scaling_and_head_supervision(self):
    from src.tasks.supermimic.rl.world_model_training import world_model_loss

    torch.set_num_threads(2)
    metadata = robot_metadata()
    world = WorldModel(metadata, (16, 16), self_collisions=True)
    state = torch.zeros(2, 1, 71)
    state[..., 2], state[..., 3] = 0.8, 1
    action, context = torch.zeros(2, 1, 29), torch.zeros(2, 1, 33)
    labels = torch.tensor([[0.0], [float(COLLISION_SUBSTEPS)]])
    # Distinguishable physical states, including a fully colliding control step.
    state[1, :, 7] = 1
    batch = {
      "mask": torch.ones(2, 1, dtype=torch.bool),
      "next_state": state,
      "self_collision_cost": labels,
    }
    optimizer = torch.optim.Adam(world.collision_head.parameters(), lr=0.01)
    losses = []
    for _ in range(50):
      optimizer.zero_grad()
      prediction = {
        "next_state": state,
        "collision_logit": world.collision_logit(state, action, context),
      }
      loss, parts = world_model_loss(prediction, batch, analytic_losses=False)
      loss.backward()
      optimizer.step()
      losses.append(float(parts["collision"].detach()))
    self.assertLess(losses[-1], losses[0] / 10)
    torch.testing.assert_close(
      world.collision_cost(state, action, context), labels, atol=0.1, rtol=0
    )
    for weight in (1, float("nan"), float("inf")):
      with self.assertRaises(ValueError):
        VelocityConfig(self_collision_weight=weight)


class CollisionCliTests(unittest.TestCase):
  def test_pretrained_migration_playback_and_resume_contract(self):
    # Use a separate baseline environment to create genuine old-physics weights.
    from src.tasks.supermimic.config.rl_cfg import OnlineConfig
    from src.tasks.supermimic.config.task_cfg import FoundationConfig
    from src.tasks.supermimic.rl.velocity_training import VelocityTrainer
    from src.tasks.supermimic.velocity_env import VelocityEnv
    from tests.supermimic_fixtures import make_motions

    torch.set_num_threads(2)
    temporary = tempfile.TemporaryDirectory()
    self.addCleanup(temporary.cleanup)
    self.root = Path(temporary.name)
    metadata = robot_metadata()
    self.manifest, self.clips = make_motions(self.root / "motions", metadata)
    old_env = VelocityEnv(
      FoundationConfig(manifest=self.manifest, num_envs=4),
      VelocityConfig(),
      "cpu",
      motion_envs=2,
      training=True,
    )
    try:
      old_config = OnlineConfig(
        num_envs=4,
        world_dims=(16, 16),
        actor_dims=(16, 8),
        critic_dims=(16, 8),
        actor_input_normalization=True,
        world_model_normalization=True,
        reference_dim=2680,
        amp="none",
        compile=False,
        policy_batch_size=4,
        critic_minibatches=1,
      )
      old = VelocityTrainer(old_env, metadata, old_config)
      actor, world = self.root / "init_actor.pt", self.root / "init_world.pt"
      torch.save(old.policy_state_dict(), actor)
      torch.save(old.world_model_state_dict(), world)
    finally:
      old_env.close()
    with tempfile.TemporaryDirectory() as folder:
      output = Path(folder) / "run"
      command = [
        sys.executable,
        "scripts/train_velocity.py",
        "--device",
        "cpu",
        "--manifest",
        str(self.manifest),
        "--actor-checkpoint",
        str(actor),
        "--world-checkpoint",
        str(world),
        "--output",
        str(output),
        "--num-envs",
        "4",
        "--capacity",
        "32",
        "--warmup-steps",
        "12",
        "--collect-steps",
        "3",
        "--updates-per-collect",
        "1",
        "--wm-batches",
        "1",
        "--wm-batch-size",
        "4",
        "--policy-batch-size",
        "4",
        "--horizon",
        "3",
        "--critic-epochs",
        "1",
        "--critic-minibatches",
        "1",
        "--warmup-updates",
        "1",
        "--initial-evaluation-steps",
        "0",
        "--iterations",
        "2",
        "--clips",
        *self.clips,
      ]
      frozen = subprocess.run(
        command + ["--freeze-world-model"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
      )
      self.assertEqual(frozen.returncode, 2, frozen.stdout + frozen.stderr)
      self.assertIn("online adaptation", frozen.stderr)
      self.assertFalse(output.exists())
      result = subprocess.run(
        command, capture_output=True, text=True, timeout=120, check=False
      )
      self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
      saved = torch.load(output / "last.pt", weights_only=True)
      self.assertTrue(saved["config"]["self_collisions"])
      self.assertEqual(
        saved["initialization_collision_transfer"], {"actor": True, "world": True}
      )
      self.assertIn("self_collision_cost", saved["velocity_replay"]["data"])
      self.assertTrue(any(k.startswith("collision_head.") for k in saved["world"]))
      resume = [
        sys.executable,
        "scripts/train_velocity.py",
        "--resume",
        str(output / "last.pt"),
        "--device",
        "cpu",
        "--iterations",
        "3",
        "--manifest",
        str(self.manifest),
      ]
      invalid = subprocess.run(
        resume + ["--no-self-collisions"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
      )
      self.assertEqual(invalid.returncode, 2, invalid.stdout + invalid.stderr)
      result = subprocess.run(
        resume, capture_output=True, text=True, timeout=120, check=False
      )
      self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
      saved = torch.load(output / "last.pt", weights_only=True)
      self.assertEqual(saved["iteration"], 3)
      result = subprocess.run(
        [
          sys.executable,
          "scripts/play_velocity.py",
          "--checkpoint",
          str(output / "velocity_policy_last.pt"),
          "--device",
          "cpu",
          "--headless",
          "--steps",
          "3",
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
      )
      self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
      diagnostic = json.loads((output / "velocity_diagnostics.json").read_text())
      self.assertIn("self_collision_cost_mae", diagnostic["horizons"]["1"])


@unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
class CollisionCudaTests(unittest.TestCase):
  def test_cuda_both_variants_update_collision_head_and_export(self):
    from src.tasks.supermimic.config.rl_cfg import OnlineConfig
    from src.tasks.supermimic.config.task_cfg import FoundationConfig
    from src.tasks.supermimic.rl.velocity_training import VelocityTrainer
    from src.tasks.supermimic.velocity_env import VelocityEnv
    from tests.supermimic_fixtures import make_motions

    torch.set_num_threads(2)
    for variant in ("g1_29dof", "g1_23dof"):
      with self.subTest(variant=variant), tempfile.TemporaryDirectory() as folder:
        metadata = robot_metadata(variant, self_collisions=True)
        manifest, clips = make_motions(Path(folder), metadata)
        cfg = OnlineConfig(
          robot=variant,
          self_collisions=True,
          num_envs=4,
          capacity=32,
          warmup_steps=12,
          collect_steps=3,
          updates_per_collect=1,
          wm_batches=1,
          wm_batch_size=4,
          policy_batch_size=4,
          horizon=3,
          critic_epochs=1,
          critic_minibatches=1,
          world_dims=(16, 16),
          actor_dims=(16, 8),
          critic_dims=(16, 8),
          amp="bf16",
          compile=False,
          world_model_normalization=True,
          clips=tuple(clips),
        )
        env = VelocityEnv(
          FoundationConfig(
            manifest=manifest, robot=variant, self_collisions=True, num_envs=4
          ),
          VelocityConfig(warmup_updates=1),
          "cuda:0",
          motion_envs=2,
          training=True,
        )
        try:
          trainer = VelocityTrainer(env, metadata, cfg, total_iterations=2)
          trainer.bootstrap()
          trainer.train_iteration()
          record = trainer.train_iteration()
          self.assertEqual(record["stage"], "joint")
          self.assertGreater(record["wm_collision"], 0)
          self.assertGreater(record["velocity/actor_gradient_norm"], 0)
          world = trainer.world
          s, a, c = [
            trainer.velocity_replay.data[k][0] for k in ("state", "action", "context")
          ]
          compiled = torch.compile(
            world.collision_cost, backend="eager", fullgraph=True
          )
          torch.testing.assert_close(compiled(s, a, c), world.collision_cost(s, a, c))
          saved = trainer.state_dict()
          trainer.load_state_dict(saved)
          self.assertTrue(trainer.simulation_parameters["self_collisions"])
          model = env.sim.mj_model
          joint_ids = [
            model.joint(f"robot/{name}").id for name in env.robot.joint_names
          ]
          bounds = model.jnt_range[joint_ids]
          q = torch.tensor(
            np.random.default_rng(19).uniform(bounds[:, 0], bounds[:, 1]),
            device=env.device,
            dtype=torch.float32,
          )
          root = env.robot.data.default_root_state.clone()
          root[:, 2] = 1.0
          env._apply_reset_state(
            torch.arange(4, device=env.device),
            root,
            q.expand(4, -1),
            torch.zeros(4, len(q), device=env.device),
          )
          _, _, _, _, extras = env.step(torch.zeros(4, len(q), device=env.device))
          self.assertTrue((extras["transition"]["self_collision_cost"] > 0).any())
          self.assertTrue((extras["metrics"]["velocity/self_collisions"] < 0).any())
        finally:
          env.close()


if __name__ == "__main__":
  unittest.main()
