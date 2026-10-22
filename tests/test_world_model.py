"""Physics oracle, gradient, split, rollout, and checkpoint tests for stage two."""

import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import mujoco
import numpy as np
import torch

from src.tasks.supermimic.data.motion_library import MotionLibrary
from src.tasks.supermimic.data.replay_buffer import ReplayBuffer
from src.tasks.supermimic.data.world_model_data import ReplayCorpus
from src.tasks.supermimic.mdp.state import quat_apply, quat_mul, rotvec_to_quat
from src.tasks.supermimic.models.kinematics import G1Kinematics, robot_metadata
from src.tasks.supermimic.models.world_model import (
  WorldModel,
  correction_targets,
  integrate,
  normalized_state_error,
)
from src.tasks.supermimic.rl.world_model_training import rollout, world_model_loss
from tests.supermimic_fixtures import make_replay

ROOT = Path(__file__).resolve().parents[1]


def poses(batch=3, dtype=torch.float32, device="cpu"):
  state = torch.randn(batch, 71, dtype=dtype, device=device) * 0.2
  state[:, 2] = 0.9
  state[:, 3:7] = rotvec_to_quat(state[:, :3])
  return state


class KinematicsTests(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    torch.set_num_threads(4)
    cls.metadata = robot_metadata()

  def test_against_mujoco_with_nonzero_joint_anchor(self):
    from mjlab.entity import Entity

    from src.tasks.supermimic.config.robot_cfg import get_supermimic_robot_cfg

    robot = Entity(get_supermimic_robot_cfg())
    model = robot.spec.compile()
    metadata = copy.deepcopy(self.metadata)
    model.jnt_pos[1] = [0.03, -0.02, 0.01]
    metadata["joint_anchor"][0] = torch.tensor([0.03, -0.02, 0.01])
    fk = G1Kinematics(metadata).double()
    states = poses(5, torch.float64)
    result = fk(states)
    data = mujoco.MjData(model)
    ids = [model.body(name).id for name in fk.body_names]
    root = int(model.jnt_bodyid[0])
    for index, state in enumerate(states.numpy()):
      data.qpos[:] = np.concatenate((state[:7], state[7:36]))
      data.qvel[:3] = state[36:39]
      inverse = state[3:7].copy()
      inverse[1:] *= -1
      mujoco.mju_rotVecQuat(data.qvel[3:6], state[39:42], inverse)
      data.qvel[6:] = state[42:71]
      mujoco.mj_kinematics(model, data)
      mujoco.mj_comPos(model, data)
      mujoco.mj_comVel(model, data)
      velocity = data.cvel[ids, 3:] + np.cross(
        data.cvel[ids, :3], data.xpos[ids] - data.subtree_com[root]
      )
      np.testing.assert_allclose(result["body_pos_w"][index], data.xpos[ids], atol=2e-7)
      np.testing.assert_allclose(result["body_lin_vel_w"][index], velocity, atol=2e-7)
      np.testing.assert_allclose(
        result["body_ang_vel_w"][index], data.cvel[ids, :3], atol=2e-7
      )
      dot = (result["body_quat_w"][index].numpy() * data.xquat[ids]).sum(-1)
      np.testing.assert_allclose(np.abs(dot), 1, atol=2e-7)

  def test_finite_difference_and_zero_rotation_gradients(self):
    fk = G1Kinematics(self.metadata).double()
    state = poses(1, torch.float64).requires_grad_()
    self.assertTrue(
      torch.autograd.gradcheck(
        lambda s: torch.cat([value.flatten(1) for value in fk(s).values()], dim=-1),
        (state,),
        fast_mode=True,
        atol=2e-5,
      )
    )
    zero = torch.zeros(1, 71, dtype=torch.float64)
    zero[:, 3] = 1
    zero.requires_grad_()
    sum(value.sum() for value in fk(zero).values()).backward()
    self.assertTrue(torch.isfinite(zero.grad).all())


class DynamicsTests(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.metadata = robot_metadata()

  def test_zero_head_is_baseline_and_residual_targets_invert(self):
    state = poses()
    model = WorldModel(self.metadata, (16,))
    action, context = torch.randn(3, 29), torch.randn(3, 33)
    torch.testing.assert_close(model(state, action, context), integrate(state))
    correction = torch.randn(3, 70) * 0.01
    torch.testing.assert_close(
      correction_targets(state, integrate(state, correction)),
      correction,
      atol=2e-7,
      rtol=1e-5,
    )

  def test_equivariance_and_quaternion_sign(self):
    model = WorldModel(self.metadata, (16,))
    torch.nn.init.normal_(model.dynamics[-1].weight, std=0.01)
    state, action, context = poses(), torch.randn(3, 29), torch.randn(3, 33)
    yaw = rotvec_to_quat(torch.tensor([[0.0, 0.0, 0.8]])).expand(3, -1)

    def transform(s):
      return torch.cat(
        (
          quat_apply(yaw, s[:, :3]) + s.new_tensor([2, -3, 0]),
          quat_mul(yaw, s[:, 3:7]),
          s[:, 7:36],
          quat_apply(yaw, s[:, 36:39]),
          quat_apply(yaw, s[:, 39:42]),
          s[:, 42:],
        ),
        dim=-1,
      )

    torch.testing.assert_close(
      model(transform(state), action, context),
      transform(model(state, action, context)),
      atol=2e-6,
      rtol=2e-5,
    )
    flipped = state.clone()
    flipped[:, 3:7] *= -1
    torch.testing.assert_close(
      model.features(state, action, context),
      model.features(flipped, action, context),
    )
    result = model(state, action, context)
    torch.testing.assert_close(result[:, 3:7].norm(dim=-1), torch.ones(3))

  def test_fit_simple_action_dependent_dynamics_and_action_gradient(self):
    torch.manual_seed(4)
    model = WorldModel(self.metadata, (32, 32))
    state = poses(128)
    action, context = torch.randn(128, 29) * 0.1, torch.zeros(128, 33)
    target = integrate(state)
    target[:, 36] += action[:, 0] * 0.7
    model.fit_normalization(state, action, context, target)
    baseline = float(normalized_state_error(integrate(state), target).mean())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    for _ in range(150):
      optimizer.zero_grad()
      loss = normalized_state_error(model(state, action, context), target).mean()
      loss.backward()
      optimizer.step()
    self.assertLess(float(loss.detach()), baseline * 0.2)
    a = action[:1].clone().requires_grad_()
    model(state[:1], a, context[:1])[:, 36].sum().backward()
    delta = torch.zeros_like(a)
    delta[:, 0] = 1e-4
    numeric = (
      model(state[:1], a.detach() + delta, context[:1])[:, 36]
      - model(state[:1], a.detach() - delta, context[:1])[:, 36]
    ) / 2e-4
    torch.testing.assert_close(a.grad[:, 0], numeric, atol=1e-3, rtol=1e-2)


class ReplayTrainingTests(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.metadata = robot_metadata()
    folder = tempfile.TemporaryDirectory()
    cls.addClassCleanup(folder.cleanup)
    cls.manifest, cls.snapshot, cls.names = make_replay(Path(folder.name), cls.metadata)

  def test_group_split_and_multiple_file_deduplication(self):
    corpus = ReplayCorpus([self.snapshot, self.snapshot], self.names)
    self.assertEqual(len(corpus.sources), 1)
    training = {
      (row["source"], row["environment"], row["episode"])
      for row in corpus.episodes
      if row["split"] == "train"
    }
    validation = {
      (row["source"], row["environment"], row["episode"])
      for row in corpus.episodes
      if row["split"] == "validation"
    }
    self.assertFalse(training & validation)
    for split in ("train", "validation"):
      rows = corpus.indices(8, split)
      batch = corpus.batch(rows[:8], 8)
      for source, time, env in rows[:8].tolist():
        self.assertEqual(
          bool(
            corpus.validation_masks[source][time % corpus.buffers[source].capacity, env]
          ),
          split == "validation",
        )
      self.assertTrue(batch["mask"][:, 0].all())
    with self.assertRaises(ValueError):
      ReplayCorpus([self.snapshot], self.names + ["absent_clip"])

  def test_normalization_train_only_autoregression_and_padding(self):
    corpus = ReplayCorpus([self.snapshot], self.names)
    model = WorldModel(self.metadata, (16,))
    transitions = corpus.training_transitions()
    model.fit_normalization(**transitions)
    expected_mean = model.features(
      transitions["state"], transitions["action"], transitions["context"]
    ).mean(0)
    torch.testing.assert_close(model.input_mean, expected_mean)
    torch.nn.init.normal_(model.dynamics[-1].weight, std=0.01)
    library = MotionLibrary(
      self.manifest,
      model.kinematics.joint_names,
      model.kinematics.body_names,
      clips=tuple(self.names),
    )
    batch = corpus.batch(corpus.indices(8, "train", True)[:2], 8)
    batch["action"].requires_grad_()
    prediction = rollout(model, batch, library)
    prediction["next_state"][:, -1, 36].sum().backward()
    self.assertGreater(float(batch["action"].grad[:, 0].abs().sum()), 0)
    batch["mask"][:, -2:] = False
    loss, _ = world_model_loss(prediction, batch)
    changed = {key: value.detach().clone() for key, value in batch.items()}
    for name in ("next_state", "next_clean_observation", "reward"):
      changed[name][:, -2:] = float("nan")
    ignored, _ = world_model_loss(prediction, changed)
    torch.testing.assert_close(loss, ignored)
    self.assertTrue(torch.isfinite(loss))

  def test_analytical_heads_match_collected_transition_labels(self):
    corpus = ReplayCorpus([self.snapshot], self.names)
    model = WorldModel(self.metadata, (16,))
    library = MotionLibrary(
      self.manifest,
      model.kinematics.joint_names,
      model.kinematics.body_names,
      clips=tuple(self.names),
    )
    batch = corpus.batch(corpus.indices(1, "train")[::7], 1)
    batch = {key: value[:, 0] for key, value in batch.items()}
    reference = library.get_frame(batch["motion_id"], batch["frame_id"] + 1)
    prediction = model.task_heads(
      batch["next_state"],
      batch["action"],
      batch["context"],
      reference,
      batch["previous_action"],
    )
    torch.testing.assert_close(
      prediction["reward"], batch["reward"], atol=2e-6, rtol=1e-4
    )
    torch.testing.assert_close(
      prediction["clean_observation"],
      batch["next_clean_observation"],
      atol=2e-6,
      rtol=1e-4,
    )

  def test_cross_backend_replay_rng_and_clip_name_remapping(self):
    saved = torch.load(self.snapshot, map_location="cpu", weights_only=True)
    saved["replay"]["generator_device_type"] = "cuda"
    saved["replay"]["generator_seed"] = 37
    first = ReplayBuffer.from_state_dict(saved["replay"])
    second = ReplayBuffer.from_state_dict(saved["replay"])
    torch.testing.assert_close(
      first.sample_windows(4)["state"], second.sample_windows(4)["state"]
    )
    self.assertEqual(first.generator.initial_seed(), 37)
    original = ReplayCorpus([self.snapshot], self.names)
    original_order = saved["clips"]
    saved["clips"] = list(reversed(original_order))
    saved["replay"]["data"]["motion_id"] = (
      len(original_order) - 1 - saved["replay"]["data"]["motion_id"]
    )
    with tempfile.TemporaryDirectory() as folder:
      path = Path(folder) / "reordered.pt"
      torch.save(saved, path)
      remapped = ReplayCorpus([path], self.names)
      count = min(original.buffers[0].steps, original.buffers[0].capacity)
      expected = original.buffers[0].data["motion_id"][:count]
      torch.testing.assert_close(
        expected, remapped.buffers[0].data["motion_id"][:count]
      )
      combined = ReplayCorpus([self.snapshot, path], self.names)
      self.assertEqual(len(combined.sources), 2)
      for buffer in combined.buffers:
        torch.testing.assert_close(buffer.data["motion_id"][:count], expected)

  def test_stable_window_indices_and_stale_index_rejection(self):
    saved = torch.load(self.snapshot, map_location="cpu", weights_only=True)
    replay = ReplayBuffer.from_state_dict(saved["replay"])
    indices = replay.window_indices(8, require_full=True)[:5]
    batch = replay.gather_windows(indices, 8)
    self.assertTrue(batch["mask"].all())
    replay.steps += replay.capacity
    with self.assertRaises(IndexError):
      replay.gather_windows(indices, 8)

  def test_checkpoint_resume_matches_uninterrupted_training(self):
    with tempfile.TemporaryDirectory() as folder:
      root = Path(folder)
      common = [
        sys.executable,
        str(ROOT / "scripts/offline/train_world_model.py"),
        "--manifest",
        str(self.manifest),
        "--replay",
        str(self.snapshot),
        "--device",
        "cpu",
        "--smoke",
        "--hidden-dims",
        "16",
        "--batch-size",
        "4",
        "--validation-windows",
        "4",
        "--validation-batch-size",
        "4",
        "--cpu-threads",
        "1",
        "--eval-interval",
        "2",
        "--checkpoint-interval",
        "2",
      ]
      environment = os.environ | {
        "MPLCONFIGDIR": str(root / "matplotlib"),
        "XDG_CACHE_HOME": str(root / "cache"),
      }
      for output, count, resume in (
        ("whole", 4, False),
        ("resumed", 2, False),
        ("resumed", 4, True),
      ):
        command = common + [
          "--steps",
          str(count),
          "--output",
          str(root / output),
        ]
        if resume:
          command += ["--resume", str(root / output / "last.pt")]
        completed = subprocess.run(
          command,
          cwd=ROOT,
          env=environment,
          capture_output=True,
          check=False,
          text=True,
          timeout=120,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
      whole = torch.load(root / "whole/last.pt", weights_only=True)
      resumed = torch.load(root / "resumed/last.pt", weights_only=True)
      for key in whole["model"]:
        torch.testing.assert_close(
          whole["model"][key], resumed["model"][key], atol=0, rtol=0
        )
      self.assertEqual(whole["split"], resumed["split"])
      torch.testing.assert_close(whole["sample_rng"], resumed["sample_rng"])
      evaluation = subprocess.run(
        common
        + [
          "--mode",
          "evaluate",
          "--checkpoint",
          str(root / "whole/last.pt"),
          "--output",
          str(root / "evaluation"),
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        check=False,
        text=True,
        timeout=120,
      )
      self.assertEqual(evaluation.returncode, 0, evaluation.stdout + evaluation.stderr)
      report = json.loads((root / "evaluation/validation.json").read_text())
      self.assertEqual(report, json.loads((root / "whole/validation.json").read_text()))
      self.assertFalse(report["qualification"]["candidate_for_bptt"])
      broken = copy.deepcopy(whole)
      broken["manifest_sha256"] = "changed"
      torch.save(broken, root / "changed.pt")
      mismatch = subprocess.run(
        common
        + [
          "--mode",
          "evaluate",
          "--checkpoint",
          str(root / "changed.pt"),
          "--output",
          str(root / "mismatch"),
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        check=False,
        text=True,
        timeout=120,
      )
      self.assertNotEqual(mismatch.returncode, 0)
      self.assertIn("does not match", mismatch.stderr)
      self.assertFalse((root / "mismatch").exists())


@unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable to this process")
class CudaTests(unittest.TestCase):
  def test_cpu_cuda_predictions_and_gradients_agree(self):
    cpu = WorldModel(robot_metadata(), (32, 32))
    torch.nn.init.normal_(cpu.dynamics[-1].weight, std=0.001)
    gpu = copy.deepcopy(cpu).cuda()
    state = poses().requires_grad_()
    action = torch.randn(3, 29).requires_grad_()
    context = torch.zeros(3, 33)
    reference = cpu(state, action, context)
    result = gpu(
      state.detach().cuda(),
      action.detach().cuda().requires_grad_(),
      context.cuda(),
    )
    torch.testing.assert_close(result.cpu(), reference, atol=1e-5, rtol=1e-4)
    reference.sum().backward()
    result.sum().backward()
    self.assertTrue(torch.isfinite(cpu.dynamics[-1].weight.grad).all())
    torch.testing.assert_close(
      gpu.dynamics[-1].weight.grad.cpu(),
      cpu.dynamics[-1].weight.grad,
      atol=1e-4,
      rtol=1e-3,
    )


if __name__ == "__main__":
  unittest.main()
