"""Explicit NPZ sources share manifest loading and evaluation behavior."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.tasks.supermimic.config.task_cfg import FoundationConfig
from src.tasks.supermimic.data.motion_library import MotionLibrary
from src.tasks.supermimic.data.motion_sources import motion_entries, select_motion_entries
from src.tasks.supermimic.models.kinematics import robot_metadata, G1Kinematics
from src.tasks.supermimic.rl.evaluation import evaluation_env_count
from tests.supermimic_fixtures import make_motions

ROOT = Path(__file__).resolve().parents[1]


class MotionSourceTests(unittest.TestCase):
  def test_direct_sources_match_manifest_for_both_robots(self):
    for robot in ('g1_23dof', 'g1_29dof'):
      with self.subTest(robot=robot), tempfile.TemporaryDirectory() as folder:
        metadata = robot_metadata(robot)
        manifest, names = make_motions(Path(folder), metadata)
        fk = G1Kinematics(metadata)
        args = (fk.joint_names, fk.body_names)
        original = MotionLibrary(manifest, *args)
        selected = MotionLibrary(manifest, *args, clips=(names[0],))
        manifest.unlink()
        directory = MotionLibrary(None, *args, motion_dir=Path(folder))
        single = MotionLibrary(None, *args, motion=Path(folder) / f'{names[0]}.npz')
        self.assertEqual(directory.names, sorted(names))
        for key in original.data:
          torch.testing.assert_close(single.data[key], selected.data[key], rtol=0, atol=0)
          for name in names:
            old = original.names.index(name)
            new = directory.names.index(name)
            length = int(original.lengths[old])
            torch.testing.assert_close(
              directory.data[key][directory.offsets[new]:directory.offsets[new] + length],
              original.data[key][original.offsets[old]:original.offsets[old] + length],
              rtol=0, atol=0,
            )
        config = FoundationConfig(robot=robot, motion_dir=Path(folder))
        self.assertIsNone(config.manifest)
        self.assertEqual(evaluation_env_count(None, motion_dir=Path(folder)), 2)
        self.assertEqual(evaluation_env_count(None, motion=single_path(Path(folder), names[0])), 1)

  def test_recursive_order_selection_and_errors(self):
    with tempfile.TemporaryDirectory() as folder:
      root = Path(folder)
      (root / 'not_a_file.npz').mkdir()
      with self.assertRaisesRegex(ValueError, 'No NPZ'):
        motion_entries(motion_dir=root)
      for relative in ('z/a.npz', 'a.npz', 'nested/a.npz'):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
      entries = motion_entries(motion_dir=root)
      self.assertEqual([e.name for e in entries], ['a', 'nested/a', 'z/a'])
      self.assertEqual([e.name for e in select_motion_entries(entries, ('nested/a.npz',))], ['nested/a'])
      for kwargs in ({}, {'motion': root / 'missing.npz'}, {'motion_dir': root / 'missing'},
                     {'motion': root / 'a.npz', 'motion_dir': root}):
        with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
          motion_entries(**kwargs)
      with self.assertRaisesRegex(ValueError, 'Unknown clips'):
        select_motion_entries(entries, ('missing',))
      with self.assertRaises(ValueError):
        FoundationConfig(manifest=root / 'manifest.json', motion_dir=root)

  def test_direct_npz_keeps_metadata_and_array_validation(self):
    with tempfile.TemporaryDirectory() as folder:
      metadata = robot_metadata()
      manifest, names = make_motions(Path(folder), metadata)
      manifest.unlink()
      path = single_path(Path(folder), names[0])
      with np.load(path) as source:
        valid = dict(source)
      fk = G1Kinematics(metadata)
      for patch in ({'fps': np.array(30)}, {'quaternion_order': np.array('xyzw')},
                    {'joint_pos': valid['joint_pos'][:-1]},
                    {'joint_vel': np.full_like(valid['joint_vel'], np.nan)}):
        np.savez(path, **(valid | patch))
        with self.subTest(fields=list(patch)), self.assertRaises(ValueError):
          MotionLibrary(None, fk.joint_names, fk.body_names, motion=path)
      np.savez(path, **{k: v for k, v in valid.items() if k != 'fps'})
      with self.assertRaisesRegex(ValueError, 'lacks required fields: fps'):
        MotionLibrary(None, fk.joint_names, fk.body_names, motion=path)

  def test_cli_requires_exactly_one_source(self):
    for script in ('evaluate_supermimic.py', 'play_supermimic.py'):
      for args in ([], ['--motion', 'a.npz', '--motion-dir', 'motions']):
        result = subprocess.run(
          [sys.executable, str(ROOT / 'scripts' / script), *args],
          capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn('--motion', result.stderr)

  def test_real_evaluation_and_playback_without_manifest(self):
    from dataclasses import asdict
    from src.tasks.supermimic.config.rl_cfg import OnlineConfig
    from src.tasks.supermimic.env import SuperMimicEnv
    from src.tasks.supermimic.models.policy import MotionActor, REFERENCE_SCHEMA
    from src.tasks.supermimic.models.world_model import WorldModel
    from src.tasks.supermimic.rl.checkpoints import policy_checkpoint, world_model_checkpoint
    from src.tasks.supermimic.rl.online_training import simulation_parameters

    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory() as folder:
      root = Path(folder)
      metadata = robot_metadata()
      manifest, names = make_motions(root / 'motions', metadata)
      env = SuperMimicEnv(FoundationConfig(manifest=manifest, num_envs=1), device='cpu')
      try:
        simulation = simulation_parameters(env)
      finally:
        env.close()
      cfg = OnlineConfig(actor_dims=(8,), world_dims=(8,), amp='none')
      actor = MotionActor(metadata['default_q'], (8,))
      world = WorldModel(metadata, (8,))
      saved = dict(
        kind='supermimic_online', schema_version=1, reference_schema=REFERENCE_SCHEMA,
        config=asdict(cfg), robot_metadata=metadata, simulation=simulation,
        clips=names, actor=actor.state_dict(), world=world.state_dict(),
      )
      torch.save(saved, root / 'full.pt')
      torch.save(policy_checkpoint(saved), root / 'policy.pt')
      torch.save(world_model_checkpoint(saved), root / 'world.pt')

      def run(script, source, checkpoint, extra=()):
        output = root / 'report.json'
        result = subprocess.run([
          sys.executable, str(ROOT / 'scripts' / script), *source,
          '--device', 'cpu', '--checkpoint', str(root / checkpoint),
          '--viewer', 'none', '--num-envs', '1', '--max-steps-per-clip', '2',
          '--output', str(output), *extra,
        ], capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return json.loads(output.read_text())

      baseline = run('evaluate_supermimic.py', ['--manifest', str(manifest)], 'full.pt')
      manifest.unlink()
      directory = run(
        'evaluate_supermimic.py', ['--motion-dir', str(manifest.parent)], 'policy.pt',
        ['--world-model', str(root / 'world.pt'), '--world-model-mode', 'compare'],
      )
      by_name = lambda report: {row['motion']: row for row in report['motions']}
      self.assertEqual(by_name(baseline), by_name(directory))
      played = run(
        'play_supermimic.py', ['--motion', str(single_path(manifest.parent, names[0]))],
        'full.pt', ['--termination', '--world-model', str(root / 'world.pt')],
      )
      self.assertEqual(played['motions'], baseline['motions'][:1])


def single_path(root, name):
  return root / f'{name}.npz'
