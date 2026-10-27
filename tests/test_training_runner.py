"""Scheduling, rollback and cleanup shared by both CLI entry points."""

import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from src.tasks.supermimic.rl.training_logger import ConsoleReporter, MetricWindow
from src.tasks.supermimic.rl.training_runner import TrainingRuntime
from src.tasks.supermimic.rl.velocity_logger import VelocityMetricWindow
from tests.test_training_logger import record


class TrainingRuntimeTests(unittest.TestCase):
  def setUp(self):
    folder = tempfile.TemporaryDirectory()
    self.addCleanup(folder.cleanup)
    self.output = Path(folder.name)
    self.args = SimpleNamespace(
      iterations=5,
      log_interval=2,
      log_format="table",
      verbose_sim=False,
      wandb=False,
      resume=None,
      checkpoint_interval=2,
      full_checkpoint_interval=3,
      eval_interval=2,
      wandb_project=None,
      wandb_entity=None,
      wandb_name=None,
      wandb_mode="offline",
    )
    self.console = MagicMock()

  def test_evaluation_precedes_checkpoint_and_final_round_always_saves(self):
    events = []
    trainer = SimpleNamespace(iteration=0)

    def step():
      trainer.iteration += 1
      print("simulator output")
      return record(trainer.iteration)

    trainer.train_iteration = step
    with TrainingRuntime(self.args, self.output, self.console) as runtime:
      runtime.trainer = trainer
      runtime.train(
        MetricWindow,
        checkpoint=lambda historical: events.append(
          ("save", trainer.iteration, historical)
        ),
        evaluate=lambda: events.append(("eval", trainer.iteration)),
      )
    self.assertEqual(
      events,
      [
        ("eval", 2),
        ("save", 2, False),
        ("save", 3, True),
        ("eval", 4),
        ("save", 4, False),
        ("eval", 5),
        ("save", 5, False),
      ],
    )
    self.assertEqual(self.console.training.call_count, 3)
    rows = [
      json.loads(line)
      for line in (self.output / "metrics.jsonl").read_text().splitlines()
    ]
    self.assertEqual([r["iteration"] for r in rows], list(range(1, 6)))
    self.assertIn("simulator output", (self.output / "simulator.log").read_text())

  def test_resume_trims_future_metrics_and_flushes_stage_before_joint(self):
    self.args.resume = self.output / "last.pt"
    self.args.iterations = 4
    self.args.log_interval = 10
    self.args.full_checkpoint_interval = 0
    trainer = SimpleNamespace(iteration=1, training_stage="warmup")
    (self.output / "metrics.jsonl").write_text(
      "\n".join(json.dumps({"iteration": i}) for i in (1, 2, 3)) + "\n"
    )

    def step():
      stage = trainer.training_stage
      trainer.iteration += 1
      trainer.training_stage = "joint"
      return record(trainer.iteration) | {"stage": stage, "warmup_completed": 1}

    trainer.train_iteration = step
    with TrainingRuntime(self.args, self.output, self.console) as runtime:
      runtime.trainer = trainer
      runtime.train(
        VelocityMetricWindow,
        checkpoint=lambda _: None,
        phase=lambda: trainer.training_stage,
      )
    stages = [call.args[0]["stage"] for call in self.console.training.call_args_list]
    self.assertEqual(stages, ["warmup", "joint"])
    rows = [
      json.loads(line)
      for line in (self.output / "metrics.jsonl").read_text().splitlines()
    ]
    self.assertEqual([r["iteration"] for r in rows], [1, 2, 3, 4])

  def test_exception_records_failure_and_closes_tracker_and_environment(self):
    env, tracker = MagicMock(), MagicMock()
    with (
      self.assertRaisesRegex(RuntimeError, "update failed"),
      TrainingRuntime(self.args, self.output, self.console) as runtime,
    ):
      runtime.env, runtime.tracking = env, tracker
      runtime.trainer = SimpleNamespace(
        iteration=7, training_stage="joint", failure_context={"bad": torch.ones(1)}
      )
      raise RuntimeError("update failed")
    tracker.close.assert_called_once_with(exit_code=1)
    env.close.assert_called_once()
    self.assertTrue(runtime.sim_log.closed)
    failure = json.loads((self.output / "failure.json").read_text())
    self.assertEqual((failure["iteration"], failure["phase"]), (7, "joint"))
    self.assertTrue((self.output / "critic_failure.pt").exists())

  def test_tracking_initialization_does_not_change_torch_rng(self):
    self.args.wandb = True
    before = torch.get_rng_state()

    def reporter(*args, **kwargs):
      torch.rand(10)
      return MagicMock()

    with TrainingRuntime(
      self.args, self.output, ConsoleReporter(5, stream=io.StringIO())
    ) as runtime:
      runtime.trainer = SimpleNamespace(iteration=0)
      runtime.start_tracking({}, reporter=reporter)
      torch.testing.assert_close(torch.get_rng_state(), before)
    runtime.tracking.close.assert_called_once_with(exit_code=0)
