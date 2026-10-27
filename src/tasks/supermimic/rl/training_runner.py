"""Shared logging, lifecycle and scheduling for online training entry points."""

import importlib.util
import json
import traceback

from .checkpoints import save_checkpoint
from .metric_schema import metric_groups
from .online_training import restore_rng, rng_state
from .training_logger import WandbReporter, simulator_output


def write_json(path, data):
  path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def add_logging_arguments(parser):
  parser.add_argument("--log-interval", type=int, default=10)
  parser.add_argument("--log-format", choices=("table", "json"), default="table")
  parser.add_argument("--verbose-sim", action="store_true")
  parser.add_argument(
    "--wandb", action="store_true", help="Track training with Weights & Biases"
  )
  parser.add_argument(
    "--wandb-project", help="Default: supermimic, or saved project on resume"
  )
  parser.add_argument("--wandb-entity", help="W&B user or team")
  parser.add_argument("--wandb-name", help="Default: output directory name")
  parser.add_argument("--wandb-mode", choices=("online", "offline"), default="online")


def validate_logging_arguments(parser, args):
  if args.wandb and importlib.util.find_spec("wandb") is None:
    parser.error(
      "W&B is optional. Install logging support with: uv sync --extra logging"
    )


class TrainingRuntime:
  """Own the simulator log, tracker and cleanup for either training task."""

  def __init__(self, args, output, console):
    self.args, self.output, self.console = args, output, console
    self.trainer = self.env = self.tracking = None

  def __enter__(self):
    self.sim_log = (self.output / "simulator.log").open("a", buffering=1)
    return self

  def start_tracking(self, config, reporter=WandbReporter):
    if not self.args.wandb:
      return None
    state = rng_state()
    try:
      self.tracking = reporter(
        self.output,
        config,
        self.trainer.iteration,
        project=self.args.wandb_project,
        entity=self.args.wandb_entity,
        name=self.args.wandb_name,
        mode=self.args.wandb_mode,
        console=self.console,
      )
    finally:
      restore_rng(state)
    return self.tracking

  def __exit__(self, error_type, error, tb):
    try:
      if error is not None:
        context = getattr(self.trainer, "failure_context", None)
        if context is not None:
          save_checkpoint(self.output / "critic_failure.pt", context)
        (self.output / "failure.json").write_text(
          json.dumps(
            {
              "iteration": getattr(self.trainer, "iteration", None),
              "phase": getattr(
                self.trainer,
                "training_stage",
                getattr(self.trainer, "phase", "initialization"),
              ),
              "error": str(error),
              "traceback": "".join(traceback.format_exception(error_type, error, tb)),
            },
            indent=2,
          )
          + "\n"
        )
    finally:
      try:
        if self.tracking is not None:
          self.tracking.close(exit_code=int(error is not None))
      finally:
        try:
          if self.env is not None:
            self.env.close()
        finally:
          self.sim_log.close()

  def train(
    self, window_factory, *, checkpoint, evaluate=None, phase=None, log_first=False
  ):
    """Run updates; task callbacks retain evaluation and checkpoint semantics."""
    args, trainer = self.args, self.trainer
    log_path = self.output / "metrics.jsonl"
    if args.resume and log_path.exists():
      rows = [
        json.loads(line) for line in log_path.read_text().splitlines() if line.strip()
      ]
      log_path.write_text(
        "".join(
          json.dumps(row) + "\n"
          for row in rows
          if row["iteration"] <= trainer.iteration
        )
      )
    window = window_factory()

    def flush():
      nonlocal window
      summary = window.summary()
      self.console.training(window.last if args.log_format == "json" else summary)
      if self.tracking:
        self.tracking.window(summary)
      window = window_factory()

    with log_path.open("a") as log:
      while trainer.iteration < args.iterations:
        if phase and window.last and window.last["stage"] != phase():
          flush()
        with simulator_output(self.sim_log, args.verbose_sim):
          record = trainer.train_iteration()
        record = record | {
          "metric_schema_version": 1,
          "metric_groups": metric_groups(record),
        }
        log.write(json.dumps(record, allow_nan=False) + "\n")
        log.flush()
        window.add(record)
        if self.tracking:
          self.tracking.training(record)
        if (
          trainer.iteration % args.log_interval == 0
          or trainer.iteration == args.iterations
          or (log_first and trainer.iteration == 1)
        ):
          flush()
        if evaluate and (
          trainer.iteration % args.eval_interval == 0
          or trainer.iteration == args.iterations
        ):
          evaluate()
        historical = bool(
          getattr(args, "full_checkpoint_interval", 0)
          and trainer.iteration % args.full_checkpoint_interval == 0
        )
        if (
          trainer.iteration % args.checkpoint_interval == 0
          or historical
          or trainer.iteration == args.iterations
        ):
          checkpoint(historical)
          self.console.message(
            f"Checkpoint saved at iteration {trainer.iteration:,}: {self.output / 'last.pt'}"
          )
