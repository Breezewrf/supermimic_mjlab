"""Velocity metrics using the motion trainer's terminal renderer."""

from .training_logger import ConsoleReporter, WandbReporter, duration


class VelocityMetricWindow:
  def __init__(self):
    self.totals = {}
    self.counts = {}
    self.last = {}
    self.transitions = 0

  def add(self, record):
    self.last = record
    collected = record["collected_this_iteration"]
    if collected:
      self.transitions += record["collected_transitions"]
    for key, value in record.items():
      # Collection metrics are retained between updates; count them only once.
      if (
        key.startswith(("motion/", "velocity/"))
        and key
        in (
          "motion/failure_rate",
          "velocity/physical_failure",
          "velocity/linear_velocity_error",
          "velocity/yaw_rate_error",
        )
        and not collected
      ):
        continue
      if isinstance(value, (int, float)) and not isinstance(value, bool):
        self.totals[key] = self.totals.get(key, 0.0) + value
        self.counts[key] = self.counts.get(key, 0) + 1

  def summary(self):
    result = {key: value / self.counts[key] for key, value in self.totals.items()}
    for key in ("iteration", "stage", "warmup_completed", "learning_rate"):
      result[key] = self.last[key]
    result["steps_per_second"] = self.transitions / max(self.totals["seconds"], 1e-9)
    return result


class VelocityConsoleReporter(ConsoleReporter):
  def training(self, record):
    if self.log_format == "json":
      self.message_record(record)
      return
    if self.training_phase != record["stage"]:
      self.training_phase = record["stage"]
      self.lines = 0
      self.message(
        f"Stage: {self.training_phase} | warmup completed: {record['warmup_completed']:,}"
      )

    def value(key, digits=4, scale=1):
      number = record.get(key)
      return "--" if number is None else f"{number * scale:.{digits}f}"

    self.table(
      (
        "Iter",
        "WM loss",
        "Align",
        "Vel J",
        "Vel V",
        "XY err",
        "Yaw err",
        "VFail%",
        "MFail%",
        "Steps/s",
        "s/it",
        "ETA",
      ),
      (
        f"{record['iteration']:,}",
        value("wm_loss"),
        value("velocity/alignment_loss"),
        value("velocity/objective"),
        value("velocity/critic_loss"),
        value("velocity/linear_velocity_error", 3),
        value("velocity/yaw_rate_error", 3),
        value("velocity/physical_failure", 2, 100),
        value("motion/failure_rate", 2, 100),
        f"{record['steps_per_second']:,.0f}",
        f"{record['seconds']:.2f}",
        duration((self.total - record["iteration"]) * record["seconds"]),
      ),
    )


class VelocityWandbReporter(WandbReporter):
  """Use the shared W&B run lifecycle with mixed-task metric names."""

  def training(self, record):
    metrics = {"train/iteration": record["iteration"], "train/stage": record["stage"]}
    updates = {
      "objective",
      "alignment_loss",
      "reward_objective",
      "bootstrap_objective",
      "survival",
      "actor_gradient_norm",
      "critic_loss",
      "warmup_skipped",
    }
    for key, value in record.items():
      if not isinstance(value, (int, float)) or isinstance(value, bool):
        continue
      if key.startswith(("motion/", "velocity/")):
        if key.split("/", 1)[1] in updates or record["collected_this_iteration"]:
          metrics[key] = value
      elif key.startswith("wm_"):
        metrics["world_model/" + key.removeprefix("wm_")] = value
      elif key in (
        "iteration",
        "environment_steps",
        "control_steps",
        "warmup_completed",
      ):
        metrics["train/" + key] = value
      elif key == "learning_rate":
        metrics["optim/learning_rate"] = value
      elif key == "seconds":
        metrics["performance/iteration_seconds"] = value
      elif (
        key in ("collected_transitions", "collection_seconds")
        and record["collected_this_iteration"]
      ):
        metrics["collection/" + key] = value
    self.run.log(metrics)
    self._persist(record["iteration"])

  def window(self, record):
    self.run.log(
      {
        "train/iteration": record["iteration"],
        "performance/window_transitions_per_second": record["steps_per_second"],
      }
    )

  def velocity_diagnostics(self, report, iteration):
    metrics = {"train/iteration": iteration}
    for horizon, row in report["horizons"].items():
      for key, value in row.items():
        if isinstance(value, (int, float)):
          metrics[f"velocity_replay_prediction/{horizon}_step/{key}"] = value
    self.run.log(metrics)
