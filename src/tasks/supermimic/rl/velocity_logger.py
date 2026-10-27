"""Velocity metrics using the motion trainer's terminal renderer."""

from ..data.episode_statistics import EpisodeSummary
from .metric_schema import classified_metrics, is_collection_metric
from .training_logger import ConsoleReporter, WandbReporter, duration


class VelocityMetricWindow:
  def __init__(self):
    self.totals = {}
    self.counts = {}
    self.last = {}
    self.transitions = 0
    self.episodes = EpisodeSummary()

  def add(self, record):
    self.last = record
    collected = record["collected_this_iteration"]
    if collected:
      self.transitions += record["collected_transitions"]
      self.episodes.add(record.get("episode_statistics", {}))
    for key, value in record.items():
      # Collection metrics are retained between updates; count them only once.
      if is_collection_metric(key) and not collected:
        continue
      if isinstance(value, (int, float)) and not isinstance(value, bool):
        weight = record["collected_transitions"] if is_collection_metric(key) else 1
        if (
          is_collection_metric(key)
          and record.get("statistics_scope") == "completed_episodes"
        ):
          task = key.split("/", 1)[0]
          weight = record["episode_statistics"].get(task, {}).get("count", 0)
          if not weight:
            continue
        self.totals[key] = self.totals.get(key, 0.0) + value * weight
        self.counts[key] = self.counts.get(key, 0) + weight

  def summary(self):
    result = {key: value / self.counts[key] for key, value in self.totals.items()}
    for key in ("iteration", "stage", "warmup_completed", "learning_rate"):
      result[key] = self.last[key]
    if not self.transitions:
      result = {
        key: value for key, value in result.items() if not is_collection_metric(key)
      }
    result["steps_per_second"] = self.transitions / max(self.totals["seconds"], 1e-9)
    if self.last.get("statistics_scope") == "completed_episodes":
      result["statistics_scope"] = "completed_episodes"
      result["episode_statistics"] = self.episodes.summary()
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

    metrics = classified_metrics(record, include_collection=True)
    episodic = record.get("statistics_scope") == "completed_episodes"
    reward_name = "total_per_second" if episodic else "total_per_step"
    failure_name = "episode_fraction" if episodic else "events_per_transition"

    def value(key, unit="", digits=4, scale=1):
      number = metrics.get(key)
      return "--" if number is None else f"{number * scale:.{digits}f}{unit}"

    rows = [
      (
        "Reward/s ↑" if episodic else "Reward ↑",
        " / ".join(
          (
            "motion=" + value(f"reward/motion/{reward_name}"),
            "velocity=" + value(f"reward/velocity/{reward_name}"),
          )
        )
        + (
          "  (completed episode sum / configured seconds)"
          if episodic
          else "  (weighted, dt-scaled per control step)"
        ),
      ),
      (
        "Error ↓",
        "velocity: XY="
        + value("error/velocity/planar_velocity_m_per_s", " m/s", 3)
        + " / yaw="
        + value("error/velocity/yaw_rate_rad_per_s", " rad/s", 3),
      ),
      (
        "",
        "motion: root position="
        + value("error/motion/root_position_m", " m", 3)
        + " / root rotation="
        + value("error/motion/root_orientation_rad", " rad", 3),
      ),
      (
        "Loss ↓",
        "world model="
        + value("loss/world_model/total")
        + " / motion="
        + value("loss/world_model/motion")
        + " / velocity="
        + value("loss/world_model/velocity"),
      ),
      (
        "",
        "latent MSE="
        + value("loss/velocity/latent_alignment_mse")
        + " / critic motion="
        + value("loss/critic/motion")
        + " / critic velocity="
        + value("loss/critic/velocity"),
      ),
      (
        "Objective ↑",
        "imagined return: motion="
        + value("objective/motion/return")
        + " / velocity="
        + value("objective/velocity/return"),
      ),
      (
        "Failure ↓",
        "motion="
        + value(f"failure/motion/{failure_name}", "%", 2, 100)
        + " / velocity="
        + value(f"failure/velocity/{failure_name}", "%", 2, 100)
        + ("  (completed episodes)" if episodic else "  (events per transition)"),
      ),
      (
        "Performance",
        f"{record['steps_per_second']:,.0f} transitions/s / "
        f"{record['seconds']:.2f} s/iteration / ETA "
        + duration((self.total - record["iteration"]) * record["seconds"]),
      ),
    ]
    if episodic:
      rows.insert(
        -1,
        (
          "Episodes",
          "motion="
          + value("episode/motion/completed", digits=0)
          + " / velocity="
          + value("episode/velocity/completed", digits=0)
          + " / mean seconds: motion="
          + value("episode/motion/duration_mean_s", digits=2)
          + " / velocity="
          + value("episode/velocity/duration_mean_s", digits=2),
        ),
      )
      rows.insert(
        -1,
        (
          "Timeouts",
          "motion="
          + value("reset/motion/time_out_count", digits=0)
          + " / velocity="
          + value("reset/velocity/time_out_count", digits=0),
        ),
      )
    self.metric_sections(
      f"Iteration {record['iteration']:,}/{self.total:,} | {record['stage']} | "
      f"lr={record['learning_rate']:.2e}",
      rows,
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
        key in ("collected_transitions", "collection_seconds", "command_resamples")
        and record["collected_this_iteration"]
      ):
        metrics["collection/" + key] = value
    if "statistics_scope" in record:
      metrics["train/statistics_scope"] = record["statistics_scope"]
    metrics.update(classified_metrics(record))
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
