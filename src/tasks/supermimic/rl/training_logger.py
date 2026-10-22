"""Readable terminal metrics and optional W&B tracking, without model uploads."""

import json
import sys
import uuid
from contextlib import nullcontext, redirect_stdout
from pathlib import Path


def duration(seconds: float) -> str:
  seconds = max(0, int(seconds))
  days, seconds = divmod(seconds, 86400)
  hours, seconds = divmod(seconds, 3600)
  minutes, seconds = divmod(seconds, 60)
  return (
    f"{days}d {hours:02}:{minutes:02}"
    if days
    else f"{hours:02}:{minutes:02}:{seconds:02}"
  )


class MetricWindow:
  """Aggregate update metrics; count each real collection block exactly once."""

  def __init__(self):
    self.count = 0
    self.totals = {}
    self.transitions = self.failures = self.endings = 0
    self.reward_sum = self.collection_seconds = 0.0
    self.last = {}
    self.peak_memory = None

  def add(self, record: dict):
    self.count += 1
    self.last = record
    if "peak_gpu_memory_gib" in record:
      self.peak_memory = max(self.peak_memory or 0, record["peak_gpu_memory_gib"])
    for key, value in record.items():
      if isinstance(value, (int, float)) and not isinstance(value, bool):
        self.totals[key] = self.totals.get(key, 0.0) + value
    if record["collected_this_iteration"]:
      count = record["collected_transitions"]
      self.transitions += count
      self.failures += record["physical_failures"]
      self.endings += record["episodes_ended"]
      self.reward_sum += record["real_reward_per_step"] * count
      self.collection_seconds += record["collection_seconds"]

  def summary(self) -> dict:
    result = {key: value / self.count for key, value in self.totals.items()}
    for key in (
      "iteration",
      "environment_steps",
      "replay_transitions",
      "learning_rate",
    ):
      result[key] = self.last[key]
    result["real_reward_per_step"] = (
      self.reward_sum / self.transitions if self.transitions else None
    )
    result["failure_events_per_transition"] = (
      self.failures / self.transitions if self.transitions else None
    )
    result["window_transitions_per_second"] = self.transitions / self.totals["seconds"]
    result["window_collection_seconds"] = self.collection_seconds
    result["window_physical_failures"] = self.failures
    result["window_episodes_ended"] = self.endings
    if self.peak_memory is not None:
      result["peak_gpu_memory_gib"] = self.peak_memory
    return result


class ConsoleReporter:
  def __init__(self, total: int, log_format: str = "table", stream=None):
    self.total, self.log_format = total, log_format
    self.stream = stream or sys.stdout
    self.lines = 0
    try:
      from rich.console import Console

      self.console = Console(file=self.stream, highlight=False)
    except ImportError:
      self.console = None

  def message(self, text: str):
    if self.log_format == "json":
      print(json.dumps({"message": text}), file=self.stream, flush=True)
    elif self.console:
      self.console.print(text, markup=False)
    else:
      print(text, file=self.stream, flush=True)

  def start(
    self, config: dict, device: str, motions: int, output: Path, iteration: int
  ):
    self.message(
      f"SuperMimic | {device} | {config['num_envs']:,} envs | {motions} motions | "
      f"iterations {iteration:,} -> {self.total:,}\n"
      f"World model: {config['wm_batches']} x {config['wm_batch_size']:,} windows x 8 steps | "
      f"Policy: {config['policy_batch_size']:,} trajectories x {config['horizon']} steps\n"
      f"Output: {output}"
    )
    if self.log_format == "table":
      self.message(
        "Table: interval means | Fail%: failure events per transition | GPU: PyTorch peak GiB"
      )

  def training(self, record: dict):
    if self.log_format == "json":
      self.message_record(record)
      return

    def value(key):
      return "--" if record.get(key) is None else f"{record[key]:.4f}"

    failure = record.get("failure_events_per_transition")
    columns = (
      "Iter",
      "Real R",
      "Fail%",
      "WM loss",
      "Actor J",
      "V loss",
      "Steps/s",
      "s/it",
      "GPU",
      "ETA",
    )
    row = (
      f"{record['iteration']:,}",
      value("real_reward_per_step"),
      "--" if failure is None else f"{100 * failure:.2f}%",
      value("wm_loss"),
      value("actor_objective"),
      value("critic_loss"),
      "--" if failure is None else f"{record['window_transitions_per_second']:,.0f}",
      f"{record['seconds']:.2f}",
      "--"
      if "peak_gpu_memory_gib" not in record
      else f"{record['peak_gpu_memory_gib']:.2f}",
      duration((self.total - record["iteration"]) * record["seconds"]),
    )
    header = self.lines % 20 == 0
    if self.console:
      from rich import box
      from rich.table import Table

      table = Table(
        box=box.SIMPLE_HEAD if header else None,
        show_header=header,
        padding=(0, 1),
        pad_edge=False,
        collapse_padding=True,
      )
      for label in columns:
        table.add_column(label, justify="right", no_wrap=True, header_style="bold cyan")
      table.add_row(*row)
      self.console.print(table)
    else:
      if header:
        self.message(" | ".join(columns))
      self.message(" | ".join(row))
    self.lines += 1

  def message_record(self, record: dict):
    print(json.dumps(record, allow_nan=False), file=self.stream, flush=True)

  def evaluation(self, report: dict):
    if self.log_format == "json":
      self.message_record(
        {
          "iteration": report["iteration"],
          "evaluation_return": report["normalized_return"],
          "successes": report["successes"],
          "motions": len(report["motions"]),
        }
      )
    else:
      self.message(
        f"Evaluation @ {report['iteration']:,} | normalized return {report['normalized_return']:.6f} | "
        f"success {report['successes']}/{len(report['motions'])} | "
        f"mean completion {sum(row['completion'] for row in report['motions']) / len(report['motions']):.1%}"
      )


class WandbReporter:
  def __init__(
    self,
    output: Path,
    config: dict,
    iteration: int,
    *,
    project=None,
    entity=None,
    name=None,
    mode="online",
    console=None,
  ):
    import wandb

    self.sdk, self.output, self.console = wandb, output, console
    self.path = output / "wandb_run.json"
    previous = json.loads(self.path.read_text()) if self.path.exists() else {}
    project = project or previous.get("project") or "supermimic"
    entity = entity or previous.get("entity")
    continuation = (
      mode == previous.get("mode") == "online"
      and previous.get("last_iteration", 0) <= iteration
      and project == previous.get("project")
      and entity == previous.get("entity")
    )
    run_id = previous["id"] if continuation else uuid.uuid4().hex[:8]
    tracking_config = {**config, "resume_from_iteration": iteration}
    if previous and not continuation:
      tracking_config["parent_run_id"] = previous["id"]
    self.run = wandb.init(
      project=project,
      entity=entity,
      name=name or previous.get("name") or output.name,
      id=run_id,
      resume="allow" if continuation else None,
      mode=mode,
      dir=str(output.resolve()),
      config=tracking_config,
      save_code=False,
      allow_val_change=True,
      settings=wandb.Settings(quiet=True, console="off", disable_git=True),
    )
    self.run.define_metric("train/iteration")
    self.run.define_metric("*", step_metric="train/iteration")
    self.run.define_metric("evaluation/normalized_return", summary="max")
    self.state = {
      "id": self.run.id,
      "project": self.run.project,
      "entity": self.run.entity,
      "name": self.run.name,
      "mode": mode,
      "last_iteration": iteration,
    }
    self._persist(iteration)
    if console:
      location = self.run.get_url() if mode == "online" else self.run.dir
      console.message(f"W&B {mode}: {location}")
      if previous and not continuation:
        console.message(
          "W&B: started a linked run segment (offline mode or checkpoint rollback)."
        )

  def _persist(self, iteration: int):
    self.state["last_iteration"] = iteration
    temporary = self.path.with_suffix(".tmp")
    temporary.write_text(json.dumps(self.state, indent=2) + "\n")
    temporary.replace(self.path)

  def training(self, record: dict):
    metrics = {
      "train/iteration": record["iteration"],
      "train/environment_steps": record["environment_steps"],
      "world_model/loss": record["wm_loss"],
      "world_model/state_loss": record["wm_state"],
      "world_model/termination_loss": record["wm_termination"],
      "diagnostics/observation_error": record["wm_observation"],
      "diagnostics/reward_error": record["wm_reward"],
      "policy/objective": record["actor_objective"],
      "policy/imagined_reward": record["imagined_reward"],
      "policy/survival_at_horizon": record["survival_at_horizon"],
      "critic/loss": record["critic_loss"],
      "gradients/world_model_preclip_norm": record["wm_gradient_norm"],
      "gradients/actor_preclip_norm": record["actor_gradient_norm"],
      "gradients/critic_preclip_max_norm": record["critic_gradient_norm"],
      "optim/learning_rate": record["learning_rate"],
      "replay/transitions": record["replay_transitions"],
      "performance/iteration_seconds": record["seconds"],
      "performance/world_model_seconds": record["world_model_seconds"],
      "performance/policy_critic_seconds": record["policy_and_critic_seconds"],
    }
    if "peak_gpu_memory_gib" in record:
      metrics["memory/torch_peak_allocated_gib"] = record["peak_gpu_memory_gib"]
    if record["collected_this_iteration"]:
      count = record["collected_transitions"]
      metrics.update(
        {
          "collection/real_reward_per_step": record["real_reward_per_step"],
          "collection/physical_failures": record["physical_failures"],
          "collection/failure_events_per_transition": record["physical_failures"]
          / count,
          "collection/episodes_ended": record["episodes_ended"],
          "collection/seconds": record["collection_seconds"],
          "collection/transitions_per_second": count / record["collection_seconds"],
        }
      )
    self.run.log(metrics)
    self._persist(record["iteration"])

  def window(self, record: dict):
    if record["real_reward_per_step"] is None:
      return
    self.run.log(
      {
        "train/iteration": record["iteration"],
        "performance/window_transitions_per_second": record[
          "window_transitions_per_second"
        ],
      }
    )

  def evaluation(self, report: dict):
    rows = report["motions"]
    table = self.sdk.Table(
      columns=[
        "motion",
        "normalized_return",
        "completion",
        "success",
        "steps",
        "failure_reason",
      ],
      data=[
        [
          row[key]
          for key in (
            "motion",
            "normalized_return",
            "completion",
            "success",
            "steps",
            "failure_reason",
          )
        ]
        for row in rows
      ],
    )
    metrics = {
      "train/iteration": report["iteration"],
      "evaluation/environment_steps": report["training_environment_steps"],
      "evaluation/normalized_return": report["normalized_return"],
      "evaluation/successes": report["successes"],
      "evaluation/success_rate": report["successes"] / len(rows),
      "evaluation/mean_completion": sum(row["completion"] for row in rows) / len(rows),
      "evaluation/limited": any(row["evaluation_limited"] for row in rows),
      "evaluation/motions": table,
      "evaluation/per_motion_return": self.sdk.plot.bar(
        table, "motion", "normalized_return", title="Normalized return by motion"
      ),
      "evaluation/per_motion_completion": self.sdk.plot.bar(
        table, "motion", "completion", title="Completion by motion"
      ),
    }
    for key in (
      "body_position_error",
      "body_orientation_error",
      "root_position_error",
      "root_orientation_error",
    ):
      errors = [
        row["mean_errors"][key] for row in rows if key in row.get("mean_errors", {})
      ]
      if errors:
        metrics[f"evaluation/{key}"] = sum(errors) / len(errors)
    self.run.log(metrics)

  def close(self, exit_code=0):
    self.run.finish(exit_code=exit_code)


def simulator_output(stream, verbose: bool):
  """Keep simulator diagnostics in a file while preserving exceptions on stderr."""
  return nullcontext() if verbose else redirect_stdout(stream)
