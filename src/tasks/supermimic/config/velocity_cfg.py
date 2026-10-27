"""Velocity interface settings, independent of motion observation options."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class VelocityConfig:
  command_scale: tuple[float, float, float] = (2.0, 1.32, 3.35)
  initial_ranges: tuple[tuple[float, float], ...] = (
    (-0.5, 1.0),
    (-0.5, 0.5),
    (-1.0, 1.0),
  )
  final_ranges: tuple[tuple[float, float], ...] = (
    (-1.5, 2.0),
    (-1.32, 1.32),
    (-3.35, 3.35),
  )
  curriculum_steps: int = 120_000
  resampling_time_range: tuple[float, float] = (3.0, 8.0)
  randomize_initial_episode_progress: bool = True
  reference_probability: float = 0.5
  standing_probability: float = 0.1
  standing_reset_probability: float = 0.5
  min_height: float = 0.35
  tilt_degrees: float = 70.0
  height_softness: float = 0.02
  tilt_softness: float = 0.04
  stand_still_weight: float = 0.0
  standing_linear_threshold: float = 0.1
  standing_angular_threshold: float = 0.1
  alignment_weight: float = 0.1
  warmup_updates: int = 500

  def __post_init__(self):
    if len(self.command_scale) != 3 or any(
      not math.isfinite(x) or x <= 0 for x in self.command_scale
    ):
      raise ValueError("command_scale must contain three positive finite scales")
    for ranges in (self.initial_ranges, self.final_ranges):
      if len(ranges) != 3 or any(
        len(r) != 2 or not all(map(math.isfinite, r)) or r[0] > r[1] for r in ranges
      ):
        raise ValueError("Command ranges must contain three finite intervals")
    for probability in (
      self.reference_probability,
      self.standing_probability,
      self.standing_reset_probability,
    ):
      if not 0 <= probability <= 1:
        raise ValueError("Probabilities must be in [0, 1]")
    if (
      len(self.resampling_time_range) != 2
      or not all(map(math.isfinite, self.resampling_time_range))
      or not 0 < self.resampling_time_range[0] <= self.resampling_time_range[1]
    ):
      raise ValueError("Command holding times must be positive finite seconds")
    if self.curriculum_steps < 0 or self.warmup_updates < 0:
      raise ValueError("Invalid velocity schedule")
    if (
      not 0 < self.tilt_degrees < 180
      or min(self.min_height, self.height_softness, self.tilt_softness) <= 0
    ):
      raise ValueError("Invalid fall thresholds")
    if self.stand_still_weight > 0 or self.alignment_weight < 0:
      raise ValueError("Invalid velocity loss weights")
