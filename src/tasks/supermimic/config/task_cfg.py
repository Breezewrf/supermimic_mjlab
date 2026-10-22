"""Defaults shared by collection, evaluation, and future model training."""

from dataclasses import dataclass
from pathlib import Path

DEFAULT_MANIFEST = (
  Path(__file__).resolve().parents[3] / "assets/motions/g1/lafan1_npz/manifest.json"
)
TRACKED_BODIES = (
  "pelvis",
  "left_hip_roll_link",
  "left_knee_link",
  "left_ankle_roll_link",
  "right_hip_roll_link",
  "right_knee_link",
  "right_ankle_roll_link",
  "torso_link",
  "left_shoulder_roll_link",
  "left_elbow_link",
  "left_wrist_yaw_link",
  "right_shoulder_roll_link",
  "right_elbow_link",
  "right_wrist_yaw_link",
)
STATE_DIM = 71
OBS_DIM = 93
ACTION_DIM = 29
CONTEXT_DIM = 33
STEP_DT = 0.02


@dataclass(frozen=True)
class FoundationConfig:
  manifest: Path = DEFAULT_MANIFEST
  clips: tuple[str, ...] = ()
  num_envs: int = 4
  seed: int = 0
  history_length: int = 10
  future_length: int = 10
  episode_length_s: float = 10.0
  randomization: bool = False
  sensor_noise: bool = False
  sampling: str = "uniform"
  action_clip: float | None = None
  reset_noise: bool = False

  def __post_init__(self):
    if self.num_envs < 1 or self.history_length < 1 or self.future_length < 1:
      raise ValueError("Environment, history and future counts must be positive")
    if self.episode_length_s <= 0 or self.sampling not in ("uniform", "return"):
      raise ValueError("Invalid episode duration or sampler")
    if self.action_clip is not None and self.action_clip <= 0:
      raise ValueError("action_clip must be positive or None")

  @classmethod
  def training(cls, **kwargs):
    return cls(
      **(
        {
          "randomization": True,
          "sensor_noise": True,
          "sampling": "return",
          "reset_noise": True,
        }
        | kwargs
      )
    )
