"""Defaults shared by collection, evaluation, and future model training."""

from dataclasses import dataclass
from pathlib import Path

from .morphology import robot_layout

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
MOTION_ANCHOR_BODY = "torso_link"
MOTION_ANCHOR_INDEX = TRACKED_BODIES.index(MOTION_ANCHOR_BODY)
STATE_DIM = 71
OBS_DIM = 93
ACTION_DIM = 29
CONTEXT_DIM = 33
STEP_DT = 0.02


@dataclass(frozen=True)
class FoundationConfig:
  manifest: Path | None = None
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
  relative_orientation: bool = False
  motion_anchor_ori_b: bool = False
  reset_noise: bool = False
  terminate_on_failure: bool = True
  robot: str = "g1_29dof"
  motion: Path | None = None
  motion_dir: Path | None = None

  @property
  def observation_dim(self) -> int:
    return self.layout.observation_dim + 6 * (
      self.relative_orientation + self.motion_anchor_ori_b
    )

  @property
  def layout(self):
    return robot_layout(self.robot)

  def __post_init__(self):
    if sum(value is not None for value in (self.manifest, self.motion, self.motion_dir)) > 1:
      raise ValueError("Select exactly one motion source")
    if self.manifest is None and self.motion is None and self.motion_dir is None:
      object.__setattr__(self, "manifest", self.layout.manifest)
    robot_layout(self.robot)
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
