"""Robot variants and tensor layouts shared by simulation, data and learning."""

from dataclasses import dataclass
from pathlib import Path

ROBOTS = ("g1_29dof", "g1_23dof")
ASSETS = Path(__file__).resolve().parents[3] / "assets"


@dataclass(frozen=True)
class RobotLayout:
  num_joints: int = 29

  def __post_init__(self):
    if self.num_joints not in (23, 29):
      raise ValueError("Only G1 23DoF and 29DoF are supported")

  @property
  def robot(self):
    return f"g1_{self.num_joints}dof"

  @property
  def q(self):
    return slice(7, 7 + self.num_joints)

  @property
  def v(self):
    return slice(7 + self.num_joints, 10 + self.num_joints)

  @property
  def w(self):
    return slice(10 + self.num_joints, 13 + self.num_joints)

  @property
  def qd(self):
    return slice(13 + self.num_joints, self.state_dim)

  @property
  def state_dim(self):
    return 13 + 2 * self.num_joints

  @property
  def observation_dim(self):
    return 6 + 3 * self.num_joints

  @property
  def context_dim(self):
    return 4 + self.num_joints

  @property
  def delta_groups(self):
    return (3, 3, self.num_joints, 3, 3, self.num_joints)

  @property
  def reference_dim(self):
    return 10 * (15 + 2 * self.num_joints)

  @property
  def full_body_reference_dim(self):
    return 10 * (210 + 2 * self.num_joints)

  @property
  def xml(self):
    return ASSETS / f"robots/g1/{self.robot}_rev_1_0.xml"

  @property
  def manifest(self):
    suffix = "_23dof" if self.num_joints == 23 else ""
    return ASSETS / f"motions/g1/lafan1_npz{suffix}/manifest.json"


def robot_layout(robot="g1_29dof"):
  if robot not in ROBOTS:
    raise ValueError(f"Unknown robot {robot!r}; expected one of {ROBOTS}")
  return RobotLayout(23 if robot == "g1_23dof" else 29)


def state_layout(state):
  layout = RobotLayout((state.shape[-1] - 13) // 2)
  if state.shape[-1] != layout.state_dim:
    raise ValueError("Physical state width does not match a supported G1 variant")
  return layout


def tracked_bodies(robot="g1_29dof"):
  from .task_cfg import TRACKED_BODIES

  robot_layout(robot)
  if robot == "g1_23dof":
    return tuple(
      name.replace("wrist_yaw_link", "wrist_roll_rubber_hand")
      for name in TRACKED_BODIES
    )
  return TRACKED_BODIES
