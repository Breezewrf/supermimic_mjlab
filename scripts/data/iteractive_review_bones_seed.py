"""Visually review BONES-SEED NPZ clips in a 3 x 4 MuJoCo grid."""

import argparse
import json
import math
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from zipfile import BadZipFile

import glfw
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from convert_lafan_to_npz import load_robot

from src.tasks.supermimic.config.morphology import ROBOTS, robot_layout
from src.tasks.supermimic.data.motion_sources import motion_entries

ROWS, COLS = 3, 4
NAVIGATION = (
  "Arrows/click select   S keep   D discard   N/B or PgDn/PgUp batch   Q quit"
)
PLAYBACK = "Space pause all   [/] step   F failure frame   0 restart   =/- speed"
CAMERA = "Right drag orbit   Scroll zoom"
COLORS = {
  "pending": (0.26, 0.30, 0.34, 1),
  "keep": (0.18, 0.67, 0.38, 1),
  "discard": (0.86, 0.26, 0.25, 1),
}


@dataclass(frozen=True)
class Candidate:
  file: str
  focus_frame: int | None = None
  failures: int | None = None


@dataclass
class PlayingClip:
  candidate: Candidate
  poses: np.ndarray
  playhead: float = 0.0

  @property
  def frame(self):
    return int(self.playhead)


def candidates_from_directory(root, queue_path):
  if queue_path is None:
    return [
      Candidate(entry.path.relative_to(root).as_posix())
      for entry in motion_entries(motion_dir=root)
    ]
  rows = json.loads(queue_path.read_text())["clips"]
  result, seen = [], set()
  for row in rows:
    relative = Path(row["file"])
    if relative.is_absolute() or ".." in relative.parts or relative.suffix != ".npz":
      raise ValueError(f"Review queue must use local relative NPZ paths: {relative}")
    file = relative.as_posix()
    if file in seen:
      raise ValueError(f"Duplicate review entry: {file}")
    seen.add(file)
    if (root / relative).is_file():
      result.append(Candidate(file, row.get("focus_frame"), row.get("failures")))
  if len(result) != len(rows):
    print(f"Found {len(result)}/{len(rows)} queued NPZs in {root}", flush=True)
  if not result:
    raise ValueError("No review candidates found on disk")
  return result


def load_poses(path, model, joints):
  with np.load(path, allow_pickle=False) as motion:
    if (
      not np.isclose(motion["fps"].item(), 50)
      or motion["quaternion_order"].item() != "wxyz"
    ):
      raise ValueError("Expected 50 Hz wxyz motion")
    names = motion["joint_names"].tolist()
    if len(names) != len(joints) or set(names) != set(joints):
      raise ValueError("Motion joints do not match the selected robot")
    pelvis = motion["body_names"].tolist().index("pelvis")
    xyz = motion["body_pos_w"][:, pelvis]
    quat = motion["body_quat_w"][:, pelvis]
    joint_pos = motion["joint_pos"][:, [names.index(name) for name in joints]]
  frames = len(xyz)
  if frames < 2 or (xyz.shape, quat.shape, joint_pos.shape) != (
    (frames, 3),
    (frames, 4),
    (frames, len(joints)),
  ):
    raise ValueError("Unexpected motion frame dimensions")
  if (
    not np.isfinite(xyz).all()
    or not np.isfinite(quat).all()
    or not np.isfinite(joint_pos).all()
  ):
    raise ValueError("Nonfinite reference pose")
  norm = np.linalg.norm(quat, axis=-1, keepdims=True)
  if not np.allclose(norm, 1.0, atol=1e-5):
    raise ValueError("Expected unit pelvis quaternions")
  poses = np.tile(model.qpos0, (frames, 1))
  free = int(np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)[0])
  address = model.jnt_qposadr[free]
  poses[:, address : address + 3] = xyz
  poses[:, address + 3 : address + 7] = quat / norm
  joint_addresses = [model.jnt_qposadr[model.joint(name).id] for name in joints]
  poses[:, joint_addresses] = joint_pos
  return poses


def write_json(path, value):
  temporary = path.with_suffix(".tmp")
  temporary.write_text(json.dumps(value, indent=2) + "\n")
  temporary.replace(path)


class Review:
  def __init__(self, root, robot):
    self.root, self.robot = root, robot
    self.path = root / "review_decisions.json"
    self.manifest = root / "manifest_reviewed.json"
    self.decisions = {}
    if self.path.is_file():
      saved = json.loads(self.path.read_text())
      if saved["robot"] != robot:
        raise ValueError(f"Saved review uses robot {saved['robot']}")
      self.decisions = saved["decisions"]
      self.write_manifest()

  def record(self, file, decision, frames, frame):
    self.decisions[file] = {
      "decision": decision,
      "frames": frames,
      "review_frame": frame,
    }
    write_json(self.path, {"robot": self.robot, "decisions": self.decisions})
    self.write_manifest()

  def write_manifest(self):
    write_json(
      self.manifest,
      {
        "schema_version": 1,
        "fps": 50,
        "quaternion_order": "wxyz",
        "robot": self.robot,
        "clips": [
          {"file": file, "frames": row["frames"]}
          for file, row in sorted(self.decisions.items())
          if row["decision"] == "keep" and (self.root / file).is_file()
        ],
      },
    )


class Grid:
  """Render independent views of one model into a shared framebuffer."""

  def __init__(self, model, batch_size):
    self.model = model
    self.data = mujoco.MjData(model)
    self.pelvis = model.body("pelvis").id
    self.scene = mujoco.MjvScene(model, maxgeom=1000)
    self.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
    self.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
    self.options = mujoco.MjvOption()
    # Use the G1 visual meshes and put the floor in a visible group of its own.
    model.geom("floor").group = 2
    self.options.geomgroup[0] = False
    self.camera = mujoco.MjvCamera()
    self.camera.distance = 2.5
    self.camera.azimuth = 135
    self.camera.elevation = -15
    self.context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_100)
    self.columns = min(COLS, batch_size)
    self.rows = math.ceil(batch_size / self.columns)
    self.tiles = []

  def draw(self, width, height, clips, selected, review, status):
    if width < 1 or height < 1:
      self.tiles = []
      return
    rect = mujoco.MjrRect
    header = min(60, height // 4)
    footer = min(76, height // 4)
    usable_height = height - header - footer
    background = (0.08, 0.10, 0.12, 1)
    mujoco.mjr_rectangle(rect(0, 0, width, height), *background)
    self.tiles = []
    for index, clip in enumerate(clips):
      row, column = divmod(index, self.columns)
      left, right = column * width // self.columns, (column + 1) * width // self.columns
      top = height - header - row * usable_height // self.rows
      bottom = height - header - (row + 1) * usable_height // self.rows
      tile = rect(left + 3, bottom + 3, right - left - 6, top - bottom - 6)
      self.tiles.append(tile)
      if tile.width < 12 or tile.height < 12:
        continue
      decision = review.decisions.get(clip.candidate.file, {}).get(
        "decision", "pending"
      )
      color = COLORS[decision]
      if index == selected:
        mujoco.mjr_rectangle(tile, 0.98, 0.81, 0.35, 1)
        tile = rect(tile.left + 3, tile.bottom + 3, tile.width - 6, tile.height - 6)
      mujoco.mjr_rectangle(tile, *color)
      view = rect(tile.left + 2, tile.bottom + 2, tile.width - 4, tile.height - 4)
      self.data.qpos[:] = clip.poses[clip.frame]
      self.data.time = clip.frame / 50
      mujoco.mj_forward(self.model, self.data)
      self.camera.lookat[:] = self.data.xpos[self.pelvis]
      mujoco.mjv_updateScene(
        self.model,
        self.data,
        self.options,
        None,
        self.camera,
        mujoco.mjtCatBit.mjCAT_ALL,
        self.scene,
      )
      mujoco.mjr_render(view, self.scene, self.context)
      label = f"{index + 1}  {Path(clip.candidate.file).stem}"
      max_chars = max(8, (view.width - 16) // 8)
      if len(label) > max_chars:
        label = label[: max_chars - 3] + "..."
      overlay = (
        f"{label}\n{decision.upper()}   Frame {clip.frame}/{len(clip.poses) - 1}"
      )
      self.overlay(view, overlay)
    self.overlay(rect(0, height - header, width, header), status)
    self.overlay(rect(0, 0, width, footer), f"{NAVIGATION}\n{PLAYBACK}\n{CAMERA}")

  def overlay(self, view, text):
    mujoco.mjr_overlay(
      mujoco.mjtFont.mjFONT_NORMAL,
      mujoco.mjtGridPos.mjGRID_TOPLEFT,
      view,
      text,
      "",
      self.context,
    )

  def tile_at(self, window, x, y):
    width, height = glfw.get_window_size(window)
    fb_width, fb_height = glfw.get_framebuffer_size(window)
    if not width or not height:
      return None
    x, y = x * fb_width / width, (height - y) * fb_height / height
    return next(
      (
        index
        for index, tile in enumerate(self.tiles)
        if tile.left <= x < tile.left + tile.width
        and tile.bottom <= y < tile.bottom + tile.height
      ),
      None,
    )


def play_batch(window, grid, clips, review, status, selected, speed, commands):
  paused = False
  last_time = time.monotonic()
  while not glfw.window_should_close(window):
    glfw.poll_events()
    if glfw.window_should_close(window):
      break
    now = time.monotonic()
    if not paused:
      for clip in clips:
        clip.playhead = (clip.playhead + (now - last_time) * 50 * speed) % len(
          clip.poses
        )
    last_time = now
    while commands:
      key = commands.popleft()
      clip = clips[selected]
      if isinstance(key, tuple):
        clicked = grid.tile_at(window, *key)
        if clicked is not None:
          selected = clicked
      elif key == glfw.KEY_RIGHT:
        if selected == len(clips) - 1:
          return 1, False, speed
        selected += 1
      elif key == glfw.KEY_LEFT:
        if selected == 0:
          return -1, True, speed
        selected -= 1
      elif key == glfw.KEY_UP:
        selected = max(0, selected - grid.columns)
      elif key == glfw.KEY_DOWN:
        selected = min(len(clips) - 1, selected + grid.columns)
      elif key in (glfw.KEY_N, glfw.KEY_PAGE_DOWN):
        return 1, False, speed
      elif key in (glfw.KEY_B, glfw.KEY_PAGE_UP):
        return -1, False, speed
      elif key in (glfw.KEY_Q, glfw.KEY_ESCAPE):
        return 0, False, speed
      elif key in (glfw.KEY_S, glfw.KEY_D):
        decision = "keep" if key == glfw.KEY_S else "discard"
        review.record(clip.candidate.file, decision, len(clip.poses), clip.frame)
        print(f"{decision}: {clip.candidate.file} (frame {clip.frame})", flush=True)
      elif key == glfw.KEY_SPACE:
        paused = not paused
      elif key in (glfw.KEY_LEFT_BRACKET, glfw.KEY_RIGHT_BRACKET):
        step = 1 if key == glfw.KEY_RIGHT_BRACKET else -1
        clip.playhead, paused = (clip.frame + step) % len(clip.poses), True
      elif key == glfw.KEY_0:
        clip.playhead = 0.0
      elif key == glfw.KEY_F and clip.candidate.focus_frame is not None:
        clip.playhead = min(len(clip.poses) - 1, max(0, clip.candidate.focus_frame))
        paused = True
      elif key in (glfw.KEY_EQUAL, glfw.KEY_MINUS):
        speed = min(8.0, max(0.1, speed * (1.25 if key == glfw.KEY_EQUAL else 0.8)))
    clip = clips[selected]
    playback = "Paused" if paused else "Playing"
    failure = ""
    if clip.candidate.failures is not None:
      failure = f"  |  Policy failures {clip.candidate.failures}"
    if clip.candidate.focus_frame is not None:
      failure += f"  |  Failure frame {clip.candidate.focus_frame}"
    width, height = glfw.get_framebuffer_size(window)
    reviewed = sum(item.candidate.file in review.decisions for item in clips)
    grid.draw(
      width,
      height,
      clips,
      selected,
      review,
      f"{status}  |  Reviewed {reviewed}/{len(clips)}  |  {playback} {speed:.2f}x\n"
      f"{clip.candidate.file}{failure}",
    )
    glfw.swap_buffers(window)
    if not width or not height:
      glfw.wait_events_timeout(0.1)
  return 0, False, speed


def review_batches(window, grid, candidates, review, model, joints, batch_size, speed):
  commands = deque()

  def on_key(_window, key, _code, action, _mods):
    if action == glfw.PRESS or (
      action == glfw.REPEAT and key in (glfw.KEY_LEFT_BRACKET, glfw.KEY_RIGHT_BRACKET)
    ):
      commands.append(key)

  def on_click(window, button, action, _mods):
    if button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS:
      commands.append(glfw.get_cursor_pos(window))

  glfw.set_key_callback(window, on_key)
  glfw.set_mouse_button_callback(window, on_click)
  glfw.set_scroll_callback(
    window,
    lambda _window, _x, y: mujoco.mjv_moveCamera(
      model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0, -0.05 * y, grid.camera
    ),
  )
  cursor = glfw.get_cursor_pos(window)

  def orbit(window, x, y):
    nonlocal cursor
    if glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS:
      height = max(1, glfw.get_window_size(window)[1])
      mujoco.mjv_moveCamera(
        model,
        mujoco.mjtMouse.mjMOUSE_ROTATE_V,
        (x - cursor[0]) / height,
        (y - cursor[1]) / height,
        grid.camera,
      )
    cursor = (x, y)

  glfw.set_cursor_pos_callback(window, orbit)
  first_pending = next(
    (
      index
      for index, clip in enumerate(candidates)
      if clip.file not in review.decisions
    ),
    0,
  )
  page = first_pending // batch_size
  pages = math.ceil(len(candidates) / batch_size)
  direction, select_last = 1, False
  unreadable = set()
  resume = True
  while not glfw.window_should_close(window):
    start = page * batch_size
    clips = []
    for candidate in candidates[start : start + batch_size]:
      if candidate.file in unreadable:
        continue
      try:
        poses = load_poses(review.root / candidate.file, model, joints)
      except (ValueError, KeyError, IndexError, OSError, EOFError, BadZipFile) as error:
        print(f"Skipping unreadable clip {candidate.file}: {error}", flush=True)
        unreadable.add(candidate.file)
        continue
      clips.append(PlayingClip(candidate, poses))
    if not clips:
      if len(unreadable) == len(candidates):
        print("No readable clips found", flush=True)
        return
      page = (page + direction) % pages
      continue
    selected = len(clips) - 1 if select_last else 0
    if resume:
      pending = next(
        (
          i
          for i, clip in enumerate(clips)
          if clip.candidate.file not in review.decisions
        ),
        None,
      )
      if pending is None:
        later_pending = next(
          (
            i
            for i in range(start + batch_size, len(candidates))
            if candidates[i].file not in review.decisions
          ),
          None,
        )
        if later_pending is not None:
          page = later_pending // batch_size
          continue
      else:
        selected = pending
      resume = False
    status = f"Batch {page + 1}/{pages}"
    print(status, flush=True)
    commands.clear()
    direction, select_last, speed = play_batch(
      window, grid, clips, review, status, selected, speed, commands
    )
    if not direction:
      return
    page = (page + direction) % pages


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--motion-dir", type=Path, required=True, help="Root of copied NPZ files"
  )
  parser.add_argument(
    "--review-queue", type=Path, help="Optional ranked queue JSON with relative paths"
  )
  parser.add_argument("--robot", choices=ROBOTS, default="g1_29dof")
  parser.add_argument("--speed", type=float, default=1.0)
  parser.add_argument(
    "--batch-size", type=int, default=ROWS * COLS, choices=range(1, ROWS * COLS + 1)
  )
  args = parser.parse_args()
  if not math.isfinite(args.speed) or not 0.1 <= args.speed <= 8.0:
    parser.error("--speed must be between 0.1 and 8.0")
  root = args.motion_dir.resolve()
  if not root.is_dir():
    parser.error(f"Motion directory does not exist: {root}")
  try:
    candidates = candidates_from_directory(root, args.review_queue)
    review = Review(root, args.robot)
  except (ValueError, OSError, KeyError) as error:
    parser.error(str(error))
  model, joints, _, _ = load_robot(robot_layout(args.robot).xml, args.robot)
  if not glfw.init():
    parser.exit(
      1, "Cannot initialize GLFW. Run the reviewer from a graphical desktop session.\n"
    )
  window = None
  grid = None
  try:
    monitor = glfw.get_primary_monitor()
    width, height = 1440, 960
    if monitor:
      _, _, available_width, available_height = glfw.get_monitor_workarea(monitor)
      width = max(640, min(width, available_width - 48))
      height = max(500, min(height, available_height - 96))
    window = glfw.create_window(width, height, "BONES-SEED Motion Review", None, None)
    if not window:
      parser.exit(1, "Cannot create a MuJoCo review window.\n")
    glfw.set_window_size_limits(window, 640, 500, glfw.DONT_CARE, glfw.DONT_CARE)
    glfw.make_context_current(window)
    glfw.swap_interval(1)
    grid = Grid(model, args.batch_size)
    print(
      f"Reviewing {len(candidates)} motions.\n{NAVIGATION}\n{PLAYBACK}\n{CAMERA}",
      flush=True,
    )
    review_batches(
      window, grid, candidates, review, model, joints, args.batch_size, args.speed
    )
  finally:
    if grid is not None:
      grid.context.free()
    if window is not None:
      glfw.destroy_window(window)
    glfw.terminate()
  if review.path.is_file():
    print(
      f"Decisions: {review.path}\nKept motions manifest: {review.manifest}", flush=True
    )


if __name__ == "__main__":
  main()
