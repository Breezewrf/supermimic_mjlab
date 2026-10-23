"""Display physics, a blue reference and an optional orange world-model prediction.

Rendering uses independent CPU MuJoCo buffers. Optional mouse perturbations send
only generalized forces back to the simulator, following mjlab's native viewer.
The terminal transition is shown before automatic reset can replace it.
"""

import copy
import math
import subprocess
import threading
import time
from itertools import pairwise
from pathlib import Path
from queue import Empty, SimpleQueue

import mujoco
import numpy as np
import torch

from ..mdp.state import quat_conjugate, quat_inverse_apply, quat_mul, yaw_quat
from .evaluation import EvaluationInterrupted


class ViewerClosed(EvaluationInterrupted):
  """The window closed before the evaluation finished."""


class EvaluationScene:
  """CPU render buffers sharing the simulator's geometry and joint ordering."""

  def __init__(self, env, *, ghost=True, global_ghost=False, world_model=None):
    self.model = copy.copy(env.sim.mj_model)
    self.data = mujoco.MjData(self.model)
    self.reference_data = mujoco.MjData(self.model)
    self.prediction_data = (
      mujoco.MjData(self.model) if world_model is not None else None
    )
    self.world_model = world_model
    self.show_world_model = True
    self.prediction_ready = False
    self.prediction_path = None
    self.ghost = ghost
    self.global_ghost = global_ghost
    self.root_adr = env.robot.indexing.free_joint_q_adr.cpu().numpy()
    self.joint_adr = env.robot.indexing.joint_q_adr.cpu().numpy()
    self.root_v_adr = env.robot.indexing.free_joint_v_adr.cpu().numpy()
    self.joint_v_adr = env.robot.indexing.joint_v_adr.cpu().numpy()
    self.origin = env.scene.env_origins[0].cpu().numpy()
    self.option = mujoco.MjvOption()
    self.perturb = mujoco.MjvPerturb()
    self.camera = mujoco.MjvCamera()
    self.camera.distance = 3.5
    self.camera.azimuth = 135
    self.camera.elevation = -15

  def update(self, env, clip, frame, state):
    state_tensor = state[0].detach().cpu()
    state = state_tensor.numpy()
    self.data.qpos[self.root_adr[:3]] = state[:3] + self.origin
    self.data.qpos[self.root_adr[3:]] = state[3:7]
    self.data.qpos[self.joint_adr] = state[7:36]
    # Perturbation damping uses body velocities, including local angular qvel.
    self.data.qvel[self.root_v_adr[:3]] = state[36:39]
    self.data.qvel[self.root_v_adr[3:]] = quat_inverse_apply(
      state_tensor[3:7], state_tensor[39:42]
    ).numpy()
    self.data.qvel[self.joint_v_adr] = state[42:71]
    self.data.time = frame * env.step_dt
    mujoco.mj_forward(self.model, self.data)
    self.camera.lookat[:] = state[:3] + self.origin
    self.prediction_ready = (
      self.world_model is not None and self.world_model.state is not None
    )
    trajectory = getattr(self.world_model, "trajectory", None)
    self.prediction_path = (
      trajectory[:, 0, :3].detach().cpu().numpy() + self.origin
      if trajectory is not None
      else None
    )
    if self.prediction_ready:
      prediction = self.world_model.state[0].detach().cpu()
      data = self.prediction_data
      data.qpos[:] = self.data.qpos
      data.qpos[self.root_adr[:3]] = prediction[:3].numpy() + self.origin
      data.qpos[self.root_adr[3:]] = prediction[3:7].numpy()
      data.qpos[self.joint_adr] = prediction[7:36].numpy()
      data.qvel[self.root_v_adr[:3]] = prediction[36:39].numpy()
      data.qvel[self.root_v_adr[3:]] = quat_inverse_apply(
        prediction[3:7], prediction[39:42]
      ).numpy()
      data.qvel[self.joint_v_adr] = prediction[42:71].numpy()
      data.time = getattr(self.world_model, "target_frame", frame) * env.step_dt
      mujoco.mj_forward(self.model, data)
    if self.ghost:
      motion = torch.tensor([clip], device=env.device, dtype=torch.long)
      frames = torch.tensor([frame], device=env.device, dtype=torch.long)
      reference = env.library.get_frame(motion, frames)
      position = reference["body_pos_w"][0, 0].cpu().numpy().copy()
      quaternion = reference["body_quat_w"][0, 0]
      if not self.global_ghost:
        # Match the heading/XY alignment used for body tracking rewards.
        robot_quat = quaternion.new_tensor(state[3:7])
        heading = yaw_quat(quat_mul(robot_quat, quat_conjugate(quaternion)))
        quaternion = quat_mul(heading, quaternion)
        position[:2] = state[:2]
      self.reference_data.qpos[:] = self.data.qpos
      self.reference_data.qpos[self.root_adr[:3]] = position + self.origin
      self.reference_data.qpos[self.root_adr[3:]] = quaternion.cpu().numpy()
      self.reference_data.qpos[self.joint_adr] = reference["joint_pos"][0].cpu().numpy()
      mujoco.mj_forward(self.model, self.reference_data)

  def add_ghost(self, scene):
    if self.ghost:
      self._add_ghost(scene, self.reference_data, (0.25, 0.6, 1.0, 0.3))
    if self.prediction_ready and self.show_world_model:
      self._add_ghost(scene, self.prediction_data, (1.0, 0.55, 0.1, 0.5))
      if self.prediction_path is not None:
        path = self.prediction_path[:: max(1, len(self.prediction_path) // 16)]
        if not np.array_equal(path[-1], self.prediction_path[-1]):
          path = np.concatenate((path, self.prediction_path[-1:]))
        for start, end in pairwise(path):
          if scene.ngeom >= scene.maxgeom:
            break
          geom = scene.geoms[scene.ngeom]
          mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_LINE,
            np.zeros(3),
            np.zeros(3),
            np.eye(3).ravel(),
            np.array([1.0, 0.55, 0.1, 0.8], dtype=np.float32),
          )
          mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, 2.0, start, end)
          geom.category = mujoco.mjtCatBit.mjCAT_DECOR
          scene.ngeom += 1

  def _add_ghost(self, scene, data, color):
    start = scene.ngeom
    mujoco.mjv_addGeoms(
      self.model,
      data,
      self.option,
      self.perturb,
      mujoco.mjtCatBit.mjCAT_DYNAMIC,
      scene,
    )
    for index in range(start, scene.ngeom):
      geom = scene.geoms[index]
      geom.rgba[:] = color
      geom.matid = -1
      geom.category = mujoco.mjtCatBit.mjCAT_DECOR


class MousePerturbations:
  """Bridge native mouse wrenches to MJWarp's generalized-force channel."""

  def __init__(self, env, model, data):
    self.env, self.model, self.data = env, model, data
    self.body_ids = set(env.robot.indexing.body_ids.cpu().tolist())
    self.applied_steps = 0
    self.qfrc = np.zeros(model.nv)

  def clear(self):
    self.env.sim.data.qfrc_applied[0].zero_()
    self.data.xfrc_applied[:] = 0

  def apply(self, perturb):
    self.clear()
    if not perturb.active or perturb.select not in self.body_ids:
      return
    # mjv_applyPerturbForce only overwrites the selected body. clear() prevents
    # a previous selection's force from remaining in the CPU render buffer.
    mujoco.mjv_applyPerturbForce(self.model, self.data, perturb)
    body = perturb.select
    self.qfrc[:] = 0
    mujoco.mj_applyFT(
      self.model,
      self.data,
      self.data.xfrc_applied[body, :3],
      self.data.xfrc_applied[body, 3:],
      self.data.xipos[body],
      body,
      self.qfrc,
    )
    target = self.env.sim.data.qfrc_applied[0]
    target.copy_(torch.from_numpy(self.qfrc).to(target))
    if np.any(self.qfrc):
      self.applied_steps += 1


class LiveViewer:
  """Real-time passive viewer; key events only queue commands on the GUI thread."""

  def __init__(
    self,
    env,
    *,
    speed=1.0,
    ghost=True,
    global_ghost=False,
    perturbations=False,
    world_model=None,
  ):
    if not math.isfinite(speed) or speed <= 0:
      raise ValueError("Playback speed must be positive and finite")
    import mujoco.viewer

    self.world_model = world_model
    self.scene = EvaluationScene(
      env, ghost=ghost, global_ghost=global_ghost, world_model=world_model
    )
    self.mouse = (
      MousePerturbations(env, self.scene.model, self.scene.data)
      if perturbations
      else None
    )
    self.speed = speed
    self.paused = False
    self.commands = SimpleQueue()
    self.deadline = None
    existing_threads = set(threading.enumerate())
    self.viewer = mujoco.viewer.launch_passive(
      self.scene.model,
      self.scene.data,
      key_callback=self.commands.put,
      show_left_ui=False,
      show_right_ui=False,
    )
    # launch_passive owns a daemon render thread. Let its GL teardown finish
    # before our buffers and the Python interpreter are destroyed on CLI exit.
    self._viewer_threads = set(threading.enumerate()) - existing_threads
    self.viewer.cam.distance = self.scene.camera.distance
    self.viewer.cam.azimuth = self.scene.camera.azimuth
    self.viewer.cam.elevation = self.scene.camera.elevation

  def begin_evaluation(self):
    if self.mouse:
      self.mouse.clear()
      self.mouse.applied_steps = 0

  def before_step(self, env):
    del env
    if not self.viewer.is_running():
      raise ViewerClosed()
    if self.mouse:
      with self.viewer.lock():
        self.mouse.apply(self.viewer.perturb)

  def _commands(self):
    advance = False
    while True:
      try:
        key = self.commands.get_nowait()
      except Empty:
        break
      if key == 32:  # Space
        self.paused = not self.paused
        self.deadline = None
      elif key == 262:  # Right arrow
        self.paused = True
        advance = True
      elif key == ord("G"):
        self.scene.ghost = not self.scene.ghost
      elif key == ord("W") and self.world_model is not None:
        self.scene.show_world_model = not self.scene.show_world_model
    return advance

  def __call__(self, env, clip, frame, state, reward_sum):
    if frame == 0:
      self.deadline = time.monotonic()
      if self.mouse:
        self.mouse.clear()
        with self.viewer.lock():
          # A drag reference from the last motion must not pull the new pose.
          self.viewer.perturb.active = 0
          self.viewer.perturb.active2 = 0
    while True:
      if not self.viewer.is_running():
        raise ViewerClosed()
      advance = self._commands()
      self.scene.update(env, clip, frame, state)
      with self.viewer.lock():
        self.viewer.cam.lookat[:] = self.scene.camera.lookat
        self.viewer.user_scn.ngeom = 0
        self.scene.add_ghost(self.viewer.user_scn)
      if not self.viewer.is_running():
        raise ViewerClosed()
      if hasattr(self.viewer, "set_texts"):
        self.viewer.set_texts(
          (
            mujoco.mjtFontScale.mjFONTSCALE_150.value,
            mujoco.mjtGridPos.mjGRID_TOPLEFT.value,
            "Motion\nTime\nReturn so far\nState\nControls\nMouse forces"
            + ("\nWorld model (orange)" if self.world_model is not None else ""),
            (
              f"{env.library.names[clip]}\n{frame * env.step_dt:.2f}s\n"
              f"{reward_sum / (5 * env.step_dt * (int(env.library.lengths[clip]) - 1)):.3f}\n"
              f"{'PAUSED' if self.paused else 'RUNNING'} ({self.speed:g}x)\n"
              "Space: pause | Right: step | G: reference"
              + (" | W: prediction" if self.world_model is not None else "")
              + "\n"
              + (
                f"ON ({self.mouse.applied_steps} perturbed steps); double-click body, Ctrl+drag"
                if self.mouse
                else "OFF"
              )
              + (
                "\n" + self.world_model.status() if self.world_model is not None else ""
              )
            ),
          )
        )
      if not self.viewer.is_running():
        raise ViewerClosed()
      self.viewer.sync()
      if not self.paused or advance:
        break
      time.sleep(0.02)
    # Rebase when inference is slower than real time; avoid later catch-up bursts.
    now = time.monotonic()
    self.deadline = max((self.deadline or now) + env.step_dt / self.speed, now)
    time.sleep(max(0.0, self.deadline - time.monotonic()))

  def close(self):
    if self.mouse:
      self.mouse.clear()
    self.viewer.close()
    for thread in self._viewer_threads:
      thread.join(timeout=5.0)


class VideoRecorder:
  """Optional headless MP4 export of exactly the frames used in evaluation."""

  def __init__(
    self, env, path: Path, *, ghost=True, global_ghost=False, world_model=None
  ):
    self.scene = EvaluationScene(
      env, ghost=ghost, global_ghost=global_ghost, world_model=world_model
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    self.scene.model.vis.global_.offwidth = 960
    self.scene.model.vis.global_.offheight = 540
    self.renderer = mujoco.Renderer(self.scene.model, height=540, width=960)
    try:
      self.encoder = subprocess.Popen(
        [
          "ffmpeg",
          "-y",
          "-loglevel",
          "error",
          "-f",
          "rawvideo",
          "-pix_fmt",
          "rgb24",
          "-s",
          "960x540",
          "-r",
          str(1 / env.step_dt),
          "-i",
          "-",
          "-an",
          "-c:v",
          "libx264",
          "-pix_fmt",
          "yuv420p",
          str(path),
        ],
        stdin=subprocess.PIPE,
      )
    except BaseException:
      self.renderer.close()
      raise

  def __call__(self, env, clip, frame, state, reward_sum):
    del reward_sum
    self.scene.update(env, clip, frame, state)
    self.renderer.update_scene(self.scene.data, camera=self.scene.camera)
    self.scene.add_ghost(self.renderer.scene)
    assert self.encoder.stdin is not None
    self.encoder.stdin.write(self.renderer.render().tobytes())

  def close(self):
    try:
      if self.encoder.stdin is not None:
        self.encoder.stdin.close()
    finally:
      status = self.encoder.wait()
      self.renderer.close()
    if status:
      raise RuntimeError(f"ffmpeg failed with exit status {status}")
