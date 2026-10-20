"""Small application-facing collection session.

The recorder itself is deliberately low-level.  This module connects it to a
robot, a gripper and an optional camera hub so a teleoperation/teaching entry
can collect a dataset without reimplementing timestamp/image plumbing.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ..camera import CameraHub
from ..errors import DatasetError
from .export import export_lerobot
from .recorder import EpisodeRecorder
from .schema import DatasetSchema


class EpisodeCollector:
    """Collect LeRobot-compatible episodes from any ``RobotBackend``.

    Cameras are optional.  If ``cameras`` is empty the schema must also have no
    cameras, and only state/action frames are written.
    """

    def __init__(self, root, schema: DatasetSchema, robot=None, gripper=None,
                 cameras=(), export=export_lerobot, fps: float | None = None,
                 min_interval_s: float = 0.0, clock=time.monotonic,
                 max_frames: int | None = None, compress: bool = False):
        self.schema = schema
        self.robot = robot
        self.gripper = gripper
        self.hub = CameraHub(cameras) if cameras else None
        self.min_interval_s = max(0.0, float(min_interval_s))
        self.clock = clock
        self._last_capture = -1e9
        recorder_kwargs = {'compress': bool(compress)}
        if max_frames is not None:
            recorder_kwargs['max_frames'] = int(max_frames)
        self.recorder = EpisodeRecorder(root, schema, export=export, fps=fps,
                                        **recorder_kwargs)
        self._started = False
        self._last_state = None

    @property
    def active(self) -> bool:
        return self._started

    @property
    def dataset_root(self) -> Path:
        return self.recorder.dataset_root

    @property
    def saved(self) -> int:
        return self.recorder.saved

    @property
    def export_errors(self) -> list:
        return self.recorder.export_errors

    def start(self, task: str, meta: dict | None = None) -> Path:
        if self.hub is not None:
            for camera in self.hub:
                if not camera.opened:
                    camera.open()
            self.hub.warmup()
        self._started = True
        self._last_capture = -1e9
        return self.recorder.start(task, meta=meta)

    def _camera_payload(self, strict: bool = True) -> tuple[dict, dict]:
        if self.hub is None:
            return {}, {}
        frames = self.hub.snapshot(strict=strict)
        images, depth = {}, {}
        for name, frame in frames.items():
            if frame.rgb is not None:
                images[name] = frame.rgb
            if frame.depth is not None:
                depth[name] = frame.depth
                depth[f'{name}_K'] = (np.eye(3) if frame.intrinsics is None
                                      else frame.intrinsics)
                depth[f'{name}_T_world_camera'] = (np.eye(4) if frame.T_world_camera is None
                                                   else frame.T_world_camera)
        return images, depth

    def capture_state(self, state=None, action=None, timestamp: float | None = None,
                      extra: dict | None = None, strict_cameras: bool = True) -> bool:
        """Capture one observation/action pair from the live robot.

        ``action`` defaults to the observation vector.  This is the right
        fallback for teleoperation where the controller target is Cartesian;
        callers with joint-space targets can pass an explicit action.
        """
        if not self._started:
            return False
        now = self.clock()
        if now - self._last_capture < self.min_interval_s:
            return True
        if self.robot is None:
            raise DatasetError('capture_state needs a robot backend')
        if state is None:
            state = self.robot.state()
        if state.q is None:
            raise DatasetError('robot state has no joint positions')
        width = None
        if self.gripper is not None:
            try:
                width = float(self.gripper.width)
            except Exception:
                width = None
        if width is None:
            # No gripper object: fall back to the width the backend reports,
            # otherwise ``pack`` substitutes max_width and the dataset is wrong.
            width = state.extra.get('gripper_width')
        observation = self.schema.observation(state.q, width)
        action_vector = observation if action is None else np.asarray(action, dtype=np.float32)
        images, depth = self._camera_payload(strict=strict_cameras)
        self._last_capture = now
        self._last_state = state
        return self.recorder.capture(observation, action_vector,
                                     images=images or None,
                                     depth=depth or None,
                                     extra=extra,
                                     timestamp=timestamp if timestamp is not None else now)

    def capture_vectors(self, observation, action, timestamp: float | None = None,
                        extra: dict | None = None, strict_cameras: bool = True) -> bool:
        """Capture already assembled state/action vectors (e.g. a trajectory)."""
        if not self._started:
            return False
        now = self.clock()
        if now - self._last_capture < self.min_interval_s:
            return True
        images, depth = self._camera_payload(strict=strict_cameras)
        self._last_capture = now
        return self.recorder.capture(np.asarray(observation, dtype=np.float32),
                                     np.asarray(action, dtype=np.float32),
                                     images=images or None,
                                     depth=depth or None,
                                     extra=extra,
                                     timestamp=timestamp if timestamp is not None else now)

    def capture_trajectory(self, trajectory, success: bool | None = True,
                           task: str | None = None) -> int:
        """Write a recorded teach trajectory as state -> next-state actions.

        Teach trajectories are commonly 50-125 Hz, while the dataset is
        configured for 10-30 Hz.  Resample before writing so the recorder queue
        and LeRobot features keep the configured dataset rate.
        """
        if not self._started:
            self.start(task or trajectory.meta.get('task') or 'teach trajectory',
                       meta=dict(trajectory.meta))
        count = len(trajectory.t)
        if count < 2:
            raise DatasetError('trajectory needs at least two samples')
        rate = max(1.0, float(self.recorder.fps))
        source_t = np.asarray(trajectory.t, dtype=float)
        source_t = source_t - float(source_t[0])
        play_t = np.arange(0.0, float(source_t[-1]) + 0.5 / rate, 1.0 / rate)
        if len(play_t) < 2:
            play_t = np.array([0.0, float(source_t[-1])])
        joints = np.stack([
            np.interp(play_t, source_t, np.asarray(trajectory.q)[:, column])
            for column in range(trajectory.q.shape[1])
        ], axis=1)
        grip = np.interp(play_t, source_t, np.asarray(trajectory.grip).reshape(-1))
        for index in range(len(play_t)):
            nxt = min(index + 1, len(play_t) - 1)
            observation = self.schema.observation(joints[index], grip[index])
            action = self.schema.observation(joints[nxt], grip[nxt])
            if not self.capture_vectors(observation, action,
                                        timestamp=float(play_t[index]),
                                        extra={'source_index': np.asarray([index])}):
                raise DatasetError('capture queue rejected a trajectory frame')
            if (index + 1) % 32 == 0:
                self.recorder.flush()
        self.stop(success=success)
        return len(play_t)

    def stop(self, success: bool | None = None) -> None:
        if not self._started:
            return
        self.recorder.save(success=success)
        self._started = False

    def discard(self) -> None:
        if self._started:
            self.recorder.discard()
            self._started = False

    def close(self, wait_export: bool = True, timeout: float = 60.0) -> None:
        self.recorder.close(wait_export=wait_export, timeout=timeout)
        if self.hub is not None:
            self.hub.close()
        self._started = False


def collect_trajectory(trajectory, schema: DatasetSchema, root, task: str | None = None,
                       success: bool | None = True, export=export_lerobot,
                       fps: float | None = None) -> EpisodeCollector:
    """Convenience wrapper: ``Trajectory`` -> LeRobot episode, no camera needed."""
    collector = EpisodeCollector(root, schema, export=export, fps=fps)
    collector.start(task or trajectory.meta.get('task') or 'teach trajectory',
                    meta=dict(trajectory.meta))
    collector.capture_trajectory(trajectory, success=success, task=task)
    collector.recorder.flush()
    return collector
