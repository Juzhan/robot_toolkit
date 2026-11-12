"""Helpers bridging camera snapshots and episode capture."""

from __future__ import annotations

import math
import time
from collections import deque

import numpy as np

from ..errors import RobotUnavailable
from ..robot.base import _json_value


class RobotCameraSynchronizer:
    """Pair images with nearby robot reads on host perf_counter time.

    No worker, connection, motor enable, or movement is started. Call
    sample_state() from the existing read loop to build history. SDK sample
    times are preserved but never compared with host reception times.
    """

    def __init__(self, robot, cameras, *, max_skew_s: float = 0.05,
                 max_frame_age_s: float = 0.3, history_s: float = 2.0,
                 max_samples: int = 256, clock=time.perf_counter):
        for value in (max_skew_s, max_frame_age_s, history_s):
            if not math.isfinite(value) or value <= 0:
                raise ValueError('synchronization time limits must be positive and finite')
        if not isinstance(max_samples, int) or max_samples < 2:
            raise ValueError('max_samples must be an integer >= 2')
        if not len(cameras):
            raise ValueError('synchronization needs at least one camera')
        self.robot = robot
        self.cameras = cameras
        self.max_skew_s = float(max_skew_s)
        self.max_frame_age_s = float(max_frame_age_s)
        self.history_s = float(history_s)
        self.clock = clock
        self._history = deque(maxlen=max_samples)

    def clear(self) -> None:
        self._history.clear()

    def add_state(self, state, *, started_at: float, finished_at: float) -> None:
        """Add an existing read with its host perf_counter acquisition window.

        These must be times around the actual read, not state.t or times
        assigned later to a cached sample. The caller owns clock alignment.
        """
        if (not math.isfinite(started_at) or not math.isfinite(finished_at)
                or finished_at < started_at):
            raise ValueError('invalid robot acquisition window')
        if self._history and started_at < self._history[-1][1]:
            raise ValueError('robot acquisition windows must be ordered and nonoverlapping')
        if (not state.connected or state.q is None or state.q.ndim != 1
                or state.q.size == 0 or not np.all(np.isfinite(state.q))):
            self.clear()
            raise RobotUnavailable('synchronization needs connected, finite joint feedback')
        self._history.append((float(started_at), float(finished_at), state.copy()))
        while self._history and self._history[0][1] < finished_at - self.history_s:
            self._history.popleft()

    def sample_state(self):
        """Read once, save the host read window, and return the original sample."""
        started = self.clock()
        try:
            if not self.robot.connected:
                raise RobotUnavailable('robot is disconnected')
            state = self.robot.state()
            self.add_state(state, started_at=started, finished_at=self.clock())
        except Exception:
            self.clear()
            raise
        return state

    def observe(self, *, strict: bool = True) -> dict:
        """Read images and pair each with one measured state; never interpolate.

        Returns native CameraFrame objects, a reference RobotState for the first
        camera, and JSON-ready synchronization evidence. Strict mode rejects any
        missing/stale/unmatched image. Reception matching is not exposure sync.
        """
        self.sample_state()
        frames = self.cameras.snapshot(strict=False)
        self.sample_state()
        now = self.clock()
        pairs = {}
        reference_state = None
        for name, frame in frames.items():
            pair = {'matched': False, 'reason': None, 'frame_received_at': float(frame.t),
                    'frame_age_s': None, 'state_minus_frame_s': None,
                    'read_window_skew_bound_s': None, 'robot_read_window': None,
                    'state': None, 'frame_metadata': _json_value(frame.meta)}
            pairs[name] = pair
            if frame.rgb is None:
                pair['reason'] = 'missing_image'
            elif not math.isfinite(frame.t) or frame.t <= 0 or frame.t > now:
                pair['frame_received_at'] = None if not math.isfinite(frame.t) else float(frame.t)
                pair['reason'] = 'invalid_frame_time'
            else:
                pair['frame_age_s'] = float(now - frame.t)
                if pair['frame_age_s'] > self.max_frame_age_s:
                    pair['reason'] = 'stale_image'
                else:
                    started, finished, sample = min(
                        self._history, key=lambda entry: abs((entry[0] + entry[1]) / 2 - frame.t))
                    pair['state_minus_frame_s'] = (started + finished) / 2 - frame.t
                    pair['read_window_skew_bound_s'] = max(abs(started - frame.t),
                                                         abs(finished - frame.t))
                    pair['robot_read_window'] = [started, finished]
                    pair['state'] = sample.to_dict()
                    pair['matched'] = pair['read_window_skew_bound_s'] <= self.max_skew_s
                    pair['reason'] = 'matched' if pair['matched'] else 'skew_exceeded'
                    if name == self.cameras.names[0] and pair['matched']:
                        reference_state = sample.copy()
        matched = bool(pairs) and all(pair['matched'] for pair in pairs.values())
        frame_times = [frame.t for frame in frames.values() if math.isfinite(frame.t) and frame.t > 0]
        evidence = {'method': 'nearest_host_receive', 'clock': 'host_perf_counter',
                    'exposure_synchronized': False, 'robot_feedback_freshness': 'unknown',
                    'max_skew_s': self.max_skew_s, 'max_frame_age_s': self.max_frame_age_s,
                'camera_received_span_s': (float(max(frame_times) - min(frame_times))
                               if len(frame_times) == len(frames) and frame_times else None),
                    'reference_camera': self.cameras.names[0], 'matched': matched,
                    'cameras': pairs}
        if strict and not matched:
            details = '; '.join(f'{name}: {pair["reason"]}' for name, pair in pairs.items()
                                if not pair['matched'])
            raise RobotUnavailable('camera/robot synchronization failed: ' + details)
        return {'frames': frames, 'state': reference_state, 'synchronization': evidence}


def split_camera_frames(frames) -> tuple[dict, dict]:
    """Convert ``{name: CameraFrame}`` into ``(images, depth_payload)``.

    ``depth_payload`` uses the flat keys understood by ``EpisodeRecorder``:
    ``<name>_K`` and ``<name>_T_world_camera`` next to ``<name>_depth_mm``.
    """
    images, depth = {}, {}
    for name, frame in frames.items():
        if frame.rgb is not None:
            images[name] = np.ascontiguousarray(frame.rgb)
        if frame.depth is not None:
            depth[name] = frame.depth
            if frame.intrinsics is not None:
                depth[f'{name}_K'] = frame.intrinsics
            if frame.T_world_camera is not None:
                depth[f'{name}_T_world_camera'] = frame.T_world_camera
    return images, depth


def camera_images(frames) -> dict:
    return {name: frame.rgb for name, frame in frames.items() if frame.rgb is not None}
