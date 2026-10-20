"""Record a human-dragged (freedrive) trajectory through the robot interface."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import numpy as np

from ..robot.base import RobotBackend
from .trajectory import Marker, Trajectory, save_trajectory


@dataclass
class TeachRecorderConfig:
    rate: float = 125.0
    max_duration: float = 900.0
    min_samples: int = 2
    meta: dict = field(default_factory=dict)


class TeachRecorder:
    """Sample robot state at a fixed rate while an operator moves the arm.

    Markers are stored with wall-clock time and converted to sample indices
    when the trajectory is built, so a paused or jittery loop cannot skew them.
    """

    def __init__(self, robot: RobotBackend, gripper=None,
                 config: TeachRecorderConfig | None = None, rate: float | None = None,
                 clock=time.monotonic):
        self.robot = robot
        self.gripper = gripper
        if rate is not None:
            config = TeachRecorderConfig(rate=float(rate),
                                         max_duration=(config.max_duration
                                                       if config else 900.0),
                                         min_samples=(config.min_samples if config else 2),
                                         meta=dict(config.meta) if config else {})
        self.config = config or TeachRecorderConfig()
        self.clock = clock
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._recording = False
        self._t0 = 0.0
        self._t: list[float] = []
        self._q: list[np.ndarray] = []
        self._qd: list[np.ndarray] = []
        self._tcp: list[np.ndarray] = []
        self._grip: list[float] = []
        self._marks: list[tuple[float, str, float | None, str]] = []
        self._error = ''

    # --------------------------------------------------------------- control
    @property
    def recording(self) -> bool:
        return self._recording

    @property
    def samples(self) -> int:
        with self._lock:
            return len(self._t)

    @property
    def duration(self) -> float:
        with self._lock:
            return self._t[-1] - self._t[0] if len(self._t) > 1 else 0.0

    def start(self) -> 'TeachRecorder':
        with self._lock:
            if self._recording:
                raise RuntimeError('already recording')
            self._t.clear()
            self._q.clear()
            self._qd.clear()
            self._tcp.clear()
            self._grip.clear()
            self._marks.clear()
            self._error = ''
            self._t0 = self.clock()
            self._recording = True
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name='teach-recorder')
        self._thread.start()
        return self

    def stop(self) -> Trajectory:
        with self._lock:
            self._recording = False
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        return self.build()

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc, traceback):
        if self._recording:
            self.stop()
        return False

    # ---------------------------------------------------------------- sample
    def _loop(self) -> None:
        period = 1.0 / max(1.0, float(self.config.rate))
        while not self._stop.is_set():
            started = self.clock()
            if not self._recording:
                break
            self.capture()
            if self.duration > self.config.max_duration:
                self._error = 'max recording duration reached'
                break
            time.sleep(max(0.0, period - (self.clock() - started)))

    def capture(self) -> None:
        try:
            state = self.robot.state()
        except Exception as exc:
            self._error = str(exc)
            return
        if state.q is None:
            self._error = 'robot reported no joint positions'
            return
        width = None
        if self.gripper is not None:
            try:
                width = float(self.gripper.width)
            except Exception:
                width = None
        if width is None:
            width = state.extra.get('gripper')
        with self._lock:
            self._t.append(self.clock() - self._t0)
            self._q.append(np.asarray(state.q, dtype=float).copy())
            self._qd.append(np.asarray(state.qd if state.qd is not None else np.zeros_like(state.q),
                                       dtype=float).copy())
            self._tcp.append(np.asarray(state.tcp if state.tcp is not None else np.zeros(6),
                                        dtype=float).copy())
            self._grip.append(float(width) if width is not None else float('nan'))

    def mark(self, action: str = 'mark', label: str = '', width: float | None = None) -> float:
        with self._lock:
            stamp = self.clock() - self._t0
            self._marks.append((stamp, action, width, label))
        return stamp

    # ----------------------------------------------------------------- build
    def build(self) -> Trajectory:
        with self._lock:
            if len(self._t) < self.config.min_samples:
                raise ValueError(f'not enough samples ({len(self._t)}) to build a trajectory')
            t = np.asarray(self._t, dtype=float)
            t = t - t[0]
            q = np.asarray(self._q, dtype=float)
            qd = np.asarray(self._qd, dtype=float)
            tcp = np.asarray(self._tcp, dtype=float)
            grip = np.asarray(self._grip, dtype=float)
            marks = list(self._marks)
        markers = []
        for stamp, action, width, label in marks:
            index = int(np.clip(np.searchsorted(t, stamp), 0, len(t) - 1))
            markers.append(Marker(t=float(t[index]), index=index, action=action,
                                  width=width, label=label))
        meta = dict(self.config.meta)
        meta.update(rate=float(self.config.rate), samples=int(len(t)),
                    duration=float(t[-1]), source='teach', error=self._error)
        return Trajectory(t=t, q=q, qd=qd, tcp=tcp, grip=grip, markers=markers, meta=meta)

    def save(self, directory, name: str = 'trajectory') -> Trajectory:
        trajectory = self.build()
        save_trajectory(directory, trajectory, name=name)
        return trajectory
