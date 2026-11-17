"""The one host-side servo stream every streamed motion runs through.

Point-to-point moves, planned trajectories, teach replay and policy chunks
all command the arm by calling ``servo_j`` at a fixed rate.  This module owns
the parts they share, so every one of them gets the same checks:

- preflight: a finite joint state, no protective stop, controller validation;
- timing: the controller period when the SDK has one, otherwise the host
  clock, with a maximum gap between commands on hardware;
- feedback: stale state, protective stop and tracking error against the
  commanded reference;
- statistics for the result reports.

What to command each tick stays with the caller.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from ..core.realtime import collect_idle, pause_gc
from ..errors import RobotUnavailable
from ..robot.base import RobotState
from ..robot.mock import MockRobot


class StreamAbort(RuntimeError):
    """A monitored stream stopped; the message starts with a short reason."""

    @property
    def reason(self) -> str:
        return str(self).split(':')[0]


@dataclass
class StreamLimits:
    """Feedback and timing limits, matching the proven deployment defaults."""

    max_tracking_error: float = 0.15
    tracking_gap: float = 0.08
    tracking_timeout_s: float = 3.0
    stale_state_s: float = 0.1
    max_servo_gap_s: float = 0.04


def is_mock(robot) -> bool:
    inner = getattr(robot, 'inner', robot)
    return bool(getattr(inner, 'mock', False)) or isinstance(inner, MockRobot)


def wait_until(deadline: float) -> None:
    remaining = deadline - time.perf_counter()
    if remaining > 0.0004:
        time.sleep(remaining - 0.0003)
    while time.perf_counter() < deadline:
        pass


def stats(values, scale: float = 1.0) -> list:
    if len(values) == 0:
        return []
    return np.percentile(np.asarray(values, dtype=float) * scale, [50, 95, 99, 100]).tolist()


def ensure_servo(robot, dt: float, ticks: int = 5, realtime: bool = True) -> bool:
    """Verify servo mode before streaming; rebuild the controller link if possible.

    The RTDE control script can time out after a long idle prompt, so this sends
    a few hold commands at the measured pose before any motion command.
    """
    for attempt in range(2):
        state = robot.state()
        if state.q is None:
            return False
        hold = np.asarray(state.q, dtype=float)
        failures = 0
        for _ in range(max(1, int(ticks))):
            try:
                if not robot.servo_j(hold, dt):
                    failures += 1
            except Exception:
                failures += 1
            if realtime:
                time.sleep(dt)
        if failures == 0:
            return True
        restart = getattr(robot, 'restart_control', None)
        if attempt == 0 and callable(restart):
            try:
                if restart():
                    continue
            except Exception:
                pass
        break
    return False


class ServoStream:
    """Fixed-rate ``servo_j`` stream with feedback monitoring.

    One tick is ``begin()``, ``measure()``, ``send()``, ``wait()``.  Every
    check raises :class:`StreamAbort` (or ``RobotUnavailable``); the caller
    decides how to stop and report.  Virtual time (``realtime=False``) is for
    mock robots only.
    """

    def __init__(self, robot, dt: float, *, realtime: bool = True,
                 limits: StreamLimits | None = None, use_sdk_period: bool = True,
                 mock_sleep_s: float = 0.0):
        self.robot = robot
        self.dt = float(dt)
        if self.dt <= 0:
            raise ValueError('dt must be positive')
        self.hardware = not is_mock(robot)
        if self.hardware and not realtime:
            raise ValueError('virtual time is forbidden on real hardware')
        self.realtime = bool(realtime)
        self.limits = limits or StreamLimits()
        self.use_sdk_period = bool(use_sdk_period)
        self.mock_sleep_s = float(mock_sleep_s)
        self.start_time = None
        self.now = 0.0
        self.ticks = 0
        self.commands = 0
        self.sends: list[float] = []
        self.work: list[float] = []
        self.max_tracking_error = 0.0
        self._began = 0.0
        self._deadline = 0.0
        self._period = None
        self._last_t = None
        self._fresh_at = 0.0
        self._bad_since = None
        self._gc_paused = False

    # -------------------------------------------------------------- startup
    def preflight(self) -> RobotState:
        """Check the robot may be streamed to; nothing is commanded."""
        state = self.robot.state()
        if state.q is None:
            raise RobotUnavailable('servo stream needs a joint state')
        if state.protective_stop:
            raise RobotUnavailable('robot is in protective stop')
        q = np.asarray(state.q, dtype=float)
        if not np.all(np.isfinite(q)):
            raise RobotUnavailable('initial joint state is not finite')
        if self.hardware and hasattr(self.robot, 'joints_within_safety'):
            if not self.robot.joints_within_safety(q):
                raise RobotUnavailable('initial pose fails controller safety validation')
        self._last_t = state.t
        return state

    def start(self) -> None:
        """Establish servo mode on hardware and the controller period if any.

        On real hardware automatic garbage collection is paused for the stream
        (``core.realtime``); ``finish()`` (or ``summary()``) collects and resumes it.
        """
        if self.hardware and not ensure_servo(self.robot, self.dt, realtime=self.realtime):
            raise RobotUnavailable('cannot establish the servo stream')
        if self.hardware and self.realtime:
            pause_gc()
            self._gc_paused = True
        if self.realtime and self.hardware and self.use_sdk_period:
            init_period = getattr(self.robot, 'init_period', None)
            frequency = float(getattr(getattr(self.robot, 'config', None),
                                      'frequency', 0.0) or 0.0)
            if callable(init_period) and (frequency <= 0
                                          or abs(frequency - 1.0 / self.dt) < 1.0):
                try:
                    self._period = init_period()
                except Exception:
                    self._period = None

    def resync(self) -> None:
        """Resume after the caller deliberately stopped sending (e.g. a blocking
        prompt): re-establish servo mode and restart the timing references."""
        if self.hardware and not ensure_servo(self.robot, self.dt, realtime=self.realtime):
            raise RobotUnavailable('cannot re-establish the servo stream')
        self.sends.clear()
        self._deadline = self._fresh_at = time.perf_counter()
        self._bad_since = None

    # ----------------------------------------------------------------- tick
    def begin(self) -> float:
        """Start a tick; returns the stream time (wall or virtual) in seconds."""
        began = time.perf_counter()
        if self.start_time is None:
            self.start_time = self._deadline = self._fresh_at = began
        self._began = began
        self.now = (began if self.realtime
                    else self.start_time + self.ticks * self.dt)
        if self._gap_exceeded(began):
            raise StreamAbort('servo_gap: target stream exceeded timing limit')
        return self.now - self.start_time

    def measure(self, reference=None):
        """Read and check feedback; returns ``(state, q, tracking_gap)``."""
        state = self.robot.state()
        if state.q is None:
            raise RobotUnavailable('robot state lost joint positions')
        q = np.asarray(state.q, dtype=float)
        if not np.all(np.isfinite(q)):
            raise StreamAbort('invalid measured joints')
        if state.protective_stop:
            raise StreamAbort('protective_stop')
        if state.t != self._last_t:
            self._last_t, self._fresh_at = state.t, self._began
        if self.hardware and self._began - self._fresh_at > self.limits.stale_state_s:
            raise StreamAbort('stale_robot_state')
        gap = 0.0
        if reference is not None:
            gap = float(np.max(np.abs(np.asarray(reference, dtype=float) - q)))
            self.max_tracking_error = max(self.max_tracking_error, gap)
            if gap > self.limits.max_tracking_error:
                raise StreamAbort(f'tracking_error: {gap:.4f} rad')
            if gap > self.limits.tracking_gap:
                if self._bad_since is None:
                    self._bad_since = self.now
                if self.now - self._bad_since > self.limits.tracking_timeout_s:
                    raise StreamAbort('tracking_timeout')
            else:
                self._bad_since = None
        return state, q, gap

    def send(self, q) -> float:
        at = time.perf_counter()
        if self._gap_exceeded(at):
            raise StreamAbort('servo_gap: computation exceeded timing limit')
        if not self.robot.servo_j(q, self.dt):
            raise StreamAbort('servo_error')
        self.sends.append(at)
        self.commands += 1
        return at

    def wait(self) -> None:
        self.work.append(time.perf_counter() - self._began)
        self.ticks += 1
        if not self.realtime:
            # sleep(0) still yields the GIL to worker threads in virtual time.
            time.sleep(max(0.0, self.mock_sleep_s))
            return
        if self._period is not None:
            try:
                if self.robot.wait_period(self._period):
                    return
            except Exception:
                pass
            self._period = None
        self._deadline += self.dt
        if self._deadline < time.perf_counter() - self.dt:
            self._deadline = time.perf_counter() + self.dt
        wait_until(self._deadline)

    # --------------------------------------------------------------- report
    def elapsed(self) -> float:
        return 0.0 if self.start_time is None else self.now - self.start_time

    def finish(self) -> None:
        """The stream is over: reclaim garbage and resume collection (idempotent)."""
        if self._gc_paused:
            self._gc_paused = False
            collect_idle(resume=True)

    def summary(self) -> dict:
        self.finish()
        gaps = np.diff(self.sends)
        return {
            'servo_commands': self.commands,
            'actual_send_hz': (float(1.0 / np.mean(gaps))
                               if len(gaps) and self.realtime else None),
            'send_gap_ms_p50_p95_p99_max': stats(gaps, 1000.0),
            'work_ms_p50_p95_p99_max': stats(self.work, 1000.0),
            'max_tracking_error': self.max_tracking_error,
        }

    def _gap_exceeded(self, at: float) -> bool:
        # A late command only matters to a controller that is waiting for it.
        return (self.realtime and self.hardware and bool(self.sends)
                and at - self.sends[-1] > self.limits.max_servo_gap_s)
