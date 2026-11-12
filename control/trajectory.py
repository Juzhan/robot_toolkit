"""Exact playback of time-parameterized joint trajectories.

A planned trajectory (robokit, or a point-to-point move) is streamed sample
by sample, without a jerk-limited filter: a filter would cut corners and
leave the path that was checked for collisions.  The samples are instead
checked against velocity / acceleration / jerk limits before the first
command, and the run is refused when they exceed them.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from ..errors import RobotUnavailable
from .stream import ServoStream, StreamAbort, StreamLimits

# Peaks of the quintic min-jerk profile over a unit move in unit time.
_MIN_JERK_VELOCITY = 1.875
_MIN_JERK_ACCELERATION = 10.0 / math.sqrt(3.0)
_MIN_JERK_JERK = 60.0


@dataclass
class TrajectoryLimits:
    """Per-joint (or scalar) bounds on the sampled trajectory derivatives."""

    velocity: float | list = 0.8
    acceleration: float | list | None = None
    jerk: float | list | None = None


def min_jerk(start, goal, dt: float, speed: float, accel: float | None = None) -> np.ndarray:
    """Synchronized quintic move whose peak joint speed (and accel) stay in bounds.

    Returns samples every ``dt`` from ``start`` (included) to ``goal`` (included).
    """
    start = np.asarray(start, dtype=float).reshape(-1)
    goal = np.asarray(goal, dtype=float).reshape(-1)
    if start.shape != goal.shape:
        raise ValueError(f'start and goal differ in shape: {start.shape} vs {goal.shape}')
    if speed <= 0 or (accel is not None and accel <= 0):
        raise ValueError('speed and accel must be positive')
    distance = float(np.max(np.abs(goal - start))) if start.size else 0.0
    if distance < 1e-9:
        return goal[None].copy()
    duration = _MIN_JERK_VELOCITY * distance / speed
    if accel is not None:
        duration = max(duration, math.sqrt(_MIN_JERK_ACCELERATION * distance / accel))
    steps = max(1, int(math.ceil(duration / dt)))
    s = np.arange(steps + 1) / steps
    blend = s ** 3 * (10.0 - 15.0 * s + 6.0 * s * s)
    return start + (goal - start) * blend[:, None]


def _natural_spline(knots: np.ndarray):
    """C2 cubic spline through ``knots`` at u = 0, 1, ..., K-1 (zero end curvature).

    Returns ``evaluate(u) -> (q, dq/du, d2q/du2, d3q/du3)``.
    """
    count = len(knots)
    curvature = np.zeros_like(knots)
    if count > 2:
        system = (np.diag(np.full(count - 2, 4.0)) + np.diag(np.ones(count - 3), 1)
                  + np.diag(np.ones(count - 3), -1))
        curvature[1:-1] = np.linalg.solve(system, 6.0 * (knots[2:] - 2.0 * knots[1:-1] + knots[:-2]))

    def evaluate(u):
        u = np.clip(np.asarray(u, dtype=float), 0.0, count - 1)
        i = np.minimum(u.astype(int), count - 2)
        t = (u - i)[:, None]
        y0, y1, m0, m1 = knots[i], knots[i + 1], curvature[i], curvature[i + 1]
        q = ((1 - t) * y0 + t * y1 + ((1 - t) ** 3 - (1 - t)) * m0 / 6.0
             + (t ** 3 - t) * m1 / 6.0)
        dq = y1 - y0 + (1.0 - 3.0 * (1 - t) ** 2) * m0 / 6.0 + (3.0 * t ** 2 - 1.0) * m1 / 6.0
        return q, dq, (1 - t) * m0 + t * m1, np.broadcast_to(m1 - m0, q.shape)

    return evaluate


def retime(knots, dt: float, limits: TrajectoryLimits) -> np.ndarray:
    """Rest-to-rest samples along a planner's path that respect ``limits``.

    A planner's frames joined by straight segments change velocity abruptly at
    every frame.  This runs a C2 cubic spline through the frames and moves
    along it with one min-jerk time scaling, stretched until velocity,
    acceleration and jerk all fit.  The path is the spline, which bows
    slightly off the straight segments: collision-check the result.
    """
    knots = np.asarray(knots, dtype=float)
    if knots.ndim != 2 or len(knots) == 0 or not np.all(np.isfinite(knots)):
        raise ValueError('knots must be a finite K x dof array')
    if len(knots) == 1 or float(np.max(np.abs(knots - knots[0]))) < 1e-9:
        return knots[:1].copy()
    if len(knots) == 2:
        knots = np.stack([knots[0], 0.5 * (knots[0] + knots[1]), knots[1]])
    spline = _natural_spline(knots)
    span = len(knots) - 1

    def along(tau, duration):
        """Joint position, velocity, acceleration, jerk at normalized times ``tau``."""
        b1 = 30.0 * tau ** 2 * (1 - tau) ** 2
        b2 = 60.0 * tau * (1 - tau) * (1 - 2 * tau)
        b3 = 60.0 * (1 - 6 * tau + 6 * tau ** 2)
        u = span * tau ** 3 * (10.0 - 15.0 * tau + 6.0 * tau ** 2)
        u1 = (span * b1 / duration)[:, None]
        u2 = (span * b2 / duration ** 2)[:, None]
        u3 = (span * b3 / duration ** 3)[:, None]
        q, d1, d2, d3 = spline(u)
        return q, d1 * u1, d2 * u1 ** 2 + d1 * u2, d3 * u1 ** 3 + 3.0 * d2 * u1 * u2 + d1 * u3

    # Velocity, acceleration and jerk scale as 1/T, 1/T^2, 1/T^3: size T from T = 1.
    dense = np.linspace(0.0, 1.0, 64 * span + 1)
    _, v, a, j = along(dense, 1.0)
    duration = 0.0
    for values, bound, power in ((v, limits.velocity, 1), (a, limits.acceleration, 2),
                                 (j, limits.jerk, 3)):
        if bound is not None:
            ratio = np.max(np.abs(values) / np.asarray(bound, dtype=float))
            duration = max(duration, float(ratio) ** (1.0 / power))
    duration = max(duration * 1.02, dt)  # margin for the dense-grid maximum
    steps = int(math.ceil(duration / dt))
    return along(np.arange(steps + 1) / steps, steps * dt)[0]


def trajectory_peaks(q, dt: float) -> dict:
    """Finite-difference peak |velocity|, |acceleration| and |jerk| per joint."""
    q = np.asarray(q, dtype=float)
    velocity = np.diff(q, axis=0) / dt
    acceleration = np.diff(velocity, axis=0) / dt
    jerk = np.diff(acceleration, axis=0) / dt
    peak = (lambda values: np.max(np.abs(values), axis=0) if len(values)
            else np.zeros(q.shape[1]))
    return {'velocity': peak(velocity), 'acceleration': peak(acceleration),
            'jerk': peak(jerk)}


def check_trajectory(q, dt: float, limits: TrajectoryLimits | None = None,
                     lower=None, upper=None) -> dict:
    """Validate shape, finiteness, joint bounds and derivative limits.

    Returns the peaks; raises ``ValueError`` naming every violated limit.
    """
    q = np.asarray(q, dtype=float)
    if q.ndim != 2 or len(q) == 0:
        raise ValueError(f'trajectory must be N x dof, got {q.shape}')
    if not np.all(np.isfinite(q)):
        raise ValueError('trajectory contains NaN/Inf')
    if dt <= 0:
        raise ValueError('dt must be positive')
    problems = []
    if lower is not None and np.any(q < np.asarray(lower, dtype=float) - 1e-9):
        problems.append('below the lower joint limit')
    if upper is not None and np.any(q > np.asarray(upper, dtype=float) + 1e-9):
        problems.append('above the upper joint limit')
    peaks = trajectory_peaks(q, dt)
    limits = limits or TrajectoryLimits()
    for name in ('velocity', 'acceleration', 'jerk'):
        bound = getattr(limits, name)
        if bound is None:
            continue
        bound = np.broadcast_to(np.asarray(bound, dtype=float), peaks[name].shape)
        if np.any(peaks[name] > bound * (1.0 + 1e-6)):
            joint = int(np.argmax(peaks[name] / bound))
            problems.append(f'{name} {peaks[name][joint]:.3f} > {bound[joint]:.3f} '
                            f'on joint {joint}')
    if problems:
        raise ValueError('trajectory rejected: ' + '; '.join(problems))
    return peaks


@dataclass
class TrajectoryResult:
    ok: bool = False
    reason: str = ''
    error: str = ''
    phase: str = 'interpolation'
    samples: int = 0
    duration_s: float = 0.0
    final_error: float | None = None
    stream: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {'ok': self.ok, 'reason': self.reason, 'error': self.error,
                'phase': self.phase, 'samples': self.samples,
                'duration_s': self.duration_s, 'final_error': self.final_error,
                **self.stream}


def run_trajectory(robot, q, dt: float, *, limits: TrajectoryLimits | None = None,
                   start_tolerance: float = 0.02, wait: bool = True,
                   tolerance: float = 0.02, timeout: float | None = None,
                   abort_event=None, realtime: bool = True,
                   stream_limits: StreamLimits | None = None,
                   use_sdk_period: bool = True, on_command=None) -> TrajectoryResult:
    """Stream ``q`` (N x dof, one sample per ``dt``) exactly, then settle.

    ``limits=None`` skips the derivative check (the caller already did it).
    The first sample must be within ``start_tolerance`` of the measured
    joints.  With ``wait`` the final sample is held until every joint is
    within ``tolerance``.  Failures stop the robot and are reported in the
    result; exceptions from ``servo_j`` (safety rejections) are re-raised
    after the stop.
    """
    q = np.asarray(q, dtype=float)
    if limits is not None:
        check_trajectory(q, dt, limits)
    elif q.ndim != 2 or len(q) == 0 or not np.all(np.isfinite(q)):
        raise ValueError('trajectory must be a finite N x dof array')
    result = TrajectoryResult(samples=len(q))
    stream = ServoStream(robot, dt, realtime=realtime, limits=stream_limits,
                         use_sdk_period=use_sdk_period)
    state = stream.preflight()
    measured = np.asarray(state.q, dtype=float)
    if measured.shape != q.shape[1:]:
        raise ValueError(f'trajectory has {q.shape[1]} joints, robot has {measured.size}')
    start_gap = float(np.max(np.abs(q[0] - measured)))
    if start_gap > start_tolerance:
        raise ValueError(f'trajectory starts {start_gap:.4f} rad from the measured joints '
                         f'(> {start_tolerance} rad); move to its first sample first')
    if timeout is None:
        timeout = len(q) * dt * 3.0 + 5.0
    deadline = time.monotonic() + float(timeout)
    stream.start()

    commanded = measured
    index = 0
    failed = None
    try:
        while True:
            elapsed = stream.begin()
            if abort_event is not None and abort_event.is_set():
                result.reason = 'aborted'
                break
            _, measured, _ = stream.measure(commanded)
            if index >= len(q):
                result.phase = 'tracking'
                result.final_error = float(np.max(np.abs(measured - q[-1])))
                if not wait:
                    result.ok, result.reason = True, 'accepted'
                    break
                if result.final_error <= tolerance:
                    result.ok, result.reason = True, 'reached'
                    break
            if time.monotonic() > deadline:
                result.reason = 'timeout'
                break
            commanded = q[min(index, len(q) - 1)]
            sent_at = stream.send(commanded)
            if on_command is not None:
                on_command({'t': elapsed, 'send_t': sent_at - stream.start_time,
                            'index': index, 'q_cmd': commanded.tolist(),
                            'q_meas': measured.tolist()})
            index += 1
            stream.wait()
    except (StreamAbort, RobotUnavailable) as exc:
        result.reason = 'servo_rejected' if str(exc) == 'servo_error' else (
            exc.reason if isinstance(exc, StreamAbort) else 'unavailable')
        result.error = str(exc)
    except BaseException as exc:
        failed = exc
        result.reason, result.error = 'error', str(exc)
    finally:
        result.duration_s = stream.elapsed()
        result.stream = stream.summary()
        if not result.ok:
            try:
                robot.stop_motion()
            except Exception as exc:
                result.error = (result.error + '; stop failed: ' + str(exc)).strip('; ')
    if failed is not None:
        raise failed
    return result
