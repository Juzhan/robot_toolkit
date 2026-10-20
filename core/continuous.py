"""Continuous, jerk-limited joint reference generation.

The Mac real-robot deployment used Ruckig at 125 Hz to turn 10 Hz policy
actions into a smooth servo stream.  Ruckig is optional in the toolkit so a
machine that only needs teleoperation or replay still imports without extra
native dependencies.  When it is installed the exact online trajectory
generator is used; otherwise a bounded fallback keeps position, velocity,
acceleration and jerk inside the configured limits.

The class is deliberately independent of any robot, policy or camera code.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:  # optional native dependency
    from ruckig import InputParameter, OutputParameter, Result, Ruckig
except Exception:  # pragma: no cover - depends on the local environment
    InputParameter = OutputParameter = Result = Ruckig = None


def _as_vector(value, dof: int, name: str) -> np.ndarray:
    values = np.asarray(value, dtype=float)
    if values.ndim == 0:
        values = np.full(dof, float(values))
    if values.shape != (dof,):
        raise ValueError(f'{name} must be a scalar or a {dof}-element vector')
    return values


def _finite_bounds(value, dof: int, default: float, name: str) -> np.ndarray:
    values = _as_vector(default if value is None else value, dof, name)
    if not np.all(np.isfinite(values)):
        # Keep an explicit infinite bound effectively free, but replace NaN
        # with the conservative default.
        values = np.where(np.isnan(values), default, values)
        values = np.where(np.isposinf(values), 1e6, values)
        values = np.where(np.isneginf(values), -1e6, values)
    return np.clip(values, -1e6, 1e6)


class ContinuousReference:
    """One online reference generator for a single joint stream.

    ``step(target)`` advances by exactly ``dt`` and returns the next joint
    position.  The target may jump; the reference itself is continuous in
    position, velocity, acceleration and (with Ruckig, or approximately with
    the fallback) jerk.
    """

    def __init__(self, q, dt: float = 0.008, speed: float = 0.8,
                 acceleration: float = 1.2, jerk: float = 8.0,
                 lower=None, upper=None, backend: str = 'auto'):
        q = np.asarray(q, dtype=float).reshape(-1)
        if q.size == 0:
            raise ValueError('q must contain at least one joint')
        if not np.all(np.isfinite(q)):
            raise ValueError('q must contain finite joint angles')
        self.dt = float(dt)
        if self.dt <= 0:
            raise ValueError('dt must be positive')
        self.dof = int(q.size)

        self.max_velocity = _as_vector(speed, self.dof, 'speed')
        self.max_acceleration = _as_vector(acceleration, self.dof, 'acceleration')
        self.max_jerk = _as_vector(jerk, self.dof, 'jerk')
        if (np.any(self.max_velocity <= 0) or np.any(self.max_acceleration <= 0)
                or np.any(self.max_jerk <= 0)):
            raise ValueError('speed, acceleration and jerk must be positive')

        self.lower = _finite_bounds(lower, self.dof, -2.0 * np.pi, 'lower')
        self.upper = _finite_bounds(upper, self.dof, 2.0 * np.pi, 'upper')
        if np.any(self.lower >= self.upper):
            raise ValueError('lower bounds must be smaller than upper bounds')

        self.q = np.clip(q, self.lower, self.upper)
        self.v = np.zeros(self.dof, dtype=float)
        self.a = np.zeros(self.dof, dtype=float)

        requested = str(backend or 'auto').strip().lower()
        if requested not in ('auto', 'ruckig', 'fallback'):
            raise ValueError("backend must be 'auto', 'ruckig' or 'fallback'")
        if requested == 'ruckig' and Ruckig is None:
            raise ImportError('Ruckig 未安装：pip install ruckig，或使用 backend="fallback"')
        self.backend = ('ruckig' if Ruckig is not None and requested in ('auto', 'ruckig')
                        else 'fallback')
        self._otg = None
        self._inp = None
        self._out = None
        if self.backend == 'ruckig':
            self._init_ruckig()

    # ---------------------------------------------------------------- helpers
    def _init_ruckig(self) -> None:
        self._otg = Ruckig(self.dof, self.dt)
        self._inp = InputParameter(self.dof)
        self._out = OutputParameter(self.dof)
        self._inp.current_position = self.q.tolist()
        self._inp.current_velocity = [0.0] * self.dof
        self._inp.current_acceleration = [0.0] * self.dof
        self._inp.target_velocity = [0.0] * self.dof
        self._inp.target_acceleration = [0.0] * self.dof
        self._inp.max_velocity = self.max_velocity.tolist()
        self._inp.max_acceleration = self.max_acceleration.tolist()
        self._inp.max_jerk = self.max_jerk.tolist()

    def validate(self, target) -> np.ndarray:
        """Validate a target position and return it as a float vector."""
        target = np.asarray(target, dtype=float).reshape(-1)
        if target.shape != (self.dof,):
            raise ValueError(f'expected {self.dof} joint angles, got {target.shape}')
        if not np.all(np.isfinite(target)):
            raise ValueError('target must contain finite joint angles')
        if np.any(target < self.lower - 1e-9) or np.any(target > self.upper + 1e-9):
            raise ValueError('joint target outside configured local limits')
        return target

    # ---------------------------------------------------------------- update
    def step(self, target) -> np.ndarray:
        target = self.validate(target)
        if self.backend == 'ruckig':
            return self._step_ruckig(target)
        return self._step_fallback(target)

    def _step_ruckig(self, target: np.ndarray) -> np.ndarray:
        self._inp.target_position = target.tolist()
        result = self._otg.update(self._inp, self._out)
        if result not in (Result.Working, Result.Finished):
            raise RuntimeError(f'Ruckig trajectory error: {result}')
        self.q = np.asarray(self._out.new_position, dtype=float)
        self.v = np.asarray(self._out.new_velocity, dtype=float)
        self.a = np.asarray(self._out.new_acceleration, dtype=float)
        self.validate(self.q)
        self._out.pass_to_input(self._inp)
        return self.q.copy()

    def _step_fallback(self, target: np.ndarray) -> np.ndarray:
        """Bounded second-order tracker.

        It is not a full time-optimal S-curve, but every output respects the
        configured velocity, acceleration and jerk bounds and converges to a
        static target.  Install Ruckig for the deployment-grade generator.
        """
        dt = self.dt
        error = target - self.q
        # Desired speed is the smaller of a proportional demand and the speed
        # from which the remaining distance can still be braked at ``amax``.
        desired = np.clip(2.0 * error, -self.max_velocity, self.max_velocity)
        # A conservative braking envelope leaves room for the jerk-limited
        # acceleration ramp; using the full ``v^2/(2a)`` distance overshoots.
        braking = np.sqrt(np.maximum(0.0, 0.5 * self.max_acceleration * np.abs(error)))
        desired = np.sign(desired) * np.minimum(np.abs(desired), braking)
        desired_acceleration = np.clip((desired - self.v) / dt,
                                       -self.max_acceleration, self.max_acceleration)
        jerk_step = self.max_jerk * dt
        self.a = self.a + np.clip(desired_acceleration - self.a, -jerk_step, jerk_step)
        self.a = np.clip(self.a, -self.max_acceleration, self.max_acceleration)
        self.v = np.clip(self.v + self.a * dt, -self.max_velocity, self.max_velocity)
        self.q = self.q + self.v * dt

        # Do not push through a joint bound; stop the reference at the bound.
        below = self.q < self.lower
        above = self.q > self.upper
        if np.any(below) or np.any(above):
            self.q = np.clip(self.q, self.lower, self.upper)
            self.v = np.where(below | above, 0.0, self.v)
            self.a = np.where(below | above, 0.0, self.a)

        # Settle exactly on a reached target instead of creeping around it.
        settled = (np.abs(target - self.q) < 1e-6) & (np.abs(self.v) < 1e-3)
        if np.any(settled):
            self.q = np.where(settled, target, self.q)
            self.v = np.where(settled, 0.0, self.v)
            self.a = np.where(settled, 0.0, self.a)
        return self.q.copy()


def progress_allowed(reference_gap: float, tracking_gap: float,
                     max_reference_gap: float, max_tracking_gap: float) -> bool:
    """Shared action clock gate: never advance actions ahead of the arm."""
    return float(reference_gap) <= float(max_reference_gap) and \
        float(tracking_gap) <= float(max_tracking_gap)


def gripper_allowed(reference_gap: float, tracking_gap: float,
                    reference_tolerance: float, tracking_tolerance: float) -> bool:
    """Gripper changes wait until the arm has reached the action target."""
    return (float(reference_gap) <= float(reference_tolerance)
            and float(tracking_gap) <= float(tracking_tolerance))


@dataclass
class ContinuousReferenceInfo:
    """Small diagnostic summary used by callers and tests."""

    backend: str
    dof: int
    dt: float

    def as_dict(self) -> dict:
        return {'backend': self.backend, 'dof': self.dof, 'dt': self.dt}
