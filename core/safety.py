"""Joint, velocity and workspace limits applied independently of the backend.

The design rule is that safety is a *wrapper*, never a backend feature.  That
way a simulator, a UR and a future arm all go through exactly the same checks.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..errors import SafetyViolation


@dataclass
class JointLimits:
    lower: np.ndarray
    upper: np.ndarray
    max_velocity: np.ndarray | float = 6
    max_acceleration: np.ndarray | float | None = None

    def __post_init__(self):
        self.lower = np.asarray(self.lower, dtype=float)
        self.upper = np.asarray(self.upper, dtype=float)
        if self.lower.shape != self.upper.shape:
            raise ValueError('lower/upper joint limits must have the same shape')
        if np.any(self.lower > self.upper):
            raise ValueError('lower joint limits must not exceed upper limits')
        self.max_velocity = np.broadcast_to(np.asarray(self.max_velocity, dtype=float),
                                            self.lower.shape).copy()

    @property
    def dof(self) -> int:
        return int(self.lower.size)

    @classmethod
    def symmetric(cls, dof: int, bound: float = np.pi, max_velocity: float | None = None) -> 'JointLimits':
        lower = np.full(dof, -abs(bound))
        upper = np.full(dof, abs(bound))
        if max_velocity is None:
            return cls(lower, upper)
        return cls(lower, upper, max_velocity)

    def clamp(self, joints) -> np.ndarray:
        return np.clip(np.asarray(joints, dtype=float), self.lower, self.upper)

    def violates(self, joints) -> bool:
        values = np.asarray(joints, dtype=float)
        return bool(np.any(values < self.lower - 1e-9) or np.any(values > self.upper + 1e-9))

    def max_step(self, dt: float) -> np.ndarray:
        return np.maximum(self.max_velocity * max(float(dt), 0.0), 1e-9)

    def rate_limit(self, previous, target, dt: float) -> np.ndarray:
        previous = np.asarray(previous, dtype=float)
        target = self.clamp(target)
        step = self.max_step(dt)
        return previous + np.clip(target - previous, -step, step)


@dataclass
class WorkspaceBox:
    """Cartesian safety box in the robot base frame."""

    lower: np.ndarray = field(default_factory=lambda: np.array([-0.75, -0.75, 0.0]))
    upper: np.ndarray = field(default_factory=lambda: np.array([0.85, 0.75, 1.0]))

    def __post_init__(self):
        self.lower = np.asarray(self.lower, dtype=float).reshape(3)
        self.upper = np.asarray(self.upper, dtype=float).reshape(3)

    def clamp(self, position) -> np.ndarray:
        return np.clip(np.asarray(position, dtype=float).reshape(3), self.lower, self.upper)

    def contains(self, position, tolerance: float = 1e-9) -> bool:
        point = np.asarray(position, dtype=float).reshape(3)
        return bool(np.all(point >= self.lower - tolerance) and
                    np.all(point <= self.upper + tolerance))


@dataclass
class SafetyDecision:
    target: np.ndarray
    accepted: bool
    reason: str = ''
    clipped: bool = False
    rate_limited: bool = False


class SafetyGuard:
    """Clamp joint targets and reject impossible jumps.

    ``max_step_rad`` is an absolute per-tick bound.  It protects against a bad
    IK solution even when the velocity limit is permissive, which is the failure
    mode that damages hardware.
    """

    def __init__(self, limits: JointLimits, max_step_rad: float = 0.8,
                 reject_on_limit: bool = False):
        self.limits = limits
        self.max_step_rad = float(max_step_rad)
        self.reject_on_limit = bool(reject_on_limit)

    def evaluate(self, current, target, dt: float) -> SafetyDecision:
        current = np.asarray(current, dtype=float)
        target = np.asarray(target, dtype=float)
        if target.shape != current.shape:
            raise ValueError(f'target shape {target.shape} != current shape {current.shape}')
        clipped = self.limits.violates(target)
        clamped = self.limits.clamp(target)
        step = min(self.max_step_rad, float(np.min(self.limits.max_step(dt))))
        delta = clamped - current
        rate_limited = bool(np.max(np.abs(delta)) > step + 1e-12)
        limited_target = current + np.clip(delta, -step, step)
        if clipped and self.reject_on_limit:
            return SafetyDecision(limited_target, False, 'joint limit exceeded', clipped, rate_limited)
        return SafetyDecision(limited_target, True, '', clipped, rate_limited)

    def check(self, current, target, dt: float) -> np.ndarray:
        decision = self.evaluate(current, target, dt)
        if not decision.accepted:
            raise SafetyViolation(decision.reason)
        return decision.target

    def check_velocity(self, joints, dt: float, speed_limit: float | None = None) -> bool:
        if dt <= 0:
            return True
        values = np.asarray(joints, dtype=float)
        velocity = np.abs(values) / dt
        limit = (self.limits.max_velocity if speed_limit is None
                 else np.broadcast_to(np.asarray(speed_limit, dtype=float), values.shape))
        return bool(np.all(velocity <= limit + 1e-9))


class Watchdog:
    """Monotonic deadline used to abort a loop when input stops arriving."""

    def __init__(self, timeout_s: float, clock=None):
        import time
        self.timeout_s = float(timeout_s)
        self.clock = clock or time.monotonic
        self._last = self.clock()

    def feed(self) -> None:
        self._last = self.clock()

    def expired(self) -> bool:
        return (self.clock() - self._last) > self.timeout_s

    @property
    def age(self) -> float:
        return self.clock() - self._last
