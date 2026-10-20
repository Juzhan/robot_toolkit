"""Safety wrapper: the only path through which a control loop should command motion."""

from __future__ import annotations

import numpy as np

from ..core.safety import SafetyGuard
from ..errors import SafetyViolation
from .base import RobotBackend, RobotState


class SafeRobot(RobotBackend):
    """Decorator that clamps every joint target to configured limits.

    Because it is a ``RobotBackend`` itself, the teaching, replay and teleop
    layers can all be written once against the safe variant and never worry
    about which concrete robot is underneath.
    """

    def __init__(self, inner: RobotBackend, guard: SafetyGuard, raise_on_limit: bool = True):
        super().__init__(inner.spec, inner.capabilities)
        self.inner = inner
        self.guard = guard
        self.raise_on_limit = bool(raise_on_limit)
        self.clamped_commands = 0
        self.rejected_commands = 0
        self._connected = inner.connected

    # ------------------------------------------------------------------ link
    def connect(self):
        self.inner.connect()
        self._connected = self.inner.connected
        return self

    def disconnect(self):
        self.inner.disconnect()
        self._connected = False

    @property
    def connected(self) -> bool:
        return self.inner.connected

    # ----------------------------------------------------------------- state
    def state(self) -> RobotState:
        return self.inner.state()

    def protective_stop(self) -> bool:
        return self.inner.protective_stop()

    def leader_joint_angles(self):
        return self.inner.leader_joint_angles()

    def joints_within_safety(self, joints) -> bool:
        return self.inner.joints_within_safety(joints)

    def telemetry(self) -> dict:
        return self.inner.telemetry()

    def init_period(self):
        return self.inner.init_period()

    def wait_period(self, started) -> bool:
        return self.inner.wait_period(started)

    # ---------------------------------------------------------------- motion
    def servo_j(self, joints, dt: float) -> bool:
        current = self.inner.state().q
        if current is None:
            raise SafetyViolation('cannot guard a servo command without joint state')
        decision = self.guard.evaluate(current, joints, dt)
        if decision.clipped or decision.rate_limited:
            self.clamped_commands += 1
        if not decision.accepted and self.raise_on_limit:
            self.rejected_commands += 1
            raise SafetyViolation(decision.reason or 'command rejected by safety guard')
        return self.inner.servo_j(decision.target, dt)

    def servo_l(self, pose, dt: float) -> bool:
        """Cartesian servo delegates: the backend runs IK, so validate afterwards."""
        before = self.inner.state().q
        ok = self.inner.servo_l(pose, dt)
        after = self.inner.state().q
        if before is not None and after is not None:
            delta = np.abs(np.asarray(after) - np.asarray(before))
            limit = self.guard.limits.max_step(dt)
            if np.any(delta > limit + 1e-6):
                self.inner.stop_motion()
                self.rejected_commands += 1
                raise SafetyViolation('Cartesian servo implied a joint step beyond the limit')
        return ok

    def move_j(self, joints, **kwargs) -> bool:
        # Use the generic interpolator so every intermediate target is guarded.
        return RobotBackend.move_j(self, joints, **kwargs)

    def move_l(self, pose, *args, **kwargs) -> bool:
        return self.inner.move_l(pose, *args, **kwargs)

    def stop_motion(self, deceleration: float = 2.0) -> bool:
        return self.inner.stop_motion(deceleration)

    def freedrive(self, enabled: bool) -> bool:
        return self.inner.freedrive(enabled)

    def __getattr__(self, item):
        # Convenience passthrough for backend-specific diagnostics only.
        return getattr(self.inner, item)

    def __repr__(self) -> str:
        return f'SafeRobot({self.inner!r})'
