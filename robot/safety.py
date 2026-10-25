"""Safety wrapper: the only path through which a control loop should command motion."""

from __future__ import annotations

import numpy as np

from ..core.safety import SafetyGuard
from ..errors import NotSupported, SafetyViolation
from .base import RobotBackend, RobotState


class SafeRobot(RobotBackend):
    """Decorator that clamps every joint target to configured limits.

    Because it is a ``RobotBackend`` itself, the teaching, replay and teleop
    layers can all be written once against the safe variant and never worry
    about which concrete robot is underneath.
    """

    def __init__(self, inner: RobotBackend, guard: SafetyGuard, raise_on_limit: bool = True,
                 workspace=None):
        super().__init__(inner.spec, inner.capabilities)
        self.inner = inner
        self.guard = guard
        self.raise_on_limit = bool(raise_on_limit)
        self.clamped_commands = 0
        self.rejected_commands = 0
        self._connected = inner.connected
        self.workspace = workspace

    # ------------------------------------------------------------------ link
    def connect(self):
        self.inner.connect()
        self._connected = self.inner.connected
        return self

    def disconnect(self):
        self.inner.disconnect()
        self._connected = False

    def connect_read_only(self):
        self.inner.connect_read_only()
        self._connected = self.inner.connected
        return self

    def prepare_motion(self):
        return self.inner.prepare_motion()

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

    def forward_kinematics(self, joints):
        return self.inner.forward_kinematics(joints)

    def check_workspace_path(self, current, target):
        if self.workspace is not None:
            self.workspace.check_path(current, target)

    def init_period(self):
        return self.inner.init_period()

    def wait_period(self, started) -> bool:
        return self.inner.wait_period(started)

    # ---------------------------------------------------------------- motion
    def servo_j(self, joints, dt: float) -> bool:
        state = self.inner.state()
        current = state.q
        if current is None:
            raise SafetyViolation('cannot guard a servo command without joint state')
        decision = self.guard.evaluate(current, joints, dt)
        if decision.clipped or decision.rate_limited:
            self.clamped_commands += 1
        if not decision.accepted and self.raise_on_limit:
            self.rejected_commands += 1
            raise SafetyViolation(decision.reason or 'command rejected by safety guard')
        if self.workspace is not None:
            try:
                self.workspace.check_feedback(state)
                self.workspace.check_path(current, decision.target)
            except Exception:
                self.inner.stop_motion()
                raise
        return self.inner.servo_j(decision.target, dt)

    def servo_l(self, pose, dt: float) -> bool:
        """Cartesian servo delegates: the backend runs IK, so validate afterwards."""
        if self.workspace is not None:
            raise SafetyViolation('workspace guarded Cartesian motion needs a validated IK path')
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

    def _check_joint_target(self, joints):
        target = np.asarray(joints, dtype=float)
        if self.guard.limits.violates(target):
            raise SafetyViolation('target violates hardware joint limits')
        if self.workspace is not None:
            state = self.inner.state()
            self.workspace.check_feedback(state)
            self.workspace.check_path(state.q, target)
        return target

    def _has_native_move_j(self) -> bool:
        """True when the inner backend has a controller-side joint planner."""
        if getattr(self.inner, 'mock', False):
            return False
        return type(self.inner).move_j is not RobotBackend.move_j

    def move_j(self, joints, use_native: bool = True, **kwargs) -> bool:
        target = self._check_joint_target(joints)
        if use_native and self._has_native_move_j():
            try:
                return self.inner.move_j(target, **kwargs)
            except NotSupported:
                pass
        return RobotBackend.move_j(self, target, **kwargs)

    def move_l(self, pose, *args, **kwargs) -> bool:
        if self.workspace is not None:
            raise SafetyViolation('workspace guarded Cartesian motion needs a validated IK path')
        return self.inner.move_l(pose, *args, **kwargs)

    def inverse_kinematics(self, pose, seed=None):
        if seed is None:
            return self.inner.inverse_kinematics(pose)
        return self.inner.inverse_kinematics(pose, seed=seed)

    def set_inverse_kinematics(self, solver):
        self.inner.set_inverse_kinematics(solver)
        self.capabilities = self.inner.capabilities
        return self

    def move_p(self, pose, wait: bool = True, timeout: float | None = None,
               use_native: bool = True, speed: float | None = None,
               accel: float | None = None) -> bool:
        if self.workspace is not None:
            raise SafetyViolation('workspace guarded Cartesian motion needs a validated IK path')
        # Prefer a native P mover (Piper).  If it is unavailable (for example
        # the Piper mock has no Cartesian model), fall back to host IK.
        if use_native and type(self.inner).move_p is not RobotBackend.move_p:
            try:
                return self.inner.move_p(pose, wait=wait, timeout=timeout,
                                         speed=speed, accel=accel)
            except NotSupported:
                if not self.capabilities.inverse_kinematics:
                    raise
        return RobotBackend.move_p(self, pose, wait=wait, timeout=timeout,
                                   use_native=False, speed=speed, accel=accel)

    def wait_until_settled(self, target=None, timeout: float = 10.0,
                           rate_hz: float = 50.0, tolerance: float = 0.02,
                           abort_event=None) -> bool:
        return self.inner.wait_until_settled(
            target, timeout=timeout, rate_hz=rate_hz,
            tolerance=tolerance, abort_event=abort_event)

    def stop_motion(self, deceleration: float = 2.0) -> bool:
        return self.inner.stop_motion(deceleration)

    def freedrive(self, enabled: bool) -> bool:
        return self.inner.freedrive(enabled)

    def __getattr__(self, item):
        # Convenience passthrough for backend-specific diagnostics only.
        return getattr(self.inner, item)

    def __repr__(self) -> str:
        return f'SafeRobot({self.inner!r})'
