"""Deterministic in-process robot used for tests, demos and dry runs.

It is a genuine first-order joint model: ``servo_j`` does not teleport, so a
recording made against it has realistic timing and a replay can be verified
without any hardware.
"""

from __future__ import annotations

import time

import numpy as np

from ..core.conventions import RobotSpec
from ..core.safety import JointLimits
from ..errors import NotSupported
from .base import Capabilities, RobotBackend, RobotState


class MockRobot(RobotBackend):
    def __init__(self, spec: RobotSpec | None = None, home=None, max_velocity: float = 2.0,
                 limits: JointLimits | None = None, forward_kinematics=None,
                 inverse_kinematics=None, clock=time.monotonic):
        spec = spec or RobotSpec.generic(6, name='mock')
        super().__init__(spec, Capabilities(servo_j=True, move_j=True,
                                            servo_l=inverse_kinematics is not None,
                                            freedrive=True, gripper=spec.gripper is not None))
        self.clock = clock
        self._t0 = clock()
        self.max_velocity = float(max_velocity)
        self.limits = limits or JointLimits.symmetric(spec.dof, np.pi, self.max_velocity)
        self.home = (np.asarray(home, dtype=float) if home is not None
                     else np.zeros(spec.dof))
        if self.home.shape != (spec.dof,):
            raise ValueError(f'home has shape {self.home.shape}, expected {(spec.dof,)}')
        self._q = self.home.copy()
        self._target = self.home.copy()
        self._qd = np.zeros(spec.dof)
        self._freedrive = False
        self._fk = forward_kinematics
        self._ik = inverse_kinematics
        self._gripper = spec.gripper.max_width if spec.gripper else None
        self.commands: list[np.ndarray] = []
        self.mock = True
        # A mock has no link to establish; being usable immediately makes it a
        # drop-in replacement in tests and demos.
        self._connected = True

    # ------------------------------------------------------------------ link
    def connect(self):
        self._connected = True
        return self

    def disconnect(self):
        self._connected = False
        self._freedrive = False

    # ----------------------------------------------------------------- state
    def _integrate(self, dt: float) -> None:
        self._integrate_to(dt)

    def _integrate_to(self, dt: float, target=None) -> None:
        target = self._target if target is None else np.asarray(target, dtype=float)
        delta = target - self._q
        step = self.max_velocity * max(0.0, dt)
        if step <= 0:
            self._qd = np.zeros_like(self._q)
            return
        clipped = np.clip(delta, -step, step)
        self._q = self._q + clipped
        self._qd = clipped / max(dt, 1e-9)

    def state(self) -> RobotState:
        tcp = None
        if self._fk is not None:
            tcp = np.asarray(self._fk(self._q), dtype=float)
        return RobotState(t=self.clock() - self._t0, q=self._q.copy(), qd=self._qd.copy(),
                          tcp=tcp, protective_stop=False, connected=self._connected,
                          extra={'target': self._target.copy(), 'gripper': self._gripper,
                                 'freedrive': self._freedrive})

    # ---------------------------------------------------------------- motion
    def servo_j(self, joints, dt: float) -> bool:
        target = np.asarray(joints, dtype=float)
        if target.shape != self._q.shape:
            raise ValueError(f'expected {self._q.shape[0]} joints, got {target.shape}')
        self._target = self.limits.clamp(target)
        self.commands.append(self._target.copy())
        self._integrate(dt)
        return self._connected

    def servo_l(self, pose, dt: float) -> bool:
        if self._ik is None:
            raise NotSupported('MockRobot needs inverse_kinematics to accept Cartesian targets')
        joints = np.asarray(self._ik(np.asarray(pose, dtype=float)), dtype=float)
        return self.servo_j(joints, dt)

    def freedrive(self, enabled: bool) -> bool:
        self._freedrive = bool(enabled)
        if enabled:
            self._target = self._q.copy()
        return True

    def set_gripper(self, width: float) -> None:
        self._gripper = self.spec.gripper.clamp(width) if self.spec.gripper else width

    # ----------------------------------------------------------------- helpers
    def simulate(self, seconds: float, dt: float = 0.01, target=None) -> None:
        """Advance the joint model without issuing new commands."""
        target = self._target if target is None else np.asarray(target, dtype=float)
        for _ in range(max(1, int(round(seconds / dt)))):
            self._integrate_to(dt, target)
