"""Wrap ``projects.ur5e_real_collect.robot.URRobot`` as a toolkit backend.

This is the migration path for the existing application: its mature UR driver
(including the CB3 freedrive fallback) keeps working, while teaching, replay and
collection can be shared with every other robot.
"""

from __future__ import annotations

import numpy as np

from ..core.conventions import RobotSpec
from ..robot.base import Capabilities, RobotBackend, RobotState


class LegacyUrRobot(RobotBackend):
    def __init__(self, ur_robot, spec: RobotSpec | None = None):
        super().__init__(spec or RobotSpec.ur5e(),
                         Capabilities(servo_j=True, servo_l=True, move_j=True,
                                      move_l=True, freedrive=True))
        self.ur = ur_robot
        self._connected = bool(getattr(ur_robot, 'connected', False))

    # ------------------------------------------------------------------ link
    def connect(self, *args, **kwargs):
        self.ur.connect(*args, **kwargs)
        self._connected = bool(getattr(self.ur, 'connected', True))
        return self

    def disconnect(self):
        close = getattr(self.ur, 'disconnect', None)
        if close is not None:
            close()
        self._connected = False

    @property
    def connected(self) -> bool:
        return bool(getattr(self.ur, 'connected', self._connected))

    # ----------------------------------------------------------------- state
    def state(self) -> RobotState:
        raw = self.ur.state()
        return RobotState(t=float(getattr(raw, 'timestamp', 0.0)),
                          q=np.asarray(raw.joints, dtype=float),
                          qd=np.asarray(getattr(raw, 'qd', []), dtype=float),
                          tcp=np.asarray(getattr(raw, 'tcp_pose', []), dtype=float),
                          protective_stop=bool(getattr(raw, 'protective_stop', False)),
                          connected=self.connected)

    def protective_stop(self) -> bool:
        return bool(getattr(self.ur.state(), 'protective_stop', False))

    def joints_within_safety(self, joints) -> bool:
        function = getattr(self.ur, 'joints_within_safety', None)
        return bool(function(joints)) if function is not None else super().joints_within_safety(joints)

    def telemetry(self) -> dict:
        function = getattr(self.ur, 'telemetry', None)
        return dict(function()) if function is not None else {}

    def init_period(self):
        function = getattr(self.ur, 'init_period', None)
        return function() if function is not None else None

    def wait_period(self, started) -> bool:
        function = getattr(self.ur, 'wait_period', None)
        return bool(function(started)) if function is not None else False

    # ---------------------------------------------------------------- motion
    def servo_j(self, joints, dt: float) -> bool:
        return bool(self.ur.servo_j(joints, dt))

    def servo_l(self, pose, dt: float) -> bool:
        return bool(self.ur.servo_l(pose, dt))

    def move_j(self, joints, speed=None, accel=None, wait=True, rate_hz=125.0,
               timeout=None, abort_event=None) -> bool:
        if hasattr(self.ur, 'move_j'):
            self.ur.move_j(joints, speed, accel, not wait)
            if not wait:
                return True
        return self.wait_until_settled(joints, timeout=timeout or 60.0,
                                       rate_hz=rate_hz, abort_event=abort_event)

    def move_l(self, pose, speed=None, accel=None, wait=True, rate_hz=125.0,
               timeout=None, abort_event=None) -> bool:
        if hasattr(self.ur, 'move_l'):
            self.ur.move_l(pose, speed, accel, not wait)
            if not wait:
                return True
        return self.wait_pose(pose, timeout=timeout or 60.0)

    def stop_motion(self, deceleration: float = 2.0) -> bool:
        for method in ('servo_stop', 'stop_motion'):
            function = getattr(self.ur, method, None)
            if function is not None:
                return bool(function(deceleration))
        return False

    def freedrive(self, enabled: bool) -> bool:
        return bool(self.ur.freedrive(enabled))

    def inverse_kinematics(self, pose):
        function = getattr(self.ur, 'get_inverse_kin', None)
        return [] if function is None else list(function(pose))

    def __getattr__(self, item):
        return getattr(self.ur, item)
