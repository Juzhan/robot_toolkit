"""Reusable Cartesian teleoperation loop.

The runner contains no robot-specific or device-specific control logic:

* a :class:`~.base.TeleopSource` produces normalized commands;
* :class:`~.mapping.CartesianTeleop` turns them into a Cartesian target;
* the backend either accepts ``servo_l`` (UR) or a slower ``move_p`` stream
  (Piper / other arms that do their own IK).

This is the layer that lets the same Joy-Con / phone / scripted source drive a
UR5e and a Piper X without changing the mapping or recording code.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from ..core.mathutils import pose_error, rotvec_to_matrix
from ..errors import NotSupported
from .base import TeleopSource
from .mapping import CartesianTeleop


@dataclass
class TeleopRunnerConfig:
    rate_hz: float = 50.0
    stale_s: float = 0.3
    max_duration_s: float = 0.0
    move_p_hz: float = 20.0
    gripper_hz: float = 20.0
    gripper_epsilon: float = 0.0015
    tracking_translation_m: float | None = None
    tracking_rotation_rad: float | None = None
    connect: bool = True
    realtime: bool = True


@dataclass
class TeleopRunResult:
    frames: int = 0
    duration_s: float = 0.0
    aborted: bool = False
    reason: str = ''
    limited: int = 0
    max_tracking_error_m: float = 0.0
    max_tracking_error_rad: float = 0.0

    def as_dict(self) -> dict:
        return dict(frames=self.frames, duration_s=self.duration_s,
                    aborted=self.aborted, reason=self.reason,
                    limited=self.limited,
                    max_tracking_error_m=self.max_tracking_error_m,
                    max_tracking_error_rad=self.max_tracking_error_rad)


class TeleopRunner:
    """Run a normalized command source against any ``RobotBackend``.

    Use ``servo_l`` when the backend advertises it.  Otherwise use a
    rate-limited ``move_p`` stream, which is what Piper exposes on top of its
    controller-side IK.  The mapping, safety wrapper and recording layers stay
    identical in both cases.
    """

    def __init__(self, robot, gripper, source: TeleopSource,
                 mapping: CartesianTeleop, config: TeleopRunnerConfig | None = None,
                 clock=time.monotonic):
        self.robot = robot
        self.gripper = gripper
        self.source = source
        self.mapping = mapping
        self.config = config or TeleopRunnerConfig()
        self.clock = clock

    def _send_cartesian(self, target, dt: float, now: float) -> bool:
        capabilities = getattr(self.robot, 'capabilities', None)
        if capabilities is not None and capabilities.servo_l:
            return bool(self.robot.servo_l(target.pose, dt))
        if capabilities is not None and capabilities.move_p:
            # The controller does its own IK; only send a new target at a
            # moderate rate.  Higher rates just restart the controller planner.
            if now - getattr(self, '_last_move_p', -1e9) < 1.0 / max(
                    self.config.move_p_hz, 1.0):
                return True
            self._last_move_p = now
            return bool(self.robot.move_p(target.pose, wait=False))
        raise NotSupported(f'{self.robot.name}: neither servo_l nor move_p is available')

    def _send_gripper(self, width: float, now: float) -> None:
        if self.gripper is None:
            return
        if now - getattr(self, '_last_gripper', -1e9) < 1.0 / max(
                self.config.gripper_hz, 1.0):
            return
        if abs(float(width) - getattr(self, '_last_width', float('inf'))) < self.config.gripper_epsilon:
            return
        self.gripper.move(width, wait=False)
        self._last_width = float(width)
        self._last_gripper = now

    def run(self, abort_event=None, on_sample=None) -> TeleopRunResult:
        result = TeleopRunResult()
        if self.config.connect:
            if not getattr(self.robot, 'connected', False):
                self.robot.connect()
            if self.gripper is not None and not getattr(self.gripper, 'connected', False):
                self.gripper.connect()
            self.source.connect()

        state = self.robot.state()
        if state.tcp is None:
            raise NotSupported(f'{self.robot.name}: state has no TCP pose')
        width = None
        if self.gripper is not None:
            try:
                width = float(self.gripper.width)
            except Exception:
                width = None
        self.mapping.reset(state.tcp[:3], rotvec_to_matrix(state.tcp[3:]),
                           width, enabled=False)
        self._last_width = width if width is not None else float('inf')
        self._last_gripper = -1e9
        self._last_move_p = -1e9
        last_command = self.clock()
        started = self.clock()
        was_active = False
        dt = 1.0 / max(1.0, float(self.config.rate_hz))
        period = dt
        try:
            while True:
                now = self.clock()
                if abort_event is not None and abort_event.is_set():
                    result.aborted = True
                    result.reason = 'operator abort'
                    break
                if self.config.max_duration_s > 0 and now - started >= self.config.max_duration_s:
                    break
                command = self.source.poll()
                if command is None:
                    if now - last_command > max(0.0, self.config.stale_s):
                        result.aborted = True
                        result.reason = 'input stale'
                        break
                    if self.config.realtime:
                        time.sleep(max(0.0, period - (self.clock() - now)))
                    continue
                last_command = now
                command = command.clipped()
                if command.buttons.get('quit'):
                    result.aborted = True
                    result.reason = 'operator quit'
                    break
                state = self.robot.state()
                if state.tcp is None:
                    result.aborted = True
                    result.reason = 'robot lost TCP pose'
                    break
                target = self.mapping.update(
                    command, dt,
                    measured_position=state.tcp[:3],
                    measured_rotation=rotvec_to_matrix(state.tcp[3:]))
                if target.limited:
                    result.limited += 1
                if not target.active:
                    if was_active:
                        self.robot.stop_motion()
                    was_active = False
                else:
                    was_active = True
                    if not self._send_cartesian(target, dt, now):
                        result.aborted = True
                        result.reason = 'cartesian command failed'
                        break
                    self._send_gripper(target.gripper, now)
                    if on_sample is not None:
                        on_sample(state, target)
                if (self.config.tracking_translation_m is not None
                        or self.config.tracking_rotation_rad is not None):
                    translation, rotation = pose_error(state.tcp, target.pose)
                    result.max_tracking_error_m = max(result.max_tracking_error_m,
                                                      translation)
                    result.max_tracking_error_rad = max(result.max_tracking_error_rad,
                                                         rotation)
                    if (self.config.tracking_translation_m is not None
                            and translation > self.config.tracking_translation_m):
                        result.aborted = True
                        result.reason = f'cartesian tracking error {translation:.3f} m'
                        break
                    if (self.config.tracking_rotation_rad is not None
                            and rotation > self.config.tracking_rotation_rad):
                        result.aborted = True
                        result.reason = f'cartesian tracking error {rotation:.3f} rad'
                        break
                result.frames += 1
                if self.config.realtime:
                    time.sleep(max(0.0, period - (self.clock() - now)))
        except KeyboardInterrupt:
            result.aborted = True
            result.reason = 'keyboard interrupt'
        finally:
            result.duration_s = self.clock() - started
            if result.aborted:
                try:
                    self.robot.stop_motion()
                except Exception:
                    pass
        return result
