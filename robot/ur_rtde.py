"""Universal Robots adapter (UR3/UR5/UR10, CB3 and e-Series) via ``ur_rtde``.

The same class is used for real hardware and for dry runs (``mock=True``), so
teaching / replay / collection code never branches on the robot.
"""

from __future__ import annotations

import platform
import time
from dataclasses import dataclass, field

import numpy as np

from ..core.conventions import RobotSpec
from ..errors import ConnectionFailed, NotSupported, RobotUnavailable
from .base import Capabilities, RobotBackend, RobotState
from .mock import MockRobot


@dataclass
class UrRtdeConfig:
    ip: str = '192.168.1.10'
    frequency: float = 125.0
    home_joints: list = field(default_factory=lambda: [0.0, -1.5708, 1.5708,
                                                       -1.5708, -1.5708, 0.0])
    payload_kg: float = 0.0
    payload_cog: list = field(default_factory=lambda: [0.0, 0.0, 0.06])
    tcp_offset: list = field(default_factory=lambda: [0.0] * 6)
    servo_lookahead: float = 0.1
    servo_gain: int = 300
    # Use RTDEControlInterface.initPeriod()/waitPeriod() for a stable 125 Hz
    # stream.  The proven Mac deployment keeps the host clock on macOS because
    # ur_rtde 1.6.5 implements waitPeriod relatively there; Windows/Linux use
    # the controller-side period.
    use_sdk_period: bool = True
    approach_speed: float = 0.4
    approach_accel: float = 0.8
    max_joint_speed: float = 1.0
    freedrive_axes: list = field(default_factory=lambda: [1, 1, 1, 1, 1, 1])
    mock: bool = False


class UrRtdeRobot(RobotBackend):
    def __init__(self, config: UrRtdeConfig | None = None, spec: RobotSpec | None = None,
                 mock: bool | None = None):
        spec = spec or RobotSpec.ur5e()
        self.config = config or UrRtdeConfig()
        self.mock = bool(self.config.mock if mock is None else mock)
        super().__init__(spec, Capabilities(servo_j=True, servo_l=True, move_j=True,
                                            move_l=True,
                                            inverse_kinematics=not self.mock,
                                            freedrive=True, gripper=False))
        self.control = None
        self.receive = None
        self._mock_robot = MockRobot(spec, home=self.config.home_joints,
                                     max_velocity=self.config.max_joint_speed,
                                     forward_kinematics=lambda q: np.zeros(6)) if self.mock else None
        self._freedrive = False
        self._t0 = time.monotonic()

    # ------------------------------------------------------------------ link
    def connect(self, with_control: bool = True):
        if self.mock:
            self._connected = True
            return self
        if with_control and not self.restart_control():
            raise ConnectionFailed(f'RTDE control 连接失败：{self.config.ip}，请检查网线和机器人 IP')
        import rtde_receive
        self.receive = rtde_receive.RTDEReceiveInterface(self.config.ip, self.config.frequency)
        if not self.receive.isConnected():
            raise ConnectionFailed(f'RTDE receive 未连接：{self.config.ip}')
        self._connected = True
        return self

    def restart_control(self) -> bool:
        if self.mock:
            return True
        import rtde_control
        if self.control is not None:
            try:
                self.control.disconnect()
            except Exception:
                pass
            self.control = None
        for _ in range(3):
            try:
                control = rtde_control.RTDEControlInterface(self.config.ip, self.config.frequency)
                if control.isConnected():
                    self.control = control
                    if self.config.payload_kg:
                        control.setPayload(float(self.config.payload_kg), list(self.config.payload_cog))
                    if any(abs(value) > 1e-9 for value in self.config.tcp_offset):
                        control.setTcp(list(self.config.tcp_offset))
                    return True
                control.disconnect()
            except Exception:
                # Transient right after a protective stop / e-stop.
                pass
            time.sleep(1.5)
        return False

    def disconnect(self):
        if self.mock:
            self._connected = False
            return
        for interface in (self.control, self.receive):
            if interface is not None:
                try:
                    interface.disconnect()
                except Exception:
                    pass
        self.control = None
        self.receive = None
        self._connected = False

    # ----------------------------------------------------------------- state
    def state(self) -> RobotState:
        if self.mock:
            sample = self._mock_robot.state()
            sample.t = time.monotonic() - self._t0
            return sample
        if self.receive is None:
            raise RobotUnavailable('RTDE receive interface is not connected')
        joints = np.asarray(self.receive.getActualQ(), dtype=float)
        qd = np.asarray(self.receive.getActualQd(), dtype=float)
        tcp = np.asarray(self.receive.getActualTCPPose(), dtype=float)
        try:
            timestamp = float(self.receive.getTimestamp())
            timestamp_clock = 'controller'
        except Exception:
            timestamp = time.monotonic() - self._t0
            timestamp_clock = 'host_elapsed'
        return RobotState(t=timestamp, q=joints, qd=qd, tcp=tcp,
                          protective_stop=bool(self.receive.isProtectiveStopped()),
                          connected=self.connected, timestamp_clock=timestamp_clock,
                          protective_stop_known=True,
                          sources={'q': 'controller', 'qd': 'controller', 'tcp': 'controller',
                                   'protective_stop': 'controller'})

    def protective_stop(self) -> bool:
        if self.mock:
            return False
        try:
            return bool(self.receive.isProtectiveStopped()) or bool(
                self.receive.isEmergencyStopped())
        except Exception:
            return True

    def joints_within_safety(self, joints) -> bool:
        if self.mock or self.control is None:
            return bool(np.all(np.isfinite(np.asarray(joints, dtype=float))))
        try:
            return bool(self.control.isJointsWithinSafetyLimits(
                list(map(float, joints))))
        except Exception as exc:
            # Unknown safety state must fail closed, not look like a green light.
            raise RobotUnavailable(f'机器人安全查询失败：{exc}') from exc

    # ------------------------------------------------------ realtime period
    def init_period(self):
        """Start the RTDE real-time period on Windows/Linux.

        macOS is deliberately excluded: the ur_rtde version used on the real
        robot implements ``waitPeriod`` relatively there, which caused drift in
        the 125 Hz loop.  The host-clock fallback is the tested Mac branch.
        """
        if (self.mock or not self.config.use_sdk_period or self.control is None
                or platform.system() == 'Darwin'):
            return None
        try:
            return float(self.control.initPeriod())
        except Exception:
            return None

    def wait_period(self, started) -> bool:
        if self.mock or started is None or self.control is None:
            return False
        try:
            self.control.waitPeriod(float(started))
            return True
        except Exception:
            return False

    def telemetry(self) -> dict:
        """Read-only controller targets used by control/quality diagnostics."""
        if self.mock:
            q = getattr(self._mock_robot, 'state')().extra.get('target')
            return {'controller_target_q': None if q is None else np.asarray(q).tolist(),
                    'controller_target_qd': [0.0] * self.dof,
                    'speed_scaling': 1.0, 'target_speed_fraction': 1.0}
        if self.receive is None:
            return {}
        values = {}
        for key, getter in (
                ('controller_target_q', 'getTargetQ'),
                ('controller_target_qd', 'getTargetQd'),
                ('speed_scaling', 'getSpeedScaling'),
                ('target_speed_fraction', 'getTargetSpeedFraction')):
            try:
                value = getattr(self.receive, getter)()
                values[key] = (float(value) if np.isscalar(value)
                               else np.asarray(value, dtype=float).tolist())
            except Exception:
                values[key] = None
        return values

    # ---------------------------------------------------------------- motion
    def servo_j(self, joints, dt: float) -> bool:
        if self.mock:
            return self._mock_robot.servo_j(joints, dt)
        if self.control is None:
            raise RobotUnavailable('RTDE control interface is not connected')
        return bool(self.control.servoJ(list(map(float, joints)), 0.0, 0.0, float(dt),
                                       float(self.config.servo_lookahead),
                                       float(self.config.servo_gain)))

    def servo_l(self, pose, dt: float) -> bool:
        if self.mock:
            return self._mock_robot.servo_l(pose, dt)
        if self.control is None:
            raise RobotUnavailable('RTDE control interface is not connected')
        return bool(self.control.servoL(list(map(float, pose)), 0.0, 0.0, float(dt),
                                       float(self.config.servo_lookahead * 0.5),
                                       float(self.config.servo_gain)))

    def move_j(self, joints, speed=None, accel=None, wait=True, rate_hz=125.0,
               timeout=None, abort_event=None) -> bool:
        if self.mock:
            return RobotBackend.move_j(self, joints, speed=speed, accel=accel, wait=wait,
                                       rate_hz=rate_hz, timeout=timeout,
                                       abort_event=abort_event)
        if self.control is None:
            raise RobotUnavailable('RTDE control interface is not connected')
        ok = bool(self.control.moveJ(
            list(map(float, joints)),
            float(speed if speed is not None else self.config.approach_speed),
            float(accel if accel is not None else self.config.approach_accel),
            not wait))
        if not ok or not wait:
            return ok
        return self.wait_until_settled(joints, timeout=timeout or 60.0,
                                       rate_hz=rate_hz, abort_event=abort_event)

    def move_l(self, pose, speed=None, accel=None, wait=True, rate_hz=125.0,
               timeout=None, abort_event=None) -> bool:
        if self.mock:
            raise NotSupported('MockRobot has no Cartesian model')
        if self.control is None:
            raise RobotUnavailable('RTDE control interface is not connected')
        ok = bool(self.control.moveL(
            list(map(float, pose)),
            float(speed if speed is not None else self.config.approach_speed * 0.5),
            float(accel if accel is not None else self.config.approach_accel * 0.5),
            not wait))
        if not ok or not wait:
            return ok
        return self.wait_pose(pose, timeout=timeout or 60.0)

    def stop_motion(self, deceleration: float = 2.0) -> bool:
        if self.mock:
            return True
        if self.control is None:
            return False
        try:
            return bool(self.control.servoStop(float(deceleration)))
        except Exception:
            return False

    def freedrive(self, enabled: bool) -> bool:
        if self.mock:
            self._freedrive = bool(enabled)
            return True
        if self.control is None:
            raise RobotUnavailable('freedrive 需要 RTDE control 连接')
        if enabled:
            ok = False
            if hasattr(self.control, 'freedriveMode'):
                try:
                    ok = bool(self.control.freedriveMode(list(self.config.freedrive_axes)))
                except Exception:
                    ok = False
            if not ok and hasattr(self.control, 'teachMode'):
                try:
                    ok = bool(self.control.teachMode())
                except Exception:
                    ok = False
            if not ok:
                raise RobotUnavailable(
                    '无法进入 Freedrive：确认机器人处于 Remote Control；CB3 需要 teach_mode 支持')
            self._freedrive = True
            return True
        ended = False
        if hasattr(self.control, 'endFreedriveMode'):
            try:
                ended = bool(self.control.endFreedriveMode())
            except Exception:
                ended = False
        if hasattr(self.control, 'endTeachMode'):
            try:
                self.control.endTeachMode()
            except Exception:
                pass
        self._freedrive = False
        return ended

    def freedrive_status(self) -> bool:
        if self.mock:
            return self._freedrive
        try:
            return bool(self.control.getFreedriveStatus())
        except Exception:
            return self._freedrive

    # --------------------------------------------------------------- extras
    def inverse_kinematics(self, pose, seed=None):
        if self._ik_solver is not None:
            return self._ik_solver.solve(pose, seed=seed)
        if self.mock or self.control is None:
            return []
        try:
            if seed is None:
                seed = self.state().q
            joints = self.control.getInverseKinematics(
                list(map(float, pose)), list(map(float, seed)))
            return [float(value) for value in joints] if joints else []
        except Exception:
            return []

    def is_pose_within_safety(self, pose) -> bool:
        if self.mock or self.control is None:
            return True
        try:
            return bool(self.control.isPoseWithinSafetyLimits(list(map(float, pose))))
        except Exception:
            return True
