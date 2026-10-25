"""AgileX Piper / Piper X backend using the official ``pyAgxArm`` Python SDK.

The SDK talks to the arm over CAN:

* Windows: ``interface="agx_cando"``, ``channel="0"`` (requires
  ``python-can-agx-cando``)
* Linux: ``interface="socketcan"``, ``channel="can0"``
* macOS: ``interface="slcan"``, ``channel="/dev/ttyACM0"``

The backend is deliberately thin: it maps the toolkit contract
``[j1..j6 (rad), gripper (m)]`` onto the SDK, keeps the SDK object private, and
provides a mock mode so the rest of the toolkit can be tested without CAN.
Hardware SDKs are imported lazily.
"""
from __future__ import annotations

import importlib.util
import platform
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from ..core.conventions import RobotSpec
from ..core.mathutils import euler_to_matrix, matrix_to_euler, matrix_to_rotvec, rotvec_to_matrix
from ..errors import (ConfigurationError, ConnectionFailed, NotSupported,
                      RobotUnavailable)
from .base import Capabilities, RobotBackend, RobotState
from .mock import MockRobot

# Piper arm status codes that mean the arm is not safely controllable.
_BAD_ARM_STATUS = {0x01, 0x02, 0x03, 0x04, 0x05, 0x06,
                   0x07, 0x08, 0x09, 0x0A, 0x0E, 0x0F}

_FIRMWARE_KEYS = {'default': 'DEFAULT', 'v183': 'V183', 'v188': 'V188', 'v189': 'V189'}
_MODEL_KEYS = {'piper': 'PIPER', 'piper_h': 'PIPER_H', 'piper_l': 'PIPER_L',
               'piper_x': 'PIPER_X'}


def _default_interface() -> str:
    system = platform.system()
    if system == 'Windows':
        return 'agx_cando'
    if system == 'Darwin':
        return 'slcan'
    if system == 'Linux':
        return 'socketcan'
    return 'socketcan'


def _default_channel() -> str:
    system = platform.system()
    if system == 'Windows':
        return '0'
    if system == 'Darwin':
        return '/dev/ttyACM0'
    return 'can0'


@dataclass
class AgxPiperConfig:
    """Connection and safety options for one Piper family arm."""

    model: str = 'piper_x'
    interface: str = 'auto'
    channel: str = 'auto'
    bitrate: int = 1_000_000
    firmware: str = 'auto'
    speed_percent: int = 100
    enable_on_connect: bool = True
    # ``False`` keeps the connection read-only: no speed/motion-mode/TCP-offset
    # frames are sent, which matters on a shared master-slave bus where a write
    # reaches both arms and can drop them out of teach mode.
    configure_on_connect: bool = True
    has_gripper: bool = True
    # Host-side IK. 'auto' attaches PyRoki when it is installed and the
    # bundled Piper URDF exists; 'none' disables it; 'pyroki' requires it.
    ik_backend: str = 'auto'
    ik_urdf: str = ''
    ik_target_link: str = 'link6'
    ik_joint_names: list = field(default_factory=lambda: [
        'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6'])
    ik_self_collision: bool = False
    # Compile the host IK once at connect instead of on the first move_p.
    ik_warmup: bool = True
    gripper_force_n: float = 1.0
    gripper_max_width: float = 0.08
    home_joints: list = field(default_factory=lambda: [0.0, 1.1, -1.1, 0.0, 0.7, 0.0])
    # Piper software limits (rad). The SDK also clamps; this is the toolkit-side
    # default used by SafeRobot. Verify with get_joint_angle_vel_limits().
    limits_lower: list = field(default_factory=lambda: [-2.617994, 0.0, -2.967060,
                                                         -1.553344, -1.553344, -2.967060])
    limits_upper: list = field(default_factory=lambda: [2.617994, 3.141593, 0.0,
                                                         1.553344, 1.553344, 2.967060])
    max_joint_speed: float = 1.0
    command_hz: float = 100.0
    connect_timeout: float = 8.0
    servo_timeout: float = 2.0
    tcp_offset: list = field(default_factory=lambda: [0.0] * 6)
    mock: bool = False


class AgxPiperRobot(RobotBackend):
    """RobotBackend implementation for AgileX Piper/Piper X."""

    def __init__(self, config: AgxPiperConfig | None = None,
                 spec: RobotSpec | None = None, mock: bool | None = None):
        spec = spec or RobotSpec.piper_x()
        self.config = config or AgxPiperConfig()
        self.mock = bool(self.config.mock if mock is None else mock)
        super().__init__(spec, Capabilities(servo_j=True,
                                            servo_l=False, move_j=True,
                                            move_l=True, move_p=True, freedrive=True,
                                            gripper=bool(self.config.has_gripper),
                                            protective_stop_report=True))
        self._sdk = None
        self._robot = None
        self._effector = None
        self._firmware_info = {}
        self._firmware_profile = ''
        self._sdk_config = {}
        self._enabled = False
        self._freedrive = False
        self._motion_mode = ''
        self._auto_motion_mode = True
        self._last_q = None
        self._last_stamp = None
        self._last_servo_at = 0.0
        self._last_gripper_width = 0.0
        self._last_gripper_force = 0.0
        self._qd = np.zeros(spec.dof)
        self._t0 = time.monotonic()
        self._mock_robot = (MockRobot(spec, home=self.config.home_joints,
                                      max_velocity=self.config.max_joint_speed)
                            if self.mock else None)
        if str(self.config.ik_backend).lower() != 'none':
            self._attach_default_ik()

    def _attach_default_ik(self):
        """Attach the bundled PyRoki IK unless it is unavailable."""
        backend = str(self.config.ik_backend or 'auto').lower()
        if backend not in ('auto', 'pyroki', 'none'):
            raise ConfigurationError(
                f"ik_backend must be 'auto', 'pyroki' or 'none', got {backend!r}")
        if backend == 'none':
            return
        try:
            available = importlib.util.find_spec('pyroki') is not None
        except Exception:
            available = False
        if not available:
            if backend == 'pyroki':
                raise ConfigurationError(
                    'ik_backend="pyroki" but PyRoki is not installed; '
                    'install it with `pip install -e .[ik]`')
            return
        urdf = self.config.ik_urdf or str(
            Path(__file__).resolve().parent.parent
            / 'assets' / 'piperx_assets' / 'assets' / 'piper_x.urdf')
        if not Path(urdf).exists():
            if backend == 'pyroki' or self.config.ik_urdf:
                raise ConfigurationError(f'IK URDF not found: {urdf}')
            return
        from .ik import PyrokiIK
        self.set_inverse_kinematics(PyrokiIK(
            urdf,
            target_link_name=self.config.ik_target_link,
            joint_names=self.config.ik_joint_names,
            self_collision=bool(self.config.ik_self_collision),
        ))

    def _warmup_ik(self):
        """Compile the optional host IK at connect time, not on first move_p."""
        solver = self._ik_solver
        if solver is None or not hasattr(solver, 'warmup'):
            return
        try:
            sample = self.state()
            if sample.tcp is None:
                return
            solver.warmup(sample.tcp, seed=sample.q)
        except Exception:
            # Warmup is an optimization; never fail a good hardware connection.
            pass

    # ------------------------------------------------------------------ sdk
    def _import_sdk(self):
        if self._sdk is not None:
            return self._sdk
        try:
            import pyAgxArm
        except ImportError as exc:
            raise ConnectionFailed(
                '未安装 pyAgxArm。请执行：\n'
                '  pip install python-can\n'
                '  pip install "git+https://github.com/agilexrobotics/pyAgxArm.git"\n'
                'Windows 还需要 python-can-agx-cando 插件。') from exc
        self._sdk = pyAgxArm
        return pyAgxArm

    def _model(self, sdk):
        key = _MODEL_KEYS.get(str(self.config.model).lower())
        return getattr(sdk.ArmModel, key, sdk.ArmModel.PIPER_X) if key else sdk.ArmModel.PIPER_X

    def _firmware(self, sdk, name: str | None = None):
        key = _FIRMWARE_KEYS.get(str(name or self.config.firmware).lower(), 'DEFAULT')
        return getattr(sdk.PiperFW, key, sdk.PiperFW.DEFAULT)

    def _interface(self) -> str:
        value = str(self.config.interface or '').strip()
        return _default_interface() if value.lower() in ('', 'auto') else value

    def _channel(self) -> str:
        value = str(self.config.channel or '').strip()
        return _default_channel() if value.lower() in ('', 'auto') else value

    def _open_sdk(self, firmware: str):
        sdk = self._import_sdk()
        cfg = sdk.create_agx_arm_config(
            robot=self._model(sdk),
            firmeware_version=self._firmware(sdk, firmware),
            interface=self._interface(),
            channel=self._channel(),
            bitrate=int(self.config.bitrate),
        )
        self._sdk_config = cfg
        robot = sdk.AgxArmFactory.create_arm(cfg)
        effector = None
        if self.config.has_gripper:
            effector = robot.init_effector(robot.OPTIONS.EFFECTOR.AGX_GRIPPER)
        return robot, effector

    @staticmethod
    def _msg_value(message):
        return None if message is None else getattr(message, 'msg', None)

    @staticmethod
    def _msg_stamp(message, fallback: float) -> float:
        value = getattr(message, 'timestamp', None) if message is not None else None
        try:
            value = float(value)
            return value if value > 0 else fallback
        except (TypeError, ValueError):
            return fallback

    def _wait_for_sample(self, timeout: float):
        deadline = time.monotonic() + max(0.5, float(timeout))
        while time.monotonic() < deadline:
            if not self.mock and self._robot is not None:
                try:
                    if not bool(self._robot.is_ok()):
                        time.sleep(0.05)
                        continue
                except Exception:
                    pass
            sample = self.state()
            if sample.q is not None:
                return sample
            time.sleep(0.05)
        raise ConnectionFailed(f'Piper {self.config.model} 未收到关节反馈（{timeout:.1f}s）')

    def _read_firmware(self, timeout: float = 2.0) -> dict:
        deadline = time.monotonic() + max(0.2, float(timeout))
        while time.monotonic() < deadline:
            try:
                info = self._robot.get_firmware(timeout=0.5)
            except TypeError:
                info = self._robot.get_firmware()
            except Exception:
                info = None
            if isinstance(info, dict) and info:
                return info
            time.sleep(0.1)
        return {}

    # ------------------------------------------------------------------ link
    def connect_read_only(self):
        if self.connected:
            raise RobotUnavailable('disconnect before opening a read-only connection')
        original = self.config
        self.config = replace(original, enable_on_connect=False, configure_on_connect=False)
        try:
            return self.connect()
        finally:
            self.config = original

    def prepare_motion(self):
        super().prepare_motion()
        return self.enable()

    def connect(self):
        if self.mock:
            self._mock_robot.connect()
            self._connected = True
            self._enabled = True
            return self
        robot = self._open_sdk(self.config.firmware)
        self._robot, self._effector = robot
        try:
            self._robot.connect()
            self._wait_for_sample(self.config.connect_timeout)

            # Firmware auto-detection: reconnect once with the profile returned
            # by the controller (Piper firmware selects the driver variant).
            # The query itself transmits (ArmMsgReqFirmware), so a read-only
            # connection must skip it.
            if str(self.config.firmware).lower() == 'auto' and self.config.configure_on_connect:
                try:
                    info = self._read_firmware()
                    version = str(info.get('software_version', '') or '')
                    if version:
                        profile = self._sdk.resolve_firmware_profile(self._model(self._sdk), version)
                        self._firmware_profile = str(profile or 'default')
                        if profile and str(profile).lower() not in ('default', ''):
                            self._robot.disconnect()
                            self._robot = self._effector = None
                            self._robot, self._effector = self._open_sdk(profile)
                            self._robot.connect()
                            self._wait_for_sample(self.config.connect_timeout)
                        self._firmware_info = info
                    else:
                        self._firmware_info = {'warning': 'firmware query returned no data'}
                except Exception as exc:
                    self._firmware_info = {'warning': f'firmware auto-detect failed: {exc}'}

            try:
                self._robot.set_joint_limits_enabled(True)
            except Exception:
                pass
            # Everything below transmits CAN frames (0x151 mode control etc.).
            # On a master-slave pair both arms share one bus, so a diagnostic
            # connection must not write: it would pull both arms out of
            # leader/follower teaching and make them hold position.
            if self.config.configure_on_connect:
                if any(abs(float(value)) > 1e-9 for value in self.config.tcp_offset):
                    try:
                        self._robot.set_tcp_offset([float(value) for value in self.config.tcp_offset])
                    except Exception:
                        pass
                if self.config.enable_on_connect:
                    self.enable()
                # Apply the speed after enable: the controller can reset the
                # speed percentage during the enable transition.  Do it before
                # the motion-mode frame so that frame carries the new value.
                try:
                    self._robot.set_speed_percent(int(np.clip(self.config.speed_percent, 0, 100)))
                except Exception:
                    pass
                # ``move_j`` otherwise sends the motion-mode command on every
                # 50 Hz replay frame, which adds latency and makes Piper playback
                # visibly stutter.  Disable auto switching and set J mode once.
                try:
                    self._robot.set_auto_set_motion_mode_enabled(False)
                    self._robot.set_motion_mode('j')
                    self._auto_motion_mode = False
                    self._motion_mode = 'j'
                except Exception:
                    self._auto_motion_mode = True
                    self._motion_mode = ''
                    try:
                        self._robot.set_auto_set_motion_mode_enabled(True)
                    except Exception:
                        pass
            self._connected = True
            if self.config.configure_on_connect and self.config.ik_warmup:
                self._warmup_ik()
            return self
        except Exception:
            self._close_sdk()
            raise

    def _close_sdk(self):
        robot, self._robot = self._robot, None
        self._effector = None
        if robot is not None:
            try:
                robot.disconnect()
            except Exception:
                pass
        self._connected = False
        self._enabled = False
        self._freedrive = False
        self._motion_mode = ''
        self._auto_motion_mode = True

    def disconnect(self):
        if self.mock:
            self._mock_robot.disconnect()
            self._connected = False
            self._enabled = False
            return
        # Leave the arm enabled/holding on purpose; disabling would drop a
        # raised arm. Use emergency_stop()/disable() explicitly when needed.
        self._close_sdk()

    # ----------------------------------------------------------------- state
    def forward_kinematics(self, joints):
        if self.mock or self._robot is None:
            raise NotSupported('Piper FK requires a connected real SDK model')
        q = np.asarray(joints, float)
        if q.shape != (self.dof,) or not np.isfinite(q).all():
            raise ValueError('invalid joint vector for FK')
        flange = np.asarray(self._robot.fk(q.tolist()), float).reshape(6)
        rotation = euler_to_matrix(flange[3:], order='xyz')
        offset = np.asarray(self.config.tcp_offset, float)
        position = flange[:3] + rotation @ offset[:3]
        rotation = rotation @ euler_to_matrix(offset[3:], order='xyz')
        return np.r_[position, matrix_to_rotvec(rotation)]

    def _read_gripper(self):
        if self._effector is None:
            return self._last_gripper_width, self._last_gripper_force
        try:
            status = self._effector.get_gripper_status()
            msg = self._msg_value(status)
            if msg is not None:
                mode = str(getattr(msg, 'mode', 'width')).lower()
                value = float(getattr(msg, 'value', 0.0) or 0.0)
                # The SDK reports metres in width mode and degrees in angle mode.
                width = value if mode == 'width' else self._last_gripper_width
                self._last_gripper_width = float(np.clip(width, 0.0, self.config.gripper_max_width))
                self._last_gripper_force = float(getattr(msg, 'force', 0.0) or 0.0)
        except Exception:
            pass
        return self._last_gripper_width, self._last_gripper_force

    def state(self) -> RobotState:
        if self.mock:
            sample = self._mock_robot.state()
            sample.t = time.monotonic() - self._t0
            sample.extra['feedback'] = 'leader' if self._freedrive else 'follower'
            return sample
        if self._robot is None:
            raise RobotUnavailable('Piper SDK is not connected')
        now = time.monotonic() - self._t0
        # In leader/freedrive mode the arm reports the dragged (leader) joints
        # separately; ``get_joint_angles`` can stay at the follower's last
        # value.  Prefer the leader feedback while dragging, then fall back.
        message = None
        used_leader = False
        if self._freedrive:
            try:
                message = self._robot.get_leader_joint_angles()
                used_leader = self._msg_value(message) is not None
            except Exception:
                message = None
        if self._msg_value(message) is None:
            message = self._robot.get_joint_angles()
        q = self._msg_value(message)
        stamp = self._msg_stamp(message, now)
        if q is None:
            return RobotState(t=stamp, q=None, qd=None, tcp=None, connected=self._connected)
        q = np.asarray(q, dtype=float).reshape(-1)
        if q.shape != (self.spec.dof,):
            raise RobotUnavailable(f'Piper 返回了 {q.shape[0]} 个关节，期望 {self.spec.dof}')

        if self._last_q is not None and self._last_stamp is not None:
            dt = stamp - self._last_stamp
            if 0.0 < dt < 0.5:
                self._qd = np.clip((q - self._last_q) / dt, -20.0, 20.0)
            elif dt >= 0.5:
                self._qd = np.zeros_like(q)
        else:
            self._qd = np.zeros_like(q)
        self._last_q = q.copy()
        self._last_stamp = stamp

        tcp = None
        tcp_rpy = None
        try:
            if self._freedrive:
                # Leader mode has no leader-pose feedback; compute the flange
                # pose from the dragged joints instead of using the static
                # follower pose.
                pose = self._robot.fk([float(value) for value in q])
            else:
                pose_msg = (self._robot.get_tcp_pose() if any(
                    abs(float(value)) > 1e-9 for value in self.config.tcp_offset)
                    else self._robot.get_flange_pose())
                pose = self._msg_value(pose_msg)
            if pose is not None:
                pose = np.asarray(pose, dtype=float).reshape(6)
                tcp_rpy = pose[3:].copy()
                # The toolkit contract is x,y,z + rotation vector; the Piper
                # SDK reports x,y,z + roll,pitch,yaw.  Normalize at the backend
                # boundary so teleop/teaching code never needs a robot branch.
                tcp = np.concatenate([pose[:3],
                                      matrix_to_rotvec(euler_to_matrix(tcp_rpy, order='xyz'))])
        except Exception:
            tcp = None
            tcp_rpy = None

        arm_status = None
        motion_status = None
        err_code = None
        protective = False
        try:
            status = self._msg_value(self._robot.get_arm_status())
            if status is not None:
                arm_status = int(getattr(status, 'arm_status', 0) or 0)
                motion_status = getattr(status, 'motion_status', None)
                err_code = getattr(status, 'err_code', None)
                protective = arm_status in _BAD_ARM_STATUS
        except Exception:
            pass

        gripper_width, gripper_force = self._read_gripper()
        extra = {
            'arm_status': arm_status,
            'motion_status': motion_status,
            'err_code': err_code,
            'firmware': self._firmware_info,
            'fps': self._safe_call('get_fps'),
            'gripper_width': gripper_width,
            'gripper_force': gripper_force,
            'enabled': self._enabled,
            'feedback': 'leader' if used_leader else 'follower',
            'tcp_rpy': None if tcp_rpy is None else tcp_rpy.tolist(),
        }
        return RobotState(t=stamp, q=q, qd=self._qd.copy(), tcp=tcp,
                          protective_stop=protective, connected=self._connected,
                          extra=extra)

    def _safe_call(self, name, default=None):
        try:
            return getattr(self._robot, name)()
        except Exception:
            return default

    def leader_joint_angles(self):
        """Read the leader arm joints already broadcast on the bus (read-only).

        In master-slave teach mode the leader streams its joints on 0x155-0x157.
        This returns them without calling ``freedrive()``, which would send a
        0x470 broadcast that every arm on the shared bus receives.  Returns
        ``None`` when no leader is broadcasting.
        """
        if self.mock:
            return None
        if self._robot is None:
            raise RobotUnavailable('Piper SDK is not connected')
        try:
            values = self._msg_value(self._robot.get_leader_joint_angles())
        except Exception:
            return None
        if values is None:
            return None
        q = np.asarray(values, dtype=float).reshape(-1)
        return q if q.shape == (self.spec.dof,) else None

    # ---------------------------------------------------------------- motion
    def enable(self, timeout: float | None = None) -> bool:
        if self.mock:
            self._enabled = True
            self._freedrive = False
            return True
        if self._robot is None:
            raise RobotUnavailable('Piper SDK is not connected')
        deadline = time.monotonic() + float(timeout if timeout is not None else 5.0)
        while time.monotonic() < deadline:
            try:
                if bool(self._robot.enable()):
                    self._enabled = True
                    self._freedrive = False
                    return True
            except Exception:
                pass
            time.sleep(0.05)
        raise ConnectionFailed('Piper 使能失败（检查急停、供电和 CAN 通信）')

    def disable(self, timeout: float | None = None) -> bool:
        """Disable all motors; retry because the status feedback lags the command.

        The SDK returns the *last received* enable state right after sending the
        disable frame.  On a real arm that is usually still ``True`` for one or
        more frames, so a single call often reports failure even though the
        command eventually takes effect.  Retry within a short deadline.
        """
        if self.mock:
            self._enabled = False
            return True
        if self._robot is None:
            return False
        deadline = time.monotonic() + float(timeout if timeout is not None else 3.0)
        ok = False
        while time.monotonic() < deadline:
            try:
                if bool(self._robot.disable()):
                    ok = True
                    break
            except Exception:
                pass
            time.sleep(0.05)
        if ok:
            self._enabled = False
            self._freedrive = False
        return ok

    def set_speed_percent(self, percent: int) -> None:
        self.config.speed_percent = int(np.clip(percent, 0, 100))
        if not self.mock and self._robot is not None:
            self._robot.set_speed_percent(self.config.speed_percent)
            # The SDK implements this by sending a temporary move_mode=255
            # mode frame and only restoring its local field. Force the next
            # motion command to re-send the real mode, otherwise changing
            # speed after connect can leave the controller ignoring moves.
            self._motion_mode = ''

    def _set_native_speed_from_rad_s(self, speed) -> None:
        """Map a generic rad/s speed onto the Piper speed percentage.

        Native Piper moves otherwise ignore ``move_j(speed=...)``.  Mapping
        keeps one speed knob for both native and generic modes; percent is
        clipped to 1..100 and ``accel`` remains host-only.
        """
        if self.mock or speed is None:
            return
        speed = float(speed)
        if speed <= 0:
            raise ValueError('speed must be positive')
        maximum = max(float(self.config.max_joint_speed), 1e-6)
        percent = int(np.clip(round(speed / maximum * 100.0), 1, 100))
        if percent != self.config.speed_percent:
            self.set_speed_percent(percent)

    def _ensure_motion_mode(self, mode: str) -> None:
        """Set the Piper motion mode once, not on every servo frame."""
        if self.mock or self._robot is None or self._auto_motion_mode:
            return
        if self._motion_mode == mode:
            return
        try:
            self._robot.set_motion_mode(mode)
            self._motion_mode = mode
        except Exception:
            # Fall back to the SDK's per-command auto switching.
            self._auto_motion_mode = True
            self._motion_mode = ''
            try:
                self._robot.set_auto_set_motion_mode_enabled(True)
            except Exception:
                pass

    def servo_j(self, joints, dt: float) -> bool:
        if self.mock:
            return self._mock_robot.servo_j(joints, dt)
        if self._robot is None or not self._connected:
            raise RobotUnavailable('Piper SDK is not connected')
        if not self._enabled or self._freedrive:
            raise RobotUnavailable('Piper 未处于受控使能状态；先退出 freedrive 并 enable()')
        target = np.asarray(joints, dtype=float).reshape(-1)
        if target.shape != (self.spec.dof,):
            raise ValueError(f'expected {self.spec.dof} joints, got {target.shape}')
        self._ensure_motion_mode('j')
        # The SDK's move_j overwrites the latest target and smooths internally.
        # Limit the command rate; a 125 Hz caller does not need 125 CAN frames/s.
        now = time.monotonic()
        if self.config.command_hz > 0 and now - self._last_servo_at < 1.0 / self.config.command_hz:
            return True
        self._last_servo_at = now
        try:
            self._robot.move_j([float(value) for value in target])
        except Exception as exc:
            raise RobotUnavailable(f'Piper move_j failed: {exc}') from exc
        return True

    def move_j(self, joints, speed=None, accel=None, wait=True, rate_hz=100.0,
               timeout=None, abort_event=None, use_native: bool = True) -> bool:
        if self.mock or not use_native:
            return RobotBackend.move_j(self, joints, speed=speed, accel=accel, wait=wait,
                                       rate_hz=rate_hz, timeout=timeout,
                                       abort_event=abort_event)
        self._check_motion_ready()
        self._set_native_speed_from_rad_s(speed)
        self._ensure_motion_mode('j')
        target = np.asarray(joints, dtype=float).reshape(-1)
        try:
            self._robot.move_j([float(value) for value in target])
        except Exception as exc:
            raise RobotUnavailable(f'Piper move_j failed: {exc}') from exc
        if not wait:
            return True
        return self.wait_until_settled(target, timeout=timeout or 60.0,
                                       rate_hz=rate_hz, tolerance=0.02,
                                       abort_event=abort_event)

    def _check_motion_ready(self):
        if self._robot is None or not self._connected:
            raise RobotUnavailable('Piper SDK is not connected')
        if not self._enabled or self._freedrive:
            raise RobotUnavailable('Piper 未处于受控使能状态；先退出 freedrive 并 enable()')

    def _cartesian_target(self, pose):
        """Convert toolkit TCP rotvec pose to the SDK's flange RPY pose."""
        target = np.asarray(pose, dtype=float).reshape(6)
        sdk_pose = np.concatenate([
            target[:3],
            matrix_to_euler(rotvec_to_matrix(target[3:]), order='xyz'),
        ])
        if any(abs(float(value)) > 1e-9 for value in self.config.tcp_offset):
            sdk_pose = np.asarray(
                self._robot.get_tcp2flange_pose([float(value) for value in sdk_pose]),
                dtype=float).reshape(6)
        return sdk_pose

    def _wait_cartesian(self, timeout: float) -> bool:
        # Piper's TCP feedback is roll/pitch/yaw, while the toolkit's generic
        # wait_pose expects a rotvec.  Use the controller's motion_status.
        deadline = time.monotonic() + max(0.5, float(timeout))
        time.sleep(0.2)
        while time.monotonic() < deadline:
            try:
                status = self._msg_value(self._robot.get_arm_status())
            except Exception:
                status = None
            if status is not None:
                if (getattr(status, 'motion_status', None) == 0
                        and getattr(status, 'arm_status', 0) == 0):
                    return True
            time.sleep(0.05)
        return False

    def move_l(self, pose, speed=None, accel=None, wait=True, rate_hz=100.0,
               timeout=None, abort_event=None) -> bool:
        if self.mock:
            raise NotSupported('Piper mock has no Cartesian model')
        self._check_motion_ready()
        self._set_native_speed_from_rad_s(speed)
        self._ensure_motion_mode('l')
        target = self._cartesian_target(pose)
        self._robot.move_l([float(value) for value in target])
        if not wait:
            return True
        return self._wait_cartesian(timeout or 60.0)

    def move_p(self, pose, wait=True, timeout=None, use_native: bool = True,
               speed=None, accel=None) -> bool:
        """Piper point-to-point Cartesian move (toolkit TCP rotvec pose)."""
        if self.mock or not use_native:
            return RobotBackend.move_p(self, pose, wait=wait, timeout=timeout,
                                       use_native=False, speed=speed, accel=accel)
        self._check_motion_ready()
        self._set_native_speed_from_rad_s(speed)
        self._ensure_motion_mode('p')
        target = self._cartesian_target(pose)
        self._robot.move_p([float(value) for value in target])
        if not wait:
            return True
        return self._wait_cartesian(timeout or 60.0)

    def stop_motion(self, deceleration: float = 2.0) -> bool:
        if self.mock:
            return True
        if self._robot is None:
            return False
        try:
            state = self.state()
            if state.q is not None:
                self._ensure_motion_mode('j')
                self._robot.move_j([float(value) for value in state.q])
            return True
        except Exception:
            return False

    def emergency_stop(self) -> bool:
        if self.mock:
            self._enabled = False
            return True
        if self._robot is None:
            return False
        try:
            self._robot.electronic_emergency_stop()
            self._enabled = False
            return True
        except Exception:
            return False

    def reset(self) -> bool:
        """Reset the controller. Warning: a raised arm may fall immediately."""
        if self.mock:
            self._enabled = False
            return True
        if self._robot is None:
            return False
        try:
            self._robot.reset()
            self._enabled = False
            return True
        except Exception:
            return False

    def freedrive(self, enabled: bool) -> bool:
        if self.mock:
            self._freedrive = bool(enabled)
            return self._mock_robot.freedrive(enabled)
        if self._robot is None:
            raise RobotUnavailable('Piper SDK is not connected')
        try:
            if enabled:
                self._robot.set_leader_mode()
                self._freedrive = True
            else:
                self._robot.set_follower_mode()
                self._freedrive = False
            # The controller may reset its motion mode after a mode switch.
            self._motion_mode = ''
            return True
        except Exception as exc:
            raise RobotUnavailable(f'Piper freedrive switch failed: {exc}') from exc

    # ---------------------------------------------------------------- gripper
    @property
    def gripper_width(self) -> float:
        if self.mock:
            return float(self._mock_robot.state().extra.get('gripper', 0.0))
        return self._read_gripper()[0]

    def move_gripper(self, width: float, force: float | None = None,
                     wait: bool = False, timeout: float | None = None) -> bool:
        if self.mock:
            self._mock_robot.set_gripper(width)
            self._last_gripper_width = float(width)
            return True
        if self._effector is None:
            raise NotSupported('Piper 没有初始化 AgxGripper（has_gripper=False）')
        target = float(np.clip(width, 0.0, self.config.gripper_max_width))
        force = float(self.config.gripper_force_n if force is None else force)
        self._effector.move_gripper_m(value=target, force=force)
        self._last_gripper_width = target
        self._last_gripper_force = force
        if not wait:
            return True
        deadline = time.monotonic() + float(timeout if timeout is not None else 5.0)
        while time.monotonic() < deadline:
            if abs(self.gripper_width - target) <= 0.003:
                return True
            time.sleep(0.05)
        return False

    def open_gripper(self, wait: bool = False, timeout: float | None = None) -> bool:
        return self.move_gripper(self.config.gripper_max_width, wait=wait, timeout=timeout)

    def close_gripper(self, wait: bool = False, timeout: float | None = None) -> bool:
        return self.move_gripper(0.0, wait=wait, timeout=timeout)

    # ---------------------------------------------------------------- extras
    def describe(self) -> dict:
        state = None
        try:
            state = self.state()
        except Exception:
            pass
        return {
            'backend': 'agx_piper',
            'model': self.config.model,
            'interface': self._interface(),
            'channel': self._channel(),
            'firmware': self._firmware_info,
            'firmware_profile': self._firmware_profile,
            'connected': self.connected,
            'enabled': self._enabled,
            'fps': None if self._robot is None else self._safe_call('get_fps'),
            'joints': None if state is None or state.q is None else state.q.tolist(),
            'tcp': None if state is None or state.tcp is None else state.tcp.tolist(),
            'gripper_width': self.gripper_width,
            'sdk_config': self._sdk_config,
        }

    def __repr__(self) -> str:
        return (f'AgxPiperRobot(model={self.config.model!r}, '
                f'interface={self._interface()!r}, channel={self._channel()!r}, '
                f'mock={self.mock})')


# ---------------------------------------------------------------- bus scanning
# Piper uses fixed arbitration IDs on the shared bus: a leader arm in
# master-slave teach mode broadcasts joint targets on 0x155..0x157, while a
# follower reports motor/driver state on 0x251..0x256 / 0x261..0x266 and arm
# state on 0x2A1..0x2A7.
PIPER_CAN_LABELS = {
    0x155: 'leader_joint_12',
    0x156: 'leader_joint_34',
    0x157: 'leader_joint_56',
    0x181: 'cpv_response_j1',
    0x182: 'cpv_response_j2',
    0x183: 'cpv_response_j3',
    0x184: 'cpv_response_j4',
    0x185: 'cpv_response_j5',
    0x186: 'cpv_response_j6',
    0x251: 'motor_state_j1',
    0x252: 'motor_state_j2',
    0x253: 'motor_state_j3',
    0x254: 'motor_state_j4',
    0x255: 'motor_state_j5',
    0x256: 'motor_state_j6',
    0x261: 'driver_state_j1',
    0x262: 'driver_state_j2',
    0x263: 'driver_state_j3',
    0x264: 'driver_state_j4',
    0x265: 'driver_state_j5',
    0x266: 'driver_state_j6',
    0x2A1: 'arm_status',
    0x2A2: 'end_pose_xy',
    0x2A3: 'end_pose_zrx',
    0x2A4: 'end_pose_ryrz',
    0x2A5: 'joint_state_12',
    0x2A6: 'joint_state_34',
    0x2A7: 'joint_state_56',
    0x2A8: 'gripper',
    0x473: 'motor_angle_limit_max_spd',
    0x476: 'resp_set_instruction',
    0x478: 'end_vel_acc_param',
    0x47B: 'crash_protection_rating',
    0x1C0: 'revo2_hand_status',
    0x1C1: 'revo2_finger_pos',
    0x1C2: 'revo2_finger_spd',
    0x1C3: 'revo2_finger_current',
}

# Single-arm feedback rates reported by the Piper firmware. Two arms wired to
# the same terminal share arbitration IDs, so a second arm shows up as roughly
# doubled rates rather than as extra IDs.
PIPER_NOMINAL_HZ = ((0x251, 0x256, 100.0), (0x261, 0x266, 20.0), (0x2A1, 0x2A7, 200.0))


def _scan_interface(value: str) -> str:
    text = str(value or '').strip()
    return _default_interface() if text.lower() in ('', 'auto') else text


def _scan_channel(value: str) -> str:
    text = str(value or '').strip()
    return _default_channel() if text.lower() in ('', 'auto') else text


def scan_piper_can(interface: str = 'auto', channel: str = 'auto',
                   bitrate: int = 1_000_000, seconds: float = 3.0) -> dict:
    """Passively listen on a Piper CAN bus and report the arbitration IDs seen.

    Nothing is transmitted, so the arm stays untouched: this only counts the
    feedback frames that already exist on the bus. Useful to tell a wiring or
    adapter problem (zero frames) from a healthy bus, and to see whether a
    leader arm is broadcasting (0x155..0x157) next to a follower.
    """
    try:
        import can
    except ImportError as exc:
        raise ConnectionFailed(
            '未安装 python-can。请执行：\n'
            '  pip install python-can\n'
            'Windows 还需要 python-can-agx-cando 插件。') from exc

    iface = _scan_interface(interface)
    chan = _scan_channel(channel)
    adapters = None
    try:
        adapter_list = can.detect_available_configs(interfaces=[iface])
        adapters = len(adapter_list)
    except Exception:
        adapters = None

    try:
        bus = can.Bus(interface=iface, channel=chan, bitrate=int(bitrate),
                      receive_own_messages=False, local_loopback=False)
    except Exception as exc:
        raise ConnectionFailed(
            f'无法打开 CAN 总线（interface={iface!r}, channel={chan!r}, '
            f'bitrate={int(bitrate)}）：{type(exc).__name__}: {exc}') from exc

    budget = max(0.2, float(seconds))
    frames = 0
    stats: dict[int, dict] = {}
    try:
        deadline = time.monotonic() + budget
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            message = bus.recv(timeout=min(0.2, remaining))
            if message is None:
                continue
            frames += 1
            can_id = int(message.arbitration_id)
            record = stats.get(can_id)
            if record is None:
                record = stats[can_id] = {
                    'id': can_id,
                    'label': PIPER_CAN_LABELS.get(can_id, 'unknown'),
                    'count': 0,
                    'first': float(message.timestamp),
                    'last': float(message.timestamp),
                    'dlc': int(message.dlc),
                    'data': bytes(message.data),
                }
            record['count'] += 1
            record['last'] = float(message.timestamp)
    finally:
        bus.shutdown()

    nodes = []
    for can_id in sorted(stats):
        record = stats[can_id]
        span = record['last'] - record['first']
        nodes.append({
            'id': can_id,
            'hex': f'0x{can_id:03X}',
            'label': record['label'],
            'count': record['count'],
            # count-1 intervals over the observed span; a single frame has no rate
            'hz': round((record['count'] - 1) / span, 1) if span > 1e-6 else None,
            'dlc': record['dlc'],
            'data': record['data'].hex(' '),
        })

    leader = any(0x155 <= node['id'] <= 0x157 for node in nodes)
    follower = any(0x2A1 <= node['id'] <= 0x2A7 or 0x251 <= node['id'] <= 0x256
                   or 0x261 <= node['id'] <= 0x266 for node in nodes)
    notes = []
    duplicated = [node['hex'] for node in nodes
                  for low, high, nominal in PIPER_NOMINAL_HZ
                  if low <= node['id'] <= high and node['hz'] and node['hz'] >= 1.6 * nominal]
    if leader and follower:
        arms = 2
        notes.append('同时看到 0x155-0x157（主臂关节广播）和从臂反馈：总线上是主从两台手臂')
    elif follower:
        arms = 1
        notes.append('只看到从臂反馈，没有 0x155-0x157 主臂广播：总线上 1 台手臂')
    elif leader:
        arms = 1
        notes.append('只看到 0x155-0x157 主臂广播，没有从臂反馈：从臂可能未上电或未接好')
    else:
        arms = 0
        notes.append('没有收到任何 Piper 已知反馈 ID：检查适配器、CAN_H/CAN_L 线序和 1 Mbps 波特率')
    if duplicated:
        arms = max(arms, 2)
        notes.insert(0, f'{"、".join(sorted(set(duplicated)))} 的帧率约为单臂标称值的两倍：'
                        '可能有第二台手臂复用了同一组 CAN ID')
    if not nodes:
        notes.append('主从手臂并到同一个接线端子只占用 1 个 CAN 适配器/1 个 channel，'
                     '适配器数量不等于手臂数量；手臂数量只能靠总线上的反馈帧判断')
    if adapters in (0, None):
        notes.append(f'{iface} 没有枚举到 USB-CAN 适配器，先确认适配器已插好并被系统识别')

    return {
        'interface': iface,
        'channel': chan,
        'bitrate': int(bitrate),
        'adapters': adapters,
        'seconds': budget,
        'frames': frames,
        'nodes': nodes,
        'leader_broadcast': leader,
        'follower_feedback': follower,
        'arms_estimate': arms,
        'notes': notes,
    }


PIPER_LINKAGE_IDS = {'leader': 0xFA, 'follower': 0xFC}


def set_piper_linkage(role: str, interface: str = 'auto', channel: str = 'auto',
                      bitrate: int = 1_000_000) -> dict:
    """Send a single 0x470 frame to (re)enter master-slave teach mode.

    ``leader`` (0xFA) makes the arm a zero-force drag input arm, ``follower``
    (0xFC) makes it mirror the leader. The frame is broadcast, so when a
    master-slave pair shares one terminal both arms receive it: power only the
    arm you want to configure while recovering a pair.

    Only 0x470 is transmitted here; no mode-control (0x151) frame is sent, so
    the arm is not pulled into CAN command mode the way ``connect()`` does.
    """
    try:
        import can
    except ImportError as exc:
        raise ConnectionFailed(
            '未安装 python-can。请执行：\n'
            '  pip install python-can\n'
            'Windows 还需要 python-can-agx-cando 插件。') from exc

    key = str(role or '').strip().lower()
    if key not in PIPER_LINKAGE_IDS:
        raise ValueError(f"role 只能是 'leader' 或 'follower'，收到 {role!r}")
    iface = _scan_interface(interface)
    chan = _scan_channel(channel)
    payload = bytes([PIPER_LINKAGE_IDS[key], 0, 0, 0, 0, 0, 0, 0])

    try:
        bus = can.Bus(interface=iface, channel=chan, bitrate=int(bitrate),
                      receive_own_messages=False, local_loopback=False)
    except Exception as exc:
        raise ConnectionFailed(
            f'无法打开 CAN 总线（interface={iface!r}, channel={chan!r}, '
            f'bitrate={int(bitrate)}）：{type(exc).__name__}: {exc}') from exc

    try:
        bus.send(can.Message(arbitration_id=0x470, is_extended_id=False,
                             data=payload, dlc=len(payload)))
        time.sleep(0.2)
    finally:
        bus.shutdown()

    return {
        'interface': iface,
        'channel': chan,
        'bitrate': int(bitrate),
        'role': key,
        'can_id': '0x470',
        'data': payload.hex(' '),
        'note': '该帧是广播：主从并线的总线上两台手臂都会收到，恢复时请只给目标手臂上电',
    }
