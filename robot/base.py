"""The single robot interface every backend (simulator or real) implements."""

from __future__ import annotations

import abc
import math
import time
from dataclasses import asdict, dataclass, field, fields, is_dataclass, replace

import numpy as np

from ..core.conventions import RobotSpec
from ..core.mathutils import pose_error
from ..errors import Aborted, NotSupported


def _json_value(value):
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if is_dataclass(value) and not isinstance(value, type):
        return _json_value({item.name: getattr(value, item.name) for item in fields(value)})
    return str(value)


@dataclass(frozen=True)
class Capabilities:
    """Advertised features; callers branch on these instead of on robot names.

    ``nonblocking_moves`` names native methods whose ``wait=False`` submits
    without host-side trajectory streaming. It never covers use_native=False.
    """

    servo_j: bool = True
    servo_l: bool = False
    move_j: bool = True
    move_l: bool = False
    move_p: bool = False
    inverse_kinematics: bool = False
    freedrive: bool = False
    gripper: bool = False
    protective_stop_report: bool = True
    servo_j_kind: str = 'unspecified'
    servo_j_timing: str = 'unspecified'
    stop_motion_kind: str = 'unspecified'
    nonblocking_moves: tuple[str, ...] = ()


@dataclass
class RobotState:
    """One synchronized sample of robot state.

    ``q`` and ``qd`` are in radians; ``tcp`` is ``[x, y, z, rx, ry, rz]`` in the
    frame given by ``tcp_frame``. ``t`` is in seconds on ``timestamp_clock``; a backend
    clock is not necessarily comparable with the host monotonic clock.
    """

    t: float = 0.0
    q: np.ndarray | None = None
    qd: np.ndarray | None = None
    tcp: np.ndarray | None = None
    protective_stop: bool = False
    connected: bool = True
    extra: dict = field(default_factory=dict)
    timestamp_clock: str = 'backend'
    enabled: bool | None = None
    fault: bool | None = None
    protective_stop_known: bool = False
    sources: dict = field(default_factory=dict)
    tcp_frame: str = 'base'

    def __post_init__(self):
        if self.q is not None:
            self.q = np.asarray(self.q, dtype=float)
        if self.qd is not None:
            self.qd = np.asarray(self.qd, dtype=float)
        if self.tcp is not None:
            self.tcp = np.asarray(self.tcp, dtype=float)

    def copy(self) -> 'RobotState':
        return RobotState(
            t=self.t,
            q=None if self.q is None else self.q.copy(),
            qd=None if self.qd is None else self.qd.copy(),
            tcp=None if self.tcp is None else self.tcp.copy(),
            protective_stop=self.protective_stop,
            connected=self.connected,
            extra=dict(self.extra),
            timestamp_clock=self.timestamp_clock,
            enabled=self.enabled,
            fault=self.fault,
            protective_stop_known=self.protective_stop_known,
            sources=dict(self.sources),
            tcp_frame=self.tcp_frame,
        )

    def to_dict(self) -> dict:
        """JSON-safe feedback, not commanded targets; missing values stay null.

        Non-finite numbers become null; opaque vendor extras become strings.
        This does not claim that separately received sensors are synchronized.
        """
        payload = {item.name: getattr(self, item.name) for item in fields(self)}
        if not self.protective_stop_known and not self.protective_stop:
            payload['protective_stop'] = None
        return _json_value(payload)

    def state_vector(self, gripper_width=None, spec: RobotSpec | None = None) -> np.ndarray:
        """Contract vector ``[joints..., gripper]`` for datasets and policies."""
        if self.q is None:
            raise ValueError('state has no joint positions')
        if spec is None:
            values = list(self.q)
            if gripper_width is not None:
                values.append(float(gripper_width))
            return np.asarray(values, dtype=float)
        return np.asarray(spec.pack(self.q, gripper_width), dtype=float)


class RobotBackend(abc.ABC):
    """Minimal, uniform robot API.

    ``servo_j`` is the primitive: a single joint target for one control tick.
    ``move_j`` has a generic implementation on top of it, so a backend only
    needs to provide ``state`` and ``servo_j`` to be fully usable.  Real
    controllers override ``move_j`` with their native planner for smoothness.
    """

    def __init__(self, spec: RobotSpec, capabilities: Capabilities | None = None):
        self.spec = spec
        self.capabilities = capabilities or Capabilities()
        self._connected = False
        self._ik_solver = None
        self._motion_detail = None

    # ------------------------------------------------------------------ link
    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def dof(self) -> int:
        return self.spec.dof

    @property
    def connected(self) -> bool:
        return self._connected

    def connect(self) -> 'RobotBackend':
        self._connected = True
        return self

    def disconnect(self) -> None:
        self._connected = False

    def connect_read_only(self) -> 'RobotBackend':
        """Connect without enabling/configuring motion; unsupported fails closed."""
        raise NotSupported(f'{self.name}: read-only connection is unavailable')

    def prepare_motion(self) -> bool:
        """Prepare an already connected controller without commanding a pose.

        Backends requiring explicit motor enable override this hook. Call only
        after task preflight; never use for passive observation or planning.
        """
        if not self.connected:
            raise Aborted('connect before preparing motion')
        return True

    def __enter__(self):
        return self.connect()

    def __exit__(self, exc_type, exc, traceback):
        self.disconnect()
        return False

    # ----------------------------------------------------------------- state
    @abc.abstractmethod
    def state(self) -> RobotState:
        """Return a fresh sample. Callers must not cache the returned object."""

    def wait_for_state(self, timeout: float = 5.0, rate_hz: float = 50.0) -> RobotState:
        deadline = time.monotonic() + timeout
        while True:
            sample = self.state()
            if sample.q is not None:
                return sample
            if time.monotonic() > deadline:
                raise TimeoutError('robot did not report joint positions')
            time.sleep(1.0 / rate_hz)

    def protective_stop(self) -> bool:
        try:
            return bool(self.state().protective_stop)
        except Exception:
            return False

    def leader_joint_angles(self):
        """Joints of an external leader input device, or ``None`` when absent.

        Passive sampling records the robot as ``observation`` and, when a leader
        device streams its joints, those as ``action``.  Backends without such
        an input device inherit this no-op instead of the caller sniffing for a
        robot-specific attribute.
        """
        return None

    def joints_within_safety(self, joints) -> bool:
        """Best-effort local check; hardware backends may query the controller.

        A backend that cannot answer must not silently turn an unknown state
        into ``True``; it should raise :class:`RobotUnavailable` instead.
        """
        values = np.asarray(joints, dtype=float)
        return bool(np.all(np.isfinite(values)))

    def telemetry(self) -> dict:
        """Optional controller diagnostics for logs and tests."""
        return {}

    def describe(self) -> dict:
        """Describe the public contract without connecting or enabling motion."""
        return {
            'schema_version': 1,
            'backend': type(self).__name__,
            'spec': self.spec.to_dict(),
            'capabilities': asdict(self.capabilities),
            'units': {'q': 'rad', 'qd': 'rad/s', 'tcp_position': 'm',
                      'tcp_rotation': 'rotation_vector_rad', 'time': 's'},
            'motion_targets': 'absolute',
            'tcp_frame': 'base',
            'motion_authorization': 'external',
        }

    def status(self) -> dict:
        """Read-only feedback with a host acquisition window, not freshness proof.

        Never connects, enables, retries, or sends a motion command. Unknown
        health values stay null; connected does not imply permission to move.
        """
        started = time.monotonic()
        state = None
        telemetry = {}
        errors = {}
        if self.connected:
            try:
                state = self.state().to_dict()
            except Exception as exc:
                errors['state'] = {'type': type(exc).__name__, 'message': str(exc)}
            try:
                telemetry = _json_value(self.telemetry())
            except Exception as exc:
                errors['telemetry'] = {'type': type(exc).__name__, 'message': str(exc)}
        return {
            'schema_version': 1, 'connected': bool(self.connected),
            'state': state, 'telemetry': telemetry, 'errors': errors,
            'acquisition': {'clock': 'host_monotonic', 'started_at': started,
                            'finished_at': time.monotonic(),
                            'synchronization': 'not_guaranteed',
                            'feedback_freshness': 'unknown'},
        }

    def forward_kinematics(self, joints):
        """Return base-to-TCP [metres, rotation vector] without commanding motion."""
        raise NotSupported(f'{self.name}: forward kinematics is unavailable')

    def inverse_kinematics(self, pose, seed=None):
        """Return joint angles for a TCP pose, or raise ``NotSupported``.

        ``pose`` uses the toolkit contract ``[x, y, z, rx, ry, rz]`` (metres
        and a rotation vector).  Backends with a native Cartesian mover may
        leave this unimplemented and override :meth:`move_p` instead.  A
        solver can be attached at runtime with :meth:`set_inverse_kinematics`.
        """
        if self._ik_solver is not None:
            return self._ik_solver.solve(pose, seed=seed)
        raise NotSupported(f'{self.name}: inverse kinematics is unavailable')

    def set_inverse_kinematics(self, solver):
        """Attach a host-side IK solver to this backend.

        ``solver`` must expose ``solve(pose, seed=None) -> joints``.  The
        capability flag is updated so callers can branch and ``move_p`` can
        use the host IK path.
        """
        self._ik_solver = solver
        self.capabilities = replace(
            self.capabilities, inverse_kinematics=solver is not None)
        return self

    def init_period(self):
        """Start an optional controller-side real-time period.

        This is the *generic hook*: every backend has it, but only a backend
        with a controller-side clock (UR RTDE) can return a token.  Returning
        ``None`` means "use the host clock"; callers must not treat it as an
        error.  The concrete ``initPeriod()/waitPeriod()`` RTDE methods are
        implementation details of :class:`UrRtdeRobot`, not a toolkit-wide API.
        """
        return None

    def wait_period(self, started) -> bool:
        """Wait for the next controller period; ``False`` means host fallback.

        Piper/MuJoCo and other backends intentionally return ``False`` unless
        their SDK exposes a real controller period.  The caller's host-clock
        branch is the common implementation for them.
        """
        return False

    # ---------------------------------------------------------------- motion
    @abc.abstractmethod
    def servo_j(self, joints, dt: float) -> bool:
        """Request an absolute joint target; True is not arrival confirmation.

        Consult capabilities.servo_j_kind and servo_j_timing: dt may be a
        controller argument, a host scheduling hint, or a simulation step.
        This interface does not guarantee hardware expiry after dt, a native
        real-time servo, or even a transmitted frame when a backend throttles.
        """

    def servo_l(self, pose, dt: float) -> bool:
        raise NotSupported(f'{self.name}: Cartesian servo is not available')

    def move_j(self, joints, speed: float | None = None, accel: float | None = None,
               wait: bool = True, rate_hz: float = 100.0, timeout: float | None = None,
               abort_event=None, use_native: bool = True) -> bool:
        """Stream a min-jerk move from the current configuration to ``joints``.

        ``speed`` (default 0.5 rad/s) and ``accel`` bound the peak joint
        velocity and acceleration.  The move runs through the monitored servo
        stream (:func:`~robot_toolkit.control.trajectory.run_trajectory`), so
        tracking error and stale feedback stop it like any other stream.
        Backends with a native planner should override this method.
        ``use_native`` is accepted for a uniform call signature; this base
        implementation is always generic.  A timeout stops motion and
        returns ``False``.
        """
        from ..control.trajectory import min_jerk, run_trajectory

        self._motion_detail = {'reason': None, 'phase': 'interpolation'}
        if not self.capabilities.servo_j:
            raise NotSupported(
            f'{self.name}: generic move_j needs servo_j')
        target = np.asarray(joints, dtype=float)
        start = self.wait_for_state().q
        if start is None:
            raise Aborted('no joint state available')
        start = np.asarray(start, dtype=float)
        if float(np.max(np.abs(target - start))) < 1e-9:
            return True
        speed = float(speed if speed is not None else 0.5)
        dt = 1.0 / max(1.0, float(rate_hz))
        q = min_jerk(start, target, dt, speed, accel if accel and accel > 0 else None)
        result = run_trajectory(self, q, dt, wait=wait, timeout=timeout,
                                abort_event=abort_event)
        self._motion_detail = {'reason': None if result.ok else result.reason,
                               'phase': result.phase}
        if result.error:
            self._motion_detail['error'] = result.error
        if result.reason == 'aborted':
            raise Aborted('motion aborted')
        return result.ok

    def move_l(self, pose, speed: float | None = None, accel: float | None = None,
               wait: bool = True) -> bool:
        raise NotSupported(f'{self.name}: Cartesian linear move is not available')

    def move_p(self, pose, wait: bool = True, timeout: float | None = None,
               use_native: bool = True, speed: float | None = None,
               accel: float | None = None) -> bool:
        """Point-to-point Cartesian move to ``[x, y, z, rx, ry, rz]``.

        Native P movers (currently Piper) override this.  The generic path
        resolves ``pose`` through :meth:`inverse_kinematics` and then calls
        :meth:`move_j`; if the backend has no IK it falls back to
        :meth:`move_l` when that capability is advertised.  ``use_native=False``
        forces the generic IK + generic joint interpolation path; ``speed``
        and ``accel`` are forwarded to that generic ``move_j``.
        """
        seed = None
        try:
            seed = self.state().q
        except Exception:
            seed = None
        joints = None
        try:
            joints = self.inverse_kinematics(pose, seed=seed)
        except TypeError:
            # Backends with an older IK signature do not accept a seed.
            joints = self.inverse_kinematics(pose)
        except NotSupported:
            joints = None
        if joints is not None and len(joints) > 0:
            if use_native:
                return self.move_j(joints, speed=speed, accel=accel,
                                   wait=wait, timeout=timeout)
            return RobotBackend.move_j(self, joints, speed=speed, accel=accel,
                                       wait=wait, timeout=timeout)
        if self.capabilities.inverse_kinematics:
            raise NotSupported(
                f'{self.name}: inverse kinematics failed for pose {pose}')
        if self.capabilities.move_l:
            return self.move_l(pose, wait=wait)
        raise NotSupported(
            f'{self.name}: no inverse kinematics or linear Cartesian move for move_p')

    def stop_motion(self, deceleration: float = 2.0) -> bool:
        """Best-effort stop. Backends override when they have a native stop."""
        return False

    stop = stop_motion

    def freedrive(self, enabled: bool) -> bool:
        raise NotSupported(f'{self.name}: freedrive is not available')

    def hold(self, seconds: float, rate_hz: float = 125.0, abort_event=None) -> bool:
        """Servo the current configuration for ``seconds`` (settle / wait)."""
        state = self.wait_for_state()
        dt = 1.0 / max(1.0, rate_hz)
        deadline = time.monotonic() + max(0.0, float(seconds))
        while time.monotonic() < deadline:
            if abort_event is not None and abort_event.is_set():
                self.stop_motion()
                raise Aborted('hold aborted')
            if not self.servo_j(state.q, dt):
                return False
            time.sleep(dt)
        return True

    # ------------------------------------------------------------------ utils
    def wait_until_settled(self, target=None, timeout: float = 10.0,
                           rate_hz: float = 50.0, tolerance: float = 0.02,
                           abort_event=None) -> bool:
        if target is None:
            target = self.state().q
        target = np.asarray(target, dtype=float)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if abort_event is not None and abort_event.is_set():
                raise Aborted('settle aborted')
            current = self.state().q
            if current is not None and float(np.max(np.abs(current - target))) <= tolerance:
                return True
            time.sleep(1.0 / max(1.0, rate_hz))
        return False

    def wait_pose(self, pose, translation_tolerance: float = 0.005,
                  rotation_tolerance: float = 0.05, timeout: float = 10.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            state = self.state()
            if state.tcp is not None:
                translation, rotation = pose_error(state.tcp, pose)
                if translation <= translation_tolerance and rotation <= rotation_tolerance:
                    return True
            time.sleep(0.02)
        return False

    def __repr__(self) -> str:
        return f'{type(self).__name__}(name={self.name!r}, dof={self.dof})'
