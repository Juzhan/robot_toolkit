"""The single robot interface every backend (simulator or real) implements."""

from __future__ import annotations

import abc
import math
import time
from dataclasses import dataclass, field, replace

import numpy as np

from ..core.conventions import RobotSpec
from ..core.mathutils import pose_error
from ..errors import Aborted, NotSupported


@dataclass(frozen=True)
class Capabilities:
    """Advertised features; callers branch on these instead of on robot names."""

    servo_j: bool = True
    servo_l: bool = False
    move_j: bool = True
    move_l: bool = False
    move_p: bool = False
    inverse_kinematics: bool = False
    freedrive: bool = False
    gripper: bool = False
    protective_stop_report: bool = True


@dataclass
class RobotState:
    """One synchronized sample of robot state.

    ``q`` and ``qd`` are in radians; ``tcp`` is ``[x, y, z, rx, ry, rz]`` in the
    robot base frame.  ``t`` is a monotonic timestamp in seconds.
    """

    t: float = 0.0
    q: np.ndarray | None = None
    qd: np.ndarray | None = None
    tcp: np.ndarray | None = None
    protective_stop: bool = False
    connected: bool = True
    extra: dict = field(default_factory=dict)

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
        )

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
        """Send one joint target valid for ``dt`` seconds. Return success."""

    def servo_l(self, pose, dt: float) -> bool:
        raise NotSupported(f'{self.name}: Cartesian servo is not available')

    def move_j(self, joints, speed: float | None = None, accel: float | None = None,
               wait: bool = True, rate_hz: float = 100.0, timeout: float | None = None,
               abort_event=None, use_native: bool = True) -> bool:
        """Interpolate from the current configuration to ``joints``.

        Uses a cosine ease-in/ease-out profile.  Backends with a native planner
        should override this method.  ``use_native`` is accepted for a uniform
        call signature; this base implementation is always generic.
        A timeout stops motion and returns ``False``.
        """
        if not self.capabilities.servo_j:
            raise NotSupported(
            f'{self.name}: generic move_j needs servo_j')
        target = np.asarray(joints, dtype=float)
        start = self.wait_for_state().q
        if start is None:
            raise Aborted('no joint state available')
        delta = target - start
        distance = float(np.max(np.abs(delta)))
        if distance < 1e-9:
            return True
        speed = float(speed if speed is not None else 0.5)
        if speed <= 0:
            raise ValueError('speed must be positive')
        duration = distance / speed
        if accel is not None and accel > 0:
            duration = max(duration, 2.0 * math.sqrt(distance / float(accel)))
        timeout = float(timeout if timeout is not None else duration * 3.0 + 5.0)
        dt = 1.0 / max(1.0, float(rate_hz))
        steps = max(1, int(math.ceil(duration * rate_hz)))
        deadline = time.monotonic() + timeout
        for index in range(1, steps + 1):
            if abort_event is not None and abort_event.is_set():
                self.stop_motion()
                raise Aborted('motion aborted')
            progress = index / steps
            eased = 0.5 - 0.5 * math.cos(math.pi * progress)
            if not self.servo_j(start + delta * eased, dt):
                return False
            if time.monotonic() > deadline:
                self.stop_motion()
                return False
            # Even with wait=False the base class has to stream the interpolation;
            # only the final settle is skipped.
            time.sleep(dt)
        if not wait:
            return True
        while True:
            if abort_event is not None and abort_event.is_set():
                self.stop_motion()
                raise Aborted('motion aborted')
            current = self.state().q
            error = (float('inf') if current is None
                     else float(np.max(np.abs(current - target))))
            if error <= 0.02:
                return True
            if time.monotonic() >= deadline:
                self.stop_motion()
                return False
            if not self.servo_j(target, dt):
                self.stop_motion()
                return False
            time.sleep(dt)

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
