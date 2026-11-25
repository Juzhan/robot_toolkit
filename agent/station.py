"""Framework-neutral execution station: one arm, its gripper and cameras.

External agent frameworks (UNI, StrangeLoop, RPent) adapt this one contract
instead of each re-implementing robot access. Every public pose is the
configured control point in the world frame as a 4x4 transform; gripper
openings are normalized (0 closed, 1 open). Motion returns a ``Receipt``
that states what ran and whether the arm was observed stationary afterwards;
it never claims a stop it did not observe. After an unknown outcome the
station refuses motion until an operator calls ``clear_block``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ..core.mathutils import invert_transform, matrix_to_pose, pose_error, pose_to_matrix
from ..errors import NotSupported, RobotUnavailable
from ..robot_platform import RobotPlatform
from .options import AgentOptions

# Platform reasons that mean nothing was sent to the arm.
_NOT_SENT = {'not_ready', 'disconnected', 'feedback_unavailable', 'busy',
             'invalid_argument', 'unsupported', 'safety_rejected'}


class Rejected(ValueError):
    """A request refused before anything was sent to the hardware."""


@dataclass
class StationState:
    t: float
    q: np.ndarray
    T_world_control: np.ndarray
    gripper: float
    gripper_width: float | None
    gripper_measured: bool
    protective_stop: bool
    fault: bool | None
    timestamp_clock: str


@dataclass
class StationFrame:
    rgb: np.ndarray
    depth_m: np.ndarray | None
    K: np.ndarray | None
    T_world_optical: np.ndarray | None
    t: float
    calibration_source: str


@dataclass
class Receipt:
    """``outcome``: completed, none (nothing ran), partial or unknown."""

    accepted: bool
    outcome: str
    stop_confirmed: bool | None
    requires_operator: bool
    reason: str
    state: StationState | None
    diagnostics: dict = field(default_factory=dict)


def _matrix(value, name) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(f'{name} must be a finite 4x4 transform')
    return matrix


def _depth_m(depth):
    if depth is None:
        return None
    depth = np.asarray(depth)
    return depth.astype(np.float32) * np.float32(0.001) if depth.dtype == np.uint16 else depth.astype(np.float32)


def _resolve(path) -> str:
    """Relative config paths name files under the current or the workspace directory."""
    from pathlib import Path
    candidate = Path(path)
    if candidate.is_absolute() or candidate.exists():
        return str(candidate)
    return str(Path(__file__).resolve().parents[2] / candidate)


class Station:
    def __init__(self, platform: RobotPlatform, options: AgentOptions | None = None, *,
                 calibrations: dict | None = None, physical: bool = True):
        self.platform = platform
        self.robot = platform.robot
        self.gripper = platform.gripper
        self.cameras = platform.cameras
        self.options = options or AgentOptions()
        self.physical = bool(physical)
        self.T_world_base = _matrix(self.options.T_world_base, 'T_world_base')
        self.T_tcp_control = _matrix(self.options.T_tcp_control, 'T_tcp_control')
        self.calibrations = dict(calibrations or {})
        self.execute = False
        self.blocked = None
        self.gripper_command = None
        self.last_stillness = None
        self._prepared = False

    # ------------------------------------------------------------- building
    @classmethod
    def from_task(cls, task, *, mock: bool = False, options: AgentOptions | None = None) -> 'Station':
        """Build devices from a task config name/path (or a loaded TaskConfig)."""
        from ..calibrate.store import CalibrationStore
        from ..profiles import build_camera_hub, build_gripper, build_robot
        from ..task_runner import load_task_config
        task = load_task_config(task) if isinstance(task, str) else task
        options = options or task.agent
        robot = build_robot(task.robot, mock=mock)
        gripper = None
        kind = str((task.gripper or {}).get('kind', '')).lower()
        if kind not in ('', 'none', 'disabled'):
            gripper = build_gripper(kind, robot=robot, options=dict(task.gripper.get('options') or {}))
        entries = list(task.collect.cameras)
        if mock:
            entries = [{'kind': 'mock', 'name': entry.get('name', 'camera'),
                        'width': int(entry.get('width', 640)), 'height': int(entry.get('height', 480)),
                        'fps': int(entry.get('fps', 30)), 'depth': bool(entry.get('depth', False))}
                       for entry in entries]
        cameras = build_camera_hub(entries) if entries else None
        calibrations = {}
        for path in options.calibration_paths:
            calibrations.update(CalibrationStore(_resolve(path)).cameras)
        return cls(RobotPlatform(robot, gripper=gripper, cameras=cameras), options,
                   calibrations=calibrations, physical=not mock)

    def open(self, *, execute: bool = False) -> 'Station':
        """Connect devices; without ``execute`` the arm link is read-only and motion is refused."""
        if execute:
            self.robot.connect()
            if self.gripper is not None:
                self.gripper.connect()
        else:
            try:
                self.robot.connect_read_only()
            except NotSupported:
                if self.physical:
                    raise
                self.robot.connect()
        if self.cameras is not None and len(self.cameras):
            self._open_cameras()
        self.execute = bool(execute)
        return self

    def _open_cameras(self) -> None:
        """Open and warm up; a camera that delivers nothing gets one hardware reset and a retry.

        Seen on a D435: the stream stalls with depth enabled until the device is reset.
        """
        for attempt in range(2):
            self.cameras.open()
            if self.cameras.warmup(min_frames=2, timeout=8.0):
                return
            stalled = [camera for camera in self.cameras if camera.frames_seen == 0]
            self.cameras.close()
            resettable = [camera for camera in stalled if hasattr(camera, 'hardware_reset')]
            if attempt or not resettable:
                break
            for camera in resettable:
                camera.hardware_reset()
        raise RobotUnavailable('cameras did not deliver frames within 8 s'
                               + (' (after a hardware reset)' if attempt else ''))

    def close(self) -> None:
        errors = []
        for device in (self.cameras, self.gripper):
            if device is not None:
                try:
                    (device.close if device is self.cameras else device.disconnect)()
                except Exception as exc:
                    errors.append(f'{type(exc).__name__}: {exc}')
        try:
            self.robot.disconnect()
        except Exception as exc:
            errors.append(f'{type(exc).__name__}: {exc}')
        self.execute = False
        if errors:
            raise RobotUnavailable('station close failed: ' + '; '.join(errors))

    # ------------------------------------------------------------- describe
    def describe(self) -> dict:
        spec = self.robot.spec
        cameras = []
        for name in (self.cameras.names if self.cameras is not None else []):
            calibration = self.calibrations.get(name)
            mounted = 'wrist' if calibration is not None and calibration.tcp_camera is not None else 'scene'
            camera = self.cameras[name]
            cameras.append({'name': name, 'role': mounted,
                            'has_depth': bool(getattr(camera, 'depth_enabled', False)),
                            'has_calibration': calibration is not None and (
                                calibration.base_camera is not None or calibration.tcp_camera is not None)})
        return {
            'arm': self.options.arm_name, 'robot_id': self.options.robot_id or spec.name,
            'robot': spec.name, 'backend': type(self.robot).__name__, 'dof': spec.dof,
            'joint_names': list(getattr(spec, 'joint_names', []) or []),
            'physical': self.physical, 'execute': self.execute,
            'control_site': self.options.control_site,
            'rotation_radius_m': self.options.rotation_radius_m,
            'gripper': None if self.gripper is None else {
                'min_width_m': self.gripper.min_width, 'max_width_m': self.gripper.max_width},
            'stop_motion_kind': self.robot.capabilities.stop_motion_kind,
            'cameras': cameras,
            'limits': {'max_step_m': self.options.max_step_m, 'max_step_rad': self.options.max_step_rad,
                       'max_joint_jump_rad': self.options.max_joint_jump_rad},
        }

    # ---------------------------------------------------------------- state
    def _control_from_tcp(self, tcp) -> np.ndarray:
        return self.T_world_base @ pose_to_matrix(tcp) @ self.T_tcp_control

    def _tcp_from_control(self, T_world_control) -> np.ndarray:
        return matrix_to_pose(invert_transform(self.T_world_base) @ T_world_control
                              @ invert_transform(self.T_tcp_control))

    def _gripper_reading(self, sample):
        if self.gripper is not None and self.gripper.connected:
            return float(self.gripper.state().width), True
        width = sample.extra.get('gripper_width')
        if isinstance(width, (int, float)) and np.isfinite(width):
            return float(width), True
        return None, False

    def _normalize(self, width) -> float:
        low, high = (self.gripper.min_width, self.gripper.max_width) if self.gripper is not None \
            else (self.robot.spec.gripper.min_width, self.robot.spec.gripper.max_width)
        return float(np.clip((width - low) / max(high - low, 1e-9), 0.0, 1.0))

    def _width(self, opening) -> float:
        return self.gripper.min_width + float(opening) * (self.gripper.max_width - self.gripper.min_width)

    def read(self) -> StationState:
        sample = self.robot.state()
        if sample.q is None or sample.tcp is None:
            raise RobotUnavailable('joint and TCP feedback are required')
        width, measured = self._gripper_reading(sample)
        opening = self._normalize(width) if measured else (
            1.0 if self.gripper_command is None else self.gripper_command)
        return StationState(t=float(sample.t), q=np.asarray(sample.q, float).copy(),
                            T_world_control=self._control_from_tcp(sample.tcp), gripper=opening,
                            gripper_width=width, gripper_measured=measured,
                            protective_stop=bool(sample.protective_stop), fault=sample.fault,
                            timestamp_clock=sample.timestamp_clock)

    def capture(self, names=None) -> dict[str, StationFrame]:
        if self.cameras is None:
            raise RobotUnavailable('this station has no cameras')
        names = list(self.cameras.names) if names is None else list(names)
        unknown = set(names) - set(self.cameras.names)
        if unknown:
            raise Rejected(f'unknown cameras: {sorted(unknown)}')
        frames = self.cameras.snapshot(strict=True)
        tcp = self.robot.state().tcp
        result = {}
        for name in names:
            frame = frames[name]
            if frame.rgb is None:
                raise RobotUnavailable(f'camera {name!r} returned no image')
            calibration = self.calibrations.get(name)
            K = frame.intrinsics if frame.intrinsics is not None else (
                None if calibration is None else calibration.intrinsics)
            base_optical, source = None, 'uncalibrated'
            if calibration is not None:
                base_optical = calibration.world_transform(tcp)
                source = 'eye_to_hand' if calibration.base_camera is not None else 'eye_in_hand'
            if base_optical is None and frame.T_world_camera is not None:
                base_optical, source = frame.T_world_camera, 'camera_driver'
            T = None if base_optical is None else self.T_world_base @ np.asarray(base_optical, float)
            if T is None:
                source = 'uncalibrated'
            result[name] = StationFrame(rgb=np.asarray(frame.rgb, np.uint8), depth_m=_depth_m(frame.depth),
                                        K=None if K is None else np.asarray(K, float), T_world_optical=T,
                                        t=float(frame.t), calibration_source=source)
        return result

    def stationary(self) -> bool | None:
        """True once fresh feedback shows the joints still within ``settle_timeout_s``.

        False when they kept moving for the whole wait; None when feedback could
        not show it. ``last_stillness`` keeps the evidence.
        """
        started = time.monotonic()
        spread = None
        while True:
            result, spread = self._still_window()
            waited = time.monotonic() - started
            if result is not False or waited >= self.options.settle_timeout_s:
                self.last_stillness = {'still': result, 'waited_s': round(waited, 3),
                                       'joint_spread_rad': spread}
                return result

    def _still_window(self):
        samples = []
        steps = 3
        for index in range(steps):
            try:
                samples.append(self.robot.state())
            except Exception:
                return None, None
            if index < steps - 1:
                time.sleep(self.options.stationary_window_s / (steps - 1))
        if any(sample.q is None for sample in samples):
            return None, None
        if self.physical and not samples[-1].t > samples[0].t:
            return None, None
        spread = float(np.max(np.ptp(np.stack([np.asarray(sample.q, float) for sample in samples]), axis=0)))
        return spread <= self.options.stationary_joint_rad, spread

    # --------------------------------------------------------------- motion
    def clear_block(self) -> None:
        """Operator confirmation that the arm was inspected and may move again."""
        self.blocked = None

    def _admit(self) -> Receipt | None:
        if not self.execute:
            raise Rejected('the station is open read-only; no command was sent')
        if self.blocked is not None:
            return self._receipt(False, 'none', 'operator_recovery_required', stop=self.stationary(),
                                 operator=True, blocked_reason=self.blocked)
        return None

    def _prepare(self) -> None:
        if not self._prepared:
            if not self.robot.prepare_motion():
                raise RobotUnavailable('robot motion preparation failed')
            self._prepared = True

    def _receipt(self, accepted, outcome, reason, *, stop, operator=False, **diagnostics) -> Receipt:
        try:
            state = self.read()
        except Exception as exc:
            state = None
            diagnostics['state_error'] = f'{type(exc).__name__}: {exc}'
            operator = operator or outcome != 'none'
        if state is not None and (state.protective_stop or state.fault):
            operator = True
            diagnostics['protective_stop'], diagnostics['fault'] = state.protective_stop, state.fault
        if outcome == 'unknown' or (outcome == 'partial' and stop is not True):
            operator = True
        if operator and self.blocked is None and outcome != 'none':
            self.blocked = reason
        if self.last_stillness is not None:
            diagnostics['stillness'] = self.last_stillness
        return Receipt(accepted, outcome, stop, operator, reason, state, diagnostics)

    def _joint_target(self, T_world_control, state: StationState) -> np.ndarray:
        T = _matrix(T_world_control, 'target')
        dp, angle = pose_error(matrix_to_pose(state.T_world_control), matrix_to_pose(T))
        if dp > self.options.max_step_m or angle > self.options.max_step_rad:
            raise Rejected(f'step of {dp:.3f} m / {angle:.3f} rad exceeds the '
                           f'{self.options.max_step_m} m / {self.options.max_step_rad} rad limit')
        joints = self.robot.inverse_kinematics(self._tcp_from_control(T), seed=state.q)
        if joints is None or len(joints) != len(state.q):
            raise Rejected('no inverse-kinematics solution for the target pose')
        return np.asarray(joints, float)

    def _run(self, joints, state: StationState, gripper, target_pose=None) -> Receipt:
        joints = np.asarray(joints, float)
        if joints.shape != state.q.shape or not np.all(np.isfinite(joints)):
            raise Rejected(f'joint target must contain {state.q.size} finite values')
        jump = float(np.max(np.abs(joints - state.q)))
        if jump > self.options.max_joint_jump_rad:
            raise Rejected(f'joint target jumps {jump:.3f} rad (limit {self.options.max_joint_jump_rad})')
        if gripper is not None and not 0.0 <= float(gripper) <= 1.0:
            raise Rejected('gripper opening must be in [0, 1]')
        self._prepare()
        diagnostics = {'joint_jump_rad': jump}
        if jump > 1e-6:
            record = self.platform.move_j(joints, joint_tolerance=self.options.joint_tolerance_rad,
                                          timeout=self.options.move_timeout_s, **self.options.move_options)
            report = record.to_report()
            diagnostics['execution'] = report
            if not record.ok:
                sent = report.get('command_ok') is not None or report['reason'] not in _NOT_SENT
                outcome = ('none' if not sent else 'partial' if report['reason'] == 'tolerance_not_met'
                           else 'unknown')
                return self._receipt(False, outcome, report['reason'], stop=self.stationary(), **diagnostics)
        if gripper is not None:
            try:
                diagnostics['gripper'] = self._command_gripper(float(gripper))
            except Exception as exc:
                return self._receipt(False, 'unknown', 'gripper_failed', stop=self.stationary(),
                                     gripper_error=f'{type(exc).__name__}: {exc}', **diagnostics)
        stop = self.stationary()
        if target_pose is not None:
            after = self.read()
            dp, angle = pose_error(matrix_to_pose(after.T_world_control), matrix_to_pose(target_pose))
            diagnostics.update(position_error_m=dp, rotation_error_rad=angle)
            if dp > self.options.position_tolerance_m or angle > self.options.rotation_tolerance_rad:
                return self._receipt(False, 'partial', 'tolerance_not_met', stop=stop, **diagnostics)
        return self._receipt(True, 'completed', 'completed', stop=stop, **diagnostics)

    def _command_gripper(self, opening) -> dict:
        if self.gripper is None:
            return {'status': 'no_gripper'}
        previous = self.gripper_command
        self.gripper_command = opening
        if previous is not None and abs(previous - opening) < self.options.gripper_deadband:
            return {'status': 'unchanged', 'opening': opening}
        width = self._width(opening)
        # Closing on an object stops short of the target; the reading reports what happened.
        reached = self.gripper.move(width, wait=True, timeout=self.options.gripper_timeout_s)
        return {'status': 'commanded', 'opening': opening, 'target_width_m': width,
                'target_reached': bool(reached), 'width_m': float(self.gripper.state().width)}

    def preview_pose(self, T_world_control) -> dict:
        """Every pre-dispatch check of ``move_pose`` without sending anything.

        Raises ``Rejected`` (or the guard's ``SafetyViolation``) exactly where a
        real command would be refused; works on a read-only connection.
        """
        state = self.read()
        joints = self._joint_target(_matrix(T_world_control, 'target'), state)
        jump = float(np.max(np.abs(joints - state.q)))
        if jump > self.options.max_joint_jump_rad:
            raise Rejected(f'joint target jumps {jump:.3f} rad (limit {self.options.max_joint_jump_rad})')
        guard = getattr(self.robot, 'guard', None)
        if guard is not None and guard.limits.violates(joints):
            raise Rejected('target violates hardware joint limits')
        if hasattr(self.robot, 'check_workspace_path'):
            self.robot.check_workspace_path(state.q, joints)
        return {'joints': joints, 'joint_jump_rad': jump}

    def move_pose(self, T_world_control, gripper: float | None = None) -> Receipt:
        refusal = self._admit()
        if refusal is not None:
            return refusal
        state = self.read()
        target = _matrix(T_world_control, 'target')
        return self._run(self._joint_target(target, state), state, gripper, target_pose=target)

    def move_joints(self, joints, gripper: float | None = None) -> Receipt:
        refusal = self._admit()
        if refusal is not None:
            return refusal
        return self._run(joints, self.read(), gripper)

    def hold(self, gripper: float | None = None) -> Receipt:
        refusal = self._admit()
        if refusal is not None:
            return refusal
        state = self.read()
        return self._run(state.q, state, gripper)

    def stop(self) -> dict:
        """Best-effort software stop plus observed stillness; never a certified E-stop."""
        command_ok, error = None, None
        if self.execute:
            try:
                command_ok = bool(self.robot.stop_motion())
            except Exception as exc:
                error = f'{type(exc).__name__}: {exc}'
        still = self.stationary()
        return {'status': 'confirmed_idle' if still else 'unknown', 'command_ok': command_ok,
                'error': error, 'stop_motion_kind': self.robot.capabilities.stop_motion_kind,
                'physical_stop_guaranteed': False}
