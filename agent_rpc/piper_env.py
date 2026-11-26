"""Piper real-robot backend for the agent ``env.*`` protocol.

The backend borrows an already-connected :class:`RobotPlatform`; it never
connects, enables or closes hardware by itself.  Motion is refused unless the
server was constructed with ``enable_motion=True``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, fields
from typing import Any

import numpy as np

from ..core.mathutils import matrix_to_rotvec, rotvec_to_matrix, rotation_angle
from ..errors import NotSupported, RobotUnavailable, SafetyViolation
from ..runtime.robot_platform import RobotPlatform
from .env import EnvService


@dataclass(frozen=True)
class PiperEnvConfig:
    main_camera: str = 'wrist'
    extra_cameras: tuple[str, ...] = ()
    task_language: str = ''
    enable_motion: bool = False
    max_delta_m: float = 0.03
    max_rotation_rad: float = 0.15
    position_tolerance_m: float = 0.005
    rotation_tolerance_rad: float = 0.05
    move_timeout_s: float = 20.0
    gripper_timeout_s: float = 5.0
    base_T_world: np.ndarray | None = None
    protocol_version: int = 1

    def __post_init__(self):
        if self.max_delta_m <= 0 or self.max_rotation_rad <= 0:
            raise ValueError('motion limits must be positive')
        if self.position_tolerance_m <= 0 or self.rotation_tolerance_rad <= 0:
            raise ValueError('motion tolerances must be positive')


def _vec3(value, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (3,):
        raise ValueError(f'{name} must contain exactly 3 values')
    if not np.isfinite(array).all():
        raise ValueError(f'{name} must contain only finite values')
    return array


def _depth_m(depth) -> np.ndarray | None:
    if depth is None:
        return None
    array = np.asarray(depth)
    if array.dtype == np.uint16:
        return array.astype(np.float32) * np.float32(0.001)
    return array.astype(np.float32)


class PiperRealBackend:
    """Expose one real Piper platform through the ``env.*`` routes."""

    rpc_methods = (
        'get_env_meta', 'reset', 'get_robot_state', 'get_observation',
        'get_camera_meta', 'move_delta', 'rotate_delta', 'set_gripper',
        'get_task_language', 'render_camera', 'step', 'chunk_step',
    )
    rpc_readonly = {
        'get_env_meta': True, 'get_robot_state': True, 'get_observation': True,
        'get_camera_meta': True, 'get_task_language': True, 'render_camera': True,
    }

    def __init__(self, platform: RobotPlatform, config: PiperEnvConfig | None = None,
                 *, robot_name: str = 'piper'):
        if platform.cameras is None or not len(platform.cameras):
            raise ValueError('PiperRealBackend requires an open CameraHub')
        self.platform = platform
        self.config = config or PiperEnvConfig()
        self.robot_name = robot_name
        names = tuple(platform.cameras.names)
        if self.config.main_camera not in names:
            raise ValueError(f'main camera {self.config.main_camera!r} is not in {names}')
        extras = self.config.extra_cameras or tuple(
            name for name in names if name != self.config.main_camera)
        missing = set(extras) - set(names)
        if missing:
            raise ValueError(f'unknown extra cameras: {sorted(missing)}')
        self.camera_names = names
        self.extra_cameras = tuple(extras)

    # ------------------------------------------------------------- helpers
    def _require_motion(self) -> None:
        if not self.config.enable_motion:
            raise SafetyViolation(
                'motion is disabled; start the env server with enable_motion=True')

    def _state(self):
        state = self.platform.robot.state()
        if state.q is None:
            raise RobotUnavailable('robot state is unavailable')
        return state

    def _feedback(self, state) -> dict:
        feedback = {item.name: getattr(state, item.name) for item in fields(state)}
        gripper = self.platform.gripper
        if gripper is not None:
            try:
                feedback['gripper'] = gripper.state()
            except Exception as exc:  # noqa: BLE001 - reported, not hidden
                feedback['gripper'] = {'error': f'{type(exc).__name__}: {exc}'}
        return feedback

    def _frame(self, name: str):
        return self.platform.cameras[name].snapshot()

    # ------------------------------------------------------------- routes
    def get_env_meta(self) -> dict:
        gripper = self.platform.gripper
        return {
            'robot': self.robot_name,
            'mode': 'real',
            'protocol_version': self.config.protocol_version,
            'action_dim': 7,
            'cameras': list(self.camera_names),
            'main_camera': self.config.main_camera,
            'extra_cameras': list(self.extra_cameras),
            'gripper_range_m': None if gripper is None else [gripper.min_width, gripper.max_width],
            'motion_enabled': bool(self.config.enable_motion),
        }

    def reset(self) -> dict:
        """Never move hardware; an operator owns the physical reset."""
        state = self._state()
        return {
            'ok': True,
            'reset': 'operator',
            'message': 'physical reset is operator-owned; no motion was issued',
            'robot_state': self._feedback(state),
        }

    def get_robot_state(self) -> dict:
        state = self._state()
        return {
            'ok': True,
            'timestamp': time.time(),
            'motion_enabled': bool(self.config.enable_motion),
            'action_dim': 7,
            'robot_state': self._feedback(state),
        }

    def get_observation(self) -> dict:
        observation = self.platform.observe(include_images=True, strict=True)
        frames = observation.frames
        if self.config.main_camera not in frames:
            raise RobotUnavailable(f'missing main camera {self.config.main_camera!r}')
        main = frames[self.config.main_camera]
        if main.rgb is None:
            raise RobotUnavailable('main camera returned no RGB frame')
        result: dict[str, Any] = {
            'ok': True,
            'observation_id': observation.observation_id,
            'timestamp': observation.finished_at,
            'clock': 'host_perf_counter',
            'main_images': np.asarray(main.rgb),
            'robot_state': self._feedback(observation.state) if observation.state else None,
            'errors': observation.errors,
        }
        if main.depth is not None:
            result['main_depths'] = _depth_m(main.depth)
        extras = [frames[name] for name in self.extra_cameras if name in frames]
        if extras:
            result['extra_view_images'] = np.stack(
                [np.asarray(frame.rgb) for frame in extras], axis=0)
            depths = [_depth_m(frame.depth) for frame in extras]
            if all(depth is not None for depth in depths):
                result['extra_view_depths'] = np.stack(depths, axis=0)
        return result

    def get_camera_meta(self) -> dict:
        cameras = {}
        for name in self.camera_names:
            frame = self._frame(name)
            world = frame.T_world_camera
            base = None
            if world is not None:
                base = (np.eye(4) if self.config.base_T_world is None
                        else np.asarray(self.config.base_T_world, dtype=float)) @ world
            cameras[name] = {
                'width': int(frame.rgb.shape[1]) if frame.rgb is not None else None,
                'height': int(frame.rgb.shape[0]) if frame.rgb is not None else None,
                'intrinsics': frame.intrinsics,
                'distortion': frame.distortion,
                'T_world_camera': world,
                'T_base_camera': base,
                'depth': frame.depth is not None,
                'depth_units': 'm',
                'meta': dict(frame.meta),
            }
        return {
            'ok': True,
            'cameras': cameras,
            'main_camera': self.config.main_camera,
            'extra_cameras': list(self.extra_cameras),
            'base_frame': 'world' if self.config.base_T_world is None else 'configured',
        }

    def move_delta(self, delta_xyz) -> dict:
        self._require_motion()
        requested = _vec3(delta_xyz, name='delta_xyz')
        limit = float(self.config.max_delta_m)
        applied = np.clip(requested, -limit, limit)
        state = self._state()
        if state.tcp is None:
            raise RobotUnavailable('TCP pose is unavailable; cannot apply a delta')
        target = np.asarray(state.tcp, dtype=float).copy()
        target[:3] += applied
        record = self.platform.move_p(
            target, position_tolerance=self.config.position_tolerance_m,
            rotation_tolerance=self.config.rotation_tolerance_rad)
        after = self.platform.robot.state()
        error = None
        if after.tcp is not None:
            error = float(np.linalg.norm(np.asarray(after.tcp)[:3] - target[:3]))
        return {
            'ok': bool(record.ok) and (error is None or error <= self.config.position_tolerance_m),
            'requested_delta_m': requested.tolist(),
            'applied_delta_m': applied.tolist(),
            'clamped': bool(np.any(applied != requested)),
            'target_tcp': target.tolist(),
            'final_tcp': None if after.tcp is None else np.asarray(after.tcp).tolist(),
            'final_error_m': error,
            'execution': record.to_report(),
            'robot_state': self._feedback(after),
        }

    def rotate_delta(self, delta_rpy) -> dict:
        self._require_motion()
        requested = _vec3(delta_rpy, name='delta_rpy')
        limit = float(self.config.max_rotation_rad)
        applied = np.clip(requested, -limit, limit)
        state = self._state()
        if state.tcp is None:
            raise RobotUnavailable('TCP pose is unavailable; cannot apply a rotation')
        tcp = np.asarray(state.tcp, dtype=float)
        target = tcp.copy()
        target[3:] = matrix_to_rotvec(
            rotvec_to_matrix(applied) @ rotvec_to_matrix(tcp[3:]))
        record = self.platform.move_p(
            target, position_tolerance=self.config.position_tolerance_m,
            rotation_tolerance=self.config.rotation_tolerance_rad)
        after = self.platform.robot.state()
        error = None
        if after.tcp is not None:
            error = float(rotation_angle(
                rotvec_to_matrix(np.asarray(after.tcp)[3:]).T
                @ rotvec_to_matrix(target[3:])))
        return {
            'ok': bool(record.ok) and (error is None or error <= self.config.rotation_tolerance_rad),
            'requested_delta_rpy_rad': requested.tolist(),
            'applied_delta_rpy_rad': applied.tolist(),
            'clamped': bool(np.any(applied != requested)),
            'target_tcp': target.tolist(),
            'final_tcp': None if after.tcp is None else np.asarray(after.tcp).tolist(),
            'final_rotation_error_rad': error,
            'execution': record.to_report(),
            'robot_state': self._feedback(after),
        }

    def set_gripper(self, *, open: bool) -> dict:
        self._require_motion()
        gripper = self.platform.gripper
        if gripper is None:
            raise NotSupported('this platform has no gripper backend')
        width = gripper.max_width if bool(open) else gripper.min_width
        ok = gripper.move(width, wait=True, timeout=self.config.gripper_timeout_s)
        state = gripper.state()
        return {'ok': bool(ok), 'open': bool(open), 'target_width_m': width,
                'width_m': float(state.width), 'moving': bool(state.moving),
                'gripper': state}

    def get_task_language(self) -> str:
        return self.config.task_language

    def render_camera(self, camera_name: str) -> np.ndarray:
        if camera_name not in self.camera_names:
            raise ValueError(f'unknown camera: {camera_name!r}')
        frame = self._frame(camera_name)
        if frame.rgb is None:
            raise RobotUnavailable(f'camera {camera_name!r} returned no RGB frame')
        return np.asarray(frame.rgb)

    def step(self, action):  # noqa: ARG002 - protocol completeness
        raise NotSupported('real Piper uses move_delta/rotate_delta/set_gripper, not env.step')

    def chunk_step(self, actions, *, return_all_frames: bool = False):  # noqa: ARG002
        raise NotSupported('real Piper chunk_step is not implemented')

    def close(self) -> None:
        # The platform borrows devices; the owner decides when to close them.
        self.platform.clear_history()


def build_piper_env_service(platform: RobotPlatform, config: PiperEnvConfig | None = None,
                            *, robot_name: str = 'piper') -> EnvService:
    return EnvService(PiperRealBackend(platform, config, robot_name=robot_name))
