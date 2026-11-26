"""Device composition, bounded observation history, and recorded execution."""

from __future__ import annotations

import math
import threading
import time
import uuid
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass, field, fields

import numpy as np

from ..camera.hub import CameraHub
from ..collect.observation import RobotCameraSynchronizer
from ..core.mathutils import pose_error
from ..errors import Aborted, NotSupported, RobotUnavailable, SafetyViolation
from ..gripper.base import GripperBackend, GripperState
from ..robot.base import RobotBackend, RobotState, _json_value
from ..perception.scene import SceneEstimator, SceneState


@dataclass
class Observation:
    """Detached device samples; images and arrays stay in their native form."""

    observation_id: str
    started_at: float
    finished_at: float
    connected: bool
    state: RobotState | None
    telemetry: dict = field(default_factory=dict)
    gripper: GripperState | None = None
    frames: dict = field(default_factory=dict)
    synchronization: dict | None = None
    errors: dict = field(default_factory=dict)
    camera_metadata: dict = field(default_factory=dict)
    scene: SceneState | None = None

    def feedback(self) -> dict:
        """Numeric robot feedback plus provenance, without image serialization."""
        state = None
        if self.state is not None:
            state = {item.name: getattr(self.state, item.name) for item in fields(self.state)}
            if not self.state.protective_stop_known and not self.state.protective_stop:
                state['protective_stop'] = None
        return deepcopy({
            'schema_version': 1, 'observation_id': self.observation_id,
            'connected': self.connected, 'state': state,
            'telemetry': self.telemetry, 'errors': self.errors,
            'gripper': self.gripper, 'gripper_synchronization': 'not_guaranteed',
            'acquisition': {'clock': 'host_perf_counter', 'started_at': self.started_at,
                            'finished_at': self.finished_at,
                            'synchronization': 'not_guaranteed',
                            'feedback_freshness': 'unknown'},
        })


@dataclass
class ExecutionRecord:
    """One attempted operation; reporting never triggers another device read."""

    execution_id: str
    started_at: float
    finished_at: float
    details: dict

    @property
    def ok(self) -> bool:
        return self.details['ok']

    def to_report(self) -> dict:
        return _json_value({**self.details, 'execution_id': self.execution_id,
                            'started_at': self.started_at, 'finished_at': self.finished_at,
                            'clock': 'host_perf_counter'})


class RobotPlatform:
    """Borrow existing devices; never connect, enable, or close them implicitly.

    History keeps detached metadata/state snapshots, not image buffers. Images
    belong to the returned observation; retain or record them explicitly.
    Overlapping motion calls on this platform are rejected. Direct device calls,
    other platform instances, observations and lifecycle calls must still be
    serialized by the owning application. wait=False has no background owner.
    """

    def __init__(self, robot: RobotBackend, *, gripper: GripperBackend | None = None,
                 cameras: CameraHub | None = None, history_size: int = 128,
                 sync_max_skew_s: float = 0.05, sync_max_frame_age_s: float = 0.3,
                 scene_estimator: SceneEstimator | None = None):
        if isinstance(history_size, bool) or not isinstance(history_size, int) or history_size < 0:
            raise ValueError('history_size must be a nonnegative integer')
        self.robot = robot
        self.gripper = gripper
        self.cameras = cameras
        if scene_estimator is not None and (
                cameras is None or scene_estimator.spec.camera not in cameras.names):
            raise ValueError('scene estimator needs its configured camera in the platform hub')
        self.scene_estimator = scene_estimator
        self.history_size = history_size
        self.synchronizer = (RobotCameraSynchronizer(
            robot, cameras, max_skew_s=sync_max_skew_s,
            max_frame_age_s=sync_max_frame_age_s) if cameras is not None and len(cameras) else None)
        self._observations = OrderedDict()
        self._executions = OrderedDict()
        self._motion_lock = threading.Lock()

    def _remember(self, history, key, value):
        if not self.history_size:
            return
        history[key] = deepcopy(value)
        while len(history) > self.history_size:
            history.popitem(last=False)

    @property
    def observation_ids(self) -> tuple[str, ...]:
        return tuple(self._observations)

    @property
    def execution_ids(self) -> tuple[str, ...]:
        return tuple(self._executions)

    def observation(self, observation_id: str) -> Observation:
        """Get retained metadata/state; image buffers are intentionally absent."""
        return deepcopy(self._observations[observation_id])

    def execution(self, execution_id: str) -> ExecutionRecord:
        return deepcopy(self._executions[execution_id])

    def clear_history(self) -> None:
        self._observations.clear()
        self._executions.clear()
        if self.synchronizer is not None:
            self.synchronizer.clear()

    def observe(self, *, include_images: bool = True, strict: bool = True) -> Observation:
        """Read once per source; optionally pair images with robot host-time reads.

        Gripper feedback is read separately and is not camera-time aligned.
        Strict image pairing may raise; ordinary state/telemetry errors are kept
        as evidence instead of fabricating a healthy sample.
        """
        started = time.perf_counter()
        state, frames, synchronization, gripper = None, {}, None, None
        errors, telemetry = {}, {}
        connected = bool(self.robot.connected)
        if include_images and self.synchronizer is not None:
            if any(not camera.opened for camera in self.cameras):
                raise RobotUnavailable('open cameras explicitly before observing')
            paired = self.synchronizer.observe(strict=strict)
            state, frames = paired['state'], paired['frames']
            synchronization = paired['synchronization']
        elif connected:
            try:
                state = self.robot.state()
            except Exception as exc:
                errors['state'] = {'type': type(exc).__name__, 'message': str(exc)}
        if connected:
            try:
                telemetry = self.robot.telemetry()
            except Exception as exc:
                errors['telemetry'] = {'type': type(exc).__name__, 'message': str(exc)}
        if self.gripper is not None:
            try:
                if not self.gripper.connected:
                    raise RobotUnavailable('gripper is disconnected')
                gripper = self.gripper.state()
            except Exception as exc:
                errors['gripper'] = {'type': type(exc).__name__, 'message': str(exc)}
        observation = Observation(
            uuid.uuid4().hex, started, time.perf_counter(), connected,
            deepcopy(state), deepcopy(telemetry), deepcopy(gripper), frames,
            deepcopy(synchronization), errors)
        observation.camera_metadata = deepcopy({
            name: {'t': frame.t, 'intrinsics': frame.intrinsics,
                   'distortion': frame.distortion, 'T_world_camera': frame.T_world_camera,
                   'meta': frame.meta,
                   'rgb_shape': None if frame.rgb is None else frame.rgb.shape,
                   'depth_shape': None if frame.depth is None else frame.depth.shape}
            for name, frame in frames.items()})
        if include_images and self.scene_estimator is not None:
            observation.scene = self.scene_estimator.estimate(observation)
        retained = Observation(
            observation.observation_id, observation.started_at, observation.finished_at,
            observation.connected, observation.state, observation.telemetry,
            observation.gripper, synchronization=observation.synchronization,
            errors=observation.errors, camera_metadata=observation.camera_metadata,
            scene=observation.scene)
        self._remember(self._observations, observation.observation_id, retained)
        return observation

    def move_j(self, joints, *, joint_tolerance: float = 0.02,
               stable_duration_s: float = 0.0, monitor_hz: float = 50.0,
               feedback_stall_timeout_s: float = 0.5, **kwargs) -> ExecutionRecord:
        """Record joint arrival; monitor explicitly nonblocking native backends.

        stable_duration_s defaults to zero. Positive dwell requires advancing
        feedback timestamps and a nonblocking native move with wait=True.
        The monitor detects timestamp stalls, not absolute sensor age. Its
        timeout defaults to 60 seconds and cannot interrupt a blocked SDK call.
        Other backends retain their own waiting/timeout behavior.
        """
        return self._execute('move_j', joints, (joint_tolerance,), kwargs,
                             stable_duration_s, monitor_hz, feedback_stall_timeout_s)

    def move_p(self, pose, *, position_tolerance: float = 0.005,
               rotation_tolerance: float = 0.05, stable_duration_s: float = 0.0,
               monitor_hz: float = 50.0, feedback_stall_timeout_s: float = 0.5,
               **kwargs) -> ExecutionRecord:
        """Record TCP arrival, using the same monitoring contract as move_j.

        The backend state timestamp does not prove TCP/joint synchronization.
        """
        return self._execute('move_p', pose,
                             (position_tolerance, rotation_tolerance), kwargs,
                             stable_duration_s, monitor_hz, feedback_stall_timeout_s)

    def _execute(self, operation, target, tolerances, kwargs,
                 stable_duration_s, monitor_hz, feedback_stall_timeout_s) -> ExecutionRecord:
        started = time.perf_counter()
        if not self._motion_lock.acquire(blocking=False):
            details = _motion_result(operation, tolerances)
            details['reason'] = 'busy'
        else:
            try:
                details = _execute_motion(
                    self, operation, target, tolerances, kwargs, started,
                    stable_duration_s, monitor_hz, feedback_stall_timeout_s)
            finally:
                self._motion_lock.release()
        finished = time.perf_counter()
        details['elapsed_s'] = finished - started
        record = ExecutionRecord(uuid.uuid4().hex, started, finished, deepcopy(details))
        self._remember(self._executions, record.execution_id, record)
        return record


def _motion_result(operation, tolerances) -> dict:
    return {
        'schema_version': 1, 'operation': operation, 'ok': False,
        'command_ok': None, 'reached': None, 'reason': None, 'error': None,
        'target': None, 'parameters': {},
        'tolerances': ({'joint_rad': tolerances[0]} if operation == 'move_j' else
                       {'position_m': tolerances[0], 'rotation_rad': tolerances[1]}),
        'before': None, 'after': None,
        'joint_error_rad': None, 'position_error_m': None, 'rotation_error_rad': None,
        'stop': {'requested': False, 'command_ok': None, 'error': None,
                 'physically_stopped': None, 'feedback': None},
        'monitoring': {'mode': 'not_started', 'samples': 0, 'stable_elapsed_s': 0.0,
                       'feedback_freshness': 'unknown'},
        'motion_detail': None,
    }


def _feedback_problem(feedback, dof):
    state = feedback['state']
    if not feedback['connected'] or (state is not None and not state['connected']):
        return 'disconnected'
    if state is None or state['q'] is None:
        return 'feedback_unavailable'
    if np.shape(state['q']) != (dof,) or not np.all(np.isfinite(state['q'])):
        return 'feedback_unavailable'
    if state['protective_stop'] or state['fault'] or state['enabled'] is False:
        return 'not_ready'
    return None


def _arrival(result, feedback, operation, target, tolerances):
    state = feedback['state']
    if state is None:
        return None
    observed = state['q'] if operation == 'move_j' else state['tcp']
    if observed is None:
        return None
    observed = np.asarray(observed, dtype=float)
    if observed.shape != target.shape or not np.all(np.isfinite(observed)):
        return None
    if operation == 'move_j':
        error = float(np.max(np.abs(observed - target)))
        result['joint_error_rad'] = error
        return bool(error <= tolerances[0])
    translation, rotation = pose_error(observed, target)
    result['position_error_m'] = translation
    result['rotation_error_rad'] = rotation
    return bool(translation <= tolerances[0] and rotation <= tolerances[1])


def _monitor_motion(platform, result, operation, target, tolerances, deadline,
                    abort_event, stable_duration_s, monitor_hz, feedback_stall_timeout_s):
    monitor = result['monitoring']
    before = result['before']['state']
    last_stamp = float(before['t'])
    clock = before['timestamp_clock']
    last_progress = time.perf_counter()
    stable_since = stable_stamp = None
    while True:
        if abort_event is not None and abort_event.is_set():
            raise Aborted('motion cancelled')
        if time.perf_counter() >= deadline:
            raise TimeoutError('motion deadline exceeded')
        feedback = platform.observe(include_images=False).feedback()
        result['after'] = feedback
        monitor['samples'] += 1
        now = time.perf_counter()
        if abort_event is not None and abort_event.is_set():
            raise Aborted('motion cancelled')
        if now >= deadline:
            raise TimeoutError('motion deadline exceeded')
        problem = _feedback_problem(feedback, platform.robot.dof)
        if problem:
            result['reason'] = problem
            return
        state = feedback['state']
        stamp = float(state['t'])
        if not math.isfinite(stamp):
            result['reason'] = 'feedback_unavailable'
            return
        if state['timestamp_clock'] != clock or stamp < last_stamp:
            result['reason'] = 'feedback_clock_changed'
            return
        if now - last_progress >= feedback_stall_timeout_s:
            result['reason'] = 'feedback_stalled'
            return
        reached = _arrival(result, feedback, operation, target, tolerances)
        result['reached'] = reached
        if reached is None:
            result['reason'] = 'feedback_unavailable'
            return
        if not reached:
            stable_since = stable_stamp = None
            monitor['stable_elapsed_s'] = 0.0
        if stamp > last_stamp:
            last_stamp, last_progress = stamp, now
            if reached:
                if stable_since is None:
                    stable_since, stable_stamp = now, stamp
                monitor['stable_elapsed_s'] = min(now - stable_since, stamp - stable_stamp)
                if monitor['stable_elapsed_s'] >= stable_duration_s:
                    result['ok'], result['reason'] = True, 'reached'
                    return
        delay = min(1.0 / monitor_hz, max(0.0, deadline - time.perf_counter()))
        if abort_event is not None:
            abort_event.wait(delay)
        else:
            time.sleep(delay)


def _execute_motion(platform, operation, target, tolerances, kwargs, started,
                    stable_duration_s, monitor_hz, feedback_stall_timeout_s) -> dict:
    robot = platform.robot
    robot._motion_detail = None
    result = _motion_result(operation, tolerances)
    dispatched = False
    try:
        result['parameters'] = {
            name: ({'is_set': value.is_set()} if name == 'abort_event' and value is not None
                   else deepcopy(value)) for name, value in kwargs.items()}
        values = np.asarray(target, dtype=float).copy()
        expected = robot.dof if operation == 'move_j' else 6
        if values.shape != (expected,) or not np.all(np.isfinite(values)):
            raise ValueError(f'target must contain {expected} finite values')
        if any(not math.isfinite(value) or value <= 0 for value in tolerances):
            raise ValueError('tolerances must be positive and finite')
        if not math.isfinite(stable_duration_s) or stable_duration_s < 0:
            raise ValueError('stable_duration_s must be nonnegative and finite')
        if any(not math.isfinite(value) or value <= 0
               for value in (monitor_hz, feedback_stall_timeout_s)):
            raise ValueError('monitor_hz and feedback_stall_timeout_s must be positive and finite')
        wait = kwargs.get('wait', True)
        monitored = (wait and operation in robot.capabilities.nonblocking_moves
                     and kwargs.get('use_native', True))
        if monitored and 1.0 / monitor_hz >= feedback_stall_timeout_s:
            raise ValueError('monitor period must be shorter than feedback_stall_timeout_s')
        if stable_duration_s and not monitored:
            raise NotSupported('stable arrival requires a nonblocking native move with wait=True')
        timeout = kwargs.get('timeout')
        timeout = 60.0 if timeout is None else float(timeout)
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError('timeout must be positive and finite')
        deadline = started + timeout
        abort_event = kwargs.get('abort_event')
        if abort_event is not None and abort_event.is_set():
            raise Aborted('motion cancelled before dispatch')
        result['monitoring'].update(
            mode='platform' if monitored else 'backend_wait' if wait else 'dispatch_only',
            stable_duration_s=stable_duration_s, monitor_hz=monitor_hz,
            feedback_stall_timeout_s=feedback_stall_timeout_s,
            timeout_s=timeout if monitored else kwargs.get('timeout'))
        result['target'] = values.copy()
        result['before'] = platform.observe(include_images=False).feedback()
        result['reason'] = _feedback_problem(result['before'], robot.dof)
        if result['reason'] is None:
            command_kwargs = dict(kwargs)
            if monitored:
                if not math.isfinite(float(result['before']['state']['t'])):
                    raise RobotUnavailable('feedback timestamp is unavailable')
                if time.perf_counter() >= deadline:
                    raise TimeoutError('motion deadline exceeded before dispatch')
                command_kwargs['wait'] = False
                command_kwargs.pop('abort_event', None)
            if abort_event is not None and abort_event.is_set():
                raise Aborted('motion cancelled before dispatch')
            dispatched = True
            result['command_ok'] = bool(getattr(robot, operation)(values, **command_kwargs))
            if monitored and result['command_ok']:
                _monitor_motion(platform, result, operation, values, tolerances, deadline,
                                abort_event, stable_duration_s, monitor_hz,
                                feedback_stall_timeout_s)
    except Exception as exc:
        result['error'] = {'type': type(exc).__name__, 'message': str(exc)}
        result['reason'] = (
            'safety_rejected' if isinstance(exc, SafetyViolation) else
            'unsupported' if isinstance(exc, NotSupported) else
            'aborted' if isinstance(exc, Aborted) else
            'timeout' if isinstance(exc, TimeoutError) else
            'unavailable' if isinstance(exc, RobotUnavailable) else
            'invalid_argument' if isinstance(exc, (ValueError, TypeError)) else
            'backend_error')
    if result['after'] is None:
        result['after'] = platform.observe(include_images=False).feedback()
    after = result['after']['state']
    if result['target'] is not None and after is not None:
        reached = _arrival(result, result['after'], operation, values, tolerances)
        if dispatched and kwargs.get('wait', True):
            result['reached'] = reached
    if result['reason'] is None:
        if not result['command_ok']:
            detail = robot._motion_detail or {}
            result['reason'] = detail.get('reason') or 'backend_failed'
        elif not result['after']['connected'] or (after is not None and not after['connected']):
            result['reason'] = 'disconnected'
        elif after is not None and (after['protective_stop'] or after['fault']):
            result['reason'] = 'not_ready'
        elif not kwargs.get('wait', True):
            result['reason'] = 'accepted'
            result['ok'] = True
        elif result['reached'] is None:
            result['reason'] = 'feedback_unavailable'
        elif result['reached']:
            result['reason'] = 'reached'
            result['ok'] = True
        else:
            result['reason'] = 'tolerance_not_met'
    if dispatched and not result['ok']:
        result['stop']['requested'] = True
        try:
            result['stop']['command_ok'] = bool(robot.stop_motion())
        except Exception as exc:
            result['stop']['error'] = {'type': type(exc).__name__, 'message': str(exc)}
        result['stop']['feedback'] = platform.observe(include_images=False).feedback()
    result['motion_detail'] = deepcopy(robot._motion_detail)
    return result