"""Offline platform composition, evidence, and history contracts."""

import json
import threading
import time
from dataclasses import replace

import numpy as np
import pytest

from robot_toolkit import RobotPlatform, with_reports
from robot_toolkit.camera import CameraHub, MockCamera
from robot_toolkit.core.conventions import RobotSpec
from robot_toolkit.errors import RobotUnavailable
from robot_toolkit.gripper.mock import MockGripper
from robot_toolkit.robot import MockRobot
from robot_toolkit.robot.base import RobotState


def test_nonblocking_motion_requires_explicit_backend_contract():
    from robot_toolkit.robot.agx_piper import AgxPiperConfig, AgxPiperRobot

    assert MockRobot().capabilities.nonblocking_moves == ()
    config = AgxPiperConfig(ik_backend='none')
    assert AgxPiperRobot(config, mock=False).capabilities.nonblocking_moves == ('move_j', 'move_p')
    assert AgxPiperRobot(config, mock=True).capabilities.nonblocking_moves == ()


def test_platform_borrows_devices_without_implicit_lifecycle(monkeypatch):
    robot = MockRobot(RobotSpec.generic(1))
    robot.disconnect()
    gripper = MockGripper()
    camera = MockCamera('wrist', 32, 24)
    cameras = CameraHub([camera])

    def forbidden(*args, **kwargs):
        raise AssertionError('implicit device lifecycle or read')

    for device, names in [(robot, ('connect', 'disconnect', 'state', 'prepare_motion')),
                          (gripper, ('connect', 'disconnect', 'state')),
                          (camera, ('open', 'close', 'read'))]:
        for name in names:
            monkeypatch.setattr(device, name, forbidden)
    platform = RobotPlatform(robot, gripper=gripper, cameras=cameras)
    assert platform.robot is robot and platform.gripper is gripper
    assert platform.cameras is cameras
    with_reports(platform)
    with pytest.raises(RobotUnavailable, match='open cameras explicitly'):
        platform.observe()
    sample = platform.observe(include_images=False)
    assert sample.state is None and not sample.connected
    assert sample.gripper is None and 'gripper' in sample.errors
    platform.clear_history()


def test_execution_is_numeric_detached_and_reported_without_rereading(monkeypatch):
    robot = MockRobot(RobotSpec.generic(1))
    platform = RobotPlatform(robot)
    reads, calls = [], []
    state = robot.state

    def read():
        reads.append(True)
        return state()

    def move(target, **kwargs):
        calls.append(target.copy())
        return True

    monkeypatch.setattr(robot, 'state', read)
    monkeypatch.setattr(robot, 'move_j', move)
    result = with_reports(platform).move_j([0.0])
    assert len(calls) == 1 and len(reads) == 2
    record = platform.execution(result['execution_id'])
    assert record.ok and isinstance(record.details['target'], np.ndarray)
    assert isinstance(record.details['after']['state']['q'], np.ndarray)
    assert record.started_at <= record.finished_at
    before_id = result['before']['observation_id']
    assert before_id in platform.observation_ids
    assert np.array_equal(platform.observation(before_id).state.q, [0.0])
    json.dumps(record.to_report(), allow_nan=False)
    assert len(calls) == 1 and len(reads) == 2
    record.details['target'][0] = 99
    result['after']['state']['q'][0] = 99
    retained = platform.execution(record.execution_id)
    assert retained.details['target'][0] == retained.details['after']['state']['q'][0] == 0


def test_bounded_history_and_clear_do_not_reset_or_close_robot():
    robot = MockRobot(RobotSpec.generic(1))
    platform = RobotPlatform(robot, history_size=1)
    first = platform.move_j([0.0])
    second = platform.move_j([0.0])
    assert platform.execution_ids == (second.execution_id,)
    assert len(platform.observation_ids) == 1
    with pytest.raises(KeyError):
        platform.execution(first.execution_id)
    assert second.details['before']['state']['q'][0] == 0
    platform.clear_history()
    assert not platform.execution_ids and not platform.observation_ids
    assert robot.connected
    disabled = RobotPlatform(robot, history_size=0)
    assert disabled.move_j([0.0]).ok
    assert not disabled.execution_ids and not disabled.observation_ids


@pytest.mark.parametrize('history_size', [-1, 1.5, True])
def test_invalid_history_size(history_size):
    with pytest.raises(ValueError, match='history_size'):
        RobotPlatform(MockRobot(), history_size=history_size)


def test_observation_pairs_images_and_keeps_only_metadata_in_history():
    robot = MockRobot(RobotSpec.generic(1))
    gripper = MockGripper().connect()
    camera = MockCamera('wrist', 32, 24, depth=True).open()
    camera.set_extrinsics(np.eye(4))
    platform = RobotPlatform(robot, gripper=gripper, cameras=CameraHub([camera]),
                             sync_max_skew_s=1.0)
    observation = platform.observe()
    assert observation.synchronization['matched']
    assert not observation.synchronization['exposure_synchronized']
    assert observation.frames['wrist'].rgb.shape == (24, 32, 3)
    assert observation.frames['wrist'].depth.dtype == np.uint16
    assert observation.gripper.width == pytest.approx(gripper.max_width)
    assert observation.feedback()['gripper_synchronization'] == 'not_guaranteed'
    observation.state.q[0] = 99
    observation.camera_metadata['wrist']['intrinsics'][0, 0] = 99
    stored = platform.observation(observation.observation_id)
    assert stored.frames == {}
    assert stored.state.q[0] == 0
    assert stored.camera_metadata['wrist']['intrinsics'][0, 0] != 99
    assert np.array_equal(stored.camera_metadata['wrist']['T_world_camera'], np.eye(4))
    platform.clear_history()
    assert robot.connected and gripper.connected and camera.opened


def test_stale_images_are_rejected_or_retained_as_rejection_evidence(monkeypatch):
    camera = MockCamera('wrist', 32, 24).open()
    frame = camera.snapshot()
    frame.t = time.perf_counter() - 10
    monkeypatch.setattr(camera, 'read', lambda: frame)
    platform = RobotPlatform(MockRobot(), cameras=CameraHub([camera]))
    with pytest.raises(RobotUnavailable, match='stale_image'):
        platform.observe()
    observation = platform.observe(strict=False)
    assert observation.state is None
    assert not observation.synchronization['matched']
    assert observation.synchronization['cameras']['wrist']['reason'] == 'stale_image'


def test_motion_does_not_capture_images_or_require_open_cameras(monkeypatch):
    camera = MockCamera('wrist', 32, 24)

    def forbidden():
        raise AssertionError('motion must not acquire images')

    monkeypatch.setattr(camera, 'snapshot', forbidden)
    platform = RobotPlatform(MockRobot(RobotSpec.generic(1)), cameras=CameraHub([camera]))
    assert platform.move_j([0.0]).ok
    assert not camera.opened


def test_aborted_and_invalid_attempts_are_recorded():
    robot = MockRobot(RobotSpec.generic(1))
    platform = RobotPlatform(robot)
    event = threading.Event()
    event.set()
    aborted = platform.move_j([0.1], abort_event=event)
    assert aborted.details['reason'] == 'aborted'
    assert aborted.details['parameters']['abort_event'] == {'is_set': True}
    assert aborted.execution_id in platform.execution_ids
    invalid = platform.move_j([np.nan])
    assert invalid.details['reason'] == 'invalid_argument'
    assert invalid.execution_id in platform.execution_ids
    assert robot.commands == []


def test_reports_require_platform_and_raw_calls_are_not_recorded():
    robot = MockRobot(RobotSpec.generic(1))
    platform = RobotPlatform(robot)
    with pytest.raises(TypeError, match='RobotPlatform'):
        with_reports(robot)
    assert robot.move_j([0.0]) is True
    assert platform.execution_ids == ()


@pytest.fixture
def monitored_motion(monkeypatch):
    import robot_toolkit.runtime.robot_platform as platform_module

    class Clock:
        now = 0.0

        def perf_counter(self):
            return self.now

        def sleep(self, duration):
            self.now += duration

    clock = Clock()
    monkeypatch.setattr(platform_module, 'time', clock)

    class NativeRobot(MockRobot):
        def __init__(self):
            super().__init__(RobotSpec.generic(1), clock=clock.perf_counter)
            self.capabilities = replace(self.capabilities, nonblocking_moves=('move_j', 'move_p'))
            self.calls = []
            self.stops = 0
            self.stop_result = True
            self.sample = lambda: {}

        def state(self):
            values = dict(t=1.0 + clock.now, q=[1.0] if self.calls else [0.0],
                          tcp=[1.0, 0, 0, 0, 0, 0] if self.calls else [0.0] * 6,
                          timestamp_clock='controller_elapsed', connected=True)
            if self.calls:
                values.update(self.sample())
            return RobotState(**values)

        def move_j(self, target, **kwargs):
            self.calls.append((target.copy(), kwargs))
            self._motion_detail = {'backend': 'test'}
            return True

        def move_p(self, target, **kwargs):
            return self.move_j(target, **kwargs)

        def stop_motion(self):
            self.stops += 1
            if isinstance(self.stop_result, Exception):
                raise self.stop_result
            return self.stop_result

    robot = NativeRobot()
    return RobotPlatform(robot), robot, clock


@pytest.mark.parametrize('operation,target', [('move_j', [1.0]), ('move_p', [1.0, 0, 0, 0, 0, 0])])
def test_platform_monitors_native_arrival_without_default_dwell(monitored_motion, operation, target):
    platform, robot, clock = monitored_motion
    result = getattr(platform, operation)(target)
    assert result.ok and result.details['reason'] == 'reached'
    assert clock.now == pytest.approx(0.02)
    assert len(robot.calls) == 1 and robot.calls[0][1]['wait'] is False
    assert result.details['monitoring']['mode'] == 'platform'
    assert result.details['monitoring']['stable_duration_s'] == 0
    assert result.details['monitoring']['feedback_freshness'] == 'unknown'
    assert robot.stops == 0
    json.dumps(result.to_report(), allow_nan=False)


@pytest.mark.parametrize('operation,target', [('move_j', [1.0]), ('move_p', [1.0, 0, 0, 0, 0, 0])])
def test_arrival_dwell_restarts_after_leaving_tolerance(monitored_motion, operation, target):
    platform, robot, clock = monitored_motion

    def sample():
        position = 0.5 if 0.04 <= clock.now < 0.08 else 1.0
        return {'q': [position], 'tcp': [position, 0, 0, 0, 0, 0]}

    robot.sample = sample
    result = getattr(platform, operation)(target, timeout=1, stable_duration_s=0.1)
    assert result.ok and clock.now >= 0.18
    assert result.details['monitoring']['stable_elapsed_s'] >= 0.1
    assert len(robot.calls) == 1


@pytest.mark.parametrize('stop_result', [True, False, RuntimeError('stop failed')])
def test_monitor_timeout_records_stop_without_claiming_standstill(monitored_motion, stop_result):
    platform, robot, clock = monitored_motion
    robot.sample = lambda: {'q': [0.4]}
    robot.stop_result = stop_result
    result = platform.move_j([1.0], timeout=0.1)
    assert not result.ok and result.details['reason'] == 'timeout'
    assert clock.now == pytest.approx(0.1)
    assert result.details['reached'] is False
    assert robot.stops == 1
    stop = result.details['stop']
    assert stop['requested'] and stop['physically_stopped'] is None
    if isinstance(stop_result, Exception):
        assert stop['command_ok'] is None and stop['error']['message'] == 'stop failed'
    else:
        assert stop['command_ok'] is stop_result


@pytest.mark.parametrize('stamp', [1.0, 1.01])
def test_repeated_feedback_cannot_complete_stable_arrival(monitored_motion, stamp):
    platform, robot, clock = monitored_motion
    robot.sample = lambda: {'t': stamp}
    result = platform.move_j([1.0], stable_duration_s=0.08,
                             feedback_stall_timeout_s=0.1, timeout=1)
    assert not result.ok and result.details['reason'] == 'feedback_stalled'
    assert result.details['monitoring']['stable_elapsed_s'] == 0
    assert clock.now < 0.2 and robot.stops == 1


@pytest.mark.parametrize('sample,reason', [
    ({'q': None}, 'feedback_unavailable'),
    ({'q': [np.nan]}, 'feedback_unavailable'),
    ({'q': [1.0, 2.0]}, 'feedback_unavailable'),
    ({'fault': True}, 'not_ready'),
    ({'enabled': False}, 'not_ready'),
    ({'protective_stop': True}, 'not_ready'),
    ({'connected': False}, 'disconnected'),
    ({'t': 0.5}, 'feedback_clock_changed'),
    ({'timestamp_clock': 'different'}, 'feedback_clock_changed'),
    ({'t': np.nan}, 'feedback_unavailable'),
])
def test_monitor_rejects_bad_feedback(monitored_motion, sample, reason):
    platform, robot, _ = monitored_motion
    robot.sample = lambda: sample
    result = platform.move_j([1.0], timeout=1)
    assert not result.ok and result.details['reason'] == reason
    assert robot.stops == 1
    json.dumps(result.to_report(), allow_nan=False)


def test_monitor_cancellation_uses_platform_not_backend_event(monitored_motion):
    platform, robot, clock = monitored_motion
    robot.sample = lambda: {'q': [0.4]}

    class Cancellation:
        def is_set(self):
            return clock.now >= 0.06

        def wait(self, duration):
            clock.sleep(duration)

    result = platform.move_j([1.0], abort_event=Cancellation(), timeout=1)
    assert result.details['reason'] == 'aborted' and not result.ok
    assert clock.now == pytest.approx(0.06)
    assert 'abort_event' not in robot.calls[0][1]
    assert robot.stops == 1


def test_precancelled_native_motion_never_dispatches(monitored_motion):
    platform, robot, _ = monitored_motion
    event = threading.Event()
    event.set()
    result = platform.move_p([1.0, 0, 0, 0, 0, 0], abort_event=event)
    assert result.details['reason'] == 'aborted'
    assert robot.calls == [] and robot.stops == 0


def test_busy_request_does_not_read_dispatch_or_stop_active_motion(monitored_motion):
    platform, robot, _ = monitored_motion
    competing = []

    def sample():
        if not competing:
            competing.append(platform.move_j([2.0]))
        return {}

    robot.sample = sample
    result = platform.move_j([1.0])
    assert result.ok and len(robot.calls) == 1 and robot.stops == 0
    rejected = competing[0]
    assert rejected.details['reason'] == 'busy' and not rejected.ok
    assert rejected.details['before'] is rejected.details['after'] is None
    assert result.details['motion_detail'] == {'backend': 'test'}
    assert platform.execution(rejected.execution_id).details['reason'] == 'busy'
    assert platform.move_j([1.0]).ok


@pytest.mark.parametrize('options', [
    {'stable_duration_s': -1}, {'stable_duration_s': np.nan},
    {'monitor_hz': 0}, {'monitor_hz': np.inf},
    {'monitor_hz': 1},
    {'feedback_stall_timeout_s': 0}, {'feedback_stall_timeout_s': np.nan},
    {'timeout': 0}, {'timeout': np.inf},
])
def test_invalid_monitor_settings_fail_before_dispatch(monitored_motion, options):
    platform, robot, _ = monitored_motion
    result = platform.move_j([1.0], **options)
    assert result.details['reason'] == 'invalid_argument'
    assert robot.calls == [] and robot.stops == 0


def test_stable_arrival_never_silently_switches_host_stream_to_nonblocking():
    robot = MockRobot(RobotSpec.generic(1))
    platform = RobotPlatform(robot)
    result = platform.move_j([1.0], stable_duration_s=0.1)
    assert result.details['reason'] == 'unsupported' and robot.commands == []
    assert platform.move_j([0.0]).details['monitoring']['mode'] == 'backend_wait'


@pytest.mark.parametrize('options', [{'wait': False}, {'use_native': False}])
def test_stable_arrival_rejects_incompatible_paths_before_dispatch(monitored_motion, options):
    platform, robot, _ = monitored_motion
    result = platform.move_j([1.0], stable_duration_s=0.1, **options)
    assert result.details['reason'] == 'unsupported' and robot.calls == []


def test_nonwaiting_native_call_remains_acceptance_only(monitored_motion):
    platform, robot, clock = monitored_motion
    result = with_reports(platform).move_j([1.0], wait=False)
    assert result['ok'] and result['reason'] == 'accepted'
    assert result['reached'] is None and result['monitoring']['mode'] == 'dispatch_only'
    assert clock.now == 0 and len(robot.calls) == 1


def test_motion_lock_released_after_failure(monitored_motion):
    platform, robot, _ = monitored_motion
    robot.sample = lambda: {'fault': True}
    assert platform.move_j([1.0]).details['reason'] == 'not_ready'
    robot.sample = lambda: {}
    assert platform.move_j([1.0]).ok


def test_dispatch_time_consumes_monitor_deadline(monitored_motion, monkeypatch):
    platform, robot, clock = monitored_motion
    original = robot.move_j

    def delayed_dispatch(target, **kwargs):
        clock.sleep(0.2)
        return original(target, **kwargs)

    monkeypatch.setattr(robot, 'move_j', delayed_dispatch)
    result = platform.move_j([1.0], timeout=0.1)
    assert not result.ok and result.details['reason'] == 'timeout'
    assert robot.stops == 1 and len(robot.calls) == 1
    assert result.details['monitoring']['samples'] == 0


def test_stable_arrival_requires_host_time_as_well_as_timestamp_progress(monitored_motion):
    platform, robot, clock = monitored_motion
    robot.sample = lambda: {'t': 1000 * clock.now + 1}
    result = platform.move_j([1.0], stable_duration_s=0.1, timeout=1)
    assert result.ok and clock.now >= 0.12


@pytest.mark.parametrize('bad_sample', [{'tcp': None}, {'tcp': [np.nan] * 6}])
def test_pose_monitor_requires_actual_pose_feedback(monitored_motion, bad_sample):
    platform, robot, _ = monitored_motion
    robot.sample = lambda: bad_sample
    result = platform.move_p([1.0, 0, 0, 0, 0, 0])
    assert result.details['reason'] == 'feedback_unavailable'
    assert not result.ok and robot.stops == 1


def test_overlapping_threads_reject_motion_without_blocking_owner(monitored_motion, monkeypatch):
    platform, robot, _ = monitored_motion
    entered, release = threading.Event(), threading.Event()
    completed = []
    original = robot.move_j

    def dispatch(target, **kwargs):
        entered.set()
        assert release.wait(2)
        return original(target, **kwargs)

    monkeypatch.setattr(robot, 'move_j', dispatch)
    owner = threading.Thread(target=lambda: completed.append(platform.move_j([1.0])))
    owner.start()
    try:
        assert entered.wait(2)
        rejected = platform.move_j([2.0])
        assert rejected.details['reason'] == 'busy'
        assert robot.stops == 0
    finally:
        release.set()
        owner.join(2)
    assert not owner.is_alive() and completed[0].ok
    assert len(robot.calls) == 1


@pytest.mark.parametrize('scenario,expected', [
    ('arrival', 'reached'), ('stalled_actuator', 'timeout'), ('cancel', 'aborted'),
])
def test_native_monitor_with_independently_stepped_physics(scenario, expected):
    mujoco = pytest.importorskip('mujoco')
    from robot_toolkit.robot.mujoco import MujocoRobot

    model = mujoco.MjModel.from_xml_string('''
        <mujoco><option gravity="0 0 0" timestep="0.002"/>
          <worldbody><body><joint name="joint" type="hinge" damping="1"/>
            <geom type="sphere" size="0.1" mass="1"/>
            <site name="tcp" pos="0.1 0 0"/>
          </body></worldbody>
          <actuator><position name="motor" joint="joint" kp="100"/></actuator>
        </mujoco>''')
    cancel, shutdown = threading.Event(), threading.Event()
    physics_lock = threading.RLock()

    class AutonomousSimulation(MujocoRobot):
        def __init__(self):
            super().__init__(model, joint_names=['joint'], arm_actuators=['motor'])
            self.capabilities = replace(self.capabilities, nonblocking_moves=('move_j',))
            self.submitted_at = None
            self.stops = 0
            self.dispatches = 0

        def state(self):
            with physics_lock:
                sample = super().state()
                sample.t = float(self.data.time)
                sample.timestamp_clock = 'simulation'
                return sample

        def move_j(self, joints, wait=True, **kwargs):
            assert wait is False
            with physics_lock:
                self.data.ctrl[self.actuator_ids] = joints
                self.submitted_at = float(self.data.time)
                self.dispatches += 1
            return True

        def stop_motion(self):
            with physics_lock:
                self.data.ctrl[self.actuator_ids] = self.data.qpos[self.qpos_indices]
                self.stops += 1
            return True

    robot = AutonomousSimulation().connect()
    if scenario == 'stalled_actuator':
        robot.model.actuator_gainprm[:] = 0
        robot.model.actuator_biasprm[:] = 0
    physics_errors = []

    def advance():
        try:
            while not shutdown.wait(0.002):
                with physics_lock:
                    robot._step(0.002)
                    if (scenario == 'cancel' and robot.submitted_at is not None
                            and robot.data.time - robot.submitted_at >= 0.04):
                        cancel.set()
        except Exception as exc:
            physics_errors.append(exc)
            cancel.set()

    worker = threading.Thread(target=advance)
    worker.start()
    try:
        result = RobotPlatform(robot).move_j(
            [0.1], stable_duration_s=0.08, joint_tolerance=0.005,
            abort_event=cancel, timeout=0.2 if scenario == 'stalled_actuator' else 2.0)
        assert result.details['reason'] == expected, result.to_report()
        assert result.ok is (scenario == 'arrival')
        assert result.details['monitoring']['mode'] == 'platform'
        assert result.details['after']['state']['extra']['simulation_time'] > 0
        assert robot.dispatches == 1
        assert robot.stops == (0 if scenario == 'arrival' else 1)
        if scenario == 'arrival':
            assert result.details['joint_error_rad'] <= 0.005
            assert result.details['monitoring']['stable_elapsed_s'] >= 0.08
        else:
            assert result.details['stop']['physically_stopped'] is None
        json.dumps(result.to_report(), allow_nan=False)
    finally:
        shutdown.set()
        worker.join(2)
        robot.disconnect()
    assert not worker.is_alive() and not physics_errors