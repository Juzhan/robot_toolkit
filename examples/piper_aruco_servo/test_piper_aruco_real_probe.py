import shlex
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import piper_aruco_real_probe as probe
from robot_toolkit.robot.base import RobotState


OBJECT_POINTS = np.array([[-.03, .03, 0], [.03, .03, 0],
                          [.03, -.03, 0], [-.03, -.03, 0]])
INTRINSICS = np.array([[600., 0, 320], [0, 600., 240], [0, 0, 1]])


def detection_at(transform):
    points = OBJECT_POINTS @ transform[:3, :3].T + transform[:3, 3]
    pixels = points[:, :2] / points[:, 2:] * 600 + [320, 240]
    return {'corners': pixels, 'T_camera_marker': transform,
            'reprojection_error_px': 0.}


def test_image_jacobian_matches_camera_motion():
    points = np.array([[-.04, -.03, .3], [.04, -.03, .31],
                       [.04, .03, .32], [-.04, .03, .33]])
    normalized = points[:, :2] / points[:, 2:]
    expected = probe.image_jacobian(normalized, points[:, 2])
    for axis in range(6):
        twist = np.eye(6)[axis] * 1e-7
        moved = (points - twist[:3]) @ Rotation.from_rotvec(twist[3:]).as_matrix()
        derivative = ((moved[:, :2] / moved[:, 2:] - normalized) / 1e-7).ravel()
        np.testing.assert_allclose(expected[:, axis], derivative, atol=2e-6)


def test_camera_jacobian_includes_lever_arm():
    base = probe.pose_matrix([.2, .1, .3, .2, -.3, .1])
    extrinsic = probe.pose_matrix([.04, -.06, .02, .1, .3, -.2])
    initial = base @ extrinsic
    expected = probe.camera_jacobian(np.eye(6), base, extrinsic)
    for axis in range(6):
        twist = np.eye(6)[axis] * 1e-7
        shifted = base.copy()
        shifted[:3, 3] += twist[:3]
        shifted[:3, :3] = Rotation.from_rotvec(twist[3:]).as_matrix() @ base[:3, :3]
        camera = shifted @ extrinsic
        translation = initial[:3, :3].T @ (camera[:3, 3] - initial[:3, 3]) / 1e-7
        angular = initial[:3, :3].T @ Rotation.from_matrix(
            camera[:3, :3] @ initial[:3, :3].T).as_rotvec() / 1e-7
        np.testing.assert_allclose(expected[:, axis], np.r_[translation, angular], atol=2e-6)


@pytest.mark.parametrize('translation,angles', [
    ([.04, -.03, .3], [0, 0, 0]),
    ([0, 0, .4], [0, 0, 0]),
    ([.02, .01, .34], [.15, -.1, .12]),
])
def test_image_control_converges(translation, angles):
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_rotvec(angles).as_matrix() @ np.diag([1., -1., -1.])
    transform[:3, 3] = translation
    for step in range(1600):
        error, interaction, pixel_rms = probe.ibvs_features(
            cv2, detection_at(transform), OBJECT_POINTS, INTRINSICS, np.zeros(5), .3, 0)
        if step == 0:
            initial_error = pixel_rms
        velocity = probe.ibvs_joint_velocity(error, interaction, np.eye(6), .8, .05, .25, .03, .2)
        assert np.max(np.abs(velocity)) <= .25 + 1e-12
        assert np.linalg.norm(velocity[:3]) <= .03 + 1e-12
        assert np.linalg.norm(velocity[3:]) <= .2 + 1e-12
        rotation = Rotation.from_rotvec(velocity[3:] * .01).as_matrix()
        transform[:3, 3] = rotation.T @ (transform[:3, 3] - velocity[:3] * .01)
        transform[:3, :3] = rotation.T @ transform[:3, :3]
    assert pixel_rms < 1.
    assert pixel_rms < initial_error * .01


def test_singular_jacobian_and_invalid_depth():
    velocity = probe.ibvs_joint_velocity(np.ones(8), np.zeros((8, 6)),
                                         np.eye(6), .8, .05, .25, .03, .2)
    np.testing.assert_array_equal(velocity, np.zeros(6))
    with pytest.raises(ValueError):
        probe.image_jacobian(np.zeros((4, 2)), [.3, -.3, .3, .3])


@pytest.mark.parametrize('extra', [
    ['--position-only'], ['--distance', '0'], ['--ibvs-gain', 'nan'],
    ['--ibvs-timeout', '-1'], ['--ibvs-resume-px', '2'],
])
def test_invalid_ibvs_options(extra):
    with pytest.raises(SystemExit):
        probe.parse_args(['--control', 'ibvs', '--motion', 'servo'] + extra)


class FakeRobot:
    def __init__(self, reject=False):
        self.q = np.zeros(6)
        self.commands = []
        self.stops = 0
        self.reject = reject

    def state(self):
        return RobotState(t=time.perf_counter(), q=self.q.copy(), tcp=np.zeros(6), connected=True)

    def servo_j(self, command, dt):
        self.q = np.asarray(command).copy()
        self.commands.append(self.q.copy())
        return not self.reject

    def stop_motion(self):
        self.stops += 1
        return True


@pytest.mark.parametrize('speed', [.1, .8])
def test_ibvs_uses_shared_joint_speed(monkeypatch, speed):
    args = probe.parse_args(['--control', 'ibvs', '--motion', 'servo',
                             '--reference-speed', str(speed)])
    robot = FakeRobot()
    stream = probe.IbvsStream(robot, robot.state(), args, [-3] * 6, [3] * 6)
    assert stream.speed == speed
    np.testing.assert_allclose(stream.inputs.max_velocity, speed)
    stream.set_velocity(np.ones(6))
    np.testing.assert_allclose(stream.velocity, speed)

    def velocity(error, interaction, camera_joints, gain, damping,
                 max_joint_speed, max_linear_speed, max_angular_speed, feedforward=None):
        assert max_joint_speed == speed
        return np.zeros(6)

    monkeypatch.setattr(probe, 'ibvs_joint_velocity', velocity)
    monkeypatch.setattr(probe, 'numerical_jacobian', lambda *unused: np.eye(6))
    robot.forward_kinematics = lambda *unused: np.zeros(6)
    marker = np.diag([1., -1., -1., 1.])
    marker[2, 3] = .3
    frame = SimpleNamespace(intrinsics=INTRINSICS, distortion=np.zeros(5))
    probe.ibvs_command(args, robot, robot.state(), detection_at(marker), cv2,
                       OBJECT_POINTS, frame, np.eye(4))


def test_legacy_joint_speed_alias_cannot_create_a_second_limit():
    base = ['--control', 'ibvs', '--motion', 'servo']
    args = probe.parse_args(base + ['--ibvs-joint-speed', '.2'])
    assert args.reference_speed == .2
    matching = probe.parse_args(base + ['--ibvs-joint-speed', '.2', '--reference-speed', '.2'])
    assert matching.reference_speed == .2
    with pytest.raises(SystemExit):
        probe.parse_args(base + ['--reference-speed', '.8', '--ibvs-joint-speed', '.25'])


def test_launcher_uses_shared_motion_limits():
    command = Path(probe.__file__).with_name('servo.sh').read_text()
    argv = shlex.split(command.replace('\\\n', ' '))[2:]
    assert '--ibvs-joint-speed' not in argv
    argv[argv.index('--control') + 1] = 'ibvs'  # the launcher may be on either controller
    ibvs = probe.parse_args(argv)
    argv[argv.index('--control') + 1] = 'pbvs'
    pbvs = probe.parse_args(argv)
    for name, value in {'reference_speed': 1.2, 'reference_accel': 4.,
                        'reference_jerk': 30., 'control_hz': 100.,
                        'speed_percent': 100, 'camera_latency': .03,
                        'distance': .3, 'fps': 60, 'exposure_ms': 8.}.items():
        assert getattr(ibvs, name) == getattr(pbvs, name) == value
    assert (ibvs.ibvs_gain, ibvs.ibvs_linear_speed, ibvs.ibvs_angular_speed) == (2., .22, .8)
    assert ibvs.ibvs_feedforward == .9 and not ibvs.position_only
    # PBVS follows continuously with the stable fixed-orientation target by default.
    assert pbvs.track_while_moving and pbvs.position_only
    assert (ibvs.ibvs_stop_px, ibvs.ibvs_resume_px, ibvs.ibvs_timeout,
            ibvs.ibvs_duration, ibvs.steps) == (3., 6., .2, 0., 0)
    robot = FakeRobot()
    stream = probe.IbvsStream(robot, robot.state(), ibvs, [-3] * 6, [3] * 6)
    np.testing.assert_allclose(stream.inputs.max_velocity, 1.2)
    np.testing.assert_allclose(stream.inputs.max_acceleration, 4.)
    np.testing.assert_allclose(stream.inputs.max_jerk, 30.)


def test_velocity_watchdog_stops_without_new_images():
    args = probe.parse_args(['--control', 'ibvs', '--motion', 'servo'])
    robot = FakeRobot()
    stream = probe.IbvsStream(robot, robot.state(), args, [-3] * 6, [3] * 6)
    stream.set_velocity(np.full(6, .1))
    stream.start()
    try:
        stream.stop_event.wait(1.)
        assert stream.error is None
        assert len(robot.commands) > 30
        assert np.max(np.abs(stream.reference_v)) < 1e-6
        assert np.max(np.abs(robot.q)) < .06
    finally:
        stream.stop_event.set()
        stream.join()
    assert robot.stops == 1


def test_rejected_servo_stops_thread():
    args = probe.parse_args(['--control', 'ibvs', '--motion', 'servo'])
    robot = FakeRobot(reject=True)
    stream = probe.IbvsStream(robot, robot.state(), args, [-3] * 6, [3] * 6)
    stream.start()
    stream.join(timeout=1.)
    assert not stream.is_alive()
    assert 'rejected' in str(stream.error)
    assert robot.stops == 1


@pytest.mark.parametrize('follow', [True, False])
def test_stopped_reference_sends_once_and_resumes_smoothly(monkeypatch, follow):
    args = probe.parse_args(['--control', 'ibvs', '--motion', 'servo'])
    clock = SimpleNamespace(t=1000., tick=0)
    telemetry_records = []
    monkeypatch.setattr(probe.time, 'perf_counter', lambda: clock.t)

    class Ticks:
        def is_set(self):
            return clock.tick >= 300

        def wait(self, timeout):
            telemetry_records.append(probe.json.loads(
                probe.json.dumps(stream.telemetry(), allow_nan=False)))
            clock.tick += 1
            clock.t += .01

    class Robot(FakeRobot):
        def servo_j(self, command, dt):
            accepted = super().servo_j(command, dt)
            if not follow:
                self.q[:] = 0.
            return accepted

        def state(self):
            if clock.tick in (10, 160):
                stream.set_velocity(np.full(6, .1))
            elif clock.tick in (30, 180):
                stream.set_velocity(np.zeros(6))
            counts.append(len(self.commands))
            return super().state()

    robot = Robot()
    counts = []
    stream = probe.IbvsStream(robot, FakeRobot().state(), args, [-3] * 6, [3] * 6)
    stream.stop_event = Ticks()
    stream.run()
    assert stream.error is None
    assert len(telemetry_records) == 300
    assert {record['reference_stopped'] for record in telemetry_records} == {True, False}
    assert len(counts) == 300
    assert counts[1] == counts[9] == 1
    assert counts[120] == counts[159]
    assert counts[200] > counts[160]
    assert counts[260] == counts[299]
    assert robot.stops == 1
    increments = np.diff(np.asarray(robot.commands), axis=0)
    assert np.max(np.abs(increments)) <= args.ibvs_joint_speed / args.control_hz + 1e-12
    np.testing.assert_allclose(stream.reference_v, 0., atol=1e-9)
    telemetry = stream.telemetry()
    assert telemetry['reference_stopped']
    assert telemetry['commands_sent'] == len(robot.commands)
    assert telemetry['command_timed_out']
    assert telemetry['last_command_t'] < clock.t - .3
    np.testing.assert_allclose(telemetry['reference_a'], 0., atol=1e-9)
    np.testing.assert_allclose(telemetry['reference_q'], robot.commands[-1])
    np.testing.assert_allclose(telemetry['measured_q'], robot.q)
    assert (telemetry['joint_error_max_rad'] > .001) == (not follow)


def test_runner_loss_and_reacquisition(monkeypatch, tmp_path):
    # --steps counts marker frames only: the lost frame waits without using one.
    args = probe.parse_args(['--control', 'ibvs', '--motion', 'servo', '--steps', '3',
                             '--hold', '--log', str(tmp_path / 'ibvs.jsonl')])
    robot = FakeRobot()
    commands = []
    readings = iter([10., None, 2., 8.])

    class Stream:
        error = None
        ident = 1
        stop_event = threading.Event()

        def __init__(self, *unused):
            pass

        def start(self):
            pass

        def state_at(self, stamp):
            return robot.state()

        def set_velocity(self, velocity):
            commands.append(velocity.copy())

        def telemetry(self):
            return {'commands_sent': len(commands)}

        def join(self):
            pass

    camera = SimpleNamespace(snapshot=lambda: SimpleNamespace(
        t=time.perf_counter(), rgb=np.zeros((2, 2, 3)), intrinsics=INTRINSICS, distortion=None))
    monkeypatch.setattr(probe, 'IbvsStream', Stream)
    monkeypatch.setattr(probe, 'detect_marker', lambda *unused: next(readings))
    monkeypatch.setattr(probe, 'ibvs_command', lambda *values: (np.ones(6) * .1, values[3], np.zeros(3)))
    config = SimpleNamespace(limits_lower=[-3] * 6, limits_upper=[3] * 6)
    probe.run_ibvs(args, robot, camera, config, cv2, None, OBJECT_POINTS, np.eye(4), robot.state())
    assert len(commands) == 4
    assert np.any(commands[0])
    assert not np.any(commands[1])
    assert not np.any(commands[2])
    assert np.any(commands[3])


@pytest.mark.parametrize('execute', [False, True])
def test_ibvs_preflight_authorization_and_enable_speed_order(monkeypatch, execute):
    from robot_toolkit import robot as backend
    from robot_toolkit.camera import realsense

    calls = []

    class Robot(FakeRobot):
        connected = True

        def __init__(self, config):
            super().__init__()

        def connect_read_only(self):
            calls.append('read-only')

        def connect(self):
            pytest.fail('dry-run connected with motion enabled')

        def prepare_motion(self):
            assert execute, 'dry-run enabled motion'
            self.speed_percent = 100
            calls.append('enable')
            return True

        def set_speed_percent(self, percent):
            assert execute, 'dry-run configured motion'
            self.speed_percent = percent
            calls.append('speed')

        def servo_j(self, command, dt):
            pytest.fail('dry-run sent a motion command')

        def disconnect(self):
            calls.append('disconnect')

    class Camera:
        name = 'fake'

        def __init__(self, *unused, **kwargs):
            pass

        def open(self):
            pass

        def set_intrinsics(self, *unused):
            pass

        def close(self):
            calls.append('camera-close')

    monkeypatch.setattr(backend, 'AgxPiperRobot', Robot)
    monkeypatch.setattr(realsense, 'RealSenseCamera', Camera)
    monkeypatch.setattr(probe, 'load_calibration', lambda *unused: SimpleNamespace(
        tcp_camera=np.eye(4), intrinsics=INTRINSICS, distortion=None))
    marker = np.diag([1., -1., -1., 1.])
    marker[2, 3] = .3
    monkeypatch.setattr(probe, 'read_scene', lambda *unused, **kwargs: (
        FakeRobot().state(), None, {'T_camera_marker': marker},
        np.eye(4), np.eye(4), np.eye(4)))

    def command(*unused):
        calls.append('preflight')
        return np.zeros(6), 0., np.zeros(3)

    def run(args, robot, *unused):
        assert execute
        assert robot.speed_percent == args.speed_percent == 20
        calls.append('run')

    monkeypatch.setattr(probe, 'ibvs_command', command)
    monkeypatch.setattr(probe, 'run_ibvs', run)
    authorization = ['--execute', '--yes'] if execute else ['--dry-run']
    assert probe.main(authorization + ['--control', 'ibvs', '--motion', 'servo',
                                       '--speed-percent', '20']) == 0
    expected = ['read-only', 'preflight']
    if execute:
        expected += ['enable', 'speed', 'run']
    assert calls == expected + ['camera-close', 'disconnect']

@pytest.mark.parametrize('speed', [.1, .8])
def test_braking_margin_follows_actual_speed(monkeypatch, speed):
    """A joint near its limit still moves toward it, then stops inside the limit."""
    args = probe.parse_args(['--control', 'ibvs', '--motion', 'servo', '--reference-speed', '.8',
                             '--reference-accel', '2', '--reference-jerk', '10'])
    clock = SimpleNamespace(t=1000., tick=0)
    monkeypatch.setattr(probe.time, 'perf_counter', lambda: clock.t)

    class Ticks:
        def is_set(self):
            return clock.tick >= 400

        def wait(self, timeout):
            clock.tick += 1
            clock.t += .01

    class Robot(FakeRobot):
        def state(self):
            if stream is not None:
                stream.set_velocity(np.full(6, speed))
            return super().state()

    stream = None
    robot = Robot()
    upper = np.full(6, .3)
    stream = probe.IbvsStream(robot, robot.state(), args, [-3] * 6, upper)
    np.testing.assert_allclose(stream.braking_margin(1), .02)
    stream.stop_event = Ticks()
    stream.run()
    assert stream.error is None
    commands = np.asarray(robot.commands)
    assert np.all(commands <= upper)
    assert np.all(commands[-1] > upper - .1)
    np.testing.assert_allclose(stream.reference_v, 0., atol=1e-9)


@pytest.mark.parametrize('execute', [False, True])
def test_execute_waits_for_marker_and_follows_a_far_start(monkeypatch, execute):
    from robot_toolkit import robot as backend
    from robot_toolkit.camera import realsense

    calls = []

    class Robot(FakeRobot):
        connected = True

        def __init__(self, config):
            super().__init__()

        def connect_read_only(self):
            pass

        def prepare_motion(self):
            return True

        def set_speed_percent(self, percent):
            pass

        def disconnect(self):
            pass

    class Camera:
        name = 'fake'

        def __init__(self, *unused, **kwargs):
            pass

        def open(self):
            pass

        def set_intrinsics(self, *unused):
            pass

        def close(self):
            pass

    far = np.eye(4)  # marker seen upside down and 0.6 m away: beyond the old start gate
    far[2, 3] = .6
    scenes = iter([None, None])

    def read_scene(*unused, **kwargs):
        if next(scenes, 'seen') is None:
            calls.append('waiting')
            raise probe.MarkerNotFound('not visible')
        return (FakeRobot().state(), None, {'T_camera_marker': far},
                np.eye(4), np.eye(4), np.eye(4))

    monkeypatch.setattr(backend, 'AgxPiperRobot', Robot)
    monkeypatch.setattr(realsense, 'RealSenseCamera', Camera)
    monkeypatch.setattr(probe, 'load_calibration', lambda *unused: SimpleNamespace(
        tcp_camera=np.eye(4), intrinsics=INTRINSICS, distortion=None))
    monkeypatch.setattr(probe, 'read_scene', read_scene)
    monkeypatch.setattr(probe, 'ibvs_command', lambda *unused: (np.zeros(6), 0.))
    monkeypatch.setattr(probe, 'run_ibvs', lambda *unused: calls.append('run'))
    argv = ['--control', 'ibvs', '--motion', 'servo']
    if not execute:
        with pytest.raises(probe.MarkerNotFound):
            probe.main(['--dry-run'] + argv)
        assert calls == ['waiting']
        return
    assert probe.main(['--execute', '--yes'] + argv) == 0
    assert calls == ['waiting', 'waiting', 'run']


def test_marker_velocity_fits_a_moving_marker_and_ignores_noise():
    rng = np.random.default_rng(0)
    estimator = probe.MarkerVelocity(window=.15, deadband=.02, max_speed=.3, reset_gap=.2)
    for index in range(30):
        stamp = index / 60
        moving = estimator.update(stamp, [.2 * stamp, 0., .3] + rng.normal(0, .001, 3))
    assert abs(moving[0] - .18) < .03 and abs(moving[1]) < .03
    estimator.reset()
    for index in range(30):
        still = estimator.update(index / 60, [0., 0., .3] + rng.normal(0, .001, 3))
    assert not np.any(still)
    assert not np.any(estimator.update(10., [1., 0., 0.]))  # gap restarts the window


def test_feedforward_moves_the_camera_with_the_marker():
    interaction = probe.image_jacobian(np.zeros((4, 2)) + [[-.1, -.1], [.1, -.1], [.1, .1], [-.1, .1]],
                                       np.full(4, .3))
    feedforward = np.array([.1, 0., 0., 0., 0., 0.])
    velocity = probe.ibvs_joint_velocity(np.zeros(8), interaction, np.eye(6), 1., .05,
                                         3., .3, .6, feedforward)
    assert np.allclose(velocity, feedforward, atol=1e-2)
    capped = probe.ibvs_joint_velocity(np.zeros(8), interaction, np.eye(6), 1., .05,
                                       3., .05, .6, feedforward)
    assert np.isclose(np.linalg.norm(capped[:3]), .05)


def test_pbvs_defaults_and_legacy_flags():
    pbvs = probe.parse_args(['--motion', 'servo'])
    assert pbvs.track_while_moving and pbvs.position_only
    legacy = probe.parse_args(['--motion', 'servo', '--settle-before-update',
                               '--pbvs-orientation', 'marker', '--track-while-moving'])
    assert not legacy.track_while_moving and not legacy.position_only
    ibvs = probe.parse_args(['--motion', 'servo', '--control', 'ibvs'])
    assert not ibvs.position_only


class _Done(Exception):
    pass


def _run_pbvs_sequence(monkeypatch, markers, extra=(), collision=None):
    """Drive run_servo with base-frame marker positions (None = not detected)."""
    args = probe.parse_args(['--motion', 'servo', '--execute', '--yes', '--steps', '0',
                             '--hold', '--distance', '.3', *extra])
    start = np.array([0., 0., .3, np.pi, 0., 0.])

    class Robot:
        forward_kinematics = staticmethod(lambda q: np.asarray(q, float))

    class Stream:
        error = None

        def __init__(self, robot, q0, *unused):
            self.goal = np.asarray(q0, float).copy()
            self.goals = [self.goal.copy()]
            self.stop_event = SimpleNamespace(set=lambda: None)

        @property
        def latest(self):
            return SimpleNamespace(q=self.goal.copy())

        def start(self):
            pass

        def join(self, timeout=None):
            pass

        def state_at(self, stamp):
            return self.latest

        def still(self):
            return True

        collision_brakes = 0

        def get_goal(self):
            return self.goal.copy()

        reference_position = get_goal

        def set_goal(self, goal):
            self.goal = np.asarray(goal, float).copy()
            self.goals.append(self.goal.copy())

    streams = []
    monkeypatch.setattr(probe, 'ServoStream',
                        lambda *values: streams.append(Stream(*values)) or streams[-1])
    sequence = iter(enumerate(markers))
    current = {}

    class Camera:
        def snapshot(self):
            try:
                index, marker = next(sequence)
            except StopIteration:
                raise _Done from None
            current['marker'] = marker
            return SimpleNamespace(t=index / 60, rgb=np.zeros((2, 2, 3)),
                                   intrinsics=INTRINSICS, distortion=np.zeros(5))

    def detect(*unused):
        marker = current['marker']
        if marker is None:
            return None
        base_camera = probe.pose_matrix(streams[-1].goal)
        base_marker = np.eye(4)
        base_marker[:3, 3] = marker
        return {'T_camera_marker': np.linalg.inv(base_camera) @ base_marker,
                'pixel_error_px': 0., 'reprojection_error_px': 0.}

    monkeypatch.setattr(probe, 'detect_marker', detect)
    config = SimpleNamespace(limits_lower=[-5] * 6, limits_upper=[5] * 6)
    args.collision = collision
    with pytest.raises(_Done):
        probe.run_servo(args, Robot(), Camera(), config, None, None, OBJECT_POINTS,
                        np.eye(4), SimpleNamespace(q=start), .3)
    return np.array(streams[-1].goals)[:, :3]


def test_pbvs_ignores_wrong_detections_and_needs_consistent_reacquire(monkeypatch):
    bogus = [.40, .10, -.20]
    markers = ([[.02, 0., 0.]] * 30 + [bogus] + [[.03, 0., 0.]] * 30
               + [None] * 20 + [bogus, None, [-.3, .2, .1], None]   # lost, isolated outliers
               + [[.05, 0., 0.]] * 60)
    goals = _run_pbvs_sequence(monkeypatch, markers)
    assert np.max(np.abs(goals[:, :2])) < .06 and np.allclose(goals[:, 2], .3, atol=.01)
    np.testing.assert_allclose(goals[-1], [.05, 0., .3], atol=.006)


def test_pbvs_target_is_clamped_to_reach(monkeypatch):
    goals = _run_pbvs_sequence(monkeypatch, [[1., 0., 0.]] * 120, ['--max-reach-m', '.2'])
    assert np.max(np.linalg.norm(goals - [0., 0., .3], axis=1)) <= .2 + 1e-6
    assert np.linalg.norm(goals[-1] - [0., 0., .3]) > .19


def test_pbvs_lead_stops_with_the_hand(monkeypatch):
    ramp = [[.005 * i, 0., 0.] for i in range(21)]   # 0.3 m/s for 1/3 s, then stop
    goals = _run_pbvs_sequence(monkeypatch, ramp + [ramp[-1]] * 90, ['--lead', '.08'])
    assert np.max(goals[:, 0]) <= .1 + .016
    np.testing.assert_allclose(goals[-1], [.1, 0., .3], atol=.006)


def test_pbvs_goal_stops_at_a_virtual_wall(monkeypatch):
    from robot_toolkit.core.collision import CollisionGuard

    class Wall(CollisionGuard):
        """Fake FK (q[:3] is the TCP): a wall face at x = 0.10 m."""

        def __init__(self):
            self.margin_m, self.step_rad = .02, .002

        def clearance(self, q):
            return .10 - q[0], 'link6', 'wall'

        def clearances(self, qs):
            return .10 - np.asarray(qs)[:, 0]

    ramp = [[.005 * i, 0., 0.] for i in range(41)]     # hand to x = 0.2 m, past the wall
    goals = _run_pbvs_sequence(monkeypatch, ramp + [ramp[-1]] * 60, collision=Wall())
    assert np.max(goals[:, 0]) <= .10 - .02 - .015 + 1e-9
    assert goals[-1, 0] > .10 - .02 - .015 - .006       # slides right up to the edge
    back = [[.2 - .005 * i, 0., 0.] for i in range(41)] # hand comes back: arm follows
    goals = _run_pbvs_sequence(monkeypatch, ramp + back + [back[-1]] * 60, collision=Wall())
    np.testing.assert_allclose(goals[-1], [0., 0., .3], atol=.006)


def test_obstacle_flags():
    with pytest.raises(SystemExit):
        probe.parse_args(['--watch-clearance'])
    with pytest.raises(SystemExit):
        probe.parse_args(['--watch-clearance', '--obstacles', 'x.json', '--execute'])
    args = probe.parse_args(['--watch-clearance', '--obstacles', 'x.json'])
    assert args.collision is None and args.collision_clearance == .015


def test_servo_stream_never_commands_inside_the_margin():
    pytest.importorskip('ruckig')
    from robot_toolkit.core.collision import CollisionGuard

    class Wall(CollisionGuard):
        def __init__(self):
            self.margin_m, self.step_rad = .02, .01

        def clearance(self, q):
            return .5 - q[0], 'link6', 'wall'

        def clearances(self, qs):
            return .5 - np.asarray(qs)[:, 0]   # joint 1 "position" against a wall at 0.5

    sent = []

    class Robot:
        def state(self):
            q = sent[-1] if sent else np.zeros(6)
            return SimpleNamespace(q=q, tcp=np.zeros(6), t=time.perf_counter())

        def servo_j(self, q, dt):
            sent.append(np.asarray(q, float).copy())
            return True

    stream = probe.ServoStream(Robot(), np.zeros(6), 100, 1.2, 4, 30, [-5] * 6, [5] * 6,
                               Wall(), .015)
    stream.set_goal([2., 0, 0, 0, 0, 0])               # ungated goal straight through the wall
    stream.start()
    stream.join(timeout=3.)
    stream.stop_event.set()
    stream.join(timeout=1.)
    assert sent and max(q[0] for q in sent) <= .5 - .02     # hard floor never crossed
    # Even an ungated full-speed goal is braked to a stop just short of the clearance band.
    assert stream.error is None and stream.collision_brakes == 1
    assert .5 - .035 - .02 < sent[-1][0] <= .5 - .035


def test_ibvs_stream_brakes_before_a_virtual_wall(monkeypatch):
    from robot_toolkit.core.collision import CollisionGuard

    class Wall(CollisionGuard):
        def __init__(self):
            self.margin_m, self.step_rad = .02, .01

        def clearance(self, q):
            return .5 - q[0], 'link6', 'wall'

        def clearances(self, qs):
            return .5 - np.asarray(qs)[:, 0]

    args = probe.parse_args(['--control', 'ibvs', '--motion', 'servo', '--reference-speed', '.8',
                             '--reference-accel', '2', '--reference-jerk', '10'])
    args.collision = Wall()
    clock = SimpleNamespace(t=1000., tick=0)
    monkeypatch.setattr(probe.time, 'perf_counter', lambda: clock.t)

    class Ticks:
        def is_set(self):
            return clock.tick >= 400

        def wait(self, timeout):
            clock.tick += 1
            clock.t += .01

    class Robot(FakeRobot):
        def state(self):
            if stream is not None:
                stream.set_velocity(np.r_[.8, np.zeros(5)])   # keep pushing into the wall
            return super().state()

    stream = None
    robot = Robot()
    stream = probe.IbvsStream(robot, robot.state(), args, [-3] * 6, [3] * 6)
    stream.stop_event = Ticks()
    stream.run()
    assert stream.error is None and stream.collision_brakes > 0
    commands = np.asarray(robot.commands)
    # Slows down approaching the clearance band and creeps up to it, never past.
    assert commands[:, 0].max() <= .5 - .02 - .015 + 1e-6 and commands[-1, 0] > .45
