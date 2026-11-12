"""Offline tests for the standalone agent env RPC protocol and Piper backend."""

from __future__ import annotations

import numpy as np
import pytest

from robot_toolkit import RobotPlatform
from robot_toolkit.agent_rpc import (EnvClient, EnvService, PiperEnvConfig,
                                     PiperRealBackend, RpcClient, RpcError,
                                     RpcServer, make_client, wait_for_ready)
from robot_toolkit.agent_rpc.codec import dumps, loads
from robot_toolkit.camera import CameraHub, MockCamera
from robot_toolkit.core.conventions import RobotSpec
from robot_toolkit.errors import NotSupported, SafetyViolation
from robot_toolkit.gripper.mock import MockGripper
from robot_toolkit.robot import MockRobot


class FakeBackend:
    def __init__(self):
        self.moved = []

    def get_env_meta(self):
        return {'robot': 'fake', 'protocol_version': 1, 'action_dim': 7}

    def reset(self):
        return {'ok': True, 'n': 1}

    def get_robot_state(self):
        return {'ok': True, 'q': np.arange(6, dtype=np.float32)}

    def get_observation(self):
        return {'ok': True, 'main_images': np.zeros((8, 8, 3), dtype=np.uint8)}

    def get_camera_meta(self):
        return {'ok': True, 'cameras': {}}

    def move_delta(self, delta_xyz):
        self.moved.append(np.asarray(delta_xyz, dtype=np.float32))
        return {'ok': True, 'applied': np.asarray(delta_xyz, dtype=np.float32)}

    def rotate_delta(self, delta_rpy):
        return {'ok': True, 'applied': np.asarray(delta_rpy, dtype=np.float32)}

    def set_gripper(self, *, open: bool):
        return {'ok': True, 'open': bool(open)}

    def get_task_language(self):
        return 'fake task'

    def render_camera(self, camera_name):
        return np.full((4, 4, 3), 7, dtype=np.uint8)


class FailingBackend(FakeBackend):
    def move_delta(self, delta_xyz):
        raise ValueError('motion rejected by test')


def test_codec_roundtrip_numpy_and_dataclasses():
    from robot_toolkit.gripper.base import GripperState

    value = {
        'image': np.arange(6, dtype=np.uint8).reshape(2, 3),
        'scalar': np.float32(1.5),
        'nested': [np.int32(2), {'x': np.array([1.0, 2.0])}],
        'gripper': GripperState(width=0.04, target=0.04),
    }
    restored = loads(dumps(value))
    assert restored['image'].dtype == np.uint8
    np.testing.assert_array_equal(restored['image'], value['image'])
    assert restored['scalar'] == np.float32(1.5)
    assert restored['nested'][0] == 2
    np.testing.assert_array_equal(restored['nested'][1]['x'], [1.0, 2.0])
    assert restored['gripper']['width'] == 0.04


def test_http_envelope_roundtrip_and_shutdown():
    service = EnvService(FakeBackend())
    host, port = service.start()
    client = EnvClient.connect(f'http://{host}:{port}')
    assert client.meta['robot'] == 'fake'
    assert client.get_task_language() == 'fake task'
    observation = client.get_observation()
    assert observation['main_images'].shape == (8, 8, 3)
    result = client.move_delta([0.01, 0.0, 0.0])
    np.testing.assert_allclose(result['applied'], [0.01, 0.0, 0.0])
    assert client.render_camera('wrist').shape == (4, 4, 3)
    client.close()
    service.stop()


def test_error_is_reported_as_rpc_error():
    service = EnvService(FailingBackend())
    host, port = service.start()
    client = EnvClient.connect(f'http://{host}:{port}')
    with pytest.raises(RpcError, match='motion rejected by test'):
        client.move_delta([0.01, 0.0, 0.0])
    client.close()
    service.stop()


def test_session_aware_client_registers_before_calls():
    server = RpcServer(enable_sessions=True)
    server.register('env.get_env_meta', lambda: {'robot': 'session'})
    host, port = server.start()
    plain = make_client(f'http://{host}:{port}')
    with pytest.raises(RpcError, match='session required'):
        plain.call('env.get_env_meta')
    aware = make_client(f'http://{host}:{port}', enable_sessions=True)
    wait_for_ready(aware, timeout_s=5.0)
    assert aware.call('env.get_env_meta')['robot'] == 'session'
    aware.close()
    server.stop()


class TcpMockRobot(MockRobot):
    """MockRobot with a Cartesian model and no IK dependency for RPC tests."""

    def __init__(self):
        super().__init__(RobotSpec.generic(6))
        self.tcp = np.array([0.0, 0.0, 0.3, 0.0, 0.0, 0.0])
        self.moves = []

    def state(self):
        state = super().state()
        state.tcp = self.tcp.copy()
        return state

    def move_p(self, pose, wait=True, **kwargs):
        self.moves.append(np.asarray(pose, dtype=float).copy())
        self.tcp = np.asarray(pose, dtype=float).copy()
        return True


def make_piper_platform():
    robot = TcpMockRobot()
    gripper = MockGripper().connect()
    wrist = MockCamera('wrist', 32, 24, depth=True).open()
    wrist.set_extrinsics(np.eye(4))
    external = MockCamera('external', 24, 16, depth=True).open()
    external.set_extrinsics(np.eye(4))
    platform = RobotPlatform(robot, gripper=gripper,
                             cameras=CameraHub([wrist, external]),
                             sync_max_skew_s=1.0)
    return platform, robot, gripper


def test_piper_backend_observation_state_and_limits():
    platform, robot, gripper = make_piper_platform()
    config = PiperEnvConfig(enable_motion=True, main_camera='wrist',
                            extra_cameras=('external',), max_delta_m=0.03)
    backend = PiperRealBackend(platform, config)
    assert backend.get_env_meta()['mode'] == 'real'
    observation = backend.get_observation()
    assert observation['main_images'].shape == (24, 32, 3)
    assert observation['extra_view_images'].shape == (1, 16, 24, 3)
    assert observation['main_depths'].dtype == np.float32
    assert observation['main_depths'].max() == pytest.approx(1.0)
    state = backend.get_robot_state()
    np.testing.assert_allclose(state['robot_state']['tcp'], [0, 0, 0.3, 0, 0, 0])
    moved = backend.move_delta([0.01, 0, 0])
    assert moved['ok'] and not moved['clamped']
    np.testing.assert_allclose(robot.moves[-1][:3], [0.01, 0, 0.3])
    clamped = backend.move_delta([1.0, 0, 0])
    assert clamped['clamped']
    np.testing.assert_allclose(clamped['applied_delta_m'], [0.03, 0, 0])
    rotated = backend.rotate_delta([0, 0, 0.001])
    assert rotated['ok']
    np.testing.assert_allclose(robot.moves[-1][3:], [0, 0, 0.001], atol=1e-8)
    opened = backend.set_gripper(open=False)
    assert opened['ok'] and opened['width_m'] == pytest.approx(gripper.min_width, abs=0.002)
    meta = backend.get_camera_meta()
    np.testing.assert_allclose(meta['cameras']['wrist']['intrinsics'],
                               np.array([[28.8, 0, 16], [0, 28.8, 12], [0, 0, 1]]))
    assert backend.render_camera('wrist').shape == (24, 32, 3)
    with pytest.raises(NotSupported):
        backend.step([0.0] * 7)
    with pytest.raises(NotSupported):
        backend.chunk_step(np.zeros((1, 7)))


def test_piper_motion_requires_explicit_enable_and_reset_never_moves():
    platform, robot, _ = make_piper_platform()
    backend = PiperRealBackend(platform, PiperEnvConfig(enable_motion=False))
    with pytest.raises(SafetyViolation, match='motion is disabled'):
        backend.move_delta([0.01, 0, 0])
    with pytest.raises(SafetyViolation, match='motion is disabled'):
        backend.set_gripper(open=True)
    before = len(robot.moves)
    result = backend.reset()
    assert result['reset'] == 'operator'
    assert len(robot.moves) == before


def test_piper_service_end_to_end():
    platform, _, _ = make_piper_platform()
    service = EnvService(PiperRealBackend(
        platform, PiperEnvConfig(enable_motion=True, main_camera='wrist')))
    host, port = service.start()
    client = EnvClient.connect(f'http://{host}:{port}')
    assert client.meta['mode'] == 'real'
    observation = client.get_observation()
    assert observation['main_images'].shape == (24, 32, 3)
    result = client.move_delta([0.005, 0.0, 0.0])
    assert result['ok']
    assert client.set_gripper(open=True)['open']
    client.close()
    service.stop()
