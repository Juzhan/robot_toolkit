"""Piper backend tests with a fake pyAgxArm module (no CAN, no hardware)."""
from __future__ import annotations

import sys
import time
import types

import numpy as np
import pytest

from robot_toolkit.core.conventions import RobotSpec
from robot_toolkit.core.mathutils import euler_to_matrix, rotvec_to_matrix
from robot_toolkit.profiles import build_gripper, build_robot
from robot_toolkit.robot import AgxPiperConfig, AgxPiperRobot


class _Message:
    def __init__(self, msg, timestamp=None, hz=100.0):
        self.msg = msg
        self.timestamp = time.monotonic() if timestamp is None else timestamp
        self.hz = hz


class _FakeEffector:
    def __init__(self):
        self.width = 0.08
        self.force = 0.0
        self.moves = []

    def get_gripper_status(self):
        return _Message(types.SimpleNamespace(value=self.width, mode='width',
                                              force=self.force))

    def move_gripper_m(self, value=0.0, force=1.0):
        self.width = float(value)
        self.force = float(force)
        self.moves.append((self.width, self.force))


class _FakeArm:
    def __init__(self):
        self.q = [0.0, 0.1, -0.1, 0.0, 0.1, 0.0]
        self.commands = []
        self.pose_commands = []
        self.connected = False
        self.enabled = False
        self.effector = None
        self.leader = False
        self.leader_q = None
        self.OPTIONS = types.SimpleNamespace(
            EFFECTOR=types.SimpleNamespace(AGX_GRIPPER='agx_gripper'))
        self.speed = None
        self.auto_motion_mode = True
        self.motion_mode = ''
        self.motion_mode_calls = 0
        self.disable_calls = 0
        self.disable_lag = 0
        self.t0 = time.monotonic()

    def connect(self):
        self.connected = True

    def disconnect(self):
        self.connected = False

    def is_connected(self):
        return self.connected

    def is_ok(self):
        return True

    def get_fps(self):
        return 100.0

    def get_joint_angles(self):
        return _Message(list(self.q), timestamp=time.monotonic() - self.t0)

    def get_leader_joint_angles(self):
        if self.leader_q is None:
            return None
        return _Message(list(self.leader_q), timestamp=time.monotonic() - self.t0)

    def fk(self, joints):
        return [0.2, 0.0, 0.3, 0.0, 1.5708, 0.0]

    def get_flange_pose(self):
        return _Message([0.2, 0.0, 0.3, 0.0, 1.5708, 0.0])

    def get_tcp_pose(self):
        return self.get_flange_pose()

    def get_arm_status(self):
        return _Message(types.SimpleNamespace(arm_status=0, motion_status=0, err_code=0))

    def get_firmware(self):
        return {'software_version': 'S-V1.8-8'}

    def init_effector(self, kind):
        assert kind == 'agx_gripper'
        self.effector = _FakeEffector()
        return self.effector

    def enable(self):
        self.enabled = True
        return True

    def disable(self):
        self.disable_calls += 1
        if self.disable_calls <= self.disable_lag:
            return False
        self.enabled = False
        return True

    def set_speed_percent(self, percent):
        self.speed = int(percent)

    def set_auto_set_motion_mode_enabled(self, enabled):
        self.auto_motion_mode = bool(enabled)

    def set_motion_mode(self, mode):
        self.motion_mode = str(mode)
        self.motion_mode_calls += 1

    def set_joint_limits_enabled(self, enabled):
        self.limits_enabled = bool(enabled)

    def set_tcp_offset(self, pose):
        self.tcp_offset = list(pose)

    def move_j(self, joints):
        self.q = [float(value) for value in joints]
        self.commands.append(list(self.q))

    def move_p(self, pose):
        self.pose_commands.append([float(value) for value in pose])

    def set_leader_mode(self):
        self.leader = True
        self.leader_q = list(self.q)

    def set_follower_mode(self):
        self.leader = False
        self.leader_q = None

    def electronic_emergency_stop(self):
        self.enabled = False

    def reset(self):
        self.enabled = False


class _FakeFactory:
    arm = _FakeArm()

    @classmethod
    def create_arm(cls, config):
        return cls.arm


def _fake_sdk():
    module = types.ModuleType('pyAgxArm')
    module.ArmModel = types.SimpleNamespace(PIPER='piper', PIPER_H='piper_h',
                                             PIPER_L='piper_l', PIPER_X='piper_x')
    module.PiperFW = types.SimpleNamespace(DEFAULT='default', V183='v183',
                                           V188='v188', V189='v189')
    module.AgxArmFactory = _FakeFactory
    module.create_agx_arm_config = lambda **kwargs: dict(kwargs)
    module.resolve_firmware_profile = lambda robot, version: 'v188'
    return module


def test_mock_backend_contract():
    robot = AgxPiperRobot(AgxPiperConfig(mock=True), mock=True)
    robot.connect()
    state = robot.state()
    assert state.q.shape == (6,)
    assert robot.servo_j([0.1] * 6, dt=0.01)
    assert robot.move_j([0.2] * 6, speed=0.5, wait=True, rate_hz=200)
    assert robot.open_gripper()
    assert robot.gripper_width > 0
    robot.disconnect()


def test_real_backend_with_fake_sdk(monkeypatch):
    monkeypatch.setitem(sys.modules, 'pyAgxArm', _fake_sdk())
    _FakeFactory.arm = _FakeArm()
    robot = AgxPiperRobot(AgxPiperConfig(model='piper_x', interface='agx_cando',
                                         channel='0', firmware='default',
                                         enable_on_connect=True), mock=False)
    robot.connect()
    state = robot.state()
    assert state.q.shape == (6,)
    assert state.tcp is not None
    assert state.tcp[3:] == pytest.approx([0.0, 1.5708, 0.0], abs=1e-5)
    assert state.extra['tcp_rpy'] == pytest.approx([0.0, 1.5708, 0.0])
    assert robot.connected and robot._enabled
    assert robot.servo_j([0.05] * 6, 0.01)
    assert robot.servo_j([0.06] * 6, 0.01)
    assert robot.move_j([0.2] * 6, wait=True, rate_hz=200)
    assert _FakeFactory.arm.commands[-1] == pytest.approx([0.2] * 6)
    assert _FakeFactory.arm.auto_motion_mode is False
    assert _FakeFactory.arm.motion_mode == 'j'
    assert _FakeFactory.arm.motion_mode_calls == 1
    assert robot.move_gripper(0.03, wait=True)
    assert robot.gripper_width == pytest.approx(0.03, abs=1e-3)
    target_rotvec = [0.1, 0.5, -0.2]
    assert robot.move_p([0.2, 0.0, 0.3, *target_rotvec], wait=False)
    sdk_pose = _FakeFactory.arm.pose_commands[-1]
    assert np.allclose(euler_to_matrix(sdk_pose[3:]),
                       rotvec_to_matrix(target_rotvec), atol=1e-6)
    assert robot.freedrive(True)
    _FakeFactory.arm.leader_q = [0.3, 0.2, -0.2, 0.1, 0.2, 0.1]
    state = robot.state()
    assert state.extra['feedback'] == 'leader'
    assert state.q == pytest.approx([0.3, 0.2, -0.2, 0.1, 0.2, 0.1])
    assert robot.freedrive(False)
    assert robot.state().extra['feedback'] == 'follower'
    robot.disconnect()


def test_disable_retries_while_feedback_lags(monkeypatch):
    monkeypatch.setitem(sys.modules, 'pyAgxArm', _fake_sdk())
    arm = _FakeArm()
    arm.enabled = True
    arm.disable_lag = 3
    _FakeFactory.arm = arm
    robot = AgxPiperRobot(AgxPiperConfig(model='piper_x', firmware='default',
                                         enable_on_connect=False), mock=False)
    robot.connect()
    assert robot.disable(timeout=1.0)
    assert arm.disable_calls >= 4
    assert not arm.enabled
    assert not robot._enabled
    robot.disconnect()


def test_profile_builds_safe_piper_and_gripper(monkeypatch):
    monkeypatch.setitem(sys.modules, 'pyAgxArm', _fake_sdk())
    _FakeFactory.arm = _FakeArm()
    robot = build_robot('piper_x', mock=True)
    robot.connect()
    assert robot.spec is not None and robot.spec.name == 'piper_x'
    assert robot.state().q.shape == (6,)
    gripper = build_gripper('agx', robot=robot)
    assert gripper.max_width == pytest.approx(0.08)
    assert gripper.close(wait=True)
    robot.disconnect()
