"""Piper backend tests with a fake pyAgxArm module (no CAN, no hardware)."""
from __future__ import annotations

import sys
import time
import types

import numpy as np
import pytest

from robot_toolkit.core.conventions import RobotSpec
from robot_toolkit.core.mathutils import euler_to_matrix, rotvec_to_matrix
from robot_toolkit.core.safety import JointLimits, SafetyGuard
from robot_toolkit.errors import ConnectionFailed, NotSupported
from robot_toolkit.profiles import build_gripper, build_robot
from robot_toolkit.robot import (AgxPiperConfig, AgxPiperRobot, SafeRobot,
                                 scan_piper_can, set_piper_linkage)


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


def test_safe_robot_uses_native_joint_planner(monkeypatch):
    monkeypatch.setitem(sys.modules, 'pyAgxArm', _fake_sdk())
    arm = _FakeArm()
    _FakeFactory.arm = arm
    robot = AgxPiperRobot(AgxPiperConfig(model='piper_x', firmware='default',
                                         enable_on_connect=True), mock=False)
    robot.connect()
    guard = SafetyGuard(JointLimits([-3.0] * 6, [3.0] * 6, 1.0))
    safe = SafeRobot(robot, guard)
    try:
        before = len(arm.commands)
        assert safe.move_j([0.2] * 6, speed=0.2, wait=True, timeout=1.0)
        assert len(arm.commands) == before + 1
        assert arm.commands[-1] == pytest.approx([0.2] * 6)
        assert arm.speed == 20
    finally:
        safe.disconnect()


def test_ik_warmup_runs_at_connect(monkeypatch):
    monkeypatch.setitem(sys.modules, 'pyAgxArm', _fake_sdk())
    arm = _FakeArm()
    _FakeFactory.arm = arm

    class _WarmupSolver:
        def __init__(self):
            self.calls = []

        def warmup(self, pose, seed=None):
            self.calls.append((list(pose), None if seed is None else list(seed)))
            return True

    robot = AgxPiperRobot(AgxPiperConfig(model='piper_x', firmware='default',
                                         enable_on_connect=True), mock=False)
    solver = _WarmupSolver()
    robot.set_inverse_kinematics(solver)
    robot.connect()
    try:
        assert len(solver.calls) == 1
        assert len(solver.calls[0][0]) == 6
    finally:
        robot.disconnect()


def test_safe_robot_can_force_generic_motion_on_piper(monkeypatch):
    monkeypatch.setitem(sys.modules, 'pyAgxArm', _fake_sdk())
    arm = _FakeArm()
    _FakeFactory.arm = arm
    robot = AgxPiperRobot(AgxPiperConfig(model='piper_x', firmware='default',
                                         enable_on_connect=True), mock=False)
    robot.connect()

    class _FixedIK:
        def solve(self, pose, seed=None):
            return [0.3] * 6

    robot.set_inverse_kinematics(_FixedIK())
    safe = SafeRobot(robot, SafetyGuard(
        JointLimits([-3.0] * 6, [3.0] * 6, 1.0)))
    try:
        assert safe.move_j([0.2] * 6, speed=1.0, accel=1.0,
                           rate_hz=200, timeout=2.0, use_native=False)
        assert np.allclose(arm.q, [0.2] * 6, atol=0.02)
        assert len(arm.commands) > 1
        assert safe.move_p([0.1, 0.0, 0.3, 0.0, 0.0, 0.0],
                           speed=1.0, accel=1.0, timeout=2.0, use_native=False)
        assert np.allclose(arm.q, [0.3] * 6, atol=0.02)
        assert arm.pose_commands == []
    finally:
        safe.disconnect()


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


def test_read_only_connect_sends_no_config(monkeypatch):
    monkeypatch.setitem(sys.modules, 'pyAgxArm', _fake_sdk())
    arm = _FakeArm()
    _FakeFactory.arm = arm
    robot = AgxPiperRobot(AgxPiperConfig(model='piper_x', firmware='default',
                                         enable_on_connect=False,
                                         configure_on_connect=False), mock=False)
    robot.connect()
    assert arm.speed is None
    assert arm.motion_mode_calls == 0
    assert not arm.enabled
    assert robot.state().q is not None
    robot.disconnect()


def test_leader_joint_angles_reads_broadcast(monkeypatch):
    monkeypatch.setitem(sys.modules, 'pyAgxArm', _fake_sdk())
    arm = _FakeArm()
    _FakeFactory.arm = arm
    robot = AgxPiperRobot(AgxPiperConfig(model='piper_x', firmware='default',
                                         enable_on_connect=False,
                                         configure_on_connect=False), mock=False)
    robot.connect()
    assert robot.leader_joint_angles() is None
    arm.leader_q = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    assert robot.leader_joint_angles() == pytest.approx([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    assert arm.motion_mode_calls == 0
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


class _FakeBus:
    def __init__(self, messages):
        self._messages = list(messages)
        self.kwargs = {}
        self.sent = []
        self.shutdown_called = False

    def recv(self, timeout=None):
        if self._messages:
            return self._messages.pop(0)
        time.sleep(0.001)
        return None

    def send(self, message, timeout=None):
        self.sent.append(message)

    def shutdown(self):
        self.shutdown_called = True


class _FakeMessage:
    def __init__(self, arbitration_id, is_extended_id=False, data=b'', dlc=0):
        self.arbitration_id = arbitration_id
        self.is_extended_id = is_extended_id
        self.data = bytes(data)
        self.dlc = dlc


def _frame(can_id, timestamp, data=b'\x01\x02\x03\x04\x05\x06\x07\x08'):
    return types.SimpleNamespace(arbitration_id=can_id, timestamp=timestamp,
                                 dlc=len(data), data=data)


def _fake_can(monkeypatch, bus, adapters=1):
    module = types.ModuleType('can')

    def detect_available_configs(interfaces=None):
        return [{'interface': name, 'channel': 0}
                for name in (interfaces or []) for _ in range(adapters)]

    def open_bus(**kwargs):
        bus.kwargs = kwargs
        return bus

    module.detect_available_configs = detect_available_configs
    module.Bus = open_bus
    module.Message = _FakeMessage
    monkeypatch.setitem(sys.modules, 'can', module)


def test_scan_sees_leader_and_follower(monkeypatch):
    bus = _FakeBus([_frame(0x155, 0.00), _frame(0x156, 0.01), _frame(0x157, 0.02),
                    _frame(0x251, 0.00), _frame(0x251, 0.01), _frame(0x251, 0.02),
                    _frame(0x2A1, 0.00)])
    _fake_can(monkeypatch, bus)
    report = scan_piper_can(interface='agx_cando', channel='0', seconds=0.2)
    assert bus.kwargs['interface'] == 'agx_cando' and bus.kwargs['channel'] == '0'
    assert bus.kwargs['receive_own_messages'] is False
    assert bus.shutdown_called
    assert report['adapters'] == 1
    assert report['frames'] == 7
    assert report['leader_broadcast'] and report['follower_feedback']
    assert report['arms_estimate'] == 2
    ids = {node['id']: node for node in report['nodes']}
    assert ids[0x251]['label'] == 'motor_state_j1'
    assert ids[0x2A1]['label'] == 'arm_status'
    assert ids[0x251]['count'] == 3
    assert ids[0x251]['hz'] == pytest.approx(100.0, abs=0.5)
    assert ids[0x251]['hex'] == '0x251'


def test_scan_reports_duplicated_ids(monkeypatch):
    frames = [_frame(0x2A1, index * 0.0025) for index in range(41)]
    _fake_can(monkeypatch, _FakeBus(frames))
    report = scan_piper_can(interface='agx_cando', channel='0', seconds=0.2)
    assert report['frames'] == 41
    assert report['arms_estimate'] == 2
    assert any('0x2A1' in note and '两倍' in note for note in report['notes'])


def test_scan_reports_silent_bus(monkeypatch):
    bus = _FakeBus([])
    _fake_can(monkeypatch, bus, adapters=0)
    report = scan_piper_can(interface='agx_cando', channel='0', seconds=0.2)
    assert report['frames'] == 0 and report['nodes'] == []
    assert report['arms_estimate'] == 0
    assert any('适配器' in note for note in report['notes'])


def test_scan_reports_open_failure(monkeypatch):
    module = types.ModuleType('can')
    module.detect_available_configs = lambda interfaces=None: []

    def open_bus(**kwargs):
        raise RuntimeError('no adapter')

    module.Bus = open_bus
    monkeypatch.setitem(sys.modules, 'can', module)
    with pytest.raises(ConnectionFailed) as info:
        scan_piper_can(interface='agx_cando', channel='0', seconds=0.2)
    assert 'no adapter' in str(info.value)


def test_set_linkage_broadcasts_one_470_frame(monkeypatch):
    bus = _FakeBus([])
    _fake_can(monkeypatch, bus)
    report = set_piper_linkage('leader', interface='agx_cando', channel='0')
    assert bus.shutdown_called
    assert len(bus.sent) == 1
    message = bus.sent[0]
    assert message.arbitration_id == 0x470
    assert message.is_extended_id is False
    assert message.data == bytes([0xFA, 0, 0, 0, 0, 0, 0, 0])
    assert report['role'] == 'leader' and report['can_id'] == '0x470'


def test_set_linkage_follower_and_bad_role(monkeypatch):
    bus = _FakeBus([])
    _fake_can(monkeypatch, bus)
    set_piper_linkage('follower', interface='agx_cando', channel='0')
    assert bus.sent[0].data == bytes([0xFC, 0, 0, 0, 0, 0, 0, 0])
    with pytest.raises(ValueError):
        set_piper_linkage('master', interface='agx_cando', channel='0')
