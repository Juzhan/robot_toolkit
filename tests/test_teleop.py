"""Teleop command bridge, Cartesian mapping and Joy-Con adapter tests."""

from __future__ import annotations

import numpy as np
import pytest

from robot_toolkit.core.conventions import GripperSpec
from robot_toolkit.core.mathutils import matrix_to_rotvec, pose_error, rotvec_to_matrix
from robot_toolkit.robot import Capabilities, MockRobot
from robot_toolkit.teleop import (CartesianMappingConfig, CartesianTeleop,
                                  EnableLatch, GripperToggle, JoyConSource,
                                  KeyboardSource, ScriptedSource, TeleopBridge,
                                  TeleopCommand,
                                  TeleopRunner, TeleopRunnerConfig,
                                  TeleopSessionConfig, build_teleop_session,
                                  stick_axes)


def test_mapping_does_nothing_while_disabled():
    teleop = CartesianTeleop(gripper=GripperSpec())
    teleop.reset([0.3, 0.0, 0.2], np.eye(3), 0.05, enabled=False)
    target = teleop.update(TeleopCommand(xyz=(1, 1, 1), grip=-1), 0.1)
    assert np.allclose(target.position, [0.3, 0.0, 0.2])
    assert target.gripper == pytest.approx(0.05)
    assert not target.active


def test_mapping_translation_scale_and_workspace():
    config = CartesianMappingConfig(translation_speed=0.1, workspace_lower=(0, 0, 0),
                                    workspace_upper=(0.32, 1, 1))
    teleop = CartesianTeleop(config, GripperSpec())
    teleop.reset([0.3, 0.0, 0.2], np.eye(3), 0.05, enabled=True)
    target = teleop.update(TeleopCommand(xyz=(1, 0, 0)), 0.05)
    assert target.position[0] == pytest.approx(0.305)
    assert not target.limited
    target = teleop.update(TeleopCommand(xyz=(1, 0, 0)), 1.0)
    assert target.position[0] == pytest.approx(0.32)
    assert target.limited


def test_mapping_acceleration_limit():
    config = CartesianMappingConfig(translation_speed=0.2, max_position_step=1.0,
                                    max_linear_acceleration=1.0)
    teleop = CartesianTeleop(config, GripperSpec())
    teleop.reset([0.0, 0.0, 0.3], np.eye(3), 0.05, enabled=True)
    first = teleop.update(TeleopCommand(xyz=(1, 0, 0)), 0.1)
    second = teleop.update(TeleopCommand(xyz=(1, 0, 0)), 0.1)
    assert first.position[0] == pytest.approx(0.01)
    assert second.position[0] == pytest.approx(0.03)
    assert teleop.linear_velocity[0] == pytest.approx(0.2)


def test_mapping_rotation_rate_and_absolute_quaternion():
    teleop = CartesianTeleop(CartesianMappingConfig(rotation_speed=0.5), GripperSpec())
    teleop.reset([0, 0, 0.3], np.eye(3), 0.05, enabled=True)
    target = teleop.update(TeleopCommand(rpy=(0, 0, 1)), 0.1)
    assert np.linalg.norm(matrix_to_rotvec(target.rotation)) == pytest.approx(0.05)
    absolute = rotvec_to_matrix([0.2, 0.0, 0.0])
    quaternion = np.array([np.sin(0.1), 0, 0, np.cos(0.1)])  # 0.2 rad about x, xyzw
    target = teleop.update(TeleopCommand(quaternion=tuple(quaternion)), 0.1)
    translation, rotation = pose_error(np.r_[target.position, matrix_to_rotvec(target.rotation)],
                                       np.r_[target.position, matrix_to_rotvec(absolute)])
    assert rotation < 0.12  # limited to max_rotation_step but pointing the right way


def test_mapping_clamps_rotation_step():
    config = CartesianMappingConfig(max_rotation_step=0.05)
    teleop = CartesianTeleop(config, GripperSpec())
    teleop.reset([0, 0, 0.3], np.eye(3), 0.05, enabled=True)
    target = teleop.update(TeleopCommand(rpy=(10, 10, 10)), 1.0)
    assert np.linalg.norm(matrix_to_rotvec(target.rotation)) == pytest.approx(0.05, abs=1e-9)
    assert target.limited


def test_mapping_gripper_integrates_and_reanchors_on_enable():
    teleop = CartesianTeleop(CartesianMappingConfig(grip_speed=0.1), GripperSpec())
    teleop.reset([0.3, 0, 0.2], np.eye(3), 0.085, enabled=True)
    target = teleop.update(TeleopCommand(grip=-1.0), 0.5)
    assert target.gripper == pytest.approx(0.035)
    teleop.update(TeleopCommand(enable=False), 0.1)
    assert not teleop.enabled
    moved = np.array([0.4, 0.1, 0.25])
    teleop.update(TeleopCommand(enable=True), 0.1, measured_position=moved)
    assert np.allclose(teleop.position, moved)


def test_bridge_expires_stale_commands():
    now = [0.0]
    bridge = TeleopBridge(stale_s=0.5, clock=lambda: now[0])
    bridge.put(TeleopCommand(xyz=(1, 0, 0)))
    assert bridge.fresh()
    now[0] = 0.6
    assert not bridge.fresh()
    assert bridge.age() == pytest.approx(0.6)


def test_scripted_source_timeline():
    now = [0.0]
    source = ScriptedSource([(0.0, TeleopCommand(xyz=(1, 0, 0))),
                             (1.0, TeleopCommand(xyz=(-1, 0, 0)))],
                            clock=lambda: now[0])
    assert source.poll().xyz == (1.0, 0.0, 0.0)
    now[0] = 1.2
    assert source.poll().xyz == (-1.0, 0.0, 0.0)


def test_stick_axes_and_buttons():
    assert stick_axes(4096, 2048, center=(2048, 2048)) == [1.0, 0.0, 0.0]
    assert stick_axes(0, 4096, center=(2048, 2048)) == [-1.0, -1.0, 0.0]
    assert stick_axes(2048, 2048, r=True, zr=True, center=(2048, 2048)) == [0.0, 0.0, 0.0]
    toggle = GripperToggle(0.085)
    assert toggle.update(True, 0.085) == 0.0
    assert toggle.update(True, 0.0) == 0.0  # held: still edge-triggered
    assert toggle.update(False, 0.0) == 0.0
    assert toggle.update(True, 0.0) == 0.085
    latch = EnableLatch()
    assert latch.update(False, False) is None
    assert latch.update(True, False) is True
    assert latch.update(True, False) is None
    assert latch.update(False, False) is None
    assert latch.update(True, False) is False
    # Calibration pauses motion; it must not turn into an enable command.
    assert latch.update(True, True) is None


class _Buttons:
    def __init__(self, y=False, b=False):
        self.y, self.b = y, b

    def get_stick_right_vertical(self):
        return 4096

    def get_stick_right_horizontal(self):
        return 2048

    def get_button_r(self):
        return False

    def get_button_zr(self):
        return False

    def get_button_b(self):
        return self.b

    def get_button_y(self):
        return self.y

    def get_button_home(self):
        return False


class _Controller:
    def __init__(self):
        self.joycon = _Buttons()
        self.gyro = type('Gyro', (), {'is_calibrating': False})()
        self.pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    def get_control(self):
        return self.pose, 0.085, None


def test_joycon_source_emits_normalized_command():
    controller = _Controller()
    source = JoyConSource(controller, gripper_max=0.085, yaw_degrees=0.0)
    source._stick_center = (2048, 2048)
    source.last_input = source.clock()
    command = source.poll()
    assert command is not None
    assert command.xyz == (1.0, 0.0, 0.0)
    assert command.enable is None
    controller.joycon.y = True
    command = source.poll()
    assert command.enable is True
    controller.joycon.y = False
    assert source.poll().enable is None


def test_teleop_runner_uses_servo_l():
    def forward(q):
        q = np.asarray(q, dtype=float)
        return np.r_[0.3 + 0.1 * q[0], 0.1 * q[1], 0.3, 0.0, 0.0, 0.0]

    def inverse(pose):
        pose = np.asarray(pose, dtype=float)
        return np.r_[(pose[0] - 0.3) / 0.1, pose[1] / 0.1, 0.0, 0.0, 0.0, 0.0]

    robot = MockRobot(forward_kinematics=forward, inverse_kinematics=inverse)
    mapping = CartesianTeleop(CartesianMappingConfig(), GripperSpec())
    source = ScriptedSource([(0.0, TeleopCommand(xyz=(1, 0, 0), enable=True))])
    runner = TeleopRunner(robot, None, source, mapping,
                          TeleopRunnerConfig(rate_hz=50, max_duration_s=0.08,
                                             realtime=False))
    result = runner.run()
    assert not result.aborted, result.reason
    assert result.frames > 0
    assert robot.commands


def test_keyboard_source_uses_same_command_contract():
    state = set()
    source = KeyboardSource(gripper_max=0.08, key_state=lambda: set(state))
    assert source.poll().xyz == (0.0, 0.0, 0.0)
    state.add('w')
    assert source.poll().xyz[0] == 1.0
    state.clear()
    assert source.poll().xyz[0] == 0.0
    state.add('y')
    assert source.poll().enable is True
    state.clear()
    source.poll()
    state.add('space')
    assert source.poll().grip_target == 0.0
    state.clear()
    source.poll()
    state.add('space')
    assert source.poll().grip_target == 0.08
    state.add('escape')
    assert source.poll().buttons['quit'] is True


def test_mapping_accepts_absolute_gripper_target():
    teleop = CartesianTeleop(CartesianMappingConfig(), GripperSpec())
    teleop.reset([0.3, 0, 0.2], np.eye(3), 0.08, enabled=True)
    target = teleop.update(TeleopCommand(grip_target=0.0), 0.1)
    assert target.gripper == pytest.approx(0.0)


def test_build_teleop_session_mock():
    session = build_teleop_session(TeleopSessionConfig(
        robot='mock', mock=True, max_duration_s=0.05, rate_hz=50,
    ))
    try:
        result = session.run()
        assert not result.aborted, result.reason
        assert result.frames > 0
    finally:
        session.close()


def test_teleop_runner_uses_move_p_without_servo_l():
    def forward(q):
        q = np.asarray(q, dtype=float)
        return np.r_[0.3 + 0.1 * q[0], 0.1 * q[1], 0.3, 0.0, 0.0, 0.0]

    robot = MockRobot(forward_kinematics=forward, inverse_kinematics=lambda pose: np.zeros(6))
    robot.capabilities = Capabilities(servo_j=True, servo_l=False, move_p=True)
    calls = []
    robot.move_p = lambda pose, wait=False: calls.append(np.asarray(pose)) or True
    mapping = CartesianTeleop(CartesianMappingConfig(), GripperSpec())
    source = ScriptedSource([(0.0, TeleopCommand(xyz=(1, 0, 0), enable=True))])
    result = TeleopRunner(robot, None, source, mapping,
                          TeleopRunnerConfig(rate_hz=50, move_p_hz=20,
                                             max_duration_s=0.1,
                                             realtime=False)).run()
    assert not result.aborted, result.reason
    assert result.frames > 0
    assert calls


def test_joycon_source_reports_stale_input_as_none():
    now = [0.0]
    controller = _Controller()
    source = JoyConSource(controller, clock=lambda: now[0])
    source._stick_center = (2048, 2048)
    now[0] = 10.0
    assert source.poll() is None
