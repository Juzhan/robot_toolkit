"""Continuous reference and action-chunk stream tests (no hardware)."""

from __future__ import annotations

import numpy as np
import pytest

from robot_toolkit.control import (ContinuousConfig, SequenceSource,
                                   run_continuous_stream)
from robot_toolkit.core.continuous import ContinuousReference
from robot_toolkit.core.conventions import RobotSpec
from robot_toolkit.gripper import MockGripper
from robot_toolkit.robot import MockRobot


def test_fallback_reference_respects_kinematic_limits():
    dt = 0.008
    reference = ContinuousReference(
        [0.0] * 6, dt, speed=0.8, acceleration=1.2, jerk=8.0,
        lower=[-6.28] * 6, upper=[6.28] * 6, backend='fallback')
    rng = np.random.default_rng(4)
    target = np.zeros(6)
    positions, velocities, accelerations = [], [], []
    for index in range(2500):
        if index % 39 == 0:
            target = rng.uniform(-0.6, 0.6, 6)
        positions.append(reference.step(target))
        velocities.append(reference.v.copy())
        accelerations.append(reference.a.copy())
    positions = np.asarray(positions)
    velocities = np.asarray(velocities)
    accelerations = np.asarray(accelerations)
    assert np.max(np.abs(velocities)) <= 0.8 + 1e-8
    assert np.max(np.abs(accelerations)) <= 1.2 + 1e-8
    assert np.max(np.abs(np.diff(accelerations, axis=0) / dt)) <= 8.0 + 1e-6
    assert np.max(np.abs(np.diff(positions, axis=0) / dt)) <= 0.8 + 1e-6


def test_reference_rejects_bad_targets():
    reference = ContinuousReference([0.0] * 6, 0.008, 0.8, 1.2, 8.0,
                                    [-1.0] * 6, [1.0] * 6, backend='fallback')
    for target in ([float('nan')] * 6, [float('inf')] * 6, [1.1] * 6):
        with pytest.raises(ValueError):
            reference.step(target)


def _actions():
    count = 20
    actions = np.zeros((count, 7), dtype=float)
    actions[:, 0] = np.linspace(0.0, 0.4, count)
    actions[:5, 6] = 0.085
    actions[5:, 6] = 0.0
    return actions


def test_stream_interpolates_chunks_and_gates_gripper():
    robot = MockRobot(RobotSpec.generic(6), max_velocity=10.0)
    gripper = MockGripper(max_width=0.085)
    config = ContinuousConfig(
        control_hz=125.0, action_hz=10.0,
        max_joint_speed=2.0, max_joint_accel=5.0, max_joint_jerk=50.0,
        max_tracking_error=1.0, progress_reference_gap=0.1,
        progress_tracking_gap=0.1, settle_s=0.1, mock_sleep_s=0.0)
    result = run_continuous_stream(
        robot, gripper,
        SequenceSource(_actions(), action_hz=10.0, chunk_size=8),
        config, realtime=False)
    assert not result.aborted, result.error
    assert result.end_reason == 'model_done'
    assert result.servo_commands > 100
    assert result.chunks >= 3
    assert 0.0 in gripper.history
    assert robot.state().q[0] == pytest.approx(0.4, abs=0.05)


class _StuckRobot(MockRobot):
    def __init__(self):
        super().__init__(RobotSpec.generic(6), max_velocity=10.0)
        self.stuck_q = self._q.copy()

    def servo_j(self, joints, dt):
        self.commands.append(np.asarray(joints, dtype=float).copy())
        return True

    def state(self):
        sample = super().state()
        sample.q = self.stuck_q.copy()
        return sample


def test_stream_stops_when_robot_cannot_track():
    robot = _StuckRobot()
    config = ContinuousConfig(
        control_hz=125.0, action_hz=10.0,
        max_joint_speed=2.0, max_joint_accel=5.0, max_joint_jerk=50.0,
        max_tracking_error=0.05, progress_reference_gap=0.1,
        progress_tracking_gap=0.1, tracking_timeout_s=0.2,
        settle_s=0.1, mock_sleep_s=0.0)
    result = run_continuous_stream(
        robot, None, SequenceSource(_actions(), action_hz=10.0, chunk_size=8),
        config, realtime=False)
    assert result.aborted
    assert result.end_reason in ('tracking_error', 'tracking_timeout')
