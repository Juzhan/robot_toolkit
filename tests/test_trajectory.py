"""Exact trajectory playback and the shared servo stream (no hardware)."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import numpy as np
import pytest

from robot_toolkit.control import (ServoStream, StreamAbort, StreamLimits, TrajectoryLimits,
                                   check_trajectory, min_jerk, retime, run_trajectory,
                                   trajectory_peaks)
from robot_toolkit.core.conventions import RobotSpec
from robot_toolkit.robot import MockRobot
from robot_toolkit.robot.base import RobotState


def test_min_jerk_respects_peak_speed_and_acceleration():
    dt = 0.008
    start, goal = np.zeros(6), np.array([0.6, -0.3, 0.0, 0.1, 0.0, 0.0])
    q = min_jerk(start, goal, dt, speed=0.5, accel=1.0)
    np.testing.assert_allclose(q[0], start)
    np.testing.assert_allclose(q[-1], goal)
    peaks = trajectory_peaks(q, dt)
    assert peaks['velocity'].max() <= 0.5 + 1e-3
    assert peaks['acceleration'].max() <= 1.0 + 1e-2
    # Joints are synchronized: every joint is the same fraction of the way.
    fraction = q[:, 0] / 0.6
    np.testing.assert_allclose(q[:, 1], -0.3 * fraction, atol=1e-12)
    assert len(min_jerk(goal, goal, dt, speed=0.5)) == 1
    with pytest.raises(ValueError):
        min_jerk(start, goal, dt, speed=0.0)


def test_check_trajectory_names_every_violation():
    dt = 0.01
    q = np.zeros((50, 2))
    q[25:, 0] = 0.1  # a 10 rad/s step
    with pytest.raises(ValueError, match='velocity') as error:
        check_trajectory(q, dt, TrajectoryLimits(velocity=1.0, acceleration=5.0),
                         upper=[0.05, 1.0])
    assert 'acceleration' in str(error.value)
    assert 'upper joint limit' in str(error.value)
    with pytest.raises(ValueError, match='NaN'):
        check_trajectory(np.full((3, 2), np.nan), dt)
    peaks = check_trajectory(min_jerk([0.0, 0.0], [0.2, 0.0], dt, 0.5), dt)
    assert peaks['velocity'][0] <= 0.5 + 1e-3


def test_retime_follows_planner_frames_within_all_limits():
    # Frames of a curved planned path; straight segments between them would
    # change velocity abruptly at every frame.
    u = np.linspace(0.0, 1.0, 21)
    knots = np.stack([0.8 * u, 0.4 * np.sin(np.pi * u), -0.3 * u ** 2], axis=1)
    dt = 0.01
    limits = TrajectoryLimits(velocity=1.0, acceleration=1.2, jerk=8.0)
    q = retime(knots, dt, limits)
    check_trajectory(q, dt, limits)
    np.testing.assert_allclose(q[0], knots[0], atol=1e-12)
    np.testing.assert_allclose(q[-1], knots[-1], atol=1e-12)
    # Rest to rest, and every frame lies on the path.
    np.testing.assert_allclose(q[1] - q[0], 0.0, atol=1e-5)
    np.testing.assert_allclose(q[-1] - q[-2], 0.0, atol=1e-5)
    gaps = [np.min(np.max(np.abs(q - knot), axis=1)) for knot in knots]
    assert max(gaps) < 2e-3
    assert len(retime(knots[:1], dt, limits)) == 1


def test_run_trajectory_tracks_every_sample_exactly():
    robot = MockRobot(RobotSpec.generic(6), max_velocity=5.0)
    dt = 0.008
    q = min_jerk(np.zeros(6), np.full(6, 0.3), dt, speed=0.6)
    seen = []
    result = run_trajectory(robot, q, dt, limits=TrajectoryLimits(velocity=0.7),
                            realtime=False, on_command=seen.append)
    assert result.ok and result.reason == 'reached', result.as_dict()
    assert result.phase == 'tracking'
    # The first len(q) commands are the samples themselves, unfiltered.
    np.testing.assert_allclose(np.asarray(robot.commands[:len(q)]), q)
    assert [sample['index'] for sample in seen[:3]] == [0, 1, 2]
    assert result.final_error <= 0.02
    assert result.stream['servo_commands'] >= len(q)


def test_run_trajectory_refuses_before_commanding():
    robot = MockRobot(RobotSpec.generic(6), max_velocity=5.0)
    dt = 0.008
    far = min_jerk(np.full(6, 0.2), np.full(6, 0.4), dt, speed=0.5)
    with pytest.raises(ValueError, match='starts'):
        run_trajectory(robot, far, dt, realtime=False)
    fast = min_jerk(np.zeros(6), np.full(6, 0.4), dt, speed=2.0)
    with pytest.raises(ValueError, match='trajectory rejected'):
        run_trajectory(robot, fast, dt, limits=TrajectoryLimits(velocity=1.0),
                       realtime=False)
    assert robot.commands == []


def test_run_trajectory_aborts_and_stops():
    robot = MockRobot(RobotSpec.generic(6), max_velocity=5.0)
    stops = []
    robot.stop_motion = lambda *unused: stops.append(True)
    abort = threading.Event()
    abort.set()
    q = min_jerk(np.zeros(6), np.full(6, 0.3), 0.008, speed=0.5)
    result = run_trajectory(robot, q, 0.008, realtime=False, abort_event=abort)
    assert not result.ok and result.reason == 'aborted'
    assert stops == [True]


def test_run_trajectory_reports_tracking_failure():
    # A slow arm cannot follow: the stream must stop on the tracking limit
    # rather than keep streaming a path the robot is not on.
    robot = MockRobot(RobotSpec.generic(6), max_velocity=0.05)
    q = min_jerk(np.zeros(6), np.full(6, 0.6), 0.008, speed=0.8)
    result = run_trajectory(robot, q, 0.008, realtime=False,
                            stream_limits=StreamLimits(max_tracking_error=0.1))
    assert not result.ok
    assert result.reason == 'tracking_error'
    assert result.stream['max_tracking_error'] > 0.1


def test_servo_stream_rejects_virtual_time_and_stale_state_on_hardware():
    state = RobotState(t=1.0, q=np.zeros(6), protective_stop=False)
    hardware = SimpleNamespace(state=lambda: state, servo_j=lambda q, dt: True)
    with pytest.raises(ValueError, match='virtual time'):
        ServoStream(hardware, 0.008, realtime=False)
    stream = ServoStream(hardware, 0.008, limits=StreamLimits(stale_state_s=0.0))
    stream.preflight()
    stream.begin()
    stream.measure()
    stream._began += 0.01  # the robot clock has not moved since
    with pytest.raises(StreamAbort, match='stale_robot_state'):
        stream.measure()
