"""Teaching layer: storage, dwell detection, trimming, recording, replay."""

from __future__ import annotations

import time

import numpy as np
import pytest

from robot_toolkit.core.conventions import RobotSpec
from robot_toolkit.gripper import MockGripper
from robot_toolkit.robot import MockRobot
from robot_toolkit.teach import (Marker, Replayer, TeachRecorder, Trajectory,
                                 add_marker, find_dwell_candidates,
                                 load_trajectory, prepare_playback, retime,
                                 save_trajectory, summarize, trim)


def _trajectory(duration=2.0, rate=100.0, dof=6):
    t = np.arange(0.0, duration, 1.0 / rate)
    q = np.zeros((len(t), dof))
    q[:, 0] = 0.5 * t / duration
    grip = np.full(len(t), 0.085)
    return Trajectory(t=t, q=q, grip=grip,
                      markers=[Marker(t=0.5, index=int(0.5 * rate), action='close'),
                               Marker(t=1.5, index=int(1.5 * rate), action='open')],
                      meta={'task': 'unit test'})


def test_trajectory_properties_and_roundtrip(tmp_path):
    trajectory = _trajectory()
    assert trajectory.dof == 6
    assert trajectory.duration == pytest.approx(2.0 - 0.01)
    assert trajectory.rate == pytest.approx(100.0)
    save_trajectory(tmp_path, trajectory)
    loaded = load_trajectory(tmp_path)
    assert np.allclose(loaded.q, trajectory.q)
    assert [marker.action for marker in loaded.markers] == ['close', 'open']
    assert loaded.meta['task'] == 'unit test'


def test_summary_and_editor_helpers():
    trajectory = _trajectory()
    summary = summarize(trajectory)
    assert summary['dof'] == 6
    assert summary['markers'] == ['0.50s:close', '1.50s:open']
    faster = retime(trajectory, 0.5)
    assert faster.duration == pytest.approx(trajectory.duration / 2)
    assert faster.markers[0].t == pytest.approx(0.25)
    add_marker(trajectory, action='phase', at=1.0, label='push')
    assert any(marker.action == 'phase' for marker in trajectory.markers)
    assert [marker.t for marker in trajectory.markers] == sorted(
        marker.t for marker in trajectory.markers)


def test_dwell_detection_finds_standstill():
    rate = 100.0
    t = np.arange(0.0, 3.0, 1.0 / rate)
    q = np.zeros((len(t), 6))
    moving = (t >= 1.0) & (t <= 2.0)
    q[moving, 0] = np.sin((t[moving] - 1.0) * np.pi) * 0.6
    trajectory = Trajectory(t=t, q=q)
    runs = find_dwell_candidates(trajectory, ignore_before=0.2)
    assert runs
    assert any(run['start'] >= 2.0 and run['duration'] > 0.5 for run in runs)


def test_trim_shifts_time_and_markers():
    trajectory = _trajectory()
    trimmed, info = trim(trajectory, start=0.4, end=1.6)
    assert trimmed.duration == pytest.approx(1.2, abs=0.02)
    assert trimmed.t[0] == pytest.approx(0.0)
    assert info['removed_head'] == pytest.approx(0.4, abs=0.02)
    assert [marker.action for marker in trimmed.markers] == ['close', 'open']
    assert trimmed.markers[0].t == pytest.approx(0.1, abs=0.02)


def test_prepare_playback_speed_scaling_and_events():
    trajectory = _trajectory()
    plan = prepare_playback(trajectory, rate=125.0, speed_scale=0.5)
    assert plan.speed_scale == 0.5
    assert plan.duration > trajectory.duration / 0.5
    assert plan.max_speed == pytest.approx(plan.source_speed * 0.5)
    assert [event['action'] for event in plan.events] == ['close', 'open']
    assert plan.schedule.width_at(plan.events[0]['t']) == 0.0
    assert plan.schedule.width_at(plan.events[1]['t']) == 0.085


def test_teach_recorder_captures_state_and_markers():
    robot = MockRobot(RobotSpec.generic(6), max_velocity=5.0)
    gripper = MockGripper(max_width=0.085)
    recorder = TeachRecorder(robot, gripper, rate=100.0)
    recorder.start()
    time.sleep(0.12)
    recorder.mark('close', label='grasp')
    robot.servo_j([0.2] * 6, 0.05)
    time.sleep(0.08)
    trajectory = recorder.stop()
    assert trajectory.meta['samples'] >= 5
    assert trajectory.q.shape[1] == 6
    assert trajectory.markers[-1].action == 'close'
    assert trajectory.duration > 0.05


def test_replay_drives_robot_and_fires_gripper_events():
    trajectory = _trajectory()
    robot = MockRobot(RobotSpec.generic(6), max_velocity=5.0)
    gripper = MockGripper(max_width=0.085)
    replayer = Replayer(robot, gripper, rate=125.0,
                        close_settle_s=0.0, open_settle_s=0.0)
    result = replayer.run(trajectory, speed_scale=1.0, realtime=False)
    assert not result.aborted, result.reason
    assert result.frames > 300
    assert [event['action'] for event in result.events] == ['close', 'open']
    assert 0.0 in gripper.history
    assert np.isclose(gripper.history[-1], 0.085)
    assert robot.state().q[0] == pytest.approx(0.5, abs=0.03)


def test_replay_rejects_too_fast_playback():
    trajectory = _trajectory(duration=0.25, rate=400.0)
    robot = MockRobot(RobotSpec.generic(6), max_velocity=5.0)
    replayer = Replayer(robot, None, rate=125.0, max_joint_speed=0.01)
    with pytest.raises(Exception):
        replayer.run(trajectory, speed_scale=1.0, realtime=False)


def test_replay_can_use_a_continuous_reference():
    trajectory = _trajectory()
    robot = MockRobot(RobotSpec.generic(6), max_velocity=5.0)
    replayer = Replayer(robot, None, rate=125.0, max_joint_speed=5.0,
                        continuous=True, max_joint_accel=5.0,
                        max_joint_jerk=50.0)
    result = replayer.run(trajectory, speed_scale=1.0, realtime=False)
    assert not result.aborted, result.reason
    assert result.continuous
    assert result.reference_backend in ('ruckig', 'fallback')
    assert robot.state().q[0] == pytest.approx(0.5, abs=0.05)


def test_replay_aborts_on_operator_request():
    import threading
    trajectory = _trajectory()
    robot = MockRobot(RobotSpec.generic(6), max_velocity=5.0)
    abort = threading.Event()
    abort.set()
    replayer = Replayer(robot, None, rate=125.0)
    result = replayer.run(trajectory, speed_scale=1.0, realtime=False, abort_event=abort)
    assert result.aborted
    assert result.reason == 'operator abort'
