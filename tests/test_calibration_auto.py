"""Hardware-free tests of the shared unattended calibration executor."""
from types import SimpleNamespace

import numpy as np
import pytest

from robot_toolkit.calibrate import auto
from robot_toolkit.calibrate.options import CalibrateOptions
from robot_toolkit.core.workspace_guard import held_object_slack


def fk(q):
    """Joint 1 lifts the TCP (m), joint 2 tilts it about world x (rad)."""
    q = np.asarray(q, float)
    return np.array([0.3, 0.0, 0.3 + q[1], q[2], 0.0, 0.0])


class Robot:
    def __init__(self, width=0.013, slip_at=None):
        self.q = np.zeros(3)
        self.moves, self.width, self.slip_at = [], width, slip_at
        self.forward_kinematics = fk

    def state(self):
        return SimpleNamespace(q=self.q.copy(), tcp=fk(self.q))

    def prepare_motion(self):
        return True

    def move_gripper(self, width, force=None):
        self.force = force

    @property
    def gripper_width(self):
        if self.slip_at is not None and len(self.moves) >= self.slip_at:
            return self.width + 0.01
        return self.width

    def move_j(self, q, speed=None, wait=True, use_native=True):
        self.moves.append((list(q), use_native))
        self.q = np.asarray(q, float)
        return True

    def stop_motion(self):
        self.stopped = True


class Camera:
    opened = True

    def snapshot(self):
        return SimpleNamespace(rgb=np.zeros((2, 2, 3), np.uint8), age=lambda: 0.0)

    def close(self):
        pass


def session():
    return SimpleNamespace(camera=Camera(), detector=SimpleNamespace(detect=lambda image: True))


def options(**kw):
    base = dict(quality_protocol=True, settle_s=0.0, min_samples=2, require_initial_board=True)
    base.update(kw)
    return CalibrateOptions(**base)


def run(stations, abort):
    return dict(source='test', stations=[dict(id=f's{i}', q_rad=q) for i, q in enumerate(stations)],
                start_q=None, abort_on_miss=abort, max_joint_error_rad=None)


def test_held_object_slack_covers_tilt_and_lift():
    start = np.zeros(3)
    assert held_object_slack(fk, fk(start), start, [0, .05, 0], .5) == pytest.approx(0.0)
    tilt = held_object_slack(fk, fk(start), start, [0, 0, .2], .5)
    assert tilt == pytest.approx(-.5 * 2 * np.sin(.1), abs=1e-6)
    assert held_object_slack(fk, fk(start), start, [0, .11, .2], .5) == pytest.approx(0.0)
    assert held_object_slack(fk, fk(start), start, [0, .09, .2], .5) < 0


def test_screen_rejects_lowering_held_object_and_chains_from_accepted():
    robot = Robot()
    cfg = options(held_object_radius_m=.5)
    poses = [[0, .02, 0], [0, .02, .2], [0, .12, .2]]
    accepted, reasons = auto.screen(robot, robot.state(), poses, cfg)
    assert accepted == [poses[0], poses[2]]
    assert reasons == [None, 'held_object_height', None]
    accepted, _ = auto.screen(robot, robot.state(), poses, options(held_object_radius_m=.5,
                                                                   held_object_drop_m=.2))
    assert accepted == poses


def test_fixed_height_reference_does_not_move_with_the_start_pose():
    robot = Robot()
    robot.q = np.array([0, -.03, 0])
    cfg = options(held_object_radius_m=.5, held_object_drop_m=.04, held_object_reference_q=[0, 0, 0])
    accepted, reasons = auto.screen(robot, robot.state(), [[0, -.035, 0], [0, -.045, 0]], cfg)
    assert reasons == [None, 'held_object_height']


def test_sweep_skips_missed_station_and_uses_host_interpolation(monkeypatch):
    captures = iter([{'board_jitter_px': .1}, None, {'board_jitter_px': .2}])
    monkeypatch.setattr(auto, 'capture_station', lambda *a, **k: next(captures))
    robot = Robot()
    views, failure = auto.run_stations(robot, Camera(), session(),
                                       options(held_object_radius_m=.5, clamp_force_n=2.0),
                                       run([[0, .01, 0], [0, .02, 0], [0, .03, 0]], False), 2)
    assert failure is None
    assert [v['view_id'] for v in views] == ['s0', 's2']
    assert [v['role'] for v in views] == ['train', 'train']
    assert robot.force == 2.0
    assert all(native is False for _, native in robot.moves)


def test_plan_aborts_on_miss_and_slip_stops(monkeypatch):
    monkeypatch.setattr(auto, 'capture_station', lambda *a, **k: None)
    robot = Robot()
    views, failure = auto.run_stations(robot, Camera(), session(), options(),
                                       run([[0, .01, 0], [0, .02, 0]], True), 2)
    assert views == [] and 'aborted' in failure and len(robot.moves) == 1
    assert robot.moves[0][1] is True

    monkeypatch.setattr(auto, 'capture_station', lambda *a, **k: {'board_jitter_px': .1})
    robot = Robot(slip_at=2)
    views, failure = auto.run_stations(robot, Camera(), session(), options(clamp_force_n=2.0),
                                       run([[0, .01, 0], [0, .02, 0], [0, .03, 0]], False), 3)
    assert len(views) == 1 and 'slipping' in failure


def test_no_motion_without_visible_board():
    robot = Robot()
    blind = SimpleNamespace(camera=Camera(), detector=SimpleNamespace(detect=lambda image: None))
    monkey = auto._wait_board
    auto._wait_board = lambda camera, detector, timeout=8.0: False
    try:
        views, failure = auto.run_stations(robot, Camera(), blind, options(),
                                           run([[0, .01, 0]], False), 1)
    finally:
        auto._wait_board = monkey
    assert views == [] and 'nothing was moved' in failure and robot.moves == []


def test_sweep_poses_reproducible_and_keep_base_yaw():
    cfg = options(pose_count=5, pose_delta_rad=.3, pose_seed=7)
    a, b = auto.sweep_poses([0.1, 0.2, 0.3], cfg), auto.sweep_poses([0.1, 0.2, 0.3], cfg)
    assert a == b and len(a) == 5 and all(p[0] == 0.1 for p in a)
