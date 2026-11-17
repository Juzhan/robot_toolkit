import numpy as np
import pytest

from robot_toolkit.control.joint_planner import PlanningFailed, RRTConnectPlanner
from robot_toolkit.core.collision import CollisionGuard, PointCloud

LOWER = [-2.618, 0., -2.967, -1.571, -1.571, -2.094]
UPPER = [2.618, 3.1416, 0., 1.571, 1.571, 2.094]
TABLE = {'type': 'plane', 'point': [0, 0, 0], 'normal': [0, 0, 1], 'ignore_links': ['link1']}
BOX = {'type': 'box', 'name': 'box', 'center': [.30, -.05, .09], 'size': [.1, .1, .18]}
# Hand low on either side of the box (from the planned-avoidance simulation).
START = np.array([0.588, 1.52, -0.97, 0.978, -0.068, 0.074])
GOAL = np.array([-0.804, 1.655, -1.107, 0.777, 0.096, 0.074])


def guard(*obstacles):
    return CollisionGuard.from_config({'obstacles': [TABLE, *obstacles]})


def test_straight_line_blocked_but_planned_around_the_box():
    boxed = guard(BOX)
    boxed.check_position(START)
    boxed.check_position(GOAL)
    assert boxed.safe_fraction(START, GOAL, .01) < 1
    planner = RRTConnectPlanner(boxed, LOWER, UPPER, seed=1)
    samples = planner.plan(START, GOAL)
    np.testing.assert_allclose(samples[0], START, atol=1e-9)
    np.testing.assert_allclose(samples[-1], GOAL, atol=1e-9)
    assert boxed.clearances(samples).min() >= boxed.margin_m
    assert np.max(np.abs(np.diff(samples, axis=0))) / .01 <= 1.0 + 1e-6   # velocity limit
    assert planner.last['plan_time_s'] < 10


def test_free_space_is_a_direct_move_and_bad_goals_fail_fast():
    free = guard()
    samples = RRTConnectPlanner(free, LOWER, UPPER).plan(START, GOAL)
    assert free.clearances(samples).min() >= free.margin_m
    into_table = np.array([0., 2.4, -.6, 0., .3, 0.])     # shoulder far forward: hand in the table
    with pytest.raises(PlanningFailed, match='goal'):
        RRTConnectPlanner(guard(BOX), LOWER, UPPER).plan(START, into_table)


def test_planner_only_obstacles_are_respected():
    free = guard()
    poses = free.robot.chain
    middle = (poses.link_poses(START)['link6'][:3, 3] + poses.link_poses(GOAL)['link6'][:3, 3]) / 2
    # A short post between start and goal that only the planner knows about.
    wall = PointCloud(middle + np.c_[np.zeros(20), np.zeros(20), np.linspace(-.12, .05, 20)],
                      .02, 'unobserved')
    planner = RRTConnectPlanner(free, LOWER, UPPER, seed=2)
    samples = planner.plan(START, GOAL, extra_obstacles=[wall])
    assert not free.dynamic                              # the guard itself is unchanged
    free.set_dynamic([wall])
    assert free.safe_fraction(START, GOAL) < 1           # the straight move would hit it
    assert free.clearances(samples).min() >= free.margin_m
