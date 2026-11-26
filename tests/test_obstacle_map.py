import os

import numpy as np
import pytest

from robot_toolkit.control.guarded import CollisionBrake
from robot_toolkit.core.collision import CollisionGuard, DistanceGrid, d435_spheres
from robot_toolkit.errors import SafetyViolation
from robot_toolkit.perception.obstacle_map import DepthObstacleMap, depth_to_points, robot_pixel_mask

pytest.importorskip('scipy')
START = np.array([0.179, 1.159, -1.051, 1.071, -0.011, 0.074])   # camera looks down
LINK_CAMERA = np.array([[.9978, -.0563, -.0346, -.0352], [.0648, .9352, .3481, -.0751],
                        [.0128, -.3496, .9368, .0154], [0., 0., 0., 1.]])
K = np.array([[280., 0., 159.5], [0., 280., 119.5], [0., 0., 1.]])


def plane_depth(T_base_camera, height, shape=(240, 320)):
    """z-depth image of the horizontal plane z = ``height`` (base frame)."""
    v, u = np.mgrid[0:shape[0], 0:shape[1]]
    rays = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones(shape)], -1)
    world = rays @ T_base_camera[:3, :3].T
    t = (height - T_base_camera[2, 3]) / world[..., 2]
    return np.where(t > 0, t, 0.).astype(np.float32)


def test_depth_to_points_round_trip():
    depth = np.full((4, 6), 1.)
    depth[0, 0] = 0.                      # invalid pixel dropped
    depth[0, 2] = 3.                      # beyond depth_max
    points = depth_to_points(depth, [[100, 0, 3], [0, 100, 2], [0, 0, 1]])
    assert len(points) == 22 and np.allclose(points[:, 2], 1.)
    np.testing.assert_allclose(points[0], [(1 - 3) / 100, (0 - 2) / 100, 1.])


def test_robot_pixel_mask_removes_only_points_on_the_robot():
    sphere = [[0., 0., .5]]
    depth = np.full((240, 320), 1.)                   # background behind the sphere: kept
    assert not robot_pixel_mask(depth, K, sphere, [.1]).any()
    depth[118:123, 158:163] = .45                     # the arm's own surface: removed
    mask = robot_pixel_mask(depth, K, sphere, [.1])
    assert mask[120, 160] and mask.sum() == 25
    depth[118:123, 158:163] = .2                      # an object in front of the arm: kept
    assert not robot_pixel_mask(depth, K, sphere, [.1]).any()
    assert not robot_pixel_mask(np.full((240, 320), .02), K, [[0., 0., .01]], [.05]).any()


def test_distance_grid_is_a_lower_bound():
    grid = DistanceGrid([0, 0, 0], .1, np.full((3, 3, 3), .5))
    assert grid.distance(np.array([[.1, .1, .1]]))[0] == pytest.approx(.5 - .1 * np.sqrt(3) / 2)
    assert grid.distance(np.array([[5., 0, 0]]))[0] == np.inf


def make_map(**kwargs):
    guard = CollisionGuard.from_config({}, attached={'link6': d435_spheres(LINK_CAMERA)})
    return guard, DepthObstacleMap(guard, LINK_CAMERA, [-.2, -.5, -.05], [.7, .5, .7],
                                   voxel=.01, stride=1, **kwargs)


def test_numpy_map_matches_a_seen_table_and_filters_the_gripper():
    guard, depth_map = make_map()
    T = depth_map.camera_pose(START)
    depth = plane_depth(T, .05)
    # Pretend the gripper is in view: sphere pixels in front of the table.
    centers = (guard.robot.spheres(START) - T[:3, 3]) @ T[:3, :3]
    radii = guard.robot.radii - guard.robot.padding_m
    # Pretend the gripper is seen: pixels at the sphere centres' depth along their rays.
    v, u = np.mgrid[0:depth.shape[0], 0:depth.shape[1]]
    rays = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones(depth.shape)], -1)
    for center, radius in zip(centers, radii):
        if center[2] <= .02:
            continue
        hit = (np.linalg.norm(np.cross(rays, center), axis=-1) / np.linalg.norm(rays, axis=-1)
               < radius * .5)
        depth[hit] = np.minimum(depth[hit], center[2])
    own = robot_pixel_mask(depth, K, centers, radii)
    assert own.any()
    depth_map.update(depth, K, START, stamp=0.)
    # Compare the hand only: the camera cannot see the table under the upper arm.
    hand = {'ignore_links': ['base_link', 'link1', 'link2', 'link3', 'link4', 'link5']}
    truth = CollisionGuard.from_config({**hand, 'obstacles': [
        {'type': 'plane', 'point': [0, 0, .05], 'normal': [0, 0, 1]}]},
        attached={'link6': d435_spheres(LINK_CAMERA)})
    guard.ignore_links, guard._masks = set(hand['ignore_links']), {}
    mapped, link, name = guard.clearance(START)
    true = truth.clearance(START)[0]
    # Conservative (never more than the truth) but within the voxel slack; no self hits.
    assert name == 'depth' and true - .03 < mapped <= true + 1e-6


def test_numpy_map_remembers_hidden_points_and_carves_seen_free_space():
    guard, depth_map = make_map()
    T = depth_map.camera_pose(START)
    depth = plane_depth(T, .05)
    depth_map.update(depth, K, START, stamp=0.)
    before = len(depth_map.points)
    depth_map.update(np.zeros_like(depth), K, START, stamp=1.)   # nothing valid: keep all
    assert len(depth_map.points) == before
    depth_map.update(np.zeros_like(depth), K, START, stamp=20.)  # too old to trust
    assert len(depth_map.points) == 0
    depth_map.update(depth, K, START, stamp=21.)
    depth_map.update(plane_depth(T, -.04), K, START, stamp=22.)  # the object at .05 left
    assert depth_map.points[:, 2].max() < -.03


def test_collision_brake_stops_short_of_the_band():
    class Wall(CollisionGuard):
        def __init__(self):
            self.margin_m, self.step_rad = .02, .01

        def clearance(self, q):
            return .5 - q[0], 'link6', 'wall'

        def clearances(self, qs):
            return .5 - np.asarray(qs)[:, 0]

    brake = CollisionBrake(Wall(), accel=4, jerk=30, dt=.01, clearance_m=.015)
    q, v = np.zeros(6), np.r_[1.2, np.zeros(5)]
    assert brake.brake_goal(q, v, np.zeros(6), q + 1) is None          # far: keep going
    goal = brake.brake_goal(np.r_[.3, np.zeros(5)], v, np.zeros(6), q + 1)
    assert goal is not None and goal[0] <= .5 - .035 + 1e-9
    assert brake.brake_goal(np.r_[.3, np.zeros(5)], v, np.zeros(6), goal) is None  # once
    with pytest.raises(SafetyViolation):
        brake.brake_goal(np.r_[.49, np.zeros(5)], v, np.zeros(6), q + 1)
    assert brake.speed_scale(np.r_[.3, np.zeros(5)], -v) == 1.          # moving away
    assert 0 < brake.speed_scale(np.r_[.3, np.zeros(5)], v) < 1.
    assert brake.speed_scale(np.r_[.47, np.zeros(5)], v) == 0.


@pytest.mark.skipif(os.environ.get('MUJOCO_GL') is None, reason='set MUJOCO_GL=egl to render')
def test_simulated_depth_avoidance_stops_before_the_box():
    pytest.importorskip('mujoco')
    pytest.importorskip('ruckig')
    from robot_toolkit.examples.sim_depth_avoidance import run

    blocked = run('numpy', (.15, 0., -.08), seconds=3., verbose=False)
    assert blocked['blocked_frames'] > 0 and blocked['closest_true_clearance_m'] > .02
    free = run('numpy', (-.05, 0., .05), seconds=2., verbose=False)
    assert free['blocked_frames'] == 0 and free['tcp_short_of_goal_m'] < .005


def test_column_prior_and_heightmap_check():
    guard, depth_map = make_map()
    T = depth_map.camera_pose(START)
    depth_map.update(plane_depth(T, .0), K, START, stamp=0.)
    seen_xy = depth_map.points[:, :2]
    depth_map.points = np.vstack([depth_map.points, [[seen_xy[0, 0], seen_xy[0, 1], .1]]])
    columns = depth_map.column_points()
    assert (np.abs(columns[:, :2] - seen_xy[0]).max(axis=1) < 1e-9).sum() >= 9  # 0.1 m tall
    check = depth_map.heightmap_check(START)
    assert check(np.array([START, START])) is None                 # the arm's own spot
    far = START + [2.2, 0, 0, 0, 0, 0]                                 # swung over unseen table
    assert 'unobserved' in check(np.array([START, far])) and check.spot is not None
    assert len(depth_map.unknown_columns(START, [-.5, -.5], [.5, .5])) > 0


def test_self_collision_ignore_keeps_distant_pairs():
    from robot_toolkit.core.collision import self_collision_ignore
    guard, _ = make_map()
    lower = [-2.618, 0., -2.967, -1.571, -1.571, -2.094]
    upper = [2.618, 3.1416, 0., 1.571, 1.571, 2.094]
    ignore = self_collision_ignore(guard.robot, lower, upper, samples=300)
    pairs = {(a, b) for a, items in ignore.items() for b in items}
    assert ('link1', 'link2') in pairs and ('link6', 'gripper_base') in pairs   # adjacent
    assert ('link2', 'gripper_base') not in pairs and ('base_link', 'gripper_base') not in pairs


@pytest.mark.skipif(not os.environ.get('ROBOKIT_CUDA_TESTS'),
                    reason='set ROBOKIT_CUDA_TESTS=1 (robokit + CUDA) to run the planner')
def test_robokit_planner_goes_around_a_box():
    pytest.importorskip('robokit')
    from robot_toolkit.adapters.robokit_planner import RobokitPlanner
    from robot_toolkit.examples.sim_depth_avoidance import position_ik

    guard = CollisionGuard.from_config({'obstacles': [
        {'type': 'plane', 'point': [0, 0, 0], 'normal': [0, 0, 1], 'ignore_links': ['link1']},
        {'type': 'box', 'center': [.30, -.05, .09], 'size': [.1, .1, .18]}]},
        attached={'link6': d435_spheres(LINK_CAMERA)})
    chain = guard.robot.chain
    tcp = chain.link_poses(START)['link6'][:3, 3]
    start = position_ik(chain, START, tcp + [0, .12, -.12])
    goal = position_ik(chain, START, tcp + [0, -.30, -.12])
    assert guard.safe_fraction(start, goal, .015) < .5                # straight line is blocked
    lower = [-2.618, 0., -2.967, -1.571, -1.571, -2.094]
    upper = [2.618, 3.1416, 0., 1.571, 1.571, 2.094]
    samples = RobokitPlanner(guard, lower, upper).plan(start, goal)
    np.testing.assert_allclose(samples[0], start, atol=1e-6)
    np.testing.assert_allclose(samples[-1], goal, atol=.01)
    assert guard.clearances(samples).min() >= guard.margin_m
