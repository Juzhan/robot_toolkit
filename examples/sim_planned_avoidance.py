"""Planned obstacle avoidance in simulation: robokit plans around a box, the guard re-checks.

Scene: Piper X with the wrist camera, a table and a 10x10x18 cm box between the
start and the goal. Three runs from the same start to the same goal:

1. ``guard only``: the straight joint path, gated by ``CollisionGuard`` (stage A/B
   behaviour): it stops in front of the box.
2. ``planned, known box``: robokit plans around the box given as a virtual obstacle.
3. ``planned, seen box``: the box is only known from wrist depth
   (``DepthObstacleMap``, tabletop 2.5-D prior). With the start view alone the plan
   is refused (its hand would pass over unobserved table); after a survey scan of
   the area it is planned.

Every planned path is retimed and densely checked by the guard before it runs;
true clearance is measured against the real (unpadded) geometry.

    MUJOCO_GL=egl python -m robot_toolkit.examples.sim_planned_avoidance --video out.mp4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from robot_toolkit.control.joint_planner import PlanningFailed, RRTConnectPlanner
from robot_toolkit.core.collision import Box, CollisionGuard, Plane, d435_spheres
from robot_toolkit.examples.sim_depth_avoidance import ROOT, START, SimCamera, position_ik
from robot_toolkit.perception.obstacle_map import DepthObstacleMap, KdPoints

LOWER = [-2.618, 0., -2.967, -1.571, -1.571, -2.094]
UPPER = [2.618, 3.1416, 0., 1.571, 1.571, 2.094]
BOX = ((.30, -.05, .09), (.1, .1, .18))
TABLE = {'type': 'plane', 'name': 'table', 'point': [0, 0, 0], 'normal': [0, 0, 1],
         'ignore_links': ['link1']}


def link_camera():
    calibration = ROOT / 'configs/calibration/piper_x_wrist_d435.json'
    if calibration.exists():
        return np.array(json.loads(calibration.read_text())['cameras']['wrist_camera']['tcp_camera'])
    return np.array([[1., -.056, -.035, -.035], [.065, .935, .348, -.075],
                     [.013, -.35, .937, .015], [0., 0., 0., 1.]])


def make_plan(kind, guard, start, goal, checks=(), unknown=None):
    """``kind``: 'rrt' (CPU, toolkit) or 'robokit' (GPU). ``unknown``: points only the
    planner avoids (unobserved space)."""
    if kind == 'robokit':
        from robot_toolkit.control.robokit_planner import RobokitPlanner

        planner = RobokitPlanner(guard, LOWER, UPPER)
        samples = planner.plan(start, goal, checks=checks, avoid_points=unknown)
    else:
        planner = RRTConnectPlanner(guard, LOWER, UPPER)
        extra = []
        if unknown is not None and len(unknown):
            # The wrist camera maps the hand's space; unobserved columns only block the
            # hand (matching heightmap_check), not the upper arm passing high above.
            unobserved = KdPoints(unknown, .01 * np.sqrt(3) / 2, 'unobserved')
            unobserved.ignore_links = {'base_link', 'link1', 'link2', 'link3', 'link4'}
            extra = [unobserved]
        samples = planner.plan(start, goal, extra_obstacles=extra, checks=checks)
    return samples, planner.last


def main(video=None, verbose=True, planner_kind='rrt'):
    camera_mount = link_camera()
    attached = {'link6': d435_spheres(camera_mount)}
    truth = CollisionGuard.from_config({}, attached=attached)
    truth.robot.radii = truth.robot.radii - truth.robot.padding_m
    truth.obstacles = [Box(*BOX, name='box'), Plane([0, 0, 0], [0, 0, 1], name='table',
                                                     ignore_links=['link1'])]
    chain = truth.robot.chain
    tcp = chain.link_poses(START)['link6'][:3, 3]
    start = position_ik(chain, START, tcp + [0, .12, -.12])
    goal = position_ik(chain, START, tcp + [0, -.30, -.12])
    camera = SimCamera(camera_mount, [BOX])
    results, frames = [], []

    def play(name, guard, samples, points=np.empty((0, 3))):
        closest = min(truth.clearance(q)[0] for q in samples)
        reached = np.linalg.norm(chain.link_poses(samples[-1])['link6'][:3, 3]
                                 - chain.link_poses(goal)['link6'][:3, 3])
        results.append({'run': name, 'seconds': round(len(samples) * .01, 2),
                        'tcp_short_of_goal_m': round(float(reached), 4),
                        'closest_true_clearance_m': round(float(closest), 4)})
        if video is not None:
            goal_point = chain.link_poses(goal)['link6'][:3, 3]
            for q in samples[::3]:
                frames.append(_frame(camera, q, points, goal_point, name, truth))
            frames.extend([frames[-1]] * 20)

    # 1. Straight line, gated: as far as the guard allows.
    guard = CollisionGuard.from_config({'obstacles': [TABLE, {'type': 'box', 'name': 'box',
                                                              'center': BOX[0], 'size': BOX[1]}]},
                                       attached=attached)
    fraction = guard.safe_fraction(start, goal, .015)
    straight = start + (goal - start) * np.linspace(0, fraction, 200)[:, None]
    play('guard only (straight line)', guard, straight)

    # 2. Planned around the virtual box.
    samples, info = make_plan(planner_kind, guard, start, goal)
    play(f'{planner_kind}: planned, known box', guard, samples)
    results[-1].update(info)

    # 3./4. The box only known from wrist depth. Unseen space is not assumed free: a
    # plan whose hand passes through it is refused until a scan has looked there.

    # Without scanning (start view only) the plan is refused. A survey (the camera
    # looking straight down from 0.40 m over a grid between start and goal) maps the
    # area first; the same plan is then accepted.
    survey = [(x, y) for x in (.2, .3, .4) for y in (-.3, -.2, -.1, 0., .1, .2)]
    for label, spots in (('start view only', []), ('after a survey scan', survey)):
        seen = CollisionGuard.from_config({'obstacles': [TABLE]}, attached=attached)
        depth_map = DepthObstacleMap(seen, camera_mount, [-.3, -.6, -.02], [.8, .6, .7])
        depth_map.max_views = 32
        depth_map.update(camera.depth(start), camera.K, start, stamp=0.)
        looked = 0
        for stamp, spot in enumerate(spots, 1):
            look = look_pose(np.array(spot), camera_mount, start, .40)   # .44 is out of reach
            if look is not None and guard_free(seen, start, look):
                depth_map.update(camera.depth(look), camera.K, look, stamp=float(stamp))
                looked += 1
        points = depth_map.column_points(floor_z=0.)   # tabletop 2.5-D prior
        seen.set_dynamic([KdPoints(points, .01 * np.sqrt(3), 'seen box')])
        name = f'{planner_kind}: planned, box seen by depth ({label})'
        try:
            samples, info = make_plan(
                planner_kind, seen, start, goal,
                checks=[depth_map.heightmap_check(start)] if planner_kind == 'robokit' else (),
                unknown=depth_map.unknown_columns(start, [0., -.5], [.6, .5], voxel=.01))
        except PlanningFailed as exc:
            results.append({'run': name, 'refused': str(exc)[:200], 'looks': looked})
            continue
        play(name, seen, samples, points)
        results[-1].update(info, looks=looked, seen_points=len(points))

    if verbose:
        for row in results:
            print(json.dumps(row))
    if video is not None and frames:
        import imageio

        Path(video).parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(video, frames, fps=30, quality=8)
        print('video:', video)
    return results


_IK = []


def look_pose(spot, camera_mount, seed, height=.36):
    """Joints that put the camera ``height`` above ``spot`` looking straight down."""
    from robot_toolkit.core.mathutils import matrix_to_rotvec
    from robot_toolkit.robot.ik import RobokitIK

    if not _IK:
        _IK.append(RobokitIK(str(ROOT / 'assets/piperx_assets/assets/piper_x.urdf'), 'link6',
                             [f'joint{i}' for i in range(1, 7)]))
    for yaw in np.linspace(0, 2 * np.pi, 8, endpoint=False):
        camera = np.eye(4)
        camera[:3, :3] = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0],
                                   [0, 0, 1]]) @ np.diag([1., -1., -1.])   # optical z down
        camera[:3, 3] = [spot[0], spot[1], height]
        link6 = camera @ np.linalg.inv(camera_mount)
        q = _IK[0].solve(np.r_[link6[:3, 3], matrix_to_rotvec(link6[:3, :3])], seed=seed)
        if len(q) and np.all(q >= LOWER) and np.all(q <= UPPER):
            return np.asarray(q, float)
    return None


def guard_free(guard, start, look):
    """The scan pose is clear of everything mapped so far (a real arm would also plan
    the move there; here the camera is placed directly)."""
    return guard.clearance(look)[0] > guard.margin_m + .02


def _frame(camera, q, points, goal, title, truth):
    import cv2

    outside, wrist = camera.views(q, points, goal, None)
    image = np.zeros((480, 960, 3), np.uint8)
    image[:, :640] = outside
    image[:240, 640:] = wrist
    lines = [title, f'true clearance {truth.clearance(q)[0]*1000:5.1f} mm']
    for i, text in enumerate(lines):
        cv2.putText(image, text, (10, 26 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, .65,
                    (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, 'wrist RGB', (648, 18), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 0), 1)
    return image


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--video', type=Path, default=None)
    parser.add_argument('--planner', choices=['rrt', 'robokit'], default='rrt',
                        help='rrt: CPU RRT-Connect on the guard; robokit: GPU trajectory optimiser')
    args = parser.parse_args()
    main(args.video, planner_kind=args.planner)
