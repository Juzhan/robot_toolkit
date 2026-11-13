"""Simulated depth-based obstacle avoidance for the Piper X wrist camera (MuJoCo).

The arm (kinematic: commanded joints are applied directly) carries the wrist
camera at the hand-eye calibration pose. A box stands on the table; the arm is
asked to go somewhere that would hit it. Every camera frame the rendered depth
goes through ``DepthObstacleMap`` (self-filter -> map -> distance grid) and each
new joint goal is gated with ``CollisionGuard.safe_fraction`` exactly like the
PBVS loop; the 100 Hz reference is hard-checked every cycle.

    MUJOCO_GL=egl python robot_toolkit/examples/sim_depth_avoidance.py [--backend robokit]

Prints the closest true (ground-truth geometry) and mapped clearances. No hardware.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from robot_toolkit.core.collision import Box, CollisionGuard, Plane, d435_spheres
from robot_toolkit.control.guarded import CollisionBrake
from robot_toolkit.core.continuous import ContinuousReference
from robot_toolkit.errors import SafetyViolation
from robot_toolkit.obstacle_map import DepthObstacleMap

ROOT = Path(__file__).resolve().parents[1]
URDF = ROOT / 'assets' / 'piperx_assets' / 'assets' / 'piper_x.urdf'
START = np.array([0.179, 1.159, -1.051, 1.071, -0.011, 0.074])   # logged servo pose
WIDTH, HEIGHT, FY = 640, 480, 559.


class SimCamera:
    """MuJoCo scene: Piper X URDF + wrist camera + table + box obstacles."""

    def __init__(self, link_camera, boxes, width=WIDTH, height=HEIGHT, fy=FY):
        import mujoco
        from scipy.spatial.transform import Rotation

        self.mujoco = mujoco
        spec = mujoco.MjSpec.from_file(str(URDF))
        link6 = next(body for body in spec.bodies if body.name == 'link6')
        camera = link6.add_camera()
        camera.name, camera.pos = 'wrist', link_camera[:3, 3]
        # OpenCV optical frame (z forward, y down) -> MuJoCo camera (looks along -z, y up).
        x, y, z, w = Rotation.from_matrix(link_camera[:3, :3] @ np.diag([1., -1., -1.])).as_quat()
        camera.quat = [w, x, y, z]
        camera.fovy = np.degrees(2 * np.arctan(height / 2 / fy))
        floor = spec.worldbody.add_geom()
        floor.type, floor.size = mujoco.mjtGeom.mjGEOM_PLANE, [2., 2., .01]
        for center, size in boxes:
            geom = spec.worldbody.add_geom()
            geom.type, geom.pos, geom.size = mujoco.mjtGeom.mjGEOM_BOX, center, np.asarray(size) / 2
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height, width)
        self.renderer.enable_depth_rendering()
        self.K = np.array([[fy, 0., width / 2 - .5], [0., fy, height / 2 - .5], [0., 0., 1.]])

    def depth(self, q):
        self.data.qpos[:6] = q
        self.mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, camera='wrist')
        return self.renderer.render().astype(np.float32)

    def _marker(self, scene, kind, size, pos, rgba):
        if scene.ngeom >= scene.maxgeom:
            return
        self.mujoco.mjv_initGeom(scene.geoms[scene.ngeom], kind, np.asarray(size, float),
                                 np.asarray(pos, float), np.eye(3).ravel(),
                                 np.asarray(rgba, np.float32))
        scene.ngeom += 1

    def views(self, q, points, goal, closest):
        """Third-person view (mapped obstacle points red, goal green, closest robot
        sphere yellow) and the wrist RGB view."""
        mujoco = self.mujoco
        if not hasattr(self, 'outside'):
            self.outside = mujoco.Renderer(self.model, 480, 640, max_geom=3000)
            self.wrist = mujoco.Renderer(self.model, 240, 320)
            self.free = mujoco.MjvCamera()
            self.free.lookat[:] = [.30, .0, .15]
            self.free.distance, self.free.azimuth, self.free.elevation = .95, 215., -22.
        self.data.qpos[:6] = q
        mujoco.mj_forward(self.model, self.data)
        self.outside.update_scene(self.data, camera=self.free)
        scene = self.outside.scene
        for point in points[np.linspace(0, len(points) - 1, min(len(points), 1500)).astype(int)] \
                if len(points) else []:
            self._marker(scene, mujoco.mjtGeom.mjGEOM_SPHERE, [.004] * 3, point, [1, .1, .1, .9])
        self._marker(scene, mujoco.mjtGeom.mjGEOM_SPHERE, [.015] * 3, goal, [.1, 1, .1, .8])
        if closest is not None:
            self._marker(scene, mujoco.mjtGeom.mjGEOM_SPHERE, [closest[1]] * 3, closest[0],
                         [1, .9, .1, .45])
        outside = self.outside.render()
        self.wrist.update_scene(self.data, camera='wrist')
        return outside, self.wrist.render()


def position_ik(chain, q, target, link='link6', iterations=60):
    """Small damped least-squares IK on the link position (orientation free)."""
    q = np.asarray(q, float).copy()
    for _ in range(iterations):
        error = target - chain.link_poses(q)[link][:3, 3]
        if np.linalg.norm(error) < 1e-4:
            break
        jacobian = np.empty((3, len(q)))
        for i in range(len(q)):
            step = np.zeros_like(q)
            step[i] = 1e-6
            jacobian[:, i] = (chain.link_poses(q + step)[link][:3, 3]
                              - chain.link_poses(q)[link][:3, 3]) / 1e-6
        q += jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + 1e-4 * np.eye(3), error)
    return q


def run(backend='numpy', move=(.15, 0., -.08), box=((.42, .1, .05), (.1, .1, .1)),
        seconds=4., clearance=.015, verbose=True, record=None):
    calibration = ROOT.parent / 'outputs/robot_tasks/piper_calibrate_auto/calibration.json'
    if calibration.exists():
        link_camera = np.array(json.loads(calibration.read_text())
                               ['cameras']['wrist_camera']['tcp_camera'])
    else:  # the calibrated mount, rounded
        link_camera = np.array([[1., -.056, -.035, -.035], [.065, .935, .348, -.075],
                                [.013, -.35, .937, .015], [0., 0., 0., 1.]])
    attached = {'link6': d435_spheres(link_camera)}
    guard = CollisionGuard.from_config({}, attached=attached)          # depth only
    truth = CollisionGuard.from_config({}, attached=attached)
    truth.robot.radii = truth.robot.radii - truth.robot.padding_m        # unpadded geometry
    truth.obstacles = [Box(*box, name='box'), Plane([0, 0, 0], [0, 0, 1], name='table',
                                                     ignore_links=['link1'])]
    camera = SimCamera(link_camera, [box])
    depth_map = DepthObstacleMap(guard, link_camera, [-.2, -.5, -.02], [.7, .5, .7],
                                 voxel=.01, backend=backend, clock=lambda: sim_t)
    chain = guard.robot.chain
    goal_target = chain.link_poses(START)['link6'][:3, 3] + move
    final_goal = position_ik(chain, START, goal_target)
    dt, frame_dt, latency = .01, 1 / 30, .03
    reference = ContinuousReference(START, dt=dt, speed=1.2, acceleration=4, jerk=30,
                                    lower=[-3] * 6, upper=[3] * 6, backend='ruckig')
    brake = CollisionBrake(guard, 4, 30, dt, clearance)
    goal, history = START.copy(), []
    sim_t, next_frame, blocked, map_ms = 0., 0., 0, []
    closest_truth = closest_map = np.inf
    closest_pair = None
    while sim_t < seconds:
        q_ref = reference.step(goal)
        history.append((sim_t, q_ref.copy()))
        stop = brake.brake_goal(q_ref, reference.v, reference.a, goal)   # + hard floor
        if stop is not None:
            goal = stop
        closest_truth = min(closest_truth, truth.clearance(q_ref)[0])
        mapped = guard.clearance(q_ref)
        if mapped[0] < closest_map:
            closest_map, closest_pair = mapped[0], mapped[1]
        if sim_t >= next_frame:
            next_frame += frame_dt
            seen = min(history, key=lambda item: abs(item[0] - (sim_t - latency)))[1]
            started = time.perf_counter()
            depth_map.update(camera.depth(seen), camera.K, seen, stamp=sim_t)
            map_ms.append((time.perf_counter() - started) * 1000)
            fraction = guard.safe_fraction(q_ref, final_goal, clearance)
            blocked += fraction < 1.
            goal = q_ref + (final_goal - q_ref) * fraction
            if record is not None:
                record.append(_frame(camera, depth_map, guard, truth, q_ref, goal_target,
                                     sim_t, fraction < 1., backend, move))
        sim_t += dt
    reached = np.linalg.norm(chain.link_poses(q_ref)['link6'][:3, 3] - goal_target)
    result = {'backend': backend, 'move_m': list(move), 'blocked_frames': int(blocked),
              'brakes': brake.brakes,
              'tcp_short_of_goal_m': round(float(reached), 4),
              'closest_true_clearance_m': round(float(closest_truth), 4),
              'closest_mapped_clearance_m': round(float(closest_map), 4),
              'closest_mapped_link': closest_pair,
              'map_update_ms_p50': round(float(np.median(map_ms)), 1),
              'map_update_ms_max': round(float(np.max(map_ms)), 1)}
    if verbose:
        print(json.dumps(result))
    return result


def _frame(camera, depth_map, guard, truth, q, goal_target, sim_t, blocked, backend, move):
    import cv2

    centers = guard.robot.spheres(q)
    mapped, link, _ = guard.clearance(q)
    gaps = [obstacle.distance(centers) - guard.robot.radii for obstacle in guard.dynamic]
    closest = None
    if gaps and np.isfinite(np.min(gaps[0])):
        index = int(np.argmin(gaps[0]))
        closest = (centers[index], guard.robot.radii[index])
    points = getattr(depth_map, 'points', np.empty((0, 3)))
    outside, wrist = camera.views(q, points, goal_target, closest)
    depth = camera.depth(q)
    filtered, _, _, mask = depth_map.filtered_depth(depth, camera.K, q)
    shade = cv2.applyColorMap(np.uint8(np.clip(filtered / .6, 0, 1) * 255), cv2.COLORMAP_TURBO)
    shade[mask] = (90, 40, 40)                       # self-filtered pixels (the gripper)
    shade = cv2.resize(shade, (320, 240), interpolation=cv2.INTER_NEAREST)
    image = np.zeros((480, 960, 3), np.uint8)
    image[:, :640] = outside
    image[:240, 640:] = wrist
    image[240:, 640:] = shade[..., ::-1]
    true = truth.clearance(q)[0]
    lines = [f'{backend}  move {list(move)}  t={sim_t:4.2f}s',
             f'mapped clearance {mapped*1000:5.1f} mm ({link})',
             f'true gap {true*1000:5.1f} mm (stops at ~45: pad 10 + margin 20 + band 15)',
             'BLOCKED: holding at the edge' if blocked else 'free: following the goal']
    for i, text in enumerate(lines):
        cv2.putText(image, text, (10, 24 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, .55,
                    (255, 80, 80) if i == 3 and blocked else (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(image, 'wrist RGB', (648, 18), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 0), 1)
    cv2.putText(image, 'depth (dark = own gripper, filtered)', (648, 258),
                cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
    return image


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--backend', choices=['numpy', 'robokit'], default='numpy')
    parser.add_argument('--seconds', type=float, default=4.)
    parser.add_argument('--video', type=Path, default=None,
                        help='write an mp4 of all runs (third-person + wrist RGB + depth)')
    args = parser.parse_args()
    frames = [] if args.video else None
    # Into the box top, sideways into its side, and a free move that must not be blocked.
    for move in ((.15, 0., -.08), (.12, -.12, -.1), (-.05, 0., .05)):
        try:
            run(args.backend, move, seconds=args.seconds, record=frames)
        except SafetyViolation as exc:
            print(json.dumps({'backend': args.backend, 'move_m': list(move),
                              'hard_stop': str(exc)}))
    if frames:
        import imageio

        args.video.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(args.video, frames, fps=30, quality=8)
        print('video:', args.video)
