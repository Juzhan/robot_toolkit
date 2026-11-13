"""robokit ``MotionPlanner`` behind a NumPy interface, re-verified by the toolkit guard.

robokit (Warp/CUDA, lazy import) plans a collision-free joint path around the
obstacles of a ``core.collision.CollisionGuard``; the toolkit then retimes it to
its own velocity/acceleration/jerk limits and checks every sample densely with
the same guard before anything may run. robokit's own success scoring skips the
goal frame and checks edges sparsely, so its plan is a proposal, never trusted
as is.

Robot geometry is the guard's: its padded spheres (including attached tool or
camera spheres) are written to robokit's YAML, plus a self-collision ignore set
computed from them. Obstacles: ``Plane``/``Box``/``Sphere`` map to robokit
geoms; dynamic depth obstacles (``KdPoints`` / ``DistanceGrid``) become an
``SdfVolume``.
"""

from __future__ import annotations

import copy
import tempfile
from pathlib import Path

import numpy as np

from ..control.trajectory import TrajectoryLimits, retime
from ..core.collision import Box, Plane, Sphere, self_collision_ignore
from ..errors import SafetyViolation


class PlanningFailed(RuntimeError):
    pass


class _Points:
    def __init__(self, points):
        self.points, self.ignore_links = np.asarray(points, float).reshape(-1, 3), set()


class RobokitPlanner:
    """Plan for the six arm joints of a URDF arm guarded by ``guard``.

    ``plan(q_start, goal)``: ``goal`` is a joint vector or a 4x4 base-frame pose
    of ``target_link``. Returns joint samples every ``dt`` that respect
    ``limits`` and keep ``guard.margin_m`` (+ ``clearance_m``) everywhere, or
    raises ``PlanningFailed``.
    """

    def __init__(self, guard, lower, upper, target_link='link6', limits=None, dt=0.01,
                 clearance_m=0.005, device='cuda:0', attempts=3, seed=0):
        self.guard = guard
        self.lower, self.upper = np.asarray(lower, float), np.asarray(upper, float)
        self.target_link = target_link
        self.limits = limits or TrajectoryLimits(velocity=1.0, acceleration=3.0, jerk=20.0)
        self.dt, self.clearance_m = float(dt), float(clearance_m)
        self.device, self.attempts, self.seed = device, int(attempts), seed
        self._planners, self._robot, self._volume = {}, None, None
        self.last = {}

    # ------------------------------------------------------------------ setup
    def _load_robot(self):
        if self._robot is not None:
            return
        import warp as wp
        import yaml  # a robokit dependency
        from robokit.robo import Robot

        wp.config.quiet = True
        robot = self.guard.robot
        spheres = {}
        for link, center, radius in zip(robot.links, robot.local, robot.radii):
            spheres.setdefault(str(link), []).append(
                {'center': [round(float(c), 5) for c in center], 'radius': round(float(radius), 5)})
        ignore = self_collision_ignore(robot, self.lower, self.upper)
        self._dir = tempfile.TemporaryDirectory(prefix='robokit_planner_')
        root = Path(self._dir.name)
        (root / 'spheres.yaml').write_text(yaml.safe_dump({'collision_spheres': spheres}))
        (root / 'ignore.yml').write_text(yaml.safe_dump({'self_collision_ignore': ignore}))
        urdf = self.guard.robot.chain.urdf_path
        self._robot = Robot.load(str(urdf), load_collision_spheres=True,
                                 collision_spheres_path=root / 'spheres.yaml',
                                 self_collision_ignore_path=root / 'ignore.yml')
        names = list(self._robot.spec.actuated_joint_names)
        self._indices = [names.index(name) for name in self.guard.robot.chain.joint_names]
        self._dof = len(names)

    def _scene(self, extra_points=None):
        """robokit scene for the guard's current obstacles, and a layout key."""
        import warp as wp
        from robokit.geom import BoxGeom, PlaneGeom, SdfVolume, SphereGeom, VolumeGeom, WarpScene

        planes, boxes, spheres, clouds = [], [], [], []
        extra = [] if extra_points is None else [_Points(extra_points)]
        for obstacle in self.guard.obstacles + self.guard.dynamic + extra:
            if obstacle.ignore_links:
                # robokit has no per-obstacle link exclusion; the guard re-checks anyway.
                pass
            if isinstance(obstacle, Plane):
                planes.append(obstacle)
            elif isinstance(obstacle, Box):
                boxes.append(obstacle)
            elif isinstance(obstacle, Sphere):
                spheres.append(obstacle)
            elif hasattr(obstacle, 'points') or hasattr(obstacle, 'values'):
                clouds.append(obstacle)
        scene = WarpScene(1, self.device)

        def poses(matrices):
            return wp.array(np.asarray(matrices, np.float32), dtype=wp.mat44, device=self.device)

        for kind, items in (('plane', planes), ('box', boxes), ('sphere', spheres)):
            if not items:
                continue
            offsets = np.array([0, len(items)], np.int32)
            frames = []
            for item in items:
                T = np.eye(4)
                if kind == 'plane':
                    z = item.normal
                    x = np.cross([0, 1, 0], z) if abs(z[1]) < .9 else np.cross([1, 0, 0], z)
                    x /= np.linalg.norm(x)
                    T[:3, :3] = np.stack([x, np.cross(z, x), z], 1)
                    T[:3, 3] = item.point
                elif kind == 'box':
                    T[:3, :3], T[:3, 3] = item.rotation, item.center
                else:
                    T[:3, 3] = item.center
                frames.append(T)
            if kind == 'plane':
                scene.add(PlaneGeom(offsets, poses=poses(frames)))
            elif kind == 'box':
                scene.add(BoxGeom(np.array([b.half for b in boxes], np.float32), offsets,
                                  poses=poses(frames)))
            else:
                scene.add(SphereGeom(np.array([s.radius for s in spheres], np.float32),
                                     offsets, poses=poses(frames)))
        if clouds:
            parts = []
            for cloud in clouds:
                if hasattr(cloud, 'points'):
                    parts.append(np.asarray(cloud.points))
                else:  # DistanceGrid: occupied cells
                    cells = np.argwhere(cloud.values <= cloud.voxel * .87)
                    parts.append(cloud.origin + cells * cloud.voxel)
            points = np.concatenate(parts or [np.empty((0, 3))])
            lo = np.minimum(points.min(0) - .1, [-.6, -.6, -.05]) if len(points) else [-.6, -.6, -.05]
            hi = np.maximum(points.max(0) + .1, [.8, .6, .8]) if len(points) else [.8, .6, .8]
            self._volume = SdfVolume.from_points(points.astype(np.float32), np.asarray(lo),
                                                 np.asarray(hi), .01, self.device)
            scene.add(VolumeGeom([self._volume], np.array([0, 1], np.int32)))
        key = (len(planes), len(boxes), len(spheres), bool(clouds))
        return scene, key

    def _planner(self, joint_goal, extra_margin=.01, extra_points=None):
        from robokit.helpers.motion_plan import MotionPlanner, presets

        scene, key = self._scene(extra_points)
        key = key + (joint_goal,)
        # A fresh planner per call: obstacle sizes/poses are baked into the scene arrays.
        if key[3]:
            # Volume scenes are checked per frame only (no swept cost): use more frames
            # so a thin observed surface cannot fall between two of them.
            import dataclasses

            pose = presets.build_offline_pose_goal_config(num_timesteps=96)
            config = pose if not joint_goal else dataclasses.replace(pose, stages=(
                dataclasses.replace(pose.stages[0], terms=[
                    term for term in copy.deepcopy(pose.stages[0].terms) if term.name != 'target'],
                    lock_frames=(0, -1)),))
            config = copy.deepcopy(config)
        else:
            config = copy.deepcopy(presets.offline_joint_lbfgs if joint_goal
                                   else presets.offline_pose_lbfgs)
        # robokit's collision terms are soft costs with a 1 cm margin; aim beyond the
        # guard's hard margin so the dense re-check passes.
        margin = self.guard.margin_m + self.clearance_m + extra_margin
        for stage in config.stages:
            try:
                task = stage.task('collision')
            except (KeyError, ValueError, AttributeError):
                continue
            task.margin = margin
            if key[3]:
                # robokit's swept/speed-scaled cost is primitive-only; with a depth volume
                # it checks frames only. The guard's dense re-check covers the edges.
                task.sweep_steps, task.speed_dt = 0, None
        planner = MotionPlanner(config, self._robot, self.target_link, scene=scene,
                                device=self.device)
        mask = np.zeros(self._dof, bool)
        mask[self._indices] = True
        planner.set_active_joint_mask(mask)
        return planner

    # --------------------------------------------------------------- planning
    def _full(self, q):
        full = np.zeros(self._dof, np.float32)
        full[self._indices] = q
        return full

    def plan(self, q_start, goal, checks=(), avoid_points=None):
        """``checks``: extra ``check(samples) -> reason or None`` (e.g. a depth map's
        ``unobserved_check``) applied to the retimed samples after the guard.
        ``avoid_points``: extra points only the planner avoids (e.g. a depth map's
        ``unknown_points``), so it proposes paths through known-free space."""
        self._load_robot()
        q_start = np.asarray(q_start, float)
        self.guard.check_position(q_start)
        goal = np.asarray(goal, float)
        joint_goal = goal.shape == q_start.shape
        if joint_goal:
            self.guard.check_position(goal, self.clearance_m)
            target = self._full(goal)
        else:
            from ..core.mathutils import matrix_to_quat

            target = {self.target_link: np.concatenate(
                [goal[:3, 3], matrix_to_quat(goal[:3, :3], order='wxyz')]).astype(np.float32)}
        failures = []
        for attempt in range(self.attempts):
            # Each retry asks robokit for a wider berth (its collision cost is soft).
            planner = self._planner(joint_goal, .02 * (attempt + 1), avoid_points)
            result = planner.solve_offline_numpy([self._full(q_start), target])
            path = np.asarray(result.q_traj[0], float)[:, self._indices]
            path[0] = q_start
            reason = self._verify_goal(path[-1], goal, joint_goal)
            if reason is None:
                samples = retime(path, self.dt, self.limits)
                reason = self._verify_samples(samples)
                for check in checks:
                    reason = reason or check(samples)
                if reason is None:
                    self.last = {'attempt': attempt, 'frames': len(path),
                                 'samples': len(samples),
                                 'seconds': round(len(samples) * self.dt, 3),
                                 'min_clearance_m': float(self.guard.clearances(samples).min())}
                    return samples
            failures.append(reason)
        raise PlanningFailed('; '.join(failures))

    def _verify_goal(self, q, goal, joint_goal):
        if np.any(q < self.lower - 1e-6) or np.any(q > self.upper + 1e-6):
            return 'goal outside joint limits'
        if joint_goal:
            return None if np.max(np.abs(q - goal)) < .01 else 'joint goal not reached'
        T = self.guard.robot.chain.link_poses(q)[self.target_link]
        position = np.linalg.norm(T[:3, 3] - goal[:3, 3])
        angle = np.arccos(np.clip((np.trace(T[:3, :3].T @ goal[:3, :3]) - 1) / 2, -1, 1))
        return None if position < .005 and angle < .05 else \
            f'pose goal missed by {position*1000:.0f} mm / {angle:.2f} rad'

    def _verify_samples(self, samples):
        if np.any(samples < self.lower - 1e-6) or np.any(samples > self.upper + 1e-6):
            return 'path leaves joint limits'
        for a, b in zip(samples[:-1], samples[1:]):
            try:
                self.guard.check_path(a, b, self.clearance_m)
            except SafetyViolation as exc:
                return f'guard rejected the path: {exc}'
        return None
