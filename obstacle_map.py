"""Depth image -> obstacle distance grid for ``core.collision.CollisionGuard``.

Pipeline per depth frame (wrist camera, metric z-depth in metres):

1. pose: ``T_base_camera = FK(q at the frame time) @ link_camera``;
2. self-filter: pixels explained by the robot's own padded spheres (the wrist
   camera always sees the gripper) or hidden behind them are dropped;
3. fuse into a map over a fixed workspace box and turn it into a distance grid;
4. publish it as a ``DistanceGrid`` dynamic obstacle on the guard.

Backends: ``numpy`` (voxel-downsampled points over a short time window in a SciPy
k-d tree, lazy import; exact distances) and ``robokit`` (``robokit.geom.SdfVolume`` TSDF fusion with
free-space clearing on the GPU, via ``adapters.robokit_sdf``). Only robokit
remembers surfaces that later fall below the camera's minimum range.
"""

from __future__ import annotations

import time
import numpy as np

from .core.collision import DistanceGrid


def depth_to_points(depth, K, stride=1, depth_min=0.05, depth_max=1.5):
    """Camera-frame points (N, 3) of valid pixels of a metric z-depth image."""
    depth = np.asarray(depth, float)[::stride, ::stride]
    v, u = np.mgrid[0:depth.shape[0], 0:depth.shape[1]] * stride
    valid = np.isfinite(depth) & (depth > depth_min) & (depth < depth_max)
    z = depth[valid]
    K = np.asarray(K, float)
    x = (u[valid] - K[0, 2] - K[0, 1] * (v[valid] - K[1, 2]) / K[1, 1]) * z / K[0, 0]
    y = (v[valid] - K[1, 2]) * z / K[1, 1]
    return np.stack([x, y, z], axis=1)


def robot_pixel_mask(depth, K, centers_camera, radii, tolerance=0.01):
    """Pixels whose 3-D point lies on the robot itself (inside a sphere + ``tolerance``).

    Only those points are removed. A pixel seen past an over-sized sphere is a
    real observation of whatever is behind the true (thinner) link and is kept,
    as is anything between the camera and the arm.
    """
    depth = np.asarray(depth, float)
    K = np.asarray(K, float)
    h, w = depth.shape
    mask = np.zeros(depth.shape, bool)
    valid = np.isfinite(depth) & (depth > 0.)
    if not valid.any():
        return mask
    v, u = np.nonzero(valid)
    z = depth[v, u]
    y = (v - K[1, 2]) / K[1, 1]
    x = (u - K[0, 2] - K[0, 1] * y) / K[0, 0]
    points = np.stack([x * z, y * z, z], 1)
    inside = np.zeros(len(points), bool)
    for center, radius in zip(np.asarray(centers_camera, float), np.asarray(radii, float)):
        reach = radius + tolerance
        if np.linalg.norm(center) <= radius or center[2] + reach <= 0.:
            continue   # the camera's own body, or behind the image plane
        near = np.abs(points[:, 2] - center[2]) <= reach      # cheap pre-filter
        if near.any():
            gap = np.linalg.norm(points[near] - center, axis=1)
            inside[np.flatnonzero(near)[gap <= reach]] = True
    mask[v[inside], u[inside]] = True
    return mask


class KdPoints:
    """Obstacle from surface points with a SciPy k-d tree: exact nearest-point distance
    (minus ``radius`` for the sampling gap between neighbouring points)."""

    def __init__(self, points, radius, name='depth', stamp=None):
        from scipy.spatial import cKDTree  # optional dependency

        self.points = np.asarray(points, float).reshape(-1, 3)
        self.tree = cKDTree(self.points) if len(self.points) else None
        self.radius = float(radius)
        self.name, self.ignore_links, self.stamp = name, set(), stamp

    def distance(self, points):
        points = np.asarray(points, float)
        if self.tree is None:
            return np.full(points.shape[:-1], np.inf)
        gap, _ = self.tree.query(points.reshape(-1, 3))
        return gap.reshape(points.shape[:-1]) - self.radius


class DepthObstacleMap:
    """Turns wrist depth frames into a ``DistanceGrid`` on a ``CollisionGuard``.

    ``bounds_min/max``: workspace box in the base frame that is mapped (outside
    it the grid reports no obstacle; keep it larger than the arm's reach).
    ``memory_s`` (numpy backend): a surface point that can no longer be checked
    (hidden behind the arm, below the minimum range, out of view) is kept this
    long; one a valid pixel looks at again is re-measured instead.
    """

    def __init__(self, guard, link_camera, bounds_min, bounds_max, voxel=0.01,
                 link='link6', backend='numpy', stride=2, depth_min=0.07, depth_max=1.2,
                 memory_s=10., self_tolerance=0.01, name='depth', clock=time.monotonic):
        self.guard = guard
        self.link_camera = np.asarray(link_camera, float)
        self.link = link
        self.bounds_min = np.asarray(bounds_min, float)
        self.bounds_max = np.asarray(bounds_max, float)
        self.voxel = float(voxel)
        self.shape = tuple(np.ceil((self.bounds_max - self.bounds_min) / self.voxel)
                           .astype(int) + 1)
        self.stride, self.depth_min, self.depth_max = int(stride), depth_min, depth_max
        self.memory_s = float(memory_s)
        self.self_tolerance = float(self_tolerance)
        self.name, self.clock = name, clock
        self.points, self.stamps = np.empty((0, 3)), np.empty(0)
        self.views = []          # (T_base_camera, K, filtered depth) of recent frames
        self.max_views = 12
        self.grid = None
        self.backend = backend
        if backend == 'robokit':
            from .adapters.robokit_sdf import RobokitSdfMap
            self._sdf = RobokitSdfMap(self.bounds_min, self.bounds_max, self.voxel,
                                      depth_max=depth_max)
        elif backend != 'numpy':
            raise ValueError(f'unknown depth map backend {backend!r}')

    def camera_pose(self, q):
        return self.guard.robot.chain.link_poses(q)[self.link] @ self.link_camera

    def filtered_depth(self, depth, K, q):
        """Depth downsampled by ``stride`` (and its intrinsics) with the robot's own
        pixels set to 0 (no observation)."""
        depth = np.asarray(depth, float)[::self.stride, ::self.stride]
        K = np.asarray(K, float).copy()
        K[:2] /= self.stride
        T_base_camera = self.camera_pose(q)
        centers = self.guard.robot.spheres(q)
        camera = (centers - T_base_camera[:3, 3]) @ T_base_camera[:3, :3]
        # The robot's appearance, not its padded safety envelope.
        radii = self.guard.robot.radii - self.guard.robot.padding_m
        mask = robot_pixel_mask(depth, K, camera, radii, self.self_tolerance)
        filtered = np.where(mask, 0., depth).astype(np.float32)
        return filtered, K, T_base_camera, mask

    def update(self, depth, K, q, stamp=None):
        """Fuse one frame taken at joint state ``q``; returns the new ``DistanceGrid``."""
        stamp = self.clock() if stamp is None else stamp
        depth, K, T_base_camera, _ = self.filtered_depth(depth, K, q)
        self.views = (self.views + [(T_base_camera, K, depth)])[-self.max_views:]
        if self.backend == 'robokit':
            values = self._sdf.update(depth, K, T_base_camera)
            self.grid = DistanceGrid(self._sdf.origin, self.voxel, values, name=self.name,
                                     stamp=stamp)
        else:
            self.grid = self._numpy_update(depth, K, T_base_camera, stamp)
        self.guard.set_dynamic([self.grid])
        return self.grid

    def _numpy_update(self, depth, K, T_base_camera, stamp):
        rotation, origin = T_base_camera[:3, :3], T_base_camera[:3, 3]
        # Forget a remembered point only where a valid pixel looks at it again (it is
        # then re-measured or seen through). Points hidden behind the gripper, below
        # the minimum range or out of view are kept until ``memory_s`` passes.
        keep = np.ones(len(self.points), bool)
        if len(self.points):
            camera = (self.points - origin) @ rotation
            z = camera[:, 2]
            front = z > 1e-6
            u = np.full(len(z), -1)
            v = np.full(len(z), -1)
            u[front] = np.rint(K[0, 0] * camera[front, 0] / z[front]
                              + K[0, 1] * camera[front, 1] / z[front] + K[0, 2])
            v[front] = np.rint(K[1, 1] * camera[front, 1] / z[front] + K[1, 2])
            inside = front & (u >= 0) & (u < depth.shape[1]) & (v >= 0) & (v < depth.shape[0])
            measured = np.zeros(len(z))
            measured[inside] = depth[v[inside], u[inside]]
            observed = inside & (measured > self.depth_min) & (measured < self.depth_max)
            keep = ~observed & (stamp - self.stamps <= self.memory_s)
        points = depth_to_points(depth, K, 1, self.depth_min, self.depth_max)
        points = points @ rotation.T + origin
        points = points[np.all((points >= self.bounds_min) & (points <= self.bounds_max), axis=1)]
        points = np.concatenate([points, self.points[keep]])
        stamps = np.concatenate([np.full(len(points) - keep.sum(), stamp), self.stamps[keep]])
        # One point per voxel (the newest); the voxel diagonal bounds the gap to the
        # nearest true surface point that was dropped. Keep voxel >= ~1 cm: surfaces
        # seen at grazing angles are sampled sparsely and a smaller slack let a
        # simulated box side come inside the margin before it was mapped.
        _, index = np.unique(np.floor(points / self.voxel).astype(np.int64), axis=0,
                             return_index=True)
        self.points, self.stamps = points[index], stamps[index]
        return KdPoints(self.points, self.voxel * np.sqrt(3), self.name, stamp)

    def seen_free(self, centers, radii):
        """True for spheres some recent view saw through: the ray to the centre hit
        a surface beyond the far side of the sphere."""
        centers = np.asarray(centers, float)
        free = np.zeros(len(centers), bool)
        for T, K, depth in self.views:
            camera = (centers - T[:3, 3]) @ T[:3, :3]
            z = camera[:, 2]
            ok = z > 1e-3
            u = np.full(len(z), -1)
            v = np.full(len(z), -1)
            u[ok] = np.rint(K[0, 0] * camera[ok, 0] / z[ok] + K[0, 1] * camera[ok, 1] / z[ok]
                            + K[0, 2])
            v[ok] = np.rint(K[1, 1] * camera[ok, 1] / z[ok] + K[1, 2])
            ok &= (u >= 0) & (u < depth.shape[1]) & (v >= 0) & (v < depth.shape[0])
            measured = np.zeros(len(z))
            measured[ok] = depth[v[ok], u[ok]]
            free |= ok & (measured > self.depth_min) & (measured >= z + np.asarray(radii))
        return free

    def column_points(self, floor_z=0.0, min_height=0.02, voxel=None):
        """Mapped points above ``floor_z + min_height`` extruded down to the floor.

        Tabletop prior (2.5-D): a top-down camera sees an object's top but not its
        sides, so the space under every observed surface is taken as solid
        (overhangs become solid too, which is conservative).
        """
        voxel = self.voxel if voxel is None else voxel
        top = self.points[self.points[:, 2] > floor_z + min_height]
        if not len(top):
            return np.empty((0, 3))
        columns = [top]
        for depth in np.arange(voxel, top[:, 2].max() - floor_z, voxel):
            below = top[top[:, 2] - depth > floor_z].copy()
            below[:, 2] -= depth
            columns.append(below)
        return np.concatenate(columns)

    def unknown_points(self, q_start, voxel=0.02, floor_z=0.0):
        """Centres of workspace cells nobody has seen through (and the arm does not
        occupy at ``q_start``): give them to a planner as obstacles so it plans in
        known-free space only."""
        axes = [np.arange(lo + voxel / 2, hi, voxel) for lo, hi in
                zip(self.bounds_min, self.bounds_max)]
        cells = np.stack(np.meshgrid(*axes, indexing='ij'), -1).reshape(-1, 3)
        cells = cells[cells[:, 2] > floor_z + voxel / 2]
        free = self.seen_free(cells, np.full(len(cells), voxel * .87))
        robot = self.guard.robot
        start = robot.spheres(q_start)
        for index in range(0, len(cells), 20000):
            chunk = cells[index:index + 20000]
            free[index:index + 20000] |= (np.linalg.norm(chunk[:, None] - start[None], axis=-1)
                                          <= robot.radii[None] + voxel).any(axis=1)
        return cells[~free]

    def unknown_columns(self, q_start, lower_xy, upper_xy, height=.3, floor_z=0.0, voxel=.02):
        """Points filling every unobserved table cell in ``lower_xy..upper_xy`` up to
        ``height`` (except where the arm stands at ``q_start``): planner obstacles
        that keep proposals over observed table, matching ``heightmap_check``."""
        observed = {tuple(c) for c in np.floor(self.points[:, :2] / self.voxel).astype(int)}
        robot = self.guard.robot
        start = robot.spheres(q_start)
        xs = np.arange(lower_xy[0], upper_xy[0], voxel)
        ys = np.arange(lower_xy[1], upper_xy[1], voxel)
        columns = []
        for x in xs:
            for y in ys:
                cell = (int(np.floor(x / self.voxel)), int(np.floor(y / self.voxel)))
                if cell in observed:
                    continue
                if (np.linalg.norm(start[:, :2] - [x, y], axis=1) <= robot.radii).any():
                    continue
                columns.append([x, y])
        if not columns:
            return np.empty((0, 3))
        columns = np.array(columns)
        zs = np.arange(floor_z + voxel / 2, height, voxel)
        return np.concatenate([np.c_[columns, np.full(len(columns), z)] for z in zs])

    def heightmap_check(self, q_start, links=('link5', 'link6', 'gripper_base'), every=4):
        """2.5-D companion of ``column_points``: the hand may only pass over table
        cells whose height was observed (or that the arm covers at ``q_start``).
        Returns ``check(samples) -> reason or None``."""
        robot = self.guard.robot
        hand = np.isin(robot.links, list(links))
        observed = {tuple(cell) for cell in np.floor(self.points[:, :2] / self.voxel).astype(int)}
        start = robot.spheres(q_start)
        offsets = np.array([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]]) * .7

        def check(samples):
            for index in sorted({*range(0, len(samples), every), len(samples) - 1}):
                centers = robot.spheres(samples[index])[hand]
                radii = robot.radii[hand]
                for center, radius in zip(centers, radii):
                    if (np.linalg.norm(start - center, axis=1) <= robot.radii).any():
                        continue
                    for offset in offsets:
                        cell = tuple(np.floor((center[:2] + offset * radius) / self.voxel)
                                     .astype(int))
                        if cell not in observed:
                            check.spot = center.copy()
                            return (f'path moves the hand over an unobserved spot '
                                    f'{np.round(center[:2], 3).tolist()} at sample {index}; '
                                    'look there first')
            return None
        check.spot = None
        return check

    def unobserved_check(self, q_start, links=('link5', 'link6', 'gripper_base'), every=4):
        """A path check for planners: the hand may only move through space that was
        seen to be free, or that the arm itself occupies at ``q_start``. Unseen space
        is not assumed empty. Returns ``check(samples) -> reason or None``."""
        robot = self.guard.robot
        hand = np.isin(robot.links, list(links))
        radii = robot.radii[hand]
        start = robot.spheres(q_start)
        start_radii = robot.radii

        def check(samples):
            for index in sorted({*range(0, len(samples), every), len(samples) - 1}):
                centers = robot.spheres(samples[index])[hand]
                inside_start = (np.linalg.norm(centers[:, None] - start[None], axis=-1)
                                <= start_radii[None]).any(axis=1)
                unknown = ~(inside_start | self.seen_free(centers, radii))
                if unknown.any():
                    return (f'path moves the hand through unobserved space at sample {index} '
                            f'({int(unknown.sum())} spheres); look there first')
            return None
        return check

    def age(self, now=None):
        if self.grid is None:
            return np.inf
        return (self.clock() if now is None else now) - self.grid.stamp
