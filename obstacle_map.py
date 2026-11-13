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


def robot_pixel_mask(depth, K, centers_camera, radii, tolerance=0.01, grow_rad=0.01):
    """Pixels showing the robot's own spheres, or anything hidden behind them.

    Exact cone test: a pixel ray is inside a sphere's silhouette if its angle to the
    centre is at most ``asin(r / |c|)`` (+ ``grow_rad``). It is masked unless its
    depth is clearly in front of the sphere (``< z - r - tolerance``): that is a
    real object between the camera and the arm and must stay an obstacle. Spheres
    around the lens (the camera's own body) are skipped.
    """
    depth = np.asarray(depth, float)
    K = np.asarray(K, float)
    h, w = depth.shape
    v, u = np.mgrid[0:h, 0:w]
    y = (v - K[1, 2]) / K[1, 1]
    x = (u - K[0, 2] - K[0, 1] * y) / K[0, 0]
    rays = np.stack([x, y, np.ones_like(x)], -1)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    unseen = ~np.isfinite(depth) | (depth <= 0.)
    mask = np.zeros(depth.shape, bool)
    for center, r in zip(np.asarray(centers_camera, float), np.asarray(radii, float)):
        distance = float(np.linalg.norm(center))
        if distance <= r or center[2] + r <= 0.:
            continue   # the camera's own body, or entirely behind the image plane
        inside = rays @ (center / distance) >= np.cos(np.arcsin(r / distance) + grow_rad)
        mask |= inside & (unseen | (depth >= center[2] - r - tolerance))
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

    def age(self, now=None):
        if self.grid is None:
            return np.inf
        return (self.clock() if now is None else now) - self.grid.stamp
