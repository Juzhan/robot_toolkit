"""robokit ``SdfVolume`` (GPU TSDF fusion + ESDF) as a dense NumPy distance grid.

The algorithm (fusion, free-space clearing, jump-flooding ESDF) is robokit's;
this adapter only feeds it depth frames and copies the distance grid back to the
host so the toolkit's NumPy ``CollisionGuard`` can query it without a GPU.
Needs ``robokit`` and ``warp`` (lazy: imported only when this backend is chosen).
"""

from __future__ import annotations

import numpy as np
import warp as wp
from robokit.geom import SdfVolume


@wp.kernel
def _read_grid(volume: wp.uint64, out: wp.array3d(dtype=float)):
    i, j, k = wp.tid()
    out[i, j, k] = wp.volume_lookup_f(volume, i, j, k)


class RobokitSdfMap:
    def __init__(self, bounds_min, bounds_max, voxel, depth_max=1.2, device='cuda:0',
                 max_weight=30.0):
        self.sdf = SdfVolume.from_points(np.empty((0, 3)), np.asarray(bounds_min, float),
                                         np.asarray(bounds_max, float), voxel, device)
        self.voxel = float(voxel)
        self.origin = np.asarray(self.sdf.aabb_min, float)
        self.shape = tuple(int(n) for n in self.sdf._fusion.shape)
        self.depth_max, self.max_weight, self.device = depth_max, max_weight, device
        self._out = wp.zeros(self.shape, dtype=float, device=device)

    def update(self, depth, K, T_base_camera):
        self.sdf.update(depth=np.asarray(depth, np.float32), intrinsics=np.asarray(K, np.float32),
                        T_world_camera=np.asarray(T_base_camera, np.float32),
                        depth_max=self.depth_max, max_weight=self.max_weight)
        wp.launch(_read_grid, dim=self.shape, inputs=[self.sdf.volume.id, self._out],
                  device=self.device)
        values = self._out.numpy().astype(float)
        values[values >= 1e5] = np.inf            # background: nothing observed nearby
        # The jump-flooding ESDF is approximate; one voxel of slack keeps it a lower bound.
        return values - self.voxel
