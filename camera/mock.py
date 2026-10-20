"""Synthetic camera with a moving target, useful for offline pipeline tests."""

from __future__ import annotations

import time

import numpy as np

from .base import Camera, CameraFrame


class MockCamera(Camera):
    def __init__(self, name: str = 'camera', width: int = 320, height: int = 240,
                 fps: int = 30, depth: bool = False, seed: int | None = None):
        super().__init__(name, width, height, fps, depth)
        self._t0 = time.perf_counter()
        self._seed = abs(hash(name)) % (2 ** 16) if seed is None else int(seed)
        grid_y, grid_x = np.mgrid[0:self.height, 0:self.width]
        self._background = np.stack([
            (grid_x * 255 // max(1, self.width - 1)),
            (grid_y * 255 // max(1, self.height - 1)),
            np.full_like(grid_x, (self._seed * 7) % 255),
        ], axis=-1).astype(np.uint8)
        self.intrinsics = np.array([[0.9 * self.width, 0.0, self.width / 2],
                                    [0.0, 0.9 * self.width, self.height / 2],
                                    [0.0, 0.0, 1.0]])
        self.distortion = np.zeros(5)

    def open(self):
        self._opened = True
        self._t0 = time.perf_counter()
        return self

    def read(self) -> CameraFrame:
        if not self._opened:
            self.open()
        now = time.perf_counter()
        phase = now - self._t0
        frame = self._background.copy()
        half = 24
        center_x = int((0.5 + 0.35 * np.sin(phase * 1.3 + self._seed)) * self.width)
        center_y = int((0.5 + 0.30 * np.cos(phase * 0.9 + self._seed)) * self.height)
        x0, x1 = max(0, center_x - half), min(self.width, center_x + half)
        y0, y1 = max(0, center_y - half), min(self.height, center_y + half)
        frame[y0:y1, x0:x1] = (255, 64, 32) if 'wrist' in self.name else (32, 255, 64)
        frame[8:20, 8:20] = (255, 255, 255)
        depth = None
        if self.depth_enabled:
            depth = np.full((self.height, self.width), 1000, dtype=np.uint16)
            depth[center_y - 4:center_y + 4, center_x - 4:center_x + 4] = 500
        self.frames_seen += 1
        return CameraFrame(name=self.name, t=now, rgb=frame, depth=depth,
                           intrinsics=self.intrinsics, distortion=self.distortion,
                           T_world_camera=self.T_world_camera,
                           meta={'frame_number': self.frames_seen})
