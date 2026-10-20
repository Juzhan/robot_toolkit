"""Synchronized multi-camera snapshot with staleness detection."""

from __future__ import annotations

import time

import numpy as np

from ..errors import RobotUnavailable
from .base import Camera, CameraFrame


class CameraHub:
    """Own a fixed set of cameras and read them as one observation.

    Cameras run their own threads; a snapshot is cheap copies of the latest
    frame.  ``stale_s`` rejects observations whose images are too old, which is
    the failure mode that silently poisons a demonstration dataset.
    """

    def __init__(self, cameras=(), stale_s: float = 0.3):
        self.cameras: dict[str, Camera] = {}
        for camera in cameras:
            self.add(camera)
        self.stale_s = float(stale_s)

    def add(self, camera: Camera) -> Camera:
        if camera.name in self.cameras:
            raise ValueError(f'duplicate camera name {camera.name!r}')
        self.cameras[camera.name] = camera
        return camera

    def __getitem__(self, name: str) -> Camera:
        return self.cameras[name]

    def __iter__(self):
        return iter(self.cameras.values())

    def __len__(self) -> int:
        return len(self.cameras)

    @property
    def names(self) -> list[str]:
        return list(self.cameras)

    def open(self) -> 'CameraHub':
        for camera in self.cameras.values():
            camera.open()
        return self

    def close(self) -> None:
        for camera in self.cameras.values():
            camera.close()

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc, traceback):
        self.close()
        return False

    def snapshot(self, strict: bool = True, stale_s: float | None = None) -> dict[str, CameraFrame]:
        limit = self.stale_s if stale_s is None else float(stale_s)
        now = time.perf_counter()
        frames = {}
        stale = []
        for name, camera in self.cameras.items():
            frame = camera.snapshot()
            if frame.rgb is None:
                stale.append(f'{name}: no frame')
            elif limit > 0 and frame.age(now) > limit:
                stale.append(f'{name}: {frame.age(now) * 1000:.0f} ms old')
            frames[name] = frame
        if strict and stale:
            raise RobotUnavailable('camera frames unavailable: ' + '; '.join(stale))
        return frames

    def warmup(self, min_frames: int = 2, timeout: float = 8.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if all(camera.frames_seen >= min_frames for camera in self.cameras.values()):
                return True
            # Threaded drivers advance frames_seen themselves.  Pull-based
            # drivers (MockCamera, some OpenCV implementations) only advance on
            # read(), so trigger one snapshot rather than waiting forever.
            for camera in self.cameras.values():
                if camera.frames_seen < min_frames:
                    try:
                        camera.snapshot()
                    except Exception:
                        pass
            time.sleep(0.01)
        return False

    def set_extrinsics(self, name: str, transform) -> None:
        self.cameras[name].set_extrinsics(transform)

    def extrinsics(self, name: str) -> np.ndarray | None:
        return self.cameras[name].T_world_camera

    def intrinsics(self, name: str) -> np.ndarray | None:
        return self.cameras[name].intrinsics

    def depth_enabled(self, name: str) -> bool:
        return self.cameras[name].depth_enabled
