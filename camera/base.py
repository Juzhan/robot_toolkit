"""Camera interface and frame container.

Frame conventions (shared with the existing projects so datasets stay
compatible):

* RGB is ``HxWx3`` ``uint8``, RGB order (not BGR).
* Depth is ``HxW`` ``uint16`` millimetres, ``0`` meaning invalid.
* ``T_world_camera`` maps camera-optical coordinates (x right, y down, z
  forward) into the robot base/world frame.
* Timestamps are host ``perf_counter`` seconds, taken when the frame was
  received, so multiple cameras can be compared even with different drivers.
"""

from __future__ import annotations

import abc
import threading
import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class CameraFrame:
    name: str = ''
    t: float = 0.0
    rgb: np.ndarray | None = None
    depth: np.ndarray | None = None
    intrinsics: np.ndarray | None = None
    distortion: np.ndarray | None = None
    T_world_camera: np.ndarray | None = None
    meta: dict = field(default_factory=dict)

    @property
    def size(self) -> tuple[int, int]:
        if self.rgb is None:
            return (0, 0)
        return int(self.rgb.shape[1]), int(self.rgb.shape[0])

    def age(self, now: float | None = None) -> float:
        return float(now if now is not None else time.perf_counter()) - float(self.t)

    def copy(self) -> 'CameraFrame':
        return CameraFrame(
            name=self.name, t=self.t,
            rgb=None if self.rgb is None else self.rgb.copy(),
            depth=None if self.depth is None else self.depth.copy(),
            intrinsics=None if self.intrinsics is None else self.intrinsics.copy(),
            distortion=None if self.distortion is None else self.distortion.copy(),
            T_world_camera=None if self.T_world_camera is None else self.T_world_camera.copy(),
            meta=dict(self.meta),
        )


class Camera(abc.ABC):
    def __init__(self, name: str = 'camera', width: int = 640, height: int = 480,
                 fps: int = 30, depth: bool = False):
        self.name = name
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.depth_enabled = bool(depth)
        self.intrinsics: np.ndarray | None = None
        self.distortion: np.ndarray | None = None
        self.T_world_camera: np.ndarray | None = None
        self.error = ''
        self.frames_seen = 0
        self._opened = False

    # ------------------------------------------------------------------ link
    @property
    def opened(self) -> bool:
        return self._opened

    def open(self) -> 'Camera':
        self._opened = True
        return self

    def close(self) -> None:
        self._opened = False

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc, traceback):
        self.close()
        return False

    # ------------------------------------------------------------------ data
    @abc.abstractmethod
    def read(self) -> CameraFrame:
        """Return the newest frame (never blocks on the camera driver)."""

    def snapshot(self) -> CameraFrame:
        frame = self.read()
        frame.intrinsics = None if self.intrinsics is None else self.intrinsics.copy()
        frame.distortion = None if self.distortion is None else self.distortion.copy()
        frame.T_world_camera = (None if self.T_world_camera is None
                                else self.T_world_camera.copy())
        return frame

    def set_extrinsics(self, transform) -> None:
        self.T_world_camera = np.asarray(transform, dtype=float).reshape(4, 4)

    def set_intrinsics(self, intrinsics, distortion=None) -> None:
        self.intrinsics = np.asarray(intrinsics, dtype=float).reshape(3, 3)
        if distortion is not None:
            self.distortion = np.asarray(distortion, dtype=float).reshape(-1)

    def __repr__(self) -> str:
        return (f'{type(self).__name__}(name={self.name!r}, '
                f'{self.width}x{self.height}@{self.fps}, depth={self.depth_enabled})')


class ThreadedCamera(Camera):
    """Base class for drivers that deliver frames on their own thread."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._rgb: np.ndarray | None = None
        self._depth: np.ndarray | None = None
        self._stamp = 0.0
        self._color_meta: dict = {}
        self._depth_meta: dict = {}

    def _start_worker(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True, name=f'cam-{self.name}')
        self._thread.start()

    def _loop(self) -> None:  # pragma: no cover - driver specific
        raise NotImplementedError

    def _publish(self, rgb, depth=None, color_meta=None, depth_meta=None) -> None:
        stamp = time.perf_counter()
        color_meta = dict(color_meta or {})
        color_meta['host_time'] = stamp
        depth_meta = dict(depth_meta or {})
        depth_meta['host_time'] = stamp
        with self._lock:
            self._rgb = rgb
            self._depth = depth
            self._stamp = stamp
            self._color_meta = color_meta
            self._depth_meta = depth_meta
        self.frames_seen += 1

    def read(self) -> CameraFrame:
        with self._lock:
            rgb = None if self._rgb is None else self._rgb.copy()
            depth = None if self._depth is None else self._depth.copy()
            stamp = self._stamp
            color_meta = dict(self._color_meta)
            depth_meta = dict(self._depth_meta)
        return CameraFrame(name=self.name, t=stamp, rgb=rgb, depth=depth,
                           intrinsics=self.intrinsics, distortion=self.distortion,
                           T_world_camera=self.T_world_camera,
                           meta={'color': color_meta, 'depth': depth_meta})

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._opened = False
