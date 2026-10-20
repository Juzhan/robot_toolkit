"""Intel RealSense colour + aligned depth capture via ``pyrealsense2``."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

from .base import ThreadedCamera


def _import_realsense():
    try:
        import pyrealsense2 as rs
        return rs
    except ImportError:
        fallback = Path(os.path.expandvars(r'%LOCALAPPDATA%/RealSense-Diagnostics/python-sdk'))
        if fallback.exists():
            sys.path.insert(0, str(fallback))
            import pyrealsense2 as rs
            return rs
        raise ImportError('RealSense 相机需要 pyrealsense2：pip install pyrealsense2') from None


class RealSenseCamera(ThreadedCamera):
    def __init__(self, name: str = 'camera', serial: str = '', width: int = 640,
                 height: int = 480, fps: int = 30, depth: bool = False):
        super().__init__(name, width, height, fps, depth)
        self.serial = str(serial)
        self._pipeline = None
        self._align = None
        self.depth_scale = 0.001
        self.distortion = np.zeros(5, dtype=float)

    def open(self):
        rs = _import_realsense()
        config = rs.config()
        if self.serial:
            config.enable_device(self.serial)
        config.enable_stream(rs.stream.color, self.width, self.height,
                             rs.format.rgb8, int(self.fps))
        if self.depth_enabled:
            config.enable_stream(rs.stream.depth, self.width, self.height,
                                 rs.format.z16, int(self.fps))
        self._pipeline = rs.pipeline()
        profile = self._pipeline.start(config)
        if self.depth_enabled:
            self._align = rs.align(rs.stream.color)
            self.depth_scale = float(profile.get_device().first_depth_sensor().get_depth_scale())
        color = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.intrinsics = np.array([[color.fx, 0.0, color.ppx],
                                    [0.0, color.fy, color.ppy],
                                    [0.0, 0.0, 1.0]])
        self.distortion = np.asarray(color.coeffs, dtype=float)[:5]
        self._stop.clear()
        self._start_worker()
        self._opened = True
        return self

    def _loop(self):  # pragma: no cover - requires hardware
        while not self._stop.is_set():
            try:
                frames = self._pipeline.wait_for_frames(2000)
            except Exception as exc:
                self.error = str(exc)
                time.sleep(0.05)
                continue
            if self._align is not None:
                frames = self._align.process(frames)
            color = frames.get_color_frame()
            if not color:
                continue
            rgb = np.asanyarray(color.get_data()).copy()
            depth = None
            depth_meta = {}
            color_meta = {'hw_time_ms': float(color.get_timestamp()),
                          'time_domain': int(color.get_frame_timestamp_domain()),
                          'frame_number': int(color.get_frame_number())}
            if self.depth_enabled:
                depth_frame = frames.get_depth_frame()
                if depth_frame:
                    depth = np.asanyarray(depth_frame.get_data()).copy()
                    depth_meta = {'hw_time_ms': float(depth_frame.get_timestamp()),
                                  'time_domain': int(depth_frame.get_frame_timestamp_domain()),
                                  'frame_number': int(depth_frame.get_frame_number())}
            self.error = ''
            self._publish(rgb, depth, color_meta, depth_meta)

    def close(self):
        super().close()
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception:
                pass
            self._pipeline = None


def list_devices():
    """Return ``[(name, serial), ...]`` for connected RealSense cameras."""
    rs = _import_realsense()
    devices = []
    for device in rs.context().query_devices():
        devices.append((device.get_info(rs.camera_info.name),
                        device.get_info(rs.camera_info.serial_number)))
    return devices
