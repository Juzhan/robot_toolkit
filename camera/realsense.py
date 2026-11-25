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
                 height: int = 480, fps: int = 30, depth: bool = False,
                 exposure_ms: float | None = None, gain: float | None = None,
                 infrared: bool = False):
        super().__init__(name, width, height, fps, depth)
        # Left/right IR (Y8) pair for on-demand stereo depth (perception.StereoDepthClient).
        self.infrared = bool(infrared)
        self._ir_pair = None
        self.ir_calibration = None
        self.serial = str(serial)
        # 手动曝光（毫秒）：短曝光可减少运动拖影；None = 自动曝光
        self.exposure_ms = exposure_ms
        self.gain = gain
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
        if self.infrared:
            for index in (1, 2):
                config.enable_stream(rs.stream.infrared, index, self.width, self.height,
                                     rs.format.y8, int(self.fps))
        self._pipeline = rs.pipeline()
        profile = self._pipeline.start(config)
        self._apply_color_options(rs, profile.get_device().first_color_sensor())
        if self.depth_enabled:
            self._align = rs.align(rs.stream.color)
            self.depth_scale = float(profile.get_device().first_depth_sensor().get_depth_scale())
        color = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.intrinsics = np.array([[color.fx, 0.0, color.ppx],
                                    [0.0, color.fy, color.ppy],
                                    [0.0, 0.0, 1.0]])
        self.distortion = np.asarray(color.coeffs, dtype=float)[:5]
        if self.infrared:
            self.ir_calibration = self._ir_calibration(rs, profile)
        self._stop.clear()
        self._start_worker()
        self._opened = True
        return self

    @staticmethod
    def _ir_calibration(rs, profile):
        left = profile.get_stream(rs.stream.infrared, 1).as_video_stream_profile()
        right = profile.get_stream(rs.stream.infrared, 2).as_video_stream_profile()
        color = profile.get_stream(rs.stream.color)

        def matrix(intr):
            return np.array([[intr.fx, 0., intr.ppx], [0., intr.fy, intr.ppy], [0., 0., 1.]])

        def transform(extrinsics):
            T = np.eye(4)
            T[:3, :3] = np.asarray(extrinsics.rotation, float).reshape(3, 3).T  # column-major
            T[:3, 3] = extrinsics.translation
            return T

        T_right_left = transform(left.get_extrinsics_to(right))
        return {'K_left': matrix(left.get_intrinsics()), 'K_right': matrix(right.get_intrinsics()),
                'baseline_m': float(abs(T_right_left[0, 3])),
                'T_color_left': transform(left.get_extrinsics_to(color))}

    def stereo_pair(self):
        """Latest ``(left, right, ir_calibration)`` uint8 IR images, or ``None``."""
        with self._lock:
            pair = self._ir_pair
        if pair is None:
            return None
        return pair[0].copy(), pair[1].copy(), dict(self.ir_calibration)

    def _apply_color_options(self, rs, sensor):
        if self.exposure_ms is not None:
            sensor.set_option(rs.option.enable_auto_exposure, 0)
            # UVC 彩色曝光单位为 100 µs
            sensor.set_option(rs.option.exposure, float(np.clip(round(self.exposure_ms * 10), 1, 10000)))
        if self.gain is not None:
            sensor.set_option(rs.option.gain, float(self.gain))

    def _loop(self):  # pragma: no cover - requires hardware
        while not self._stop.is_set():
            try:
                frames = self._pipeline.wait_for_frames(2000)
            except Exception as exc:
                self.error = str(exc)
                time.sleep(0.05)
                continue
            if self.infrared:
                left, right = frames.get_infrared_frame(1), frames.get_infrared_frame(2)
                if left and right:
                    pair = (np.asanyarray(left.get_data()).copy(),
                            np.asanyarray(right.get_data()).copy())
                    with self._lock:
                        self._ir_pair = pair
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
            capture_t = capture_time(color_meta['hw_time_ms'], color_meta['time_domain'])
            if capture_t is not None:
                color_meta['capture_t'] = capture_t   # perf_counter clock, like frame.t
            if self.depth_enabled:
                depth_frame = frames.get_depth_frame()
                if depth_frame:
                    depth = np.asanyarray(depth_frame.get_data()).copy()
                    depth_meta = {'depth_scale': self.depth_scale,
                                  'hw_time_ms': float(depth_frame.get_timestamp()),
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

    def hardware_reset(self, wait_s: float = 15.0) -> bool:
        """Power-cycle this device over USB and wait until it enumerates again."""
        rs = _import_realsense()
        if not self.serial:
            return False
        devices = [d for d in rs.context().query_devices()
                   if d.get_info(rs.camera_info.serial_number) == self.serial]
        if not devices:
            return False
        devices[0].hardware_reset()
        time.sleep(3.0)
        deadline = time.monotonic() + wait_s
        while time.monotonic() < deadline:
            if any(d.get_info(rs.camera_info.serial_number) == self.serial
                   for d in rs.context().query_devices()):
                time.sleep(1.0)
                return True
            time.sleep(0.5)
        return False


def capture_time(hw_time_ms, time_domain, now_perf=None, now_wall=None):
    """Frame capture time on the host ``perf_counter`` clock, or ``None``.

    RealSense ``global_time`` (2) and ``system_time`` (1) timestamps are host wall
    clock milliseconds of the frame (the device clock mapped to the host for
    global time). The frame's age is measured on the wall clock and subtracted
    from ``perf_counter``. A ``hardware_clock`` (0) stamp has no host meaning.
    """
    if int(time_domain) not in (1, 2):
        return None
    now_perf = time.perf_counter() if now_perf is None else now_perf
    now_wall = time.time() if now_wall is None else now_wall
    age = now_wall - float(hw_time_ms) / 1000.
    if not 0. <= age < 1.:        # clock jump or bogus metadata: do not trust it
        return None
    return now_perf - age


def list_devices():
    """Return ``[(name, serial), ...]`` for connected RealSense cameras."""
    rs = _import_realsense()
    devices = []
    for device in rs.context().query_devices():
        devices.append((device.get_info(rs.camera_info.name),
                        device.get_info(rs.camera_info.serial_number)))
    return devices
