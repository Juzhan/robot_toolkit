"""USB / built-in camera via OpenCV, with a background capture thread."""

from __future__ import annotations

import sys
import time

import numpy as np

from .base import ThreadedCamera


class OpenCVCamera(ThreadedCamera):
    def __init__(self, name: str = 'camera', index: int = 0, width: int = 640,
                 height: int = 480, fps: int = 30, depth: bool = False):
        super().__init__(name, width, height, fps, depth)
        self.index = int(index)
        self._capture = None

    def open(self):
        import cv2
        # OpenCV 5.x ships builds where DSHOW is advertised but cannot open a
        # device by index, while MSMF works.  Try the platform-preferred
        # backend first and fall back instead of failing the whole capture.
        backends = ([cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
                    if sys.platform == 'win32' else [cv2.CAP_ANY])
        capture = None
        for backend in backends:
            candidate = cv2.VideoCapture(self.index, backend)
            if candidate.isOpened():
                capture = candidate
                break
            candidate.release()
        if capture is None:
            raise RuntimeError(f'无法打开 OpenCV 相机 {self.index}（{self.name}）')
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        capture.set(cv2.CAP_PROP_FPS, self.fps)
        self._capture = capture
        self._stop.clear()
        self._start_worker()
        self._opened = True
        return self

    def _loop(self):  # pragma: no cover - requires a camera
        while not self._stop.is_set():
            ok, frame = self._capture.read()
            if not ok:
                self.error = 'read failed'
                time.sleep(0.02)
                continue
            self.error = ''
            self._publish(np.ascontiguousarray(frame[:, :, ::-1]))

    def close(self):
        super().close()
        if self._capture is not None:
            self._capture.release()
            self._capture = None
