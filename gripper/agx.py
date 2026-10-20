"""AgxGripper adapter for the AgileX Piper / Piper X arm.

The gripper is created through ``robot.init_effector(...)`` by the arm backend;
this class only exposes the toolkit's normalized metre interface around it.
"""
from __future__ import annotations

import threading
import time

from .base import GripperBackend, GripperState


class AgxGripper(GripperBackend):
    def __init__(self, robot, name: str = 'piper_gripper', max_width: float = 0.08,
                 min_width: float = 0.0, force: float = 1.0,
                 close_time_s: float = 0.0, open_time_s: float = 0.0,
                 ramp_hz: float = 50.0):
        super().__init__(name=name, max_width=max_width, min_width=min_width)
        self.robot = robot
        self.force = float(force)
        self.close_time_s = max(0.0, float(close_time_s))
        self.open_time_s = max(0.0, float(open_time_s))
        self.ramp_hz = max(1.0, float(ramp_hz))
        self._target = float(max_width)
        self._connected = bool(getattr(robot, 'connected', False))
        self._lock = threading.RLock()
        self._ramp_thread = None
        self._ramp_stop = threading.Event()
        self._ramp_errors: list[str] = []

    def connect(self):
        if not getattr(self.robot, 'connected', False):
            self.robot.connect()
        self._connected = True
        return self

    def disconnect(self):
        # The arm owns the CAN link; do not close it here.
        self._stop_ramp()
        self._connected = False

    def _stop_ramp(self):
        with self._lock:
            self._ramp_stop.set()
            thread, self._ramp_thread = self._ramp_thread, None
        if (thread is not None and thread.is_alive()
                and thread is not threading.current_thread()):
            thread.join(timeout=0.5)

    def _ramp(self, start: float, target: float, duration: float, force: float):
        steps = max(1, int(round(duration * self.ramp_hz)))
        period = 1.0 / self.ramp_hz
        try:
            for index in range(1, steps + 1):
                if self._ramp_stop.is_set():
                    return
                # Smoothstep avoids an abrupt start/stop of the target ramp.
                alpha = index / steps
                alpha = alpha * alpha * (3.0 - 2.0 * alpha)
                width = start + (target - start) * alpha
                self.robot.move_gripper(width, force=force, wait=False)
                if index < steps:
                    time.sleep(period)
        except Exception as exc:
            self._ramp_errors.append(f'{type(exc).__name__}: {exc}')
            self._ramp_stop.set()

    def move(self, width, force=None, speed=None, wait: bool = False,
             timeout: float | None = None) -> bool:
        target = self.clamp(width)
        force_value = self.force if force is None else float(force)
        with self._lock:
            self._stop_ramp()
            try:
                start = float(self.width)
            except Exception:
                start = float(self._target)
            self._target = target
            self._ramp_errors.clear()
            duration = self.close_time_s if target < start else self.open_time_s
            if duration <= 0.0 or abs(target - start) < 1e-4:
                return bool(self.robot.move_gripper(
                    target, force=force_value, wait=wait, timeout=timeout))
            self._ramp_stop.clear()
            thread = threading.Thread(
                target=self._ramp,
                args=(start, target, duration, force_value),
                name='agx-gripper-ramp',
                daemon=True,
            )
            self._ramp_thread = thread
            thread.start()
        if not wait:
            return True
        thread.join(timeout=float(timeout if timeout is not None else duration + 2.0))
        if self._ramp_errors:
            return False
        try:
            return abs(float(self.width) - target) <= 0.005
        except Exception:
            return False

    def stop(self) -> bool:
        self._stop_ramp()
        return True

    @property
    def width(self) -> float:
        return self.clamp(self.robot.gripper_width)

    @property
    def commanded_width(self) -> float:
        return self._target

    def state(self) -> GripperState:
        width = self.width
        return GripperState(width=width, target=self._target,
                            moving=abs(width - self._target) > 0.002,
                            connected=self.connected)
