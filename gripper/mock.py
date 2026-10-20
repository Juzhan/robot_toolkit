"""In-memory gripper that moves at a finite speed (never teleports)."""

from __future__ import annotations

import time

from .base import GripperBackend, GripperState


class MockGripper(GripperBackend):
    def __init__(self, name: str = 'gripper', max_width: float = 0.085,
                 min_width: float = 0.0, speed: float = 0.08, clock=time.monotonic):
        super().__init__(name, max_width, min_width)
        self.motion_speed = float(speed)
        self.clock = clock
        self._width = self.max_width
        self._target = self.max_width
        self._last = clock()
        self._moving = False
        self.history: list[float] = []

    def _advance(self) -> None:
        now = self.clock()
        dt = max(0.0, now - self._last)
        self._last = now
        delta = self._target - self._width
        step = self.motion_speed * dt
        if abs(delta) <= step:
            self._width = self._target
            self._moving = False
        else:
            self._width += step if delta > 0 else -step
            self._moving = True

    @property
    def width(self) -> float:
        self._advance()
        return self._width

    @property
    def commanded_width(self) -> float:
        return self._target

    def connect(self):
        self._connected = True
        self._last = self.clock()
        return self

    def move(self, width, force=None, speed=None, wait=False, timeout=None) -> bool:
        self._advance()
        self._target = self.clamp(width)
        self._moving = self._target != self._width
        self.history.append(self._target)
        if wait:
            return self.wait(timeout)
        return True

    def stop(self) -> bool:
        self._advance()
        self._target = self._width
        self._moving = False
        return True

    def state(self) -> GripperState:
        self._advance()
        return GripperState(width=self._width, target=self._target,
                            moving=self._moving, connected=self.connected,
                            t=self.clock())

    def wait(self, timeout=None, tolerance=0.002, rate_hz=20.0) -> bool:
        deadline = self.clock() + float(timeout if timeout is not None else 5.0)
        while self.clock() < deadline:
            self._advance()
            if abs(self._target - self._width) <= tolerance:
                return True
            time.sleep(1.0 / rate_hz)
        return False
