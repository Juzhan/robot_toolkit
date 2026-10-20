"""Adapt arbitrary read/write callables to the gripper interface.

Useful to attach a toolkit pipeline to an existing driver (or a MuJoCo
actuator) without writing a new class::

    gripper = CallbackGripper(read=lambda: sim.width(),
                              write=lambda m: sim.set_width(m),
                              max_width=0.085)
"""

from __future__ import annotations

from .base import GripperBackend, GripperState


class CallbackGripper(GripperBackend):
    def __init__(self, read, write, name: str = 'gripper', max_width: float = 0.085,
                 min_width: float = 0.0, stop=None, connect=None, disconnect=None,
                 target=None):
        super().__init__(name, max_width, min_width)
        self._read = read
        self._write = write
        self._stop = stop
        self._connect = connect
        self._disconnect = disconnect
        self._target = target or read
        self._connected = True

    def connect(self):
        if self._connect is not None:
            self._connect()
        self._connected = True
        return self

    def disconnect(self):
        if self._disconnect is not None:
            self._disconnect()
        self._connected = False

    def move(self, width, force=None, speed=None, wait=False, timeout=None) -> bool:
        self._target = self.clamp(width)
        self._write(self._target)
        return self.wait(timeout) if wait else True

    def stop(self) -> bool:
        return True if self._stop is None else bool(self._stop())

    @property
    def width(self) -> float:
        return self.clamp(self._read())

    @property
    def commanded_width(self) -> float:
        return self._target

    def state(self) -> GripperState:
        returned = self.width
        return GripperState(width=returned, target=self._target,
                            moving=abs(returned - self._target) > 0.002,
                            connected=self.connected)
