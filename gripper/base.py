"""Uniform gripper interface: width in metres, 0 = closed, max_width = open."""

from __future__ import annotations

import abc
import time
from dataclasses import dataclass


@dataclass
class GripperState:
    width: float
    target: float
    moving: bool = False
    connected: bool = True
    t: float = 0.0


class GripperBackend(abc.ABC):
    """Every gripper is addressed by physical opening width in metres.

    Normalizing to metres is what lets the same dataset action vector, the same
    teleop mapping and the same replay schedule work with a Robotiq 2F-85, a
    Piper gripper or a simulated parallel jaw.
    """

    def __init__(self, name: str = 'gripper', max_width: float = 0.085,
                 min_width: float = 0.0):
        self.name = name
        self.max_width = float(max_width)
        self.min_width = float(min_width)
        self._connected = False

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def range(self) -> tuple[float, float]:
        return self.min_width, self.max_width

    def clamp(self, width: float) -> float:
        return float(min(max(float(width), self.min_width), self.max_width))

    def connect(self) -> 'GripperBackend':
        self._connected = True
        return self

    def disconnect(self) -> None:
        self._connected = False

    def __enter__(self):
        return self.connect()

    def __exit__(self, exc_type, exc, traceback):
        self.disconnect()
        return False

    @abc.abstractmethod
    def move(self, width: float, force=None, speed=None, wait: bool = False,
             timeout: float | None = None) -> bool:
        """Command an opening width. ``force``/``speed`` are device-specific 0-255."""

    def open(self, wait: bool = False, timeout: float | None = None) -> bool:
        return self.move(self.max_width, wait=wait, timeout=timeout)

    def close(self, wait: bool = False, timeout: float | None = None) -> bool:
        return self.move(self.min_width, wait=wait, timeout=timeout)

    def stop(self) -> bool:
        return True

    @property
    @abc.abstractmethod
    def width(self) -> float:
        """Latest measured (or best known) opening width."""

    @property
    def commanded_width(self) -> float:
        return self.width

    def state(self) -> GripperState:
        return GripperState(width=self.width, target=self.commanded_width,
                            moving=False, connected=self.connected, t=time.monotonic())

    def wait(self, timeout: float | None = None, tolerance: float = 0.002,
             rate_hz: float = 20.0) -> bool:
        deadline = time.monotonic() + float(timeout if timeout is not None else 5.0)
        while time.monotonic() < deadline:
            if abs(self.width - self.commanded_width) <= tolerance:
                return True
            time.sleep(1.0 / rate_hz)
        return False

    @classmethod
    def from_spec(cls, gripper_spec, **kwargs):
        return cls(name=gripper_spec.name, max_width=gripper_spec.max_width,
                   min_width=gripper_spec.min_width, **kwargs)

    def __repr__(self) -> str:
        return f'{type(self).__name__}(name={self.name!r}, width={self.width:.4f})'
