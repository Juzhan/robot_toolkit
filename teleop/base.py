"""Input-device abstraction and thread-safe command bridge.

A teleoperation device produces ``TeleopCommand`` objects.  Everything after
the device (mapping, safety, robot, recording) is identical for a Joy-Con, a
phone, a keyboard, a VR controller or a scripted test.
"""

from __future__ import annotations

import abc
import threading
import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class TeleopCommand:
    """Normalized operator input.

    ``xyz`` and ``rpy`` are in ``[-1, 1]``; the mapping layer decides what one
    unit means.  ``grip`` is a normalized opening/closing rate.  ``quaternion``
    is an optional absolute orientation (xyzw) when the device tracks attitude
    instead of reporting angular velocity.
    """

    t: float = 0.0
    xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)
    quaternion: tuple[float, float, float, float] | None = None
    grip: float = 0.0
    grip_target: float | None = None
    enable: bool | None = None
    buttons: dict = field(default_factory=dict)

    def __post_init__(self):
        self.xyz = tuple(float(value) for value in self.xyz)
        self.rpy = tuple(float(value) for value in self.rpy)
        self.grip = float(self.grip)

    @property
    def is_idle(self) -> bool:
        return (max(abs(value) for value in self.xyz) < 1e-9 and
                max(abs(value) for value in self.rpy) < 1e-9 and
                abs(self.grip) < 1e-9)

    def clipped(self) -> 'TeleopCommand':
        clip = lambda values: tuple(float(np.clip(value, -1.0, 1.0)) for value in values)
        return TeleopCommand(t=self.t, xyz=clip(self.xyz), rpy=clip(self.rpy),
                             quaternion=self.quaternion,
                             grip=float(np.clip(self.grip, -1.0, 1.0)),
                             grip_target=self.grip_target,
                             enable=self.enable, buttons=dict(self.buttons))


class TeleopSource(abc.ABC):
    def connect(self) -> 'TeleopSource':
        return self

    def close(self) -> None:
        pass

    def __enter__(self):
        return self.connect()

    def __exit__(self, exc_type, exc, traceback):
        self.close()
        return False

    @abc.abstractmethod
    def poll(self) -> TeleopCommand | None:
        """Return the latest command, or ``None`` when no fresh input exists."""


class TeleopBridge:
    """Latest-command mailbox used between an input thread and a control loop.

    Dropping old commands is intentional: a delayed joystick sample must never
    be executed later.  ``stale_s`` lets the control loop detect a dead device.
    """

    def __init__(self, stale_s: float = 0.3, clock=time.monotonic):
        self.stale_s = float(stale_s)
        self.clock = clock
        self._lock = threading.Lock()
        self._command: TeleopCommand | None = None
        self._stamp = 0.0
        self._sequence = 0

    def put(self, command: TeleopCommand) -> None:
        with self._lock:
            self._command = command
            self._stamp = self.clock()
            self._sequence += 1

    def get(self, consume: bool = False) -> TeleopCommand | None:
        with self._lock:
            command = self._command
            if consume:
                self._command = None
                self._stamp = 0.0
            return command

    def fresh(self, now: float | None = None) -> bool:
        with self._lock:
            return self._command is not None and (
                (now if now is not None else self.clock()) - self._stamp) <= self.stale_s

    def age(self) -> float:
        with self._lock:
            return float('inf') if self._command is None else self.clock() - self._stamp

    @property
    def sequence(self) -> int:
        with self._lock:
            return self._sequence

    def clear(self) -> None:
        with self._lock:
            self._command = None
            self._stamp = 0.0


class BridgeSource(TeleopSource):
    """Expose a ``TeleopBridge`` as a source (input produced by another thread)."""

    def __init__(self, bridge: TeleopBridge):
        self.bridge = bridge

    def poll(self) -> TeleopCommand | None:
        return self.bridge.get() if self.bridge.fresh() else None


class ScriptedSource(TeleopSource):
    """Deterministic command sequence for tests and demos.

    Each entry is ``(start_time_s, TeleopCommand)``; ``poll`` returns the most
    recent command or ``None`` before the first one.
    """

    def __init__(self, entries, clock=time.monotonic, loop: bool = False):
        self.entries = list(entries)
        self.clock = clock
        self.loop = bool(loop)
        self._t0 = clock()
        self._index = 0

    def restart(self) -> None:
        self._t0 = self.clock()
        self._index = 0

    def poll(self) -> TeleopCommand | None:
        if not self.entries:
            return None
        elapsed = self.clock() - self._t0
        if self.loop:
            duration = self.entries[-1][0] + 1e-6
            elapsed = elapsed % duration
            self._index = 0
        while self._index + 1 < len(self.entries) and elapsed >= self.entries[self._index + 1][0]:
            self._index += 1
        start, command = self.entries[self._index]
        return command if elapsed >= start else None


class CallableSource(TeleopSource):
    """Wrap a callable returning a ``TeleopCommand`` (or ``None``)."""

    def __init__(self, function):
        self.function = function

    def poll(self) -> TeleopCommand | None:
        return self.function()
