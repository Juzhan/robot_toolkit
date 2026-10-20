"""Keyboard teleoperation source with the same command contract as Joy-Con.

The keyboard is just another ``TeleopSource``:

* W/S/A/D/R/F: Cartesian X/Y/Z
* Arrow keys: yaw/pitch (Q/E: roll)
* Space/B: absolute gripper toggle
* Y or Enter: enable/pause
* Esc: quit

Windows uses ``GetAsyncKeyState`` so held keys work.  POSIX terminals do not
report key-up events; each keypress becomes a short configurable pulse.  Tests
can inject ``key_state`` and avoid both platform readers.
"""
from __future__ import annotations

import os
import select
import sys
import threading
import time

from .base import TeleopCommand, TeleopSource


_WINDOWS_KEYS = {
    'w': 0x57, 's': 0x53, 'a': 0x41, 'd': 0x44,
    'r': 0x52, 'f': 0x46, 'q': 0x51, 'e': 0x45,
    'y': 0x59, 'b': 0x42, 'space': 0x20, 'enter': 0x0D,
    'escape': 0x1B,
    'up': 0x26, 'down': 0x28, 'left': 0x25, 'right': 0x27,
}

_POSIX_KEYS = {
    'w': 'w', 's': 's', 'a': 'a', 'd': 'd', 'r': 'r', 'f': 'f',
    'q': 'q', 'e': 'e', 'y': 'y', 'b': 'b', 'space': ' ',
    'enter': '\r', 'escape': '\x1b',
}


class KeyboardSource(TeleopSource):
    def __init__(self, gripper_max: float = 0.08, pulse_s: float = 0.15,
                 key_state=None, clock=time.monotonic):
        self.gripper_max = float(gripper_max)
        self.pulse_s = max(0.02, float(pulse_s))
        self.clock = clock
        self._injected_state = key_state
        self._pulse: dict[str, float] = {}
        self._last_state: set[str] = set()
        self._enabled = False
        self._grip_target = self.gripper_max
        self._quit = False
        self._stop = threading.Event()
        self._thread = None
        self._fd = None
        self._old_termios = None

    # ------------------------------------------------------------------ link
    def connect(self):
        if self._injected_state is None and sys.platform != 'win32':
            self._start_posix_reader()
        return self

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._fd is not None and self._old_termios is not None:
            try:
                import termios
                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_termios)
            except Exception:
                pass
        self._fd = None
        self._old_termios = None

    def _start_posix_reader(self):
        try:
            fd = sys.stdin.fileno()
            import termios
            old = termios.tcgetattr(fd)
        except Exception:
            return
        self._fd = fd
        self._old_termios = old
        self._thread = threading.Thread(target=self._posix_loop, daemon=True,
                                        name='keyboard-teleop')
        self._thread.start()

    def _posix_loop(self):
        import termios
        import tty
        fd = self._fd
        try:
            tty.setcbreak(fd)
            while not self._stop.is_set():
                ready, _, _ = select.select([fd], [], [], 0.05)
                if not ready:
                    continue
                data = os.read(fd, 1)
                if not data:
                    continue
                if data == b'\x1b':
                    # Arrow/function key sequence; bare Esc becomes quit.
                    ready2, _, _ = select.select([fd], [], [], 0.02)
                    if ready2:
                        rest = os.read(fd, 2)
                        arrows = {b'[A': 'up', b'[B': 'down',
                                  b'[C': 'right', b'[D': 'left'}
                        name = arrows.get(rest)
                        if name:
                            self._mark(name)
                        continue
                    self._mark('escape')
                    continue
                try:
                    char = data.decode('utf-8')
                except UnicodeDecodeError:
                    continue
                if char == '\x03':
                    self._mark('escape')
                    continue
                if char == '\n':
                    self._mark('enter')
                    continue
                for name, value in _POSIX_KEYS.items():
                    if char == value:
                        self._mark(name)
                        break
        except Exception:
            return
        finally:
            try:
                termios.tcsetattr(fd, termios.TCSADRAIN, self._old_termios)
            except Exception:
                pass

    def _mark(self, name: str):
        self._pulse[name] = self.clock() + self.pulse_s

    def _windows_state(self) -> set[str]:
        import ctypes
        user32 = ctypes.windll.user32
        state = set()
        for name, code in _WINDOWS_KEYS.items():
            try:
                if user32.GetAsyncKeyState(code) & 0x8000:
                    state.add(name)
            except Exception:
                pass
        return state

    def _state(self) -> set[str]:
        if self._injected_state is not None:
            try:
                return set(self._injected_state())
            except Exception:
                return set()
        if sys.platform == 'win32':
            return self._windows_state()
        now = self.clock()
        self._pulse = {name: expiry for name, expiry in self._pulse.items()
                       if expiry > now}
        return set(self._pulse)

    # ------------------------------------------------------------------ data
    def poll(self) -> TeleopCommand | None:
        state = self._state()
        new = state - self._last_state
        enable = None
        grip_target = None
        if 'escape' in new:
            self._quit = True
        if 'y' in new or 'enter' in new:
            self._enabled = not self._enabled
            enable = self._enabled
        if 'space' in new or 'b' in new:
            midpoint = self.gripper_max * 0.5
            self._grip_target = self.gripper_max if self._grip_target <= midpoint else 0.0
            grip_target = self._grip_target
        self._last_state = state
        xyz = (
            float('w' in state) - float('s' in state),
            float('a' in state) - float('d' in state),
            float('r' in state) - float('f' in state),
        )
        rpy = (
            float('q' in state) - float('e' in state),
            float('up' in state) - float('down' in state),
            float('left' in state) - float('right' in state),
        )
        if self._quit:
            return TeleopCommand(t=self.clock(), enable=False,
                                 buttons={'quit': True})
        return TeleopCommand(t=self.clock(), xyz=xyz, rpy=rpy, grip=0.0,
                             grip_target=grip_target, enable=enable,
                             buttons={'enable': self._enabled,
                                      'grip': self._grip_target})
