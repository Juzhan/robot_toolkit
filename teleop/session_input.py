"""Joy-Con buttons and keyboard keys as one timestamped event stream.

Both inputs use the Joy-Con button names, so a session reacts identically to
``B`` on the controller and ``b`` on the keyboard. Events older than
``max_age_s`` when read (the loop was stalled) are dropped and counted instead of
being replayed late, which would silently flip a session's state.
"""
from __future__ import annotations

import queue
import threading
import time

KEYMAP = {'a': 'a', 'b': 'b', 'x': 'x', 'y': 'y', 'h': 'home', 'r': 'r', 'd': 'plus', 'q': 'quit'}


class DoublePress:
    """Second press of the same button within ``timeout_s`` confirms."""

    def __init__(self, timeout_s: float = 8.0):
        self.timeout_s = float(timeout_s)
        self._until: dict[str, float] = {}

    def confirm(self, name: str, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        until = self._until.get(name)
        if until is not None and now <= until:
            self._until.pop(name)
            return True
        self._until[name] = now + self.timeout_s
        return False

    def reset(self) -> None:
        self._until.clear()


class SessionInput:
    def __init__(self, joycon: bool = True, keyboard: bool = True, max_age_s: float = 0.5,
                 hub=None, keys=None):
        self.events: queue.Queue = queue.Queue()
        self.max_age_s = float(max_age_s)
        self.dropped = 0
        self._stop = threading.Event()
        self._threads = []
        self.hub = hub
        if self.hub is None and joycon:
            import importlib.util
            from .joycon import ensure_joyconrobotics_path
            missing = [name for name in ('hid', 'glm') if importlib.util.find_spec(name) is None]
            if not ensure_joyconrobotics_path() or missing:
                print('Joy-Con 不可用（缺少 '
                      f'{", ".join(missing) or "joyconrobotics"}：pip install hidapi PyGLM）；'
                      '使用键盘操作。', flush=True)
            else:
                from .joycon_buttons import ButtonHub
                self.hub = ButtonHub()
        self.keys = keys
        if self.keys is None and keyboard:
            from ..teach.console import KeyReader
            self.keys = KeyReader()

    def start(self) -> 'SessionInput':
        if self.hub is not None:
            if not self.hub.is_alive():
                self.hub.start()
            self._threads.append(threading.Thread(target=self._pump_hub, daemon=True,
                                                  name='session-joycon'))
        if self.keys is not None:
            self.keys.start()
            self._threads.append(threading.Thread(target=self._pump_keys, daemon=True,
                                                  name='session-keys'))
        for thread in self._threads:
            thread.start()
        return self

    def _pump_hub(self):
        while not self._stop.is_set():
            try:
                self.events.put(self.hub.events.get(timeout=0.05))
            except queue.Empty:
                continue

    def _pump_keys(self):
        while not self._stop.is_set():
            key = self.keys.get()
            if key is None:
                time.sleep(0.01)
                continue
            name = KEYMAP.get(key.lower())
            if name is not None:
                self.events.put((name, time.monotonic()))

    def get(self, timeout: float = 0.0):
        """Next fresh button name, or None when nothing arrives within ``timeout``."""
        deadline = time.monotonic() + max(0.0, timeout)
        while True:
            remaining = deadline - time.monotonic()
            try:
                name, stamp = (self.events.get(timeout=remaining) if remaining > 0
                               else self.events.get_nowait())
            except queue.Empty:
                return None
            if time.monotonic() - stamp > self.max_age_s:
                self.dropped += 1
                continue
            return name

    def flush(self) -> None:
        while True:
            try:
                self.events.get_nowait()
            except queue.Empty:
                return

    def stop(self) -> None:
        self._stop.set()
        if self.keys is not None:
            self.keys.stop()
        if self.hub is not None:
            self.hub.stop()
        for thread in self._threads:
            thread.join(timeout=0.5)
