"""Joy-Con button hub for teach/session workflows.

This is intentionally separate from :mod:`teleop.joycon`: teleoperation needs
stick/gyro commands, while a teach/session loop only needs edge-triggered
buttons.  The hub keeps a long-lived connection, reconnects after Bluetooth
drops, and pushes ``(button, timestamp)`` events.
"""
from __future__ import annotations

import queue
import sys
import threading
import time
from pathlib import Path

BUTTONS = ('get_button_a', 'get_button_b', 'get_button_x', 'get_button_y',
           'get_button_r', 'get_button_zr', 'get_button_home', 'get_button_plus',
           'get_button_minus', 'get_button_right_sr')


def connect_joycon():
    """Connect a right (preferred) or left Joy-Con; return ``None`` on failure."""
    root = Path(__file__).resolve().parents[3]
    library = str(root / 'projects' / 'joycon-robotics')
    if library not in sys.path:
        sys.path.insert(0, library)
    from joyconrobotics.joycon import JoyCon

    class RobustJoyCon(JoyCon):
        # A finite HID read timeout lets the reader thread exit on stop.
        _READ_TIMEOUT_MS = 250

        def _read_input_report(self) -> bytes:
            return bytes(self._joycon_device.read(self._INPUT_REPORT_SIZE,
                                                   self._READ_TIMEOUT_MS))

        def _update_input_report(self):
            while self.enable:
                try:
                    report = self._read_input_report()
                except Exception:
                    break
                if not self.enable:
                    break
                if len(report) < 1 or report[0] != 0x30:
                    continue
                self._input_report = report
                self.report_received_at = time.monotonic()
                for callback in self._input_hooks:
                    callback(self)

    for product_id in (0x2007, 0x2006):  # right, then left
        try:
            return RobustJoyCon(vendor_id=0x057E, product_id=product_id)
        except Exception:
            continue
    return None


class ButtonHub(threading.Thread):
    """Poll a Joy-Con and push rising-edge button events."""

    _NO_REPORT_LIMIT = 2.0

    def __init__(self):
        super().__init__(daemon=True, name='joycon-buttons')
        self.events = queue.Queue()
        self.pressed: dict[str, bool] = {}
        self.connected = threading.Event()
        self.joycon = None
        self._stop_event = threading.Event()
        self._abandoned: list = []

    def is_pressed(self, button: str) -> bool:
        return bool(self.pressed.get(button, False))

    def _connect(self):
        return connect_joycon()

    def _retire(self, stale) -> None:
        stale.enable = False
        reader = getattr(stale, '_update_input_report_thread', None)
        if reader is not None:
            reader.join(1.0)
        if reader is None or not reader.is_alive():
            try:
                stale._close()
            except Exception:
                pass
        else:
            self._abandoned.append(stale)

    def run(self):
        states = {name: False for name in BUTTONS}
        last_press: dict[str, float] = {}
        last_report = None
        last_change = time.monotonic()
        last_notice = 0.0
        while not self._stop_event.is_set():
            if self.joycon is None:
                self.joycon = self._connect()
                if self.joycon is None:
                    now = time.monotonic()
                    if now - last_notice > 30.0:
                        print('Joy-Con 未连接（按一下手柄任意键唤醒），每 2 秒重试……',
                              flush=True)
                        last_notice = now
                    if self._stop_event.wait(2.0):
                        break
                    continue
                last_notice = 0.0
                print('Joy-Con 已连接，按键就绪。', flush=True)
                time.sleep(0.3)
                for name in BUTTONS:
                    try:
                        states[name] = bool(getattr(self.joycon, name)())
                    except Exception:
                        states[name] = False
                last_report = None
                last_change = time.monotonic()
                self.connected.set()
            try:
                reader = getattr(self.joycon, '_update_input_report_thread', None)
                if reader is not None and not reader.is_alive():
                    raise RuntimeError('后台读取线程已退出')
                report = getattr(self.joycon, '_input_report', None)
                if report is not None:
                    received_at = getattr(self.joycon, 'report_received_at', None)
                    if received_at is not None:
                        last_change = received_at
                        if time.monotonic() - last_change > self._NO_REPORT_LIMIT:
                            raise RuntimeError(
                                f'读取数据 {self._NO_REPORT_LIMIT:.1f}s 无更新')
                    elif report != last_report:
                        last_report = report
                        last_change = time.monotonic()
                    elif time.monotonic() - last_change > self._NO_REPORT_LIMIT:
                        raise RuntimeError(
                            f'读取数据 {self._NO_REPORT_LIMIT:.1f}s 无更新')
                current = {name: bool(getattr(self.joycon, name)()) for name in BUTTONS}
                self.pressed = {name.replace('get_button_', ''): value
                                for name, value in current.items()}
            except Exception as exc:
                self.pressed = {}
                print(f'Joy-Con 读取失败（{exc}），尝试重连……', flush=True)
                stale = self.joycon
                self.joycon = None
                if stale is not None:
                    self._retire(stale)
                self.connected.clear()
                last_report = None
                last_change = time.monotonic()
                if self._stop_event.wait(0.5):
                    break
                continue
            now = time.monotonic()
            for name, pressed in current.items():
                if pressed and not states[name] and now - last_press.get(name, -9.0) > 0.10:
                    last_press[name] = now
                    button = name.replace('get_button_', '')
                    self.events.put(('sr' if button == 'right_sr' else button, now))
                states[name] = pressed
            self._stop_event.wait(1.0 / 200.0)
        if self.joycon is not None:
            self._retire(self.joycon)
            self.joycon = None

    def stop(self):
        self._stop_event.set()
        if self.joycon is not None:
            self._retire(self.joycon)
            self.joycon = None
