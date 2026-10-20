"""Robotiq 2F gripper without ROS.

Transport options, chosen with ``RobotiqConfig.backend``:

``mock``
    Offline; width follows commands through the shared mock model.
``urscript``
    URCap ``rq_*`` functions over the URScript secondary port (30002).
    Works on CB3, where tool RS485 pass-through does not exist.
``urcap_socket``
    Robotiq URCap XML-RPC-ish socket (63352) when the pendant runs the URCap.
``tool_modbus``
    Modbus RTU tunnelled through the UR tool RS485 interface (e-Series only).
``modbus_rtu``
    A USB-RS485 adapter on the PC.

All widths are metres; register values are percentages, so the conversion lives
in exactly one place.
"""

from __future__ import annotations

import platform
import socket
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from ..errors import ConnectionFailed, RobotUnavailable
from .base import GripperBackend, GripperState

ACTIVATE_REG = 0x03E8
STATUS_REG = 0x07D0


@dataclass
class RobotiqConfig:
    backend: str = 'mock'
    ip: str = ''
    max_width_m: float = 0.085
    force: int = 100
    speed: int = 150
    tool_comm_port: int = 54321
    script_port: int = 30002
    urcap_port: int = 63352
    serial_port: str = field(default_factory=lambda: ('COM5' if platform.system() == 'Windows'
                                                        else '/dev/ttyUSB0'))
    baudrate: int = 115200
    slave: int = 1
    urscript_prefix: str = 'rq_'
    command_timeout_s: float = 2.0
    poll_hz: float = 5.0
    close_settle_s: float = 0.4
    open_settle_s: float = 0.2
    # If true, a stale URCap measurement raises instead of silently returning
    # the last commanded width (the Mac deployment control fix).
    require_fresh_measurement: bool = False


def _rtu_framer():
    try:
        from pymodbus.framer import FramerType
        return FramerType.RTU
    except ImportError:  # older pymodbus
        from pymodbus.framer.rtu_framer import ModbusRtuFramer
        return ModbusRtuFramer


class RobotiqGripper(GripperBackend):
    def __init__(self, config: RobotiqConfig | None = None, robot_ip: str | None = None):
        self.config = config or RobotiqConfig()
        super().__init__(name='robotiq_2f85', max_width=self.config.max_width_m)
        if robot_ip:
            self.config.ip = robot_ip
        self.client = None
        self._script_socket = None
        self._urcap_socket = None
        self._urcap_lock = threading.Lock()
        self._commanded = self.config.max_width_m
        self._measured = self.config.max_width_m
        self._measured_at = 0.0
        self._lock = threading.Lock()
        self._poll_stop = threading.Event()
        self._poll_thread = None

    @property
    def mock(self) -> bool:
        return self.config.backend == 'mock'

    # ---------------------------------------------------------- unit mapping
    def _position(self, width: float) -> int:
        width = float(np.clip(width, 0.0, self.config.max_width_m))
        return int(round(255.0 * (1.0 - width / self.config.max_width_m)))

    def _width(self, position: int) -> float:
        return self.config.max_width_m * (1.0 - position / 255.0)

    @property
    def width(self) -> float:
        return self.latest_width()

    @property
    def commanded_width(self) -> float:
        with self._lock:
            return self._commanded

    def latest_width(self) -> float:
        with self._lock:
            fresh = self.mock or time.perf_counter() - self._measured_at < 1.0
            if fresh:
                return self._measured
            if self.config.require_fresh_measurement:
                raise RobotUnavailable(
                    '夹爪实测宽度缺失或超过 1 秒，拒绝用目标宽度代替观测')
            return self._commanded

    # ------------------------------------------------------------------ link
    def connect(self) -> 'RobotiqGripper':
        if self.mock:
            self._connected = True
            return self
        if self.config.backend == 'urscript':
            self._script_socket = self._connect_script_socket()
            self._connected = True
            return self
        if self.config.backend == 'urcap_socket':
            self._urcap_socket = self._connect_urcap_socket()
            self._connected = True
            return self
        if self.config.backend in ('tool_modbus', 'modbus_rtu'):
            from pymodbus.client import ModbusSerialClient, ModbusTcpClient
            if self.config.backend == 'tool_modbus':
                address = self._robot_address()
                self.client = ModbusTcpClient(address, port=self.config.tool_comm_port,
                                              framer=_rtu_framer(), timeout=0.2)
            else:
                self.client = ModbusSerialClient(port=self.config.serial_port,
                                                 framer=_rtu_framer(),
                                                 baudrate=self.config.baudrate, bytesize=8,
                                                 parity='N', stopbits=1, timeout=0.2)
            if not self.client.connect():
                raise ConnectionFailed(f'Robotiq Modbus 连接失败：{self.config.backend}')
            self._connected = True
            return self
        raise ValueError(f'未知 Robotiq backend：{self.config.backend}')

    def _robot_address(self) -> str:
        if not self.config.ip:
            raise RobotUnavailable('需要机器人 IP：设置 RobotiqConfig.ip 或传入 robot_ip')
        return str(self.config.ip)

    def _connect_script_socket(self):
        connection = socket.create_connection((self._robot_address(), int(self.config.script_port)),
                                              timeout=2.0)
        connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        return connection

    def _send_script(self, line: str) -> None:
        payload = (line.rstrip('\n') + '\n').encode('ascii')
        try:
            self._script_socket.sendall(payload)
        except OSError:
            self._script_socket = self._connect_script_socket()
            self._script_socket.sendall(payload)

    def _connect_urcap_socket(self):
        address = self._robot_address()
        try:
            connection = socket.create_connection((address, int(self.config.urcap_port)),
                                                  timeout=3.0)
        except OSError as exc:
            raise ConnectionFailed(
                f'无法连接 Robotiq URCap 服务 {address}:{self.config.urcap_port}：{exc}；'
                '确认示教器已加载 Gripper URCap 且夹爪面板在线') from exc
        connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        return connection

    def _urcap_cmd(self, line: str, timeout: float = 2.0) -> str:
        with self._urcap_lock:
            payload = (line.rstrip('\n') + '\n').encode('ascii')
            try:
                self._urcap_socket.sendall(payload)
            except OSError:
                self._urcap_socket = self._connect_urcap_socket()
                self._urcap_socket.sendall(payload)
            self._urcap_socket.settimeout(min(0.2, max(0.02, float(timeout))))
            deadline = time.monotonic() + timeout
            buffer = b''
            while time.monotonic() < deadline:
                try:
                    chunk = self._urcap_socket.recv(256)
                except socket.timeout:
                    continue
                except OSError:
                    break
                if not chunk:
                    break
                buffer += chunk
                if b'\n' in buffer:
                    break
            return buffer.decode('ascii', errors='replace').strip()

    def _urcap_int(self, field: str):
        parts = self._urcap_cmd(f'GET {field}').split()
        if len(parts) >= 2 and parts[0].upper() == field.upper():
            try:
                return int(parts[1], 0)
            except ValueError:
                return None
        return None

    def get_position(self):
        return self._urcap_int('POS')

    # --------------------------------------------------------------- modbus
    def _write(self, address: int, values):
        errors = []
        for key in ('slave', 'unit'):
            try:
                return self.client.write_registers(address, values, **{key: self.config.slave})
            except TypeError as exc:
                errors.append(exc)
        raise errors[-1]

    def _status_registers(self, count: int = 8):
        errors = []
        for key in ('slave', 'unit'):
            for positional in (False, True):
                try:
                    if positional:
                        result = self.client.read_input_registers(
                            STATUS_REG, count, **{key: self.config.slave})
                    else:
                        result = self.client.read_input_registers(
                            STATUS_REG, count=count, **{key: self.config.slave})
                    if result is None:
                        continue
                    if getattr(result, 'isError', lambda: False)():
                        raise IOError(f'Robotiq 读取失败：{result}')
                    return list(result.registers)
                except TypeError as exc:
                    errors.append(exc)
        raise errors[-1]

    # ------------------------------------------------------------- commands
    def activate(self, timeout: float | None = None) -> bool:
        timeout = float(timeout if timeout is not None else self.config.command_timeout_s)
        if self.mock:
            with self._lock:
                self._commanded = self.config.max_width_m
                self._measured = self.config.max_width_m
                self._measured_at = time.perf_counter()
            return True
        if self.config.backend == 'urscript':
            self._send_script(f'{self.config.urscript_prefix}activate_and_wait()')
            return True
        if self.config.backend == 'urcap_socket':
            if self._urcap_int('ACT') == 1:
                return True
            self._urcap_cmd('SET ACT 1', timeout=0.05)
            deadline = time.monotonic() + max(timeout, 5.0)
            while time.monotonic() < deadline:
                if self._urcap_int('ACT') == 1:
                    return True
                time.sleep(0.25)
            raise RobotUnavailable('Robotiq 激活超时（urcap_socket）')
        self._write(ACTIVATE_REG, [0x00, 0x00, 0x00, 0x00])
        time.sleep(0.05)
        self._write(ACTIVATE_REG, [0x01, 0x00, 0x00, 0x00])
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            registers = self._status_registers(3)
            if registers[1] == 3:
                return True
            time.sleep(0.02)
        raise RobotUnavailable('Robotiq 激活超时：检查工具端供电、Tool Communication 与波特率 115200')

    def move(self, width, force=None, speed=None, wait=False, timeout=None) -> bool:
        width = self.clamp(width)
        with self._lock:
            self._commanded = width
        if self.mock:
            with self._lock:
                self._measured = width
                self._measured_at = time.perf_counter()
            return True
        force = int(self.config.force if force is None else force)
        speed = int(self.config.speed if speed is None else speed)
        if self.config.backend == 'urscript':
            command = 'move_and_wait' if wait else 'move'
            self._send_script(f'{self.config.urscript_prefix}{command}({self._position(width)})')
            return True
        if self.config.backend == 'urcap_socket':
            self._urcap_cmd(f'SET SPE {speed}', timeout=0.05)
            self._urcap_cmd(f'SET FOR {force}', timeout=0.05)
            self._urcap_cmd('SET GTO 1', timeout=0.05)
            self._urcap_cmd(f'SET POS {self._position(width)}', timeout=0.05)
            return self.wait(timeout) if wait else True
        self._write(ACTIVATE_REG, [0x09, self._position(width), speed, force])
        return self.wait(timeout) if wait else True

    def wait(self, timeout=None, tolerance: float = 0.002, rate_hz: float = 20.0) -> bool:
        if self.mock:
            return True
        timeout = float(timeout if timeout is not None else self.config.command_timeout_s)
        if self.config.backend == 'urcap_socket':
            settle = self.config.close_settle_s if self._commanded <= 1e-6 else self.config.open_settle_s
            start = time.monotonic()
            deadline = start + max(timeout, settle + 2.0)
            last = None
            stable = 0
            seen = False
            while time.monotonic() < deadline:
                position = self.get_position()
                if position is None:
                    time.sleep(0.1)
                    continue
                seen = True
                with self._lock:
                    self._measured = self._width(position)
                    self._measured_at = time.perf_counter()
                if last is not None and position == last and time.monotonic() - start >= settle:
                    stable += 1
                    if stable >= 2:
                        break
                else:
                    stable = 0
                last = position
                time.sleep(0.15)
            return seen
        start = time.monotonic()
        deadline = start + timeout
        while time.monotonic() < deadline:
            registers = self._status_registers(3)
            if registers[1] in (1, 2, 3) and time.monotonic() - start > 0.2:
                return True
            time.sleep(0.02)
        return False

    def stop(self) -> bool:
        if self.mock:
            return True
        if self.config.backend == 'urscript':
            self._send_script(f'{self.config.urscript_prefix}stop()')
            return True
        if self.config.backend == 'urcap_socket':
            self._urcap_cmd('SET GTO 0', timeout=0.05)
            return True
        self._write(ACTIVATE_REG, [0x00, 0x00, 0x00, 0x00])
        return True

    def state(self) -> GripperState:
        return GripperState(width=self.width, target=self._commanded,
                            moving=abs(self.width - self._commanded) > 0.002,
                            connected=self.connected, t=time.monotonic())

    # -------------------------------------------------------------- polling
    def start_polling(self) -> None:
        if self._poll_thread is not None or self.mock or self.config.backend == 'urscript':
            return
        # Seed the cache before the first strict read: otherwise a caller that
        # starts polling and immediately reads would see the zero timestamp.
        try:
            if self.config.backend == 'urcap_socket':
                position = self.get_position()
            else:
                position = self._status_registers(8)[7]
            if position is not None:
                with self._lock:
                    self._measured = self._width(position)
                    self._measured_at = time.perf_counter()
        except Exception:
            pass
        self._poll_stop.clear()
        self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._poll_thread.start()

    def _poll_loop(self) -> None:
        period = 1.0 / max(1.0, float(self.config.poll_hz))
        while not self._poll_stop.wait(period):
            try:
                if self.config.backend == 'urcap_socket':
                    position = self.get_position()
                    if position is None:
                        continue
                else:
                    registers = self._status_registers(8)
                    position = registers[7]
            except Exception:
                continue
            with self._lock:
                self._measured = self._width(position)
                self._measured_at = time.perf_counter()

    def disconnect(self) -> None:
        self._poll_stop.set()
        if self._poll_thread is not None:
            self._poll_thread.join(timeout=1.0)
            self._poll_thread = None
        for attribute in ('client', '_script_socket', '_urcap_socket'):
            handle = getattr(self, attribute)
            if handle is not None:
                try:
                    handle.close()
                except Exception:
                    pass
                setattr(self, attribute, None)
        self._connected = False
