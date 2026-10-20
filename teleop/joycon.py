"""Joy-Con input source and edge-triggered button helpers.

The Joy-Con driver (``joycon-robotics``) lives in the repository but is not a
normal importable package because of the hyphen in its folder name.  This module
therefore accepts an already constructed controller, and offers
``load_joyconrobotics`` for callers that want the default one.

Nothing here talks to a robot: the source only emits ``TeleopCommand``.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from ..core.mathutils import euler_to_matrix, matrix_to_quat
from .base import TeleopCommand, TeleopSource


def stick_axes(vertical: int, horizontal: int, r: bool = False, zr: bool = False,
               center=(2048, 2048), deadzone: int = 350):
    """Map the right stick + R/ZR triggers to a normalized XYZ command.

    Matches the mapping already used by the simulators: stick = XY plane,
    R = +Z, ZR = -Z.
    """
    v = int(vertical) - int(center[0])
    h = int(horizontal) - int(center[1])
    x = int(v > deadzone) - int(v < -deadzone)
    y = int(h < -deadzone) - int(h > deadzone)
    z = int(bool(r)) - int(bool(zr))
    return [float(x), float(y), float(z)]


class GripperToggle:
    """Edge-triggered open/close toggle, like the existing Joy-Con behaviour."""

    def __init__(self, width: float, midpoint: float | None = None,
                 max_width: float = 0.085):
        self.goal = float(width)
        self.held = False
        self.max_width = float(max_width)
        self.midpoint = float(width) * 0.5 if midpoint is None else float(midpoint)

    def update(self, pressed: bool, width: float, enabled: bool = True) -> float:
        pressed = bool(pressed)
        if not enabled:
            self.goal = float(width)
        elif pressed and not self.held:
            self.goal = 0.0 if self.goal > self.midpoint else self.max_width
        self.held = pressed
        return self.goal


class EnableLatch:
    """Edge-triggered enable toggle; ``paused`` forces the disabled state.

    The previous version returned ``bool(paused)`` on a press, which is the
    current pause state rather than the desired new state.  With an initially
    disabled ``CartesianTeleop`` that meant the first Y press could disable
    instead of enable.
    """

    def __init__(self, initially_held: bool = True, enabled: bool = False):
        self.held = bool(initially_held)
        self.enabled = bool(enabled)

    def update(self, pressed: bool, paused: bool):
        pressed = bool(pressed)
        paused = bool(paused)
        action = None
        if paused:
            if self.enabled:
                self.enabled = False
                action = False
        elif pressed and not self.held:
            self.enabled = not self.enabled
            action = self.enabled
        self.held = pressed
        return action


def load_joyconrobotics(repo_root=None):
    """Import ``joyconrobotics`` from ``projects/joycon-robotics``."""
    root = Path(repo_root).resolve() if repo_root else Path(__file__).resolve().parents[3]
    candidate = root / 'projects' / 'joycon-robotics'
    if not candidate.exists():
        candidate = root / 'joycon-robotics'
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))
    from joyconrobotics import JoyconRobotics  # noqa: WPS433 (dynamic path)
    return JoyconRobotics


class JoyConSource(TeleopSource):
    """Adapt a ``JoyconRobotics``-style controller to ``TeleopCommand``.

    Expected duck-typed controller::

        controller.get_control() -> (pose7, gripper_width, button)
        controller.joycon.get_button_r() / get_button_zr() / get_button_b()
        controller.joycon.get_stick_right_vertical() / ..._horizontal()

    ``pose7`` is ``[x, y, z, roll, pitch, yaw]`` in radians.
    """

    def __init__(self, controller, gripper_max: float = 0.085, deadzone: int = 350,
                 stick_center=None, orientation_offset=None, yaw_degrees: float = 90.0,
                 pivot: str = 'wrist', stale_s: float = 0.3, clock=time.monotonic,
                 gripper_button: str = 'b', enable_button: str = 'y'):
        self.controller = controller
        self.gripper_max = float(gripper_max)
        self.deadzone = int(deadzone)
        self.clock = clock
        self.stale_s = float(stale_s)
        self.pivot = pivot
        self.gripper_button = str(gripper_button).lower()
        self.enable_button = str(enable_button).lower()
        self.last_input = clock()
        if orientation_offset is None:
            offset = euler_to_matrix([0.0, np.radians(yaw_degrees), 0.0])
        else:
            offset = np.asarray(orientation_offset, dtype=float).reshape(3, 3)
        self.orientation_offset = offset
        self.toggle = GripperToggle(self.gripper_max, midpoint=self.gripper_max * 0.5,
                                    max_width=self.gripper_max)
        self.latch = EnableLatch(enabled=False)
        self._stick_center = stick_center
        self._last_pose = None
        self._hooked = False
        self._attach_freshness_hook()

    def _attach_freshness_hook(self):
        joycon = getattr(self.controller, 'joycon', None)
        hooks = getattr(joycon, '_input_hooks', None)
        if hooks is not None:
            hooks.append(lambda *args: setattr(self, 'last_input', self.clock()))
            self._hooked = True

    def connect(self):
        self.last_input = self.clock()
        self.latch = EnableLatch(enabled=False)
        return self

    def close(self):
        for name in ('disconnnect', 'disconnect', 'close'):
            method = getattr(self.controller, name, None)
            if callable(method):
                try:
                    method()
                except Exception:
                    pass
                return

    def calibrate(self):
        """Run the driver's horizontal calibration when it is available."""
        method = getattr(self.controller, 'reset_joycon', None)
        if callable(method):
            method()
            self.last_input = self.clock()
            return True
        return False

    def reset_yaw(self):
        """Reset the gyro yaw reference without a full re-calibration."""
        changed = False
        gyro = getattr(self.controller, 'gyro', None)
        reset = getattr(gyro, 'reset_orientation', None)
        if callable(reset):
            reset()
            changed = True
        sensor = getattr(self.controller, 'orientation_sensor', None)
        reset = getattr(sensor, 'reset_yaw', None)
        if callable(reset):
            reset()
            changed = True
        return changed

    def _buttons(self):
        return getattr(self.controller, 'joycon', None)

    def _stick_center_value(self):
        if self._stick_center is not None:
            return self._stick_center
        buttons = self._buttons()
        self._stick_center = (buttons.get_stick_right_vertical(),
                              buttons.get_stick_right_horizontal())
        return self._stick_center

    def _button(self, name: str) -> bool:
        buttons = self._buttons()
        method = getattr(buttons, f'get_button_{name}', None)
        if method is None:
            return False
        try:
            return bool(method())
        except Exception:
            return False

    def _button_snapshot(self) -> dict:
        names = ('a', 'b', 'x', 'y', 'r', 'zr', 'home', 'plus', 'minus')
        snapshot = {name: self._button(name) for name in names}
        snapshot['stick'] = self._button('right_stick') or self._button('stick_r')
        return snapshot

    def fresh(self) -> bool:
        joycon = self._buttons()
        received = getattr(joycon, 'report_received_at', None)
        if received is not None:
            try:
                self.last_input = max(self.last_input, float(received))
            except (TypeError, ValueError):
                pass
        return (self.clock() - self.last_input) < self.stale_s

    def poll(self) -> TeleopCommand | None:
        if not self.fresh():
            return None
        try:
            pose, grip, _button = self.controller.get_control()
        except Exception:
            return None
        buttons = self._buttons()
        if not self._hooked and getattr(buttons, 'report_received_at', None) is None:
            # A generic controller returns live data; only drivers that expose
            # cached values need the explicit input hook above.  If the driver
            # exposes report_received_at, preserve it so stale input can be
            # detected instead of being refreshed by every poll.
            self.last_input = self.clock()
        if buttons is None:
            return None
        pose = np.asarray(pose, dtype=float)
        vertical = buttons.get_stick_right_vertical()
        horizontal = buttons.get_stick_right_horizontal()
        xyz = stick_axes(vertical, horizontal, buttons.get_button_r(),
                         buttons.get_button_zr(), center=self._stick_center_value(),
                         deadzone=self.deadzone)
        rotation = euler_to_matrix(pose[3:6]) @ self.orientation_offset
        quaternion = matrix_to_quat(rotation).tolist()
        paused = bool(getattr(getattr(self.controller, 'gyro', None), 'is_calibrating', False))
        enable = self.latch.update(self._button(self.enable_button), paused)
        grip_delta = 0.0
        if enable is True or (enable is None and self.latch.enabled and not paused):
            goal = self.toggle.update(self._button(self.gripper_button), float(grip),
                                      enabled=True)
            grip_delta = float(np.clip((goal - float(grip)) / max(self.gripper_max, 1e-9), -1, 1))
        return TeleopCommand(t=self.clock(), xyz=tuple(xyz), rpy=(0.0, 0.0, 0.0),
                             quaternion=tuple(quaternion), grip=grip_delta,
                             enable=enable,
                             buttons=self._button_snapshot())
