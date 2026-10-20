"""Device-independent Cartesian mapping.

The same mapping is used for a simulated arm and a real one: it turns a
normalized ``TeleopCommand`` into a Cartesian target (position + rotation +
gripper width) inside a workspace box.  Safety and IK stay downstream.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.conventions import GripperSpec
from ..core.mathutils import (matrix_to_rotvec, quat_to_matrix, rotvec_to_matrix)
from .base import TeleopCommand


@dataclass
class CartesianMappingConfig:
    translation_speed: float = 0.08       # m/s at full stick deflection
    rotation_speed: float = 0.45          # rad/s at full deflection
    grip_speed: float = 0.0425            # m/s at full grip deflection
    workspace_lower: tuple = (-0.75, -0.75, 0.018)
    workspace_upper: tuple = (0.85, 0.75, 1.00)
    translation_deadband: float = 0.0     # normalized units
    max_position_step: float = 0.02       # m per update
    max_rotation_step: float = 0.12       # rad per update
    smoothing: float = 0.0                # 0 = raw, 1 = no movement
    max_linear_acceleration: float = 0.0  # m/s^2; 0 = no acceleration limit
    max_angular_acceleration: float = 0.0 # rad/s^2; 0 = no acceleration limit


@dataclass
class TeleopTarget:
    position: np.ndarray
    rotation: np.ndarray                  # 3x3
    gripper: float
    active: bool = True
    limited: bool = False
    t: float = 0.0

    @property
    def pose(self) -> np.ndarray:
        return np.concatenate([self.position, matrix_to_rotvec(self.rotation)])

    def copy(self) -> 'TeleopTarget':
        return TeleopTarget(self.position.copy(), self.rotation.copy(),
                            self.gripper, self.active, self.limited, self.t)


class CartesianTeleop:
    def __init__(self, config: CartesianMappingConfig | None = None,
                 gripper: GripperSpec | None = None, quaternion_order: str = 'xyzw'):
        self.config = config or CartesianMappingConfig()
        self.gripper = gripper
        self.quaternion_order = quaternion_order
        self.lower = np.asarray(self.config.workspace_lower, dtype=float)
        self.upper = np.asarray(self.config.workspace_upper, dtype=float)
        self.position = np.zeros(3)
        self.rotation = np.eye(3)
        self.grip = gripper.max_width if gripper else 0.0
        self.enabled = False
        self.home_position = None
        self.home_rotation = None
        self.limited = False
        self.linear_velocity = np.zeros(3)
        self.angular_velocity = np.zeros(3)

    # ---------------------------------------------------------------- setup
    def reset(self, position, rotation=None, gripper_width=None, enabled: bool = False):
        self.position = np.asarray(position, dtype=float).reshape(3).copy()
        if rotation is not None:
            rotation = np.asarray(rotation, dtype=float)
            self.rotation = (rotation.reshape(3, 3) if rotation.size == 9
                             else rotvec_to_matrix(rotation.reshape(3)))
        if gripper_width is not None:
            self.grip = self._clamp_grip(gripper_width)
        self.home_position = self.position.copy()
        self.home_rotation = self.rotation.copy()
        self.linear_velocity[:] = 0.0
        self.angular_velocity[:] = 0.0
        self.enabled = bool(enabled)
        return self

    def set_home(self, position=None, rotation=None):
        self.home_position = (self.position.copy() if position is None
                              else np.asarray(position, dtype=float).reshape(3))
        self.home_rotation = (self.rotation.copy() if rotation is None
                              else np.asarray(rotation, dtype=float).reshape(3, 3))

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)
        if not self.enabled:
            self.linear_velocity[:] = 0.0
            self.angular_velocity[:] = 0.0

    @property
    def target(self) -> TeleopTarget:
        return TeleopTarget(self.position.copy(), self.rotation.copy(), self.grip,
                            active=self.enabled, limited=self.limited)

    # ----------------------------------------------------------------- update
    def update(self, command: TeleopCommand, dt: float,
               measured_position=None, measured_rotation=None) -> TeleopTarget:
        dt = max(0.0, float(dt))
        if command.enable is not None:
            was_enabled = self.enabled
            self.enabled = bool(command.enable)
            if self.enabled and not was_enabled:
                # Re-anchor on the real robot so enabling never causes a jump
                # (the arm may have been moved by hand while disabled).
                if measured_position is not None:
                    self.position = np.asarray(measured_position, dtype=float).reshape(3).copy()
                if measured_rotation is not None:
                    self.rotation = np.asarray(measured_rotation, dtype=float).reshape(3, 3).copy()
                self.linear_velocity[:] = 0.0
                self.angular_velocity[:] = 0.0
        if not self.enabled:
            self.limited = False
            self.linear_velocity[:] = 0.0
            self.angular_velocity[:] = 0.0
            return self.target

        # ------------------------------------------------------ translation
        xyz = np.asarray(command.xyz, dtype=float).reshape(3)
        if self.config.translation_deadband > 0:
            magnitude = float(np.linalg.norm(xyz))
            if magnitude <= self.config.translation_deadband:
                xyz = np.zeros(3)
            else:
                span = max(1e-9, 1.0 - self.config.translation_deadband)
                magnitude_new = min(1.0, (magnitude - self.config.translation_deadband) / span)
                xyz = xyz / magnitude * magnitude_new
        step = xyz * self.config.translation_speed * dt
        length = float(np.linalg.norm(step))
        if length > self.config.max_position_step:
            step *= self.config.max_position_step / length
        if self.config.max_linear_acceleration > 0 and dt > 0:
            desired_velocity = step / dt
            delta_velocity = desired_velocity - self.linear_velocity
            max_delta = self.config.max_linear_acceleration * dt
            delta_length = float(np.linalg.norm(delta_velocity))
            if delta_length > max_delta > 0:
                delta_velocity *= max_delta / delta_length
            self.linear_velocity += delta_velocity
            step = self.linear_velocity * dt
        elif dt > 0:
            self.linear_velocity = step / dt
        raw_position = self.position + step
        clamped_position = np.clip(raw_position, self.lower, self.upper)
        limited = not np.allclose(clamped_position, raw_position, atol=1e-12)
        alpha = float(np.clip(self.config.smoothing, 0.0, 1.0))
        self.position = (clamped_position if alpha <= 0
                         else self.position + (clamped_position - self.position) * (1.0 - alpha))

        # -------------------------------------------------------- rotation
        if command.quaternion is not None:
            desired = quat_to_matrix(command.quaternion, order=self.quaternion_order)
            delta = matrix_to_rotvec(desired @ self.rotation.T)
        else:
            rates = np.asarray(command.rpy, dtype=float).reshape(3)
            delta = rates * self.config.rotation_speed * dt
        angle = float(np.linalg.norm(delta))
        if angle > self.config.max_rotation_step:
            delta *= self.config.max_rotation_step / angle
            limited = True
        if self.config.max_angular_acceleration > 0 and dt > 0:
            desired_velocity = delta / dt
            delta_velocity = desired_velocity - self.angular_velocity
            max_delta = self.config.max_angular_acceleration * dt
            delta_length = float(np.linalg.norm(delta_velocity))
            if delta_length > max_delta > 0:
                delta_velocity *= max_delta / delta_length
            self.angular_velocity += delta_velocity
            delta = self.angular_velocity * dt
        elif dt > 0:
            self.angular_velocity = delta / dt
        self.rotation = rotvec_to_matrix(delta) @ self.rotation

        # ---------------------------------------------------------- gripper
        if self.gripper is not None:
            if command.grip_target is not None:
                self.grip = self._clamp_grip(command.grip_target)
            else:
                self.grip = self._clamp_grip(
                    self.grip + float(command.grip) * self.config.grip_speed * dt)

        self.limited = limited
        return self.target

    def _clamp_grip(self, width: float) -> float:
        if self.gripper is None:
            return float(width)
        return self.gripper.clamp(width)

    def home(self, dt: float = 0.02) -> TeleopTarget:
        if self.home_position is not None:
            self.position = self.home_position.copy()
        if self.home_rotation is not None:
            self.rotation = self.home_rotation.copy()
        self.linear_velocity[:] = 0.0
        self.angular_velocity[:] = 0.0
        return self.target
