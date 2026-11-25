"""Task-config section ``agent``: how one station is exposed to agent frameworks."""

from __future__ import annotations

from dataclasses import dataclass, field

_IDENTITY = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
             [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]


@dataclass
class AgentOptions:
    arm_name: str = 'arm'
    robot_id: str = ''
    # World pose of the robot base, and the public control point in the backend TCP frame.
    T_world_base: list = field(default_factory=lambda: [row[:] for row in _IDENTITY])
    T_tcp_control: list = field(default_factory=lambda: [row[:] for row in _IDENTITY])
    control_site: str = 'grasp site between the fingertips'
    rotation_radius_m: float = 0.1
    calibration_paths: list = field(default_factory=list)
    # One command may move the control point at most this far from the measured pose.
    max_step_m: float = 0.05
    max_step_rad: float = 0.35
    max_joint_jump_rad: float = 0.5
    position_tolerance_m: float = 0.01
    rotation_tolerance_rad: float = 0.05
    joint_tolerance_rad: float = 0.02
    move_timeout_s: float = 20.0
    # Extra keyword arguments for RobotPlatform.move_j, e.g. {"use_native": false, "speed": 0.5}.
    move_options: dict = field(default_factory=dict)
    gripper_timeout_s: float = 5.0
    gripper_deadband: float = 0.05
    stationary_window_s: float = 0.2
    stationary_joint_rad: float = 0.002
    # Arrival may still ring; stillness is awaited this long before it counts as not observed.
    settle_timeout_s: float = 1.5
    control_frequency_hz: float = 2.0
    max_episode_steps: int = 400
