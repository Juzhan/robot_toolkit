"""Reusable teleoperation session assembly.

Control flow lives in :class:`TeleopRunner`; this module only wires a robot,
gripper, Joy-Con source and Cartesian mapping together.  Applications such as
``joycon_teleop`` and ``piper_collect`` must use this builder instead of
reimplementing the same setup.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..core.conventions import RobotSpec
from ..gripper import MockGripper
from ..profiles import build_gripper, build_robot
from ..robot import MockRobot
from .base import ScriptedSource, TeleopCommand, TeleopSource
from .joycon import JoyConSource, load_joyconrobotics
from .keyboard import KeyboardSource
from .mapping import CartesianMappingConfig, CartesianTeleop
from .runner import TeleopRunner, TeleopRunnerConfig


PIPER_WORKSPACE = ((0.12, -0.35, 0.05), (0.45, 0.35, 0.35))
UR_WORKSPACE = ((-0.75, -0.75, 0.018), (0.85, 0.75, 1.0))


@dataclass
class TeleopSessionConfig:
    robot: str = 'piper_x'
    ip: str = '192.168.1.10'
    mock: bool = False
    input: str = 'joycon'
    keyboard_pulse_s: float = 0.15
    speed_percent: int = 20
    max_joint_speed: float = 1.0
    translation_speed: float = 0.08
    rotation_speed: float = 0.45
    smoothing: float = 0.2
    linear_acceleration: float = 0.6
    angular_acceleration: float = 1.0
    rate_hz: float = 50.0
    move_p_hz: float = 20.0
    stale_s: float = 0.4
    max_duration_s: float = 60.0
    tracking_translation_m: float | None = 0.03
    gripper_close_time_s: float = 0.8
    gripper_open_time_s: float = 0.5
    gripper_button: str = 'b'
    enable_button: str = 'y'
    calibrate: bool = False
    workspace_lower: tuple | None = None
    workspace_upper: tuple | None = None


@dataclass
class TeleopSession:
    robot: object
    gripper: object
    source: TeleopSource
    mapping: CartesianTeleop
    runner: TeleopRunner
    config: TeleopSessionConfig
    controller: object | None = None

    def run(self, on_sample=None, abort_event=None):
        return self.runner.run(on_sample=on_sample, abort_event=abort_event)

    def close(self):
        if self.source is not None:
            try:
                self.source.close()
            except Exception:
                pass
        if self.gripper is not None:
            try:
                self.gripper.stop()
            except Exception:
                pass
        if self.robot is not None:
            try:
                self.robot.disconnect()
            except Exception:
                pass


def _mock_robot():
    scale, offset = 0.1, (0.3, 0.0, 0.3)

    def forward(q):
        q = list(q)
        return [offset[0] + scale * q[0], offset[1] + scale * q[1],
                offset[2] + scale * q[2], 0.0, 0.0, 0.0]

    def inverse(pose):
        pose = list(pose)
        return [(pose[0] - offset[0]) / scale, (pose[1] - offset[1]) / scale,
                (pose[2] - offset[2]) / scale, 0.0, 0.0, 0.0]

    spec = RobotSpec.piper_x()
    robot = MockRobot(spec=spec, forward_kinematics=forward, inverse_kinematics=inverse)
    return robot, MockGripper(name=spec.gripper.name, max_width=spec.gripper.max_width)


def _joycon_source(config: TeleopSessionConfig) -> tuple[JoyConSource, object]:
    JoyconRobotics = load_joyconrobotics()
    controller = JoyconRobotics('right', gripper_open=0.08, gripper_close=0.0)
    source = JoyConSource(controller, gripper_max=0.08,
                          gripper_button=config.gripper_button,
                          enable_button=config.enable_button)
    if config.calibrate:
        source.calibrate()
    return source, controller


def build_teleop_session(config: TeleopSessionConfig | None = None,
                         source: TeleopSource | None = None) -> TeleopSession:
    """Build the same teleop stack for Piper, UR or the offline mock."""
    config = config or TeleopSessionConfig()
    if config.input not in ('joycon', 'keyboard'):
        raise ValueError(f'unknown teleop input {config.input!r}')
    mock = bool(config.mock or config.robot == 'mock')
    controller = None

    def keyboard_source() -> KeyboardSource:
        width = (robot.spec.gripper.max_width
                 if getattr(robot.spec, 'gripper', None) is not None else 0.08)
        return KeyboardSource(gripper_max=width, pulse_s=config.keyboard_pulse_s)

    if mock:
        robot, gripper = _mock_robot()
        lower, upper = PIPER_WORKSPACE
        if source is None:
            source = (keyboard_source() if config.input == 'keyboard'
                      else ScriptedSource([
                          (0.0, TeleopCommand(xyz=(1.0, 0.0, 0.0), enable=True)),
                      ], loop=True))
    elif config.robot == 'piper_x':
        robot = build_robot({
            'kind': 'agx_piper',
            'name': 'piper_x',
            'options': {'model': 'piper_x', 'speed_percent': config.speed_percent,
                        'max_joint_speed': config.max_joint_speed,
                        'enable_on_connect': True},
            'limits': {'max_velocity': config.max_joint_speed},
            'safety': True,
        })
        gripper = build_gripper('agx', robot=robot, options={
            'close_time_s': config.gripper_close_time_s,
            'open_time_s': config.gripper_open_time_s,
        })
        lower, upper = PIPER_WORKSPACE
        if source is None:
            if config.input == 'keyboard':
                source = keyboard_source()
            else:
                source, controller = _joycon_source(config)
    elif config.robot == 'ur_rtde':
        robot = build_robot({
            'kind': 'ur_rtde',
            'name': 'ur5e',
            'options': {'ip': config.ip, 'max_joint_speed': config.max_joint_speed},
            'safety': True,
        })
        gripper = None
        lower, upper = UR_WORKSPACE
        if source is None:
            if config.input == 'keyboard':
                source = keyboard_source()
            else:
                source, controller = _joycon_source(config)
    else:
        raise ValueError(f'unknown teleop robot {config.robot!r}')

    if config.workspace_lower is not None:
        lower = tuple(config.workspace_lower)
    if config.workspace_upper is not None:
        upper = tuple(config.workspace_upper)
    mapping = CartesianTeleop(CartesianMappingConfig(
        translation_speed=config.translation_speed,
        rotation_speed=config.rotation_speed,
        workspace_lower=lower,
        workspace_upper=upper,
        smoothing=config.smoothing,
        max_linear_acceleration=config.linear_acceleration,
        max_angular_acceleration=config.angular_acceleration,
    ), robot.spec.gripper)
    runner = TeleopRunner(robot, gripper, source, mapping, TeleopRunnerConfig(
        rate_hz=config.rate_hz,
        stale_s=config.stale_s,
        max_duration_s=config.max_duration_s,
        move_p_hz=config.move_p_hz,
        tracking_translation_m=config.tracking_translation_m,
    ))
    return TeleopSession(robot=robot, gripper=gripper, source=source,
                         mapping=mapping, runner=runner, config=config,
                         controller=controller)
