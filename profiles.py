"""Profile builders: turn plain configuration into concrete backends.

Applications should depend on this module rather than on a hardware class, so
switching from ``mock`` to ``ur_rtde`` or ``mujoco`` is a configuration change.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .camera import Camera, CameraHub, MockCamera, OpenCVCamera, RealSenseCamera
from .core.conventions import GripperSpec, RobotSpec
from .core.safety import JointLimits, SafetyGuard, WorkspaceBox
from .errors import ConfigurationError
from .gripper import AgxGripper, GripperBackend, MockGripper, RobotiqConfig, RobotiqGripper
from .robot import (AgxPiperConfig, AgxPiperRobot, MockRobot, SafeRobot,
                    UrRtdeConfig, UrRtdeRobot)
from .robot.base import RobotBackend


@dataclass
class RobotProfile:
    name: str = 'mock'
    kind: str = 'mock'
    spec: RobotSpec = field(default_factory=lambda: RobotSpec.generic(6, 'mock'))
    options: dict = field(default_factory=dict)
    limits: dict = field(default_factory=dict)
    safety: bool = True


def build_robot(profile: RobotProfile | dict | str = 'mock', mock: bool | None = None) -> RobotBackend:
    piper_kinds = ('agx_piper', 'piper', 'piper_x', 'piper_h', 'piper_l')
    if isinstance(profile, str):
        spec = (RobotSpec.ur5e() if profile == 'ur_rtde'
                else RobotSpec.piper_x() if profile in piper_kinds
                else RobotSpec.generic(6, profile))
        profile = RobotProfile(name=profile,
                               kind='agx_piper' if profile in piper_kinds else profile,
                               spec=spec)
    elif isinstance(profile, dict):
        kind = profile.get('kind', profile.get('name', 'mock'))
        if kind in piper_kinds:
            kind = 'agx_piper'
        default_spec = (RobotSpec.piper_x() if kind == 'agx_piper'
                        else RobotSpec.ur5e() if kind == 'ur_rtde'
                        else RobotSpec.generic(6, profile.get('name', 'mock')))
        profile = RobotProfile(name=profile.get('name', profile.get('kind', 'mock')),
                               kind=kind,
                               spec=RobotSpec.from_dict(profile['spec']) if 'spec' in profile
                               else default_spec,
                               options=dict(profile.get('options', {})),
                               limits=dict(profile.get('limits', {})),
                               safety=bool(profile.get('safety', True)))

    default_lower = default_upper = None
    if profile.kind == 'mock':
        robot = MockRobot(profile.spec, **profile.options)
    elif profile.kind == 'ur_rtde':
        config = UrRtdeConfig(**{key: value for key, value in profile.options.items()
                                 if key in UrRtdeConfig.__dataclass_fields__})
        robot = UrRtdeRobot(config, spec=profile.spec or RobotSpec.ur5e(), mock=mock)
    elif profile.kind == 'agx_piper':
        options = dict(profile.options)
        options.setdefault('model', profile.name if profile.name in piper_kinds else 'piper_x')
        config = AgxPiperConfig(**{key: value for key, value in options.items()
                                   if key in AgxPiperConfig.__dataclass_fields__})
        robot = AgxPiperRobot(config, spec=profile.spec or RobotSpec.piper_x(), mock=mock)
        default_lower, default_upper = config.limits_lower, config.limits_upper
    elif profile.kind == 'mujoco':
        from .robot.mujoco import MujocoRobot
        robot = MujocoRobot(spec=profile.spec, **profile.options)
    else:
        raise ConfigurationError(f'unknown robot kind {profile.kind!r}')

    if profile.safety:
        dof = robot.dof
        lower = profile.limits.get('lower', default_lower or [-3.15] * dof)
        upper = profile.limits.get('upper', default_upper or [3.15] * dof)
        velocity = profile.limits.get('max_velocity', 1.0)
        guard = SafetyGuard(JointLimits(lower, upper, velocity),
                            max_step_rad=float(profile.limits.get('max_step_rad', 0.8)))
        robot = SafeRobot(robot, guard,
                          raise_on_limit=bool(profile.limits.get('raise_on_limit', True)))
    return robot


def build_gripper(kind: str = 'mock', spec: GripperSpec | None = None,
                  options: dict | None = None, robot=None) -> GripperBackend:
    options = dict(options or {})
    if robot is not None:
        options['robot'] = robot
    if spec is None:
        provided_robot = options.get('robot')
        spec = getattr(getattr(provided_robot, 'spec', None), 'gripper', None)
    spec = spec or GripperSpec()
    if kind == 'mock':
        return MockGripper(name=spec.name, max_width=spec.max_width,
                           min_width=spec.min_width, **options)
    if kind in ('robotiq', 'robotiq_2f85'):
        if 'max_width_m' not in options:
            options['max_width_m'] = spec.max_width
        config = RobotiqConfig(**{key: value for key, value in options.items()
                                  if key in RobotiqConfig.__dataclass_fields__})
        return RobotiqGripper(config)
    if kind in ('agx', 'agx_gripper', 'piper', 'piper_gripper'):
        robot = options.pop('robot', None)
        if robot is None:
            raise ConfigurationError('AgxGripper 需要传入 robot=AgxPiperRobot')
        return AgxGripper(robot, name=spec.name, max_width=spec.max_width,
                          min_width=spec.min_width,
                          force=float(options.pop('force', 1.0)),
                          close_time_s=float(options.pop('close_time_s', 0.0)),
                          open_time_s=float(options.pop('open_time_s', 0.0)),
                          ramp_hz=float(options.pop('ramp_hz', 50.0)))
    raise ConfigurationError(f'unknown gripper kind {kind!r}')


def build_camera(kind: str = 'mock', name: str = 'camera', width: int = 640,
                 height: int = 480, fps: int = 30, depth: bool = False,
                 **options) -> Camera:
    if kind == 'mock':
        return MockCamera(name=name, width=width, height=height, fps=fps, depth=depth,
                          **options)
    if kind == 'opencv':
        index = int(options.pop('index', 0))
        return OpenCVCamera(name=name, index=index, width=width, height=height,
                            fps=fps, depth=depth)
    if kind == 'realsense':
        serial = str(options.pop('serial', ''))
        return RealSenseCamera(name=name, serial=serial, width=width, height=height,
                               fps=fps, depth=depth)
    raise ConfigurationError(f'unknown camera kind {kind!r}')


def build_camera_hub(entries, stale_s: float = 0.3) -> CameraHub:
    cameras = []
    for entry in entries:
        data = dict(entry)
        kind = data.pop('kind', 'mock')
        name = data.pop('name', 'camera')
        width = int(data.pop('width', 640))
        height = int(data.pop('height', 480))
        fps = int(data.pop('fps', 30))
        depth = bool(data.pop('depth', False))
        cameras.append(build_camera(kind, name=name, width=width, height=height,
                                    fps=fps, depth=depth, **data))
    return CameraHub(cameras, stale_s=stale_s)


def workspace_from_limits(limits: dict) -> WorkspaceBox:
    return WorkspaceBox(limits.get('lower', [-0.75, -0.75, 0.0]),
                        limits.get('upper', [0.85, 0.75, 1.0]))
