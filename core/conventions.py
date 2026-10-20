"""Dataset and control contracts shared by simulation and real robots.

The 7-dimensional contract used by the existing projects is kept intact:

    observation.state = [j1..j6 (rad), gripper opening (m)]
    action            = [j1..j6 (rad), gripper opening (m)]

Actions are absolute joint positions, not deltas.  Every backend, simulator or
real, must expose the same ordering, which is what allows one recorded dataset
to be replayed on another arm (after retargeting) and one teleoperation loop to
drive either backend.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: Depth maps are unsigned 16-bit millimetres, 0 meaning "invalid".
DEPTH_ENCODING = 'uint16 millimeters, 0 invalid'

#: Camera extrinsics are stored as ``T_world_camera`` in the optical frame.
EXTRINSIC_CONVENTION = 'T_world_camera, optical: x right, y down, z forward'

CAMERA_FRAME_CONVENTION = EXTRINSIC_CONVENTION


@dataclass(frozen=True)
class GripperSpec:
    """Normalized description of a parallel-jaw gripper."""

    name: str = 'gripper'
    unit: str = 'm'
    max_width: float = 0.085
    min_width: float = 0.0

    @property
    def action_name(self) -> str:
        return f'{self.name}.width_{self.unit}'

    def clamp(self, width: float) -> float:
        return float(min(max(float(width), self.min_width), self.max_width))

    def to_dict(self) -> dict:
        return {'name': self.name, 'unit': self.unit,
                'max_width': self.max_width, 'min_width': self.min_width}

    @classmethod
    def from_dict(cls, data: dict) -> 'GripperSpec':
        return cls(**{key: data[key] for key in ('name', 'unit', 'max_width', 'min_width')
                      if key in data})


@dataclass(frozen=True)
class RobotSpec:
    """Joint ordering plus the optional gripper axis of one robot family.

    ``name`` is metadata only; nothing in the toolkit branches on it.  Backends
    receive the spec and use ``joint_names`` to address their own hardware.
    """

    name: str = 'generic'
    joint_names: tuple[str, ...] = ('j1', 'j2', 'j3', 'j4', 'j5', 'j6')
    gripper: GripperSpec | None = field(default_factory=GripperSpec)
    joint_units: str = 'rad'

    @property
    def dof(self) -> int:
        return len(self.joint_names)

    @property
    def state_names(self) -> list[str]:
        names = [f'{joint}.{self.joint_units}' for joint in self.joint_names]
        if self.gripper is not None:
            names.append(self.gripper.action_name)
        return names

    @property
    def dim(self) -> int:
        return len(self.state_names)

    def pack(self, joints, gripper_width=None) -> list[float]:
        """Assemble one contract vector from joint values and a gripper width."""
        values = [float(value) for value in joints]
        if len(values) != self.dof:
            raise ValueError(f'{self.name}: expected {self.dof} joints, got {len(values)}')
        if self.gripper is not None:
            width = self.gripper.max_width if gripper_width is None else gripper_width
            values.append(self.gripper.clamp(width))
        return values

    def to_dict(self) -> dict:
        return {'name': self.name, 'joint_names': list(self.joint_names),
                'joint_units': self.joint_units,
                'gripper': None if self.gripper is None else self.gripper.to_dict()}

    @classmethod
    def from_dict(cls, data: dict) -> 'RobotSpec':
        gripper = data.get('gripper')
        return cls(name=data.get('name', 'generic'),
                   joint_names=tuple(data.get('joint_names', cls.joint_names)),
                   gripper=None if gripper is None else GripperSpec.from_dict(gripper),
                   joint_units=data.get('joint_units', 'rad'))

    @classmethod
    def generic(cls, dof: int = 6, name: str = 'generic',
                with_gripper: bool = True) -> 'RobotSpec':
        return cls(name=name, joint_names=tuple(f'joint_{index + 1}' for index in range(dof)),
                   gripper=GripperSpec() if with_gripper else None)

    @classmethod
    def ur5e(cls) -> 'RobotSpec':
        return cls(name='ur5e',
                   joint_names=('shoulder_pan', 'shoulder_lift', 'elbow',
                                'wrist_1', 'wrist_2', 'wrist_3'),
                   gripper=GripperSpec(name='robotiq_2f85', max_width=0.085))

    @classmethod
    def piper_x(cls) -> 'RobotSpec':
        return cls(name='piper_x',
                   joint_names=('joint_1', 'joint_2', 'joint_3',
                                'joint_4', 'joint_5', 'joint_6'),
                   gripper=GripperSpec(name='piper_gripper', max_width=0.08))


def state_names(spec: RobotSpec) -> list[str]:
    """Convenience wrapper so callers do not need the property spelling."""
    return spec.state_names
