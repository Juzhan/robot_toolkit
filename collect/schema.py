"""Dataset contract: names, shapes, units and LeRobot feature description.

A schema is the single source of truth for what a recorded episode contains.
Backends produce state; this module decides how it is named and exported, so a
UR5e episode and a Piper episode differ only in metadata, never in structure.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core.conventions import (CAMERA_FRAME_CONVENTION, DEPTH_ENCODING,
                                EXTRINSIC_CONVENTION, RobotSpec)
from ..errors import DatasetError


@dataclass(frozen=True)
class CameraSpec:
    name: str = 'camera'
    width: int = 640
    height: int = 480
    channels: int = 3
    depth: bool = False

    def to_dict(self) -> dict:
        return {'name': self.name, 'width': self.width, 'height': self.height,
                'channels': self.channels, 'depth': self.depth}

    @classmethod
    def from_dict(cls, data: dict) -> 'CameraSpec':
        return cls(**{key: data[key] for key in
                      ('name', 'width', 'height', 'channels', 'depth')
                      if key in data})

    @classmethod
    def from_camera(cls, camera) -> 'CameraSpec':
        """Describe an already constructed ``Camera`` without opening it again."""
        return cls(name=camera.name, width=int(camera.width), height=int(camera.height),
                   depth=bool(getattr(camera, 'depth_enabled', False)))


@dataclass
class DatasetSchema:
    robot: RobotSpec = field(default_factory=RobotSpec.ur5e)
    cameras: tuple[CameraSpec, ...] = (
        CameraSpec('front_left', 640, 360),
        CameraSpec('front_right', 640, 360),
        CameraSpec('wrist_camera', 640, 360),
    )
    fps: int = 10
    image_feature_dtype: str = 'video'
    include_time: bool = False
    time_name: str = 'observation.sim_time'
    robot_type: str = ''
    repo_id: str = 'local/robot_toolkit'
    require_images: bool = True

    def __post_init__(self):
        self.cameras = tuple(self.cameras)
        if not self.robot_type:
            self.robot_type = self.robot.name
        # A schema with no cameras can only record robot state/action.  Do not
        # require image keys that no camera can provide.
        if not self.cameras:
            self.require_images = False

    # ------------------------------------------------------------------ names
    @property
    def state_names(self) -> list[str]:
        return self.robot.state_names

    @property
    def action_names(self) -> list[str]:
        return self.robot.state_names

    @property
    def dim(self) -> int:
        return self.robot.dim

    @property
    def camera_names(self) -> list[str]:
        return [camera.name for camera in self.cameras]

    @classmethod
    def for_robot(cls, robot: RobotSpec, cameras=(), fps: int = 10,
                  **kwargs) -> 'DatasetSchema':
        """Compose a schema from a robot and any subset of cameras.

        ``cameras=()`` is the robot-only dataset (state/action, no images).
        ``require_images`` can be supplied explicitly; otherwise it follows
        whether any camera was declared.  Items may be ``CameraSpec`` objects
        or already constructed ``Camera`` objects."""
        specs = tuple(item if isinstance(item, CameraSpec)
                      else CameraSpec.from_camera(item) for item in cameras)
        kwargs.setdefault('require_images', bool(specs))
        return cls(robot=robot, cameras=specs, fps=int(fps), **kwargs)

    # --------------------------------------------------------------- lerobot
    def features(self) -> dict:
        state = {'dtype': 'float32', 'shape': (self.dim,), 'names': self.state_names}
        features = {
            'observation.state': dict(state),
            'action': dict(state),
        }
        if self.include_time:
            features[self.time_name] = {'dtype': 'float64', 'shape': (1,),
                                        'names': ['seconds']}
        for camera in self.cameras:
            features[f'observation.images.{camera.name}'] = {
                'dtype': self.image_feature_dtype,
                'shape': (camera.height, camera.width, camera.channels),
                'names': ['height', 'width', 'channels'],
            }
        return features

    # ------------------------------------------------------------------ frame
    def observation(self, joints, gripper_width=None) -> np.ndarray:
        return np.asarray(self.robot.pack(joints, gripper_width), dtype=np.float32)

    def frame(self, observation_state, action, images=None, task: str = '',
              time_s: float | None = None, extra: dict | None = None) -> dict:
        state = np.asarray(observation_state, dtype=np.float32).reshape(-1)
        action_array = np.asarray(action, dtype=np.float32).reshape(-1)
        if state.size != self.dim:
            raise DatasetError(f'observation.state has {state.size} values, expected {self.dim}')
        if action_array.size != self.dim:
            raise DatasetError(f'action has {action_array.size} values, expected {self.dim}')
        frame = {
            'observation.state': state,
            'action': action_array,
            'task': str(task),
        }
        if self.include_time and time_s is not None:
            frame[self.time_name] = np.array([float(time_s)], dtype=np.float64)
        for camera in self.cameras:
            if images is None or camera.name not in images:
                continue
            image = images[camera.name]
            if image is None:
                continue
            expected = (camera.height, camera.width, camera.channels)
            if tuple(image.shape) != expected:
                raise DatasetError(f'camera {camera.name}: image shape {image.shape} != {expected}')
            frame[f'observation.images.{camera.name}'] = image
        if extra:
            frame.update(extra)
        self.validate(frame)
        return frame

    def validate(self, frame: dict) -> None:
        for key in ('observation.state', 'action'):
            if key not in frame:
                raise DatasetError(f'missing {key}')
            value = np.asarray(frame[key])
            if value.shape != (self.dim,):
                raise DatasetError(f'{key} has shape {value.shape}, expected {(self.dim,)}')
            if not np.all(np.isfinite(value)):
                raise DatasetError(f'{key} contains non-finite values')
        for camera in self.cameras:
            key = f'observation.images.{camera.name}'
            if key not in frame:
                continue
            value = np.asarray(frame[key])
            if value.dtype != np.uint8:
                raise DatasetError(f'{key} must be uint8, got {value.dtype}')

    def contract(self) -> dict:
        """Machine-readable description written next to every dataset."""
        return {
            'toolkit': 'robot_toolkit',
            'fps': self.fps,
            'robot': self.robot.name,
            'robot_type': self.robot_type,
            'state_action': {
                'names': self.state_names,
                'units': [self.robot.joint_units] * self.robot.dof +
                         ([] if self.robot.gripper is None else [self.robot.gripper.unit]),
                'action': 'absolute joint targets plus absolute gripper opening',
                'dim': self.dim,
            },
            'cameras': [camera.to_dict() for camera in self.cameras],
            'depth': DEPTH_ENCODING,
            'extrinsics': EXTRINSIC_CONVENTION,
            'camera_frame': CAMERA_FRAME_CONVENTION,
            'alignment': ('observation at t, action computed from the same control '
                          'sample and applied after that observation'),
        }

    # ------------------------------------------------------------------ io
    def to_dict(self) -> dict:
        return {'robot': self.robot.to_dict(),
                'cameras': [camera.to_dict() for camera in self.cameras],
                'fps': self.fps, 'image_feature_dtype': self.image_feature_dtype,
                'include_time': self.include_time, 'time_name': self.time_name,
                'robot_type': self.robot_type, 'repo_id': self.repo_id,
                'require_images': self.require_images}

    @classmethod
    def from_dict(cls, data: dict) -> 'DatasetSchema':
        # An explicit ``cameras: []`` means robot-only data.  Only a completely
        # missing key falls back to the default three-camera layout.
        cameras = (tuple(CameraSpec.from_dict(item) for item in data['cameras'])
                   if 'cameras' in data else DatasetSchema().cameras)
        return cls(robot=RobotSpec.from_dict(data.get('robot', {})),
                   cameras=cameras,
                   fps=int(data.get('fps', 10)),
                   image_feature_dtype=data.get('image_feature_dtype', 'video'),
                   include_time=bool(data.get('include_time', False)),
                   time_name=data.get('time_name', 'observation.sim_time'),
                   robot_type=data.get('robot_type', ''),
                   repo_id=data.get('repo_id', 'local/robot_toolkit'),
                   require_images=bool(data.get('require_images', bool(cameras))))
