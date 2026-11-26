"""Robot-agnostic toolkit for teleoperation, teaching, collection and calibration.

Design in one sentence: *one contract, many backends*.  Every layer above the
hardware talks to ``RobotBackend`` / ``GripperBackend`` / ``Camera`` and to the
7-dimensional state/action convention, so a MuJoCo scene and a real UR5e are
interchangeable.

Importing the package is cheap: hardware SDKs (``ur_rtde``, ``mujoco``,
``pyrealsense2``, ``lerobot``) are only imported when their backend is used.
"""

from .core.conventions import (CAMERA_FRAME_CONVENTION, DEPTH_ENCODING,
                               EXTRINSIC_CONVENTION, GripperSpec, RobotSpec)
from .errors import (Aborted, CalibrationError, ConfigurationError, ConnectionFailed,
                     DatasetError, NotSupported, ProtectiveStop, RobotToolkitError,
                     RobotUnavailable, SafetyViolation)
from .robot.base import Capabilities, RobotBackend, RobotState
from .runtime.robot_platform import ExecutionRecord, Observation, RobotPlatform
from .runtime.reports import PlatformReports, with_reports
from .perception.scene import (LayoutCheck, ObjectSpec, ObjectState, PlacementCondition,
                    PlanarColorEstimator, ResetLayoutVerifier, SceneSpec, SceneState)

__version__ = '0.1.0'

__all__ = [
    '__version__',
    'CAMERA_FRAME_CONVENTION',
    'DEPTH_ENCODING',
    'EXTRINSIC_CONVENTION',
    'GripperSpec',
    'RobotSpec',
    'Aborted',
    'CalibrationError',
    'ConfigurationError',
    'ConnectionFailed',
    'DatasetError',
    'NotSupported',
    'ProtectiveStop',
    'RobotToolkitError',
    'RobotUnavailable',
    'SafetyViolation',
    'Capabilities',
    'RobotBackend',
    'RobotState',
    'ExecutionRecord',
    'Observation',
    'RobotPlatform',
    'PlatformReports',
    'with_reports',
    'ObjectSpec',
    'ObjectState',
    'SceneSpec',
    'SceneState',
    'PlanarColorEstimator',
    'PlacementCondition',
    'ResetLayoutVerifier',
    'LayoutCheck',
]
