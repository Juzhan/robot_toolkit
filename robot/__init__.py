"""Concrete robot backends.

Importing this package is free: hardware SDKs (``ur_rtde``, ``mujoco``) are
imported lazily when the corresponding backend is constructed.
"""

from .agx_piper import (PIPER_CAN_LABELS, AgxPiperConfig, AgxPiperRobot,
                        scan_piper_can, set_piper_linkage)
from .base import Capabilities, RobotBackend, RobotState
from .ik import PyrokiIK
from .mock import MockRobot
from .safety import SafeRobot
from .ur_rtde import UrRtdeConfig, UrRtdeRobot

__all__ = [
    'Capabilities',
    'RobotBackend',
    'RobotState',
    'AgxPiperConfig',
    'AgxPiperRobot',
    'PIPER_CAN_LABELS',
    'scan_piper_can',
    'set_piper_linkage',
    'MockRobot',
    'PyrokiIK',
    'SafeRobot',
    'UrRtdeConfig',
    'UrRtdeRobot',
    'MujocoRobot',
]


def __getattr__(name):
    if name == 'MujocoRobot':
        from .mujoco import MujocoRobot
        return MujocoRobot
    raise AttributeError(name)
