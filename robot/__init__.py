"""Concrete robot backends.

Importing this package is free: hardware SDKs (``ur_rtde``, ``mujoco``) are
imported lazily when the corresponding backend is constructed.
"""

from .agx_piper import AgxPiperConfig, AgxPiperRobot
from .base import Capabilities, RobotBackend, RobotState
from .mock import MockRobot
from .safety import SafeRobot
from .ur_rtde import UrRtdeConfig, UrRtdeRobot

__all__ = [
    'Capabilities',
    'RobotBackend',
    'RobotState',
    'AgxPiperConfig',
    'AgxPiperRobot',
    'MockRobot',
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
