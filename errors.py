"""Exception hierarchy shared by every robot backend.

Keeping these in one place means a control loop can catch ``RobotUnavailable``
regardless of whether it is talking to a simulator or a real controller.
"""


class RobotToolkitError(Exception):
    """Base class for all toolkit errors."""


class ConfigurationError(RobotToolkitError):
    """A profile or JSON config is missing or inconsistent."""


class ConnectionFailed(RobotToolkitError):
    """The backend could not establish a link (network, driver, permissions)."""


class RobotUnavailable(ConnectionFailed):
    """The link existed but the robot is not ready for motion right now."""


class NotSupported(RobotToolkitError):
    """The backend does not implement the requested capability."""


class SafetyViolation(RobotToolkitError):
    """A command was rejected because it would exceed a configured limit."""


class Aborted(RobotToolkitError):
    """An operator or watchdog aborted the operation."""

    def __init__(self, reason: str = 'aborted'):
        super().__init__(reason)
        self.reason = reason


class ProtectiveStop(Aborted):
    """The controller reported a protective stop while a motion was running."""


class DatasetError(RobotToolkitError):
    """An episode is incomplete or does not match the dataset contract."""


class CalibrationError(RobotToolkitError):
    """A calibration input or solve is invalid."""
