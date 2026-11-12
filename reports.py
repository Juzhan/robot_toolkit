"""Agent-facing views of recorded platform operations."""

from __future__ import annotations

from .robot_platform import RobotPlatform


class PlatformReports:
    """Convert platform execution records to reports without extra device reads."""

    def __init__(self, platform: RobotPlatform):
        if not isinstance(platform, RobotPlatform):
            raise TypeError('reports require a RobotPlatform, not a robot backend')
        self.platform = platform

    def __getattr__(self, name):
        return getattr(self.platform, name)

    def move_j(self, joints, *, joint_tolerance: float = 0.02, **kwargs) -> dict:
        return self.platform.move_j(joints, joint_tolerance=joint_tolerance, **kwargs).to_report()

    def move_p(self, pose, *, position_tolerance: float = 0.005,
               rotation_tolerance: float = 0.05, **kwargs) -> dict:
        return self.platform.move_p(
            pose, position_tolerance=position_tolerance,
            rotation_tolerance=rotation_tolerance, **kwargs).to_report()


def with_reports(platform: RobotPlatform | PlatformReports) -> PlatformReports:
    """Borrow a platform; repeated wrapping is idempotent and performs no I/O."""
    return platform if isinstance(platform, PlatformReports) else PlatformReports(platform)