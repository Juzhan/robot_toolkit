"""A mock station whose joints are its TCP pose, so IK is exact and offline."""

import numpy as np

from robot_toolkit.agent import AgentOptions, Station
from robot_toolkit.calibrate.store import CameraCalibration
from robot_toolkit.camera import CameraHub, MockCamera
from robot_toolkit.core.conventions import RobotSpec
from robot_toolkit.gripper import MockGripper
from robot_toolkit.robot import MockRobot
from robot_toolkit.runtime.robot_platform import RobotPlatform

HOME = [0.3, 0.0, 0.3, 0.0, 0.0, 0.0]


class PoseIK:
    def solve(self, pose, seed=None):
        return list(pose)


def make_station(execute=True, **changes) -> Station:
    robot = MockRobot(RobotSpec.ur5e(), home=HOME, forward_kinematics=lambda q: np.asarray(q, float))
    robot.set_inverse_kinematics(PoseIK())
    gripper = MockGripper(max_width=0.08, speed=10.0)
    cameras = CameraHub([MockCamera('scene', 32, 24, depth=True), MockCamera('wrist', 32, 24)])
    world_base = np.eye(4)
    world_base[:3, 3] = [1.0, 2.0, 0.0]
    options = AgentOptions(T_world_base=world_base.tolist(), stationary_window_s=0.02,
                           position_tolerance_m=0.002, **changes)
    scene = np.eye(4)
    scene[:3, 3] = [0.5, 0.0, 1.0]
    calibrations = {'scene': CameraCalibration('scene', intrinsics=np.diag([100.0, 100.0, 1.0]),
                                               base_camera=scene)}
    station = Station(RobotPlatform(robot, gripper=gripper, cameras=cameras), options,
                      calibrations=calibrations, physical=False)
    return station.open(execute=execute)
