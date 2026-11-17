import unittest
import numpy as np

from robot_toolkit.core.workspace_guard import WorkspaceGuard
from robot_toolkit.core.safety import SafetyGuard, JointLimits
from robot_toolkit.errors import SafetyViolation
from robot_toolkit.robot.safety import SafeRobot
from robot_toolkit.robot.base import RobotState
from robot_toolkit.robot.mock import MockRobot


class WorkspaceTests(unittest.TestCase):
    def test_y_walls_and_table_only(self):
        guard = WorkspaceGuard([-1]*3, [1]*3, lambda q: q,
                               lower_enabled=[False, True, True],
                               upper_enabled=[False, True, False])
        guard.check_path([10, 0, 2], [20, .5, 3], clearance_m=.1)
        for target in ([0, 1.1, 0], [0, -1.1, 0], [0, 0, -1.1]):
            with self.assertRaises(SafetyViolation):
                guard.check_path([0, 0, 0], target)

    def test_safe_endpoints_unsafe_middle(self):
        guard = WorkspaceGuard([-1, -1, -1], [1, 1, 1],
                               lambda q: [0, 2*np.sin(np.pi*q[0]), 0])
        with self.assertRaises(SafetyViolation):
            guard.check_path([0], [1])

    def test_margin_and_invalid_data(self):
        guard = WorkspaceGuard([-1]*3, [1]*3, lambda q: q, margin_m=.1)
        guard.check_path([0]*3, [.8]*3)
        with self.assertRaises(SafetyViolation):
            guard.check_path([0]*3, [.95]*3)
        with self.assertRaises(SafetyViolation):
            guard.check_path([0]*3, [float('nan')]*3)

    def test_model_mismatch(self):
        guard = WorkspaceGuard([-1]*3, [1]*3, lambda q: [0, 0, 0])
        with self.assertRaises(SafetyViolation):
            guard.check_feedback(RobotState(q=[0], tcp=[.02, 0, 0, 0, 0, 0]))

    def test_wrapper_rejects_before_command(self):
        inner = MockRobot()
        inner.connect()
        guard = WorkspaceGuard([-1]*3, [1]*3, lambda q: [0, 2*np.sin(np.pi*q[0]), 0])
        state = RobotState(q=np.zeros(6), tcp=np.zeros(6))
        inner.state = lambda: state
        calls = []
        inner.servo_j = lambda *args: calls.append(args)
        safe = SafeRobot(inner, SafetyGuard(JointLimits.symmetric(6)), workspace=guard)
        with self.assertRaises(SafetyViolation):
            safe.move_j([1, 0, 0, 0, 0, 0])
        self.assertEqual(calls, [])


if __name__ == '__main__':
    unittest.main()
