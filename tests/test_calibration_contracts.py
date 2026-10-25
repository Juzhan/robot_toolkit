"""Offline contracts: no controller, CAN interface or camera is opened."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from robot_toolkit.calibrate.identity import calibration_identity
from robot_toolkit.core.safety import JointLimits, SafetyGuard
from robot_toolkit.core.safety_envelope import SafetyEnvelope
from robot_toolkit.errors import NotSupported, Aborted
from robot_toolkit.robot.agx_piper import AgxPiperRobot, AgxPiperConfig
from robot_toolkit.robot.mock import MockRobot
from robot_toolkit.robot.safety import SafeRobot


class ContractTests(unittest.TestCase):
    def test_identity_json_and_objects_match(self):
        camera = dict(name='wrist', serial='123', width=640, height=480)
        cfg = dict(eye='eye_in_hand', board='charuco', squares=[7, 10],
                   square_size=.026, marker_size=.0195, aruco_dict='DICT_5X5_100')
        self.assertEqual(calibration_identity(camera, cfg),
                         calibration_identity(SimpleNamespace(**camera),
                                              SimpleNamespace(**cfg)))

    def test_read_only_is_explicit_and_wrapper_delegates(self):
        inner = MockRobot()
        inner.disconnect()
        safe = SafeRobot(inner, SafetyGuard(JointLimits.symmetric(inner.dof)))
        with self.assertRaises(NotSupported):
            safe.connect_read_only()
        self.assertFalse(inner.connected)
        with self.assertRaises(Aborted):
            safe.prepare_motion()
        with patch.object(inner, 'connect_read_only', side_effect=inner.connect) as connect:
            self.assertIs(safe.connect_read_only(), safe)
            connect.assert_called_once()
        with patch.object(inner, 'prepare_motion', return_value=False) as prepare:
            self.assertFalse(safe.prepare_motion())
            prepare.assert_called_once()
        safe.disconnect()

    def test_piper_read_only_options_restored_even_on_failure(self):
        robot = AgxPiperRobot(AgxPiperConfig(mock=True))
        original = robot.config
        def inspect_options():
            self.assertFalse(robot.config.enable_on_connect)
            self.assertFalse(robot.config.configure_on_connect)
            raise RuntimeError('simulated connection failure')
        with patch.object(robot, 'connect', side_effect=inspect_options):
            with self.assertRaises(RuntimeError):
                robot.connect_read_only()
        self.assertIs(robot.config, original)

    def test_envelope_units_and_invalid_bounds(self):
        tcp = np.array([.1, .2, .3, 1., 2., 3.])
        envelope = SafetyEnvelope.from_samples([([0., 0.], tcp)], tcp_margin_m=.02)
        np.testing.assert_allclose(envelope.tcp_lower[:3], tcp[:3]-.02)
        np.testing.assert_allclose(envelope.tcp_lower[3:], tcp[3:])
        for bad in (float('nan'), float('inf'), -.1):
            with self.assertRaises(ValueError):
                SafetyEnvelope.from_samples([([0.], tcp)], tcp_margin_m=bad)
        with self.assertRaises(ValueError):
            SafetyEnvelope([float('nan')], [1.], tcp, tcp)


if __name__ == '__main__':
    unittest.main()
