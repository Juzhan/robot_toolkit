"""Hardware-free tests of independent reprojection verification."""
import copy
import unittest
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from robot_toolkit.calibrate.quality import project, solve, verify
from robot_toolkit.core.mathutils import pose_to_matrix
from robot_toolkit.errors import CalibrationError
from robot_toolkit.calibrate.protocol import capture_station, finish
from robot_toolkit.calibrate.session import BoardObservation
from robot_toolkit.camera.base import CameraFrame


def dataset(mode):
    rng = np.random.default_rng(15)
    state = dict(mode=mode, camera='test', serial='123', board={'test': True},
                 image_size=[640, 480], K=[[600., 0, 320], [0, 600., 240], [0, 0, 1]],
                 dist=[0.]*5, extrinsic=np.eye(4),
                 target=pose_to_matrix([0, 0, .7, 0, 0, 0]))
    grid = np.zeros((30, 3))
    grid[:, :2] = np.mgrid[0:6, 0:5].T.reshape(-1, 2)*.026
    views = []
    for i in range(35):
        pose = np.r_[rng.uniform(-.04, .04, 3), rng.uniform(-.25, .25, 3)]
        view = dict(index=i, base_tcp=pose.tolist(), object_points=grid.tolist())
        view['image_points'] = project(state, view).tolist()
        views.append(view)
    doc = dict(schema='robot_toolkit.calibration.views.v1', camera='test', serial='123',
               mode=mode, board=state['board'], image_size=state['image_size'],
               intrinsics=dict(K=state['K'], dist=state['dist']), views=views[:27])
    return state, doc, dict(doc, views=views[27:])


class QualityTests(unittest.TestCase):
    def test_multiframe_is_one_station(self):
        objects = np.c_[np.arange(6)*.026, np.zeros((6, 2))]
        points = np.c_[np.arange(6)+10., np.full(6, 10.)]
        camera = SimpleNamespace(snapshot=lambda: CameraFrame(
            t=time.perf_counter(), rgb=np.zeros((24, 24, 3), np.uint8)))
        session = SimpleNamespace(camera=camera,
            robot=SimpleNamespace(state=lambda: SimpleNamespace(tcp=np.zeros(6))),
            detector=SimpleNamespace(min_corners=6,
                detect=lambda image: BoardObservation(objects, points)))
        result = capture_station(session, frames=6)
        self.assertEqual(result['frames_used'], 6)
        self.assertEqual(len(result['image_points']), 6)
        stale = camera.snapshot()
        camera.snapshot = lambda: stale
        self.assertIsNone(capture_station(session, frames=6, timeout=.05))

    def test_partial_capture_is_saved_without_solution(self):
        camera = SimpleNamespace(name='test', serial='123', width=640, height=480,
                                 intrinsics=np.eye(3), distortion=np.zeros(5))
        cfg = SimpleNamespace(eye='eye_in_hand', board='charuco', squares=[7, 10],
            square_size=.026, marker_size=.0195, aruco_dict='DICT_5X5_100',
            action='solve', min_samples=27)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'calibration.json'
            self.assertEqual(finish(SimpleNamespace(camera=camera), cfg, path, []), 1)
            self.assertTrue((path.parent / 'views.json').exists())
            self.assertFalse(path.exists())

    def test_both_mountings_fit_and_validate(self):
        for mode in ('eye_in_hand', 'eye_to_hand'):
            truth, train, hold = dataset(mode)
            state, report = solve(train, source='factory')
            np.testing.assert_allclose(state['extrinsic'], truth['extrinsic'], atol=1e-5)
            self.assertFalse(report['independent_verified'])
            self.assertEqual(verify(state, hold)['status'], 'pass')
            state['extrinsic'][0, 3] += .02
            self.assertEqual(verify(state, hold)['status'], 'fail')

    def test_train_data_cannot_verify_itself(self):
        _, train, _ = dataset('eye_in_hand')
        state, _ = solve(train, source='factory')
        with self.assertRaises(CalibrationError):
            verify(state, train)

    def test_missing_or_wrong_data_is_not_pass(self):
        state, _, hold = dataset('eye_in_hand')
        small = dict(hold, views=hold['views'][:2])
        self.assertEqual(verify(state, small)['status'], 'insufficient')
        wrong = dict(hold, serial='another camera')
        with self.assertRaises(CalibrationError):
            verify(state, wrong)
        repeated = dict(hold, views=[hold['views'][0]]*8)
        with self.assertRaises(CalibrationError):
            verify(state, repeated)
        invalid = copy.deepcopy(hold)
        invalid['views'][0]['image_points'][0][0] = float('nan')
        with self.assertRaises(CalibrationError):
            verify(state, invalid)


if __name__ == '__main__':
    unittest.main()
