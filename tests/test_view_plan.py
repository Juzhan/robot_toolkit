import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np

from robot_toolkit.calibrate.view_plan import generate, load_plan, predict
from robot_toolkit.calibrate.quality import write_json
from robot_toolkit.core.mathutils import pose_to_matrix
from robot_toolkit.core.safety import JointLimits
from robot_toolkit.core.workspace_guard import WorkspaceGuard
from robot_toolkit.errors import CalibrationError


class PlanTests(unittest.TestCase):
    def setUp(self):
        self.state = dict(mode='eye_in_hand', camera='test', serial='123',
            board=dict(pattern='charuco', squares=[7, 10], square_size=.026,
                       marker_size=.0195, aruco_dict='DICT_5X5_100'),
            image_size=[640, 480], K=[[600., 0, 320], [0, 600., 240], [0, 0, 1]],
            dist=[0.]*5, extrinsic=np.eye(4).tolist(),
            target=pose_to_matrix([-.09, -.13, 1., 0, 0, 0]).tolist())
        self.fk = lambda q: np.r_[np.asarray(q)[:3]*.1, np.asarray(q)[3:]]
        self.workspace = WorkspaceGuard([-1]*3, [1]*3, self.fk)
        self.limits = JointLimits.symmetric(6)

    def make_plan(self):
        return generate(self.state, np.zeros(6), self.fk, self.limits, self.workspace,
                        count=12, verify_count=8, candidate_count=256)

    def test_reproducible_diverse_visible_plan_and_roles(self):
        first, second = self.make_plan(), self.make_plan()
        self.assertEqual(first['plan_id'], second['plan_id'])
        self.assertEqual([v['role'] for v in first['views']], ['train']*12+['verify']*8)
        qs = [tuple(v['q_rad']) for v in first['views']]
        self.assertEqual(len(set(qs)), 20)
        previous = first['anchor_q_rad']
        for row in first['views']:
            self.workspace.check_path(previous, row['q_rad'], clearance_m=.01)
            corners = np.asarray(row['corners_px'])
            self.assertTrue(np.all(corners >= 24))
            self.assertTrue(np.all(corners <= [616, 456]))
            previous = row['q_rad']

    def test_modified_plan_and_workspace_rejected(self):
        plan = self.make_plan()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'plan.json'
            write_json(path, plan)
            load_plan(path, self.state, self.workspace, 12, 8)
            other = WorkspaceGuard([-1]*3, [.9]*3, self.fk)
            with self.assertRaises(CalibrationError):
                load_plan(path, self.state, other, 12, 8)
            plan['views'][0]['q_rad'][0] += .1
            write_json(path, plan)
            with self.assertRaises(CalibrationError):
                load_plan(path, self.state, self.workspace, 12, 8)

    def test_out_of_frame_and_behind_camera_rejected(self):
        for target in ([2, 0, 1, 0, 0, 0], [0, 0, -1, 0, 0, 0]):
            state = copy.deepcopy(self.state)
            state['target'] = pose_to_matrix(target).tolist()
            with self.assertRaises(CalibrationError):
                predict(state, np.zeros(6))


if __name__ == '__main__':
    unittest.main()
