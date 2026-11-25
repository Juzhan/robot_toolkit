import numpy as np
import pytest

from robot_toolkit.agent import Rejected
from robot_toolkit.core.mathutils import matrix_to_pose, pose_to_matrix

from .station_support import make_station as _station


def _shifted(T, dx):
    target = T.copy()
    target[0, 3] += dx
    return target


def test_state_is_the_control_point_in_world():
    station = _station()
    state = station.read()
    np.testing.assert_allclose(state.T_world_control[:3, 3], [1.3, 2.0, 0.3])
    assert state.gripper == pytest.approx(1.0) and state.gripper_measured
    station.close()


def test_move_pose_completes_and_confirms_stillness():
    station = _station()
    target = _shifted(station.read().T_world_control, 0.03)
    receipt = station.move_pose(target, gripper=0.0)
    assert receipt.accepted and receipt.outcome == 'completed'
    assert receipt.stop_confirmed is True and not receipt.requires_operator
    np.testing.assert_allclose(receipt.state.T_world_control[:3, 3], [1.33, 2.0, 0.3], atol=1e-3)
    assert station.gripper.state().target == pytest.approx(0.0)
    station.close()


def test_large_step_and_read_only_are_refused_before_sending():
    station = _station()
    before = station.read().q
    with pytest.raises(Rejected):
        station.move_pose(_shifted(station.read().T_world_control, 0.2))
    np.testing.assert_allclose(station.read().q, before)
    station.close()
    observer = _station(execute=False)
    with pytest.raises(Rejected):
        observer.hold(gripper=0.5)
    observer.close()


def test_capture_applies_world_base_and_keeps_uncalibrated_views_unclaimed():
    station = _station()
    frames = station.capture()
    np.testing.assert_allclose(frames['scene'].T_world_optical[:3, 3], [1.5, 2.0, 1.0])
    assert frames['scene'].depth_m is not None and frames['scene'].calibration_source == 'eye_to_hand'
    assert frames['wrist'].K is None and frames['wrist'].T_world_optical is None
    station.close()


def test_unknown_outcome_blocks_motion_until_operator_clears(monkeypatch):
    station = _station()
    real = station.platform.move_j

    def timed_out(joints, **kwargs):
        record = real(station.read().q, **kwargs)
        record.details.update(ok=False, reason='timeout', command_ok=True)
        return record

    monkeypatch.setattr(station.platform, 'move_j', timed_out)
    receipt = station.move_pose(_shifted(station.read().T_world_control, 0.01))
    assert receipt.outcome == 'unknown' and receipt.requires_operator and not receipt.accepted
    monkeypatch.setattr(station.platform, 'move_j', real)
    refused = station.move_pose(_shifted(station.read().T_world_control, 0.01))
    assert refused.outcome == 'none' and refused.requires_operator and refused.reason == 'operator_recovery_required'
    station.clear_block()
    assert station.move_pose(_shifted(station.read().T_world_control, 0.01)).accepted
    station.close()


def test_control_offset_maps_targets_back_to_the_backend_tcp():
    offset = np.eye(4)
    offset[2, 3] = 0.1
    station = _station(T_tcp_control=offset.tolist())
    state = station.read()
    np.testing.assert_allclose(state.T_world_control[:3, 3], [1.3, 2.0, 0.4])
    receipt = station.move_pose(_shifted(state.T_world_control, 0.02))
    assert receipt.accepted
    np.testing.assert_allclose(matrix_to_pose(pose_to_matrix(station.robot.state().tcp))[:3], [0.32, 0.0, 0.3],
                               atol=1e-3)
    station.close()
