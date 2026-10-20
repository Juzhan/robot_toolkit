"""Calibration tests: Zhang intrinsics, hand-eye AX=XB and persistence."""

from __future__ import annotations

import json

import numpy as np
import pytest

from robot_toolkit.calibrate import (CalibrationSession, CalibrationStore,
                                     estimate_intrinsics, solve_hand_eye)
from robot_toolkit.calibrate.handeye import solve_ax_xb
from robot_toolkit.calibrate.intrinsics import (project_points,
                                                undistort_points)
from robot_toolkit.calibrate.store import compose_camera_chain
from robot_toolkit.camera import MockCamera
from robot_toolkit.core.mathutils import (invert_transform, matrix_to_pose,
                                          pose_to_matrix, rotvec_to_matrix)
from robot_toolkit.errors import CalibrationError


def _random_transform(rng, translation=0.3):
    matrix = np.eye(4)
    matrix[:3, :3] = rotvec_to_matrix(rng.normal(0, 0.7, 3))
    matrix[:3, 3] = rng.uniform(-translation, translation, 3)
    return matrix


def _hand_eye_case(mode, samples=14, seed=5):
    rng = np.random.default_rng(seed)
    truth = _random_transform(rng, 0.05)
    truth[:3, 3] = [0.04, -0.02, 0.08]
    base, camera = [], []
    if mode == 'eye_in_hand':
        target = _random_transform(rng, 0.03)
        target[:3, 3] = [0.0, 0.0, 0.5]
        for _ in range(samples):
            robot = _random_transform(rng, 0.12)
            robot[:3, 3] += [0.3, 0.0, 0.3]
            base.append(robot)
            camera.append(invert_transform(robot @ truth) @ target)
    else:
        target_in_tcp = _random_transform(rng, 0.03)
        for _ in range(samples):
            robot = _random_transform(rng, 0.12)
            robot[:3, 3] += [0.3, 0.0, 0.3]
            base.append(robot)
            camera.append(invert_transform(truth) @ robot @ target_in_tcp)
    return base, camera, truth


@pytest.mark.parametrize('mode', ['eye_in_hand', 'eye_to_hand'])
def test_hand_eye_recovers_exact_transform(mode):
    base, camera, truth = _hand_eye_case(mode)
    result = solve_hand_eye(mode, base, camera)
    assert result.mode == mode
    assert result.rotation_rms < 1e-9
    assert result.translation_rms < 1e-9
    assert np.allclose(result.transform, truth, atol=1e-7)


@pytest.mark.parametrize('mode', ['eye_in_hand', 'eye_to_hand'])
def test_hand_eye_tolerates_noise(mode):
    rng = np.random.default_rng(11)
    base, camera, truth = _hand_eye_case(mode)
    noisy = []
    for matrix in camera:
        item = matrix.copy()
        item[:3, 3] += rng.normal(0, 5e-5, 3)
        item[:3, :3] = rotvec_to_matrix(rng.normal(0, 5e-5, 3)) @ item[:3, :3]
        noisy.append(item)
    result = solve_hand_eye(mode, base, noisy)
    assert np.allclose(result.transform, truth, atol=5e-3)


def test_calibration_session_collects_and_saves(tmp_path):
    base, camera, truth = _hand_eye_case('eye_in_hand', samples=8)
    session = CalibrationSession(mode='eye_in_hand', camera_name='wrist_camera')
    for robot_pose, board_pose in zip(base, camera):
        session.add_sample(matrix_to_pose(robot_pose), board_pose)
    result = session.solve_hand_eye()
    assert np.allclose(result.transform, truth, atol=1e-7)
    summary = session.save(tmp_path / 'calibration.json')
    assert summary['camera'] == 'wrist_camera'
    store = CalibrationStore(tmp_path / 'calibration.json')
    assert np.allclose(store.get('wrist_camera').tcp_camera, truth, atol=1e-7)


def test_hand_eye_rejects_insufficient_motion():
    transform = np.eye(4)
    with pytest.raises(CalibrationError):
        solve_hand_eye('eye_in_hand', [transform, transform], [transform, transform])
    with pytest.raises(CalibrationError):
        solve_hand_eye('unknown', [transform] * 3, [transform] * 3)


def test_solve_ax_xb_matches_constructed_solution():
    rng = np.random.default_rng(2)
    x = _random_transform(rng, 0.05)
    a_list, b_list = [], []
    for _ in range(6):
        a = _random_transform(rng, 0.1)
        b = invert_transform(x) @ a @ x
        a_list.append(a)
        b_list.append(b)
    solved = solve_ax_xb(a_list, b_list)
    assert np.allclose(solved, x, atol=1e-8)


def _intrinsics_views(image_size=(640, 480), views=16, seed=9):
    rng = np.random.default_rng(seed)
    truth = np.array([[610.0, 1.0, 318.0], [0.0, 605.0, 242.0], [0.0, 0.0, 1.0]])
    columns, rows, square = 8, 6, 0.025
    grid = np.zeros((columns * rows, 3))
    grid[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2) * square
    samples = []
    while len(samples) < views:
        rotation = rotvec_to_matrix(rng.normal(0, 0.35, 3))
        translation = np.array([rng.uniform(-0.05, 0.05), rng.uniform(-0.04, 0.04),
                                rng.uniform(0.35, 0.6)])
        image = project_points(grid, truth, None, rotation, translation)
        if (image[:, 0].min() < 5 or image[:, 0].max() > image_size[0] - 5 or
                image[:, 1].min() < 5 or image[:, 1].max() > image_size[1] - 5):
            continue
        samples.append({'object_points': grid, 'image_points': image})
    return samples, truth


def test_zhang_recovers_intrinsics():
    samples, truth = _intrinsics_views()
    result = estimate_intrinsics(samples, image_size=(640, 480), refinement=0)
    assert result.method == 'zhang'
    assert result.K[0, 0] / truth[0, 0] == pytest.approx(1.0, abs=0.01)
    assert result.K[1, 1] / truth[1, 1] == pytest.approx(1.0, abs=0.01)
    assert result.K[0, 2] == pytest.approx(truth[0, 2], abs=2.0)
    assert result.K[1, 2] == pytest.approx(truth[1, 2], abs=2.0)
    assert result.rms < 0.5
    assert len(result.per_view) == len(samples)


def test_zhang_needs_three_views():
    samples, _ = _intrinsics_views(views=3)
    with pytest.raises(CalibrationError):
        estimate_intrinsics(samples[:2])


def test_project_undistort_roundtrip():
    K = np.array([[600.0, 0.0, 320.0], [0.0, 605.0, 240.0], [0.0, 0.0, 1.0]])
    dist = np.array([0.1, -0.05, 0.001, -0.002, 0.0])
    points = np.array([[0.1, 0.05, 1.0], [-0.2, 0.1, 1.5], [0.0, 0.0, 0.8]])
    image = project_points(points, K, dist)
    recovered = undistort_points(image, K, dist)
    normalized = points[:, :2] / points[:, 2:3]
    assert np.allclose(recovered, normalized * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]], atol=1e-6)


def test_calibration_store_roundtrip_and_apply(tmp_path):
    path = tmp_path / 'calibration.json'
    store = CalibrationStore(path)
    K = np.array([[600.0, 0.0, 320.0], [0.0, 610.0, 240.0], [0.0, 0.0, 1.0]])
    transform = pose_to_matrix([0.01, 0.02, 0.03, 0.0, 0.0, 0.1])
    store.set_intrinsics('wrist_camera', K, np.zeros(5), (640, 480))
    store.set_hand_eye('wrist_camera', 'eye_in_hand', transform, residual=1e-5)
    store.save()

    reloaded = CalibrationStore(path)
    entry = reloaded.get('wrist_camera')
    assert np.allclose(entry.intrinsics, K)
    assert np.allclose(entry.tcp_camera, transform)
    base_tcp = [0.3, 0.0, 0.2, 0.0, 0.0, 0.0]
    expected = pose_to_matrix(base_tcp) @ transform
    assert np.allclose(entry.world_transform(base_tcp), expected)

    camera = MockCamera('wrist_camera', 64, 48)
    assert reloaded.apply(camera, base_tcp)
    assert np.allclose(camera.intrinsics, K)
    assert np.allclose(camera.T_world_camera, expected)
    summary = reloaded.summary()
    assert summary[0]['tcp_camera'] and summary[0]['intrinsics']


def test_store_import_export_legacy_config(tmp_path):
    legacy = {'cameras': [{'name': 'external_camera', 'width': 640, 'height': 480,
                           'tcp_camera': None,
                           'base_camera': [0.1, 0.2, 0.3, 0.0, 0.0, 0.0]}]}
    config_path = tmp_path / 'config.json'
    config_path.write_text(json.dumps(legacy), encoding='utf-8')
    store = CalibrationStore(tmp_path / 'calibration.json')
    assert store.import_legacy(config_path) == 1
    entry = store.get('external_camera')
    assert np.allclose(entry.base_camera, pose_to_matrix(legacy['cameras'][0]['base_camera']))
    exported = store.export_legacy()
    item = exported['cameras'][0]
    assert np.allclose(item['base_camera'],
                       matrix_to_pose(pose_to_matrix(legacy['cameras'][0]['base_camera'])))


def test_compose_camera_chain_and_inverse():
    base_tcp = [0.3, 0.1, 0.2, 0.0, 0.0, 0.0]
    tcp_camera = pose_to_matrix([0.0, 0.0, 0.05, 0.0, 0.0, 0.0])
    chain = compose_camera_chain(base_tcp, tcp_camera)
    expected = pose_to_matrix(base_tcp) @ tcp_camera
    assert np.allclose(chain, expected)
    assert np.allclose(invert_transform(chain) @ chain, np.eye(4), atol=1e-10)


def test_checkerboard_requires_opencv(tmp_path):
    import importlib.util
    if importlib.util.find_spec('cv2') is None:
        from robot_toolkit.calibrate import calibrate_checkerboard
        with pytest.raises(ImportError):
            calibrate_checkerboard([])
    else:
        pytest.skip('opencv is installed')
