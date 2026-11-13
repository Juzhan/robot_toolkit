"""Perception protocol tests; no devices, models, service startup or motion."""

import base64
import io
from types import SimpleNamespace

import numpy as np
import pytest

from robot_toolkit.camera.base import CameraFrame
from robot_toolkit.external_tools import ExternalToolError, ToolRuntime, Sam3Client, Segmentation, SegmentationResult
from robot_toolkit.perception import (
    AnyGraspClient, BundleSdfClient, PerceptionFrame, _encode_array, inspect_object,
)


def frame_at(stamp=10., observation_id='obs-1'):
    transform = np.eye(4)
    transform[:3, 3] = [1, 2, 3]
    camera_frame = CameraFrame('top', stamp, np.full((8, 10, 3), [255, 20, 0], dtype=np.uint8),
                               np.full((8, 10), 500, dtype=np.uint16),
                               np.array([[100, 0, 5], [0, 100, 4], [0, 0, 1]]),
                               np.zeros(5), transform)
    observation = SimpleNamespace(observation_id=observation_id, frames={'top': camera_frame})
    return PerceptionFrame.from_observation(observation, 'top', world_frame='base',
        calibration_id='cal-1', depth_aligned_to_rgb=True, now=stamp + 0.01)


def test_frame_units_geometry_and_png_contract():
    import cv2

    frame = frame_at()
    assert np.all(frame.depth_m == 0.5)
    mask = np.zeros((8, 10), dtype=bool)
    mask[4, 5] = True
    assert np.allclose(frame.masked_points(mask), [[1, 2, 3.5]])
    payload = frame.bundlesdf_payload()
    image = cv2.imdecode(np.frombuffer(base64.b64decode(payload['image_base64']), dtype=np.uint8), 1)
    assert np.array_equal(image[0, 0], [0, 20, 255])
    assert np.all(np.load(io.BytesIO(base64.b64decode(payload['depth_base64'])), allow_pickle=False) == 0.5)
    assert payload['intrinsics'] == [100, 100, 5, 4]
    assert not frame.rgb.flags.writeable


@pytest.mark.parametrize('problem', ['stale', 'future', 'unaligned', 'distortion', 'sync', 'depth_dtype'])
def test_observation_preconditions(problem):
    valid = frame_at()
    camera = CameraFrame('top', 10., valid.rgb, (valid.depth_m * 1000).astype(np.uint16),
                         valid.intrinsics, np.zeros(5), valid.T_world_camera)
    observation = SimpleNamespace(observation_id='obs', frames={'top': camera}, synchronization=None)
    if problem == 'distortion':
        camera.distortion[0] = 0.1
    if problem == 'sync':
        observation.synchronization = {'cameras': {'top': {'matched': False}}}
    if problem == 'depth_dtype':
        camera.depth = camera.depth.astype(float)
    now = 11. if problem == 'stale' else 9. if problem == 'future' else 10.01
    with pytest.raises(ValueError):
        PerceptionFrame.from_observation(observation, 'top', world_frame='base', calibration_id='cal',
                                         depth_aligned_to_rgb=problem != 'unaligned', now=now)


def test_anygrasp_protocol_and_explicit_tcp_transform(tmp_path, monkeypatch):
    client = AnyGraspClient(ToolRuntime('anygrasp', tmp_path))
    poses = np.eye(4)[None]
    poses[0, :3, 3] = [0.1, 0.2, 0.5]

    def request(path, payload):
        assert path == '/plan' and payload['collision_detection'] is True
        assert payload['object_input_mode'] == 'segmented_object_cloud'
        assert np.all(np.load(io.BytesIO(base64.b64decode(payload['depth_base64'])), allow_pickle=False) == 0.5)
        return {'n_grasps': 1, 'grasps_base64': _encode_array(poses),
                'scores_base64': _encode_array(np.array([0.8])),
                'widths_base64': _encode_array(np.array([0.04]))}

    monkeypatch.setattr(client, '_request', request)
    result = client.plan(frame_at(), np.ones((8, 10), dtype=bool))
    assert result.observation_id == 'obs-1'
    assert np.allclose(result.T_world_grasp[0, :3, 3], [1.1, 2.2, 3.5])
    offset = np.eye(4)
    offset[0, 3] = 0.1
    assert np.allclose(result.tcp_targets(offset)[0, :3, 3], [1.2, 2.2, 3.5])
    assert 'not robot TCP' in result.frame_convention
    assert result.frame_received_at == 10.


def test_empty_grasp_candidates_are_valid(tmp_path, monkeypatch):
    client = AnyGraspClient(ToolRuntime('anygrasp', tmp_path))
    response = {'n_grasps': 0, 'grasps_base64': _encode_array(np.empty((0, 4, 4))),
                'scores_base64': _encode_array(np.empty(0)), 'widths_base64': _encode_array(np.empty(0))}
    monkeypatch.setattr(client, '_request', lambda *args: response)
    result = client.plan(frame_at(), np.ones((8, 10), dtype=bool))
    assert result.T_world_grasp.shape == (0, 4, 4)


@pytest.mark.parametrize('corruption', ['count', 'shape', 'rotation', 'nan', 'width'])
def test_anygrasp_rejects_invalid_evidence(tmp_path, monkeypatch, corruption):
    client = AnyGraspClient(ToolRuntime('anygrasp', tmp_path))
    poses = np.eye(4)[None]
    if corruption == 'rotation':
        poses[0, 0, 0] = 2
    if corruption == 'nan':
        poses[0, 0, 3] = np.nan
    response = {'n_grasps': 1000 if corruption == 'count' else 1,
                'grasps_base64': _encode_array(poses[0] if corruption == 'shape' else poses),
                'scores_base64': _encode_array(np.array([0.8])),
                'widths_base64': _encode_array(np.array([-0.1 if corruption == 'width' else 0.04]))}
    monkeypatch.setattr(client, '_request', lambda *args: response)
    with pytest.raises(ExternalToolError):
        client.plan(frame_at(), np.ones((8, 10), dtype=bool))


def test_tracking_keeps_one_session_but_does_not_claim_pose_freshness(tmp_path, monkeypatch):
    client = BundleSdfClient(ToolRuntime('bundlesdf', tmp_path))
    calls = []

    def request(path, payload=None):
        calls.append(path)
        if path == '/push_frame':
            return {'ok': True, 'camera': 'top', 'seq': len(calls)}
        if path == '/add_detection':
            return {'name': 'block', 'bbox': [0, 0, 10, 8], 'first_score': 0.9}
        if path == '/get_detection/block':
            return {'tracking': True, 'frame_idx': 2, 'score': 0.9,
                    'ob_in_cam': np.eye(4).tolist(), 'ob_in_world': (np.eye(4) * 99).tolist()}
        return {'ok': True}

    monkeypatch.setattr(client, '_request', request)
    session = client.start(frame_at(), 'red block', name='block')
    client.push(session, frame_at(11., 'in-hand'))
    client.push(session, frame_at(12., 'placed'))
    result = client.pose(session)
    assert result.source_observation_id is None
    assert result.last_submitted_observation_id == 'placed'
    assert result.freshness.startswith('unverified')
    assert np.allclose(result.T_world_object[:3, 3], [1, 2, 3])
    assert calls.count('/add_detection') == 1
    with pytest.raises(ValueError, match='new observation'):
        client.push(session, frame_at(12., 'placed'))
    client.end(session)
    client.end(session)
    assert calls.count('/end_detection/block') == 1
    with pytest.raises(ValueError, match='closed'):
        client.pose(session)


def test_parent_roi_part_masks_are_returned_in_full_image_coordinates(tmp_path, monkeypatch):
    sam = Sam3Client(ToolRuntime('sam3', tmp_path))
    shapes = []

    def segment(rgb, query, *, observation_id, camera):
        shapes.append(rgb.shape)
        mask = np.zeros(rgb.shape[:2], dtype=bool)
        bbox = (2, 2, 6, 4) if query == 'drawer' else (1, 1, 2, 1)
        left, top, width, height = bbox
        mask[top:top + height, left:left + width] = True
        return SegmentationResult(observation_id, camera, query,
                                  (Segmentation(mask, bbox, 0.9, int(mask.sum())),))

    monkeypatch.setattr(sam, 'segment', segment)
    inspection = inspect_object(sam, frame_at(), 'drawer', part_queries={'handle': 'drawer handle'})
    handle = inspection.parts['handle'].detections[0]
    assert shapes == [(8, 10, 3), (4, 6, 3)]
    assert handle.bbox_xywh == (3, 3, 2, 1) and handle.mask.shape == (8, 10)
    assert len(inspection.points_world) == 24
    assert inspection.visible_bounds_world.shape == (2, 3)
    assert 'not a verified joint' in inspection.part_relationship_status


@pytest.mark.parametrize('count', [0, 2])
def test_inspection_does_not_choose_an_ambiguous_parent(tmp_path, monkeypatch, count):
    sam = Sam3Client(ToolRuntime('sam3', tmp_path))
    candidate = Segmentation(np.ones((8, 10), dtype=bool), (0, 0, 10, 8), 0.9, 80)
    monkeypatch.setattr(sam, 'segment', lambda *args, **kwargs:
                        SegmentationResult('obs-1', 'top', 'drawer', (candidate,) * count))
    with pytest.raises(ExternalToolError, match='expected one target'):
        inspect_object(sam, frame_at(), 'drawer', part_queries={'handle': 'handle'})


def test_bundle_health_is_not_model_readiness(tmp_path, monkeypatch):
    client = BundleSdfClient(ToolRuntime('bundlesdf', tmp_path))
    monkeypatch.setattr(client, '_request', lambda *args: {'status': 'ok'})
    assert client.health()['ready'] is None


def test_new_clients_stay_disabled_by_default(tmp_path):
    from robot_toolkit.external_tools import ExternalTools

    tools = ExternalTools.load(root=tmp_path)
    for client in (tools.bundlesdf(), tools.anygrasp()):
        with pytest.raises(ExternalToolError, match='disabled'):
            client.health()


def test_reconstruction_preserves_metric_frame_and_rejects_stale_pose():
    pytest.importorskip('open3d')
    from robot_toolkit.reconstruction import ObjectReconstruction, distinct_masks

    frame = frame_at()
    mask = np.ones(frame.depth_m.shape, dtype=bool)
    fusion = ObjectReconstruction(voxel_size_m=.002)
    with pytest.raises(ValueError, match='this observation'):
        fusion.integrate(frame, mask, T_world_object=np.eye(4),
                         pose_observation_id='old', pose_source='test')
    pose = np.eye(4)
    pose[:3, 3] = [1, 2, 3]
    fusion.integrate(frame, mask, T_world_object=pose,
                     pose_observation_id=frame.observation_id, pose_source='synthetic stationary fixture')
    points = np.asarray(fusion.point_cloud().points)
    assert len(points) > 16 and np.max(np.abs(points[:, 2] - .5)) < .003
    assert len(distinct_masks([mask, mask])) == 1
    with pytest.raises(ValueError, match='duplicate'):
        fusion.integrate(frame, mask, T_world_object=pose,
                         pose_observation_id=frame.observation_id, pose_source='test')


def test_stationary_registration_requires_independent_synchronized_views():
    o3d = pytest.importorskip('open3d')
    from dataclasses import replace
    from robot_toolkit.reconstruction import register_stationary_views

    frame = frame_at()
    mask = np.ones(frame.depth_m.shape, dtype=bool)
    reference = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(frame.masked_points(mask)))
    second = replace(frame, camera='second')
    aligned = register_stationary_views(reference, [(frame, mask), (second, mask)],
                                        initial_world_pose=np.eye(4))
    assert len(aligned) == 2
    assert max(result['rmse_m'] for result in aligned) < .003
    with pytest.raises(ValueError, match='distinct camera'):
        register_stationary_views(reference, [(frame, mask), (frame, mask)], initial_world_pose=np.eye(4))
    with pytest.raises(ValueError, match='one stationary observation'):
        register_stationary_views(reference, [(frame, mask), (replace(second, observation_id='old'), mask)],
                                  initial_world_pose=np.eye(4))
    shifted = second.T_world_camera.copy()
    shifted[0, 3] += .007
    with pytest.raises(ValueError, match='disagree'):
        register_stationary_views(reference, [(frame, mask), (replace(second, T_world_camera=shifted), mask)],
                                  initial_world_pose=np.eye(4), agreement_m=.0001)

def test_stereo_depth_client_and_registration(monkeypatch):
    import base64
    import io
    from robot_toolkit.external_tools import ToolRuntime
    from robot_toolkit.perception import StereoDepthClient, register_depth

    sent = {}

    def fake(self, path, payload=None):
        if payload is None:
            return {'status': 'ok', 'model_loaded': True, 'model_ref': 'abc'}
        sent.update(payload)
        buffer = io.BytesIO()
        np.save(buffer, np.full((24, 32), .5, np.float32), allow_pickle=False)
        return {'depth': base64.b64encode(buffer.getvalue()).decode(), 'shape': [24, 32],
                'scale': .5, 'seconds': .1, 'model_ref': 'abc'}

    monkeypatch.setattr(StereoDepthClient, '_request', fake)
    client = StereoDepthClient(ToolRuntime('fast_foundationstereo', root='/tmp/tools-test',
                                           endpoint='http://127.0.0.1:6790', enabled=True))
    assert client.health()['ready']
    K = np.array([[100., 0, 32], [0, 100., 24], [0, 0, 1]])
    image = np.zeros((48, 64), np.uint8)
    result = client.depth(image, image, K, baseline_m=.05, width=32)
    assert result.depth.shape == (24, 32) and sent['width'] == 32 and sent['fx'] == 100.
    np.testing.assert_allclose(result.intrinsics[:2], K[:2] * .5)
    with pytest.raises(ValueError):
        client.depth(image, image.astype(np.float32), K, baseline_m=.05)
    # Registration: identity keeps depth; a 1 cm shift moves pixels by fx * 0.01 / z.
    depth = np.full((48, 64), .5, np.float32)
    same = register_depth(depth, K, K, np.eye(4), (48, 64))
    np.testing.assert_allclose(same[5:-5, 5:-5], .5)
    T = np.eye(4)
    T[0, 3] = .01
    shifted = register_depth(depth, K, K, T, (48, 64))
    assert (shifted[:, :2] == 0).all() and shifted[10, 10] == pytest.approx(.5)
