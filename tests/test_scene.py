"""Synthetic-image contracts for scene estimates and layout acceptance."""

import numpy as np
import pytest

from robot_toolkit import RobotPlatform
from robot_toolkit.camera import CameraHub, MockCamera
from robot_toolkit.camera.base import CameraFrame
from robot_toolkit.robot import MockRobot
from robot_toolkit.runtime.robot_platform import Observation
from robot_toolkit.perception.scene import (ObjectSpec, PlacementCondition, PlanarColorEstimator,
                                 ResetLayoutVerifier, SceneSpec)


def make_spec(**changes):
    values = dict(camera='top', calibration_id='synthetic-v1', coordinate_frame='table',
                  image_size=(100, 80), image_to_plane=np.diag([0.001, 0.001, 1.0]),
                  workspace_xy=(0.0, 0.0, 0.1, 0.08),
                  objects=(ObjectSpec('block', (((0, 150, 150), (10, 255, 255)),)),))
    values.update(changes)
    return SceneSpec(**values)


def make_observation(stamp=10.0, *, duplicate=False, missing=False):
    image = np.zeros((80, 100, 3), dtype=np.uint8)
    if not missing:
        image[20:40, 30:50] = [255, 0, 0]
    if duplicate:
        image[50:65, 65:80] = [255, 0, 0]
    frame = CameraFrame('top', stamp, rgb=image)
    return Observation(str(stamp), stamp, stamp, True, None, frames={'top': frame})


def test_planar_detection_retains_numeric_position_and_provenance():
    pytest.importorskip('cv2')
    spec = make_spec()
    scene = PlanarColorEstimator(spec, clock=lambda: 10.1).estimate(make_observation())
    assert scene.valid and scene.spec_id == spec.spec_id
    assert scene.calibration_id == 'synthetic-v1' and scene.observation_id == '10.0'
    assert np.allclose(scene.objects['block'].xy_m, [0.0395, 0.0295])
    assert scene.objects['block'].area_px == 400


@pytest.mark.parametrize('options, reason', [({'missing': True}, 'missing'),
                                           ({'duplicate': True}, 'ambiguous')])
def test_missing_or_ambiguous_object_has_no_position(options, reason):
    pytest.importorskip('cv2')
    estimator = PlanarColorEstimator(make_spec(), clock=lambda: 10.1)
    assert estimator.estimate(make_observation()).valid
    scene = estimator.estimate(make_observation(**options))
    assert not scene.valid
    assert scene.objects['block'].reason == reason
    assert scene.objects['block'].xy_m is None


def test_layout_requires_fresh_stable_frames_and_rejects_replay():
    pytest.importorskip('cv2')
    now = [10.01]
    spec = make_spec()
    estimator = PlanarColorEstimator(spec, clock=lambda: now[0])
    verifier = ResetLayoutVerifier(
        spec, (PlacementCondition('block', (0.0395, 0.0295)),), clock=lambda: now[0])
    for stamp, ready in [(10.0, False), (10.11, False), (10.22, True)]:
        now[0] = stamp + 0.01
        scene = estimator.estimate(make_observation(stamp))
        check = verifier.check(scene)
        assert check.layout_ready is ready
    assert check.consecutive_frames == 3
    assert not verifier.check(scene).layout_ready
    assert verifier.check(scene).reason == 'repeated_or_out_of_order_frame'
    now[0] = 10.4
    missing = estimator.estimate(make_observation(10.39, missing=True))
    assert verifier.check(missing).reason == 'invalid_scene'


@pytest.mark.parametrize('stamp, reason', [(9.0, 'stale_image'),
                                         (11.0, 'invalid_frame_time'),
                                         (float('nan'), 'invalid_frame_time'),
                                         (0.0, 'invalid_frame_time')])
def test_invalid_frame_timing_never_produces_pose(stamp, reason):
    pytest.importorskip('cv2')
    scene = PlanarColorEstimator(make_spec(), clock=lambda: 10.1).estimate(make_observation(stamp))
    assert not scene.valid and scene.objects['block'].reason == reason
    assert scene.objects['block'].xy_m is None


@pytest.mark.parametrize('problem, reason', [
    ('missing_camera', 'missing_image'), ('resolution', 'image_contract_mismatch'),
    ('dtype', 'image_contract_mismatch'), ('distortion', 'undistortion_required'),
    ('synchronization', 'unsynchronized'), ('border', 'clipped'),
])
def test_image_evidence_gates(problem, reason):
    pytest.importorskip('cv2')
    observation = make_observation()
    frame = observation.frames['top']
    if problem == 'missing_camera':
        observation.frames = {}
    elif problem == 'resolution':
        frame.rgb = frame.rgb[:40]
    elif problem == 'dtype':
        frame.rgb = frame.rgb.astype(float)
    elif problem == 'distortion':
        frame.distortion = np.array([0.1, 0, 0, 0, 0])
    elif problem == 'synchronization':
        observation.synchronization = {'cameras': {'top': {'matched': False}}}
    elif problem == 'border':
        frame.rgb[:] = 0
        frame.rgb[:20, :20] = [255, 0, 0]
    scene = PlanarColorEstimator(make_spec(), clock=lambda: 10.1).estimate(observation)
    assert not scene.valid and scene.objects['block'].reason == reason


def test_color_overlap_cannot_assign_one_object_to_two_ids():
    pytest.importorskip('cv2')
    first = make_spec().objects[0]
    second = ObjectSpec('other', first.hsv_ranges)
    spec = make_spec(objects=(first, second))
    scene = PlanarColorEstimator(spec, clock=lambda: 10.1).estimate(make_observation())
    assert not scene.valid
    assert all(item.reason == 'ambiguous_identity' and item.xy_m is None
               for item in scene.objects.values())


@pytest.mark.parametrize('changes, reason', [
    ({'workspace_xy': (0.0, 0.0, 0.04, 0.08)}, 'outside_workspace'),
    ({'image_to_plane': [[0.001, 0, 0], [0, 0.001, 0], [0.02, 0, -0.8]]},
     'invalid_projection'),
])
def test_projection_rejects_footprint_outside_workspace_and_horizon(changes, reason):
    pytest.importorskip('cv2')
    scene = PlanarColorEstimator(make_spec(**changes), clock=lambda: 10.1).estimate(make_observation())
    assert not scene.valid and scene.objects['block'].reason == reason


def test_oversized_component_is_not_silently_accepted():
    pytest.importorskip('cv2')
    definition = ObjectSpec('block', make_spec().objects[0].hsv_ranges, max_area_px=200)
    scene = PlanarColorEstimator(make_spec(objects=(definition,)), clock=lambda: 10.1).estimate(
        make_observation())
    assert scene.objects['block'].reason == 'size_mismatch'


@pytest.mark.parametrize('changes', [
    {'image_to_plane': np.zeros((3, 3))}, {'calibration_id': ''},
    {'objects': ()}, {'image_size': (0, 80)}, {'workspace_xy': (1, 0, 0, 1)},
    {'max_frame_age_s': float('nan')},
])
def test_invalid_scene_configuration_is_rejected(changes):
    with pytest.raises(ValueError):
        make_spec(**changes)


def test_configuration_is_detached_and_versioned():
    transform = np.diag([0.001, 0.001, 1.0])
    spec = make_spec(image_to_plane=transform)
    original_id = spec.spec_id
    transform[0, 0] = 99
    assert spec.spec_id == original_id
    assert make_spec(calibration_id='different').spec_id != original_id


def test_layout_rejects_drift_gap_contract_changes_and_stale_results():
    pytest.importorskip('cv2')
    now = [10.01]
    spec = make_spec()
    estimator = PlanarColorEstimator(spec, clock=lambda: now[0])
    verifier = ResetLayoutVerifier(spec, (PlacementCondition('block', (0.0395, 0.0295)),),
                                   clock=lambda: now[0])
    for stamp, displacement in [(10.0, 0.0), (10.11, 0.002), (10.22, 0.004)]:
        now[0] = stamp + 0.01
        scene = estimator.estimate(make_observation(stamp))
        scene.objects['block'].xy_m[0] += displacement
        check = verifier.check(scene)
        assert not check.layout_ready
    assert check.consecutive_frames == 1
    now[0] = 11.01
    scene = estimator.estimate(make_observation(11.0))
    assert verifier.check(scene).consecutive_frames == 1
    scene.calibration_id = 'wrong-camera-calibration'
    assert verifier.check(scene).reason == 'scene_contract_mismatch'
    scene.calibration_id = spec.calibration_id
    now[0] = 12.0
    assert verifier.check(scene).reason == 'stale_image'


def test_layout_requires_all_objects_and_valid_targets():
    spec = make_spec()
    with pytest.raises(ValueError, match='every scene object'):
        ResetLayoutVerifier(spec, ())
    with pytest.raises(ValueError, match='inside the workspace'):
        ResetLayoutVerifier(spec, (PlacementCondition('block', (1.0, 1.0)),))
    with pytest.raises(ValueError, match='required_frames'):
        ResetLayoutVerifier(spec, (PlacementCondition('block', (0.04, 0.03)),), required_frames=1)


def test_platform_estimates_once_and_retains_detached_scene(monkeypatch):
    pytest.importorskip('cv2')
    camera = MockCamera('top', 100, 80).open()
    calls = []

    def read():
        import time
        calls.append(True)
        return make_observation(time.perf_counter()).frames['top']

    monkeypatch.setattr(camera, 'read', read)
    robot = MockRobot()
    estimator = PlanarColorEstimator(make_spec())
    platform = RobotPlatform(robot, cameras=CameraHub([camera]), scene_estimator=estimator,
                             sync_max_skew_s=1.0)
    observation = platform.observe()
    assert len(calls) == 1 and observation.scene.valid
    assert observation.scene.observation_id == observation.observation_id
    observation.scene.objects['block'].xy_m[:] = 99
    retained = platform.observation(observation.observation_id)
    assert retained.frames == {} and retained.scene.valid
    assert np.allclose(retained.scene.objects['block'].xy_m, [0.0395, 0.0295])
    assert platform.observe(include_images=False).scene is None
    assert platform.move_j(np.zeros(robot.dof)).ok
    assert len(calls) == 1 and robot.commands == []
    with pytest.raises(ValueError, match='configured camera'):
        RobotPlatform(robot, scene_estimator=estimator)