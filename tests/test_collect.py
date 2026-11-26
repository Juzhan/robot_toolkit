"""Dataset contract, async recorder and exporter tests."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from robot_toolkit.camera import MockCamera
from robot_toolkit.camera.base import ThreadedCamera
from robot_toolkit.camera import CameraHub
from robot_toolkit.collect.observation import RobotCameraSynchronizer
from robot_toolkit.collect import (CameraSpec, DatasetSchema, EpisodeCollector,
                                   EpisodeRecorder, collect_trajectory,
                                   export_lerobot, export_manifest,
                                   split_camera_frames)
from robot_toolkit.core.conventions import RobotSpec
from robot_toolkit.errors import DatasetError
from robot_toolkit.errors import RobotUnavailable
from robot_toolkit.robot import MockRobot
from robot_toolkit.robot.base import RobotState
from robot_toolkit.runtime.profiles import build_robot
from robot_toolkit.teach import Trajectory


def _schema(fps=10):
    return DatasetSchema(robot=RobotSpec.ur5e(),
                         cameras=(CameraSpec('wrist_camera', 32, 24),
                                  CameraSpec('external_camera', 32, 24)),
                         fps=fps)


def _image():
    return np.zeros((24, 32, 3), dtype=np.uint8)


def test_camera_read_copies_outside_lock_without_mixing_frames():
    camera = ThreadedCamera('wrist')
    next_rgb = np.full((3, 4, 3), 2, dtype=np.uint8)
    next_depth = np.full((3, 4), 200, dtype=np.uint16)

    class PublishingArray(np.ndarray):
        def copy(self, order='C'):
            assert camera._lock.acquire(blocking=False), 'image copy blocks the publisher'
            camera._lock.release()
            camera._publish(next_rgb, next_depth, {'frame_number': 2}, {'frame_number': 2})
            return super().copy(order=order)

    rgb = np.ones((3, 4, 3), dtype=np.uint8).view(PublishingArray)
    depth = np.full((3, 4), 100, dtype=np.uint16)
    camera._publish(rgb, depth, {'frame_number': 1}, {'frame_number': 1})
    stamp = camera._stamp
    first = camera.read()
    second = camera.read()
    assert first.t == stamp
    assert np.all(first.rgb == 1) and np.all(first.depth == 100)
    assert first.meta['color']['frame_number'] == first.meta['depth']['frame_number'] == 1
    assert np.all(second.rgb == 2) and np.all(second.depth == 200)
    assert second.meta['color']['frame_number'] == second.meta['depth']['frame_number'] == 2
    first.rgb.fill(9)
    first.depth.fill(999)
    second.rgb.fill(8)
    second.depth.fill(888)
    assert np.all(rgb == 1) and np.all(depth == 100)
    assert np.all(next_rgb == 2) and np.all(next_depth == 200)


def test_camera_read_before_first_frame_remains_empty():
    frame = ThreadedCamera('empty').read()
    assert frame.rgb is None and frame.depth is None and frame.t == 0.0


def test_synchronizer_pairs_host_history_not_robot_sdk_time():
    camera = ThreadedCamera('wrist')
    camera._publish(_image())
    camera._stamp = 10.01
    robot = MockRobot(RobotSpec.generic(1))
    times = iter([10.1, 10.101, 10.102, 10.103, 10.104])
    sync = RobotCameraSynchronizer(robot, CameraHub([camera]), clock=lambda: next(times))
    sample = RobotState(t=900000.0, q=[0.3], timestamp_clock='controller')
    sync.add_state(sample, started_at=10.0, finished_at=10.002)
    sample.q[0] = 99
    result = sync.observe()
    evidence = result['synchronization']
    assert evidence['matched'] and not evidence['exposure_synchronized']
    assert result['state'].q == pytest.approx([0.3])
    pair = evidence['cameras']['wrist']
    assert pair['state']['t'] == 900000.0
    assert pair['state_minus_frame_s'] == pytest.approx(-0.009)
    assert pair['read_window_skew_bound_s'] == pytest.approx(0.01)
    assert robot.commands == []
    json.dumps(evidence, allow_nan=False)


@pytest.mark.parametrize('frame_time, expected', [
    (9.0, 'stale_image'), (9.9, 'skew_exceeded'),
    (11.0, 'invalid_frame_time'), (float('nan'), 'invalid_frame_time')])
def test_synchronizer_rejects_unmatched_images(frame_time, expected):
    camera = ThreadedCamera('wrist')
    camera._publish(_image())
    camera._stamp = frame_time
    robot = MockRobot(RobotSpec.generic(1))
    sync = RobotCameraSynchronizer(robot, CameraHub([camera]), clock=lambda: 10.0)
    result = sync.observe(strict=False)
    assert not result['synchronization']['matched']
    assert result['state'] is None
    assert result['synchronization']['cameras']['wrist']['reason'] == expected
    json.dumps(result['synchronization'], allow_nan=False)
    with pytest.raises(RobotUnavailable, match=expected):
        sync.observe()
    assert robot.commands == []


def test_synchronizer_rejects_slow_reads_even_when_midpoint_matches():
    camera = ThreadedCamera('wrist')
    camera._publish(_image())
    camera._stamp = 10.1
    times = iter([10.0, 10.2, 10.201, 10.202, 10.203])
    sync = RobotCameraSynchronizer(MockRobot(RobotSpec.generic(1)), CameraHub([camera]),
                                   clock=lambda: next(times))
    pair = sync.observe(strict=False)['synchronization']['cameras']['wrist']
    assert pair['state_minus_frame_s'] == pytest.approx(0.0)
    assert pair['read_window_skew_bound_s'] == pytest.approx(0.1)
    assert not pair['matched']


def test_synchronizer_bounds_history_and_rejects_bad_feedback():
    sync = RobotCameraSynchronizer(MockRobot(RobotSpec.generic(1)),
                                   CameraHub([ThreadedCamera('wrist')]), max_samples=2, history_s=1)
    for stamp in (1.0, 1.1, 1.2):
        sync.add_state(RobotState(q=[stamp]), started_at=stamp, finished_at=stamp)
    assert len(sync._history) == 2
    sync.add_state(RobotState(q=[3]), started_at=3.0, finished_at=3.0)
    assert len(sync._history) == 1
    with pytest.raises(ValueError, match='ordered'):
        sync.add_state(RobotState(q=[0]), started_at=2.0, finished_at=2.0)
    with pytest.raises(RobotUnavailable):
        sync.add_state(RobotState(q=[np.nan]), started_at=4.0, finished_at=4.0)
    assert len(sync._history) == 0


def test_synchronizer_preserves_each_cameras_matched_state():
    wrist, external = ThreadedCamera('wrist'), ThreadedCamera('external')
    wrist._publish(_image())
    external._publish(_image())
    wrist._stamp, external._stamp = 10.0, 10.1
    sync = RobotCameraSynchronizer(MockRobot(RobotSpec.generic(1)),
                                   CameraHub([wrist, external]), clock=lambda: 10.2)
    sync.add_state(RobotState(q=[0.1]), started_at=10.0, finished_at=10.001)
    sync.add_state(RobotState(q=[0.2]), started_at=10.1, finished_at=10.101)
    result = sync.observe()
    evidence = result['synchronization']
    assert evidence['reference_camera'] == 'wrist'
    assert evidence['camera_received_span_s'] == pytest.approx(0.1)
    assert result['state'].q == pytest.approx([0.1])
    assert evidence['cameras']['external']['state']['q'] == [0.2]
    assert evidence['matched'] and not evidence['exposure_synchronized']


@pytest.mark.parametrize('kind', ['mock', 'piper_x', 'ur_rtde'])
def test_synchronizer_supports_uniform_robot_backends(kind):
    robot = build_robot(kind, mock=True).connect()
    camera = MockCamera('wrist', 32, 24).open()
    before = robot.state().q.copy()
    try:
        result = RobotCameraSynchronizer(robot, CameraHub([camera])).observe()
        assert result['synchronization']['matched']
        assert result['state'].q == pytest.approx(before)
        assert robot.state().q == pytest.approx(before)
    finally:
        camera.close()
        robot.disconnect()


def test_synchronizer_refuses_disconnected_robot_without_reconnecting():
    robot = MockRobot(RobotSpec.generic(1))
    sync = RobotCameraSynchronizer(robot, CameraHub([ThreadedCamera('wrist')]))
    sync.add_state(robot.state(), started_at=1.0, finished_at=1.0)
    robot.disconnect()
    with pytest.raises(RobotUnavailable, match='disconnected'):
        sync.observe()
    assert not robot.connected and not sync._history
    assert robot.commands == []


def test_synchronizer_missing_camera_and_constructor_validation():
    robot = MockRobot(RobotSpec.generic(1))
    hub = CameraHub([ThreadedCamera('wrist')])
    sync = RobotCameraSynchronizer(robot, hub)
    with pytest.raises(RobotUnavailable, match='missing_image'):
        sync.observe()
    for options in ({'max_skew_s': 0}, {'max_frame_age_s': float('nan')},
                    {'history_s': -1}, {'max_samples': 1}):
        with pytest.raises(ValueError):
            RobotCameraSynchronizer(robot, hub, **options)
    with pytest.raises(ValueError, match='camera'):
        RobotCameraSynchronizer(robot, CameraHub())


def test_synchronized_collector_persists_pairs_without_double_camera_read(tmp_path):
    robot = MockRobot(RobotSpec.generic(1, with_gripper=False))
    camera = MockCamera('wrist', 32, 24)
    schema = DatasetSchema.for_robot(robot.spec, cameras=[camera])
    collector = EpisodeCollector(tmp_path, schema, robot=robot, cameras=[camera],
                                 export=export_manifest, synchronize=True)
    take = collector.start('synchronized observation')
    try:
        initial_frames = camera.frames_seen
        assert collector.capture_state(extra={'source_index': [7]})
        assert collector.capture_state(action=[0.2])
        assert camera.frames_seen == initial_frames + 2
        collector.stop(success=True)
    finally:
        collector.close()
    frames = sorted(take.glob('[0-9]*.npz'))
    assert len(frames) == 2
    with np.load(frames[0], allow_pickle=False) as saved:
        evidence = json.loads(saved['synchronization_json'].item())
        assert evidence['matched'] and not evidence['exposure_synchronized']
        pair = evidence['cameras']['wrist']
        assert pair['frame_metadata']['frame_number'] == initial_frames + 1
        assert saved['observation_state'] == pytest.approx(pair['state']['q'])
        assert evidence['action_alignment'] == 'reference_state_copy'
        assert saved['source_index'].tolist() == [7]
        assert saved['wrist_rgb'].shape == (24, 32, 3)
    with np.load(frames[1], allow_pickle=False) as saved:
        evidence = json.loads(saved['synchronization_json'].item())
        assert evidence['action_alignment'] == 'caller_supplied_not_time_aligned'
        assert saved['action'] == pytest.approx([0.2])
    assert robot.commands == []


def test_synchronized_collector_rejects_untimed_and_stale_inputs(tmp_path):
    robot = MockRobot(RobotSpec.generic(1, with_gripper=False))
    camera = MockCamera('wrist', 32, 24)
    collector = EpisodeCollector(tmp_path, DatasetSchema.for_robot(robot.spec, cameras=[camera]),
                                 robot=robot, cameras=[camera], synchronize=True)
    collector.start('rejections')
    try:
        with pytest.raises(DatasetError, match='untimed'):
            collector.capture_state(state=robot.state())
        with pytest.raises(DatasetError, match='bare vectors'):
            collector.capture_vectors([0], [0])
        with pytest.raises(DatasetError, match='reserved'):
            collector.capture_state(extra={'synchronization_json': 'forged'})
        collector.synchronizer.clock = lambda: 1e15
        with pytest.raises(DatasetError, match='stale_image'):
            collector.capture_state(strict_cameras=False)
        assert collector.recorder.frames == 0
    finally:
        collector.discard()
        collector.close()


def test_synchronized_collector_does_not_substitute_live_gripper(tmp_path, monkeypatch):
    robot = MockRobot(RobotSpec.generic(1))
    camera = MockCamera('wrist', 32, 24)
    class UntimedGripper:
        @property
        def width(self):
            raise AssertionError('must not read a live gripper for an old image')
    monkeypatch.setattr(robot, 'state', lambda: RobotState(q=[0], extra={}))
    collector = EpisodeCollector(tmp_path, DatasetSchema.for_robot(robot.spec, cameras=[camera]),
                                 robot=robot, gripper=UntimedGripper(), cameras=[camera],
                                 synchronize=True)
    collector.start('missing matched gripper')
    try:
        with pytest.raises(DatasetError, match='gripper width'):
            collector.capture_state()
        assert collector.recorder.frames == 0
    finally:
        collector.discard()
        collector.close()


def test_schema_features_match_contract():
    schema = _schema()
    features = schema.features()
    assert features['observation.state']['shape'] == (7,)
    assert features['observation.state']['names'] == schema.state_names
    assert features['action']['shape'] == (7,)
    assert features['observation.images.wrist_camera']['shape'] == (24, 32, 3)
    contract = schema.contract()
    assert contract['state_action']['dim'] == 7
    assert contract['depth'].startswith('uint16')
    assert len(contract['cameras']) == 2
    restored = DatasetSchema.from_dict(schema.to_dict())
    assert restored.state_names == schema.state_names
    assert restored.cameras == schema.cameras


def test_robot_only_schema_has_no_image_features():
    schema = DatasetSchema.for_robot(RobotSpec.piper_x(), cameras=(), fps=10)
    assert schema.cameras == ()
    assert schema.require_images is False
    assert not any(key.startswith('observation.images') for key in schema.features())
    frame = schema.frame(np.zeros(7), np.zeros(7), task='robot only')
    assert frame['task'] == 'robot only'


def test_schema_from_dict_keeps_explicit_empty_cameras():
    schema = DatasetSchema.for_robot(RobotSpec.piper_x(), cameras=(), fps=10)
    restored = DatasetSchema.from_dict({**schema.to_dict(), 'cameras': []})
    assert restored.cameras == ()
    assert restored.require_images is False


def test_camera_spec_from_camera():
    camera = MockCamera('wrist_camera', 32, 24, depth=True)
    spec = CameraSpec.from_camera(camera)
    assert spec.name == 'wrist_camera'
    assert (spec.width, spec.height, spec.depth) == (32, 24, True)
    schema = DatasetSchema.for_robot(RobotSpec.piper_x(), cameras=(camera,))
    assert schema.cameras == (spec,)
    assert schema.require_images is True


def test_recorder_robot_only(tmp_path):
    schema = DatasetSchema.for_robot(RobotSpec.piper_x(), cameras=(), fps=10)
    recorder = EpisodeRecorder(tmp_path, schema, export=export_manifest)
    recorder.start('robot-only task')
    for index in range(3):
        assert recorder.capture(np.zeros(7), np.zeros(7))
    recorder.save(success=True)
    recorder.close()
    assert recorder.saved == 1
    entry = json.loads((recorder.dataset_root / 'episodes.jsonl')
                       .read_text(encoding='utf-8').strip())
    assert entry['cameras'] == []
    assert entry['frames'] == 3


def _spooled_frames(recorder) -> list[Path]:
    return sorted(recorder.session.glob('take_*/0*.npz'))


def _spool_compress_types(tmp_path, **kwargs) -> set:
    schema = DatasetSchema.for_robot(RobotSpec.piper_x(), cameras=(), fps=10)
    recorder = EpisodeRecorder(tmp_path, schema, export=export_manifest, **kwargs)
    recorder.start('spool task')
    for _ in range(3):
        assert recorder.capture(np.zeros(7), np.zeros(7))
    recorder.save(success=True)
    recorder.close()
    frames = _spooled_frames(recorder)
    assert frames, 'no spooled frames'
    with zipfile.ZipFile(frames[0]) as archive:
        return {item.compress_type for item in archive.infolist()}


def test_spool_frames_are_uncompressed_by_default(tmp_path):
    # Deflating a 640x480 frame costs ~50 ms and starves the spool queue at 30 Hz.
    assert _spool_compress_types(tmp_path) == {zipfile.ZIP_STORED}


def test_spool_frames_compress_when_requested(tmp_path):
    assert _spool_compress_types(tmp_path, compress=True) == {zipfile.ZIP_DEFLATED}


def test_episode_collector_robot_only(tmp_path):
    schema = DatasetSchema.for_robot(RobotSpec.piper_x(), cameras=(), fps=10)
    robot = MockRobot(spec=schema.robot)
    collector = EpisodeCollector(tmp_path / 'session', schema, robot=robot,
                                 export=export_manifest, fps=10)
    collector.start('collect task')
    for _ in range(3):
        assert collector.capture_state()
    collector.stop(success=True)
    collector.close()
    assert collector.saved == 1


def test_collect_trajectory_to_dataset(tmp_path):
    schema = DatasetSchema.for_robot(RobotSpec.piper_x(), cameras=(), fps=10)
    trajectory = Trajectory(t=np.arange(4) / 10.0,
                            q=np.zeros((4, 6)),
                            grip=np.full(4, 0.08))
    collector = collect_trajectory(trajectory, schema, tmp_path / 'trajectory_session',
                                   task='trajectory task', export=export_manifest)
    collector.close()
    assert collector.saved == 1


def test_schema_validates_frames():
    schema = _schema()
    frame = schema.frame(np.zeros(7), np.zeros(7),
                         images={'wrist_camera': _image(),
                                 'external_camera': _image()},
                         task='test')
    assert frame['task'] == 'test'
    with pytest.raises(DatasetError):
        schema.frame(np.zeros(7), np.zeros(6))
    with pytest.raises(DatasetError):
        schema.frame(np.zeros(7), np.zeros(7),
                     images={'wrist_camera': np.zeros((10, 10, 3), np.uint8)})
    with pytest.raises(DatasetError):
        schema.frame(np.full(7, np.nan), np.zeros(7),
                     images={'wrist_camera': _image(),
                             'external_camera': _image()})


def test_recorder_requires_all_schema_cameras(tmp_path):
    schema = _schema()
    recorder = EpisodeRecorder(tmp_path, schema)
    recorder.start('task')
    with pytest.raises(DatasetError):
        recorder.capture(np.zeros(7), np.zeros(7),
                         images={'wrist_camera': _image()})
    recorder.discard()
    recorder.close()


def test_recorder_background_export(tmp_path):
    exported = []

    def export(take, dataset_root):
        exported.append(take.name)
        Path(dataset_root).mkdir(parents=True, exist_ok=True)
        (Path(dataset_root) / 'exported.txt').write_text(take.name, encoding='utf-8')

    recorder = EpisodeRecorder(tmp_path, _schema(), export=export)
    take = recorder.start('pick cube', meta={'teleoperator': 'test'})
    for index in range(5):
        assert recorder.capture(np.full(7, index * 0.01), np.full(7, index * 0.01),
                                images={'wrist_camera': _image(),
                                        'external_camera': _image()},
                                timestamp=index / 10)
    recorder.save(success=True)
    recorder.close()
    assert exported == [take.name]
    assert (recorder.dataset_root / 'exported.txt').exists()
    assert recorder.saved == 1
    assert recorder.status().phase == 'idle'
    frames = sorted(take.glob('[0-9]*.npz'))
    assert len(frames) == 5
    with np.load(frames[0]) as sample:
        assert sample['observation_state'].shape == (7,)
        assert sample['wrist_camera_rgb'].shape == (24, 32, 3)
        assert sample['timestamp'][0] == 0.0
    meta = json.loads((take / 'take.json').read_text(encoding='utf-8'))
    assert meta['status'] == 'saved'
    assert meta['success'] is True
    assert meta['task'] == 'pick cube'


def test_recorder_deferred_export(tmp_path):
    exported = []
    recorder = EpisodeRecorder(tmp_path, _schema(),
                               export=lambda take, root: exported.append(take.name),
                               export_mode='deferred')
    recorder.start('deferred task')
    for _ in range(3):
        recorder.capture(np.zeros(7), np.zeros(7),
                         images={'wrist_camera': _image(),
                                 'external_camera': _image()})
    recorder.save()
    recorder.flush()
    assert exported == []
    assert recorder.unfinished == 1
    attempted = recorder.process_pending()
    assert attempted == 1
    assert len(exported) == 1
    recorder.close()


def test_recorder_deferred_export_on_close(tmp_path):
    exported = []
    recorder = EpisodeRecorder(tmp_path, _schema(),
                               export=lambda take, root: exported.append(take.name),
                               export_mode='deferred')
    recorder.start('deferred close')
    for _ in range(3):
        recorder.capture(np.zeros(7), np.zeros(7),
                         images={'wrist_camera': _image(),
                                 'external_camera': _image()})
    recorder.save()
    recorder.close()
    assert len(exported) == 1


def test_manifest_export(tmp_path):
    recorder = EpisodeRecorder(tmp_path, _schema(), export=export_manifest)
    recorder.start('manifest task')
    for index in range(3):
        recorder.capture(np.zeros(7), np.zeros(7),
                         images={'wrist_camera': _image(),
                                 'external_camera': _image()})
    recorder.save()
    recorder.close()
    root = recorder.dataset_root
    lines = (root / 'episodes.jsonl').read_text(encoding='utf-8').strip().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry['frames'] == 3
    assert entry['task'] == 'manifest task'
    assert (root / 'schema.json').exists()
    assert (root / 'capture_contract.json').exists()


def test_lerobot_export_requires_lerobot(tmp_path):
    import importlib.util
    if importlib.util.find_spec('lerobot') is None:
        with pytest.raises(ImportError):
            export_lerobot(tmp_path / 'missing', tmp_path / 'dataset')
    else:
        pytest.skip('lerobot is installed')


def test_split_camera_frames_with_mock_camera():
    camera = MockCamera('wrist_camera', 32, 24, depth=True).open()
    camera.T_world_camera = np.eye(4)
    frame = camera.read()
    images, depth = split_camera_frames({'wrist_camera': frame})
    assert images['wrist_camera'].shape == (24, 32, 3)
    assert depth['wrist_camera'].dtype == np.uint16
    assert depth['wrist_camera_K'].shape == (3, 3)
    assert depth['wrist_camera_T_world_camera'].shape == (4, 4)
    camera.close()


def test_master_slave_sampling_writes_episode(tmp_path):
    from robot_toolkit.examples import piper_collect

    exit_code = piper_collect.main([
        '--source', 'master-slave', '--mock', '--auto-record',
        '--sample-seconds', '0.4', '--fps', '10',
        '--dataset-root', str(tmp_path), '--repo-id', 'local/ms_test',
        '--export', 'manifest',
    ])
    assert exit_code == 0
    takes = list(Path(tmp_path).rglob('take.json'))
    assert len(takes) == 1
    frames = sorted(takes[0].parent.glob('*.npz'))
    assert len(frames) >= 2


def test_capture_state_uses_reported_gripper_width(tmp_path):
    from types import SimpleNamespace

    schema = DatasetSchema.for_robot(RobotSpec.piper_x(), cameras=(), fps=10)
    collector = EpisodeCollector(tmp_path, schema, robot=SimpleNamespace(),
                                 export=export_manifest, fps=10,
                                 min_interval_s=0.0)
    collector.start('gripper width')
    for _ in range(2):
        state = SimpleNamespace(q=np.zeros(6), extra={'gripper_width': 0.03}, t=0.0)
        assert collector.capture_state(state=state)
    collector.stop(success=True)
    collector.close()
    path = sorted(Path(tmp_path).rglob('*.npz'))[0]
    with np.load(path) as data:
        assert data['observation_state'][-1] == pytest.approx(0.03)
        assert data['action'][-1] == pytest.approx(0.03)


def test_schema_feature_name_styles():
    unit = DatasetSchema.for_robot(RobotSpec.piper_x(), cameras=(), fps=30)
    assert unit.state_names[0] == 'joint_1.rad'
    assert unit.state_names[-1] == 'piper_gripper.width_m'

    pos = DatasetSchema.for_robot(RobotSpec.piper_x(), cameras=(), fps=30,
                                  name_style='pos')
    assert pos.state_names == [f'joint_{index}.pos' for index in range(1, 7)] + ['gripper.pos']
    # Labels only: feature shape and order are identical to the unit style.
    assert pos.dim == unit.dim
    assert pos.features()['action']['shape'] == unit.features()['action']['shape']

    with pytest.raises(DatasetError):
        DatasetSchema.for_robot(RobotSpec.piper_x(), cameras=(), name_style='bogus')


def _fake_lerobot(monkeypatch, store):
    """A minimal LeRobotDataset that records add_frame calls (no video encoding)."""
    import sys
    import types

    class Dataset:
        def __init__(self, repo_id, root=None, **kwargs):
            self.root, self.frames, self.num_episodes = Path(root), [], 0
            store.append(self)

        @classmethod
        def create(cls, repo_id, **kwargs):
            return cls(repo_id, root=kwargs['root'])

        def add_frame(self, frame):
            self.frames.append(frame)

        def save_episode(self, **kwargs):
            self.root.mkdir(parents=True, exist_ok=True)

        def finalize(self):
            pass

        def __len__(self):
            return len(self.frames)

    module = types.ModuleType('lerobot.datasets.lerobot_dataset')
    module.LeRobotDataset = Dataset
    for name in ('lerobot', 'lerobot.datasets'):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, 'lerobot.datasets.lerobot_dataset', module)


def test_fast_rows_store_each_camera_frame_once(tmp_path, monkeypatch):
    # 100 Hz rows with a 30 fps camera: every image is spooled once, rows that
    # show the same frame reference it, and the export still has one image per row.
    schema = DatasetSchema(robot=RobotSpec.ur5e(), cameras=(CameraSpec('wrist_camera', 32, 24),),
                           fps=100)
    recorder = EpisodeRecorder(tmp_path, schema, export_mode='deferred', export=None)
    recorder.start('fast rows')
    for row in range(10):
        frame_id = row // 3                       # a new camera frame every 3 rows
        image = np.full((24, 32, 3), frame_id, dtype=np.uint8)
        depth = {'wrist_camera': np.full((24, 32), 100 + frame_id, np.uint16),
                 'wrist_camera_K': np.eye(3)}
        assert recorder.capture(np.full(7, row), np.full(7, row), images={'wrist_camera': image},
                                depth=depth, image_ids={'wrist_camera': frame_id})
    take = recorder.take
    recorder.save(success=True)
    recorder.flush()
    files = sorted(take.glob('0*.npz'))
    stored = [index for index, file in enumerate(files)
              if 'wrist_camera_rgb' in np.load(file).files]
    assert stored == [0, 3, 6, 9]                 # 4 camera frames, not 10 images
    recorder.close()

    store = []
    _fake_lerobot(monkeypatch, store)
    export_lerobot(take, tmp_path / 'dataset', validate=False)
    frames = store[0].frames
    assert len(frames) == 10
    for row, frame in enumerate(frames):
        assert frame['observation.images.wrist_camera'][0, 0, 0] == row // 3
        assert frame['action'][0] == row
    episode = json.loads(next((tmp_path / 'dataset').glob('depth/episode_*/episode.json'))
                         .read_text())
    assert episode['depth_rows'] == [0, 0, 0, 3, 3, 3, 6, 6, 6, 9]
    for row in (0, 3, 6, 9):
        depth_file = next((tmp_path / 'dataset').glob(f'depth/episode_*/{row:06d}.npz'))
        assert np.load(depth_file)['wrist_camera_depth_mm'][0, 0] == 100 + row // 3


def test_collector_tags_frames_by_identity(tmp_path):
    camera = MockCamera('wrist_camera', width=32, height=24)
    schema = DatasetSchema.for_robot(RobotSpec.piper_x(), cameras=(camera,), fps=100)
    robot = MockRobot()
    robot.connect()
    collector = EpisodeCollector(tmp_path, schema, robot=robot, cameras=(camera,), export=None)
    collector.start('ids')
    images, _, ids = collector._camera_payload()
    assert set(ids) == {'wrist_camera'} and ids['wrist_camera'] > 0
    collector.recorder.discard()
    collector.recorder.close()
