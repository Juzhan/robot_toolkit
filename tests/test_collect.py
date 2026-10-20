"""Dataset contract, async recorder and exporter tests."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from robot_toolkit.camera import MockCamera
from robot_toolkit.collect import (CameraSpec, DatasetSchema, EpisodeCollector,
                                   EpisodeRecorder, collect_trajectory,
                                   export_lerobot, export_manifest,
                                   split_camera_frames)
from robot_toolkit.core.conventions import RobotSpec
from robot_toolkit.errors import DatasetError
from robot_toolkit.robot import MockRobot
from robot_toolkit.teach import Trajectory


def _schema(fps=10):
    return DatasetSchema(robot=RobotSpec.ur5e(),
                         cameras=(CameraSpec('wrist_camera', 32, 24),
                                  CameraSpec('external_camera', 32, 24)),
                         fps=fps)


def _image():
    return np.zeros((24, 32, 3), dtype=np.uint8)


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
