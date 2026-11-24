"""Optional LeRobot integration test; skipped when LeRobot is not installed.

Run it with the environment that actually exports datasets (the one where
``import lerobot`` works)::

    python -m pytest tests/test_lerobot_optional.py -q
"""

from __future__ import annotations

import numpy as np
import pytest

lerobot = pytest.importorskip('lerobot')

from robot_toolkit.collect import (CameraSpec, DatasetSchema, EpisodeRecorder,
                                   export_lerobot)
from robot_toolkit.core.conventions import RobotSpec


def test_robot_only_export(tmp_path):
    schema = DatasetSchema.for_robot(RobotSpec.piper_x(), cameras=(), fps=10,
                                     repo_id='local/robot_toolkit_robot_only_test')
    recorder = EpisodeRecorder(tmp_path, schema, export=export_lerobot)
    recorder.start('robot only')
    for frame in range(2):
        recorder.capture(np.full(schema.dim, 0.1), np.full(schema.dim, 0.1),
                         timestamp=frame / schema.fps)
    recorder.save(success=True)
    recorder.close()
    assert recorder.export_errors == [], recorder.export_errors
    assert recorder.saved == 1
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    dataset = LeRobotDataset('local/robot_toolkit_robot_only_test',
                             root=recorder.dataset_root)
    assert dataset.num_episodes == 1
    sample = dataset[0]
    assert tuple(sample['action'].shape) == (schema.dim,)
    assert not any(key.startswith('observation.images') for key in sample)


def _episode(recorder, schema, index):
    rng = np.random.default_rng(index)
    recorder.start(f'episode {index}')
    for frame in range(4):
        images = {'wrist_camera': rng.integers(0, 255, (24, 32, 3), dtype=np.uint8)}
        recorder.capture(np.full(schema.dim, index * 0.1),
                         np.full(schema.dim, index * 0.1),
                         images=images, timestamp=frame / schema.fps)
    recorder.save(success=True)


def test_single_and_appended_episode(tmp_path):
    schema = DatasetSchema(robot=RobotSpec.ur5e(),
                           cameras=(CameraSpec('wrist_camera', 32, 24),),
                           fps=10, repo_id='local/robot_toolkit_optional_test')
    # One recorder owns one dataset session; repeated save() calls append
    # episodes to that same dataset.  Two different EpisodeRecorder instances
    # intentionally create two timestamped sessions.
    recorder = EpisodeRecorder(tmp_path, schema, export=export_lerobot)
    _episode(recorder, schema, 0)
    _episode(recorder, schema, 1)
    recorder.close()
    assert recorder.export_errors == [], recorder.export_errors
    assert recorder.saved == 2

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    dataset = LeRobotDataset('local/robot_toolkit_optional_test',
                             root=recorder.dataset_root)
    assert dataset.num_episodes == 2
    sample = dataset[0]
    assert tuple(sample['action'].shape) == (schema.dim,)
    assert tuple(sample['observation.images.wrist_camera'].shape) == (3, 24, 32)
    depth = recorder.dataset_root / 'depth' / 'episode_000001' / 'episode.json'
    assert depth.exists()


def test_merge_keeps_files_and_realigns_indices(tmp_path):
    import json
    from robot_toolkit.collect.merge import merge_datasets, verify_merge
    schema = DatasetSchema(robot=RobotSpec.ur5e(),
                           cameras=(CameraSpec('wrist_camera', 32, 24),),
                           fps=10, repo_id='local/robot_toolkit_merge_test')
    roots = []
    for batch, episodes in ((0, (0, 1)), (1, (2,))):
        recorder = EpisodeRecorder(tmp_path / f'batch{batch}', schema, export=export_lerobot)
        for index in episodes:
            _episode(recorder, schema, index)
        recorder.close()
        assert recorder.export_errors == []
        roots.append(recorder.dataset_root)
    out = tmp_path / 'merged'
    assert merge_datasets(roots, out)['episodes'] == 3
    report = verify_merge(roots, out)
    assert report['checked_rows'] == 12 and report['depth_ok'] == 3
    moved = json.loads((out / 'depth' / 'episode_000002' / 'episode.json').read_text())
    assert moved['episode_index'] == 2 and moved['merged_from']['episode_index'] == 0
    with pytest.raises(FileExistsError):
        merge_datasets(roots, out)

    from robot_toolkit.collect.dataset_edit import delete_last
    from robot_toolkit.collect.merge import _pointer_errors
    assert _pointer_errors(out) == 0
    assert delete_last(out) == {'deleted': 2, 'episodes': 2}
    assert sorted(p.name for p in (out / 'depth').glob('episode_*')) == [
        'episode_000000', 'episode_000001']


def test_sessions_append_to_one_dataset_through_child_process(tmp_path):
    from robot_toolkit.collect.export import isolated_exporter
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    schema = DatasetSchema(robot=RobotSpec.ur5e(),
                           cameras=(CameraSpec('wrist_camera', 32, 24),),
                           fps=10, repo_id='local/robot_toolkit_shared_test')
    shared = tmp_path / 'lerobot'
    for session, index in enumerate((0, 1)):
        recorder = EpisodeRecorder(tmp_path / 'raw', schema, dataset_root=shared,
                                   export=isolated_exporter('lerobot'))
        _episode(recorder, schema, index)
        recorder.close()
        assert recorder.export_errors == [], recorder.export_errors
        assert recorder.dataset_root == shared
    dataset = LeRobotDataset('local/robot_toolkit_shared_test', root=shared)
    assert dataset.num_episodes == 2
    assert sorted(p.name for p in (shared / 'depth').glob('episode_*')) == [
        'episode_000000', 'episode_000001']
    assert len(list((tmp_path / 'raw').glob('*/take_*/export.log'))) == 2


def test_collector_writes_per_frame_timestamps(tmp_path):
    import json
    from types import SimpleNamespace
    from robot_toolkit.collect import EpisodeCollector
    from robot_toolkit.profiles import build_camera
    camera = build_camera('mock', name='wrist_camera', width=32, height=24, fps=30, depth=True)
    schema = DatasetSchema(robot=RobotSpec.ur5e(), cameras=(CameraSpec.from_camera(camera),),
                           fps=10, repo_id='local/robot_toolkit_timing_test')
    robot = SimpleNamespace(state=lambda: None)
    collector = EpisodeCollector(tmp_path, schema, robot=robot, cameras=[camera],
                                 export=export_lerobot, fps=10)
    collector.start('timing')
    for step in range(3):
        state = SimpleNamespace(q=np.full(6, 0.1 * step), t=float(step), tcp=np.arange(6.0),
                                extra={'gripper_width': 0.02})
        assert collector.capture_state(state=state)
    collector.stop(success=True)
    collector.close()
    assert collector.export_errors == [], collector.export_errors
    stamps = np.load(collector.dataset_root / 'timestamps' / 'episode_000000.npz')
    assert stamps['time_robot_state'].tolist() == [0.0, 1.0, 2.0]
    assert stamps['time_robot_tcp'].shape == (3, 6)
    assert 'time_wrist_camera_color_host' in stamps.files
    summary = json.loads((collector.dataset_root / 'timestamps' / 'episode_000000.json').read_text())
    assert summary['available'] and summary['rows'] == 3 and 'row_gap_ms' in summary
