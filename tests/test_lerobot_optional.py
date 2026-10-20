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
