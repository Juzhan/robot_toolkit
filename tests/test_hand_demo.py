"""Vision-only human demonstration collection tests (same pipeline as robot collection)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from robot_toolkit.camera import MockCamera
from robot_toolkit.collect import (CameraSpec, DatasetSchema, EpisodeCollector,
                                   export_manifest, record_hand_demos)
from robot_toolkit.core.conventions import RobotSpec


class _IdleKeys:
    def get(self):
        return None

    def stop(self):
        pass


def _silent(*args, **kwargs):
    pass


def _schema(fps: int = 20) -> DatasetSchema:
    return DatasetSchema.for_robot(
        RobotSpec(name='hand_demo', joint_names=(), gripper=None),
        cameras=(CameraSpec('front', 32, 24),), fps=fps,
        repo_id='local/test_hand', include_state_action=False)


def _manifest(root: Path) -> dict:
    manifest = Path(root) / 'episodes.jsonl'
    lines = [line for line in manifest.read_text(encoding='utf-8').splitlines() if line]
    assert len(lines) == 1
    return json.loads(lines[0])


def _spooled_frames(root: Path) -> list[Path]:
    return sorted(Path(root).parent.glob('take_*/*.npz'))


def test_hand_demo_records_vision_only_dataset(tmp_path):
    stats = record_hand_demos(
        [MockCamera('front', 32, 24)], tmp_path / 'session', _schema(),
        fps=20.0, task='hand pick', auto_record=True, sample_seconds=0.4,
        export=export_manifest, keys=_IdleKeys(), printer=_silent)
    assert stats['saved'] == 1
    assert stats['episodes'] == 1
    assert stats['frames'] >= 2
    root = Path(stats['dataset_root'])
    entry = _manifest(root)
    assert entry['task'] == 'hand pick'
    assert entry['cameras'] == ['front']
    assert entry['state_names'] == []
    assert entry['frames'] == stats['frames']
    contract = json.loads((root / 'capture_contract.json').read_text(encoding='utf-8'))
    assert contract['modalities'] == ['cameras']
    assert contract['state_action'] is None
    frames = _spooled_frames(root)
    assert frames
    with np.load(frames[0]) as raw:
        assert 'front_rgb' in raw.files
        assert 'observation_state' not in raw.files
        assert 'action' not in raw.files
    take = next(root.parent.glob('take_*/take.json'))
    assert json.loads(take.read_text(encoding='utf-8'))['status'] == 'saved'


def test_hand_demo_tracker_payload_is_spooled_as_extra(tmp_path):
    class Tracker:
        name = 'test_tracker'

        def __init__(self):
            self.calls = 0

        def read(self, timestamp, images):
            self.calls += 1
            return {'points': [[0.1, 0.2], [0.3, 0.4]], 'call': self.calls}

    tracker = Tracker()
    stats = record_hand_demos(
        [MockCamera('front', 32, 24)], tmp_path / 'session', _schema(),
        fps=20.0, task='hand track', auto_record=True, sample_seconds=0.4,
        export=export_manifest, tracker=tracker, keys=_IdleKeys(),
        printer=_silent)
    assert stats['saved'] == 1
    assert tracker.calls >= 2
    frames = _spooled_frames(Path(stats['dataset_root']))
    with np.load(frames[0]) as raw:
        assert 'tracker_points' in raw.files
        assert 'tracker_call' in raw.files


def test_hand_demo_max_frames_bounds_one_episode(tmp_path):
    stats = record_hand_demos(
        [MockCamera('front', 32, 24)], tmp_path / 'session', _schema(30),
        fps=30.0, task='bounded', auto_record=True, max_frames=3,
        export=export_manifest, keys=_IdleKeys(), printer=_silent)
    assert stats['saved'] == 1
    assert stats['frames'] == 3


def test_hand_demo_announces_start_save_and_end(tmp_path):
    cues = []

    def announce(text, preempt=False):
        cues.append(text)

    stats = record_hand_demos(
        [MockCamera('front', 32, 24)], tmp_path / 'session', _schema(),
        fps=20.0, task='voice', auto_record=True, sample_seconds=0.4,
        export=export_manifest, announce=announce, keys=_IdleKeys(),
        printer=_silent)
    assert stats['saved'] == 1
    assert cues[0] == '开始录制'
    assert any('已保存' in cue or '录制完成' in cue for cue in cues)
    assert cues[-1] == '采集结束'


def test_episode_collector_stop_is_idempotent_and_reusable(tmp_path):
    collector = EpisodeCollector(
        tmp_path / 'session', _schema(10),
        cameras=(MockCamera('front', 32, 24),),
        export=export_manifest, fps=10)
    try:
        for _ in range(2):
            collector.start('test')
            for _ in range(3):
                assert collector.capture_frame()
            collector.stop(success=True)
            collector.stop(success=True)   # must be a no-op
    finally:
        collector.close()
    assert collector.saved == 2


@pytest.mark.skipif(importlib.util.find_spec('lerobot') is None,
                    reason='lerobot is not installed')
def test_delete_last_episode_on_lerobot(tmp_path):
    from robot_toolkit.collect import delete_last_episode, export_lerobot

    collector = EpisodeCollector(
        tmp_path / 'session', _schema(10), cameras=(MockCamera('front', 32, 24),),
        export=export_lerobot, fps=10)
    try:
        for _ in range(2):
            collector.start('delete me')
            for _ in range(3):
                assert collector.capture_frame()
            collector.stop(success=True)
    finally:
        collector.close()
    root = collector.dataset_root
    assert json.loads((root / 'meta' / 'info.json').read_text(encoding='utf-8'))[
        'total_episodes'] == 2
    removed = delete_last_episode(root)
    assert removed == {'episode': 1, 'remaining': 1}
    assert json.loads((root / 'meta' / 'info.json').read_text(encoding='utf-8'))[
        'total_episodes'] == 1


@pytest.mark.skipif(importlib.util.find_spec('lerobot') is None,
                    reason='lerobot is not installed')
def test_lerobot_delete_preserves_depth_sidecar(tmp_path):
    from robot_toolkit.collect import delete_last_episode, export_lerobot

    schema = DatasetSchema.for_robot(
        RobotSpec(name='hand_demo', joint_names=(), gripper=None),
        cameras=(CameraSpec('front', 32, 24, depth=True),), fps=10,
        repo_id='local/test_depth', include_state_action=False)
    collector = EpisodeCollector(
        tmp_path / 'session', schema,
        cameras=(MockCamera('front', 32, 24, depth=True),),
        export=export_lerobot, fps=10)
    try:
        for _ in range(2):
            collector.start('depth task')
            for _ in range(3):
                assert collector.capture_frame()
            collector.stop(success=True)
    finally:
        collector.close()
    root = collector.dataset_root
    assert (root / 'depth' / 'episode_000000' / '000000.npz').exists()
    assert (root / 'depth' / 'episode_000001' / '000000.npz').exists()
    removed = delete_last_episode(root)
    assert removed == {'episode': 1, 'remaining': 1}
    assert (root / 'depth' / 'episode_000000' / '000000.npz').exists()
    assert not (root / 'depth' / 'episode_000001').exists()


def test_hand_demo_mode_is_config_driven(tmp_path):
    from robot_toolkit.task_runner import execute, load_task_config

    payload = {
        'slug': 'hand_task',
        'task': 'hand task',
        'mode': 'hand_demo',
        'root': str(tmp_path),
        'collect': {
            'fps': 20.0,
            'export': 'manifest',
            'record_state_action': False,
            'cameras': [{
                'kind': 'opencv', 'name': 'front', 'index': 0,
                'width': 32, 'height': 24, 'fps': 20,
            }],
        },
        'hand_demo': {
            'fps': 20.0, 'auto_record': True, 'sample_seconds': 0.4,
            'tracker': '', 'input': 'keyboard',
        },
    }
    path = tmp_path / 'hand_task.json'
    path.write_text(json.dumps(payload), encoding='utf-8')
    config = load_task_config(str(path))
    assert execute(config, mode='hand_demo', mock=True) == 0
    roots = list((tmp_path / 'hand_task' / 'dataset').glob('*/dataset/episodes.jsonl'))
    assert roots, 'hand_demo did not write an episode index'
    entry = json.loads(roots[0].read_text(encoding='utf-8').splitlines()[0])
    assert entry['state_names'] == []


def test_joycon_keys_map_to_record_actions():
    import queue

    from robot_toolkit.task_runner import _JoyConKeys

    class Hub:
        def __init__(self):
            self.events = queue.Queue()

    hub = Hub()
    keys = _JoyConKeys(hub)
    keys.sync(False)
    hub.events.put(('b', 0.0))
    hub.events.put(('a', 0.0))
    assert keys.get() == ' '       # B starts while idle
    assert keys.get() is None      # A does nothing while idle
    keys.sync(True)
    hub.events.put(('b', 0.0))
    hub.events.put(('a', 0.0))
    assert keys.get() is None      # B does nothing while recording
    assert keys.get() == ' '       # A saves while recording
    for button in ('x', 'r'):
        hub.events.put((button, 0.0))
    assert keys.get() is None
    assert keys.get() is None
    hub.events.put(('y', 0.0))
    hub.events.put(('plus', 0.0))
    hub.events.put(('home', 0.0))
    hub.events.put(('minus', 0.0))
    assert keys.get() == 'cancel'
    assert keys.get() == 'delete'
    assert keys.get() == 'status'
    assert keys.get() == 'q'


def test_hand_demo_rejects_unknown_input(tmp_path):
    from robot_toolkit.task_runner import load_task_config

    payload = {
        'slug': 'hand_task', 'task': 'hand task', 'mode': 'hand_demo',
        'root': str(tmp_path),
        'collect': {'record_state_action': False,
                    'cameras': [{'kind': 'opencv', 'name': 'front'}]},
        'hand_demo': {'input': 'gamepad'},
    }
    path = tmp_path / 'bad_input.json'
    path.write_text(json.dumps(payload), encoding='utf-8')
    with pytest.raises(ValueError):
        load_task_config(str(path))

