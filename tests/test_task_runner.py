"""Config-driven Piper task runner tests (no CAN, no camera hardware)."""
from __future__ import annotations

import json
import queue

import numpy as np

import robot_toolkit.task_runner as task_runner
from robot_toolkit.task_runner import (dataset_root, execute,
                                       load_task_config,
                                       trajectory_root)
from robot_toolkit.teach import Trajectory, save_trajectory


def _write_config(path, root):
    payload = {
        'slug': 'unit_task',
        'task': 'unit task',
        'mode': 'collect',
        'root': str(root),
        'robot': {
            'kind': 'agx_piper', 'name': 'piper_x',
            'options': {'model': 'piper_x', 'speed_percent': 20},
            'limits': {'max_velocity': 1.5}, 'safety': True,
        },
        'gripper': {'kind': 'agx'},
        'teach': {'rate': 50.0, 'gripper_close_time_s': 0.2,
                 'gripper_open_time_s': 0.2},
        'replay': {'rate': 50.0, 'speed_scale': 1.0, 'smoothing': 3,
                   'realtime': False},
        'collect': {'repo_id': 'local/unit_task', 'fps': 10.0,
                    'export': 'manifest', 'cameras': []},
    }
    path.write_text(json.dumps(payload), encoding='utf-8')
    return path


def test_task_config_resolves_task_paths(tmp_path):
    config = load_task_config(str(_write_config(tmp_path / 'unit_task.json',
                                                 tmp_path)))
    assert config.slug == 'unit_task'
    assert config.source_path.endswith('unit_task.json')
    assert trajectory_root(config) == tmp_path / 'unit_task' / 'trajectories'
    assert dataset_root(config) == tmp_path / 'unit_task' / 'dataset'


def _write_trajectory(config, name='unit_01'):
    trajectory_dir = trajectory_root(config) / name
    samples = 30
    t = np.arange(samples, dtype=float) / 50.0
    q = np.stack([
        np.linspace(0.0, 0.08, samples),
        np.linspace(0.5, 0.58, samples),
        np.linspace(-0.5, -0.58, samples),
        np.zeros(samples), np.linspace(0.5, 0.54, samples), np.zeros(samples),
    ], axis=1)
    save_trajectory(trajectory_dir,
                    Trajectory(t=t, q=q, grip=np.full(samples, 0.06)),
                    name='trajectory')
    return name


def test_collect_mode_replays_and_writes_dataset(tmp_path):
    config = load_task_config(str(_write_config(tmp_path / 'unit_task.json',
                                                 tmp_path)))
    name = _write_trajectory(config)

    code = execute(config, mode='collect', name=name, mock=True, yes=True)
    assert code == 0
    dataset = dataset_root(config)
    takes = list(dataset.glob('*/dataset/episodes.jsonl'))
    assert takes, f'no manifest under {dataset}'
    assert json.loads(takes[0].read_text(encoding='utf-8').splitlines()[0])['task'] == 'unit task'


def test_teach_mode_saves_a_trajectory(monkeypatch, tmp_path):
    config = load_task_config(str(_write_config(tmp_path / 'unit_task.json',
                                                 tmp_path)))
    samples = 20
    t = np.arange(samples, dtype=float) / 50.0
    q = np.stack([np.linspace(0.0, 0.05, samples), np.full(samples, 0.5),
                  np.full(samples, -0.5), np.zeros(samples),
                  np.full(samples, 0.5), np.zeros(samples)], axis=1)
    trajectory = Trajectory(t=t, q=q, grip=np.full(samples, 0.06))
    monkeypatch.setattr(task_runner, 'keyboard_teach',
                        lambda *args, **kwargs: trajectory)

    assert execute(config, mode='teach', name='unit_teach', mock=True) == 0
    assert (trajectory_root(config) / 'unit_teach' / 'trajectory.json').exists()


def test_export_mode_does_not_need_a_connection(tmp_path):
    config = load_task_config(str(_write_config(tmp_path / 'unit_task.json',
                                                 tmp_path)))
    name = _write_trajectory(config)
    assert execute(config, mode='export', name=name) == 0
    assert list(dataset_root(config).glob('*/dataset/episodes.jsonl'))


def test_collect_mode_is_robot_agnostic(tmp_path):
    """The same runner should use UR5e when the config says so."""
    path = _write_config(tmp_path / 'unit_task.json', tmp_path)
    payload = json.loads(path.read_text(encoding='utf-8'))
    payload['robot'] = {'kind': 'ur_rtde', 'name': 'ur5e',
                        'options': {'ip': '127.0.0.1'},
                        'limits': {'max_velocity': 1.5}, 'safety': True}
    payload['gripper'] = {'kind': 'robotiq',
                          'options': {'backend': 'urcap_socket'}}
    payload['collect']['cameras'] = [{
        'kind': 'realsense', 'name': 'wrist_camera', 'serial': 'not-connected',
        'width': 64, 'height': 48, 'fps': 30, 'depth': True,
    }]
    path.write_text(json.dumps(payload), encoding='utf-8')
    config = load_task_config(str(path))
    name = _write_trajectory(config)

    assert execute(config, mode='collect', name=name, mock=True, yes=True) == 0


class _FakeEventQueue:
    def __init__(self, events):
        self._events = list(events)

    def get(self, timeout=None):
        if self._events:
            return self._events.pop(0)
        raise KeyboardInterrupt

    def get_nowait(self):
        if self._events:
            return self._events.pop(0)
        raise queue.Empty

    def __bool__(self):
        return bool(self._events)


class _FakeHub:
    def __init__(self, events):
        self.events = _FakeEventQueue(events)
        self.stopped = False

    def start(self):
        return self

    def stop(self):
        self.stopped = True


def test_session_loops_teach_then_replay_collect(monkeypatch, tmp_path):
    path = _write_config(tmp_path / 'unit_task.json', tmp_path)
    payload = json.loads(path.read_text(encoding='utf-8'))
    payload['mode'] = 'session'
    payload['session'] = {'prefix': 'unit', 'confirm_replay': True,
                          'double_press_timeout_s': 5.0}
    path.write_text(json.dumps(payload), encoding='utf-8')
    config = load_task_config(str(path))
    trajectory = Trajectory(
        t=np.arange(20, dtype=float) / 50.0,
        q=np.stack([np.linspace(0.0, 0.05, 20), np.full(20, 0.5),
                    np.full(20, -0.5), np.zeros(20), np.full(20, 0.5),
                    np.zeros(20)], axis=1),
        grip=np.full(20, 0.06))
    fake_hub = _FakeHub([('b', 0.0)])

    def fake_teach(*args, **kwargs):
        fake_hub.events._events.extend([('x', 0.1), ('x', 0.2)])
        return trajectory

    monkeypatch.setattr(task_runner, 'joycon_teach', fake_teach)
    monkeypatch.setattr(task_runner, 'ButtonHub', lambda: fake_hub)

    assert task_runner.run_session(config, mock=True) == 0
    assert fake_hub.stopped
    assert (trajectory_root(config) / 'unit_01' / 'trajectory.json').exists()
    assert list(dataset_root(config).glob('*/dataset/episodes.jsonl'))


def test_collect_mode_can_capture_a_mock_camera(tmp_path):
    path = _write_config(tmp_path / 'unit_task.json', tmp_path)
    payload = json.loads(path.read_text(encoding='utf-8'))
    payload['collect']['cameras'] = [{
        'kind': 'mock', 'name': 'wrist_camera',
        'width': 64, 'height': 48, 'fps': 30, 'depth': False,
    }]
    path.write_text(json.dumps(payload), encoding='utf-8')
    config = load_task_config(str(path))
    name = _write_trajectory(config)

    assert execute(config, mode='collect', name=name, mock=True, yes=True) == 0
