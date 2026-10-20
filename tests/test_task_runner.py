"""Config-driven Piper task runner tests (no CAN, no camera hardware)."""
from __future__ import annotations

import json
import queue

import numpy as np
import pytest

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


def _write_master_slave_config(path, root):
    payload = {
        'slug': 'ms_task',
        'task': 'master slave task',
        'mode': 'master_slave',
        'root': str(root),
        'robot': {'kind': 'agx_piper', 'name': 'piper_x',
                  'options': {'model': 'piper_x', 'enable_on_connect': False,
                              'configure_on_connect': False},
                  'safety': False},
        'gripper': {'kind': 'none'},
        'collect': {'repo_id': 'local/ms_task', 'fps': 10.0, 'export': 'manifest',
                    'cameras': [{'kind': 'mock', 'name': 'wrist_camera',
                                 'width': 32, 'height': 24, 'fps': 30}]},
        'master_slave': {'auto_record': True, 'sample_seconds': 0.3, 'fps': 10.0},
    }
    path.write_text(json.dumps(payload), encoding='utf-8')
    return path


def test_feature_names_option_is_validated(tmp_path):
    path = _write_master_slave_config(tmp_path / 'ms3.json', tmp_path / 'out')
    payload = json.loads(path.read_text(encoding='utf-8'))
    payload['collect']['feature_names'] = 'POS'
    path.write_text(json.dumps(payload), encoding='utf-8')
    assert load_task_config(str(path)).collect.feature_names == 'pos'

    payload['collect']['feature_names'] = 'whatever'
    path.write_text(json.dumps(payload), encoding='utf-8')
    with pytest.raises(ValueError):
        load_task_config(str(path))


def test_master_slave_mode_samples_from_config(tmp_path):
    config = load_task_config(str(_write_master_slave_config(tmp_path / 'ms.json', tmp_path)))
    assert config.mode == 'master_slave'
    assert config.master_slave.auto_record is True

    assert execute(config, mock=True) == 0
    takes = list(dataset_root(config).rglob('take.json'))
    assert len(takes) == 1
    assert len(list(takes[0].parent.glob('*.npz'))) >= 2


def test_repo_ships_demo_config(tmp_path, monkeypatch):
    # A fresh clone has no workspace configs; the packaged demo must still load.
    monkeypatch.setattr(task_runner, 'ROOT', tmp_path)
    task = load_task_config('piper_master_slave')
    assert task.mode == 'master_slave'
    assert task.robot['options']['configure_on_connect'] is False
    assert task.master_slave.sample_seconds == 0.0
    assert task.master_slave.auto_record is True
    assert task.collect.max_frames == 0

    ur5 = load_task_config('ur5e_session')
    assert ur5.mode == 'session'
    assert ur5.robot['kind'] == 'ur_rtde'
    assert ur5.gripper['kind'] == 'robotiq'


def test_master_slave_duration_starts_recording(tmp_path):
    path = _write_master_slave_config(tmp_path / 'ms2.json', tmp_path / 'out')
    payload = json.loads(path.read_text(encoding='utf-8'))
    # No auto_record flag: a configured duration is enough to go hands-free.
    payload['master_slave'] = {'auto_record': False, 'sample_seconds': 0.3, 'fps': 10.0}
    path.write_text(json.dumps(payload), encoding='utf-8')
    config = load_task_config(str(path))

    assert execute(config, mock=True) == 0
    assert list(dataset_root(config).rglob('take.json'))


def _write_calibrate_config(path, root, cameras=None):
    payload = {
        'slug': 'cal_task',
        'task': 'calibrate task',
        'mode': 'calibrate',
        'root': str(root),
        'robot': {'kind': 'agx_piper', 'name': 'piper_x',
                  'options': {'model': 'piper_x', 'enable_on_connect': False,
                              'configure_on_connect': False},
                  'safety': False},
        'gripper': {'kind': 'none'},
        'collect': {'repo_id': 'local/cal_task', 'fps': 30.0, 'export': 'manifest',
                    'cameras': cameras if cameras is not None else []},
        'calibrate': {'kind': 'handeye', 'eye': 'eye_in_hand',
                      'camera': 'wrist_camera', 'pattern': [9, 6],
                      'square_size': 0.025, 'min_samples': 8},
    }
    path.write_text(json.dumps(payload), encoding='utf-8')
    return path


def test_calibrate_mode_requires_a_camera(tmp_path):
    config = load_task_config(str(_write_calibrate_config(tmp_path / 'cal.json', tmp_path)))
    assert config.mode == 'calibrate'
    assert config.calibrate.kind == 'handeye'
    # No camera in the profile: the mode must refuse instead of failing later.
    assert execute(config, mock=True) == 2


def test_calibrate_config_validation(tmp_path):
    path = _write_calibrate_config(tmp_path / 'cal2.json', tmp_path)
    payload = json.loads(path.read_text(encoding='utf-8'))
    payload['calibrate']['kind'] = 'nope'
    path.write_text(json.dumps(payload), encoding='utf-8')
    with pytest.raises(ValueError):
        load_task_config(str(path))

    payload['calibrate']['kind'] = 'intrinsics'
    payload['calibrate']['eye'] = 'eye_to_hand'
    path.write_text(json.dumps(payload), encoding='utf-8')
    task = load_task_config(str(path))
    assert task.calibrate.kind == 'intrinsics'
    assert execute(task, mode='calibrate', mock=True, dry_run=True) == 0


def test_auto_calibration_requires_confirmation(tmp_path):
    path = _write_calibrate_config(
        tmp_path / 'cal_auto.json', tmp_path,
        cameras=[{'kind': 'mock', 'name': 'wrist_camera', 'width': 32,
                  'height': 24, 'fps': 30}])
    payload = json.loads(path.read_text(encoding='utf-8'))
    payload['calibrate'].update({'auto': True, 'pose_count': 3, 'settle_s': 0.0})
    path.write_text(json.dumps(payload), encoding='utf-8')
    config = load_task_config(str(path))

    # Moving the arm needs an explicit --yes; without it the mode refuses.
    assert execute(config, mock=True) == 2
    # With --yes it runs the sweep and reports the missing samples instead of
    # crashing (mock camera sees no board and mock robot has no TCP).
    assert execute(config, mock=True, yes=True) == 1
