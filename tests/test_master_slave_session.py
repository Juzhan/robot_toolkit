"""Hardware-free tests of the master-slave button session and shared input."""
import json
import queue
import time

import numpy as np

from robot_toolkit import task_runner
from robot_toolkit.task_runner import execute, load_task_config
from robot_toolkit.teleop import session_input
from robot_toolkit.teleop.session_input import DoublePress, SessionInput


def _config(tmp_path):
    payload = {
        'slug': 'ms_session', 'task': 'pick', 'mode': 'master_slave', 'root': str(tmp_path),
        'robot': {'kind': 'agx_piper', 'name': 'piper_x',
                  'options': {'model': 'piper_x', 'enable_on_connect': False,
                              'configure_on_connect': False},
                  'safety': False},
        'gripper': {'kind': 'none'},
        'collect': {'repo_id': 'local/ms_session', 'fps': 50.0, 'export': 'manifest',
                    'cameras': [{'kind': 'mock', 'name': 'wrist_camera',
                                 'width': 32, 'height': 24, 'fps': 50}]},
        'master_slave': {'fps': 50.0, 'joycon': False, 'voice': False},
    }
    path = tmp_path / 'ms.json'
    path.write_text(json.dumps(payload), encoding='utf-8')
    return load_task_config(str(path))


class Script:
    """Stand-in for SessionInput: each name is returned after ``gap`` s."""

    def __init__(self, names, gap=0.15):
        self.names, self.gap, self.dropped = list(names), gap, 0
        self.next_at = time.monotonic() + gap

    def start(self):
        return self

    def get(self, timeout=0.0):
        time.sleep(timeout)
        if time.monotonic() < self.next_at:
            return None
        self.next_at = time.monotonic() + self.gap
        return self.names.pop(0) if self.names else 'quit'

    def stop(self):
        pass


def _run(tmp_path, monkeypatch, names):
    monkeypatch.setattr(session_input, 'SessionInput', lambda **kw: Script(names))
    config = _config(tmp_path)
    assert execute(config, mock=True) == 0
    takes = sorted(task_runner.dataset_root(config).rglob('take.json'))
    return [json.loads(p.read_text()) for p in takes], takes


def test_b_starts_a_saves_and_extra_b_is_ignored(tmp_path, monkeypatch, capsys):
    metas, takes = _run(tmp_path, monkeypatch, ['a', 'b', 'b', 'a'])
    out = capsys.readouterr().out
    assert [m['status'] for m in metas] == ['saved']
    assert '当前没有在录制' in out and '正在录制' in out
    frame = np.load(sorted(takes[0].parent.glob('0*.npz'))[1])
    assert 'action_gripper_from_leader' in frame.files


def test_y_needs_two_presses_and_quit_waits_for_idle(tmp_path, monkeypatch, capsys):
    metas, _ = _run(tmp_path, monkeypatch, ['b', 'y', 'quit', 'y'])
    out = capsys.readouterr().out
    assert [m['status'] for m in metas] == ['discarded']
    assert '再按一次 Y' in out and '先 A 保存' in out


def test_double_press_and_stale_events():
    confirm = DoublePress(timeout_s=1.0)
    assert not confirm.confirm('y', now=0.0) and confirm.confirm('y', now=0.5)
    assert not confirm.confirm('y', now=0.6) and not confirm.confirm('y', now=2.0)

    keys = type('K', (), {'start': lambda self: None, 'stop': lambda self: None,
                          'get': lambda self: None})()
    inputs = SessionInput(joycon=False, keys=keys, max_age_s=0.2)
    inputs.events.put(('b', time.monotonic() - 5.0))
    inputs.events.put(('a', time.monotonic()))
    assert inputs.get() == 'a' and inputs.dropped == 1
    assert session_input.KEYMAP['h'] == 'home'
    assert isinstance(inputs.events, queue.Queue)
