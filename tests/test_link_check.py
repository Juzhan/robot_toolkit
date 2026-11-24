from types import SimpleNamespace

from robot_toolkit import link_check


class _Robot:
    mock = False

    def __init__(self, q=(0.0, 0.1), updating=True, stop=False, leader=None):
        self.q, self.updating, self.stop, self.leader, self.t = q, updating, stop, leader, 0.0
        self.gripper_width = 0.042

    def state(self):
        if self.updating:
            self.t += 1.0
        return SimpleNamespace(q=self.q, t=self.t, protective_stop=self.stop, fault=0, enabled=True)


class _Camera:
    def __init__(self, name, age=0.01, opened=True):
        self.name, self.serial, self.opened, self._age = name, '', opened, age

    def snapshot(self):
        return SimpleNamespace(rgb=object(), age=lambda: self._age)


def _check(robot, cameras=(), monkeypatch=None):
    monkeypatch.setattr(link_check.time, 'sleep', lambda _s: None)
    lines = []
    ok, summary = link_check.run_link_check(robot, cameras=cameras, log=lines.append)
    return ok, summary, lines


def test_link_check_all_ok(monkeypatch):
    ok, summary, lines = _check(_Robot(), [_Camera('wrist_camera'), _Camera('external_camera')],
                                monkeypatch)
    assert ok
    assert '机器人在线' in summary and '2 台相机正常' in summary
    assert any('42.0mm' in line for line in lines)


def test_link_check_flags_stale_robot_and_camera(monkeypatch):
    ok, summary, _ = _check(_Robot(updating=False),
                            [_Camera('wrist_camera', age=3.0), _Camera('external_camera', opened=False)],
                            monkeypatch)
    assert not ok
    assert '机器人离线' in summary and '腕部相机、外部相机异常' in summary


def test_link_check_flags_protective_stop_and_silent_can(monkeypatch):
    monkeypatch.setattr(link_check, '_can_link', lambda _robot: ('can0', 'up'))
    monkeypatch.setattr(link_check, '_rx_packets', lambda _channel: 100)
    ok, summary, _ = _check(_Robot(stop=True), (), monkeypatch)
    assert not ok
    assert '保护停止' in summary and 'CAN 总线无数据' in summary


def test_stationary_leader_keeps_last_command():
    from robot_toolkit.collect.passive import leader_label, read_leader

    robot = SimpleNamespace(leader_feedback=lambda: dict(q=[0.0] * 6, q_age_s=30.0,
                                                         gripper=0.05, gripper_age_s=30.0))
    leader = read_leader(robot, max_age_s=0.3)
    assert leader['ok'] and leader['gripper_ok'] and not leader['moving']
    assert leader_label(leader) == '主臂静止'
    silent = SimpleNamespace(leader_feedback=lambda: dict(q=None, q_age_s=None,
                                                          gripper=None, gripper_age_s=None))
    assert not read_leader(silent)['ok']
