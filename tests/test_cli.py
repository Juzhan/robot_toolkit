"""CLI smoke tests: every offline command must work without hardware."""

from __future__ import annotations

import json

from robot_toolkit.cli import main


def test_info_and_doctor(capsys):
    assert main(['info']) == 0
    assert main(['doctor']) == 0
    output = capsys.readouterr().out
    assert 'robot_toolkit' in output


def test_selftest():
    assert main(['selftest']) == 0


def test_demo_end_to_end(tmp_path):
    output = tmp_path / 'demo'
    assert main(['demo', '--out', str(output)]) == 0
    summary = json.loads((output / 'summary.json').read_text(encoding='utf-8'))
    assert summary['trajectory']['samples'] > 100
    assert summary['replay']['frames'] > 100
    assert not summary['replay']['aborted']
    assert summary['collector']['saved'] == 1
    assert (output / 'trajectories' / 'trajectory.npz').exists()
    manifests = list((output / 'collect').glob('*/dataset/episodes.jsonl'))
    assert manifests


def test_calibration_demos(tmp_path):
    handeye = tmp_path / 'handeye.json'
    assert main(['calib-handeye', '--demo', '--out', str(handeye)]) == 0
    payload = json.loads(handeye.read_text(encoding='utf-8'))
    assert payload['eye_in_hand']['truth_error'] < 1e-6
    assert payload['eye_to_hand']['truth_error'] < 1e-6

    intrinsics = tmp_path / 'intrinsics.json'
    assert main(['calib-intrinsics', '--synthetic', '--out', str(intrinsics)]) == 0
    payload = json.loads(intrinsics.read_text(encoding='utf-8'))
    assert abs(payload['focal_error'][0]) < 0.02
