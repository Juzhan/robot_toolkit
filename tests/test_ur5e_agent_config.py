from pathlib import Path

import numpy as np

from robot_toolkit.agent.station import Station
from robot_toolkit.task_runner import load_task_config

TASK = Path(__file__).resolve().parents[1] / 'configs' / 'tasks' / 'ur5e_agent.json'


def test_ur5e_agent_task_uses_ur5_project_serials_and_calibration():
    task = load_task_config(str(TASK))
    assert task.mode == 'agent'
    assert task.robot['kind'] == 'ur_rtde' and task.gripper['kind'] == 'robotiq'
    serials = {c['name']: c['serial'] for c in task.collect.cameras}
    assert serials == {'wrist_camera': '846112070112', 'external_camera': '342522071753'}


def test_ur5e_agent_station_loads_both_calibrations_in_mock_dry_run():
    station = Station.from_task(str(TASK), mock=True).open(execute=False)
    try:
        cal = station.calibrations
        wrist = cal['wrist_camera']
        external = cal['external_camera']
        assert wrist.tcp_camera is not None and wrist.base_camera is None
        assert external.base_camera is not None and external.tcp_camera is None
        assert wrist.intrinsics is not None and external.intrinsics is not None
        for T in (wrist.tcp_camera, external.base_camera):
            np.testing.assert_allclose(np.asarray(T)[:3, :3] @ np.asarray(T)[:3, :3].T, np.eye(3), atol=1e-6)
    finally:
        station.close()
