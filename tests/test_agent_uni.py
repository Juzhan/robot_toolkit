"""End to end: UNI SDK tools -> UNI environment service -> UniBackend -> Station -> mock arm.

Runs only where ``uni_robot_agent`` (with fastapi/uvicorn) is importable.
"""
import socket
import threading
import time

import numpy as np
import pytest

pytest.importorskip('uni_robot_agent.services.environment.service')
pytest.importorskip('uvicorn')

from .station_support import make_station  # noqa: E402


def _free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


@pytest.fixture
def served(tmp_path):
    import uvicorn
    from fastapi import FastAPI
    from uni_robot_agent.services.environment.routes import build_router
    from robot_toolkit.agent.uni import build_service
    station = make_station(robot_id='test-arm')
    service = build_service(station, tmp_path / 'env', episode_id='trial-1', instruction='touch the block')
    app = FastAPI()
    app.include_router(build_router(service))
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(app, host='127.0.0.1', port=port, log_level='warning'))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.05)
    yield f'http://127.0.0.1:{port}', station, tmp_path
    server.should_exit = True
    thread.join(5)
    service.close()


def test_uni_tools_drive_a_toolkit_station(served):
    from uni_robot_agent import Session, SessionConfig
    url, station, tmp_path = served
    config = SessionConfig(integration='robot_toolkit', env_url=url, no_sam3=True,
                           integration_options={'robot_id': 'test-arm'}, output_dir=tmp_path / 'runs')
    with Session(config) as session:
        names = {tool['name'] for tool in session.tools.list()}
        assert {'move', 'grasp', 'capture_scene', 'finish'} <= names
        state = session.tools.call('view_env_state').to_dict()
        assert state['success'], state
        capture = session.tools.call('capture_scene').to_dict()
        assert capture['success'], capture
        before = station.read().T_world_control[:3, 3].copy()
        target = [float(before[0] + 0.06), float(before[1]), float(before[2]), 0.0, 0.0, 0.0]
        moved = session.tools.call('move', {'targets': [{'arm': 'arm', 'pose': target, 'gripper': 1.0}]}).to_dict()
        assert moved['success'], moved
        np.testing.assert_allclose(station.read().T_world_control[:3, 3], before + [0.06, 0, 0], atol=0.03)
        grasped = session.tools.call('grasp', {'targets': [{'arm': 'arm'}]}).to_dict()
        assert grasped['success'], grasped
        assert grasped['feedback']['arms']['arm']['opening_status'] == 'reached', grasped['feedback']
        assert station.gripper.state().target == pytest.approx(station.gripper.min_width)
        finished = session.tools.call('finish', {'status': 'success', 'summary': 'test done'}).to_dict()
        assert finished['success'], finished
    assert session.result is not None
    outcome = session.result.record
    assert outcome is not None
    assert station.blocked is None
