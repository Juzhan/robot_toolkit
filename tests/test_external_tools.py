"""External-tool lifecycle and wire contracts, without models or devices."""

import base64
import io
import json
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread

import numpy as np
import pytest

from robot_toolkit.external_tools import ExternalToolError, ExternalTools, Sam3Client, ToolRuntime, main


def test_catalog_load_is_offline_and_creates_nothing(tmp_path):
    root = tmp_path / 'tools'
    tools = ExternalTools.load(root=root)
    assert set(tools.definitions) == {'sam3', 'bundlesdf', 'anygrasp', 'curobo', 'fast_foundationstereo'}
    assert not root.exists()
    description = tools.describe('sam3')
    assert description['status'] == 'disabled' and description['ready'] is None
    assert description['process_owner'] == 'external'
    assert not description['source_present']
    json.dumps(description)


def test_runtime_paths_must_stay_outside_toolkit():
    root = Path(__file__).resolve().parents[1]
    with pytest.raises(ValueError, match='outside'):
        ExternalTools.load(root=root / 'models')


@pytest.mark.parametrize('endpoint', ['file:///tmp/service', 'http://user:secret@localhost',
                                     'http://localhost/?token=secret'])
def test_credentials_and_non_http_endpoints_rejected(tmp_path, endpoint):
    with pytest.raises(ValueError, match='endpoint'):
        ToolRuntime('sam3', tmp_path, endpoint=endpoint)


def response_for(mask, *, score=0.8):
    buffer = io.BytesIO()
    np.save(buffer, mask, allow_pickle=False)
    rows, columns = np.where(mask)
    entry = {'mask_b64': base64.b64encode(buffer.getvalue()).decode(), 'score': score,
             'bbox_xywh': [int(columns.min()), int(rows.min()),
                           int(columns.max() - columns.min() + 1), int(rows.max() - rows.min() + 1)],
             'mask_area': int(mask.sum())}
    return {'height': mask.shape[0], 'width': mask.shape[1], 'count': 1, 'detections': [entry]}


def test_sam_contract_preserves_all_masks_and_observation_provenance(tmp_path, monkeypatch):
    client = Sam3Client(ToolRuntime('sam3', tmp_path))
    mask = np.zeros((8, 10), dtype=np.uint8)
    mask[2:5, 3:7] = 1
    response = response_for(mask)
    response['detections'].append(dict(response['detections'][0]))
    response['count'] = 2
    calls = []

    def request(path, payload):
        calls.append(path)
        encoded = base64.b64decode(payload['image_b64'])
        assert np.array_equal(np.load(io.BytesIO(encoded), allow_pickle=False), rgb)
        assert payload['text'] == 'red block'
        return response

    monkeypatch.setattr(client, '_request', request)
    rgb = np.zeros((8, 10, 3), dtype=np.uint8)
    result = client.segment(rgb, 'red block', observation_id='obs-1', camera='top')
    assert calls == ['/segment_all']
    assert result.observation_id == 'obs-1' and result.camera == 'top'
    assert len(result.detections) == 2
    assert result.detections[0].mask.dtype == bool
    assert np.array_equal(result.detections[0].mask, mask)


@pytest.mark.parametrize('corruption', ['dimensions', 'count', 'mask', 'bbox', 'score'])
def test_malformed_sam_evidence_is_error_not_empty_detection(tmp_path, monkeypatch, corruption):
    client = Sam3Client(ToolRuntime('sam3', tmp_path))
    response = response_for(np.ones((8, 10), dtype=np.uint8))
    if corruption == 'dimensions':
        response['height'] = 99
    elif corruption == 'count':
        response['count'] = 0
    elif corruption == 'mask':
        response['detections'][0]['mask_b64'] = 'not base64'
    elif corruption == 'bbox':
        response['detections'][0]['bbox_xywh'] = [0, 0, 1, 1]
    elif corruption == 'score':
        response['detections'][0]['score'] = float('nan')
    monkeypatch.setattr(client, '_request', lambda *args: response)
    with pytest.raises(ExternalToolError, match='invalid SAM3'):
        client.segment(np.zeros((8, 10, 3), dtype=np.uint8), 'block',
                       observation_id='obs', camera='top')


def test_disabled_client_never_sends_a_request(tmp_path):
    client = Sam3Client(ToolRuntime('sam3', tmp_path, endpoint='http://127.0.0.1:1'))
    with pytest.raises(ExternalToolError, match='disabled'):
        client.health()


def test_sam_health_distinguishes_server_from_loaded_model(tmp_path, monkeypatch):
    client = Sam3Client(ToolRuntime('sam3', tmp_path))
    monkeypatch.setattr(client, '_request', lambda *args: {'status': 'ok', 'model_loaded': False})
    assert client.health() == {'reachable': True, 'ready': False, 'model_loaded': False,
                               'model_ref': None, 'process_owner': 'external'}


def test_launch_plan_is_loopback_offline_and_does_not_create_runtime(tmp_path):
    root = tmp_path / 'external'
    tools = ExternalTools.load(root=root)
    plan = tools.launch_plan('sam3')
    assert plan['executed'] is False and plan['validated'] is False
    assert plan['argv'][-4:] == ['--host', '127.0.0.1', '--port', '6767']
    assert plan['env_overrides']['YAM_UV_ABS_BOOTSTRAPPED'] == '1'
    assert plan['env_overrides']['HF_HUB_OFFLINE'] == '1'
    assert not root.exists()
    with pytest.raises(ValueError, match='no verified'):
        tools.launch_plan('bundlesdf')


def test_launch_plan_matches_implicit_http_port(tmp_path):
    tools = ExternalTools.load(root=tmp_path)
    tools.runtimes['sam3'] = ToolRuntime('sam3', tmp_path, endpoint='http://127.0.0.1')
    assert tools.launch_plan('sam3')['argv'][-1] == '80'


def test_local_config_and_cli_do_not_probe_services(tmp_path, capsys):
    config = tmp_path / 'tools.json'
    config.write_text(json.dumps({'schema_version': 1, 'root': str(tmp_path / 'runtime'),
                                  'tools': {'sam3': {'endpoint': 'http://127.0.0.1:1', 'enabled': True}}}))
    assert main(['list', '--config', str(config)]) == 0
    status = json.loads(capsys.readouterr().out)
    sam = next(entry for entry in status if entry['name'] == 'sam3')
    assert sam['enabled'] and sam['status'] == 'not_checked' and sam['ready'] is None
    assert not (tmp_path / 'runtime').exists()


@pytest.mark.parametrize('reply', ['empty', 'multiple', 'http_error', 'redirect', 'bad_json'])
def test_loopback_http_transport(tmp_path, reply):
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            calls.append(self.path)
            payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            rgb = np.load(io.BytesIO(base64.b64decode(payload['image_b64'])), allow_pickle=False)
            assert rgb.shape == (8, 10, 3) and payload['text'] == 'block'
            if reply == 'http_error':
                self.send_error(503)
                return
            if reply == 'redirect':
                self.send_response(307)
                self.send_header('Location', '/unexpected')
                self.end_headers()
                return
            response = {'height': 8, 'width': 10, 'count': 0, 'detections': []}
            if reply == 'multiple':
                response = response_for(np.ones((8, 10), dtype=np.uint8))
                response['detections'] *= 2
                response['count'] = 2
            data = b'not json' if reply == 'bad_json' else json.dumps(response).encode()
            self.send_response(200)
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = HTTPServer(('127.0.0.1', 0), Handler)
    thread = Thread(target=server.handle_request, daemon=True)
    thread.start()
    client = Sam3Client(ToolRuntime('sam3', tmp_path, enabled=True, timeout_s=2,
                                    endpoint=f'http://127.0.0.1:{server.server_port}'))
    try:
        if reply in ('empty', 'multiple'):
            result = client.segment(np.zeros((8, 10, 3), dtype=np.uint8), 'block',
                                    observation_id='obs', camera='top')
            assert len(result.detections) == (0 if reply == 'empty' else 2)
        else:
            with pytest.raises(ExternalToolError):
                client.segment(np.zeros((8, 10, 3), dtype=np.uint8), 'block',
                               observation_id='obs', camera='top')
    finally:
        thread.join(timeout=3)
        server.server_close()
    assert not thread.is_alive()
    assert calls == ['/segment_all']