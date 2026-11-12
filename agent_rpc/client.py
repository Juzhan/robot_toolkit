"""HTTP RPC client for the agent-facing robot environment."""

from __future__ import annotations

import atexit
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from typing import Any

from .codec import dumps, loads


class RpcError(RuntimeError):
    """A remote call returned ``ok=false`` or the transport failed."""

    def __init__(self, method: str, message: str, *, traceback: str | None = None):
        super().__init__(f'{method}: {message}')
        self.method = method
        self.server_traceback = traceback


def _direct_opener(url: str):
    host = urllib.parse.urlsplit(url).hostname
    if host in {'127.0.0.1', 'localhost'}:
        return urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return None


class RpcClient:
    def __init__(self, base_url: str, *, timeout_s: float = 30.0,
                 enable_sessions: bool = False):
        self.base_url = base_url.rstrip('/')
        self.timeout_s = float(timeout_s)
        self._session_id = f'rpc_{uuid.uuid4().hex}' if enable_sessions else None
        self._opener = _direct_opener(self.base_url)
        self._closed = False
        if self._session_id is not None:
            atexit.register(self.close)

    def call(self, method: str, args: tuple = (), kwargs: dict | None = None,
             *, timeout_s: float | None = None) -> Any:
        payload = {
            'method': method,
            'args': list(args),
            'kwargs': kwargs or {},
            'session_id': self._session_id,
        }
        request = urllib.request.Request(
            f'{self.base_url}/call', data=dumps(payload).encode('utf-8'),
            headers={'Content-Type': 'application/json'}, method='POST')
        opener = self._opener.open if self._opener is not None else urllib.request.urlopen
        try:
            with opener(request, timeout=timeout_s or self.timeout_s) as response:
                body = response.read()
        except urllib.error.HTTPError as exc:
            body = exc.read()
        except OSError as exc:
            raise RpcError(method, f'HTTP request failed: {exc}') from exc
        try:
            envelope = loads(body)
        except Exception as exc:  # noqa: BLE001 - malformed remote response
            raise RpcError(method, f'invalid JSON response: {exc}') from exc
        if not isinstance(envelope, dict):
            raise RpcError(method, 'response is not a JSON object')
        if not envelope.get('ok'):
            raise RpcError(method, str(envelope.get('error', 'unknown error')),
                           traceback=envelope.get('traceback'))
        return envelope.get('result')

    def healthz(self, *, timeout_s: float = 1.0) -> Any:
        return self.call('healthz', timeout_s=timeout_s)

    def shutdown(self, *, timeout_s: float = 2.0) -> Any:
        return self.call('shutdown', timeout_s=timeout_s)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._session_id is None:
            return
        try:
            self.healthz(timeout_s=0.5)
            self.call('session.close', timeout_s=1.0)
        except Exception:
            pass


def parse_endpoint(endpoint: str) -> tuple[str, str, int]:
    if '://' in endpoint:
        protocol, _, rest = endpoint.partition('://')
    else:
        protocol, rest = 'http', endpoint
    host, _, port = rest.partition(':')
    if protocol != 'http':
        raise ValueError(f'only http endpoints are supported, got {protocol!r}')
    if not host or not port:
        raise ValueError(f'endpoint must be [http://]host:port, got {endpoint!r}')
    return protocol, host, int(port)


def make_client(endpoint: str, *, timeout_s: float = 30.0,
                enable_sessions: bool = False) -> RpcClient:
    _, host, port = parse_endpoint(endpoint)
    return RpcClient(f'http://{host}:{port}', timeout_s=timeout_s,
                     enable_sessions=enable_sessions)


def wait_for_ready(client: RpcClient, *, timeout_s: float = 30.0,
                   poll_interval_s: float = 0.1) -> None:
    deadline = time.monotonic() + float(timeout_s)
    last: Exception | None = None
    while time.monotonic() < deadline:
        try:
            client.healthz(timeout_s=1.0)
        except Exception as exc:  # noqa: BLE001 - polling
            last = exc
            time.sleep(poll_interval_s)
            continue
        if client._session_id is not None:
            client.call('session.register', timeout_s=5.0)
        return
    raise TimeoutError(f'RPC server did not become ready within {timeout_s:.0f}s: {last}')
