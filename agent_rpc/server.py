"""Small HTTP RPC server for agent-facing robot environments.

Wire format (compatible with the RPent ``env.*`` convention)::

    POST /call
    {"method": "env.move_delta", "args": [], "kwargs": {"delta_xyz": ...},
     "session_id": null}
    -> {"ok": true, "result": ...}
    -> {"ok": false, "error": "...", "traceback": "..."}

Failures always use HTTP 200; the envelope carries the error.  Numpy arrays are
tagged by :mod:`robot_toolkit.agent_rpc.codec`.
"""

from __future__ import annotations

import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable

from .codec import dumps, loads

SESSION_TIMEOUT_S = 3600.0


class RpcServer:
    """Method registry plus an HTTP transport.

    All business methods run under one lock: a robot environment is a single
    physical resource, so concurrent motion calls are never correct.  Health
    checks bypass the lock.
    """

    def __init__(self, *, enable_sessions: bool = False,
                 session_timeout_s: float = SESSION_TIMEOUT_S):
        self._methods: dict[str, Callable[..., Any]] = {}
        self._readonly: set[str] = set()
        self._lock = threading.RLock()
        self._shutdown = threading.Event()
        self._sessions: dict[str, float] = {}
        self._session_timeout_s = float(session_timeout_s)
        self._enable_sessions = bool(enable_sessions)
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self.address: tuple[str, int] | None = None

    # ------------------------------------------------------------- registry
    def register(self, method: str, handler: Callable[..., Any], *,
                 readonly: bool = False) -> None:
        if not callable(handler):
            raise TypeError('RPC handler must be callable')
        self._methods[str(method)] = handler
        if readonly:
            self._readonly.add(str(method))
        else:
            self._readonly.discard(str(method))

    def method(self, name: str, *, readonly: bool = False):
        def decorator(handler):
            self.register(name, handler, readonly=readonly)
            return handler
        return decorator

    def register_object(self, backend: Any, *, prefix: str = 'env.') -> None:
        """Register public methods advertised by ``backend.rpc_methods``."""
        methods = getattr(backend, 'rpc_methods', None)
        if methods is None:
            raise ValueError('backend does not declare rpc_methods')
        for name in methods:
            handler = getattr(backend, name)
            readonly = bool(getattr(backend, 'rpc_readonly', {}).get(name, False))
            self.register(prefix + name, handler, readonly=readonly)

    @property
    def methods(self) -> tuple[str, ...]:
        return tuple(sorted(self._methods))

    # ------------------------------------------------------------- dispatch
    def dispatch(self, method: str, args: tuple = (), kwargs: dict | None = None,
                 *, session_id: str | None = None) -> Any:
        kwargs = kwargs or {}
        if method == 'healthz':
            return {'status': 'ok'}
        if method == 'shutdown':
            with self._lock:
                self._shutdown.set()
            return {'ok': True}
        if self._enable_sessions:
            if method == 'session.register':
                return self._register_session(session_id)
            if method == 'session.close':
                return self._drop_session(session_id)
            if not session_id or session_id not in self._sessions:
                raise ValueError('session required: call session.register first')
            self._sessions[session_id] = time.monotonic()
        handler = self._methods.get(method)
        if handler is None:
            raise ValueError(f'unknown RPC method: {method!r}')
        with self._lock:
            return handler(*args, **kwargs)

    def _register_session(self, session_id: str | None) -> dict:
        if not isinstance(session_id, str) or not session_id:
            raise ValueError('session_id must be a non-empty string')
        with self._lock:
            self._sessions[session_id] = time.monotonic()
        return {'ok': True, 'session_id': session_id}

    def _drop_session(self, session_id: str | None) -> dict:
        with self._lock:
            self._sessions.pop(session_id or '', None)
        return {'ok': True, 'session_id': session_id}

    # ------------------------------------------------------------- transport
    def start(self, host: str = '127.0.0.1', port: int = 0) -> tuple[str, int]:
        """Bind and serve in a background thread; returns the effective address."""
        if self._httpd is not None:
            raise RuntimeError('server is already running')
        httpd = _HttpServer((host, int(port)), self)
        self._httpd = httpd
        self.address = httpd.server_address
        self._thread = threading.Thread(target=httpd.serve_forever,
                                        name='agent-rpc-http', daemon=True)
        self._thread.start()
        return self.address

    def serve(self, host: str = '127.0.0.1', port: int = 0, *,
              parent_watch: bool = False) -> None:
        """Blocking variant used by a standalone env server process."""
        address = self.start(host, port)
        print(f'RPC server listening on http://{address[0]}:{address[1]}', flush=True)
        if parent_watch:
            threading.Thread(target=self._watch_parent, name='agent-rpc-parent',
                             daemon=True).start()
        try:
            while not self._shutdown.wait(0.2):
                pass
        finally:
            self.stop()

    def _watch_parent(self) -> None:
        try:
            while sys.stdin.buffer.read(1):
                pass
        except Exception:
            pass
        self._shutdown.set()

    def stop(self) -> None:
        self._shutdown.set()
        httpd, thread = self._httpd, self._thread
        self._httpd = self._thread = None
        if httpd is not None:
            httpd.shutdown()
            httpd.server_close()
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback_value):
        self.stop()
        return False


class _HttpServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, address, rpc: RpcServer):
        self.rpc = rpc
        super().__init__(address, _Handler)


class _Handler(BaseHTTPRequestHandler):
    server_version = 'robot-toolkit-agent-rpc/1'

    def log_message(self, fmt, *args):
        return

    def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler API
        self.send_response(404)
        self.end_headers()

    def do_POST(self):  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path != '/call':
            self.send_response(404)
            self.end_headers()
            return
        length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(length) if length else b''
        try:
            request = loads(body)
            if not isinstance(request, dict):
                raise ValueError('request body must be a JSON object')
            result = self.server.rpc.dispatch(
                request['method'], tuple(request.get('args', ())),
                dict(request.get('kwargs', {})),
                session_id=request.get('session_id'))
            response = {'ok': True, 'result': result}
        except Exception as exc:  # noqa: BLE001 - the envelope reports it
            response = {'ok': False, 'error': str(exc),
                        'traceback': ''.join(traceback.format_exception(
                            type(exc), exc, exc.__traceback__))}
        payload = dumps(response).encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        try:
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            pass
