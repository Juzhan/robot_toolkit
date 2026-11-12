"""Generic ``env.*`` service and client shared by simulator and real backends."""

from __future__ import annotations

from typing import Any

from .client import RpcClient, make_client, wait_for_ready
from .server import RpcServer

_COMMON = (
    'get_env_meta',
    'reset',
    'get_robot_state',
    'get_observation',
    'get_camera_meta',
    'move_delta',
    'rotate_delta',
    'set_gripper',
    'get_task_language',
    'render_camera',
    'step',
    'chunk_step',
)

_READONLY = {
    'get_env_meta', 'get_robot_state', 'get_observation',
    'get_camera_meta', 'get_task_language', 'render_camera',
}


class EnvService:
    """Register a backend's ``env.*`` methods on an :class:`RpcServer`."""

    def __init__(self, backend: Any, *, server: RpcServer | None = None):
        self.backend = backend
        self.server = server or RpcServer()
        self.meta = backend.get_env_meta()
        self._register()

    def _register(self) -> None:
        for name in _COMMON:
            handler = getattr(self.backend, name, None)
            if handler is None:
                continue
            self.server.register('env.' + name, handler,
                                 readonly=name in _READONLY)

    def register(self, name: str, handler, *, readonly: bool = False) -> None:
        """Expose a robot-specific skill next to the common env routes."""
        self.server.register(name, handler, readonly=readonly)

    def start(self, host: str = '127.0.0.1', port: int = 0):
        return self.server.start(host, port)

    def serve(self, host: str = '127.0.0.1', port: int = 0, *,
              parent_watch: bool = False) -> None:
        self.server.serve(host, port, parent_watch=parent_watch)

    def stop(self) -> None:
        self.server.stop()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback_value):
        self.stop()
        return False


class EnvClient:
    """Typed convenience wrapper over the ``env.*`` routes."""

    def __init__(self, client: RpcClient, *, reset_on_connect: bool = False):
        self.client = client
        self.meta = self.client.call('env.get_env_meta', timeout_s=30.0)
        if reset_on_connect:
            self.reset()

    @classmethod
    def connect(cls, endpoint: str, *, timeout_s: float = 30.0,
                enable_sessions: bool = False,
                reset_on_connect: bool = False) -> 'EnvClient':
        client = make_client(endpoint, timeout_s=timeout_s,
                             enable_sessions=enable_sessions)
        wait_for_ready(client, timeout_s=timeout_s)
        return cls(client, reset_on_connect=reset_on_connect)

    def reset(self) -> dict:
        return self.client.call('env.reset', timeout_s=120.0)

    def get_robot_state(self) -> dict:
        return self.client.call('env.get_robot_state')

    def get_observation(self) -> dict:
        return self.client.call('env.get_observation')

    def get_camera_meta(self) -> dict:
        return self.client.call('env.get_camera_meta')

    def get_task_language(self) -> str | None:
        return self.client.call('env.get_task_language')

    def move_delta(self, delta_xyz) -> dict:
        return self.client.call('env.move_delta',
                                kwargs={'delta_xyz': delta_xyz}, timeout_s=120.0)

    def rotate_delta(self, delta_rpy) -> dict:
        return self.client.call('env.rotate_delta',
                                kwargs={'delta_rpy': delta_rpy}, timeout_s=120.0)

    def set_gripper(self, *, open: bool) -> dict:
        return self.client.call('env.set_gripper',
                                kwargs={'open': bool(open)}, timeout_s=120.0)

    def render_camera(self, camera_name: str) -> Any:
        return self.client.call('env.render_camera',
                                kwargs={'camera_name': camera_name},
                                timeout_s=30.0)

    def step(self, action) -> Any:
        return self.client.call('env.step', args=(action,), timeout_s=120.0)

    def chunk_step(self, actions, *, return_all_frames: bool = False) -> Any:
        return self.client.call('env.chunk_step', kwargs={
            'actions': actions, 'return_all_frames': bool(return_all_frames),
        }, timeout_s=300.0)

    def close(self) -> None:
        self.client.close()
