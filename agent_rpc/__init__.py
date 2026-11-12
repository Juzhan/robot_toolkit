"""Agent-facing ``env.*`` RPC for robot_toolkit.

The wire format follows the RPent-style convention, but this package does not
import or depend on RPent.  Simulator and real-robot backends implement the same
``EnvService`` contract.
"""

from .client import RpcClient, RpcError, make_client, parse_endpoint, wait_for_ready
from .env import EnvClient, EnvService
from .piper_env import PiperEnvConfig, PiperRealBackend, build_piper_env_service
from .server import RpcServer

__all__ = [
    'RpcClient', 'RpcError', 'RpcServer', 'EnvClient', 'EnvService',
    'PiperEnvConfig', 'PiperRealBackend', 'build_piper_env_service',
    'make_client', 'parse_endpoint', 'wait_for_ready',
]
