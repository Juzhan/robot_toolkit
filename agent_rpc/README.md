# agent_rpc

Standalone HTTP RPC for agent-facing robot environments. The wire format follows
the RPent-style `env.*` convention, but this package does not import or depend on
RPent.

## Wire format

```
POST /call
{"method": "env.move_delta", "args": [], "kwargs": {"delta_xyz": [...]},
 "session_id": null}

200 {"ok": true, "result": ...}
200 {"ok": false, "error": "...", "traceback": "..."}
```

Numpy arrays are tagged as
`{"__ndarray__": "<base64>", "dtype": "...", "shape": [...]}`; numpy scalars use
`{"__npscalar__": ..., "dtype": "..."}`. `healthz` and `shutdown` are always
available. Sessions are optional.

The server serializes all business calls with one lock: one environment is one
physical resource, so concurrent motion calls are never correct.

## Routes

| Route | Meaning |
| --- | --- |
| `env.get_env_meta` | robot, mode, cameras, action dimensions, protocol version |
| `env.reset` | simulator reset; real robot returns operator-owned reset, no motion |
| `env.get_robot_state` | joints, TCP, gripper, protective state |
| `env.get_observation` | wrist + external RGB, depths in metres, robot state |
| `env.get_camera_meta` | intrinsics, distortion, `T_world_camera`, `T_base_camera` |
| `env.move_delta` | bounded base-frame xyz delta, blocking, returns final error |
| `env.rotate_delta` | bounded base-frame rpy delta, blocking |
| `env.set_gripper` | open/close and measured width |
| `env.get_task_language` | task instruction |
| `env.render_camera` | one live RGB frame |
| `env.step` / `env.chunk_step` | simulator/VLA only; real Piper reports `NotSupported` |

## Real Piper

`PiperRealBackend` borrows an already-connected `RobotPlatform`; it never
connects, enables, or closes devices. Cameras must be open, and motion is refused
unless `PiperEnvConfig(enable_motion=True)` is set.

```python
from robot_toolkit import RobotPlatform
from robot_toolkit.agent_rpc import EnvService, PiperEnvConfig, PiperRealBackend

platform = RobotPlatform(robot, gripper=gripper, cameras=cameras)
backend = PiperRealBackend(
    platform,
    PiperEnvConfig(main_camera='wrist', extra_cameras=('external',),
                   enable_motion=True, max_delta_m=0.03))
service = EnvService(backend)
service.serve('127.0.0.1', 8765)
```

Client:

```python
from robot_toolkit.agent_rpc import EnvClient

env = EnvClient.connect('http://127.0.0.1:8765')
state = env.get_robot_state()
image = env.get_observation()['main_images']
env.move_delta([0.005, 0.0, 0.0])
env.set_gripper(open=False)
env.close()
```

## Depth and frames

`main_images` and `extra_view_images` are RGB `uint8` in HWC layout.
`main_depths` and `extra_view_depths` are `float32` metres; the source camera
convention is `uint16` millimetres, converted at this boundary. Camera extrinsics
are `T_world_camera` in the optical frame. `T_base_camera` is identity-composed
from world unless a `base_T_world` transform is configured.

## Tests

`tests/test_agent_rpc.py` covers codec round-trips, HTTP envelopes, errors,
sessions, and the Piper backend with mock devices. No hardware is touched.
