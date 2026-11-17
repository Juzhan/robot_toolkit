# Agent-facing robot feedback

The same methods are inherited by real backends, mock robots, MuJoCo, and
`SafeRobot`. No MCP server, automatic motor enable, task reward, or autonomous
retry is introduced. Existing `move_j()` and `move_p()` still return bool.

## Calls

| Call | Result and effects |
| --- | --- |
| `robot.describe()` | Robot specification, joint ordering, capabilities, units, TCP frame; no connection or enable. Piper retains its existing read-only diagnostic fields. |
| `robot.status()` | JSON-ready state, optional telemetry, read errors, host acquisition window; never connects or commands motion. |
| `robot.state().to_dict()` | Serialize an existing sample without another read; missing or non-finite values become null. |
| `robot.move_j_result(q, **kwargs)` | Calls the existing joint mover through the same safety wrapper; returns execution and feedback evidence. |
| `robot.move_p_result(pose, **kwargs)` | Calls the existing pose mover; checks measured translation and rotation error. |

Offline example (no hardware):

```python
from robot_toolkit import build_robot

robot = build_robot('piper_x', mock=True)
robot.connect()
try:
    print(robot.describe())
    print(robot.status())
    result = robot.move_j_result(
        [0.1, 0.7, -0.7, 0.1, 0.1, 0.1],
        use_native=False, speed=0.5, timeout=5.0)
    print(result['reason'], result['joint_error_rad'])
    print(result['after']['state'])
finally:
    robot.disconnect()
```

Pose example for an already authorized, connected robot with a valid TCP:
`robot.move_p_result(pose, use_native=False, speed=0.2, timeout=10.0)`.
The caller supplies a validated absolute target in the backend's advertised
frame; no implicit relative offsets or quaternion ordering conversions apply.

## State evidence

`q` is radians, `qd` rad/s, and `tcp` is metres plus a rotation vector in radians.
`tcp_frame` is `base` for the hardware backends. MuJoCo currently reports `world`;
align the model's robot base with the world before treating those frames as equal.
The API does not invent a tool offset or certify the configured TCP calibration.

`timestamp_clock` identifies the sample clock, not a synchronization guarantee.
UR uses controller time when available; mock/MuJoCo use host elapsed time;
Piper currently exposes SDK time with a host-elapsed fallback. MuJoCo also
reports `extra.simulation_time`. Do not subtract these timestamps from host
monotonic time. `status.acquisition` bounds the host read only: cached SDK
feedback can still be stale. Freshness is explicitly `unknown`.

`sources` distinguishes controller feedback, simulation, FK, and estimation.
Piper velocity is clipped finite difference, not a directly measured motor
velocity; its legacy `extra.enabled` is cached command state. The normalized
`enabled` stays null until a backend supplies actual enable feedback.
`fault` and `protective_stop` are null when unavailable; missing Piper arm status
is not converted into a healthy zero. Simulator health fields remain unknown
where that physical protection mechanism is not modeled. Extra vendor values
are preserved; opaque non-JSON vendor objects are serialized as strings.

## Motion result

Results contain `operation`, `target`, `parameters`, named `tolerances`,
`before`, `after`, `elapsed_s`, `command_ok`, `reached`, `ok`, `reason`, and `error`.
Snapshots use the same schema as `status()`. Targets never replace measured state.
Errors are maximum absolute joint difference, Euclidean TCP translation, and
SO(3) rotation angle, in radians/metres. Default tolerances are 0.02 rad,
0.005 m, and 0.05 rad; override with `joint_tolerance`, `position_tolerance`,
or `rotation_tolerance` on the corresponding result call.

- `command_ok`: the existing mover's bool result, null when it did not return.
- `reached`: final feedback meets tolerance; null if not checked or unavailable.
- Blocking `ok`: the mover returned true AND final feedback met tolerance,
  with no observed disconnect/fault. This is endpoint agreement, not task success,
  certified settling, freshness, collision clearance, or absolute accuracy.
- With `wait=False`, `ok` means only the mover returned true; `reason=accepted`,
  `reached=null`. Generic movement still streams synchronously before returning.
- Reasons include `disconnected`, `feedback_unavailable`, `not_ready`,
  `invalid_argument`, `safety_rejected`, `unsupported`, `aborted`, `timeout`,
  `servo_rejected`, `backend_failed`, `backend_error`, and `tolerance_not_met`.
  A native false without a detailed cause remains `backend_failed`.

Failed dispatched calls request a best-effort `stop_motion()` without retrying
the movement. `stop.command_ok` reports only that stop method's return;
`stop.physically_stopped` remains null. `after` is the post-move observation;
`stop.feedback` is another observation after the stop attempt. Stop errors do
not overwrite the original motion error. Backends may already have requested
a stop while handling the failure. No assumption of physical emergency-stop
equivalence is made, and base/mock/MuJoCo currently do not implement a native stop.

## Boundary and remaining gaps

These are library conveniences, not a secure Agent execution service. The caller
must serialize calls and independently enforce operator authorization, controller
ownership, feedback freshness, and execution budgets. Never expose arbitrary
Python access to an untrusted Agent and assume the wrapper is a security boundary.
Capability declarations are not runtime readiness checks; unknown health does
not grant permission. Existing native/generic timing and speed differences remain.
`SafeRobot.describe().safety` states the applied target-lead semantics; this is
not measured acceleration enforcement or a full-arm collision checker.

This increment follows the 2026-10-22 RAX session's distinction between common
contracts and vendor implementations, especially measured arrival/error reporting
in [Franka servo](https://github.com/robot-agent-exploration/rax/blob/8858d7f3e34eff382abae738cf0578d2726a9a0a/services/franka/servo.py).
It does not copy Franka's tolerances, hand-written IK, integral controller, or
hardware assumptions. Camera exposure/robot synchronization, normalized gripper
force/contact evidence, model identity and full path preflight, ownership leases,
operator authorization, and task-level observe/reset/verify remain separate work.
Offline/fake SDK and MuJoCo tests validate the contract, not real hardware safety.