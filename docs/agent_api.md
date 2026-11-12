# Agent-facing robot feedback

Real backends, mock robots, MuJoCo, and `SafeRobot` share state and description
methods. Compose devices with `platform = RobotPlatform(robot, ...)`, then opt
in to reports with `reported = with_reports(platform)`.
No MCP server, automatic motor enable, task reward, or autonomous retry is
introduced. Original `robot.move_j()` and `robot.move_p()` still return bool.

## Calls

| Call | Result and effects |
| --- | --- |
| `robot.describe()` | Robot specification, joint ordering, capabilities, units, TCP frame; no connection or enable. Piper retains its existing read-only diagnostic fields. |
| `robot.status()` | JSON-ready state, optional telemetry, read errors, host acquisition window; never connects or commands motion. |
| `robot.state().to_dict()` | Serialize an existing sample without another read; missing or non-finite values become null. |
| `platform.observe()` | Native robot/gripper samples and optional host-time-paired camera frames, with an observation ID. |
| `platform.move_j(q, **kwargs)` / `platform.move_p(pose, **kwargs)` | Execute through the original robot and return a numeric `ExecutionRecord`. |
| `platform.execution(execution_id)` | Retrieve a detached retained record without device I/O. |
| `record.to_report()` | Convert an existing record to JSON-ready feedback without another read or motion. |
| `reported.move_j(q, **kwargs)` | Calls the existing joint mover through the same safety wrapper; returns execution and feedback evidence. |
| `reported.move_p(pose, **kwargs)` | Calls the existing pose mover; checks measured translation and rotation error. |

Offline example (no hardware):

```python
from robot_toolkit import RobotPlatform, with_reports
from robot_toolkit.profiles import build_robot

robot = build_robot('piper_x', mock=True)
platform = RobotPlatform(robot)
reported = with_reports(platform)
robot.connect()
try:
    print(robot.describe())
    print(robot.status())
    result = reported.move_j(
        [0.1, 0.7, -0.7, 0.1, 0.1, 0.1],
        use_native=False, speed=0.5, timeout=5.0)
    print(result['reason'], result['joint_error_rad'])
    print(result['after']['state'])
    record = platform.execution(result['execution_id'])
    print(record.ok, record.details['after']['observation_id'])
finally:
    robot.disconnect()
```

Pose example for an already authorized, connected robot with a valid TCP:
`reported.move_p(pose, use_native=False, speed=0.2, timeout=10.0)`.
The caller supplies a validated absolute target in the backend's advertised
frame; no implicit relative offsets or quaternion ordering conversions apply.

## Platform and history

An optional `scene_estimator=PlanarColorEstimator(spec)` adds native scene
estimates to image observations without a second camera read. See
[Planar scenes and initial-layout checks](scene.md) for the fixed-camera
contract, read-only reset-layout verifier, offline demo, and safety limits.

`RobotPlatform(robot, gripper=None, cameras=None, history_size=128)` borrows the
exact objects supplied, including the robot's safety wrapper and any shared
arm/gripper transport. Construction performs no device I/O. Open/connect devices
explicitly and close them through their original owners. Neither the platform
nor reporting view is a context manager, a device backend, or an authorization
boundary. Overlapping motion calls on the same platform return a `busy` record
without device I/O. Calls must still be serialized with direct device users,
other platform instances, observations and lifecycle operations. Nonwaiting
calls do not reserve ownership beyond their return.

External model clients are independent of device ownership. See
[External tools](external_tools.md) for the offline tool catalogue, isolated
runtime configuration, SAM3 service client, readiness limits and RAX comparison.

`observe()` returns an `Observation`: `state` is a `RobotState`, `gripper` is an
optional `GripperState`, and `frames` contains native `CameraFrame` objects.
Configured cameras must already be open. Their frames use one persistent
`RobotCameraSynchronizer`; `strict=False` retains mismatch evidence instead of
rejecting unmatched images. It does not suppress device read errors. Gripper
feedback is read separately, not time-aligned to the images. Absent/disconnected
gripper feedback is recorded as unavailable, never substituted with a target.

Use `observe(include_images=False)` for a lightweight device observation. Motion
operations use this path for before/after/stop feedback: they do not capture
images or require open cameras. Explicit image observations are not silently
labelled as the before/after images of a later movement. Camera metadata includes
the supplied intrinsics, distortion, and `T_world_camera`; the platform does not
compute moving wrist-camera extrinsics or infer calibration/frame identities.

`move_j` and `move_p` return `ExecutionRecord` with `execution_id`,
`started_at`, `finished_at`, `ok`, and `details`. Targets and feedback arrays stay
numeric; use `record.ok`, not the truthiness of the record object. Snapshot
feedback carries `observation_id`. Invalid/rejected operations are also recorded.
The platform owns preflight checks, arrival checks, and best-effort failure
stopping regardless of whether reports are enabled. Controller motion remains
solely in the original robot backend.

`execution_ids` and `observation_ids` list retained IDs in chronological order.
`execution(id)` and `observation(id)` return detached copies; caller edits cannot
rewrite history. Each history retains at most `history_size` entries; zero
disables retention without changing execution checks. Evicted IDs raise
`KeyError`. Execution records retain their numeric feedback even if the separate
observation entries are evicted. `clear_history()` clears records and pairing
history, not physical device state, connections, or motion.

Image buffers are returned to the caller but not retained in platform history.
Historical observations have empty `frames` and preserve `camera_metadata` and
synchronization evidence. There is no background sampling, disk recorder, task
reset/reward, batch-training backend, or automatic collector migration. This
first version records joint/pose moves; grippers participate in observation,
not a new generic grasp-success implementation. Direct device calls are outside
platform history. Metadata retention is count-bounded, not a total byte quota.

## Native motion monitoring

`platform.move_j` and `platform.move_p` monitor arrival when `wait=True`,
`use_native` is not false, and the backend explicitly lists the operation in
`capabilities.nonblocking_moves`. Real Piper declares both operations; its
`SafeRobot` wrapper preserves this contract and is not bypassed. Other backends
retain their original waiting behavior; no controller support is inferred from
method names. A host-streamed `move_j(wait=False)` is not a nonblocking dispatch.

```python
record = platform.move_j(
        target_joints, wait=True, timeout=5.0, joint_tolerance=0.02,
        stable_duration_s=0.1,
)
if not record.ok:
        print(record.to_report())
```

The example dwell is a task choice, not a validated hardware setting. New
platform-only options (also accepted by the reporting view) are:

- `stable_duration_s=0.0`: no mandatory dwell. Positive values require sampled
    arrival within tolerance for this duration on both the host and advancing
    feedback clocks. Leaving tolerance resets the interval. This is position
    stability, not a measured zero-velocity or physical-stop guarantee.
- `monitor_hz=50.0`: nominal polling rate, not a hard-real-time guarantee.
- `feedback_stall_timeout_s=0.5`: fail on a nonadvancing feedback timestamp or
    an observation gap this long. The polling period must be shorter. Timestamp
    regression or a clock-name change fails separately. These are configurable
    software defaults, not hardware-qualified freshness thresholds.

The platform issues one native `wait=False` move, then observes without images.
`timeout` covers preflight, dispatch and monitoring (60 seconds when omitted on
this path). `abort_event` is checked before dispatch and throughout monitoring;
setting it interrupts the polling wait. Neither cancellation nor timeout can
interrupt a blocked SDK call; there is no background watchdog or device worker.
Piper feedback may use a host timestamp fallback, so advancing timestamps do
not prove sensor freshness. `monitoring.feedback_freshness` stays `unknown`.
The timestamp also does not establish synchronization between TCP and joints.

Failures after dispatch request the backend's stop and retain its result;
`stop.physically_stopped` remains unknown. No automatic retry, enable, reset,
disable, teaching-mode switch, or gripper command is performed.

`monitoring` records the execution mode, requested settings, sample count and
qualified dwell time. A legacy path is labelled `backend_wait`; requesting
positive dwell on it fails before dispatch with `unsupported`. `wait=False`
retains `reason='accepted'`, `reached=None` and mode `dispatch_only`, without
ongoing monitoring or an asynchronous handle; positive dwell is unsupported.

## Reporting view

`with_reports(platform)` requires a `RobotPlatform`, not a robot backend. The
earlier robot-level wrapper and `move_j_result`/`move_p_result` methods have been
removed. Repeated `with_reports(reported)` returns the same view without I/O.

`reported.move_j(...)` and `reported.move_p(...)` invoke the platform once and
call `to_report()` on that execution record. Reports never repeat device reads,
motion, preflight checks, or stopping. Options are keyword-only after the target;
arrival tolerances are consumed by the platform, not passed to the controller.
Other platform methods pass through unchanged, so `reported.observe()` still
returns a native `Observation`. Direct `robot.state()`/`robot.servo_j()` calls
remain the low-level numeric path, with no platform history or report overhead.

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
Reports retain the robot `status()` feedback fields and add execution and
observation IDs plus separately sampled gripper feedback. Platform acquisition
windows and execution timestamps use host `perf_counter`; the original robot
`status()` uses its labelled host-monotonic clock. Neither is a sensor timestamp
or proof of freshness. Targets never replace measured state.
Errors are maximum absolute joint difference, Euclidean TCP translation, and
SO(3) rotation angle, in radians/metres. Default tolerances are 0.02 rad,
0.005 m, and 0.05 rad; override with `joint_tolerance`, `position_tolerance`,
or `rotation_tolerance` on the corresponding reporting-view call.

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

## Camera/robot time pairing

`RobotCameraSynchronizer` matches each image's host reception time to the nearest
robot read midpoint. It uses a bounded history (2 seconds / 256 samples by
default), preserving joint/TCP feedback instead of interpolating or substituting
a command target. The full robot read window must be within `max_skew_s` of the
image time; a slow read is not accepted merely because its midpoint looks close.
Images older than `max_frame_age_s` are rejected. Defaults are 0.05 and 0.3 seconds,
respectively; these are pairing limits, not guarantees of positional accuracy.

Offline example, no movement:

```python
from robot_toolkit.camera import CameraHub, MockCamera
from robot_toolkit.collect import RobotCameraSynchronizer
from robot_toolkit.robot import MockRobot

robot = MockRobot()
with CameraHub([MockCamera('wrist', 32, 24)]) as cameras:
    sync = RobotCameraSynchronizer(robot, cameras, max_skew_s=0.05)
    observation = sync.observe()
    print(observation['state'].q)
    print(observation['synchronization'])
```

`observe()` reads robot state before and after one camera-hub snapshot. It does
not open devices, enable motors, send motion, start workers, or sleep/retry for a
match. Devices must already be ready. `strict=False` returns rejection evidence
instead of raising on unmatched images; robot read failures still raise.

For delayed images, populate history in your existing serialized feedback loop
with `sync.sample_state()`. To reuse a read without repeating it, call
`sync.add_state(sample, started_at=..., finished_at=...)` with actual
`time.perf_counter()` values taken around that read. Do not timestamp a cached
sample afterwards, pass `sample.t`, or use `status()`'s differently labelled
clock without establishing equivalence. Sparse history may cause validly
rejected pairs; the synchronizer does not invent earlier states.

Each camera gets its own `state`, read window, signed `state_minus_frame_s`,
`read_window_skew_bound_s`, age, original frame metadata, and matched/reason
fields. The top-level state uses the first camera as reference and is null if
that camera cannot be matched. `camera_received_span_s` shows the inter-camera
reception spread; matching all images does not mean simultaneous capture.

To record pairs, construct `EpisodeCollector(..., synchronize=True,
sync_max_skew_s=0.05, sync_max_frame_age_s=0.3)` and call `capture_state()` without
an untimed state argument. This optional mode writes the reference camera's
paired joints and stores all pairing evidence as a Unicode JSON scalar named
`synchronization_json` in each raw NPZ (loadable with `allow_pickle=False`).
It rejects unmatched frames even with `strict_cameras=False`, and rejects
`capture_vectors()`, which has no robot-read timing. A configured gripper must
have a width in the matched robot sample; the current gripper value and a guessed
open width cannot replace it. That width still shares the backend's acquisition
limitations. Supplied actions are kept but labelled not time-aligned; default
actions copy the reference observation. The ordinary record timestamp keeps its
existing collection/caller clock semantics; camera times live in pairing metadata.
Existing LeRobot feature exports do not include this new JSON field; keep the
raw NPZ files for pairing evidence. Default collection remains unchanged.

**This is host-reception pairing, not hardware exposure synchronization.** SDK
feedback can be cached and camera processing/transport can delay reception.
`exposure_synchronized` therefore stays false and `robot_feedback_freshness`
unknown. Original camera hardware timestamps are retained, not blindly compared
with robot timestamps. Accurate dynamic hand-eye use still requires verified
device-to-host clock mappings, sensor latency bounds, and appropriate calibration.

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
hardware assumptions. Hardware exposure/robot synchronization, normalized gripper
force/contact evidence, model identity and full path preflight, ownership leases,
operator authorization, and task-level observe/reset/verify remain separate work.
Offline/fake SDK and MuJoCo tests validate the contract, not real hardware safety.