# Control Contracts and Real-Robot Lessons

Status: 2026-11-08. These are software contracts and bounded offline checks,
not a certification of physical stability, stopping distance, or every robot.

## The actual motion chain

The current workspace IBVS probe uses:

```text
ArUco corners -> image error / interaction matrix -> bounded joint velocity
 -> Ruckig velocity reference -> absolute joint position samples
 -> AgxPiperRobot.servo_j -> pyAgxArm.move_j -> CAN -> Piper controller
```

Ruckig velocity mode is a HOST reference generator. It does not put Piper into
a hardware velocity or torque mode. The official SDK remains responsible for
protocol encoding and transmission; the firmware controls the motors.

The PBVS probe instead constructs a Cartesian pose target, solves IK, and uses
a position-reference generator. Thus the current implementations differ in
more than their feature input. Their physical limits can be shared, but a
per-update pose correction fraction is not the same parameter as an IBVS gain.

## Query semantics, not just method names

`robot.describe()['capabilities']` includes:

| Backend | `servo_j_kind` | `servo_j_timing` | `stop_motion_kind` |
| --- | --- | --- | --- |
| Real Piper | `planned_position` | `host_scheduled` | `measured_position_hold` |
| Real UR RTDE | `native_servo` | `controller_dt` | `vendor_servo_stop` |
| Mock / MuJoCo | `simulation` | `simulation_step` | `unsupported` |
| Piper / UR in mock mode | `simulation` | `simulation_step` | `no_op` |
| Undeclared backend | `unspecified` | `unspecified` | `unspecified` |

- Piper `servo_j` calls SDK `move_j`, not UR-style `servoJ`. `dt` is not a
  controller command-expiry time. Firmware planning and its speed percentage
  still affect the response. A host jerk bound is not a measured motor jerk bound.
- UR calls RTDE `servoJ` with `dt`, lookahead, and gain; stopping calls
  `servoStop`. This still requires the correct firmware, timing, and feedback.
- MuJoCo writes actuator targets and advances physics. Mock integrates a
  simplified velocity-limited model. Neither reproduces CAN latency, firmware
  replanning, quantization, holding noise, or real drive faults by default.
- A true return from `servo_j` is not arrival confirmation. Piper's optional
  `command_hz` throttle can return true without transmitting that call. Executor
  call counts / send timestamps do not by themselves prove CAN packet rates.
- A true return from Piper `stop_motion` means a measured-position hold was
  submitted, not verified standstill. Its `deceleration` argument is not
  implemented by this transport. Missing or invalid joint feedback returns false.
- Disconnecting Piper deliberately leaves it enabled. Process exit is not
  torque-off or emergency stop. Do not automatically switch to teaching, reset,
  or disable an elevated arm as an error-recovery shortcut.

These fields describe the implementation; they are not a safety certificate.
A client requiring native servo semantics must reject `planned_position` and
`unspecified`, rather than merely checking `capabilities.servo_j`.

## What changed during visual-servo development

| Change | Evidence / purpose | Boundary |
| --- | --- | --- |
| Blocking Cartesian steps to an independent reference thread | Decouples camera / IK latency from target publication | Host scheduling, not hard real time |
| Require Ruckig | A fixed-goal limit cycle was reproduced in the legacy fallback; user reported the original jitter disappeared with Ruckig | Does not establish every later noise cause |
| Track both reference and measured joints | Stops reference runaway and distinguishes command convergence from arm convergence | Requires fresh, trustworthy feedback |
| Frame-to-state matching, filtering and reacquisition changes | Reduces some delay/noise effects and stale target handling | Nearest host timestamp is not exposure synchronization |
| IBVS four-corner velocity control | Uses image error directly; PnP supplies depth | Current version changes orientation as well as position |
| Image stop/resume hysteresis and zero-command braking | Avoids some small-error retargeting; handles target loss / timeout | Thresholds trade accuracy against activity |
| Stop repeating a settled Piper position target in the probe | Offline test proves send count stops and tracking can resume | Not proven to be the cause of the wrist noise; do not copy this policy to native servos needing a stream |
| Shared joint speed / acceleration / jerk configuration | Removes the hidden second IBVS joint-speed cap | Cartesian image-controller limits still apply |
| Enable before applying speed | Avoids an enable transition overwriting configured speed | A Piper adapter ordering issue, not an IBVS equation |
| Serialize actual telemetry in tests | Caught NumPy boolean JSON failure that terminated execution | Mock telemetry alone had missed the bug |

The user reported better real tracking after these iterations. There is no
controlled hardware benchmark attributing that improvement to every individual
change. Ideal image-control convergence tests do not establish physical gains.

The wrist noise after program exit remains unassigned. The 2026-11-05 passive
CAN capture was taken AFTER the user switched to teaching to stop the sound.
Its mode and J6 micro-motion cannot establish what occurred during noisy IBVS
position holding. Lack of a drive alarm does not prove normal operation.

## Changes extracted into the toolkit

1. `ContinuousReference(backend='auto')` no longer silently substitutes the
   legacy fallback when Ruckig is missing. Importing the module still works;
   constructing the default reference raises an actionable dependency error.
2. The legacy fallback remains explicitly selectable for offline experiments,
   not a repaired or convergent hardware controller. `run_continuous_stream`
   rejects it on non-mock backends, even when explicitly requested.
3. The continuous executor validates the reference before `ensure_servo` sends
   any initial hold commands. Missing dependencies and invalid reference limits
   must fail before this active handshake, not after it.
4. Piper normal connections and read-only-to-motion upgrades share J-mode
   preparation. `prepare_motion` enables, applies speed, and prepares J mode
   without sending a joint target. Supported SDKs disable per-command mode
   switching. Existing compatibility fallback for older SDKs remains; it must
   not be assumed to have identical timing.
5. Piper stop/hold rejects missing, non-finite, or malformed feedback rather
   than reporting success or submitting invalid angles.
6. Backend capabilities expose motion semantics without hardware connection.
   `SafeRobot` preserves these declarations through the wrapper.

## Keep responsibilities separate

- Task controller: feature extraction, task error, target generation, image-loss
  policy, desired pose / velocity, task-specific convergence thresholds.
- Generic execution: finite values, units, shared physical limits, reference
  generation, one command owner, source timeout, feedback freshness, tracking
  error gates, diagnostics, and explicit success/failure states.
- Hardware adapter: official SDK calls, firmware profile, native command type,
  mode setup, enable/speed ordering, transmission throttling, feedback clocks,
  and honest stop/hold behavior.
- Hardware validation: payload, mounting, reachable posture, communication
  budget, drive behavior, stopping procedure, and measured tracking performance.

Do not implement Piper mode frames or sign conventions in a generic IBVS loop.
Do not assume matching units and method signatures imply matching dynamics.
Do not replace `move_j` with `move_js`, CPV, or MIT to make it "more real time":
the installed SDK explicitly warns that `move_js` has no smoothing/planning
and can cause severe shock or instability. Such a backend needs separate review
and supervised qualification, not an automatic fallback.

## Validation gates and remaining work

### Platform monitoring update (2026-11-09)

`RobotPlatform.move_j` and `move_p` now monitor explicitly declared native
nonblocking moves. Real Piper advertises these via `nonblocking_moves`;
`use_native=False` and other backends retain backend-owned waits. There is no
automatic conversion of host interpolation into an asynchronous command.

The platform submits once, checks feedback, handles timeout/cancellation, and
records stop attempts separately from physical standstill. Optional positional
dwell defaults to zero; positive dwell requires advancing sample timestamps.
Timestamp stalls, clock changes, invalid feedback and known faults fail the
action. Freshness remains unknown: host fallback timestamps can advance even
when cached sensor values do not. This is not a hardware watchdog or a proof of
zero velocity. See [the API contract](agent_api.md#native-motion-monitoring).

Overlapping calls on one platform return `busy` without reading or stopping
the active arm. This does not arbitrate direct backend calls, other platform
instances, or motion continuing after `wait=False` returns. Synchronous SDK
calls must return before the platform can handle cancellation or timeout.
Offline tests cover deterministic feedback sequences, cancellation, deadline
handling, stop failure, same-platform ownership and fake-SDK Piper integration;
no physical robot was connected for this update.

Offline gates covered in existing test files:

- Fixed-target settling AND retargeting, not just bounded velocity/acceleration.
- Missing Ruckig, invalid reference limits, and hardware fallback rejection
  before active servo handshakes.
- A lagging/stuck fake robot and action-clock / gripper gating.
- Read-only connection, authorized motion upgrade, and no pose command during
  preparation; normal SDK path does not reissue J mode on each target.
- Missing / invalid stop feedback and observable command semantics.
- In the workspace probe: real telemetry JSON, shared limits, image geometry,
  reference stop/resume, command timeout, and dry-run authorization.

Run the toolkit slice with an environment containing NumPy, pytest and Ruckig:

```sh
python -m pytest tests/test_continuous.py tests/test_agx_piper.py tests/test_robot.py -q
```

Before a new physical deployment, separately qualify stationary hold, small
steps, moving targets, target loss, saturation, feedback delay, and interruption.
Log image/source timestamps, requested velocity, reference q/v/a, measured q,
actual adapter transmission times, mode transitions, drive status, and exit
reason. Record whether data was acquired before or after an operator mode change.
Compare identical tasks, payloads, limits and postures; do not tune several
layers at once. Do not deliberately induce a dangerous hardware fault.

Still not established by this change:

- The standalone workspace visual-servo probe is not yet a shared executor;
  its raw Piper backend does not automatically inherit every `SafeRobot` guard.
- Not every other legacy call site that explicitly selects the fallback is
  audited or forbidden; the default generator and this executor are covered.
- Actual CAN cadence, feedback exposure synchronization, firmware-specific
  planning and optimal host rate still require measurements. Read-only Piper
  connection skips firmware querying; upgrading it does not auto-negotiate a
  firmware profile. Use a qualified explicit profile where required.
- Software stop/hold return values are not proof of standstill, and not all
  legacy callers propagate every stop failure. A stalled process cannot run a
  host watchdog; hardware stopping must be an independent safety provision.
- Cross-robot physical stability and the earlier wrist noise have not been
  resolved by mock, simulation, or mathematical tests.