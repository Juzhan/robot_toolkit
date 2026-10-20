"""One-owner continuous servo stream.

This is the robot-agnostic part of the Mac real-robot policy executor that ran
at 125 Hz: action chunks arriving at 10 Hz are interpolated, a jerk-limited
reference keeps the commanded joint stream continuous, and a shared action
clock stops advancing when the arm cannot keep up.  Network / encoding work is
kept off the servo thread by the chunk and gripper workers.

The executor does not know about HTTP, LeRobot or evaluation labels.  Feed it
any :class:`~.chunks.ChunkSource`, including a local :class:`SequenceSource`.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from ..core.continuous import (ContinuousReference, gripper_allowed,
                               progress_allowed)
from ..errors import RobotUnavailable
from ..robot.mock import MockRobot
from .chunks import ActionChunk, ChunkSource


@dataclass
class ContinuousConfig:
    """Control and safety parameters, matching the proven deployment defaults."""

    control_hz: float = 125.0
    action_hz: float = 10.0
    max_joint_speed: float = 0.8
    max_joint_accel: float = 1.2
    max_joint_jerk: float = 8.0
    joint_lower: list | None = None
    joint_upper: list | None = None
    reference_backend: str = 'auto'

    # Safety / tracking gates.
    max_action_jump_rad: float = 1.0
    max_tracking_error: float = 0.15
    tracking_timeout_s: float = 3.0
    progress_reference_gap: float = 0.05
    progress_tracking_gap: float = 0.08
    max_servo_gap_s: float = 0.04
    max_trial_s: float = 60.0
    first_chunk_timeout_s: float = 30.0
    chunk_wait_timeout_s: float = 20.0
    prefetch_ratio: float = 0.5
    settle_s: float = 0.25

    # Gripper gating / actuation.
    actuate_gripper: bool = True
    gripper_reference_tolerance: float = 0.025
    gripper_tracking_tolerance: float = 0.06
    gripper_target_tolerance: float = 0.035
    gripper_change_m: float = 0.004
    gripper_min_interval_s: float = 0.25

    # Optional sample callback rate.
    capture_fps: float = 10.0
    mock_sleep_s: float = 0.0002


@dataclass
class ContinuousResult:
    aborted: bool = False
    end_reason: str = ''
    error: str = ''
    duration_s: float = 0.0
    phase_s: float = 0.0
    frames: int = 0
    servo_commands: int = 0
    chunks: int = 0
    requests: int = 0
    actions_received: int = 0
    actions_played: int = 0
    gripper_commands: int = 0
    phase_paused_ticks: int = 0
    gripper_gated_ticks: int = 0
    actual_send_hz: float | None = None
    send_gap_ms_p50_p95_p99_max: list = field(default_factory=list)
    work_ms_p50_p95_p99_max: list = field(default_factory=list)
    max_tracking_error: float = 0.0
    reference_backend: str = ''
    meta: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.aborted

    def as_dict(self) -> dict:
        return {
            'ok': self.ok,
            'aborted': self.aborted,
            'end_reason': self.end_reason,
            'error': self.error,
            'duration_s': self.duration_s,
            'phase_s': self.phase_s,
            'frames': self.frames,
            'servo_commands': self.servo_commands,
            'chunks': self.chunks,
            'requests': self.requests,
            'actions_received': self.actions_received,
            'actions_played': self.actions_played,
            'gripper_commands': self.gripper_commands,
            'phase_paused_ticks': self.phase_paused_ticks,
            'gripper_gated_ticks': self.gripper_gated_ticks,
            'actual_send_hz': self.actual_send_hz,
            'send_gap_ms_p50_p95_p99_max': list(self.send_gap_ms_p50_p95_p99_max),
            'work_ms_p50_p95_p99_max': list(self.work_ms_p50_p95_p99_max),
            'max_tracking_error': self.max_tracking_error,
            'reference_backend': self.reference_backend,
            'meta': dict(self.meta),
        }


class _ChunkWorker(threading.Thread):
    """Fetch chunks off the servo thread; at most one request is outstanding."""

    def __init__(self, source: ChunkSource, stop: threading.Event):
        super().__init__(daemon=True, name='continuous-chunks')
        self.source = source
        self.stop = stop
        self.requests: queue.Queue = queue.Queue(maxsize=1)
        self.results: queue.Queue = queue.Queue()

    def request(self, action_index: int) -> bool:
        try:
            self.requests.put_nowait(int(action_index))
            return True
        except queue.Full:
            return False

    def run(self):
        while not self.stop.is_set():
            try:
                index = self.requests.get(timeout=0.05)
            except queue.Empty:
                continue
            if self.stop.is_set():
                break
            try:
                chunk = self.source.next_chunk(index)
                if chunk is None:
                    raise RuntimeError('action source returned no chunk')
                self.results.put(('chunk', chunk))
            except Exception as exc:
                self.results.put(('error', exc))
                return


class _GripperWorker(threading.Thread):
    """Serialize gripper writes without blocking the servo stream."""

    def __init__(self, gripper, stop: threading.Event):
        super().__init__(daemon=True, name='continuous-gripper')
        self.gripper = gripper
        self.stop = stop
        self.commands: queue.Queue = queue.Queue(maxsize=1)
        self.error = None
        self.sent = 0

    def submit(self, width: float) -> bool:
        try:
            self.commands.put_nowait(float(width))
            return True
        except queue.Full:
            return False

    def flush(self, timeout: float = 0.5) -> None:
        """Give an already-submitted width a chance to reach the gripper."""
        deadline = time.perf_counter() + max(0.0, float(timeout))
        while not self.commands.empty() and time.perf_counter() < deadline:
            time.sleep(0.005)
        if time.perf_counter() < deadline:
            time.sleep(0.02)

    def run(self):
        while not self.stop.is_set():
            try:
                width = self.commands.get(timeout=0.02)
            except queue.Empty:
                continue
            if self.stop.is_set():
                break
            try:
                if self.gripper.move(width, wait=False) is False:
                    raise RuntimeError('gripper move returned false')
                self.sent += 1
            except Exception as exc:
                self.error = str(exc)
                return


def _is_mock(robot) -> bool:
    inner = getattr(robot, 'inner', robot)
    return bool(getattr(inner, 'mock', False)) or isinstance(inner, MockRobot)


def _wait_until(deadline: float) -> None:
    remaining = deadline - time.perf_counter()
    if remaining > 0.0004:
        time.sleep(remaining - 0.0003)
    while time.perf_counter() < deadline:
        pass


def ensure_servo(robot, dt: float, ticks: int = 5, realtime: bool = True) -> bool:
    """Verify servo mode before streaming; rebuild the controller link if possible.

    The RTDE control script can time out after a long idle prompt, so this sends
    a few hold commands at the measured pose before any motion command.
    """
    for attempt in range(2):
        state = robot.state()
        if state.q is None:
            return False
        hold = np.asarray(state.q, dtype=float)
        failures = 0
        for _ in range(max(1, int(ticks))):
            try:
                if not robot.servo_j(hold, dt):
                    failures += 1
            except Exception:
                failures += 1
            if realtime:
                time.sleep(dt)
        if failures == 0:
            return True
        restart = getattr(robot, 'restart_control', None)
        if attempt == 0 and callable(restart):
            try:
                if restart():
                    continue
            except Exception:
                pass
        break
    return False


def _validate_chunk(chunk: ActionChunk, reference: ContinuousReference,
                    action_hz: float, max_width: float) -> np.ndarray:
    if abs(float(chunk.action_hz) - float(action_hz)) > 1e-6:
        raise ValueError('policy action frequency changed')
    actions = np.asarray(chunk.actions, dtype=float)
    if actions.size == 0:
        if chunk.done:
            return actions
        raise ValueError('empty action chunk with done=false')
    if actions.ndim != 2 or actions.shape[1] != reference.dof + 1:
        raise ValueError(f'action chunk must be Nx{reference.dof + 1}, got {actions.shape}')
    if not np.all(np.isfinite(actions)):
        raise ValueError('action chunk contains NaN/Inf')
    for action in actions:
        reference.validate(action[:reference.dof])
    if np.any(actions[:, -1] < 0) or np.any(actions[:, -1] > float(max_width) + 1e-6):
        raise ValueError('policy gripper width outside physical bounds')
    return actions


def _stats(values, scale: float = 1.0) -> list:
    if len(values) == 0:
        return []
    return np.percentile(np.asarray(values, dtype=float) * scale, [50, 95, 99, 100]).tolist()


def run_continuous_stream(robot, gripper, source: ChunkSource,
                          config: ContinuousConfig | None = None, *,
                          abort_event: threading.Event | None = None,
                          end_event: threading.Event | None = None,
                          realtime: bool = True,
                          on_command=None, on_sample=None,
                          on_start=None) -> ContinuousResult:
    """Execute absolute action chunks through a continuous joint reference.

    ``source.next_chunk(action_index)`` is called on a worker thread.  The
    executor never waits for the network in the servo loop; it holds the last
    action and pauses the shared action clock until the next chunk arrives.
    """
    cfg = config or ContinuousConfig()
    dt = 1.0 / float(cfg.control_hz)
    if dt <= 0 or cfg.action_hz <= 0:
        raise ValueError('control_hz and action_hz must be positive')
    hardware = not _is_mock(robot)
    if not realtime and hardware:
        raise ValueError('virtual time is forbidden on real hardware')
    abort_event = abort_event or threading.Event()
    end_event = end_event or threading.Event()
    stop = threading.Event()
    result = ContinuousResult(meta={'control_hz': cfg.control_hz,
                                    'action_hz': cfg.action_hz})

    initial = robot.state()
    if initial.q is None:
        raise RobotUnavailable('continuous stream needs a joint state')
    if initial.protective_stop:
        raise RobotUnavailable('robot is in protective stop')
    initial_q = np.asarray(initial.q, dtype=float)
    if not np.all(np.isfinite(initial_q)):
        raise RobotUnavailable('initial joint state is not finite')
    if hardware and hasattr(robot, 'joints_within_safety'):
        if not robot.joints_within_safety(initial_q):
            raise RobotUnavailable('initial pose fails controller safety validation')

    if hardware:
        if not ensure_servo(robot, dt, realtime=realtime):
            raise RobotUnavailable('cannot establish the servo stream')
        try:
            robot.stop_motion(0.5)
        except Exception:
            pass

    try:
        grip_last = float(gripper.latest_width())
    except Exception:
        grip_last = float(getattr(gripper, 'commanded_width', 0.0))
    max_width = float(getattr(gripper, 'max_width', 0.085)) if gripper is not None else 0.085
    dof = int(initial_q.size)
    lower = cfg.joint_lower
    upper = cfg.joint_upper
    if lower is None or upper is None:
        guard = getattr(getattr(robot, 'guard', None), 'limits', None)
        if guard is not None and getattr(guard, 'lower', None) is not None:
            lower = guard.lower.tolist()
            upper = guard.upper.tolist()
        else:
            lower = [-2.0 * np.pi] * dof
            upper = [2.0 * np.pi] * dof
    reference = ContinuousReference(initial_q, dt, cfg.max_joint_speed,
                                    cfg.max_joint_accel, cfg.max_joint_jerk,
                                    lower, upper, backend=cfg.reference_backend)
    result.reference_backend = reference.backend
    target = np.r_[initial_q, grip_last]
    if on_start is not None:
        on_start({'config': cfg, 'initial_state': initial, 'max_width': max_width})

    worker = _ChunkWorker(source, stop)
    grip_worker = _GripperWorker(gripper, stop) if gripper is not None else None
    worker.start()
    if grip_worker is not None:
        grip_worker.start()
    if not worker.request(0):
        raise RuntimeError('cannot submit the first action request')

    start = time.perf_counter()
    phase = 0.0
    chunk_start = 0.0
    offset = 0
    current: ActionChunk | None = None
    following: ActionChunk | None = None
    pending = True
    requests = 1
    chunks = 0
    received = 0
    commands = 0
    tick = 0
    captures = 0
    phase_paused = 0
    gripper_gated = 0
    last_grip_time = -1e9
    next_capture = 0.0
    tracking_bad_since = None
    phase_bad_since = None
    waiting_since = None
    final_settle = None
    last_robot_time = initial.t
    fresh_at = start
    sends: list[float] = []
    work_times: list[float] = []
    target_gap = 0.0
    tracking_gap = 0.0
    action_index = 0
    next_deadline = start
    now = start
    sdk_period = None
    if realtime and hardware:
        init_period = getattr(robot, 'init_period', None)
        frequency = float(getattr(getattr(robot, 'config', None), 'frequency', 0.0) or 0.0)
        if callable(init_period) and (frequency <= 0 or abs(frequency - cfg.control_hz) < 1.0):
            try:
                sdk_period = init_period()
            except Exception:
                sdk_period = None

    try:
        while True:
            began = time.perf_counter()
            now = began if realtime else start + tick * dt
            wall_elapsed = now - start
            if abort_event.is_set():
                raise RuntimeError('operator_abort')
            if end_event.is_set():
                result.end_reason = 'manual'
                break
            if cfg.max_trial_s > 0 and wall_elapsed >= cfg.max_trial_s:
                result.end_reason = 'timeout'
                break
            if realtime and sends and began - sends[-1] > cfg.max_servo_gap_s:
                raise RuntimeError('servo_gap: target stream exceeded timing limit')

            measured = robot.state()
            if measured.q is None:
                raise RobotUnavailable('robot state lost joint positions')
            measured_q = np.asarray(measured.q, dtype=float)
            if not np.all(np.isfinite(measured_q)):
                raise RuntimeError('invalid measured joints')
            if measured.protective_stop:
                raise RuntimeError('protective_stop')
            if measured.t != last_robot_time:
                last_robot_time, fresh_at = measured.t, began
            if hardware and began - fresh_at > 0.1:
                raise RuntimeError('stale_robot_state')

            tracking_gap = float(np.max(np.abs(reference.q - measured_q)))
            result.max_tracking_error = max(result.max_tracking_error, tracking_gap)
            if tracking_gap > cfg.max_tracking_error:
                raise RuntimeError(f'tracking_error: {tracking_gap:.4f} rad')
            if tracking_gap > cfg.progress_tracking_gap:
                tracking_bad_since = now if tracking_bad_since is None else tracking_bad_since
                if now - tracking_bad_since > cfg.tracking_timeout_s:
                    raise RuntimeError('tracking_timeout')
            else:
                tracking_bad_since = None
            if grip_worker is not None and grip_worker.error:
                raise RuntimeError('gripper_error: ' + grip_worker.error)

            while True:
                try:
                    kind, payload = worker.results.get_nowait()
                except queue.Empty:
                    break
                pending = False
                if kind != 'chunk':
                    raise RuntimeError('policy_error: ' + str(payload))
                chunk: ActionChunk = payload
                actions = _validate_chunk(chunk, reference, cfg.action_hz, max_width)
                received += int(len(actions))
                if current is None:
                    current = chunk
                    offset = int(chunk.action_index)
                    chunk_start = phase
                    chunks += 1
                else:
                    following = chunk
            if current is None and wall_elapsed > cfg.first_chunk_timeout_s > 0:
                raise RuntimeError('first_chunk_timeout')

            elapsed = phase - chunk_start
            if current is not None:
                count = int(len(current.actions))
                duration = count / cfg.action_hz if count else 0.0
                if count == 0:
                    if current.done:
                        result.end_reason = 'model_done'
                        break
                    raise RuntimeError('empty nonterminal action chunk')
                if elapsed >= duration and not current.done:
                    if following is not None:
                        next_index = offset + count
                        skip = max(0, next_index - int(following.action_index))
                        if skip >= len(following.actions):
                            # Never replay an expired final action by clamping.
                            following = None
                        else:
                            current = following.slice(skip)
                            offset = int(current.action_index)
                            chunk_start = phase
                            following = None
                            chunks += 1
                            count = len(current.actions)
                            elapsed = 0.0
                            duration = count / cfg.action_hz
                            waiting_since = None
                    if elapsed >= duration:
                        waiting_since = now if waiting_since is None else waiting_since
                        if now - waiting_since > cfg.chunk_wait_timeout_s:
                            raise RuntimeError('chunk_timeout')
                if len(current.actions):
                    position = max(0.0, min(elapsed * cfg.action_hz, count - 1))
                    index0 = int(position)
                    fraction = position - index0
                    target = (current.actions[index0] * (1.0 - fraction)
                              + current.actions[min(index0 + 1, count - 1)] * fraction)
                    # Never anticipate a discrete grip marker.
                    target[-1] = current.actions[index0, -1]
                    if not pending and following is None and not current.done:
                        if elapsed >= duration * cfg.prefetch_ratio:
                            index = offset + min(count, int(max(0.0, elapsed) * cfg.action_hz))
                            if worker.request(index):
                                requests += 1
                                pending = True

            target_gap = float(np.max(np.abs(target[:dof] - reference.q)))
            if cfg.max_action_jump_rad > 0 and target_gap > cfg.max_action_jump_rad:
                raise RuntimeError(f'action_jump: {target_gap:.4f} rad')
            allowed = progress_allowed(target_gap, tracking_gap,
                                       cfg.progress_reference_gap, cfg.progress_tracking_gap)
            if allowed and (current is None or elapsed < duration):
                phase += dt
                phase_bad_since = None
            else:
                phase_paused += 1
                if target_gap > cfg.progress_reference_gap:
                    phase_bad_since = now if phase_bad_since is None else phase_bad_since
                    if now - phase_bad_since > cfg.tracking_timeout_s:
                        raise RuntimeError('reference_catchup_timeout')

            q = reference.step(target[:dof])
            send_at = time.perf_counter()
            if realtime and sends and send_at - sends[-1] > cfg.max_servo_gap_s:
                raise RuntimeError('servo_gap: computation exceeded timing limit')
            if not robot.servo_j(q, dt):
                raise RuntimeError('servo_error')
            sends.append(send_at)
            commands += 1

            grip_ok = gripper_allowed(float(np.max(np.abs(target[:dof] - q))), tracking_gap,
                                      cfg.gripper_reference_tolerance,
                                      cfg.gripper_tracking_tolerance)
            grip_ok = grip_ok and float(
                np.max(np.abs(target[:dof] - measured_q))) <= cfg.gripper_target_tolerance
            width = float(target[-1])
            if (cfg.actuate_gripper and grip_worker is not None
                    and abs(width - grip_last) >= cfg.gripper_change_m):
                if grip_ok and now - last_grip_time >= cfg.gripper_min_interval_s:
                    if grip_worker.submit(width):
                        grip_last, last_grip_time = width, now
                else:
                    gripper_gated += 1

            action_index = offset
            if current is not None and len(current.actions):
                action_index = offset + min(len(current.actions) - 1,
                                            int(max(0.0, phase - chunk_start) * cfg.action_hz))
            if on_command is not None:
                on_command({
                    't': wall_elapsed, 'send_t': send_at - start,
                    'q_cmd': q.tolist(), 'target': target[:dof].tolist(),
                    'gripper_cmd': grip_last, 'chunk': chunks,
                    'request': requests, 'action_index': action_index,
                    'qd_cmd': reference.v.tolist(), 'qdd_cmd': reference.a.tolist(),
                    'phase_s': phase, 'tracking_gap': tracking_gap,
                    'reference_gap': target_gap,
                    'cycle_ms': 0.0 if len(sends) < 2 else (sends[-1] - sends[-2]) * 1000.0,
                })
            if on_sample is not None and wall_elapsed >= next_capture - 1e-6:
                if gripper is not None:
                    try:
                        width_meas = float(gripper.latest_width())
                    except Exception:
                        width_meas = float(getattr(gripper, 'commanded_width', grip_last))
                else:
                    width_meas = grip_last
                on_sample({
                    't': wall_elapsed, 'stamp': began,
                    'joints': measured_q.tolist(),
                    'state': np.r_[measured_q, width_meas].tolist(),
                    'tcp_pose': None if measured.tcp is None else np.asarray(measured.tcp).tolist(),
                    'robot_time': measured.t, 'q_cmd': q.tolist(),
                    'q_meas': measured_q.tolist(),
                    'gripper_cmd': grip_last, 'gripper_meas': width_meas,
                    'chunk': chunks, 'request': requests, 'action_index': action_index,
                    'phase_s': phase, 'tracking_gap': tracking_gap,
                    'reference_gap': target_gap,
                })
                captures += 1
                next_capture += 1.0 / max(1e-6, float(cfg.capture_fps))
                if next_capture < wall_elapsed - 1.0 / max(1e-6, float(cfg.capture_fps)):
                    next_capture = wall_elapsed + 1.0 / max(1e-6, float(cfg.capture_fps))

            if current is not None and current.done and elapsed >= duration:
                settled = (target_gap < 0.01 and tracking_gap < 0.03
                           and float(np.max(np.abs(reference.v))) < 0.02)
                final_settle = (now if final_settle is None else final_settle) if settled else None
                if final_settle is not None and now - final_settle >= cfg.settle_s:
                    result.end_reason = 'model_done'
                    break

            work_times.append(time.perf_counter() - began)
            if realtime:
                if sdk_period is not None:
                    try:
                        if robot.wait_period(sdk_period):
                            tick += 1
                            continue
                    except Exception:
                        pass
                    sdk_period = None
                next_deadline += dt
                if next_deadline < time.perf_counter() - dt:
                    next_deadline = time.perf_counter() + dt
                _wait_until(next_deadline)
            else:
                time.sleep(max(0.0, float(cfg.mock_sleep_s)))
            tick += 1
    except Exception as exc:
        result.aborted = True
        result.end_reason = str(exc).split(':')[0] or type(exc).__name__
        result.error = str(exc)
    finally:
        # Stop first: joining workers or aborting a network request must never
        # postpone stopping the robot.
        try:
            robot.stop_motion(0.5)
        except Exception as exc:
            result.aborted = True
            result.error = (result.error + '; stop failed: ' + str(exc)).strip('; ')
        if grip_worker is not None and not result.aborted:
            # A width submitted just before model_done is still a valid final
            # command; let it leave the queue before the worker is stopped.
            grip_worker.flush(0.5)
        stop.set()
        abort_source = getattr(source, 'abort', None)
        if callable(abort_source):
            try:
                abort_source()
            except Exception:
                pass
        worker.join(timeout=2.0)
        if grip_worker is not None:
            grip_worker.join(timeout=2.0)
            if grip_worker.error:
                result.aborted = True
                result.end_reason = 'gripper_error'
                result.error = grip_worker.error

    gaps = np.diff(sends)
    result.duration_s = max(0.0, now - start)
    result.phase_s = phase
    result.frames = tick + 1
    result.servo_commands = commands
    result.chunks = chunks
    result.requests = requests
    result.actions_received = received
    result.actions_played = action_index if commands else 0
    result.gripper_commands = 0 if grip_worker is None else grip_worker.sent
    result.phase_paused_ticks = phase_paused
    result.gripper_gated_ticks = gripper_gated
    result.actual_send_hz = (float(1.0 / np.mean(gaps)) if len(gaps) and realtime else None)
    result.send_gap_ms_p50_p95_p99_max = _stats(gaps, 1000.0)
    result.work_ms_p50_p95_p99_max = _stats(work_times, 1000.0)
    return result
