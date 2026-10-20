"""Trajectory playback with speed scaling, gripper events and abort handling."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ..core.continuous import ContinuousReference
from ..errors import Aborted, SafetyViolation
from ..robot.base import RobotBackend
from .trajectory import (GripperSchedule, Trajectory, build_events, max_joint_speed,
                         resample, smooth_joints)


@dataclass
class ReplayPlan:
    t: np.ndarray
    q: np.ndarray
    events: list
    schedule: GripperSchedule
    initial_width: float
    duration: float
    rate: float
    speed_scale: float
    max_speed: float
    source_speed: float


@dataclass
class ReplayResult:
    frames: int = 0
    duration: float = 0.0
    aborted: bool = False
    reason: str = ''
    events: list = field(default_factory=list)
    max_speed_rad_s: float = 0.0
    source_speed_rad_s: float = 0.0
    overrun_p95_ms: float = 0.0
    captured: int = 0
    continuous: bool = False
    reference_backend: str = ''

    def as_dict(self) -> dict:
        return dict(frames=self.frames, duration=self.duration, aborted=self.aborted,
                    reason=self.reason, events=list(self.events),
                    max_speed_rad_s=self.max_speed_rad_s,
                    source_speed_rad_s=self.source_speed_rad_s,
                    overrun_p95_ms=self.overrun_p95_ms, captured=self.captured,
                    continuous=self.continuous,
                    reference_backend=self.reference_backend)


def prepare_playback(trajectory: Trajectory, rate: float = 125.0, speed_scale: float = 0.3,
                     smoothing: int = 5, start_hold_s: float = 1.0, end_hold_s: float = 1.0,
                     max_width: float = 0.085) -> ReplayPlan:
    rate = float(rate)
    speed_scale = float(np.clip(speed_scale, 0.02, 2.0))
    q_smooth = smooth_joints(trajectory.q, smoothing)
    t_source = np.asarray(trajectory.t, dtype=float) - float(trajectory.t[0])
    t_play, q_play = resample(t_source / speed_scale, q_smooth, rate)
    prepend = max(1, int(round(float(start_hold_s) * rate)))
    append = max(1, int(round(float(end_hold_s) * rate)))
    start_offset = prepend / rate
    t_full = np.concatenate([
        np.arange(prepend) / rate,
        t_play + start_offset,
        start_offset + t_play[-1] + (np.arange(1, append + 1) / rate),
    ])
    q_full = np.concatenate([
        np.repeat(q_play[:1], prepend, axis=0), q_play,
        np.repeat(q_play[-1:], append, axis=0),
    ], axis=0)
    grip = np.asarray(trajectory.grip, dtype=float).reshape(-1)
    initial_width = (float(np.clip(grip[0], 0.0, max_width))
                     if len(grip) and np.isfinite(grip[0]) else float(max_width))
    events = build_events(trajectory.markers, initial_width, max_width,
                          start_offset, speed_scale)
    schedule = GripperSchedule([event for event in events if event['width'] is not None],
                               initial_width)
    source_speed = max_joint_speed(t_source, q_smooth)
    return ReplayPlan(t=t_full, q=q_full, events=events, schedule=schedule,
                      initial_width=initial_width, duration=float(t_full[-1]),
                      rate=rate, speed_scale=speed_scale,
                      max_speed=source_speed * speed_scale, source_speed=source_speed)


class Replayer:
    """Execute a ``Trajectory`` on any backend via ``servo_j``.

    The loop is deliberately the same for simulation and hardware: the only
    difference is whether ``servo_j`` talks to MuJoCo or RTDE.
    """

    def __init__(self, robot: RobotBackend, gripper=None, rate: float = 125.0,
                 max_joint_speed: float = 1.0, close_settle_s: float = 0.4,
                 open_settle_s: float = 0.2, approach_speed: float = 0.4,
                 approach_accel: float = 0.8, gripper_free: bool = False,
                 capture_fps: float = 0.0, smoothing: int = 5,
                 continuous: bool = False, max_joint_accel: float = 1.2,
                 max_joint_jerk: float = 8.0, use_sdk_period: bool = True):
        self.robot = robot
        self.gripper = gripper
        self.rate = float(rate)
        self.max_joint_speed = float(max_joint_speed)
        self.max_joint_accel = float(max_joint_accel)
        self.max_joint_jerk = float(max_joint_jerk)
        self.smoothing = max(1, int(smoothing))
        self.close_settle_s = float(close_settle_s)
        self.open_settle_s = float(open_settle_s)
        self.approach_speed = float(approach_speed)
        self.approach_accel = float(approach_accel)
        self.gripper_free = bool(gripper_free)
        self.capture_fps = float(capture_fps)
        self.continuous = bool(continuous)
        self.use_sdk_period = bool(use_sdk_period)
        self.gripper_errors: list[str] = []

    def run(self, trajectory: Trajectory, speed_scale: float | None = None,
            allow_fast: bool = False, confirm=None, capture_hook=None, on_event=None,
            abort_event=None, realtime: bool = True, hold_callback=None) -> ReplayResult:
        speed_scale = 0.3 if speed_scale is None else float(speed_scale)
        plan = prepare_playback(trajectory, rate=self.rate, speed_scale=speed_scale,
                                smoothing=self.smoothing,
                                max_width=self._max_width())
        result = ReplayResult(max_speed_rad_s=plan.max_speed,
                              source_speed_rad_s=plan.source_speed,
                              continuous=self.continuous)
        if plan.max_speed > self.max_joint_speed and not allow_fast:
            raise Aborted(f'replay joint speed {plan.max_speed:.2f} rad/s exceeds the '
                          f'{self.max_joint_speed:.2f} rad/s limit; reduce speed_scale')
        if confirm is not None and not confirm('确认急停在手、工作区无人后开始回放'):
            result.aborted = True
            result.reason = 'cancelled'
            return result

        dt = 1.0 / plan.rate
        q_full, t_full = plan.q, plan.t
        total = plan.duration
        pending = list(plan.events)
        if self.gripper is not None and not self.gripper_free:
            self._fire('move', plan.initial_width)

        # Approach the first waypoint slowly instead of jumping in servo mode.
        current = self.robot.state().q
        if current is not None:
            gap = float(np.max(np.abs(current - q_full[0])))
            if gap > 0.02:
                try:
                    self.robot.move_j(q_full[0], speed=self.approach_speed,
                                      accel=self.approach_accel, wait=True,
                                      abort_event=abort_event)
                except Exception as exc:
                    result.aborted = True
                    result.reason = f'approach failed: {exc}'
                    return result
            current = self.robot.state().q

        reference = None
        if self.continuous and current is not None:
            limits = getattr(getattr(self.robot, 'guard', None), 'limits', None)
            if limits is not None:
                lower, upper = limits.lower, limits.upper
            elif getattr(self.robot, 'limits', None) is not None:
                lower = self.robot.limits.lower
                upper = self.robot.limits.upper
            else:
                lower = [-2.0 * np.pi] * self.robot.dof
                upper = [2.0 * np.pi] * self.robot.dof
            reference = ContinuousReference(
                current, dt, self.max_joint_speed, self.max_joint_accel,
                self.max_joint_jerk, lower, upper)
            result.reference_backend = reference.backend

        # The RTDE control script can time out while waiting at a prompt; make
        # sure servo mode is alive immediately before the stream starts.
        if not getattr(self.robot, 'mock', False):
            from ..control.continuous import ensure_servo
            if not ensure_servo(self.robot, dt, realtime=bool(realtime)):
                result.aborted = True
                result.reason = 'servo channel unavailable'
                return result

        sdk_period = None
        if (realtime and self.use_sdk_period
                and not getattr(self.robot, 'mock', False)):
            frequency = float(getattr(getattr(self.robot, 'config', None),
                                      'frequency', 0.0) or 0.0)
            init_period = getattr(self.robot, 'init_period', None)
            if callable(init_period) and (frequency <= 0 or abs(frequency - plan.rate) < 1.0):
                try:
                    sdk_period = init_period()
                except Exception:
                    sdk_period = None

        frozen_q = None
        freeze_until = -1.0
        capture_period = 1.0 / self.capture_fps if self.capture_fps > 0 else 0.0
        next_capture = 0.0
        overruns: list[float] = []
        tick = 0
        started = time.perf_counter()
        try:
            while True:
                tick_start = time.perf_counter()
                if abort_event is not None and abort_event.is_set():
                    result.aborted = True
                    result.reason = 'operator abort'
                    break
                t = (tick + 1) * dt
                if t > total + dt:
                    break
                index = min(int(round((t - t_full[0]) * plan.rate)), len(q_full) - 1)
                q_command = q_full[max(index, 0)]

                if frozen_q is not None and t < freeze_until:
                    q_command = frozen_q
                elif frozen_q is not None:
                    frozen_q = None

                if reference is not None:
                    try:
                        q_command = reference.step(q_command)
                    except Exception as exc:
                        result.aborted = True
                        result.reason = f'continuous reference: {exc}'
                        break

                try:
                    sent = self.robot.servo_j(q_command, dt)
                except SafetyViolation as exc:
                    result.aborted = True
                    result.reason = f'safety rejection: {exc}'
                    break
                if not sent:
                    result.aborted = True
                    result.reason = 'servo_j failed (robot may have left servo mode)'
                    break

                if tick % 25 == 0 and self.robot.protective_stop():
                    result.aborted = True
                    result.reason = 'protective stop'
                    break

                while pending and pending[0]['t'] <= t:
                    event = pending.pop(0)
                    result.events.append(dict(t=event['t'], action=event['action'],
                                              width=event['width'], label=event['label']))
                    if on_event is not None:
                        on_event(event)
                    if event['action'] == 'hold':
                        if hold_callback is not None:
                            hold_callback(event)
                        else:
                            frozen_q, freeze_until = q_command, t + 0.5
                    elif not self.gripper_free and event['action'] in ('close', 'open', 'move'):
                        self._fire(event['action'], event['width'])
                        settle = (self.close_settle_s if event['action'] == 'close'
                                  else self.open_settle_s)
                        if settle > 0:
                            frozen_q, freeze_until = q_command, t + settle

                if capture_hook is not None and t >= next_capture:
                    # Pass the commanded joint target and scheduled gripper
                    # width so a collector can store the action that produced
                    # this observation, not just the measured state.
                    capture_hook(t, result.frames, q_command,
                                 plan.schedule.width_at(t))
                    result.captured += 1
                    next_capture += capture_period
                result.frames += 1
                tick += 1
                elapsed = time.perf_counter() - tick_start
                overruns.append(max(0.0, elapsed - dt))
                if realtime:
                    if sdk_period is not None:
                        try:
                            if self.robot.wait_period(sdk_period):
                                continue
                        except Exception:
                            pass
                        sdk_period = None
                    time.sleep(max(0.0, dt - elapsed))
        except KeyboardInterrupt:
            result.aborted = True
            result.reason = 'keyboard interrupt'
        finally:
            result.duration = time.perf_counter() - started
            if overruns:
                ordered = sorted(overruns)
                result.overrun_p95_ms = ordered[int(0.95 * (len(ordered) - 1))] * 1000.0
            if result.aborted:
                try:
                    self.robot.stop_motion()
                except Exception:
                    pass
        return result

    def _max_width(self) -> float:
        if self.gripper is not None:
            return float(getattr(self.gripper, 'max_width', 0.085))
        return 0.085

    def _fire(self, action: str, width: float | None) -> None:
        if self.gripper is None:
            return
        try:
            if action == 'close':
                self.gripper.close(wait=False)
            elif action == 'open':
                self.gripper.open(wait=False)
            elif width is not None:
                self.gripper.move(width, wait=False)
        except Exception as exc:
            self.gripper_errors.append(f'{action}: {exc}')
