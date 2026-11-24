"""Unattended calibration capture shared by every mounting and pose source.

A pose source (seeded sweep around the current joints, or a saved view plan)
produces stations; ``screen`` applies the same safety checks to both, and
``run_stations`` executes them: board precheck, motion preparation, optional
gripper clamp, move, settle, slip check, capture. Eye-in-hand and eye-to-hand
differ only in the solve, which ``protocol.finish`` already handles.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from ..core.workspace_guard import held_object_slack
from ..errors import CalibrationError
from .identity import calibration_identity
from .protocol import capture_station, finish
from .quality import write_json
from .view_plan import load_plan


def wait_for_feedback(robot, timeout=3.0):
    """First state with joints and a populated TCP (Piper's first frame can be zero)."""
    deadline = time.monotonic() + timeout
    state = robot.state()
    while ((state.q is None or state.tcp is None or not np.any(state.tcp[:3]))
           and time.monotonic() < deadline):
        time.sleep(.02)
        state = robot.state()
    if state.q is None or state.tcp is None:
        raise CalibrationError('robot joint/TCP feedback unavailable')
    return state


def sweep_poses(start_q, cfg):
    """Explicit poses when configured, otherwise a seeded sweep around ``start_q``."""
    if cfg.poses:
        return [[float(value) for value in pose] for pose in cfg.poses]
    base = np.asarray(start_q, dtype=float)
    rng = np.random.default_rng(int(cfg.pose_seed))
    delta = max(0.02, float(cfg.pose_delta_rad))
    poses = []
    for _ in range(max(3, int(cfg.pose_count))):
        offset = rng.uniform(-delta, delta, size=base.shape)
        offset[0] = 0.0                      # keep the base yaw
        poses.append((base + offset).tolist())
    return poses


def screen(robot, state, poses, cfg, clearance_m=None, first_clearance_m=None):
    """Check each pose from the previously accepted one; return (accepted, reasons).

    ``reasons[i]`` is None for accepted poses. Held-object height is relative to
    ``cfg.held_object_reference_q`` when set, else to ``state`` (the run's start).
    """
    clearance_m = cfg.path_clearance_m if clearance_m is None else clearance_m
    workspace = getattr(robot, 'workspace', None)
    guard = getattr(robot, 'guard', None)
    previous, accepted, reasons = np.asarray(state.q, float), [], []
    reference_tcp = (robot.forward_kinematics(cfg.held_object_reference_q)
                     if cfg.held_object_reference_q else state.tcp)
    for pose in poses:
        reason = None
        if guard is not None and guard.limits.violates(pose):
            reason = 'joint_limits'
        elif workspace is not None:
            try:
                workspace.check_path(previous, pose, clearance_m=(
                    first_clearance_m if first_clearance_m is not None and not accepted
                    else clearance_m))
            except Exception:
                reason = 'workspace'
        if reason is None and cfg.held_object_radius_m > 0:
            slack = held_object_slack(robot.forward_kinematics, reference_tcp, previous, pose,
                                      cfg.held_object_radius_m)
            if slack < -cfg.held_object_drop_m - 1e-4:
                reason = 'held_object_height'
        reasons.append(reason)
        if reason is None:
            accepted.append(pose)
            previous = np.asarray(pose, float)
    return accepted, reasons


def _count(reasons):
    counts = {}
    for reason in reasons:
        if reason is not None:
            counts[reason] = counts.get(reason, 0) + 1
    return counts


def sweep_stations(robot, cfg, state, need, envelope=None):
    """Seeded sweep; unsafe poses are dropped, missed captures are skipped."""
    poses = sweep_poses(state.q, cfg)
    if getattr(robot, 'workspace', None) is None and envelope is not None:
        poses = [pose for pose in poses if envelope.contains_q(pose)]
    accepted, reasons = screen(robot, state, poses, cfg)
    rejected = _count(reasons)
    print(f'自动扫描候选 {len(poses)}，通过 {len(accepted)}；剔除 {rejected or "无"}')
    if len(accepted) < need:
        raise CalibrationError(f'only {len(accepted)} safe sweep poses; need {need}')
    stations = [dict(id=f'sweep_{i:03d}', q_rad=list(q)) for i, q in enumerate(accepted)]
    return dict(source='sweep', stations=stations, start_q=None, abort_on_miss=False,
                max_joint_error_rad=None, rejected=rejected)


def plan_stations(robot, camera, cfg, state):
    """Saved plan; any unsafe station or missed capture stops the run."""
    if not cfg.quality_protocol or cfg.action != 'solve' or not cfg.verify_after:
        raise CalibrationError('fixed plan requires quality protocol, solve and verify_after')
    if getattr(robot, 'workspace', None) is None:
        raise CalibrationError('fixed plan requires workspace guard')
    plan = load_plan(cfg.pose_plan_path, calibration_identity(camera, cfg), robot.workspace,
                     cfg.min_samples, cfg.verify_samples)
    for row in plan['views']:
        if not np.allclose(robot.forward_kinematics(row['q_rad']), row['predicted_tcp'],
                           atol=1e-4, rtol=0):
            raise CalibrationError('robot FK/TCP model differs from saved plan')
    targets = [plan['anchor_q_rad']] + [row['q_rad'] for row in plan['views']]
    accepted, reasons = screen(robot, state, targets, cfg, clearance_m=plan['clearance_m'],
                               first_clearance_m=0.0)
    if len(accepted) != len(targets):
        raise CalibrationError(f'saved plan fails current safety checks: {_count(reasons)}')
    stations = [dict(id=row['id'], q_rad=row['q_rad'], role=row['role'], plan_id=plan['plan_id'])
                for row in plan['views']]
    return dict(source='plan', plan=plan, stations=stations, start_q=plan['anchor_q_rad'],
                abort_on_miss=True, max_joint_error_rad=.02)


def _wait_board(camera, detector, timeout=8.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        frame = camera.snapshot()
        if frame.rgb is not None and frame.age() < .5 and detector.detect(frame.rgb) is not None:
            return True
        time.sleep(.05)
    return False


def _legacy_capture(session, cfg):
    """Single-frame capture used when the quality protocol is off."""
    frame = session.camera.snapshot()
    if frame.rgb is None:
        return None
    try:
        if cfg.kind == 'intrinsics':
            observation = session.detector.detect(frame.rgb)
            if observation is None:
                return None
            session.views.append(observation.view)
            session.images.append(frame.rgb)
            return {}
        return None if session.capture(image=frame.rgb) is None else {}
    except Exception as exc:
        print(f'单帧采集失败：{type(exc).__name__}: {exc}', flush=True)
        return None


def run_stations(robot, camera, session, cfg, run, need):
    """Execute stations; returns (views, failure). Robot must already be connected."""
    views, failure = [], None
    stations = run['stations']
    try:
        if not camera.opened:
            camera.open()
        if (cfg.require_initial_board or run['abort_on_miss']) and not _wait_board(
                camera, session.detector):
            raise CalibrationError('board not visible before motion; nothing was moved')
        print('预检通过：当前位姿已检测到标定板。', flush=True)
        if not robot.prepare_motion():
            raise CalibrationError('controller did not become ready for motion')
        clamp_width = None
        if cfg.clamp_force_n > 0:
            robot.move_gripper(0.0, force=float(cfg.clamp_force_n))
            time.sleep(1.0)
            clamp_width = float(robot.gripper_width)
            print(f'夹爪夹紧：{cfg.clamp_force_n:g} N，开度 {clamp_width * 1000:.1f} mm', flush=True)
            if clamp_width < 0.002:
                raise CalibrationError('gripper closed fully; nothing is held')
        # Held-object screening assumes the host's joint-linear interpolation.
        native = cfg.held_object_radius_m <= 0
        if run['start_q'] is not None and not robot.move_j(
                run['start_q'], speed=.1, wait=True, use_native=native):
            raise CalibrationError('anchor move did not settle')
        for index, station in enumerate(stations, 1):
            tag = f'[{index}/{len(stations)}] {station["id"]}'
            if getattr(robot, 'protective_stop', lambda: False)():
                raise CalibrationError('protective stop active')
            if not robot.move_j(station['q_rad'], speed=float(cfg.speed_scale), wait=True,
                                use_native=native):
                raise CalibrationError(f'{station["id"]}: move did not settle')
            time.sleep(max(0.0, float(cfg.settle_s)))
            if clamp_width is not None:
                slip = abs(float(robot.gripper_width) - clamp_width)
                if slip > cfg.clamp_slip_m:
                    raise CalibrationError(f'gripper width changed {slip * 1000:.1f} mm; '
                                           'held object may be slipping')
            if cfg.quality_protocol:
                view = capture_station(session, frames=cfg.frames_per_pose,
                                       timeout=cfg.capture_timeout_s,
                                       max_jitter_px=cfg.board_jitter_px or None)
            else:
                view = _legacy_capture(session, cfg)
            if view is None:
                if run['abort_on_miss']:
                    raise CalibrationError(f'{station["id"]}: board not captured; '
                                           'fixed sequence aborted')
                print(f'{tag} 未采到（板子未稳定或未检测到），跳过', flush=True)
                continue
            if cfg.quality_protocol:
                actual = view.get('actual_q_rad')
                error = (None if actual is None else
                         float(np.max(np.abs(np.asarray(actual) - station['q_rad']))))
                limit = run['max_joint_error_rad']
                if limit is not None and (error is None or error > limit):
                    raise CalibrationError(f'{station["id"]}: joint repeat error {error}')
                view.update({k: v for k, v in station.items() if k not in ('id', 'q_rad')},
                            view_id=station['id'], target_q_rad=station['q_rad'],
                            max_joint_error_rad=error)
                if 'role' not in station:
                    view['role'] = 'train' if len(views) < cfg.min_samples else 'verify'
                print(f'{tag} 有效 {len(views) + 1}/{need}，'
                      f'板抖动 {view["board_jitter_px"]:.2f} px', flush=True)
            else:
                print(f'{tag} 有效 {len(views) + 1}/{need}', flush=True)
            views.append(view)
            if len(views) >= need:
                break
    except (Exception, KeyboardInterrupt) as exc:
        failure = f'{type(exc).__name__}: {exc}' if str(exc) else type(exc).__name__
        try:
            robot.stop_motion()
        except Exception:
            pass
        print(f'自动标定停止：{failure}', flush=True)
    finally:
        try:
            camera.close()
        except Exception:
            pass
    return views, failure


def run_auto(robot, camera, session, cfg, path, need, envelope=None):
    """Plan or sweep, execute, then persist and solve. Returns a process exit code."""
    try:
        robot.connect()
        state = wait_for_feedback(robot)
        workspace = getattr(robot, 'workspace', None)
        if workspace is not None:
            workspace.check_feedback(state)
        run = (plan_stations(robot, camera, cfg, state) if cfg.pose_plan_path
               else sweep_stations(robot, cfg, state, need, envelope))
    except Exception as exc:
        print(f'自动标定预检失败，未运动：{exc}')
        try:
            robot.disconnect()
        except Exception:
            pass
        return 2
    print(f'自动标定（{run["source"]}）：{len(run["stations"])} 个位姿，速度 '
          f'{cfg.speed_scale:g}，每处停 {cfg.settle_s:g}s；目标 ≥{need} 组', flush=True)
    try:
        views, failure = run_stations(robot, camera, session, cfg, run, need)
    finally:
        try:
            robot.disconnect()
        except Exception:
            pass
    if not cfg.quality_protocol:
        count = len(session.views) if cfg.kind == 'intrinsics' else len(session.base_tcp)
        if count < need:
            print(f'样本不足（{count}），没有保存')
            return 1
        summary = session.save(path, camera_name=camera.name, apply=bool(cfg.apply))
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
        return 0
    directory = Path(path).parent
    write_json(directory / 'execution.json', dict(
        source=run['source'], stations=len(run['stations']), captured=len(views),
        failure=failure, rejected=run.get('rejected'), frames_per_pose=cfg.frames_per_pose,
        settle_s=cfg.settle_s, board_jitter_px=cfg.board_jitter_px,
        held_object_radius_m=cfg.held_object_radius_m,
        held_object_drop_m=cfg.held_object_drop_m, clamp_force_n=cfg.clamp_force_n,
        plan_id=run['plan']['plan_id'] if 'plan' in run else None,
        plan_path=str(Path(cfg.pose_plan_path).resolve()) if 'plan' in run else None))
    if 'plan' in run:
        write_json(directory / 'plan.json', run['plan'])
    if camera.intrinsics is None:
        return 1
    return finish(session, cfg, path, views)
