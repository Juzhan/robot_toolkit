"""Execute a fixed view plan, preserving roles when a capture fails."""
import time
from pathlib import Path

import numpy as np

from ..errors import CalibrationError
from .protocol import capture_station, finish
from .quality import write_json
from .view_plan import load_plan
from .identity import calibration_identity


def run_plan(robot, camera, session, cfg, path):
    if not cfg.quality_protocol or cfg.action != 'solve' or not cfg.verify_after:
        raise CalibrationError('fixed plan requires quality protocol, solve and verify_after')
    if getattr(robot, 'workspace', None) is None:
        raise CalibrationError('fixed plan requires workspace guard')
    identity = calibration_identity(camera, cfg)
    plan = load_plan(cfg.pose_plan_path, identity, robot.workspace,
                     cfg.min_samples, cfg.verify_samples)
    rows, failure = [], None
    try:
        robot.connect()
        deadline = time.monotonic()+3
        state = robot.state()
        while state.tcp is None and time.monotonic() < deadline:
            time.sleep(.02)
            state = robot.state()
        robot.workspace.check_feedback(state)
        targets = [plan['anchor_q_rad']] + [v['q_rad'] for v in plan['views']]
        previous = state.q
        for i, target in enumerate(targets):
            if robot.guard.limits.violates(target):
                raise CalibrationError('plan target exceeds joint limits')
            if i > 0 and not np.allclose(robot.forward_kinematics(target),
                                        plan['views'][i-1]['predicted_tcp'], atol=1e-4, rtol=0):
                raise CalibrationError('robot FK/TCP model differs from saved plan')
            robot.workspace.check_path(previous, target,
                clearance_m=0 if i == 0 else plan['clearance_m'])
            previous = target
        camera.open()
        deadline = time.monotonic()+8
        while time.monotonic() < deadline:
            frame = camera.snapshot()
            if frame.rgb is not None and frame.age() < .5 and session.detector.detect(frame.rgb) is not None:
                break
            time.sleep(.05)
        else:
            raise CalibrationError('board not visible before motion')
        if not robot.prepare_motion():
            raise CalibrationError('controller did not become ready for motion')
        if not robot.move_j(plan['anchor_q_rad'], speed=.1, wait=True):
            raise CalibrationError('anchor move did not settle')
        for index, row in enumerate(plan['views']):
            if not robot.move_j(row['q_rad'], speed=cfg.speed_scale, wait=True):
                raise CalibrationError(f'{row["id"]}: move did not settle')
            time.sleep(cfg.settle_s)
            view = capture_station(session, frames=cfg.frames_per_pose)
            if view is None:
                raise CalibrationError(f'{row["id"]}: board not detected; fixed sequence aborted')
            if view['actual_q_rad'] is None:
                raise CalibrationError('joint feedback missing during capture')
            error = float(np.max(np.abs(np.asarray(view['actual_q_rad'])-row['q_rad'])))
            if error > .02:
                raise CalibrationError(f'{row["id"]}: joint repeat error {error:.4f} rad')
            view.update(plan_id=plan['plan_id'], view_id=row['id'], role=row['role'],
                        target_q_rad=row['q_rad'], max_joint_error_rad=error)
            rows.append(view)
            print(f'固定清单 {index+1}/{len(plan["views"])} {row["role"]} '
                  f'到位偏差 {np.degrees(error):.3f}°', flush=True)
    except (Exception, KeyboardInterrupt) as exc:
        failure = str(exc) or 'interrupted'
        robot.stop_motion()
        print(f'固定清单停止：{failure}', flush=True)
    finally:
        try:
            camera.close()
        finally:
            robot.disconnect()
    write_json(Path(path).parent/'execution.json', dict(plan_id=plan['plan_id'],
        source_digest=plan['source_digest'], plan_path=str(Path(cfg.pose_plan_path).resolve()),
        frames_per_pose=cfg.frames_per_pose, settle_s=cfg.settle_s,
        captured=len(rows), failure=failure))
    write_json(Path(path).parent/'plan.json', plan)
    if camera.intrinsics is None:
        return 1
    return finish(session, cfg, path, rows)
