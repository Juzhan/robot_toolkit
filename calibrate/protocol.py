"""Shared capture/report protocol; depends only on Camera and Robot contracts."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ..core.mathutils import (average_transforms, matrix_to_pose, pose_to_matrix,
                              matrix_to_rotvec)
from ..errors import CalibrationError
from .quality import write_json, solve, verify, save_overlays
from .store import CalibrationStore
from .identity import calibration_identity


def capture_station(session, frames=6, timeout=5.0):
    """Aggregate distinct, fresh frames at a stationary pose into one view."""
    camera, robot, detector = session.camera, session.robot, session.detector
    deadline = time.monotonic() + timeout
    started = time.perf_counter()
    last = None
    records = []
    joints = []
    while len(records) < frames and time.monotonic() < deadline:
        frame = camera.snapshot()
        if frame.rgb is None or frame.age() > .5 or frame.t <= started or frame.t == last:
            time.sleep(.02)
            continue
        last = frame.t
        obs = detector.detect(frame.rgb)
        if obs is None:
            time.sleep(.02)
            continue
        state = robot.state()
        tcp = state.tcp
        if tcp is None:
            raise CalibrationError('robot has no TCP feedback')
        records.append((frame, obs, np.asarray(tcp)))
        if getattr(state, 'q', None) is not None:
            joints.append(np.asarray(state.q).copy())
    if len(records) < frames:
        return None
    points = {}
    for frame, obs, _ in records:
        for obj, pixel in zip(obs.object_points, obs.image_points):
            key = tuple(obj)
            depth = None
            if frame.depth is not None:
                x, y = np.round(pixel).astype(int)
                h, w = frame.depth.shape
                if 0 <= x < w and 0 <= y < h:
                    window = frame.depth[max(0, y-2):y+3, max(0, x-2):x+3]
                    valid = window[window > 0]
                    if valid.size:
                        depth = float(np.median(valid))
            points.setdefault(key, []).append((pixel, depth))
    keys = [k for k, v in points.items() if len(v) >= max(3, frames//2)]
    if len(keys) < getattr(detector, 'min_corners', 6):
        return None
    poses = [r[2] for r in records]
    mean_pose = matrix_to_pose(average_transforms([pose_to_matrix(p) for p in poses]))
    spread = max(np.linalg.norm(p[:3]-mean_pose[:3]) for p in poses)*1000
    rotation = pose_to_matrix(mean_pose)[:3, :3]
    angular_spread = max(np.linalg.norm(matrix_to_rotvec(
        rotation.T @ pose_to_matrix(p)[:3, :3])) for p in poses)*180/np.pi
    if spread > 1.0 or angular_spread > .25:
        raise CalibrationError(f'arm moved during capture ({spread:.2f} mm, {angular_spread:.2f} deg)')
    depth = []
    for k in keys:
        valid = [d for _, d in points[k] if d is not None]
        depth.append(float(np.median(valid)) if valid else None)
    return dict(object_points=np.asarray(keys).tolist(),
                image_points=[np.median([p for p, _ in points[k]], axis=0).tolist() for k in keys],
                base_tcp=mean_pose.tolist(), frames_used=len(records), pose_spread_mm=spread,
                pose_spread_deg=angular_spread,
                actual_q_rad=np.median(joints, axis=0).tolist() if joints else None,
                point_depth_mm=depth, _rgb=records[-1][0].rgb)


def document(session, cfg, views):
    return dict(schema='robot_toolkit.calibration.views.v1',
                **calibration_identity(session.camera, cfg),
                intrinsics=dict(K=session.camera.intrinsics.tolist(),
                    dist=np.asarray(session.camera.distortion).tolist()),
                views=views)


def finish(session, cfg, path, views):
    """Always persist observations; never promote insufficient data to a solution."""
    import json
    import cv2
    directory = Path(path).parent
    directory.mkdir(parents=True, exist_ok=True)
    doc = document(session, cfg, views)
    for i, view in enumerate(views):
        view['index'] = i
        if '_rgb' in view:
            image_path = directory / f'capture_{i:03d}.png'
            cv2.imwrite(str(image_path), cv2.cvtColor(view['_rgb'], cv2.COLOR_RGB2BGR))
            view['image'] = image_path.name
    serializable = dict(doc, views=[{k: v for k, v in row.items() if k != '_rgb'} for row in views])
    write_json(directory / 'views.json', serializable)
    required = cfg.verify_samples if cfg.action == 'verify' else cfg.min_samples
    if len(views) < required:
        write_json(directory / 'incomplete.json',
                   dict(status='insufficient', views=len(views), required=required))
        print(f'样本不足 {len(views)}/{required}；已保留原始观测：{directory}')
        return 1
    if cfg.action == 'verify':
        state = json.loads(Path(cfg.solution_path).read_text())
        report = verify(state, doc, min_views=cfg.verify_samples)
        write_json(directory / 'verify.json', report)
    else:
        training = dict(doc, views=views[:cfg.min_samples] if cfg.verify_after else views)
        state, report = solve(training, source=cfg.intrinsics_mode, min_views=cfg.min_samples)
        write_json(directory / 'solution.json', state)
        write_json(directory / 'report.json', report)
        store = CalibrationStore(path)
        store.set_intrinsics(session.camera.name, state['K'], state['dist'], doc['image_size'])
        entry = store.set_hand_eye(session.camera.name, cfg.eye, state['extrinsic'])
        entry.meta.update(quality_report='report.json', independent_verified=False,
                          serial=doc['serial'], board=doc['board'])
        if cfg.verify_after:
            heldout = dict(doc, views=views[cfg.min_samples:])
            if heldout['views']:
                validation = verify(state, heldout, min_views=cfg.verify_samples)
                save_overlays(state, heldout, directory / 'verify_overlays')
            else:
                validation = dict(status='insufficient', views=0, independent_verified=False)
            write_json(directory / 'verify.json', validation)
            report['verification'] = validation
            entry.meta['independent_verified'] = validation['independent_verified']
            write_json(directory / 'report.json', report)
        store.save()
    save_overlays(state, doc, directory / 'overlays')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f'结果：{directory}')
    status = report.get('verification', report).get('status', 'pass')
    return 0 if status == 'pass' else 1
