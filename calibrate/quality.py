"""Robot-independent reprojection fitting and held-out verification.

The UR5 workflow supplies the protocol (27 stations, six frames, eight held-out
stations). No UR controller, camera SDK or motion planner is used here.
Transforms are metres and rotation vectors; pixel RMS is per 2-D corner.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..core.mathutils import (average_transforms, invert_transform, matrix_to_pose,
                              matrix_to_rotvec, pose_to_matrix)
from ..errors import CalibrationError
from .handeye import solve_hand_eye


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False,
                              default=lambda x: x.tolist() if isinstance(x, np.ndarray)
                              else float(x)), encoding='utf-8')


def normalize(doc):
    """Accept native views or the supplied UR5 views.json (ChArUco)."""
    if doc.get('schema') == 'robot_toolkit.calibration.views.v1':
        return doc
    if doc['board']['pattern'] != 'charuco':
        raise CalibrationError('legacy import currently supports ChArUco only')
    import cv2
    b = doc['board']
    scale = float(doc.get('scale', 1))
    board = cv2.aruco.CharucoBoard(
        (b['squares_x'], b['squares_y']), b['square_mm'] * scale / 1000,
        b['marker_mm'] * scale / 1000,
        cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, b['dict_name'])))
    grid = np.asarray(board.getChessboardCorners(), float)
    return dict(schema='robot_toolkit.calibration.views.v1', camera=doc['camera'],
                mode={'wrist': 'eye_in_hand', 'external': 'eye_to_hand'}[doc['mode']],
                board=b, image_size=doc['image_size'], intrinsics=doc['intrinsics'],
                views=[dict(v, object_points=grid[v['point_ids']].tolist(),
                            image_points=v['points'], base_tcp=v['tcp_pose'])
                       for v in doc['views']])


def board_pose(state, base_tcp):
    z = pose_to_matrix(base_tcp)
    x, y = np.asarray(state['extrinsic']), np.asarray(state['target'])
    if state['mode'] == 'eye_in_hand':
        return invert_transform(z @ x) @ y
    return invert_transform(x) @ z @ y


def project(state, view, require_positive_depth=True):
    import cv2
    t = board_pose(state, view['base_tcp'])
    objects = np.asarray(view['object_points'], float)
    if require_positive_depth and np.any((objects @ t[:3, :3].T + t[:3, 3])[:, 2] <= 0):
        raise CalibrationError('predicted board is behind camera')
    points, _ = cv2.projectPoints(objects, cv2.Rodrigues(t[:3, :3])[0],
                                 t[:3, 3], np.asarray(state['K']),
                                 np.asarray(state['dist']))
    return points.reshape(-1, 2)


def evaluate(state, views):
    rows, errors, depths = [], [], []
    for i, v in enumerate(views):
        residual = project(state, v) - np.asarray(v['image_points'])
        if not np.isfinite(residual).all():
            raise CalibrationError('non-finite reprojection')
        rms = float(np.sqrt(np.mean(np.sum(residual**2, axis=1))))
        row = dict(index=v.get('index', i), rms_px=rms)
        measured = v.get('point_depth_mm')
        if measured is not None:
            d = np.asarray([np.nan if x is None else x for x in measured])
            t = board_pose(state, v['base_tcp'])
            predicted = (np.asarray(v['object_points']) @ t[:3, :3].T + t[:3, 3])[:, 2]*1000
            mask = np.isfinite(d) & (d > 0)
            if mask.sum() >= 4:
                row['depth_median_abs_mm'] = float(np.median(np.abs(d[mask]-predicted[mask])))
                depths.append(row['depth_median_abs_mm'])
        rows.append(row)
        errors.extend(residual)
    if not rows:
        raise CalibrationError('no views to evaluate')
    rms = [r['rms_px'] for r in rows]
    return dict(views=len(rows), mean_rms_px=float(np.mean(rms)),
                worst_rms_px=float(np.max(rms)),
                pooled_rms_px=float(np.sqrt(np.mean(np.sum(np.asarray(errors)**2, axis=1)))),
                bias_px=np.mean(errors, axis=0).tolist(), per_view=rows,
                depth_error_mm_median=float(np.median(depths)) if depths else None)


def _fit(doc, views, source):
    import cv2
    from scipy.optimize import least_squares
    if len(views) < 6:
        raise CalibrationError('need at least six views for fitting')
    k = np.asarray(doc['intrinsics']['K'], float).copy()
    dist = np.asarray(doc['intrinsics']['dist'], float).copy()
    if source == 'calibrated':
        _, k, dist, _, _ = cv2.calibrateCamera(
            [np.asarray(v['object_points'], np.float32) for v in views],
            [np.asarray(v['image_points'], np.float32) for v in views],
            tuple(doc['image_size']), k, None, flags=cv2.CALIB_USE_INTRINSIC_GUESS)
    ms = []
    for v in views:
        ok, rv, tv = cv2.solvePnP(np.asarray(v['object_points'], float),
                                  np.asarray(v['image_points'], float), k, dist)
        if not ok:
            raise CalibrationError('PnP failed')
        ms.append(pose_to_matrix(np.r_[tv.ravel(), rv.ravel()]))
    zs = [pose_to_matrix(v['base_tcp']) for v in views]
    mode = doc['mode']
    # Invert robot poses to obtain the fixed-camera AX=XB equation.
    x = solve_hand_eye('eye_in_hand',
                      zs if mode == 'eye_in_hand' else [invert_transform(z) for z in zs],
                      ms).transform
    ys = [(z @ x @ m if mode == 'eye_in_hand' else invert_transform(z) @ x @ m)
          for z, m in zip(zs, ms)]
    y = average_transforms(ys)
    state = dict(mode=mode, extrinsic=x, target=y, K=k, dist=dist.ravel(), source=source)
    active = list(views)
    rejected = []
    for iteration in range(3):
        def residual(p):
            candidate = dict(state, extrinsic=pose_to_matrix(p[:6]), target=pose_to_matrix(p[6:]))
            return np.concatenate([(project(candidate, v, require_positive_depth=False)-v['image_points']).ravel()
                                   for v in active])
        opt = least_squares(residual, np.r_[matrix_to_pose(state['extrinsic']),
                                           matrix_to_pose(state['target'])],
                            loss='soft_l1', f_scale=1.2, max_nfev=160)
        if not opt.success or not np.isfinite(opt.x).all():
            raise CalibrationError(f'nonlinear fit did not converge: {opt.message}')
        state.update(extrinsic=pose_to_matrix(opt.x[:6]), target=pose_to_matrix(opt.x[6:]))
        rms = np.array([r['rms_px'] for r in evaluate(state, active)['per_view']])
        keep = rms <= max(1.0, 3*float(np.median(rms)))
        if keep.all() or iteration == 2:
            break
        if keep.sum() < 6:
            raise CalibrationError('outlier rejection leaves fewer than six views')
        rejected.extend(v['index'] for v, flag in zip(active, keep) if not flag)
        active = [v for v, flag in zip(active, keep) if flag]
    return state, dict(evaluate(state, active), rejected=rejected)


def solve(doc, source='auto', min_views=27):
    doc = normalize(doc)
    views = doc['views']
    _distinct(views)
    if len(views) < max(12, min_views):
        raise CalibrationError(f'need {max(12, min_views)} views; got {len(views)}')
    candidates, fits = {}, {}
    for name in (['factory', 'calibrated'] if source == 'auto' else [source]):
        if name not in ('factory', 'calibrated'):
            raise CalibrationError('intrinsics must be auto, factory or calibrated')
        try:
            scores = []
            for parity in (0, 1):
                train = views[parity::2]
                test = views[1-parity::2]
                state, _ = _fit(doc, train, name)
                scores.extend(r['rms_px']**2 for r in evaluate(state, test)['per_view'])
            state, report = _fit(doc, views, name)
            # min_views is the capture requirement. Outlier rejection is
            # allowed to reduce the fit set, but never below six (_fit).
            candidates[name] = dict(cv_rms_px=float(np.sqrt(np.mean(scores))),
                                    train_rms_px=report['pooled_rms_px'])
            fits[name] = state, report
        except (CalibrationError, ValueError, np.linalg.LinAlgError) as exc:
            candidates[name] = {'error': str(exc)}
    if not fits:
        raise CalibrationError(f'all candidates failed: {candidates}')
    chosen = min(fits, key=lambda name: candidates[name]['cv_rms_px'])
    state, report = fits[chosen]
    state.update(camera=doc['camera'], serial=doc.get('serial'), board=doc.get('board'),
                 image_size=doc['image_size'],
                 training_poses=[v['base_tcp'] for v in views])
    report.update(candidates=candidates, chosen_source=chosen, independent_verified=False,
                  views_captured=len(views), required_capture_views=min_views,
                  note='Training/CV residuals are not absolute positioning accuracy.')
    return state, report


def _distinct(views):
    for i, view in enumerate(views):
        if any(np.allclose(view['base_tcp'], other['base_tcp'], atol=1e-5, rtol=0)
               for other in views[:i]):
            raise CalibrationError('duplicate stationary pose cannot count as a new view')


def validate_identity(state, doc):
    """Check the camera, board and pose convention before capture or evaluation."""
    for key in ('camera', 'mode', 'board', 'image_size'):
        if state.get(key) != doc.get(key):
            raise CalibrationError(f'verification {key} does not match solution')
    if state.get('serial') and state['serial'] != doc.get('serial'):
        raise CalibrationError('verification camera serial does not match solution')


def verify(state, doc, min_views=8):
    doc = normalize(doc)
    _distinct(doc['views'])
    validate_identity(state, doc)
    for v in doc['views']:
        if any(np.allclose(v['base_tcp'], p, atol=1e-6, rtol=0)
               for p in state.get('training_poses', [])):
            raise CalibrationError('verification reuses a training pose')
    report = evaluate(state, doc['views'])
    good, acceptable = (1., 2.) if state['mode'] == 'eye_in_hand' else (1.5, 3.)
    report['thresholds_px'] = dict(good=good, acceptable=acceptable)
    report['status'] = ('insufficient' if len(doc['views']) < min_views else
                        'pass' if report['mean_rms_px'] <= good else
                        'warning' if report['mean_rms_px'] <= acceptable else 'fail')
    report['independent_verified'] = report['status'] == 'pass'
    depth_good, depth_ok = (4., 8.) if state['mode'] == 'eye_in_hand' else (6., 12.)
    depth = report['depth_error_mm_median']
    report['depth_status'] = ('unavailable' if depth is None else
                              'pass' if depth <= depth_good else
                              'warning' if depth <= depth_ok else 'fail')
    report['depth_thresholds_mm'] = dict(good=depth_good, acceptable=depth_ok)
    report['note'] = 'Board must remain fixed; pixel/depth consistency is not absolute TCP accuracy.'
    if len(doc['views']) >= min_views:
        try:
            fixed = dict(doc, intrinsics=dict(K=state['K'], dist=state['dist']))
            redo, _ = _fit(fixed, doc['views'], 'factory')
            delta = invert_transform(np.asarray(state['extrinsic'])) @ redo['extrinsic']
            report['drift'] = dict(trans_mm=float(np.linalg.norm(delta[:3, 3])*1000),
                                  angle_deg=float(np.linalg.norm(matrix_to_rotvec(delta[:3, :3]))*180/np.pi))
        except (CalibrationError, ValueError, np.linalg.LinAlgError) as exc:
            report['drift_error'] = str(exc)
            report['independent_verified'] = False
            report['status'] = 'inconclusive'
    return report


def save_overlays(state, doc, output):
    """Green measured corners, red projected corners and residual segments."""
    import cv2
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for i, view in enumerate(doc['views']):
        width, height = doc['image_size']
        image = view.get('_rgb')
        canvas = (cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if image is not None
                  else np.full((height, width, 3), 35, np.uint8))
        for observed, predicted in zip(view['image_points'], project(state, view)):
            a, b = tuple(np.round(observed).astype(int)), tuple(np.round(predicted).astype(int))
            cv2.line(canvas, a, b, (0, 255, 255), 1)
            cv2.circle(canvas, a, 3, (0, 220, 0), 1)
            cv2.drawMarker(canvas, b, (0, 0, 255), cv2.MARKER_CROSS, 7, 1)
        cv2.imwrite(str(output / f'view_{i:03d}.png'), canvas)
