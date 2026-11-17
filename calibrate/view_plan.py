"""Deterministic, coarse-calibration-guided joint view selection.

Uses FK and a workspace guard supplied by the caller, never a robot SDK.
The coarse model predicts visibility; it is not evidence of calibration quality.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from ..core.mathutils import matrix_to_rotvec
from ..errors import CalibrationError, SafetyViolation
from .quality import board_pose, project, validate_identity, write_json


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def workspace_signature(workspace):
    return dict(lower=workspace.box.lower.tolist(), upper=workspace.box.upper.tolist(),
                lower_enabled=workspace.lower_enabled.tolist(),
                upper_enabled=workspace.upper_enabled.tolist(), step_rad=workspace.step_rad)


def predict(state, tcp, image_margin_px=24):
    b = state['board']
    if b['pattern'] != 'charuco':
        raise CalibrationError('view planner currently supports ChArUco boards')
    width, height = np.asarray(b['squares']) * b['square_size']
    # Include the full board perimeter, not just detected inner corners.
    objects = np.array([[0, 0, 0], [width, 0, 0], [width, height, 0], [0, height, 0]])
    view = dict(base_tcp=tcp, object_points=objects)
    corners = project(state, view)
    w, h = state['image_size']
    if (np.any(corners < image_margin_px) or
            np.any(corners > np.array([w, h])-image_margin_px)):
        raise CalibrationError('predicted board outside image margin')
    t = board_pose(state, tcp)
    distance = float(np.linalg.norm(t[:3, 3]))
    normal = t[:3, 2]
    tilt = float(np.degrees(np.arccos(np.clip(abs(normal @ t[:3, 3])/distance, 0, 1))))
    area = abs(np.dot(corners[:, 0], np.roll(corners[:, 1], 1)) -
               np.dot(corners[:, 1], np.roll(corners[:, 0], 1))) / (2*w*h)
    if not .06 <= area <= .75 or tilt > 55:
        raise CalibrationError('board too small, too large or too oblique')
    rotation = matrix_to_rotvec(t[:3, :3])
    center = corners.mean(axis=0)/np.array([w, h])
    feature = np.r_[rotation/.35, center/.2, np.log(distance)/.25]
    return dict(corners_px=corners.tolist(), center_normalized=center.tolist(),
                distance_m=distance, tilt_deg=tilt, image_area_fraction=float(area),
                feature=feature.tolist())


def generate(state, anchor, fk, joint_limits, workspace, *, count=27, verify_count=8,
             candidate_count=4096, delta_rad=.25, clearance_m=.01, image_margin_px=24):
    from scipy.stats import qmc
    if count < 12 or verify_count < 8 or candidate_count < count+verify_count:
        raise ValueError('need >=12 training, >=8 verification and enough candidates')
    if workspace is None or not 0 < delta_rad <= .5:
        raise ValueError('workspace and bounded joint sampling are required')
    anchor = np.asarray(anchor, float)
    if joint_limits.violates(anchor):
        raise SafetyViolation('anchor violates joint limits')
    workspace.check_path(anchor, anchor, clearance_m=clearance_m)
    sampler = qmc.Halton(len(anchor), scramble=False)
    # Deterministic joint-space proposals; selection is in camera-view space.
    qs = anchor + (sampler.random(candidate_count)*2-1)*delta_rad
    candidates, rejected = [], dict(joint=0, visibility=0, workspace=0)
    for q in qs:
        if joint_limits.violates(q):
            rejected['joint'] += 1
            continue
        tcp = np.asarray(fk(q))
        try:
            prediction = predict(state, tcp, image_margin_px)
        except CalibrationError:
            rejected['visibility'] += 1
            continue
        try:
            workspace.check_path(anchor, q, clearance_m=clearance_m)
        except SafetyViolation:
            rejected['workspace'] += 1
            continue
        candidates.append(dict(q_rad=q.tolist(), predicted_tcp=tcp.tolist(), **prediction))
    if len(candidates) < count+verify_count:
        raise CalibrationError(f'only {len(candidates)} valid full-board views; '
                               f'need {count+verify_count}; rejected={rejected}')
    features = np.asarray([c['feature'] for c in candidates])
    selected, available = [], set(range(len(candidates)))
    previous = anchor
    min_distance = np.full(len(candidates), np.inf)
    for index in range(count+verify_count):
        # Start with the most central visible view; then farthest-point sampling.
        if index == 0:
            scores = -np.linalg.norm(np.asarray([c['center_normalized'] for c in candidates])-.5, axis=1)
        else:
            scores = min_distance
        ordered = sorted(available, key=lambda j: (-scores[j], j))
        chosen = None
        for j in ordered:
            if selected and min_distance[j] < .15:
                continue
            try:
                workspace.check_path(previous, candidates[j]['q_rad'], clearance_m=clearance_m)
            except SafetyViolation:
                continue
            chosen = j
            break
        if chosen is None:
            raise CalibrationError('not enough diverse views with safe connecting paths')
        available.remove(chosen)
        row = dict(candidates[chosen], id=f'view_{index:03d}',
                   role='train' if index < count else 'verify')
        selected.append(row)
        previous = np.asarray(row['q_rad'])
        min_distance = np.minimum(min_distance, np.linalg.norm(features-features[chosen], axis=1))
    plan = dict(schema='robot_toolkit.view_plan.v1', source_digest=digest(state),
                camera=state['camera'], serial=state.get('serial'), mode=state['mode'],
                board=state['board'], image_size=state['image_size'],
                anchor_q_rad=anchor.tolist(), workspace=workspace_signature(workspace),
                clearance_m=clearance_m, image_margin_px=image_margin_px,
                train_count=count, verify_count=verify_count,
                candidate_count=candidate_count, accepted_candidates=len(candidates),
                rejected=rejected, delta_rad=delta_rad, views=selected,
                note='Predicted visibility only; full-board border and TCP clearance required. '
                     'Board and camera mounting must remain fixed. Not whole-arm collision checking.')
    plan['plan_id'] = digest(plan)
    return plan


def load_plan(path, identity, workspace, train_count, verify_count):
    plan = json.loads(Path(path).read_text())
    payload = {k: v for k, v in plan.items() if k != 'plan_id'}
    if plan.get('schema') != 'robot_toolkit.view_plan.v1' or digest(payload) != plan.get('plan_id'):
        raise CalibrationError('invalid or modified pose plan')
    validate_identity(plan, identity)
    if plan['workspace'] != workspace_signature(workspace):
        raise CalibrationError('workspace changed since planning; regenerate pose plan')
    roles = ['train']*train_count + ['verify']*verify_count
    if ([r['role'] for r in plan['views']] != roles or
            plan['train_count'] != train_count or plan['verify_count'] != verify_count):
        raise CalibrationError('plan sample counts/roles do not match task')
    return plan


def save_preview(plan, path):
    """SVG contact sheet: predicted board footprints, explicitly not photographs."""
    width, height = plan['image_size']
    tiles = []
    for i, row in enumerate(plan['views']):
        x, y = 20+(i % 5)*220, 70+(i//5)*190
        points = ' '.join(f'{p[0]*200/width:.1f},{p[1]*150/height:.1f}'
                          for p in row['corners_px'])
        color = '#43c6ac' if row['role'] == 'train' else '#f4b942'
        tiles.append(f'<g transform="translate({x},{y})"><rect width="200" height="150" '
                     f'fill="#182331" stroke="#556"/><polygon points="{points}" '
                     f'fill="{color}" fill-opacity=".25" stroke="{color}" stroke-width="2"/>'
                     f'<text y="169" fill="white" font-size="12">{row["id"]} {row["role"]} '
                     f'{row["distance_m"]*100:.0f}cm / {row["tilt_deg"]:.0f}deg</text></g>')
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="{80+190*((len(tiles)+4)//5)}">'
           '<rect width="100%" height="100%" fill="#101722"/>'
           '<text x="20" y="30" fill="white" font-size="20">Predicted views — not camera images</text>'
           '<text x="20" y="52" fill="#aab" font-size="14">Green: training / Yellow: held-out verification</text>'
           + ''.join(tiles) + '</svg>')
    Path(path).write_text(svg, encoding='utf-8')
