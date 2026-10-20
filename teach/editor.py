"""Small, composable trajectory edits (mark, trim, retime, inspect)."""

from __future__ import annotations

import numpy as np

from .trajectory import (Marker, Trajectory, find_dwell_candidates, save_trajectory,
                         trim_trajectory)


def add_marker(trajectory: Trajectory, action: str = 'mark', at: float | None = None,
               index: int | None = None, width: float | None = None,
               label: str = '') -> Trajectory:
    if index is None:
        time = float(trajectory.t[0] if at is None else at)
        index = int(np.clip(np.searchsorted(trajectory.t, time), 0, len(trajectory) - 1))
        time = float(trajectory.t[index])
    else:
        index = int(np.clip(index, 0, len(trajectory) - 1))
        time = float(trajectory.t[index])
    trajectory.markers.append(Marker(t=time, index=index, action=action,
                                     width=width, label=label))
    trajectory.markers.sort(key=lambda marker: marker.t)
    return trajectory


def remove_marker(trajectory: Trajectory, *, index: int | None = None,
                  action: str | None = None, at: float | None = None) -> int:
    keep, removed = [], 0
    for marker in trajectory.markers:
        hit = ((index is not None and marker.index == index) or
               (action is not None and marker.action == action) or
               (at is not None and abs(marker.t - float(at)) <= 0.05))
        if hit:
            removed += 1
        else:
            keep.append(marker)
    trajectory.markers = keep
    return removed


def mark_dwells(trajectory: Trajectory, action: str = 'mark',
                ignore_before: float = 1.0, **kwargs) -> Trajectory:
    for run in find_dwell_candidates(trajectory, ignore_before=ignore_before, **kwargs):
        add_marker(trajectory, action=action, at=run['start'], label='dwell')
    return trajectory


def trim(trajectory: Trajectory, **kwargs):
    return trim_trajectory(trajectory, **kwargs)


def retime(trajectory: Trajectory, factor: float) -> Trajectory:
    """Scale time (``factor < 1`` makes the demonstration faster)."""
    factor = float(factor)
    if factor <= 0:
        raise ValueError('factor must be positive')
    meta = dict(trajectory.meta)
    meta['retime'] = factor
    return Trajectory(t=trajectory.t * factor, q=trajectory.q, qd=trajectory.qd / factor,
                      tcp=trajectory.tcp, grip=trajectory.grip,
                      markers=[Marker(t=marker.t * factor, index=marker.index,
                                      action=marker.action, width=marker.width,
                                      label=marker.label)
                               for marker in trajectory.markers],
                      meta=meta)


def summarize(trajectory: Trajectory) -> dict:
    return {
        'samples': len(trajectory),
        'dof': trajectory.dof,
        'duration_s': round(trajectory.duration, 4),
        'rate_hz': round(trajectory.rate, 2),
        'markers': [f'{marker.t:.2f}s:{marker.action}' for marker in trajectory.markers],
        'tcp_start': None if trajectory.tcp is None else trajectory.tcp[0].round(4).tolist(),
        'tcp_end': None if trajectory.tcp is None else trajectory.tcp[-1].round(4).tolist(),
        'gripper_start': None if not len(trajectory.grip) else float(np.asarray(trajectory.grip).reshape(-1)[0]),
        'gripper_end': None if not len(trajectory.grip) else float(np.asarray(trajectory.grip).reshape(-1)[-1]),
        'meta': dict(trajectory.meta),
    }


def save(directory, trajectory: Trajectory, name: str = 'trajectory'):
    return save_trajectory(directory, trajectory, name=name)
