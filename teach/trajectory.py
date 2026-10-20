"""Robot-agnostic teach trajectory: storage, filtering, dwell detection, schedule.

A trajectory is joint-space data plus TCP and gripper channels, recorded at a
fixed rate.  Markers are semantic events (gripper close/open, pause, phase)
that survive replay and dataset export.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

ACTIONS = ('mark', 'close', 'open', 'move', 'hold', 'phase')

DWELL_SPEED_THRESHOLD = 0.04      # rad/s
DWELL_MIN_DURATION = 0.5          # s
DWELL_BRIDGE = 0.3               # s
MOTION_MIN_DURATION = 0.3        # s


@dataclass
class Marker:
    t: float = 0.0
    index: int = 0
    action: str = 'mark'
    width: float | None = None
    label: str = ''

    def __post_init__(self):
        if self.action not in ACTIONS:
            raise ValueError(f'unknown marker action {self.action!r}; expected one of {ACTIONS}')


@dataclass
class Trajectory:
    t: np.ndarray
    q: np.ndarray
    qd: np.ndarray | None = None
    tcp: np.ndarray | None = None
    grip: np.ndarray | None = None
    markers: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.t = np.asarray(self.t, dtype=float).reshape(-1)
        self.q = np.asarray(self.q, dtype=float)
        if self.q.ndim == 1:
            self.q = self.q.reshape(-1, 1)
        if self.t.shape[0] != self.q.shape[0]:
            raise ValueError('t and q must have the same number of samples')
        for name in ('qd', 'tcp', 'grip'):
            value = getattr(self, name)
            if value is None:
                columns = self.q.shape[1] if name == 'qd' else (6 if name == 'tcp' else None)
                value = (np.zeros((len(self.t), columns)) if columns is not None
                         else np.zeros(len(self.t)))
            value = np.asarray(value, dtype=float)
            if value.ndim == 1:
                if name == 'tcp':
                    value = value.reshape(-1, 6)
                elif name == 'grip':
                    value = value.reshape(-1)
                else:
                    value = value.reshape(-1, 1)
            if len(value) != len(self.t):
                raise ValueError(f'{name} has {len(value)} samples, expected {len(self.t)}')
            setattr(self, name, value)

    @property
    def dof(self) -> int:
        return int(self.q.shape[1])

    @property
    def duration(self) -> float:
        return float(self.t[-1] - self.t[0]) if len(self.t) else 0.0

    @property
    def rate(self) -> float:
        return float(1.0 / np.median(np.diff(self.t))) if len(self.t) > 1 else 0.0

    def __len__(self) -> int:
        return int(len(self.t))

    def subset(self, mask) -> 'Trajectory':
        mask = np.asarray(mask, dtype=bool)
        markers = [marker for marker in self.markers
                   if 0 <= marker.index < len(mask) and mask[marker.index]]
        return Trajectory(self.t[mask] - self.t[mask][0] if np.any(mask) else self.t[mask],
                          self.q[mask], self.qd[mask], self.tcp[mask], self.grip[mask],
                          markers=markers, meta=dict(self.meta))


def save_trajectory(directory, trajectory: Trajectory, name: str = 'trajectory') -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(directory / f'{name}.npz', t=trajectory.t, q=trajectory.q,
                        qd=trajectory.qd, tcp=trajectory.tcp, grip=trajectory.grip)
    meta = dict(trajectory.meta)
    meta['markers'] = [asdict(marker) for marker in trajectory.markers]
    meta.setdefault('samples', len(trajectory))
    meta.setdefault('duration', trajectory.duration)
    (directory / f'{name}.json').write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    return directory


def load_trajectory(directory, name: str = 'trajectory') -> Trajectory:
    directory = Path(directory)
    with np.load(directory / f'{name}.npz', allow_pickle=False) as raw:
        arrays = {key: raw[key] for key in ('t', 'q', 'qd', 'tcp', 'grip')}
    meta = json.loads((directory / f'{name}.json').read_text(encoding='utf-8'))
    markers = [Marker(**entry) for entry in meta.pop('markers', [])]
    return Trajectory(markers=markers, meta=meta, **arrays)


def smooth_joints(q, window: int = 5) -> np.ndarray:
    window = int(window)
    q = np.asarray(q, dtype=float)
    if window < 3 or len(q) < window:
        return q
    window = window if window % 2 else window + 1
    pad = window // 2
    padded = np.pad(q, ((pad, pad), (0, 0)), mode='edge')
    kernel = np.ones(window) / window
    return np.stack([np.convolve(padded[:, column], kernel, mode='valid')
                     for column in range(q.shape[1])], axis=1)


def resample(t, q, rate: float):
    t = np.asarray(t, dtype=float)
    q = np.asarray(q, dtype=float)
    if len(t) > 2:
        keep = np.concatenate(([True], np.diff(t) > 1e-9))
        t = t[keep]
        q = q[keep]
    step = 1.0 / float(rate)
    t_new = np.arange(0.0, t[-1] + step * 0.5, step)
    t_new[-1] = min(t_new[-1], t[-1])
    q_new = np.stack([np.interp(t_new, t, q[:, column])
                      for column in range(q.shape[1])], axis=1)
    return t_new, q_new


def joint_motion(trajectory: 'Trajectory') -> float:
    """Largest peak-to-peak joint change in a trajectory (rad)."""
    return (float(np.max(np.ptp(trajectory.q, axis=0)))
            if trajectory.q.size else 0.0)


def max_joint_speed(t, q, min_window_s: float = 0.02) -> float:
    """Peak joint speed over a short window (robust to recorder timestamp jitter)."""
    t = np.asarray(t, dtype=float)
    q = np.asarray(q, dtype=float)
    if len(t) < 3:
        return 0.0
    step = float(np.median(np.diff(t)))
    half = max(1, int(round(float(min_window_s) / max(step, 1e-6))))
    lo = np.clip(np.arange(len(t)) - half, 0, len(t) - 1)
    hi = np.clip(np.arange(len(t)) + half, 0, len(t) - 1)
    span = t[hi] - t[lo]
    valid = span > 1e-6
    if not np.any(valid):
        return 0.0
    speeds = np.max(np.abs(q[hi[valid]] - q[lo[valid]]), axis=1) / span[valid]
    return float(speeds.max())


def motion_speed(t, q, baseline_s: float = 0.05, smooth: int = 9) -> np.ndarray:
    """Per-sample max joint speed, central difference and optional smoothing."""
    t = np.asarray(t, dtype=float)
    q = np.asarray(q, dtype=float)
    count = len(t)
    if count < 3:
        return np.zeros(count)
    step = float(np.median(np.diff(t)))
    half = max(2, int(round(float(baseline_s) / max(step, 1e-6))))
    lo = np.clip(np.arange(count) - half, 0, count - 1)
    hi = np.clip(np.arange(count) + half, 0, count - 1)
    span = np.maximum(t[hi] - t[lo], 1e-6)
    speed = np.max(np.abs(q[hi] - q[lo]), axis=1) / span
    smooth = max(1, int(smooth))
    if smooth > 1 and count >= smooth:
        pad = smooth // 2
        padded = np.pad(speed, (pad, pad), mode='edge')
        speed = np.convolve(padded, np.ones(smooth) / smooth, mode='valid')
    return speed


def _runs_from_flags(flags, t, min_duration, bridge):
    spans = []
    begin = None
    for position, value in enumerate(flags):
        if value and begin is None:
            begin = position
        elif not value and begin is not None:
            spans.append([begin, position - 1])
            begin = None
    if begin is not None:
        spans.append([begin, len(flags) - 1])
    merged = []
    for span in spans:
        if merged and t[span[0]] - t[merged[-1][1]] <= float(bridge):
            merged[-1][1] = span[1]
        else:
            merged.append(list(span))
    return [span for span in merged
            if float(t[span[1]] - t[span[0]]) >= float(min_duration)]


def find_still_runs(t, q, tcp=None, min_duration=DWELL_MIN_DURATION,
                    speed_threshold=DWELL_SPEED_THRESHOLD, bridge=DWELL_BRIDGE):
    t = np.asarray(t, dtype=float)
    q = np.asarray(q, dtype=float)
    if len(t) < 2:
        return []
    still = motion_speed(t, q) < float(speed_threshold)
    runs = []
    for first, last in _runs_from_flags(still, t, min_duration, bridge):
        entry = dict(start=float(t[first]), end=float(t[last]), t=float(t[first]),
                     duration=float(t[last] - t[first]), index=int(first))
        if tcp is not None and len(tcp) == len(t):
            entry['tcp'] = [round(float(value), 4)
                            for value in np.asarray(tcp, dtype=float)[first]]
        runs.append(entry)
    return runs


def find_dwell_candidates(trajectory: Trajectory, ignore_before: float = 1.0, **kwargs):
    runs = find_still_runs(trajectory.t, trajectory.q, tcp=trajectory.tcp, **kwargs)
    return [run for run in runs if run['index'] > 0 and run['t'] >= float(ignore_before)]


def clean_timing(trajectory: Trajectory, min_gap_s: float = 0.004) -> Trajectory:
    t = np.asarray(trajectory.t, dtype=float)
    if len(t) < 3:
        return trajectory
    keep = np.ones(len(t), dtype=bool)
    last = t[0]
    for index in range(1, len(t)):
        if t[index] - last < float(min_gap_s):
            keep[index] = False
        else:
            last = t[index]
    if keep.all():
        return trajectory
    offsets = np.nonzero(keep)[0]
    new_t = t[offsets]
    markers = [Marker(t=float(marker.t),
                      index=int(np.clip(np.searchsorted(new_t, marker.t), 0, len(new_t) - 1)),
                      action=marker.action, width=marker.width, label=marker.label)
               for marker in trajectory.markers]
    meta = dict(trajectory.meta)
    meta['samples'] = int(len(new_t))
    meta['duration'] = float(new_t[-1] - new_t[0])
    return Trajectory(t=new_t, q=trajectory.q[offsets], qd=trajectory.qd[offsets],
                      tcp=trajectory.tcp[offsets], grip=trajectory.grip[offsets],
                      markers=markers, meta=meta)


def trim_trajectory(trajectory: Trajectory, start=None, end=None, head_margin: float = 0.4,
                    tail_margin: float = 0.7, speed_threshold: float = DWELL_SPEED_THRESHOLD):
    t = np.asarray(trajectory.t, dtype=float)
    info = dict(start=float(t[0]) if len(t) else 0.0, end=float(t[-1]) if len(t) else 0.0,
                removed_head=0.0, removed_tail=0.0, dropped_markers=[])
    if len(t) < 3:
        return trajectory, info
    if start is None or end is None:
        motion = _runs_from_flags(motion_speed(t, trajectory.q) >= float(speed_threshold),
                                  t, MOTION_MIN_DURATION, 0.0)
        auto_start = float(t[motion[0][0]]) - float(head_margin) if motion else float(t[0])
        auto_end = float(t[motion[-1][1]]) + float(tail_margin) if motion else float(t[-1])
        action_times = [marker.t for marker in trajectory.markers
                        if marker.action in ('close', 'open', 'move')]
        if action_times:
            auto_start = min(auto_start, min(action_times) - float(head_margin))
            auto_end = max(auto_end, max(action_times) + float(tail_margin))
        start = auto_start if start is None else float(start)
        end = auto_end if end is None else float(end)
    start = float(np.clip(start, t[0], t[-1]))
    end = float(np.clip(end, t[0], t[-1]))
    kept = np.nonzero((t >= start) & (t <= end))[0]
    if len(kept) < 2 or (end - start) >= (float(t[-1]) - float(t[0])) * 0.999:
        return trajectory, info
    new_t = t[kept]
    shift = float(new_t[0])
    kept_markers, dropped = [], []
    for marker in trajectory.markers:
        (kept_markers if start <= marker.t <= end else dropped).append(marker)
    markers = [Marker(t=float(marker.t) - shift,
                      index=int(np.clip(np.searchsorted(new_t, marker.t), 0, len(new_t) - 1)),
                      action=marker.action, width=marker.width, label=marker.label)
               for marker in kept_markers]
    meta = dict(trajectory.meta)
    meta['samples'] = int(len(new_t))
    meta['duration'] = float(new_t[-1] - new_t[0])
    meta['trim'] = dict(start=shift, end=float(new_t[-1]),
                        removed_head=float(shift - t[0]),
                        removed_tail=float(t[-1] - new_t[-1]))
    info.update(meta['trim'])
    info['dropped_markers'] = [f'{marker.t:.2f}s:{marker.action}' for marker in dropped]
    trimmed = Trajectory(t=new_t - shift, q=trajectory.q[kept], qd=trajectory.qd[kept],
                         tcp=trajectory.tcp[kept], grip=trajectory.grip[kept],
                         markers=markers, meta=meta)
    return trimmed, info


class GripperSchedule:
    """Gripper width over playback time, derived from replay events."""

    def __init__(self, events, initial_width: float):
        self.initial_width = float(initial_width)
        ordered = sorted(events, key=lambda event: event['t'])
        self.times = np.asarray([event['t'] for event in ordered], dtype=float)
        self.widths = np.asarray([event['width'] for event in ordered], dtype=float)

    def width_at(self, t) -> float:
        if not len(self.times):
            return self.initial_width
        index = int(np.searchsorted(self.times, float(t), side='right')) - 1
        return self.initial_width if index < 0 else float(self.widths[index])


def build_events(markers, initial_width: float, max_width: float,
                 start_offset: float, speed_scale: float):
    events = []
    for marker in markers:
        time = start_offset + marker.t / max(float(speed_scale), 1e-6)
        if marker.action == 'close':
            width = 0.0
        elif marker.action == 'open':
            width = max_width
        elif marker.action == 'move':
            width = float(max_width if marker.width is None
                          else np.clip(marker.width, 0.0, max_width))
        else:
            width = None
        events.append(dict(t=time, action=marker.action, width=width,
                           label=marker.label, marker=marker))
    return events
