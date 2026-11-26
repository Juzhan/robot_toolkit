"""Recorded, robot-agnostic safety envelope for human-guided setup.

    The envelope is deliberately based on observed joint and TCP ranges. Joint
    ranges are the hard pre-motion guard; TCP ranges are retained for inspection
    and for backends that can provide forward-kinematics checks. No robot SDK is
    imported here.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class SafetyEnvelope:
    q_lower: np.ndarray
    q_upper: np.ndarray
    tcp_lower: np.ndarray
    tcp_upper: np.ndarray
    samples: int = 0
    source: str = 'freedrive'

    def __post_init__(self):
        self.q_lower = np.asarray(self.q_lower, dtype=float).reshape(-1)
        self.q_upper = np.asarray(self.q_upper, dtype=float).reshape(-1)
        self.tcp_lower = np.asarray(self.tcp_lower, dtype=float).reshape(6)
        self.tcp_upper = np.asarray(self.tcp_upper, dtype=float).reshape(6)
        if self.q_lower.shape != self.q_upper.shape:
            raise ValueError('q_lower/q_upper must have the same shape')
        if not self.q_lower.size or not all(np.all(np.isfinite(v)) for v in
                (self.q_lower, self.q_upper, self.tcp_lower, self.tcp_upper)):
            raise ValueError('safety envelope requires finite, nonempty bounds')
        if np.any(self.q_lower > self.q_upper) or np.any(self.tcp_lower > self.tcp_upper):
            raise ValueError('safety envelope lower bounds must not exceed upper bounds')

    @classmethod
    def from_samples(cls, samples, margin_rad: float = 0.0,
                     tcp_margin_m: float = 0.0) -> 'SafetyEnvelope':
        items = list(samples)
        if not items:
            raise ValueError('at least one safety sample is required')
        q = np.asarray([item[0] for item in items], dtype=float)
        tcp = np.asarray([item[1] for item in items], dtype=float)
        margin_rad, tcp_margin_m = float(margin_rad), float(tcp_margin_m)
        if not all(np.isfinite(v) and v >= 0 for v in (margin_rad, tcp_margin_m)):
            raise ValueError('safety margins must be finite and nonnegative')
        # Rotation components are radians; translation components are metres.
        tcp_margin = np.array([tcp_margin_m] * 3 + [0.0] * 3)
        return cls(q.min(axis=0) - margin_rad, q.max(axis=0) + margin_rad,
                   tcp.min(axis=0) - tcp_margin, tcp.max(axis=0) + tcp_margin,
                   samples=len(items))

    def contains_q(self, q, tolerance: float = 1e-9) -> bool:
        value = np.asarray(q, dtype=float).reshape(self.q_lower.shape)
        return bool(np.all(value >= self.q_lower - tolerance) and
                    np.all(value <= self.q_upper + tolerance))

    def to_dict(self) -> dict:
        return {'version': 1, 'source': self.source, 'samples': int(self.samples),
                'q_lower_rad': self.q_lower.tolist(), 'q_upper_rad': self.q_upper.tolist(),
                'tcp_lower': self.tcp_lower.tolist(), 'tcp_upper': self.tcp_upper.tolist(),
                'tcp_convention': '[x,y,z,rx,ry,rz], metres/radians'}

    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding='utf-8')
        return path

    @classmethod
    def load(cls, path) -> 'SafetyEnvelope':
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        return cls(data['q_lower_rad'], data['q_upper_rad'], data['tcp_lower'],
                   data['tcp_upper'], samples=int(data.get('samples', 0)),
                   source=str(data.get('source', 'unknown')))
