"""Action-chunk sources for continuous policy / scripted control.

A chunk is an array of absolute ``[joints..., gripper_width]`` rows sampled at
``action_hz``.  Keeping this small and transport-agnostic lets HTTP policy
clients, recorded demonstrations and tests feed the same executor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


@dataclass
class ActionChunk:
    """One absolute action chunk; rows are ``[q..., gripper_width]``."""

    actions: np.ndarray
    action_hz: float = 10.0
    done: bool = False
    action_index: int = 0
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        actions = np.asarray(self.actions, dtype=float)
        if actions.size == 0:
            actions = np.empty((0, 0), dtype=float)
        elif actions.ndim == 1:
            actions = actions.reshape(1, -1)
        if actions.ndim != 2:
            raise ValueError(f'actions must be a 2-D array, got {actions.shape}')
        if not np.all(np.isfinite(actions)):
            raise ValueError('actions contain NaN/Inf')
        self.actions = actions
        self.action_hz = float(self.action_hz)
        if self.action_hz <= 0:
            raise ValueError('action_hz must be positive')
        self.action_index = int(self.action_index)
        self.done = bool(self.done)

    @property
    def count(self) -> int:
        return int(len(self.actions))

    def slice(self, skip: int) -> 'ActionChunk':
        skip = max(0, int(skip))
        return ActionChunk(actions=self.actions[skip:], action_hz=self.action_hz,
                           done=self.done, action_index=self.action_index + skip,
                           meta=dict(self.meta))


class ChunkSource(Protocol):
    """Minimal interface a policy client or a local source must provide."""

    def next_chunk(self, action_index: int, timeout: float | None = None) -> ActionChunk:
        """Return the chunk starting at ``action_index``."""


class CallableSource:
    """Adapt any ``fn(action_index) -> ActionChunk`` callable to the protocol."""

    def __init__(self, function, abort=None):
        self.function = function
        self.abort = abort

    def next_chunk(self, action_index: int, timeout: float | None = None) -> ActionChunk:
        return self.function(int(action_index))


class SequenceSource:
    """Serve a fixed action array in chunks (demos, tests, offline validation)."""

    def __init__(self, actions, action_hz: float = 10.0, chunk_size: int = 8,
                 action_index: int = 0):
        actions = np.asarray(actions, dtype=float)
        if actions.ndim != 2 or actions.shape[0] == 0:
            raise ValueError('actions must be a non-empty 2-D array')
        self.actions = actions
        self.action_hz = float(action_hz)
        self.chunk_size = max(1, int(chunk_size))
        self.action_index = max(0, int(action_index))
        self._served = set()

    def next_chunk(self, action_index: int, timeout: float | None = None) -> ActionChunk:
        start = max(0, int(action_index))
        end = min(start + self.chunk_size, len(self.actions))
        self._served.add(start)
        if start >= len(self.actions):
            return ActionChunk(np.empty((0, self.actions.shape[1])), action_hz=self.action_hz,
                               done=True, action_index=start,
                               meta={'source': 'sequence', 'served': len(self._served)})
        return ActionChunk(self.actions[start:end], action_hz=self.action_hz,
                           done=end >= len(self.actions), action_index=start,
                           meta={'source': 'sequence', 'served': len(self._served)})
