"""Fixed-rate control loop helper with overrun statistics."""

from __future__ import annotations

import time
from collections import deque


class Rate:
    """Sleep so that successive loop iterations start at a fixed period.

    ``overruns`` records iterations that took longer than the period, which is
    the signal a real-time servo loop needs in order to warn the operator.
    """

    def __init__(self, hz: float, clock=time.perf_counter):
        if hz <= 0:
            raise ValueError('rate must be positive')
        self.hz = float(hz)
        self.period = 1.0 / self.hz
        self.clock = clock
        self._started = self.clock()
        self._next = self._started
        self.iterations = 0
        self.overruns = 0
        self.max_overrun = 0.0
        self.recent = deque(maxlen=256)

    def reset(self) -> None:
        now = self.clock()
        self._started = now
        self._next = now
        self.iterations = 0
        self.overruns = 0
        self.max_overrun = 0.0
        self.recent.clear()

    def wait(self) -> float:
        """Block until the next tick; return the measured iteration period."""
        self._next += self.period
        now = self.clock()
        delay = self._next - now
        if delay > 0:
            time.sleep(delay)
        else:
            overrun = -delay
            self.overruns += 1
            self.max_overrun = max(self.max_overrun, overrun)
            # Do not accumulate lag: resynchronise instead of sprinting.
            self._next = now
        self.iterations += 1
        self.recent.append(now)
        return self.period

    def elapsed(self) -> float:
        return self.clock() - self._started

    def p95_overrun_ms(self) -> float:
        if len(self.recent) < 2:
            return 0.0
        gaps = [b - a for a, b in zip(self.recent, list(self.recent)[1:])]
        excess = sorted(max(0.0, gap - self.period) for gap in gaps)
        if not excess:
            return 0.0
        index = min(len(excess) - 1, int(round(0.95 * (len(excess) - 1))))
        return excess[index] * 1000.0
