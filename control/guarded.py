"""Collision-aware braking for joint-space servo references.

``CollisionBrake`` turns a ``core.collision.CollisionGuard`` into the three
checks a servo loop needs every control cycle:

* ``brake_goal`` (position goals, e.g. PBVS): if the jerk-limited stopping point
  of the current reference motion would get inside ``margin_m + clearance_m``,
  return the farthest clear point of that stopping path as the new goal;
* ``speed_scale`` (velocity commands, e.g. IBVS): scale a commanded velocity so
  its stop keeps the clearance band (moving away is never slowed);
* the hard floor: a reference inside ``margin_m`` raises ``SafetyViolation``.

NumPy only; the guard's obstacles may change between calls (depth maps).
"""

from __future__ import annotations

import numpy as np


class CollisionBrake:
    def __init__(self, guard, accel, jerk, dt, clearance_m=0.015):
        if not (accel > 0 and jerk > 0 and dt > 0 and clearance_m >= 0):
            raise ValueError('invalid braking limits')
        self.guard = guard
        self.accel, self.jerk, self.dt = float(accel), float(jerk), float(dt)
        self.clearance_m = float(clearance_m)
        self.brakes = 0
        self._brake_goal = None

    def stopping_point(self, q, v, a=None):
        """Joint-wise jerk-limited stopping point (conservative ramps, 3 cycles latency)."""
        v = np.asarray(v, float)
        a = np.zeros_like(v) if a is None else np.asarray(a, float)
        direction = np.sign(v)
        push = np.maximum(direction * a, 0.)              # still accelerating along v
        speed = np.abs(v) + push ** 2 / (2 * self.jerk)  # speed gained while that ramps down
        distance = (speed ** 2 / (2 * self.accel) + speed * (self.accel + push) / self.jerk
                    + speed * 3 * self.dt)
        return np.asarray(q, float) + direction * distance

    def brake_goal(self, q, v, a, goal):
        """Hard-check ``q``; return a replacement goal if the current motion must stop."""
        if not np.any(v) or (self._brake_goal is not None
                             and np.array_equal(goal, self._brake_goal)):
            self.guard.check_position(q)
            return None
        stop = self.stopping_point(q, v, a)
        here, ahead = self.guard.clearances(np.stack([q, stop]))    # one batched FK
        if here < self.guard.margin_m:
            self.guard.check_position(q)   # raises with the details
        if ahead >= min(self.guard.margin_m + self.clearance_m, here):
            return None
        # Brake once per approach (re-braking every cycle would creep on).
        self.brakes += 1
        self._brake_goal = q + (stop - q) * self.guard.safe_fraction(q, stop, self.clearance_m)
        return self._brake_goal.copy()

    def speed_scale(self, q, v, a=None):
        """Scale for velocity ``v`` so its stop keeps the clearance band.

        Linearises the clearance lost per rad along ``v`` (worst of two probe
        lengths); moving away from every obstacle is never slowed.
        """
        speed = float(np.max(np.abs(v)))
        if speed == 0.:
            return 1.
        direction = np.asarray(v, float) / speed
        here, near, far = self.guard.clearances(
            np.stack([q, q + direction * .01, q + direction * .05]))
        rate = max((here - near) / .01, (here - far) / .05)
        if rate <= 0.:
            return 1.
        room = (here - self.guard.margin_m - self.clearance_m) / rate
        if room <= 0.:
            return 0.
        # Largest s with s^2/(2a) + s*(a/j + 3dt) <= room (conservative jerk ramps), less
        # the speed still gained while a current acceleration along v ramps down.
        b = self.accel / self.jerk + 3 * self.dt
        allowed = self.accel * (-b + np.sqrt(b * b + 2 * room / self.accel))
        if a is not None:
            push = max(float(np.max(np.asarray(a) * np.sign(direction))), 0.)
            allowed -= push ** 2 / (2 * self.jerk)
        scale = min(1., max(allowed, 0.) / speed)
        self.brakes += scale < 1.
        return scale
