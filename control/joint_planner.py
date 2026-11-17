"""CPU joint-space motion planner (RRT-Connect + shortcutting) on the collision guard.

The fallback/default when robokit (GPU) is unavailable, and the more literal one:
every configuration and edge is checked by the same ``CollisionGuard`` that
guards execution (a hard constraint, not a cost), so a returned path cannot pass
through a thin point cloud or an obstacle between samples. NumPy only.

The path is shortcut, retimed to the toolkit's velocity/acceleration/jerk limits
(``control.trajectory.retime``, a spline through the waypoints) and re-checked
densely; if the spline bows into the clearance band, the waypoints are densified
and retimed again.
"""

from __future__ import annotations

import time

import numpy as np

from .trajectory import TrajectoryLimits, retime
from ..errors import SafetyViolation


class PlanningFailed(RuntimeError):
    pass


class RRTConnectPlanner:
    """Plan joint paths for ``guard.robot`` between joint vectors.

    ``plan(q_start, q_goal, extra_obstacles=(), checks=())`` returns joint samples
    every ``dt`` or raises ``PlanningFailed``. ``extra_obstacles`` are avoided by
    the planner only (e.g. unobserved space); ``checks`` are additional
    ``check(samples) -> reason or None`` run on the final samples.
    """

    def __init__(self, guard, lower, upper, *, clearance_m=0.01, step_rad=0.15,
                 time_limit_s=10.0, max_nodes=20000, goal_bias=0.1, shortcut_iters=150,
                 limits=None, dt=0.01, seed=0):
        self.guard = guard
        self.lower, self.upper = np.asarray(lower, float), np.asarray(upper, float)
        self.clearance_m, self.step_rad = float(clearance_m), float(step_rad)
        self.time_limit_s, self.max_nodes = float(time_limit_s), int(max_nodes)
        self.goal_bias, self.shortcut_iters = float(goal_bias), int(shortcut_iters)
        self.limits = limits or TrajectoryLimits(velocity=1.0, acceleration=3.0, jerk=20.0)
        self.dt = float(dt)
        self.rng = np.random.default_rng(seed)
        self.last = {}
        self._limit = None

    # ----------------------------------------------------------- validity
    def _clearance(self, qs):
        qs = np.atleast_2d(qs)
        distance = self.guard.clearances(qs)
        if self._extra:
            saved = self.guard.dynamic
            self.guard.dynamic = saved + list(self._extra)
            try:
                distance = np.minimum(distance, self.guard.clearances(qs))
            finally:
                self.guard.dynamic = saved
        return distance

    def _valid_edge(self, a, b):
        steps = max(1, int(np.ceil(np.max(np.abs(b - a)) / self.guard.step_rad)))
        qs = a + (b - a) * (np.arange(1, steps + 1) / steps)[:, None]
        inside = np.all((qs >= self.lower) & (qs <= self.upper))
        return bool(inside and np.all(self._clearance(qs) >= self._limit))

    # ----------------------------------------------------------- RRT-Connect
    def _extend(self, tree, parents, target):
        nearest = int(np.argmin(np.max(np.abs(np.asarray(tree) - target), axis=1)))
        q = tree[nearest]
        delta = target - q
        distance = np.max(np.abs(delta))
        new = target if distance <= self.step_rad else q + delta * (self.step_rad / distance)
        if not self._valid_edge(q, new):
            return None
        tree.append(new)
        parents.append(nearest)
        return len(tree) - 1

    def _connect(self, tree, parents, target):
        while True:
            index = self._extend(tree, parents, target)
            if index is None:
                return None
            if np.max(np.abs(tree[index] - target)) < 1e-9:
                return index

    @staticmethod
    def _branch(tree, parents, index):
        path = []
        while index >= 0:
            path.append(tree[index])
            index = parents[index]
        return path

    def _search(self, q_start, q_goal, started):
        if self._valid_edge(q_start, q_goal):
            return [q_start, q_goal]
        trees = ([q_start], [-1]), ([q_goal], [-1])
        while time.perf_counter() - started < self.time_limit_s:
            if len(trees[0][0]) + len(trees[1][0]) > self.max_nodes:
                break
            grow, other = trees
            target = (other[0][0] if self.rng.random() < self.goal_bias
                      else self.rng.uniform(self.lower, self.upper))
            index = self._extend(grow[0], grow[1], target)
            if index is not None:
                joined = self._connect(other[0], other[1], grow[0][index])
                if joined is not None:
                    a = self._branch(*grow, index)[::-1]
                    b = self._branch(*other, joined)
                    path = a + b[1:]
                    return path if np.allclose(path[0], q_start) else path[::-1]
            trees = trees[::-1]
        return None

    def _shortcut(self, path):
        path = [np.asarray(q, float) for q in path]
        for _ in range(self.shortcut_iters):
            if len(path) < 3:
                break
            i, j = sorted(self.rng.choice(len(path), 2, replace=False))
            if j - i < 2:
                continue
            if self._valid_edge(path[i], path[j]):
                path = path[:i + 1] + path[j:]
        return path

    # --------------------------------------------------------------- public
    def plan(self, q_start, q_goal, extra_obstacles=(), checks=()):
        started = time.perf_counter()
        q_start, q_goal = np.asarray(q_start, float), np.asarray(q_goal, float)
        self._extra = list(extra_obstacles)
        self.guard.check_position(q_start)
        band = self.guard.margin_m + self.clearance_m
        start_clear, goal_clear = self._clearance(np.stack([q_start, q_goal]))
        if goal_clear < band:
            raise PlanningFailed(f'goal is {goal_clear*1000:.0f} mm from an obstacle '
                                 f'(needs {band*1000:.0f} mm)')
        if np.any(q_goal < self.lower) or np.any(q_goal > self.upper):
            raise PlanningFailed('goal outside joint limits')
        # A start already inside the band may still leave it (never below margin_m).
        self._limit = min(band, max(start_clear - 1e-4, self.guard.margin_m))
        path = self._search(q_start, q_goal, started)
        if path is None:
            raise PlanningFailed(f'no path within {self.time_limit_s:g} s')
        path = self._shortcut(path)
        self._limit = max(self._limit, self.guard.margin_m)
        for density in (1, 2, 4, 8):          # densify until the spline stays clear
            knots = [path[0]]
            for a, b in zip(path[:-1], path[1:]):
                pieces = max(1, int(np.ceil(np.max(np.abs(b - a)) / .2)) * density)
                knots.extend(a + (b - a) * k / pieces for k in range(1, pieces + 1))
            samples = retime(np.asarray(knots), self.dt, self.limits)
            reason = self._verify(samples)
            for check in checks:
                reason = reason or check(samples)
            if reason is None:
                self.last = {'waypoints': len(path), 'samples': len(samples),
                             'seconds': round(len(samples) * self.dt, 3),
                             'plan_time_s': round(time.perf_counter() - started, 3),
                             'min_clearance_m': float(self._clearance(samples).min())}
                return samples
        raise PlanningFailed(reason)

    def _verify(self, samples):
        if np.any(samples < self.lower - 1e-6) or np.any(samples > self.upper + 1e-6):
            return 'path leaves joint limits'
        for a, b in zip(samples[:-1], samples[1:]):
            try:
                self.guard.check_path(a, b)
            except SafetyViolation as exc:
                return f'guard rejected the path: {exc}'
        if self._extra:
            # Planner-only obstacles (e.g. unobserved space) hold for the final samples too.
            dense = np.concatenate([np.linspace(a, b, max(2, int(np.ceil(
                np.max(np.abs(b - a)) / self.guard.step_rad)) + 1)) for a, b in
                zip(samples[:-1], samples[1:])])
            if np.min(self._clearance(dense)) < self.guard.margin_m:
                return 'path enters space the planner must avoid (e.g. unobserved)'
        return None
