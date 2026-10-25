"""TCP box and sampled joint-path checks, independent of robot hardware."""
import numpy as np

from .safety import WorkspaceBox
from ..errors import SafetyViolation


class WorkspaceGuard:
    def __init__(self, lower, upper, fk, margin_m=0.0, step_rad=0.01,
                 lower_enabled=(True, True, True), upper_enabled=(True, True, True)):
        lower, upper = np.asarray(lower, float), np.asarray(upper, float)
        if (lower.shape != (3,) or upper.shape != (3,) or
                not np.isfinite([lower, upper]).all() or
                not np.isfinite(margin_m) or margin_m < 0 or
                not np.isfinite(step_rad) or step_rad <= 0):
            raise ValueError('invalid workspace bounds, margin or sampling step')
        self.box = WorkspaceBox(lower + margin_m, upper - margin_m)
        if np.any(self.box.lower >= self.box.upper):
            raise ValueError('workspace margin leaves no usable volume')
        self.fk = fk
        self.step_rad = step_rad
        self.lower_enabled = np.asarray(lower_enabled, bool).reshape(3)
        self.upper_enabled = np.asarray(upper_enabled, bool).reshape(3)

    def check_position(self, tcp):
        if tcp is None:
            raise SafetyViolation('TCP feedback unavailable')
        value = np.asarray(tcp, float)
        if value.shape not in ((3,), (6,)) or not np.isfinite(value).all():
            raise SafetyViolation('invalid TCP position')
        if (np.any((value[:3] < self.box.lower) & self.lower_enabled) or
                np.any((value[:3] > self.box.upper) & self.upper_enabled)):
            raise SafetyViolation(f'TCP {value[:3].round(4).tolist()} outside workspace '
                                  f'{self.box.lower.round(4).tolist()} .. '
                                  f'{self.box.upper.round(4).tolist()}')

    def check_path(self, start, target, clearance_m=0.0):
        if not np.isfinite(clearance_m) or clearance_m < 0:
            raise ValueError('invalid planning clearance')
        start, target = np.asarray(start, float), np.asarray(target, float)
        if (start.ndim != 1 or target.shape != start.shape or not start.size or
                not np.isfinite([start, target]).all()):
            raise SafetyViolation('invalid joint path')
        steps = max(1, int(np.ceil(np.max(np.abs(target-start))/self.step_rad)))
        if steps > 10000:
            raise SafetyViolation('joint path exceeds sampling budget')
        for i in range(steps+1):
            tcp = np.asarray(self.fk(start+(target-start)*i/steps), float)
            self.check_position(tcp)
            if (np.any((tcp[:3] < self.box.lower+clearance_m) & self.lower_enabled) or
                    np.any((tcp[:3] > self.box.upper-clearance_m) & self.upper_enabled)):
                raise SafetyViolation('path lacks reserved TCP tracking clearance')

    def check_feedback(self, state, model_tolerance_m=0.01):
        if state.q is None or not state.connected or state.protective_stop:
            raise SafetyViolation('robot feedback unavailable or protective stop active')
        self.check_position(state.tcp)
        predicted = np.asarray(self.fk(state.q), float)
        self.check_position(predicted)
        if np.linalg.norm(predicted[:3]-state.tcp[:3]) > model_tolerance_m:
            raise SafetyViolation('FK / measured TCP mismatch exceeds 10 mm')
