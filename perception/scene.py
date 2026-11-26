"""Task-scoped planar scene estimation from existing camera observations."""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from ..runtime.robot_platform import Observation


@dataclass(frozen=True)
class ObjectSpec:
    """One uniquely color-identifiable rigid object; HSV uses OpenCV units."""

    object_id: str
    hsv_ranges: tuple
    min_area_px: int = 100
    max_area_px: int | None = None

    def __post_init__(self):
        if not self.object_id or not self.hsv_ranges:
            raise ValueError('object_id and HSV ranges are required')
        ranges = []
        for lower, upper in self.hsv_ranges:
            bounds = np.asarray([lower, upper], dtype=float)
            if (bounds.shape != (2, 3) or not np.all(np.isfinite(bounds))
                    or np.any(bounds != np.floor(bounds)) or np.any(bounds < 0)
                    or np.any(bounds > [179, 255, 255]) or np.any(bounds[0] > bounds[1])):
                raise ValueError('HSV bounds must be ordered integers within OpenCV ranges')
            ranges.append(tuple(tuple(int(value) for value in row) for row in bounds))
        if (isinstance(self.min_area_px, bool) or not isinstance(self.min_area_px, int)
                or self.min_area_px < 1):
            raise ValueError('min_area_px must be a positive integer')
        if self.max_area_px is not None and (
                isinstance(self.max_area_px, bool) or not isinstance(self.max_area_px, int)
                or self.max_area_px < self.min_area_px):
            raise ValueError('max_area_px must be an integer >= min_area_px')
        object.__setattr__(self, 'hsv_ranges', tuple(ranges))


@dataclass(frozen=True)
class SceneSpec:
    """Frozen fixed-camera calibration and object definitions for one plane.

    image_to_plane maps undistorted pixel coordinates into metres on the named
    plane. It is not valid for a moving camera or objects above that plane.
    """

    camera: str
    calibration_id: str
    coordinate_frame: str
    image_size: tuple[int, int]
    image_to_plane: tuple
    workspace_xy: tuple[float, float, float, float]
    objects: tuple[ObjectSpec, ...]
    max_frame_age_s: float = 0.3

    def __post_init__(self):
        if not self.camera or not self.calibration_id or not self.coordinate_frame:
            raise ValueError('camera, calibration_id and coordinate_frame are required')
        if (len(self.image_size) != 2 or any(isinstance(value, bool) or
                not isinstance(value, int) or value <= 0 for value in self.image_size)):
            raise ValueError('image_size must be positive integer (width, height)')
        transform = np.asarray(self.image_to_plane, dtype=float)
        if (transform.shape != (3, 3) or not np.all(np.isfinite(transform))
                or np.linalg.matrix_rank(transform) != 3):
            raise ValueError('image_to_plane must be a finite invertible 3x3 matrix')
        workspace = np.asarray(self.workspace_xy, dtype=float)
        if (workspace.shape != (4,) or not np.all(np.isfinite(workspace))
                or np.any(workspace[2:] <= workspace[:2])):
            raise ValueError('workspace_xy must be ordered (xmin, ymin, xmax, ymax) metres')
        objects = tuple(self.objects)
        if not objects or len({item.object_id for item in objects}) != len(objects):
            raise ValueError('scene needs unique object IDs')
        if not math.isfinite(self.max_frame_age_s) or self.max_frame_age_s <= 0:
            raise ValueError('max_frame_age_s must be positive and finite')
        object.__setattr__(self, 'objects', objects)
        object.__setattr__(self, 'image_size', tuple(self.image_size))
        object.__setattr__(self, 'workspace_xy', tuple(float(value) for value in workspace))
        object.__setattr__(self, 'image_to_plane', tuple(tuple(row) for row in transform.tolist()))

    @property
    def spec_id(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, allow_nan=False)
        return hashlib.sha256(payload.encode('utf-8')).hexdigest()


@dataclass
class ObjectState:
    object_id: str
    valid: bool = False
    reason: str = 'not_observed'
    xy_m: np.ndarray | None = None
    centroid_px: np.ndarray | None = None
    bbox_xywh: tuple[int, int, int, int] | None = None
    area_px: int | None = None


@dataclass
class SceneState:
    spec_id: str
    observation_id: str
    camera: str
    calibration_id: str
    coordinate_frame: str
    frame_received_at: float | None
    estimated_at: float
    objects: dict[str, ObjectState] = field(default_factory=dict)
    source: str = 'planar_color_v1'

    @property
    def valid(self) -> bool:
        return bool(self.objects) and all(item.valid for item in self.objects.values())


class SceneEstimator(Protocol):
    spec: SceneSpec

    def estimate(self, observation: 'Observation') -> SceneState: ...


@dataclass(frozen=True)
class PlacementCondition:
    object_id: str
    target_xy_m: tuple[float, float]
    tolerance_m: float = 0.01

    def __post_init__(self):
        target = np.asarray(self.target_xy_m, dtype=float)
        if not self.object_id or target.shape != (2,) or not np.all(np.isfinite(target)):
            raise ValueError('placement needs an object ID and finite XY target')
        if not math.isfinite(self.tolerance_m) or self.tolerance_m <= 0:
            raise ValueError('placement tolerance must be positive and finite')
        object.__setattr__(self, 'target_xy_m', tuple(float(value) for value in target))


@dataclass
class LayoutCheck:
    """Visual layout acceptance only, never permission to enable or move."""

    layout_ready: bool
    reason: str
    observation_id: str
    spec_id: str
    layout_id: str
    consecutive_frames: int
    stable_for_s: float
    errors_m: dict[str, float]


class ResetLayoutVerifier:
    """Accept initial XY placement after distinct fresh frames remain stable.

    This does not verify orientation, grasp release, robot pose, collision
    clearance, force, or physical stopping. It never issues device commands.
    """

    def __init__(self, spec: SceneSpec, placements: tuple[PlacementCondition, ...], *,
                 required_frames: int = 3, min_stable_s: float = 0.2,
                 max_step_m: float = 0.003, max_gap_s: float = 0.5,
                 clock=time.perf_counter):
        placements = tuple(placements)
        names = [item.object_id for item in placements]
        if len(set(names)) != len(names) or set(names) != {item.object_id for item in spec.objects}:
            raise ValueError('placements must cover every scene object exactly once')
        if (isinstance(required_frames, bool) or not isinstance(required_frames, int)
                or required_frames < 2):
            raise ValueError('required_frames must be an integer >= 2')
        if any(not math.isfinite(value) or value <= 0
               for value in (min_stable_s, max_step_m, max_gap_s)):
            raise ValueError('stability limits must be positive and finite')
        workspace = np.asarray(spec.workspace_xy)
        if any(np.any(np.asarray(item.target_xy_m) < workspace[:2]) or
               np.any(np.asarray(item.target_xy_m) > workspace[2:]) for item in placements):
            raise ValueError('placement targets must be inside the workspace')
        self.spec = spec
        self.placements = placements
        self.required_frames = required_frames
        self.min_stable_s = min_stable_s
        self.max_step_m = max_step_m
        self.max_gap_s = max_gap_s
        self.clock = clock
        config = {'spec_id': spec.spec_id, 'placements': [asdict(item) for item in placements],
                  'required_frames': required_frames, 'min_stable_s': min_stable_s,
                  'max_step_m': max_step_m, 'max_gap_s': max_gap_s}
        self.layout_id = hashlib.sha256(json.dumps(
            config, sort_keys=True, allow_nan=False).encode('utf-8')).hexdigest()
        self.clear()

    def clear(self) -> None:
        """Clear acceptance progress, not the scene or any physical device."""
        self._last_stamp = None
        self._last_observation_id = None
        self._started_at = None
        self._anchor_positions = None
        self._count = 0

    def check(self, scene: SceneState) -> LayoutCheck:
        now = self.clock()
        errors = {}

        def result(ready, reason, stable=0.0):
            return LayoutCheck(ready, reason, scene.observation_id, scene.spec_id,
                               self.layout_id, self._count, stable, errors)

        def reject(reason):
            self._count = 0
            self._started_at = None
            self._anchor_positions = None
            return result(False, reason)

        if (scene.spec_id != self.spec.spec_id or scene.camera != self.spec.camera
                or scene.calibration_id != self.spec.calibration_id
                or scene.coordinate_frame != self.spec.coordinate_frame):
            return reject('scene_contract_mismatch')
        stamp = scene.frame_received_at
        if stamp is None or not math.isfinite(now) or not math.isfinite(stamp) or stamp <= 0 or stamp > now:
            return reject('invalid_frame_time')
        if now - stamp > self.spec.max_frame_age_s:
            return reject('stale_image')
        if (scene.observation_id == self._last_observation_id
                or self._last_stamp is not None and stamp <= self._last_stamp):
            return reject('repeated_or_out_of_order_frame')
        gap = self._last_stamp is not None and stamp - self._last_stamp > self.max_gap_s
        self._last_stamp = stamp
        self._last_observation_id = scene.observation_id
        if not scene.valid or set(scene.objects) != {item.object_id for item in self.placements}:
            return reject('invalid_scene')
        positions = {}
        for condition in self.placements:
            position = np.asarray(scene.objects[condition.object_id].xy_m, dtype=float)
            if position.shape != (2,) or not np.all(np.isfinite(position)):
                return reject('invalid_position')
            positions[condition.object_id] = position.copy()
            errors[condition.object_id] = float(np.linalg.norm(position - condition.target_xy_m))
        if any(errors[item.object_id] > item.tolerance_m for item in self.placements):
            return reject('outside_initial_region')
        moving = self._anchor_positions is not None and any(
            np.linalg.norm(position - self._anchor_positions[name]) > self.max_step_m
            for name, position in positions.items())
        if self._started_at is None or gap or moving:
            self._started_at = stamp
            self._anchor_positions = positions
            self._count = 1
        else:
            self._count += 1
        stable = stamp - self._started_at
        ready = self._count >= self.required_frames and stable >= self.min_stable_s
        return result(ready, 'layout_ready' if ready else 'stabilizing', stable)


class PlanarColorEstimator:
    """Stateless detections; never carry a lost object's old pose forward."""

    def __init__(self, spec: SceneSpec, *, clock=time.perf_counter):
        self.spec = spec
        self.clock = clock

    def estimate(self, observation: 'Observation') -> SceneState:
        import cv2

        now = self.clock()
        frame = observation.frames.get(self.spec.camera)
        stamp = None if frame is None else float(frame.t)
        state = SceneState(
            self.spec.spec_id, observation.observation_id, self.spec.camera,
            self.spec.calibration_id, self.spec.coordinate_frame, stamp, now,
            {item.object_id: ObjectState(item.object_id) for item in self.spec.objects})

        def reject(reason):
            for item in state.objects.values():
                item.reason = reason
            return state

        if frame is None or frame.rgb is None:
            return reject('missing_image')
        if not math.isfinite(now) or not math.isfinite(stamp) or stamp <= 0 or stamp > now:
            return reject('invalid_frame_time')
        if now - stamp > self.spec.max_frame_age_s:
            return reject('stale_image')
        width, height = self.spec.image_size
        if frame.rgb.shape != (height, width, 3) or frame.rgb.dtype != np.uint8:
            return reject('image_contract_mismatch')
        if observation.synchronization is not None:
            pair = observation.synchronization.get('cameras', {}).get(self.spec.camera, {})
            if not pair.get('matched', False):
                return reject('unsynchronized')
        if frame.distortion is not None and np.any(np.asarray(frame.distortion) != 0):
            return reject('undistortion_required')
        hsv = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2HSV)
        transform = np.asarray(self.spec.image_to_plane)
        workspace = np.asarray(self.spec.workspace_xy)
        masks = {}
        for definition in self.spec.objects:
            item = state.objects[definition.object_id]
            mask = np.zeros((height, width), dtype=np.uint8)
            for lower, upper in definition.hsv_ranges:
                mask |= cv2.inRange(hsv, np.array(lower, dtype=np.uint8),
                                   np.array(upper, dtype=np.uint8))
            count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
            candidates = [index for index in range(1, count)
                          if stats[index, cv2.CC_STAT_AREA] >= definition.min_area_px]
            if len(candidates) != 1:
                item.reason = 'missing' if not candidates else 'ambiguous'
                continue
            index = candidates[0]
            left, top, box_width, box_height, area = (int(value) for value in stats[index])
            item.area_px = area
            item.centroid_px = centroids[index].copy()
            item.bbox_xywh = (left, top, box_width, box_height)
            if definition.max_area_px is not None and area > definition.max_area_px:
                item.reason = 'size_mismatch'
                continue
            if left == 0 or top == 0 or left + box_width >= width or top + box_height >= height:
                item.reason = 'clipped'
                continue
            pixels = np.array([centroids[index], [left, top], [left + box_width, top],
                               [left, top + box_height], [left + box_width, top + box_height]])
            projected = np.column_stack((pixels, np.ones(len(pixels)))) @ transform.T
            denominator = projected[:, 2]
            if (np.any(np.abs(denominator) < 1e-12)
                    or np.any(np.sign(denominator) != np.sign(denominator[0]))):
                item.reason = 'invalid_projection'
                continue
            positions = projected[:, :2] / denominator[:, None]
            if (not np.all(np.isfinite(positions)) or np.any(positions < workspace[:2])
                    or np.any(positions > workspace[2:])):
                item.reason = 'outside_workspace'
                continue
            item.xy_m = positions[0].copy()
            item.valid, item.reason = True, 'detected'
            masks[item.object_id] = labels == index
        names = list(masks)
        for offset, name in enumerate(names):
            for other in names[offset + 1:]:
                if np.any(masks[name] & masks[other]):
                    for conflicted in (name, other):
                        state.objects[conflicted].valid = False
                        state.objects[conflicted].reason = 'ambiguous_identity'
                        state.objects[conflicted].xy_m = None
        return state