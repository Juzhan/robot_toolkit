"""RGB-D inspection and ENPIRE clients. No camera acquisition or robot commands."""

from __future__ import annotations

import base64
import io
import math
from dataclasses import dataclass
from time import perf_counter
from urllib.parse import quote

import numpy as np

from .external_tools import (
    ExternalToolError, Sam3Client, Segmentation, SegmentationResult, _HttpToolClient,
)


def _transform(value) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-6)
            or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-3)
            or not np.isclose(np.linalg.det(matrix[:3, :3]), 1, atol=1e-3)):
        raise ValueError('expected a finite rigid 4x4 transform')
    return matrix.copy()


def _encode_array(value: np.ndarray) -> str:
    buffer = io.BytesIO()
    np.save(buffer, value, allow_pickle=False)
    return base64.b64encode(buffer.getvalue()).decode('ascii')


def _decode_array(value: str, shape: tuple[int, ...]) -> np.ndarray:
    maximum = math.prod(shape) * 8 + 4096
    if not isinstance(value, str) or len(value) > maximum * 2:
        raise ValueError('invalid array payload size')
    stream = io.BytesIO(base64.b64decode(value, validate=True))
    if np.lib.format.read_magic(stream) != (1, 0):
        raise ValueError('expected NPY v1')
    actual_shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
    if actual_shape != shape or dtype.kind != 'f' or dtype.itemsize not in (4, 8):
        raise ValueError('unexpected array shape or dtype')
    stream.seek(0)
    result = np.load(stream, allow_pickle=False)
    if not np.isfinite(result).all():
        raise ValueError('nonfinite service array')
    return result


@dataclass(frozen=True)
class PerceptionFrame:
    observation_id: str
    camera: str
    received_at: float
    rgb: np.ndarray
    depth_m: np.ndarray
    intrinsics: np.ndarray
    T_world_camera: np.ndarray
    world_frame: str
    calibration_id: str

    def __post_init__(self):
        if not all((self.observation_id, self.camera, self.world_frame, self.calibration_id)):
            raise ValueError('observation, camera, coordinate frame and calibration IDs are required')
        rgb = np.asarray(self.rgb)
        depth = np.asarray(self.depth_m, dtype=np.float32)
        intrinsics = np.asarray(self.intrinsics, dtype=float)
        if (rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8
                or min(rgb.shape[:2]) == 0 or rgb.shape[0] * rgb.shape[1] > 8_000_000):
            raise ValueError('expected uint8 HxWx3 RGB, at most eight million pixels')
        if depth.shape != rgb.shape[:2] or not np.isfinite(depth).all() or (depth < 0).any():
            raise ValueError('depth must be aligned to RGB, finite metres, with zero for invalid pixels')
        if (intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all()
                or intrinsics[0, 0] <= 0 or intrinsics[1, 1] <= 0
                or not np.allclose(intrinsics[2], [0, 0, 1])
                or not np.allclose([intrinsics[0, 1], intrinsics[1, 0]], 0)):
            raise ValueError('expected rectified pinhole intrinsics')
        if not math.isfinite(self.received_at) or self.received_at <= 0:
            raise ValueError('invalid frame timestamp')
        values = {'rgb': rgb, 'depth_m': depth, 'intrinsics': intrinsics,
                  'T_world_camera': _transform(self.T_world_camera)}
        for name, value in values.items():
            value = value.copy()
            value.setflags(write=False)
            object.__setattr__(self, name, value)

    @classmethod
    def from_observation(cls, observation, camera: str, *, world_frame: str,
                         calibration_id: str, depth_aligned_to_rgb: bool,
                         max_frame_age_s: float = 0.3, now: float | None = None):
        """Convert an existing frame, explicitly asserting RGB/depth registration.

        T_world_camera must be calibrated for this frame. This method does not
        solve wrist-camera kinematics or promise exposure synchronization.
        """
        if depth_aligned_to_rgb is not True:
            raise ValueError('RGB/depth registration must be explicitly confirmed')
        if not math.isfinite(max_frame_age_s) or max_frame_age_s <= 0:
            raise ValueError('max_frame_age_s must be positive and finite')
        frame = observation.frames[camera]
        synchronization = getattr(observation, 'synchronization', None)
        if synchronization is not None:
            pair = synchronization.get('cameras', {}).get(camera, {})
            if pair.get('matched') is not True:
                raise ValueError('observation has rejected camera/robot synchronization')
        age = (perf_counter() if now is None else now) - frame.t
        if not math.isfinite(age) or not 0 <= age <= max_frame_age_s:
            raise ValueError('frame is stale or from the future')
        if frame.distortion is not None and np.any(np.asarray(frame.distortion) != 0):
            raise ValueError('rectified input is required')
        if frame.depth is None or frame.depth.dtype != np.uint16:
            raise ValueError('CameraFrame depth must be uint16 millimetres')
        return cls(observation.observation_id, camera, frame.t, frame.rgb,
                   frame.depth.astype(np.float32) * 0.001, frame.intrinsics,
                   frame.T_world_camera, world_frame, calibration_id)

    def masked_points(self, mask: np.ndarray) -> np.ndarray:
        """Return observed surface points, not an inferred volume or object centre."""
        mask = np.asarray(mask)
        if mask.shape != self.depth_m.shape or not np.isin(mask, [0, 1]).all():
            raise ValueError('mask must be binary and match RGB/depth dimensions')
        rows, columns = np.where(mask.astype(bool) & (self.depth_m > 0))
        depth = self.depth_m[rows, columns]
        points = np.column_stack(((columns - self.intrinsics[0, 2]) * depth / self.intrinsics[0, 0],
                                  (rows - self.intrinsics[1, 2]) * depth / self.intrinsics[1, 1], depth))
        return points @ self.T_world_camera[:3, :3].T + self.T_world_camera[:3, 3]

    def bundlesdf_payload(self) -> dict:
        """Encode camera-space input; identity extrinsics avoid Portal fallback.

        ENPIRE's station-dependent world transform is not used by our client.
        World coordinates are always recomputed locally from ob_in_cam.
        """
        import cv2

        ok, encoded = cv2.imencode('.png', cv2.cvtColor(self.rgb, cv2.COLOR_RGB2BGR))
        if not ok:
            raise ValueError('RGB PNG encoding failed')
        return {
            'camera': self.camera, 'image_base64': base64.b64encode(encoded).decode('ascii'),
            'depth_base64': _encode_array(self.depth_m),
            'intrinsics': [float(self.intrinsics[0, 0]), float(self.intrinsics[1, 1]),
                           float(self.intrinsics[0, 2]), float(self.intrinsics[1, 2])],
            'extrinsics': {'position': [0., 0., 0.], 'rotation': np.eye(3).ravel().tolist()},
        }


@dataclass
class GraspCandidates:
    observation_id: str
    camera: str
    world_frame: str
    calibration_id: str
    T_camera_grasp: np.ndarray
    T_world_grasp: np.ndarray
    scores: np.ndarray
    widths_m: np.ndarray
    frame_received_at: float
    frame_convention: str = 'AnyGrasp: +X approach, +Y finger opening, +Z height; not robot TCP'

    def tcp_targets(self, T_grasp_tcp: np.ndarray) -> np.ndarray:
        """Requires an explicit calibrated grasp-frame-to-TCP transform; no dispatch."""
        return self.T_world_grasp @ _transform(T_grasp_tcp)


class AnyGraspClient(_HttpToolClient):
    def health(self) -> dict:
        response = self._request('/health')
        return {'reachable': True,
                'ready': response.get('status') == 'ok' and response.get('model_loaded') is True,
                'model_loaded': response.get('model_loaded'),
                'checkpoint_path': response.get('checkpoint_path'), 'process_owner': 'external'}

    def plan(self, frame: PerceptionFrame, mask: np.ndarray, *, max_grasps: int = 20,
             z_range_m: tuple[float, float] = (1e-6, 1.5)) -> GraspCandidates:
        mask = np.asarray(mask)
        if (mask.shape != frame.depth_m.shape or not np.isin(mask, [0, 1]).all()
                or not np.any(mask.astype(bool) & (frame.depth_m > 0))):
            raise ValueError('a binary target mask with valid depth is required')
        if type(max_grasps) is not int or not 1 <= max_grasps <= 256:
            raise ValueError('max_grasps must be between 1 and 256')
        if (len(z_range_m) != 2 or not np.isfinite(z_range_m).all()
                or not 0 < z_range_m[0] < z_range_m[1]):
            raise ValueError('invalid depth range')
        response = self._request('/plan', {
            'rgb_base64': _encode_array(frame.rgb), 'depth_base64': _encode_array(frame.depth_m),
            'cam_K_base64': _encode_array(frame.intrinsics),
            'segmap_base64': _encode_array(mask.astype(np.int32)), 'segmap_id': 1,
            'max_grasps': max_grasps, 'z_range': list(z_range_m), 'workspace_margin': 0.02,
            'collision_detection': True, 'object_input_mode': 'segmented_object_cloud',
        })
        try:
            count = response['n_grasps']
            if type(count) is not int or not 0 <= count <= max_grasps:
                raise ValueError('invalid grasp count')
            poses = _decode_array(response['grasps_base64'], (count, 4, 4))
            scores = _decode_array(response['scores_base64'], (count,))
            widths = _decode_array(response['widths_base64'], (count,))
            for pose in poses:
                _transform(pose)
            if (widths < 0).any() or ((scores < 0) | (scores > 1)).any():
                raise ValueError('invalid widths or scores')
        except (KeyError, TypeError, ValueError, EOFError, OSError) as exc:
            raise ExternalToolError(f'invalid AnyGrasp response: {exc}') from exc
        return GraspCandidates(frame.observation_id, frame.camera, frame.world_frame,
                               frame.calibration_id, poses, frame.T_world_camera @ poses,
                               scores, widths, frame.received_at)


@dataclass(frozen=True)
class StereoDepth:
    """Metric z-depth in the LEFT IR camera (0 = invalid) at the inference size."""

    depth: np.ndarray
    intrinsics: np.ndarray       # left IR K scaled to ``depth``
    seconds: float
    model_ref: str


class StereoDepthClient(_HttpToolClient):
    """On-demand stereo depth (Fast-FoundationStereo service) from a D435 IR pair.

    Not for the real-time safety loop (~0.4 s per pair). Use ``register_depth``
    to move the result into the colour camera.
    """

    def health(self) -> dict:
        response = self._request('/health')
        return {'reachable': True,
                'ready': response.get('status') == 'ok' and response.get('model_loaded') is True,
                'model_loaded': response.get('model_loaded'),
                'model_ref': response.get('model_ref'), 'process_owner': 'external'}

    def depth(self, left: np.ndarray, right: np.ndarray, K_left, *, baseline_m: float,
              ppx_right: float | None = None, width: int = 640) -> StereoDepth:
        left, right = np.asarray(left), np.asarray(right)
        if left.dtype != np.uint8 or left.shape != right.shape or right.dtype != np.uint8:
            raise ValueError('expected two uint8 images of the same shape')
        K_left = np.asarray(K_left, dtype=float)
        if not (math.isfinite(baseline_m) and 0 < baseline_m < 1 and 32 <= width <= 2048):
            raise ValueError('invalid baseline or width')

        def encode(image):
            buffer = io.BytesIO()
            np.save(buffer, image, allow_pickle=False)
            return base64.b64encode(buffer.getvalue()).decode('ascii')

        response = self._request('/depth', {
            'left': encode(left), 'right': encode(right), 'fx': float(K_left[0, 0]),
            'baseline_m': float(baseline_m), 'ppx_left': float(K_left[0, 2]),
            'ppx_right': float(K_left[0, 2] if ppx_right is None else ppx_right),
            'width': int(width)})
        try:
            shape = tuple(int(v) for v in response['shape'])
            depth = _decode_array(response['depth'], shape)
            scale = float(response['scale'])
        except (KeyError, TypeError, ValueError) as exc:
            raise ExternalToolError(f'invalid stereo depth response: {exc}') from exc
        K = K_left.copy()
        K[:2] *= scale
        return StereoDepth(depth.astype(np.float32), K, float(response.get('seconds', 0.)),
                           str(response.get('model_ref', '')))


def register_depth(depth, K_depth, K_color, T_color_depth, color_shape, max_depth_m=6.0):
    """Z-depth from one camera into another (e.g. IR-left -> colour), z-buffered.

    Each depth pixel is splatted over 2x2 target pixels so a lower-resolution source
    leaves no grid of holes; 0 marks pixels without data.
    """
    depth = np.nan_to_num(np.asarray(depth, float), nan=0.)
    K_depth, K_color = np.asarray(K_depth, float), np.asarray(K_color, float)
    T = np.asarray(T_color_depth, float)
    v, u = np.nonzero((depth > 0) & (depth < max_depth_m))
    z = depth[v, u]
    points = np.stack([(u - K_depth[0, 2]) * z / K_depth[0, 0],
                       (v - K_depth[1, 2]) * z / K_depth[1, 1], z], 1) @ T[:3, :3].T + T[:3, 3]
    front = points[:, 2] > 1e-3
    points = points[front]
    uc = np.rint(K_color[0, 0] * points[:, 0] / points[:, 2] + K_color[0, 2]).astype(int)
    vc = np.rint(K_color[1, 1] * points[:, 1] / points[:, 2] + K_color[1, 2]).astype(int)
    h, w = color_shape[:2]
    out = np.full((h, w), np.inf)
    for du in (0, 1):
        for dv in (0, 1):
            uu, vv = uc + du, vc + dv
            ok = (uu >= 0) & (uu < w) & (vv >= 0) & (vv < h)
            np.minimum.at(out, (vv[ok], uu[ok]), points[ok, 2])
    out[~np.isfinite(out)] = 0.
    return out.astype(np.float32)


@dataclass
class TrackingSession:
    name: str
    camera: str
    world_frame: str
    calibration_id: str
    intrinsics: np.ndarray
    T_world_camera: np.ndarray
    last_submitted_observation_id: str
    last_received_at: float
    closed: bool = False


@dataclass
class TrackingPose:
    name: str
    tracking: bool
    frame_idx: int
    score: float
    T_camera_object: np.ndarray | None
    T_world_object: np.ndarray | None
    world_frame: str
    last_submitted_observation_id: str
    source_observation_id: str | None = None
    freshness: str = 'unverified: upstream has no source frame ID or pose timestamp'
    reconstruction_status: str = 'unverified: pose endpoint does not certify a mesh'


class BundleSdfClient(_HttpToolClient):
    """Fixed-camera streaming adapter. Calls do not execute robot motion.

    Remote world poses are intentionally ignored: ENPIRE applies station-specific
    optical flips. Raw camera poses are transformed locally using our calibration.
    """

    def health(self) -> dict:
        response = self._request('/health')
        return {'reachable': True, 'ready': None, 'service_status': response.get('status'),
                'reason': 'upstream health does not report model or reconstruction readiness',
                'process_owner': 'external'}

    def _push(self, frame: PerceptionFrame) -> None:
        response = self._request('/push_frame', frame.bundlesdf_payload())
        if (response.get('ok') is not True or response.get('camera') != frame.camera
                or type(response.get('seq')) is not int or response['seq'] < 0):
            raise ExternalToolError('invalid BundleSDF push acknowledgement')

    def start(self, frame: PerceptionFrame, query: str, *, name: str) -> TrackingSession:
        if not query.strip() or not name or any(character not in
                'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for character in name):
            raise ValueError('query and a simple unique session name are required')
        self._push(frame)
        response = self._request('/add_detection', {
            'text': query, 'camera': frame.camera, 'name': name,
            'intrinsics': [float(frame.intrinsics[0, 0]), float(frame.intrinsics[1, 1]),
                           float(frame.intrinsics[0, 2]), float(frame.intrinsics[1, 2])],
        })
        if response.get('name') != name:
            raise ExternalToolError('BundleSDF did not acknowledge the requested session')
        return TrackingSession(name, frame.camera, frame.world_frame, frame.calibration_id,
                               frame.intrinsics.copy(), frame.T_world_camera.copy(),
                               frame.observation_id, frame.received_at)

    def push(self, session: TrackingSession, frame: PerceptionFrame) -> None:
        if session.closed:
            raise ValueError('tracking session is closed')
        if (frame.camera != session.camera or frame.world_frame != session.world_frame
                or frame.calibration_id != session.calibration_id
                or not np.allclose(frame.intrinsics, session.intrinsics)
                or not np.allclose(frame.T_world_camera, session.T_world_camera)):
            raise ValueError('this adapter requires a fixed camera and unchanged calibration')
        if (frame.received_at <= session.last_received_at
                or frame.observation_id == session.last_submitted_observation_id):
            raise ValueError('tracking requires a new observation with increasing frame time')
        self._push(frame)
        session.last_received_at = frame.received_at
        session.last_submitted_observation_id = frame.observation_id

    def pose(self, session: TrackingSession) -> TrackingPose:
        if session.closed:
            raise ValueError('tracking session is closed')
        response = self._request('/get_detection/' + quote(session.name, safe=''))
        try:
            tracking, frame_idx, score = response['tracking'], response['frame_idx'], float(response['score'])
            if (type(tracking) is not bool or type(frame_idx) is not int or frame_idx < 0
                    or not math.isfinite(score) or not 0 <= score <= 1):
                raise ValueError('invalid tracking status')
            camera_pose = response.get('ob_in_cam')
            camera_pose = None if camera_pose is None else _transform(camera_pose)
        except (KeyError, TypeError, ValueError) as exc:
            raise ExternalToolError(f'invalid BundleSDF pose: {exc}') from exc
        world_pose = None if camera_pose is None else session.T_world_camera @ camera_pose
        return TrackingPose(session.name, tracking, frame_idx, score, camera_pose, world_pose,
                            session.world_frame, session.last_submitted_observation_id)

    def end(self, session: TrackingSession) -> None:
        if session.closed:
            return
        response = self._request('/end_detection/' + quote(session.name, safe=''), {})
        if response.get('ok') is not True:
            raise ExternalToolError('BundleSDF did not acknowledge session end')
        session.closed = True


@dataclass
class ObjectInspection:
    segmentation: SegmentationResult
    points_world: np.ndarray
    visible_bounds_world: np.ndarray | None
    parts: dict[str, SegmentationResult]
    part_relationship_status: str = 'user-specified parent; not a verified joint or affordance'


def inspect_object(sam: Sam3Client, frame: PerceptionFrame, query: str, *,
                   part_queries: dict[str, str] | None = None) -> ObjectInspection:
    """Inspect one unambiguous target and optionally prompt for parts in its ROI."""
    result = sam.segment(frame.rgb, query, observation_id=frame.observation_id, camera=frame.camera)
    if len(result.detections) != 1:
        raise ExternalToolError(f'expected one target, found {len(result.detections)}; refine the prompt')
    parent = result.detections[0]
    points = frame.masked_points(parent.mask)
    bounds = np.stack((points.min(axis=0), points.max(axis=0))) if len(points) else None
    left, top, width, height = parent.bbox_xywh
    parts = {}
    for part_id, part_query in (part_queries or {}).items():
        candidates = sam.segment(frame.rgb[top:top + height, left:left + width], part_query,
                                 observation_id=frame.observation_id, camera=frame.camera)
        full_detections = []
        for candidate in candidates.detections:
            full_mask = np.zeros(frame.depth_m.shape, dtype=bool)
            full_mask[top:top + height, left:left + width] = candidate.mask
            part_left, part_top, part_width, part_height = candidate.bbox_xywh
            full_detections.append(Segmentation(full_mask,
                (part_left + left, part_top + top, part_width, part_height), candidate.score, candidate.area_px))
        parts[part_id] = SegmentationResult(frame.observation_id, frame.camera, part_query,
                                            tuple(full_detections))
    return ObjectInspection(result, points, bounds, parts)