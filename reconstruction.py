"""Measured object-surface fusion. No device acquisition or motion dispatch."""

from __future__ import annotations

import math

import numpy as np

from .perception import PerceptionFrame, _transform


def camera_roll_near(pose, reference_pose) -> np.ndarray:
    """Keep optical position/aim, selecting roll near the current camera right axis."""
    pose = _transform(pose)
    reference = _transform(reference_pose)
    forward = pose[:3, 2]
    right = reference[:3, 0] - forward * np.dot(reference[:3, 0], forward)
    if np.linalg.norm(right) < 1e-6:
        right = np.cross(reference[:3, 1], forward)
    right /= np.linalg.norm(right)
    pose[:3, :3] = np.column_stack((right, np.cross(forward, right), forward))
    return _transform(pose)


def front_hemisphere_views(center, robot_position, *, radius_m=.26,
                           azimuth_degrees=(-55., 0., 55.),
                           elevation_degrees=(30., 45., 60.)) -> list[dict]:
    """Optical camera poses on the robot-facing upper hemisphere, aimed at center."""
    center = np.asarray(center, dtype=float)
    robot_position = np.asarray(robot_position, dtype=float)
    if (center.shape != (3,) or robot_position.shape != (3,)
            or not np.isfinite(center).all() or not np.isfinite(robot_position).all()
            or not math.isfinite(radius_m) or radius_m <= 0):
        raise ValueError('finite 3D positions and positive radius are required')
    front = robot_position - center
    front[2] = 0
    if np.linalg.norm(front) < 1e-6:
        raise ValueError('robot-facing direction is undefined')
    front /= np.linalg.norm(front)
    up = np.array([0., 0., 1.])
    side = np.cross(up, front)
    azimuths = tuple(azimuth_degrees)
    elevations = tuple(elevation_degrees)
    if (not azimuths or not elevations
            or any(not math.isfinite(value) or abs(value) >= 90 for value in azimuths)
            or any(not math.isfinite(value) or not 0 < value < 90 for value in elevations)):
        raise ValueError('azimuth must be inside +/-90 and elevation inside (0, 90) degrees')
    poses = []
    for elevation in elevations:
        pitch = math.radians(elevation)
        for azimuth in azimuths:
            yaw = math.radians(azimuth)
            offset = radius_m * (math.cos(pitch) * (math.cos(yaw) * front + math.sin(yaw) * side)
                                 + math.sin(pitch) * up)
            forward = -offset / radius_m
            right = np.cross(forward, up)
            right /= np.linalg.norm(right)
            down = np.cross(forward, right)
            pose = np.eye(4)
            pose[:3, :3] = np.column_stack((right, down, forward))
            pose[:3, 3] = center + offset
            poses.append({'azimuth_degrees': float(azimuth), 'elevation_degrees': float(elevation),
                          'radius_m': radius_m, 'T_world_camera': _transform(pose)})
    return poses


def distinct_masks(masks, *, iou_threshold: float = 0.5) -> list[np.ndarray]:
    if not 0 < iou_threshold <= 1:
        raise ValueError('IoU threshold must be in (0, 1]')
    unique = []
    shape = None
    for value in masks:
        mask = np.asarray(value)
        if mask.ndim != 2 or not np.isin(mask, [0, 1]).all():
            raise ValueError('expected binary image masks')
        if shape is not None and mask.shape != shape:
            raise ValueError('masks must belong to the same image')
        shape = mask.shape
        mask = mask.astype(bool)
        if not mask.any():
            continue
        if any(np.count_nonzero(mask & other) / np.count_nonzero(mask | other)
               >= iou_threshold for other in unique):
            continue
        unique.append(mask.copy())
    return unique


def register_stationary_views(reference, views, *, initial_world_pose: np.ndarray,
                              correspondence_m: float = .01,
                              agreement_m: float = .006,
                              agreement_rad: float = .08) -> list[dict]:
    """Independently align stopped views; require two-camera pose agreement.

    This verifies local geometric consistency, not unique orientation for a
    symmetric object. The initial pose is a coarse prior, not rigid attachment.
    """
    import open3d as o3d

    initial = _transform(initial_world_pose)
    if len(views) < 2 or len({frame.camera for frame, _ in views}) != len(views):
        raise ValueError('at least two distinct camera views are required')
    if len({(frame.world_frame, frame.observation_id) for frame, _ in views}) != 1:
        raise ValueError('views must describe one stationary observation in one world frame')
    if not all(math.isfinite(value) and value > 0
               for value in (correspondence_m, agreement_m, agreement_rad)):
        raise ValueError('registration tolerances must be finite and positive')
    if len(reference.points) < 16:
        raise ValueError('insufficient reference surface')
    aligned = []
    for frame, mask in views:
        points = frame.masked_points(mask)
        if len(points) < 80:
            raise ValueError('insufficient stationary surface pixels')
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points)).voxel_down_sample(.003)
        result = o3d.pipelines.registration.registration_icp(
            cloud, reference, correspondence_m, np.linalg.inv(initial),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=80))
        if result.fitness < .6 or result.inlier_rmse > .003:
            raise ValueError('stationary surface registration failed')
        pose = np.linalg.inv(result.transformation)
        delta = np.linalg.inv(initial) @ pose
        angle = math.acos(float(np.clip((np.trace(delta[:3, :3]) - 1) / 2, -1, 1)))
        if np.linalg.norm(delta[:3, 3]) > .05 or angle > math.pi / 4:
            raise ValueError('registration left the coarse pose search region')
        aligned.append({'camera': frame.camera, 'T_world_object': pose,
                        'fitness': result.fitness, 'rmse_m': result.inlier_rmse})
    for index, first in enumerate(aligned):
        for second in aligned[index + 1:]:
            delta = np.linalg.inv(first['T_world_object']) @ second['T_world_object']
            angle = math.acos(float(np.clip((np.trace(delta[:3, :3]) - 1) / 2, -1, 1)))
            if np.linalg.norm(delta[:3, 3]) > agreement_m or angle > agreement_rad:
                raise ValueError('stationary cameras disagree on object alignment')
    return aligned


class ObjectReconstruction:
    """Fuse explicit per-frame object poses into an object-local TSDF.

    Pose provenance is an input contract, not a pose estimator. Callers must
    stop fusion when tracking, rigid attachment or calibration is uncertain.
    Unknown surfaces stay unknown; extracted meshes are not certified closed.
    """

    def __init__(self, *, voxel_size_m: float = 0.002, depth_max_m: float = 3.0):
        import open3d as o3d

        if not all(math.isfinite(value) and value > 0
                   for value in (voxel_size_m, depth_max_m)):
            raise ValueError('voxel size and depth limit must be positive finite metres')
        self.voxel_size_m = voxel_size_m
        self.depth_max_m = depth_max_m
        self.volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel_size_m, sdf_trunc=4 * voxel_size_m,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
        self.records = []
        self._seen = set()
        self._world_frame = None
        self._calibrations = {}

    def integrate(self, frame: PerceptionFrame, mask: np.ndarray, *,
                  T_world_object: np.ndarray, pose_observation_id: str,
                  pose_source: str) -> dict:
        import open3d as o3d

        key = (frame.camera, frame.observation_id)
        if key in self._seen:
            raise ValueError('duplicate camera observation')
        if pose_observation_id != frame.observation_id or not pose_source.strip():
            raise ValueError('object pose must identify this observation and its source')
        if self._world_frame not in (None, frame.world_frame):
            raise ValueError('world coordinate frame changed')
        if self._calibrations.get(frame.camera, frame.calibration_id) != frame.calibration_id:
            raise ValueError('camera calibration changed')
        transform = _transform(T_world_object)
        mask = np.asarray(mask)
        frame.masked_points(mask)
        valid = mask.astype(bool) & (frame.depth_m > 0) & (frame.depth_m < self.depth_max_m)
        if np.count_nonzero(valid) < 16:
            raise ValueError('insufficient measured surface pixels')
        depth = np.where(valid, frame.depth_m, 0).astype(np.float32)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(frame.rgb.copy()), o3d.geometry.Image(depth),
            depth_scale=1.0, depth_trunc=self.depth_max_m, convert_rgb_to_intensity=False)
        height, width = depth.shape
        intrinsics = o3d.camera.PinholeCameraIntrinsic(
            width, height, frame.intrinsics[0, 0], frame.intrinsics[1, 1],
            frame.intrinsics[0, 2], frame.intrinsics[1, 2])
        camera_from_object = np.linalg.inv(frame.T_world_camera) @ transform
        self.volume.integrate(rgbd, intrinsics, camera_from_object)
        record = {
            'camera': frame.camera, 'observation_id': frame.observation_id,
            'pose_source': pose_source, 'surface_pixels': int(valid.sum()),
            'T_world_object': transform.tolist(),
            'T_object_camera': np.linalg.inv(camera_from_object).tolist(),
        }
        self.records.append(record)
        self._seen.add(key)
        self._world_frame = frame.world_frame
        self._calibrations[frame.camera] = frame.calibration_id
        return record

    def point_cloud(self):
        return self.volume.extract_point_cloud()

    def mesh(self):
        mesh = self.volume.extract_triangle_mesh()
        mesh.compute_vertex_normals()
        return mesh