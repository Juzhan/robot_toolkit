"""Fuse measured surfaces from one synchronized capture and real detection artifacts."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from robot_toolkit.perception.perception import PerceptionFrame
from robot_toolkit.perception.reconstruction import ObjectReconstruction, distinct_masks


def reconstruct(frames: Path, detections: Path, output: Path, step: int = 0):
    import open3d as o3d

    output.mkdir(parents=True, exist_ok=False)
    report = json.loads((detections / 'report.json').read_text())
    objects = []
    for camera in dict.fromkeys(record['camera'] for record in report['records']):
        records = [record for record in report['records']
                   if record['step'] == step and record['camera'] == camera]
        if not records:
            continue
        with np.load(frames / f'{camera}_{step:04d}.npz', allow_pickle=False) as data:
            frame = PerceptionFrame(str(data['observation_id']), camera, 1., data['rgb'],
                                    data['depth_m'], data['intrinsics'], data['T_world_camera'],
                                    'capture_world', 'capture_calibration')
        rgb_hash = hashlib.sha256(frame.rgb.tobytes()).hexdigest()
        if any(record['rgb_sha256'] != rgb_hash for record in records):
            raise ValueError('detection RGB hash does not match the depth capture')
        masks = []
        for record in records:
            with np.load(detections / record['artifact'], allow_pickle=False) as data:
                masks.extend(data['masks'])
        for mask in distinct_masks(masks):
            points = frame.masked_points(mask)
            if len(points) < 16:
                continue
            measured = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
            matches = []
            for candidate in objects:
                distances = np.asarray(measured.compute_point_cloud_distance(candidate['cloud']))
                if np.mean(distances < .006) > .5:
                    matches.append(candidate)
            if len(matches) > 1:
                raise ValueError('ambiguous cross-camera association; do not merge')
            if matches:
                candidate = matches[0]
            else:
                pose = np.eye(4)
                pose[:3, 3] = np.median(points, axis=0)
                candidate = {'id': f'object_{len(objects):03d}', 'fusion': ObjectReconstruction(),
                             'pose': pose, 'cloud': measured, 'points': [], 'masks': []}
                objects.append(candidate)
            candidate['fusion'].integrate(
                frame, mask, T_world_object=candidate['pose'],
                pose_observation_id=frame.observation_id, pose_source='single synchronized static capture')
            candidate['points'].append(points)
            candidate['masks'].append({'camera': camera, 'pixels': int(mask.sum())})
            candidate['cloud'] += measured
    summaries = []
    for candidate in objects:
        name = candidate['id']
        fusion = candidate['fusion']
        cloud, mesh = fusion.point_cloud(), fusion.mesh()
        if not o3d.io.write_point_cloud(str(output / f'{name}.ply'), cloud):
            raise RuntimeError('point cloud export failed')
        if not o3d.io.write_triangle_mesh(str(output / f'{name}_mesh.ply'), mesh):
            raise RuntimeError('mesh export failed')
        points = np.concatenate(candidate['points'])
        summaries.append({'id': name, 'T_world_object': candidate['pose'].tolist(),
                          'observed_bounds_world': np.quantile(points, [.01, .99], axis=0).tolist(),
                          'points': len(cloud.points), 'triangles': len(mesh.triangles),
                          'watertight': mesh.is_watertight(), 'observations': fusion.records,
                          'next_action': 'acquire additional view; grasp feasibility not verified',
                          'complete': False})
    result = {'input_fields': ['rgb', 'depth_m', 'intrinsics', 'T_world_camera', 'observation_id'],
              'detection_report': str(detections / 'report.json'), 'step': step,
              'scope': 'Measured static partial surfaces; no hidden-surface completion or motion',
              'objects': summaries}
    (output / 'report.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frames', type=Path, required=True)
    parser.add_argument('--detections', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--step', type=int, default=0)
    args = parser.parse_args()
    reconstruct(args.frames, args.detections, args.output, args.step)


if __name__ == '__main__':
    main()