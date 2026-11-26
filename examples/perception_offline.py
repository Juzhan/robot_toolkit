"""Synthetic object/part inspection; no models, cameras, network or robot motion."""

import json

import numpy as np

from robot_toolkit.perception.external_tools import Segmentation, SegmentationResult
from robot_toolkit.perception.perception import PerceptionFrame, inspect_object


class SyntheticSegmenter:
    def segment(self, rgb, query, *, observation_id, camera):
        bbox = (2, 2, 6, 4) if query == 'drawer' else (1, 1, 2, 1)
        left, top, width, height = bbox
        mask = np.zeros(rgb.shape[:2], dtype=bool)
        mask[top:top + height, left:left + width] = True
        return SegmentationResult(observation_id, camera, query,
            (Segmentation(mask, bbox, 1.0, int(mask.sum())),), provider='synthetic_demo')


def main():
    frame = PerceptionFrame(
        'synthetic-1', 'top', 1.0,
        np.zeros((8, 10, 3), dtype=np.uint8), np.full((8, 10), 0.5, dtype=np.float32),
        np.array([[100, 0, 5], [0, 100, 4], [0, 0, 1]]), np.eye(4), 'demo_world', 'synthetic-cal',
    )
    result = inspect_object(SyntheticSegmenter(), frame, 'drawer',
                            part_queries={'handle': 'drawer handle'})
    handle = result.parts['handle'].detections[0]
    assert result.points_world.shape == (24, 3)
    assert handle.bbox_xywh == (3, 3, 2, 1)
    print(json.dumps({
        'source': 'synthetic arrays and masks; NO model inference',
        'observed_surface_points': len(result.points_world),
        'visible_bounds_m': result.visible_bounds_world.tolist(),
        'handle_bbox_xywh': handle.bbox_xywh,
        'part_relationship': result.part_relationship_status,
        'complete_reconstruction': False, 'robot_commands': 0,
    }, indent=2))


if __name__ == '__main__':
    main()