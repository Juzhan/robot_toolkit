"""Synthetic recorded-frame replay; no devices, motion, workers, or waits."""

from __future__ import annotations

import numpy as np

from robot_toolkit import (ObjectSpec, Observation, PlacementCondition,
                           PlanarColorEstimator, ResetLayoutVerifier, SceneSpec)
from robot_toolkit.camera.base import CameraFrame


def main():
    spec = SceneSpec(
        camera='top', calibration_id='synthetic-only-v1', coordinate_frame='synthetic_table',
        image_size=(100, 80), image_to_plane=np.diag([0.001, 0.001, 1.0]),
        workspace_xy=(0.0, 0.0, 0.1, 0.08),
        objects=(ObjectSpec('red_block', (((0, 150, 150), (10, 255, 255)),),
                            min_area_px=200, max_area_px=500),))
    clock = [10.01]
    estimator = PlanarColorEstimator(spec, clock=lambda: clock[0])
    verifier = ResetLayoutVerifier(
        spec, (PlacementCondition('red_block', (0.0395, 0.0295), tolerance_m=0.005),),
        clock=lambda: clock[0])
    scenarios = [('stable', 30), ('stable', 30), ('stable', 30),
                 ('missing', None), ('displaced', 55)]
    results = []
    for index, (label, left) in enumerate(scenarios):
        stamp = 10.0 + index * 0.12
        clock[0] = stamp + 0.01
        image = np.zeros((80, 100, 3), dtype=np.uint8)
        if left is not None:
            image[20:40, left:left + 20] = [255, 0, 0]
        observation = Observation(
            f'frame-{index}', stamp, stamp, False, None,
            frames={'top': CameraFrame('top', stamp, rgb=image)})
        scene = estimator.estimate(observation)
        check = verifier.check(scene)
        results.append(check)
        print(f'{label:10s} valid={scene.valid!s:5s} '
              f'layout_ready={check.layout_ready!s:5s} reason={check.reason}')
    assert [item.layout_ready for item in results] == [False, False, True, False, False]
    print('PASS: layout checks only; no robot was connected or moved')
    return results


if __name__ == '__main__':
    main()