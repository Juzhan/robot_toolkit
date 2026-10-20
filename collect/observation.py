"""Helpers bridging camera snapshots and episode capture."""

from __future__ import annotations

import numpy as np


def split_camera_frames(frames) -> tuple[dict, dict]:
    """Convert ``{name: CameraFrame}`` into ``(images, depth_payload)``.

    ``depth_payload`` uses the flat keys understood by ``EpisodeRecorder``:
    ``<name>_K`` and ``<name>_T_world_camera`` next to ``<name>_depth_mm``.
    """
    images, depth = {}, {}
    for name, frame in frames.items():
        if frame.rgb is not None:
            images[name] = np.ascontiguousarray(frame.rgb)
        if frame.depth is not None:
            depth[name] = frame.depth
            if frame.intrinsics is not None:
                depth[f'{name}_K'] = frame.intrinsics
            if frame.T_world_camera is not None:
                depth[f'{name}_T_world_camera'] = frame.T_world_camera
    return images, depth


def camera_images(frames) -> dict:
    return {name: frame.rgb for name, frame in frames.items() if frame.rgb is not None}
