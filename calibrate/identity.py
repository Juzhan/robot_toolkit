"""Camera/board identity shared by capture, planning and verification.

Accepts JSON mappings or configuration/camera objects, without device imports.
"""
from collections.abc import Mapping


def _get(value, key, default=None):
    return value.get(key, default) if isinstance(value, Mapping) else getattr(value, key, default)


def calibration_identity(camera, cfg):
    return dict(
        camera=_get(camera, 'name'), serial=_get(camera, 'serial', ''),
        mode=_get(cfg, 'eye'),
        image_size=[_get(camera, 'width'), _get(camera, 'height')],
        board=dict(pattern=_get(cfg, 'board'), squares=list(_get(cfg, 'squares')),
                   square_size=_get(cfg, 'square_size'), marker_size=_get(cfg, 'marker_size'),
                   aruco_dict=_get(cfg, 'aruco_dict')),
    )
