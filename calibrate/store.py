"""Persistent calibration store shared by every robot and camera driver.

The file is plain JSON so it can be inspected, diffed and copied between
machines.  Extrinsics are stored in the same two forms the existing projects
use: ``tcp_camera`` (eye-in-hand) and ``base_camera`` (fixed camera).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from ..core.mathutils import (invert_transform, matrix_to_pose, pose_to_matrix,
                              rotvec_to_matrix)
from ..errors import CalibrationError


@dataclass
class CameraCalibration:
    name: str
    intrinsics: np.ndarray | None = None
    distortion: np.ndarray | None = None
    tcp_camera: np.ndarray | None = None
    base_camera: np.ndarray | None = None
    width: int = 0
    height: int = 0
    updated: str = ''
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'intrinsics': None if self.intrinsics is None else np.asarray(self.intrinsics).tolist(),
            'distortion': None if self.distortion is None else np.asarray(self.distortion).reshape(-1).tolist(),
            'tcp_camera': None if self.tcp_camera is None else np.asarray(self.tcp_camera).tolist(),
            'base_camera': None if self.base_camera is None else np.asarray(self.base_camera).tolist(),
            'width': self.width, 'height': self.height,
            'updated': self.updated, 'meta': dict(self.meta),
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'CameraCalibration':
        def array(value, shape=None):
            if value is None:
                return None
            result = np.asarray(value, dtype=float)
            return result.reshape(shape) if shape else result

        return cls(name=data.get('name', ''),
                   intrinsics=array(data.get('intrinsics'), (3, 3)),
                   distortion=array(data.get('distortion')),
                   tcp_camera=array(data.get('tcp_camera'), (4, 4)),
                   base_camera=array(data.get('base_camera'), (4, 4)),
                   width=int(data.get('width', 0)), height=int(data.get('height', 0)),
                   updated=data.get('updated', ''), meta=dict(data.get('meta', {})))

    def world_transform(self, base_tcp=None) -> np.ndarray | None:
        if self.base_camera is not None:
            return self.base_camera
        if self.tcp_camera is not None and base_tcp is not None:
            return pose_to_matrix(base_tcp) @ self.tcp_camera
        return None


class CalibrationStore:
    VERSION = 1

    def __init__(self, path):
        self.path = Path(path)
        self.cameras: dict[str, CameraCalibration] = {}
        self.load()

    # ------------------------------------------------------------------- io
    def load(self) -> 'CalibrationStore':
        self.cameras = {}
        if not self.path.exists():
            return self
        raw = json.loads(self.path.read_text(encoding='utf-8'))
        for name, entry in raw.get('cameras', {}).items():
            entry.setdefault('name', name)
            self.cameras[name] = CameraCalibration.from_dict(entry)
        return self

    def save(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {'version': self.VERSION,
                   'cameras': {name: calibration.to_dict()
                               for name, calibration in self.cameras.items()}}
        temporary = self.path.with_suffix(self.path.suffix + '.tmp')
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
        temporary.replace(self.path)
        return self.path

    # --------------------------------------------------------------- update
    def get(self, name: str) -> CameraCalibration:
        if name not in self.cameras:
            self.cameras[name] = CameraCalibration(name=name)
        return self.cameras[name]

    def set_intrinsics(self, name: str, K, dist=None, image_size=None, meta=None) -> CameraCalibration:
        entry = self.get(name)
        entry.intrinsics = np.asarray(K, dtype=float).reshape(3, 3)
        entry.distortion = np.zeros(5) if dist is None else np.asarray(dist, dtype=float).reshape(-1)
        if image_size is not None:
            entry.width, entry.height = int(image_size[0]), int(image_size[1])
        entry.updated = _now()
        if meta:
            entry.meta.update(meta)
        return entry

    def set_hand_eye(self, name: str, mode: str, transform, residual=None) -> CameraCalibration:
        entry = self.get(name)
        matrix = _as_matrix(transform)
        if mode == 'eye_in_hand':
            entry.tcp_camera = matrix
        elif mode == 'eye_to_hand':
            entry.base_camera = matrix
        else:
            raise CalibrationError("mode must be 'eye_in_hand' or 'eye_to_hand'")
        entry.updated = _now()
        if residual is not None:
            entry.meta['hand_eye_residual'] = float(residual)
        entry.meta['hand_eye_mode'] = mode
        return entry

    def remove(self, name: str) -> bool:
        return self.cameras.pop(name, None) is not None

    # ---------------------------------------------------------------- apply
    def apply(self, camera, base_tcp=None, require_extrinsics: bool = False) -> bool:
        """Push intrinsics/extrinsics into a ``Camera``-like object."""
        entry = self.cameras.get(getattr(camera, 'name', ''))
        if entry is None:
            if require_extrinsics:
                raise CalibrationError(f'no calibration for camera {getattr(camera, "name", "?")}')
            return False
        if entry.intrinsics is not None:
            camera.intrinsics = entry.intrinsics.copy()
            camera.distortion = (None if entry.distortion is None else entry.distortion.copy())
        transform = entry.world_transform(base_tcp)
        if transform is not None:
            camera.T_world_camera = transform.copy()
        return True

    def apply_all(self, hub, base_tcp=None, require_extrinsics: bool = False) -> dict:
        result = {}
        for name, camera in hub.cameras.items():
            result[name] = self.apply(camera, base_tcp, require_extrinsics)
        return result

    # -------------------------------------------------------------- legacy
    def import_legacy(self, config_path, camera_names=None) -> int:
        """Import ``tcp_camera`` / ``base_camera`` from an existing JSON config."""
        raw = json.loads(Path(config_path).read_text(encoding='utf-8'))
        imported = 0
        for item in raw.get('cameras', []):
            name = item.get('name')
            if not name or (camera_names and name not in camera_names):
                continue
            entry = self.get(name)
            tcp = item.get('tcp_camera')
            base = item.get('base_camera')
            if tcp is not None and len(tcp) == 6:
                entry.tcp_camera = pose_to_matrix(tcp)
                entry.meta['imported_from'] = str(config_path)
                imported += 1
            if base is not None and len(base) == 6:
                entry.base_camera = pose_to_matrix(base)
                entry.meta['imported_from'] = str(config_path)
                imported += 1
            entry.width = int(item.get('width', entry.width))
            entry.height = int(item.get('height', entry.height))
        return imported

    def export_legacy(self, camera_names=None) -> dict:
        """Return camera config entries with 6-vector extrinsics."""
        cameras = []
        for name, entry in self.cameras.items():
            if camera_names and name not in camera_names:
                continue
            item = {'name': name,
                    'width': entry.width or 640,
                    'height': entry.height or 480}
            if entry.intrinsics is not None:
                item['intrinsics'] = entry.intrinsics.tolist()
            if entry.tcp_camera is not None:
                item['tcp_camera'] = matrix_to_pose(entry.tcp_camera).tolist()
            if entry.base_camera is not None:
                item['base_camera'] = matrix_to_pose(entry.base_camera).tolist()
            cameras.append(item)
        return {'cameras': cameras}

    def summary(self) -> list[dict]:
        rows = []
        for name, entry in sorted(self.cameras.items()):
            rows.append({
                'camera': name,
                'intrinsics': entry.intrinsics is not None,
                'distortion': entry.distortion is not None,
                'tcp_camera': entry.tcp_camera is not None,
                'base_camera': entry.base_camera is not None,
                'updated': entry.updated,
            })
        return rows


def _as_matrix(value) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape == (4, 4):
        return array
    if array.shape == (3, 3):
        result = np.eye(4)
        result[:3, :3] = array
        return result
    if array.size == 6:
        return pose_to_matrix(array.reshape(6))
    raise CalibrationError(f'cannot interpret calibration transform of shape {array.shape}')


def _now() -> str:
    return datetime.now().isoformat(timespec='seconds')


def compose_camera_chain(base_tcp, tcp_camera) -> np.ndarray:
    """``T_base_camera = T_base_tcp @ T_tcp_camera``."""
    return pose_to_matrix(base_tcp) @ _as_matrix(tcp_camera)


def inverse_extrinsics(transform) -> np.ndarray:
    return invert_transform(_as_matrix(transform))


def rotation_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    return rotvec_to_matrix([roll, pitch, yaw])
