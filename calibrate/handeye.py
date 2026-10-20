"""Hand-eye calibration with a closed-form AX = XB solver (no OpenCV needed).

Two mountings are supported:

``eye_in_hand``
    Camera rigidly attached to the tool.  Returns ``T_tcp_camera``.
``eye_to_hand``
    Camera fixed in the cell.  Returns ``T_base_camera``.

Inputs are corresponding transforms sampled at several robot poses:

* ``T_base_tcp[i]`` from the robot
* ``T_cam_target[i]`` from a marker/board detection

The solver is Park & Martin (1994): it is closed-form, deterministic and easy
to test offline, which matters more for a shared toolkit than a marginally
better nonlinear refinement.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core.mathutils import (invert_transform, matrix_to_rotvec, orthonormalize,
                              pose_to_matrix)
from ..errors import CalibrationError

MODES = ('eye_in_hand', 'eye_to_hand')


@dataclass
class HandEyeResult:
    transform: np.ndarray
    mode: str
    pairs: int
    rotation_rms: float
    translation_rms: float
    per_pair: list = field(default_factory=list)
    method: str = 'park-martin'

    def to_dict(self) -> dict:
        return {'transform': np.asarray(self.transform).round(8).tolist(),
                'mode': self.mode, 'pairs': self.pairs,
                'rotation_rms_rad': self.rotation_rms,
                'translation_rms_m': self.translation_rms,
                'method': self.method,
                'per_pair': self.per_pair}


def _as_transform(value) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape == (4, 4):
        return array
    if array.shape in ((6,), (7,)):
        return pose_to_matrix(array[:6])
    raise ValueError(f'expected a 4x4 transform or a 6-vector pose, got {array.shape}')


def _relative_pairs(base_tcp, cam_target):
    """A_i = inv(T_base_tcp_i) T_base_tcp_j and B_i = T_cam_target_i inv(T_cam_target_j)."""
    base = [_as_transform(item) for item in base_tcp]
    camera = [_as_transform(item) for item in cam_target]
    if len(base) != len(camera):
        raise CalibrationError('base_tcp and cam_target must have the same length')
    if len(base) < 3:
        raise CalibrationError('at least three pose pairs are required')
    a_list, b_list = [], []
    for i in range(len(base)):
        for j in range(len(base)):
            if i == j:
                continue
            a_list.append(invert_transform(base[i]) @ base[j])
            b_list.append(camera[i] @ invert_transform(camera[j]))
    return a_list, b_list


def _eye_to_hand_pairs(base_tcp, cam_target):
    base = [_as_transform(item) for item in base_tcp]
    camera = [_as_transform(item) for item in cam_target]
    if len(base) != len(camera):
        raise CalibrationError('base_tcp and cam_target must have the same length')
    if len(base) < 3:
        raise CalibrationError('at least three pose pairs are required')
    a_list, b_list = [], []
    for i in range(len(base)):
        for j in range(len(base)):
            if i == j:
                continue
            a_list.append(base[i] @ invert_transform(base[j]))
            b_list.append(camera[i] @ invert_transform(camera[j]))
    return a_list, b_list


def solve_ax_xb(a_transforms, b_transforms) -> np.ndarray:
    """Closed-form rotation then least-squares translation for A X = X B."""
    alpha = np.array([matrix_to_rotvec(item[:3, :3]) for item in a_transforms])
    beta = np.array([matrix_to_rotvec(item[:3, :3]) for item in b_transforms])
    if np.linalg.matrix_rank(alpha) < 2 or np.linalg.matrix_rank(beta) < 2:
        raise CalibrationError('insufficient rotation diversity: move the wrist through '
                               'at least two non-parallel rotations')
    scatter = alpha.T @ beta
    u, _, vt = np.linalg.svd(scatter)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u = u.copy()
        u[:, -1] *= -1
        rotation = u @ vt
    # (R_A - I) t_X = R_X t_B - t_A, stacked over every pair.
    rows, rhs = [], []
    for a_matrix, b_matrix in zip(a_transforms, b_transforms):
        rows.append(a_matrix[:3, :3] - np.eye(3))
        rhs.append(rotation @ b_matrix[:3, 3] - a_matrix[:3, 3])
    solution, *_ = np.linalg.lstsq(np.vstack(rows), np.concatenate(rhs), rcond=None)
    transform = np.eye(4)
    transform[:3, :3] = orthonormalize(rotation)
    transform[:3, 3] = solution
    return transform


def solve_hand_eye(mode: str, base_tcp, cam_target) -> HandEyeResult:
    if mode not in MODES:
        raise CalibrationError(f'mode must be one of {MODES}')
    if mode == 'eye_in_hand':
        a_list, b_list = _relative_pairs(base_tcp, cam_target)
    else:
        a_list, b_list = _eye_to_hand_pairs(base_tcp, cam_target)
    transform = solve_ax_xb(a_list, b_list)
    rotations, translations = [], []
    per_pair = []
    for index, (a_matrix, b_matrix) in enumerate(zip(a_list, b_list)):
        left = a_matrix @ transform
        right = transform @ b_matrix
        relative = invert_transform(left) @ right
        rotation_error = float(np.linalg.norm(matrix_to_rotvec(relative[:3, :3])))
        translation_error = float(np.linalg.norm(relative[:3, 3]))
        rotations.append(rotation_error)
        translations.append(translation_error)
        if index < 64:
            per_pair.append({'rotation_rad': round(rotation_error, 8),
                             'translation_m': round(translation_error, 8)})
    return HandEyeResult(
        transform=transform, mode=mode, pairs=len(a_list),
        rotation_rms=float(np.sqrt(np.mean(np.square(rotations)))),
        translation_rms=float(np.sqrt(np.mean(np.square(translations)))),
        per_pair=per_pair)


def camera_pose_from_robot(result: HandEyeResult, base_tcp) -> np.ndarray:
    """World pose of an eye-in-hand camera for a given robot TCP pose."""
    if result.mode != 'eye_in_hand':
        return result.transform.copy()
    return _as_transform(base_tcp) @ result.transform


def target_pose(result: HandEyeResult, base_tcp=None) -> np.ndarray:
    """Convenience: ``T_base_camera`` regardless of mounting."""
    return camera_pose_from_robot(result, base_tcp)


def pose_pairs_from_observations(base_poses, camera_poses) -> tuple[list, list]:
    """Normalize two lists of 6-vectors into 4x4 transforms."""
    return ([pose_to_matrix(pose) for pose in base_poses],
            [pose_to_matrix(pose) for pose in camera_poses])


def synthetic_pair_error(result: HandEyeResult) -> float:
    return float(max(result.rotation_rms, result.translation_rms))
