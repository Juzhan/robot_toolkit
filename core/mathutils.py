"""Pose / transform math used by every layer of the toolkit.

Conventions
-----------
* A *pose* is always a 6-vector ``[x, y, z, rx, ry, rz]`` where the rotation is
  an axis-angle (Rodrigues) vector in radians, expressed in the parent frame.
  This is the same convention as UR ``getActualTCPPose`` and ROS ``Pose``.
* A *transform* is a homogeneous ``4x4`` matrix ``T_parent_child``.
* No dependency on SciPy: only NumPy is required.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-12


def _as_rotation_matrix(matrix) -> np.ndarray:
    array = np.asarray(matrix, dtype=float)
    if array.shape != (3, 3):
        raise ValueError(f'expected a 3x3 rotation matrix, got {array.shape}')
    return array


def _skew(vector) -> np.ndarray:
    x, y, z = np.asarray(vector, dtype=float).reshape(3)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def _vee(skew_matrix) -> np.ndarray:
    return np.array([skew_matrix[2, 1], skew_matrix[0, 2], skew_matrix[1, 0]], dtype=float)


def rotvec_to_matrix(rotvec) -> np.ndarray:
    """Rodrigues formula: axis-angle vector -> rotation matrix."""
    vector = np.asarray(rotvec, dtype=float).reshape(3)
    angle = float(np.linalg.norm(vector))
    if angle < _EPS:
        # First-order expansion, also keeps gradients finite at zero.
        skew = _skew(vector)
        return np.eye(3) + skew + 0.5 * skew @ skew
    axis = vector / angle
    skew = _skew(axis)
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


def matrix_to_rotvec(matrix) -> np.ndarray:
    """Rotation matrix -> axis-angle vector, robust near 0 and pi."""
    rotation = _as_rotation_matrix(matrix)
    trace = float(np.clip(np.trace(rotation), -1.0, 3.0))
    cos_angle = float(np.clip((trace - 1.0) * 0.5, -1.0, 1.0))
    angle = float(np.arccos(cos_angle))
    if angle < 1e-7:
        return 0.5 * _vee(rotation - rotation.T)
    if angle < np.pi - 1e-6:
        axis = _vee(rotation - rotation.T) / (2.0 * np.sin(angle))
        return axis * angle
    # Near pi the antisymmetric part vanishes; recover the axis from R + I.
    symmetric = (rotation + np.eye(3)) * 0.5
    axis = np.sqrt(np.clip(np.diag(symmetric), 0.0, None))
    largest = int(np.argmax(axis))
    if axis[largest] < _EPS:
        # Degenerate (should not happen for a proper rotation).
        return np.zeros(3)
    for index in range(3):
        if index != largest:
            axis[index] = symmetric[largest, index] / axis[largest]
    axis /= np.linalg.norm(axis)
    return axis * angle


def quat_to_matrix(quaternion, order: str = 'xyzw') -> np.ndarray:
    """Unit quaternion -> rotation matrix.

    ``order='xyzw'`` matches SciPy / MuJoCo; ``order='wxyz'`` matches UR and
    many ROS tools.  The quaternion is normalized here, so callers may pass
    raw controller output.
    """
    values = np.asarray(quaternion, dtype=float).reshape(4)
    if order == 'wxyz':
        w, x, y, z = values
    elif order == 'xyzw':
        x, y, z, w = values
    else:
        raise ValueError("order must be 'xyzw' or 'wxyz'")
    norm = float(np.linalg.norm([w, x, y, z]))
    if norm < _EPS:
        raise ValueError('cannot convert a zero quaternion')
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def matrix_to_quat(matrix, order: str = 'xyzw') -> np.ndarray:
    """Rotation matrix -> unit quaternion (Shepperd's method)."""
    rotation = _as_rotation_matrix(matrix)
    trace = float(np.trace(rotation))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (rotation[2, 1] - rotation[1, 2]) / s
        y = (rotation[0, 2] - rotation[2, 0]) / s
        z = (rotation[1, 0] - rotation[0, 1]) / s
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        s = np.sqrt(max(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2], 0.0)) * 2.0
        w = (rotation[2, 1] - rotation[1, 2]) / s if s else 0.0
        x = 0.25 * s
        y = (rotation[0, 1] + rotation[1, 0]) / s if s else 0.0
        z = (rotation[0, 2] + rotation[2, 0]) / s if s else 0.0
    elif rotation[1, 1] > rotation[2, 2]:
        s = np.sqrt(max(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2], 0.0)) * 2.0
        w = (rotation[0, 2] - rotation[2, 0]) / s if s else 0.0
        x = (rotation[0, 1] + rotation[1, 0]) / s if s else 0.0
        y = 0.25 * s
        z = (rotation[1, 2] + rotation[2, 1]) / s if s else 0.0
    else:
        s = np.sqrt(max(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1], 0.0)) * 2.0
        w = (rotation[1, 0] - rotation[0, 1]) / s if s else 0.0
        x = (rotation[0, 2] + rotation[2, 0]) / s if s else 0.0
        y = (rotation[1, 2] + rotation[2, 1]) / s if s else 0.0
        z = 0.25 * s
    quaternion = np.array([x, y, z, w])
    quaternion /= max(float(np.linalg.norm(quaternion)), _EPS)
    return quaternion if order == 'xyzw' else np.array([w, x, y, z])


def euler_to_matrix(angles, order: str = 'xyz', degrees: bool = False) -> np.ndarray:
    """Extrinsic Euler angles -> rotation matrix (SciPy-compatible lowercase orders).

    Supported orders: ``xyz`` (Joy-Con attitude) and ``ZXY`` (phone attitude),
    both interpreted as extrinsic rotations, matching SciPy's lowercase orders.
    """
    values = np.asarray(angles, dtype=float).reshape(3)
    if degrees:
        values = np.radians(values)
    if order in ('xyz', 'XYZ'):
        rx, ry, rz = values
        return rotvec_to_matrix([0, 0, rz]) @ rotvec_to_matrix([0, ry, 0]) @ rotvec_to_matrix([rx, 0, 0])
    if order in ('ZXY', 'zxy'):
        a, b, c = values
        return rotvec_to_matrix([0, 0, a]) @ rotvec_to_matrix([b, 0, 0]) @ rotvec_to_matrix([0, c, 0])
    raise ValueError(f'unsupported Euler order {order!r}; use xyz or ZXY')


def matrix_to_euler(matrix, order: str = 'xyz') -> np.ndarray:
    """Rotation matrix -> extrinsic Euler angles (inverse of :func:`euler_to_matrix`)."""
    rotation = _as_rotation_matrix(matrix)
    if order in ('xyz', 'XYZ'):
        # R = Rz(rz) @ Ry(ry) @ Rx(rx)
        sy = float(np.clip(-rotation[2, 0], -1.0, 1.0))
        pitch = float(np.arcsin(sy))
        cos_pitch = float(np.cos(pitch))
        if abs(cos_pitch) > 1e-8:
            roll = float(np.arctan2(rotation[2, 1], rotation[2, 2]))
            yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
        else:
            # Gimbal lock: yaw is arbitrary; keep roll meaningful.
            roll = float(np.arctan2(-rotation[1, 2], rotation[1, 1]))
            yaw = 0.0
        return np.array([roll, pitch, yaw], dtype=float)
    if order in ('ZXY', 'zxy'):
        # R = Rz(a) @ Rx(b) @ Ry(c)
        sb = float(np.clip(rotation[2, 1], -1.0, 1.0))
        b = float(np.arcsin(sb))
        cb = float(np.cos(b))
        if abs(cb) > 1e-8:
            a = float(np.arctan2(-rotation[0, 1], rotation[1, 1]))
            c = float(np.arctan2(-rotation[2, 0], rotation[2, 2]))
        else:
            a = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
            c = 0.0
        return np.array([a, b, c], dtype=float)
    raise ValueError(f'unsupported Euler order {order!r}; use xyz or ZXY')


def pose_to_matrix(pose) -> np.ndarray:
    """``[x, y, z, rx, ry, rz]`` -> homogeneous transform."""
    values = np.asarray(pose, dtype=float).reshape(6)
    matrix = np.eye(4)
    matrix[:3, :3] = rotvec_to_matrix(values[3:])
    matrix[:3, 3] = values[:3]
    return matrix


def matrix_to_pose(matrix) -> np.ndarray:
    """Homogeneous transform -> ``[x, y, z, rx, ry, rz]``."""
    array = np.asarray(matrix, dtype=float)
    if array.shape != (4, 4):
        raise ValueError(f'expected a 4x4 matrix, got {array.shape}')
    return np.concatenate([array[:3, 3], matrix_to_rotvec(array[:3, :3])])


def invert_transform(matrix) -> np.ndarray:
    array = np.asarray(matrix, dtype=float)
    rotation = array[:3, :3]
    translation = array[:3, 3]
    result = np.eye(4)
    result[:3, :3] = rotation.T
    result[:3, 3] = -rotation.T @ translation
    return result


def compose_transforms(*matrices) -> np.ndarray:
    """Multiply transforms left-to-right: ``compose(A, B) == A @ B``."""
    result = np.eye(4)
    for matrix in matrices:
        result = result @ np.asarray(matrix, dtype=float)
    return result


def transform_points(matrix, points) -> np.ndarray:
    """Apply a transform to an ``(N, 3)`` or ``(3,)`` array of points."""
    array = np.asarray(points, dtype=float)
    single = array.ndim == 1
    array = np.atleast_2d(array)
    matrix = np.asarray(matrix, dtype=float)
    result = array @ matrix[:3, :3].T + matrix[:3, 3]
    return result[0] if single else result


def pose_error(left, right) -> tuple[float, float]:
    """Return ``(translation_error_m, rotation_error_rad)`` between two poses."""
    relative = invert_transform(pose_to_matrix(left)) @ pose_to_matrix(right)
    return float(np.linalg.norm(relative[:3, 3])), float(np.linalg.norm(matrix_to_rotvec(relative[:3, :3])))


def rotation_angle(matrix) -> float:
    rotation = _as_rotation_matrix(matrix)
    return float(np.arccos(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0)))


def average_transforms(matrices) -> np.ndarray:
    """Karcher-style average of transforms without SciPy.

    Translations are averaged directly; rotations use the principal
    eigenvector of the quaternion scatter matrix (Markley's method).
    """
    items = [np.asarray(matrix, dtype=float) for matrix in matrices]
    if not items:
        raise ValueError('average_transforms needs at least one transform')
    translation = np.mean([item[:3, 3] for item in items], axis=0)
    scatter = np.zeros((4, 4))
    for item in items:
        quaternion = matrix_to_quat(item[:3, :3], order='wxyz')
        scatter += np.outer(quaternion, quaternion)
    eigenvalues, eigenvectors = np.linalg.eigh(scatter)
    quaternion = eigenvectors[:, int(np.argmax(eigenvalues))]
    if quaternion[0] < 0:
        quaternion = -quaternion
    result = np.eye(4)
    result[:3, :3] = quat_to_matrix(quaternion, order='wxyz')
    result[:3, 3] = translation
    return result


def orthonormalize(matrix) -> np.ndarray:
    """Project the rotation block back onto SO(3) (used after linear solves)."""
    array = np.array(matrix, dtype=float, copy=True)
    u, _, vt = np.linalg.svd(array[:3, :3])
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    array[:3, :3] = rotation
    return array
