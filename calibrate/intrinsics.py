"""Camera intrinsics: planar calibration with a pure-NumPy Zhang solver.

OpenCV is used to refine the result when it is installed, but it is never
required.  The fallback matters because the toolkit is also used on machines
that only have NumPy/SciPy for data processing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core.mathutils import orthonormalize
from ..errors import CalibrationError


@dataclass
class IntrinsicsResult:
    K: np.ndarray
    dist: np.ndarray
    rms: float
    image_size: tuple
    per_view: list = field(default_factory=list)
    method: str = 'zhang'

    def to_dict(self) -> dict:
        return {'K': np.asarray(self.K).tolist(),
                'dist': np.asarray(self.dist).reshape(-1).tolist(),
                'rms_px': float(self.rms),
                'image_size': list(self.image_size),
                'per_view': self.per_view,
                'method': self.method}


def project_points(object_points, K, dist=None, rotation=None, translation=None) -> np.ndarray:
    """Project ``(N, 3)`` board points through a pinhole + radial-tangential model."""
    points = np.asarray(object_points, dtype=float)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    if points.shape[1] == 2:
        points = np.concatenate([points, np.zeros((len(points), 1))], axis=1)
    if rotation is not None:
        matrix = np.asarray(rotation, dtype=float)
        if matrix.shape == (3,):
            from ..core.mathutils import rotvec_to_matrix
            matrix = rotvec_to_matrix(matrix)
        points = points @ matrix.T
    if translation is not None:
        points = points + np.asarray(translation, dtype=float).reshape(1, 3)
    normalized = points[:, :2] / np.where(np.abs(points[:, 2:3]) < 1e-12, 1e-12, points[:, 2:3])
    x, y = normalized[:, 0], normalized[:, 1]
    coefficients = (np.zeros(5) if dist is None else np.asarray(dist, dtype=float).reshape(-1))
    k1, k2, p1, p2, k3 = (list(coefficients[:5]) + [0.0] * 5)[:5]
    r2 = x * x + y * y
    radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
    x_distorted = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    y_distorted = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    K = np.asarray(K, dtype=float).reshape(3, 3)
    u = K[0, 0] * x_distorted + K[0, 1] * y_distorted + K[0, 2]
    v = K[1, 1] * y_distorted + K[1, 2]
    return np.stack([u, v], axis=1)


def undistort_points(points, K, dist=None) -> np.ndarray:
    """Iterative inverse of the distortion model (five iterations are plenty)."""
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    K = np.asarray(K, dtype=float).reshape(3, 3)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    xd = (points[:, 0] - cx) / fx
    yd = (points[:, 1] - cy) / fy
    coefficients = (np.zeros(5) if dist is None else np.asarray(dist, dtype=float).reshape(-1))
    k1, k2, p1, p2, k3 = (list(coefficients[:5]) + [0.0] * 5)[:5]
    x, y = xd.copy(), yd.copy()
    for _ in range(5):
        r2 = x * x + y * y
        radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
        dx = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
        dy = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
        x = (xd - dx) / radial
        y = (yd - dy) / radial
    return np.stack([x * fx + cx, y * fy + cy], axis=1)


def _normalize_points(points):
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    mean = points.mean(axis=0)
    centered = points - mean
    mean_distance = np.mean(np.linalg.norm(centered, axis=1))
    scale = np.sqrt(2.0) / max(mean_distance, 1e-12)
    transform = np.array([[scale, 0, -scale * mean[0]],
                          [0, scale, -scale * mean[1]],
                          [0, 0, 1.0]])
    return transform @ np.hstack([points, np.ones((len(points), 1))]).T, transform


def planar_homography(object_points, image_points) -> np.ndarray:
    """DLT homography from board XY to image pixels, with normalization."""
    object_points = np.asarray(object_points, dtype=float).reshape(-1, 2)
    image_points = np.asarray(image_points, dtype=float).reshape(-1, 2)
    if len(object_points) != len(image_points) or len(object_points) < 4:
        raise CalibrationError('homography needs at least four matching points')
    object_norm, t_object = _normalize_points(object_points)
    image_norm, t_image = _normalize_points(image_points)
    rows = []
    for index in range(object_norm.shape[1]):
        x, y, w = object_norm[:, index]
        u, v, _ = image_norm[:, index]
        rows.append([-x, -y, -w, 0, 0, 0, u * x, u * y, u * w])
        rows.append([0, 0, 0, -x, -y, -w, v * x, v * y, v * w])
    _, _, vt = np.linalg.svd(np.asarray(rows))
    homography = vt[-1].reshape(3, 3)
    homography = np.linalg.inv(t_image) @ homography @ t_object
    return homography / homography[2, 2]


def _v_column(homography, i, j):
    return np.array([
        homography[0, i] * homography[0, j],
        homography[0, i] * homography[1, j] + homography[1, i] * homography[0, j],
        homography[1, i] * homography[1, j],
        homography[2, i] * homography[0, j] + homography[0, i] * homography[2, j],
        homography[2, i] * homography[1, j] + homography[1, i] * homography[2, j],
        homography[2, i] * homography[2, j],
    ])


def _intrinsics_from_b(b):
    b11, b12, b22, b13, b23, b33 = b
    denominator = b11 * b22 - b12 * b12
    if abs(denominator) < 1e-15:
        raise CalibrationError('degenerate Zhang solution (singular B matrix)')
    v0 = (b12 * b13 - b11 * b23) / denominator
    lam = b33 - (b13 * b13 + v0 * (b12 * b13 - b11 * b23)) / b11
    if lam <= 0 or b11 <= 0:
        raise CalibrationError('Zhang solution violates the pinhole constraints')
    alpha = np.sqrt(lam / b11)
    beta = np.sqrt(lam * b11 / denominator)
    gamma = -b12 * alpha * alpha * beta / lam
    u0 = gamma * v0 / beta - b13 * alpha * alpha / lam
    return np.array([[alpha, gamma, u0], [0.0, beta, v0], [0.0, 0.0, 1.0]])


def _view_extrinsics(homography, K):
    inverse = np.linalg.inv(K)
    h1, h2, h3 = homography[:, 0], homography[:, 1], homography[:, 2]
    lambda1 = 1.0 / max(np.linalg.norm(inverse @ h1), 1e-12)
    lambda2 = 1.0 / max(np.linalg.norm(inverse @ h2), 1e-12)
    scale = float(np.sqrt(lambda1 * lambda2))
    r1 = scale * inverse @ h1
    r2 = scale * inverse @ h2
    translation = scale * inverse @ h3
    if translation[2] < 0:
        # The board must be in front of the camera.
        r1, r2, translation = -r1, -r2, -translation
    r3 = np.cross(r1, r2)
    rotation = orthonormalize(np.column_stack([r1, r2, r3]))
    return rotation, translation


def estimate_intrinsics(views, image_size=None, refinement: int = 1) -> IntrinsicsResult:
    """Zhang's planar calibration.

    ``views`` is a sequence of dicts with ``object_points`` (Nx3 with Z=0, or
    Nx2) and ``image_points`` (Nx2).  Distortion is left at zero unless OpenCV
    is available for the optional refinement step.
    """
    if len(views) < 3:
        raise CalibrationError('planar calibration needs at least three views')
    homographies = []
    for view in views:
        object_points = np.asarray(view['object_points'], dtype=float)
        object_xy = object_points[:, :2]
        homographies.append(planar_homography(object_xy, view['image_points']))
    rows = []
    for homography in homographies:
        rows.append(_v_column(homography, 0, 1))
        rows.append(_v_column(homography, 0, 0) - _v_column(homography, 1, 1))
    _, _, vt = np.linalg.svd(np.asarray(rows))
    b = vt[-1]
    if b[0] < 0:
        b = -b
    K = _intrinsics_from_b(b)
    if image_size is not None:
        # Zhang is scale-ambiguous about image axes; trust the configured size
        # for the principal point sign when the estimate lands outside.
        width, height = image_size
        if not (0 <= K[0, 2] <= width) or not (0 <= K[1, 2] <= height):
            raise CalibrationError('estimated principal point is outside the image')
    per_view = []
    squared = []
    for view, homography in zip(views, homographies):
        rotation, translation = _view_extrinsics(homography, K)
        projected = project_points(view['object_points'], K, None, rotation, translation)
        observed = np.asarray(view['image_points'], dtype=float).reshape(-1, 2)
        error = np.linalg.norm(projected - observed, axis=1)
        squared.extend((error ** 2).tolist())
        per_view.append({'rms_px': float(np.sqrt(np.mean(error ** 2))),
                         'rotation': rotation.tolist(),
                         'translation': translation.tolist()})
    result = IntrinsicsResult(K=K, dist=np.zeros(5),
                              rms=float(np.sqrt(np.mean(squared))) if squared else 0.0,
                              image_size=tuple(image_size) if image_size else (0, 0),
                              per_view=per_view, method='zhang')
    if refinement:
        refined = _refine_with_cv2(views, result)
        if refined is not None:
            return refined
    return result


def _refine_with_cv2(views, result: IntrinsicsResult):
    try:
        import cv2
    except ImportError:
        return None
    object_points, image_points = [], []
    for view in views:
        points = np.asarray(view['object_points'], dtype=np.float32)
        if points.shape[1] == 2:
            points = np.concatenate([points, np.zeros((len(points), 1), np.float32)], axis=1)
        object_points.append(points.reshape(-1, 3))
        image_points.append(np.asarray(view['image_points'], dtype=np.float32).reshape(-1, 2))
    width, height = result.image_size
    try:
        rms, K, dist, _, _ = cv2.calibrateCamera(
            object_points, image_points, (int(width), int(height)),
            result.K.astype(np.float64), np.zeros(5),
            flags=cv2.CALIB_USE_INTRINSIC_GUESS)
    except Exception:
        return None
    return IntrinsicsResult(K=K, dist=np.asarray(dist, dtype=float).reshape(-1),
                            rms=float(rms), image_size=result.image_size,
                            per_view=result.per_view, method='zhang+cv2')


def calibrate_checkerboard(images, pattern=(9, 6), square_size: float = 0.025,
                           image_size=None, refinement: int = 1):
    """Detect a chessboard in each image and solve intrinsics (requires OpenCV)."""
    import cv2

    columns, rows = int(pattern[0]), int(pattern[1])
    grid = np.zeros((columns * rows, 3), np.float32)
    grid[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2) * float(square_size)
    views = []
    for image in images:
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        found, corners = cv2.findChessboardCorners(gray, (columns, rows), None)
        if not found:
            continue
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                   (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        views.append({'object_points': grid.reshape(-1, 3),
                      'image_points': corners.reshape(-1, 2)})
    if len(views) < 3:
        raise CalibrationError(f'found the board in only {len(views)} images')
    return estimate_intrinsics(views, image_size=image_size, refinement=refinement)


def reprojection_rms(views, K, dist=None, extrinsics=None) -> float:
    squared = []
    for index, view in enumerate(views):
        if extrinsics is None:
            rotation, translation = None, None
        else:
            rotation, translation = extrinsics[index]
        projected = project_points(view['object_points'], K, dist, rotation, translation)
        observed = np.asarray(view['image_points'], dtype=float).reshape(-1, 2)
        squared.extend((np.linalg.norm(projected - observed, axis=1) ** 2).tolist())
    return float(np.sqrt(np.mean(squared))) if squared else 0.0
