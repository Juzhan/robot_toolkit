"""Board/pose collection session for intrinsics and hand-eye calibration.

The numerical solvers and the JSON store are already robot-independent.  This
module adds the missing application layer:

* read a camera frame and detect a board;
* pair ``T_cam_board`` with the robot's current ``T_base_tcp``;
* solve intrinsics and/or hand-eye;
* persist the result through :class:`CalibrationStore`.

The robot pose comes from the shared ``state().tcp`` contract (rotation vector),
so the same session works for UR5e and Piper X.  ``ChessboardDetector`` handles a
plain checkerboard; ``CharucoDetector`` handles the ChArUco board shipped in
``assets/`` (more robust: partial views and per-marker IDs).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..core.mathutils import pose_to_matrix
from ..errors import CalibrationError
from .handeye import HandEyeResult, solve_hand_eye
from .intrinsics import IntrinsicsResult, estimate_intrinsics
from .store import CalibrationStore


@dataclass
class BoardObservation:
    object_points: np.ndarray
    image_points: np.ndarray
    T_cam_board: np.ndarray | None = None
    image_size: tuple[int, int] = (0, 0)

    @property
    def view(self) -> dict:
        return {'object_points': self.object_points.copy(),
                'image_points': self.image_points.copy()}


class ChessboardDetector:
    """OpenCV chessboard detector with optional PnP pose.

    ``pattern`` is the number of inner corners ``(columns, rows)``.
    """

    def __init__(self, pattern=(9, 6), square_size: float = 0.025):
        self.pattern = (int(pattern[0]), int(pattern[1]))
        self.square_size = float(square_size)

    def object_grid(self) -> np.ndarray:
        columns, rows = self.pattern
        grid = np.zeros((columns * rows, 3), dtype=np.float64)
        grid[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2) * self.square_size
        return grid

    def detect(self, image, K=None, dist=None) -> BoardObservation | None:
        import cv2

        image = np.asarray(image)
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        found, corners = cv2.findChessboardCorners(gray, self.pattern, None)
        if not found:
            return None
        corners = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        image_points = corners.reshape(-1, 2).astype(np.float64)
        object_points = self.object_grid()
        transform = None
        if K is not None:
            K = np.asarray(K, dtype=np.float64).reshape(3, 3)
            dist = np.zeros(5) if dist is None else np.asarray(dist, dtype=np.float64).reshape(-1)
            ok, rvec, tvec = cv2.solvePnP(object_points, image_points, K, dist,
                                          flags=cv2.SOLVEPNP_ITERATIVE)
            if ok:
                rotation = cv2.Rodrigues(rvec)[0]
                transform = np.eye(4)
                transform[:3, :3] = rotation
                transform[:3, 3] = np.asarray(tvec).reshape(3)
        return BoardObservation(object_points=object_points, image_points=image_points,
                                T_cam_board=transform,
                                image_size=(gray.shape[1], gray.shape[0]))


class CharucoDetector:
    """ChArUco detector for the board shipped in ``assets/``.

    Defaults match ``board_charuco_300dpi.pdf``: 7x10 squares, 35 markers of the
    5x5 dictionary.  ``square_size`` is what you measure on the printed board —
    300 dpi print scaling makes it differ from the nominal value.
    """

    def __init__(self, squares=(7, 10), square_size: float = 0.025,
                 marker_size: float = 0.0, aruco_dict: str = 'DICT_5X5_100',
                 min_corners: int = 6):
        self.squares = (int(squares[0]), int(squares[1]))
        self.square_size = float(square_size)
        self.marker_size = float(marker_size) if float(marker_size) > 0 else self.square_size * 0.75
        self.aruco_dict = str(aruco_dict)
        self.min_corners = max(4, int(min_corners))
        self._board = None
        self._detector = None

    def _ensure(self):
        if self._detector is not None:
            return
        import cv2
        dictionary_id = getattr(cv2.aruco, self.aruco_dict, None)
        if dictionary_id is None:
            raise CalibrationError(f'unknown ArUco dictionary {self.aruco_dict!r}')
        dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        self._board = cv2.aruco.CharucoBoard(self.squares, self.square_size,
                                             self.marker_size, dictionary)
        self._detector = cv2.aruco.CharucoDetector(self._board)

    def object_grid(self) -> np.ndarray:
        self._ensure()
        return np.asarray(self._board.getChessboardCorners(), dtype=np.float64)

    def detect(self, image, K=None, dist=None) -> BoardObservation | None:
        import cv2

        self._ensure()
        image = np.asarray(image)
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        corners, ids, _, _ = self._detector.detectBoard(gray)
        if corners is None or ids is None or len(ids) < self.min_corners:
            return None
        index = ids.reshape(-1).astype(int)
        image_points = corners.reshape(-1, 2).astype(np.float64)
        object_points = self.object_grid()[index]
        transform = None
        if K is not None:
            K = np.asarray(K, dtype=np.float64).reshape(3, 3)
            dist = np.zeros(5) if dist is None else np.asarray(dist, dtype=np.float64).reshape(-1)
            ok, rvec, tvec = cv2.solvePnP(object_points, image_points, K, dist,
                                          flags=cv2.SOLVEPNP_ITERATIVE)
            if ok:
                transform = np.eye(4)
                transform[:3, :3] = cv2.Rodrigues(rvec)[0]
                transform[:3, 3] = np.asarray(tvec).reshape(3)
        return BoardObservation(object_points=object_points, image_points=image_points,
                                T_cam_board=transform,
                                image_size=(gray.shape[1], gray.shape[0]))


class CalibrationSession:
    """Collect paired robot/board observations and solve hand-eye calibration."""

    def __init__(self, robot=None, camera=None, detector=None,
                 mode: str = 'eye_in_hand', camera_name: str | None = None):
        if mode not in ('eye_in_hand', 'eye_to_hand'):
            raise CalibrationError("mode must be 'eye_in_hand' or 'eye_to_hand'")
        self.robot = robot
        self.camera = camera
        self.detector = detector
        self.mode = mode
        self.camera_name = camera_name or getattr(camera, 'name', 'camera')
        self.base_tcp: list[np.ndarray] = []
        self.cam_target: list[np.ndarray] = []
        self.views: list[dict] = []
        self.images: list[np.ndarray] = []
        self.last_observation: BoardObservation | None = None

    def add_sample(self, base_tcp, cam_target, view: dict | None = None,
                   image=None) -> None:
        """Add one known pair (useful for tests or an external detector)."""
        base = np.asarray(base_tcp, dtype=float).reshape(6)
        target = np.asarray(cam_target, dtype=float)
        if target.shape == (6,):
            target = pose_to_matrix(target)
        elif target.shape != (4, 4):
            raise CalibrationError(f'cam_target must be 6 or 4x4, got {target.shape}')
        self.base_tcp.append(base.copy())
        self.cam_target.append(target.copy())
        if view is not None:
            self.views.append(dict(view))
        if image is not None:
            self.images.append(np.asarray(image).copy())

    def capture(self, image=None, base_tcp=None) -> BoardObservation | None:
        """Read one robot pose + camera frame and detect the board."""
        if base_tcp is None:
            if self.robot is None:
                raise CalibrationError('capture needs a robot or an explicit base_tcp')
            state = self.robot.state()
            if state.tcp is None:
                raise CalibrationError('robot state has no tcp pose')
            base_tcp = state.tcp
        if image is None:
            if self.camera is None:
                raise CalibrationError('capture needs a camera or an explicit image')
            frame = self.camera.snapshot()
            image = frame.rgb
        if image is None:
            raise CalibrationError('camera returned no RGB frame')
        if self.detector is None:
            raise CalibrationError('capture needs a board detector')
        K = getattr(self.camera, 'intrinsics', None) if self.camera is not None else None
        dist = getattr(self.camera, 'distortion', None) if self.camera is not None else None
        observation = self.detector.detect(image, K=K, dist=dist)
        if observation is None:
            return None
        if observation.T_cam_board is None:
            raise CalibrationError('board detected but PnP failed; check camera intrinsics')
        self.last_observation = observation
        self.add_sample(base_tcp, observation.T_cam_board,
                        view=observation.view, image=image)
        return observation

    def solve_hand_eye(self) -> HandEyeResult:
        if len(self.base_tcp) < 3:
            raise CalibrationError('need at least three robot/board pose pairs')
        return solve_hand_eye(self.mode, self.base_tcp, self.cam_target)

    def solve_intrinsics(self, image_size=None) -> IntrinsicsResult:
        if len(self.views) < 3:
            raise CalibrationError('need at least three board views')
        if image_size is None:
            image_size = (getattr(self.camera, 'width', 0),
                          getattr(self.camera, 'height', 0))
        return estimate_intrinsics(self.views, image_size=image_size or None)

    def save(self, path, camera_name: str | None = None, apply: bool = False) -> dict:
        """Solve available data, write ``CalibrationStore`` and optionally apply it."""
        name = camera_name or self.camera_name
        store = CalibrationStore(path)
        summary = {'camera': name, 'mode': self.mode}
        # Hand-eye PnP used these intrinsics. Do not silently replace them with
        # a separately fitted camera model while keeping the old extrinsics.
        if len(self.base_tcp) >= 3 and self.camera is not None and self.camera.intrinsics is not None:
            store.set_intrinsics(name, self.camera.intrinsics, self.camera.distortion,
                                 (self.camera.width, self.camera.height))
            summary['intrinsics_source'] = 'capture'
        elif len(self.views) >= 3:
            intrinsics = self.solve_intrinsics()
            store.set_intrinsics(name, intrinsics.K, intrinsics.dist,
                                 intrinsics.image_size)
            summary['intrinsics_rms_px'] = intrinsics.rms
        if len(self.base_tcp) >= 3:
            result = self.solve_hand_eye()
            store.set_hand_eye(name, self.mode, result.transform,
                              residual=max(result.rotation_rms, result.translation_rms))
            summary['hand_eye'] = result.to_dict()
        if len(self.views) < 3 and len(self.base_tcp) < 3:
            raise CalibrationError('no enough observations to save calibration')
        store.save()
        summary['path'] = str(Path(path))
        if apply and self.camera is not None:
            base = self.base_tcp[-1] if self.base_tcp else None
            store.apply(self.camera, base_tcp=base, require_extrinsics=True)
            summary['applied'] = True
        return summary

    def summary(self) -> dict:
        return {'mode': self.mode, 'camera': self.camera_name,
                'pairs': len(self.base_tcp), 'views': len(self.views)}
