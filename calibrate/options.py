"""Calibration task options; the JSON ``calibrate`` section maps onto these fields."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CalibrateOptions:
    """Board-based calibration.

    ``kind``: ``handeye`` (robot/board pose pairs), ``intrinsics`` (board views
    only) or ``both``. ``eye`` selects the mounting; capture and execution are
    identical for both, only the solve differs.
    """

    kind: str = 'handeye'
    eye: str = 'eye_in_hand'
    camera: str = ''
    # ``chessboard``: ``pattern`` = inner corners.  ``charuco``: ``squares`` =
    # squares per side of the board in assets/board_charuco_300dpi.pdf.
    board: str = 'chessboard'
    pattern: list = field(default_factory=lambda: [9, 6])
    squares: list = field(default_factory=lambda: [7, 10])
    aruco_dict: str = 'DICT_5X5_100'
    marker_size: float = 0.0
    min_corners: int = 6
    square_size: float = 0.025
    min_samples: int = 8
    path: str = ''
    apply: bool = False
    # Unattended capture (calibrate/auto.py). Moves the robot; needs --yes.
    # Single host-controlled arm only, not a hardware-linked leader/follower pair.
    auto: bool = False
    poses: list = field(default_factory=list)
    pose_count: int = 10
    pose_delta_rad: float = 0.25
    pose_seed: int = 0
    pose_plan_path: str = ''
    settle_s: float = 0.8
    speed_scale: float = 0.3
    safety_path: str = ''
    require_safety_envelope: bool = True
    require_initial_board: bool = True
    path_clearance_m: float = 0.01
    # Rigidly held object (board in the gripper): every point within this radius
    # of the TCP stays above its starting height minus ``held_object_drop_m``.
    held_object_radius_m: float = 0.0
    held_object_drop_m: float = 0.0
    # Joints where the drop allowance was given; empty means the run's start pose.
    held_object_reference_q: list = field(default_factory=list)
    # Clamp the gripper (N) after enabling and stop on width change. 0 disables.
    clamp_force_n: float = 0.0
    clamp_slip_m: float = 0.003
    # Capture only once board corners move less than this between frames (px).
    board_jitter_px: float = 0.0
    capture_timeout_s: float = 5.0
    quality_protocol: bool = False
    action: str = 'solve'
    frames_per_pose: int = 6
    verify_samples: int = 8
    verify_after: bool = False
    solution_path: str = ''
    intrinsics_mode: str = 'auto'
