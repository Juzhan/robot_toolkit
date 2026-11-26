"""Task options and the parsed task configuration."""
from __future__ import annotations

import argparse
import json
import os
import queue
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..agent.options import AgentOptions
from ..calibrate.options import CalibrateOptions
from ..collect import (CameraSpec, DatasetSchema, EpisodeCollector, collect_trajectory,
                       exporter_for)
from ..collect.passive import sample_episodes
from ..config import build_dataclass
from ..core.safety_envelope import SafetyEnvelope
from ..errors import DatasetError
from ..profiles import build_camera_hub, build_gripper, build_robot
from ..teach import (Replayer, KeyReader, joycon_teach, joint_motion, keyboard_teach,
                     load_trajectory, save_trajectory, state_line, summarize)
from ..teleop import ButtonHub

from .constants import DEFAULT_ROOT


@dataclass
class TeachOptions:
    rate: float = 50.0
    max_duration: float = 300.0
    allow_static: bool = False
    freedrive: bool = True
    wait_feedback: str = ''
    gripper_close_time_s: float = 0.8
    gripper_open_time_s: float = 0.5
    gripper_ramp_hz: float = 50.0


@dataclass
class ReplayOptions:
    rate: float = 50.0
    speed_scale: float = 0.3
    smoothing: int = 9
    allow_fast: bool = False
    realtime: bool = True
    # Use the Mac-proven jerk-limited reference and RTDE period clock.
    continuous: bool = False
    max_joint_accel: float = 1.2
    max_joint_jerk: float = 8.0
    use_sdk_period: bool = True


@dataclass
class CollectOptions:
    trajectory_root: str = ''
    dataset_root: str = ''
    repo_id: str = ''
    fps: float = 10.0
    export: str = 'lerobot'
    cameras: list = field(default_factory=list)
    # Per-episode frame cap; 0 (default) means no cap.  Set it explicitly when a
    # bounded episode is wanted, e.g. 18000 frames at 30 Hz = 10 minutes.
    max_frames: int = 0
    # Per-dimension labels: 'unit' (joint_1.rad, piper_gripper.width_m) keeps
    # explicit units; 'pos' follows LeRobot's motor naming (joint_1.pos ...
    # gripper.pos).  Labels only, values and column order never change.
    feature_names: str = 'unit'
    # Compress the spooled .npz frames.  Off (default) because deflate costs
    # ~50 ms per 640x480 RGB+depth frame and cannot keep up with a 30 Hz camera;
    # turn it on only when disk space matters more than sustained frame rate.
    compress: bool = False
    # One LeRobot dataset per task that every session appends to; empty means
    # <task dir>/lerobot. Raw takes stay under dataset_root/<session>.
    lerobot_root: str = ''
    # Export in a low-priority child process so encoding cannot stall or crash capture.
    isolated_export: bool = True
    # Record robot state/action alongside cameras.  False records cameras only
    # (human demonstrations) using the same spool/LeRobot export pipeline.
    record_state_action: bool = True


@dataclass
class SessionOptions:
    """Persistent Joy-Con teach/replay loop options."""

    prefix: str = ''
    confirm_replay: bool = True
    double_press_timeout_s: float = 8.0
    replay_speed_scale: float | None = None
    auto_replay: bool = False
    voice: bool = False


@dataclass
class MasterSlaveOptions:
    """Passive sampling of a leader/follower pair that teaches itself.

    Nothing is commanded: the follower state is recorded as observation and the
    leader's broadcast joints as action, so the pair keeps working normally.
    """

    enabled: bool = True
    # Recording starts as soon as the link is ready, like the UR5 collect flow;
    # the operator stops the episode (space) instead of starting it.  No time or
    # frame limit is applied unless sample_seconds / collect.max_frames are set.
    auto_record: bool = True
    sample_seconds: float = 0.0
    fps: float = 30.0
    # Button session (Joy-Con and/or keyboard): B start, A save, Y×2 discard.
    # sample_seconds > 0 keeps the hands-free single-episode loop instead.
    interactive: bool = True
    joycon: bool = True
    voice: bool = False
    leader_timeout_s: float = 0.3


@dataclass
class HandDemoOptions:
    """Video-only human demonstration recording.

    No robot state or action is written: an episode is one video per camera,
    plus per-frame timestamps.  The robot / gripper profiles are unused in this
    mode; cameras come from ``collect.cameras``.  ``tracker`` is reserved for a
    future hand-pose channel and stays empty today.
    """

    # 0 means "use collect.fps"; set a positive value only for a
    # hand-demo-specific override.
    fps: float = 0.0
    # Like session mode: wait for B/space instead of recording on startup.
    auto_record: bool = False
    sample_seconds: float = 0.0
    tracker: str = ''
    # 'keyboard' or 'joycon'; Joy-Con maps B/A to start/save.
    input: str = 'keyboard'
    # Reuse collect.speech (ported from ur5e_real_collect) for clip feedback.
    voice: bool = True


@dataclass
class CalibrateOptions:
    """Board-based calibration driven from the task file.

    ``kind``: ``handeye`` (robot/board pose pairs), ``intrinsics`` (board views
    only) or ``both``.  Hand-eye needs the robot's TCP; intrinsics only needs
    the camera, but the robot profile is still used when present.
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
    # Automated scanning: the host drives the arm through the pose list, checks
    # the board at every stop and solves at the end.  Only for a single,
    # host-controlled arm (not for a hardware-linked leader/follower pair) and
    # requires --yes because it moves the robot.
    auto: bool = False
    poses: list = field(default_factory=list)
    pose_count: int = 10
    pose_delta_rad: float = 0.25
    pose_seed: int = 0
    settle_s: float = 0.8
    speed_scale: float = 0.3
    safety_path: str = ''
    require_safety_envelope: bool = True
    require_initial_board: bool = True
    path_clearance_m: float = 0.01
    quality_protocol: bool = False
    action: str = 'solve'
    frames_per_pose: int = 6
    verify_samples: int = 8
    verify_after: bool = False
    solution_path: str = ''
    intrinsics_mode: str = 'auto'
    pose_plan_path: str = ''


@dataclass
class SafetyOptions:
    """Human-guided safety envelope capture for any RobotBackend."""

    path: str = ''
    rate: float = 25.0
    max_duration: float = 300.0
    margin_rad: float = 0.05
    tcp_margin_m: float = 0.02
    min_samples: int = 100
    freedrive: bool = True
    wait_feedback: str = ''


@dataclass
class TaskConfig:
    slug: str = ''
    task: str = ''
    mode: str = 'teach_collect'
    name: str = ''
    root: str = DEFAULT_ROOT
    # ``robot`` and ``gripper`` are the same plain profile dicts accepted by
    # profiles.build_robot/build_gripper.  Nothing here names a robot.
    robot: dict = field(default_factory=lambda: {'kind': 'mock', 'name': 'mock'})
    gripper: dict = field(default_factory=lambda: {'kind': 'mock'})
    teach: TeachOptions = field(default_factory=TeachOptions)
    replay: ReplayOptions = field(default_factory=ReplayOptions)
    collect: CollectOptions = field(default_factory=CollectOptions)
    session: SessionOptions = field(default_factory=SessionOptions)
    master_slave: MasterSlaveOptions = field(default_factory=MasterSlaveOptions)
    hand_demo: HandDemoOptions = field(default_factory=HandDemoOptions)
    calibrate: CalibrateOptions = field(default_factory=CalibrateOptions)
    safety: SafetyOptions = field(default_factory=SafetyOptions)
    agent: AgentOptions = field(default_factory=AgentOptions)
    source_path: str = ''


# --------------------------------------------------------------------- config
