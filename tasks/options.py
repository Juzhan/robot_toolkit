"""Task options and the parsed task configuration."""

from __future__ import annotations
from dataclasses import dataclass, field
from ..agent.options import AgentOptions
from ..calibrate.options import CalibrateOptions
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
    # Speak the clip start/save/end cues through collect.speech.
    voice: bool = True

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
