"""Config-driven robot task runner (teach / replay / collect).

One JSON file describes a task and the robot profile.  The same entry point
drives Piper X, UR5e, MuJoCo or mock because it only uses
``profiles.build_robot`` / ``build_gripper`` and the shared backend contracts::

    run.bat --list
    run.bat stack_the_cube                 # mode from the JSON
    run.bat stack_the_cube teach
    run.bat stack_the_cube collect --yes
    run.bat stack_the_cube --dry-run

The default config directory is ``configs/tasks``.  A path passed as the first
argument is used directly when it points to an existing ``.json`` file.
"""
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

from .collect import (CameraSpec, DatasetSchema, EpisodeCollector, collect_trajectory,
                      exporter_for)
from .collect.passive import sample_episodes
from .config import build_dataclass
from .profiles import build_camera_hub, build_gripper, build_robot
from .teach import (Replayer, joycon_teach, joint_motion, keyboard_teach,
                    load_trajectory, save_trajectory, state_line, summarize)
from .teleop import ButtonHub

# In the host workspace the package lives at <root>/projects/robot_toolkit,
# while a standalone clone has this file at <repo>/robot_toolkit/task_runner.py.
_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
ROOT = (Path(__file__).resolve().parents[2]
        if (Path(__file__).resolve().parents[2] / 'configs').exists()
        else _PACKAGE_ROOT)
DEFAULT_ROOT = 'outputs/robot_tasks'
MODES = ('teach', 'replay', 'collect', 'teach_collect', 'export', 'session',
         'master_slave', 'calibrate')


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


@dataclass
class SessionOptions:
    """Persistent Joy-Con teach/replay loop options."""

    prefix: str = ''
    confirm_replay: bool = True
    double_press_timeout_s: float = 8.0
    replay_speed_scale: float | None = None
    auto_replay: bool = False


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
    calibrate: CalibrateOptions = field(default_factory=CalibrateOptions)
    source_path: str = ''


# --------------------------------------------------------------------- config
def _slug(text: str) -> str:
    value = re.sub(r'[^a-z0-9]+', '_', str(text or '').strip().lower()).strip('_')
    return value or 'default'


def config_dir() -> Path:
    """Primary task-config directory (first entry of :func:`config_dirs`)."""
    return config_dirs()[0]


def config_dirs() -> list[Path]:
    """Task-config search path, most specific first.

    ``ROBOT_TASK_CONFIG_DIR`` replaces the list entirely; otherwise the workspace
    ``configs/tasks`` wins over the demo configs shipped inside the package.
    """
    override = os.environ.get('ROBOT_TASK_CONFIG_DIR')
    if override:
        return [Path(override)]
    # Demo configs ship next to the package code, so a fresh clone can run
    # ``run.bat piper_master_slave`` without copying anything.
    code_root = Path(__file__).resolve().parent
    dirs = [ROOT / 'configs' / 'tasks', code_root / 'configs' / 'tasks']
    unique = []
    for path in dirs:
        if path not in unique:
            unique.append(path)
    return unique


def discover_configs() -> list[Path]:
    found: dict[str, Path] = {}
    for directory in config_dirs():
        if not directory.exists():
            continue
        for path in sorted(directory.glob('*.json')):
            found.setdefault(path.stem, path)
    return [found[key] for key in sorted(found)]


def resolve_config_path(name: str) -> Path:
    direct = Path(name).expanduser()
    if direct.is_file():
        return direct
    candidates = []
    for directory in config_dirs():
        candidates.append(directory / name)
        if not name.lower().endswith('.json'):
            candidates.append(directory / f'{name}.json')
    root = ROOT / DEFAULT_ROOT / 'tasks'
    candidates.extend([root / name / 'task.json',
                       root / name / 'config.json'])
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    available = ', '.join(path.stem for path in discover_configs()) or '(none)'
    raise FileNotFoundError(
        f'找不到任务配置 {name!r}；可用任务：{available}。'
        f'搜索目录：{", ".join(str(path) for path in config_dirs())}')


def _profile(value, what: str) -> dict:
    if isinstance(value, str):
        value = {'kind': value}
    if not isinstance(value, dict):
        raise ValueError(f'{what} 必须是对象或字符串')
    profile = dict(value)
    if not str(profile.get('kind', '')).strip():
        raise ValueError(f'{what}.kind 不能为空')
    profile['kind'] = str(profile['kind']).strip().lower()
    return profile


def load_task_config(name_or_path: str) -> TaskConfig:
    path = resolve_config_path(name_or_path)
    # utf-8-sig: Windows editors/PowerShell often prepend a BOM.
    raw = json.loads(path.read_text(encoding='utf-8-sig'))
    task = build_dataclass(TaskConfig, raw)
    task.source_path = str(path)

    robot = _profile(task.robot if task.robot else {'kind': 'mock', 'name': 'mock'},
                     'robot')
    if robot['kind'] == 'ur5e':
        robot['kind'] = 'ur_rtde'
        robot.setdefault('name', 'ur5e')
    if raw.get('gripper'):
        gripper = _profile(task.gripper, 'gripper')
    else:
        kind = robot['kind']
        default = ('agx' if kind in ('agx_piper', 'piper', 'piper_x', 'piper_h', 'piper_l')
                   else 'robotiq' if kind in ('ur_rtde', 'ur5e') else 'mock')
        gripper = {'kind': default}
    task.robot, task.gripper = robot, gripper

    task.slug = _slug(str(task.slug or '') or str(task.task or '') or path.stem)
    task.task = str(task.task or '').strip() or task.slug.replace('_', ' ')
    task.mode = str(task.mode or 'teach_collect').strip().lower()
    task.name = str(task.name or '')
    task.root = str(task.root or DEFAULT_ROOT)
    if task.mode not in MODES:
        raise ValueError(f'{path}: mode={task.mode!r} 不在 {MODES}')
    if str(task.collect.export).lower() not in ('lerobot', 'manifest'):
        raise ValueError(f'{path}: collect.export 只能是 lerobot 或 manifest')
    if str(task.collect.feature_names).lower() not in ('unit', 'pos'):
        raise ValueError(f'{path}: collect.feature_names 只能是 unit 或 pos')
    task.collect.feature_names = str(task.collect.feature_names).strip().lower()
    if str(task.calibrate.kind).lower() not in ('handeye', 'intrinsics', 'both'):
        raise ValueError(f'{path}: calibrate.kind 只能是 handeye / intrinsics / both')
    if str(task.calibrate.eye).lower() not in ('eye_in_hand', 'eye_to_hand'):
        raise ValueError(f'{path}: calibrate.eye 只能是 eye_in_hand 或 eye_to_hand')
    if str(task.calibrate.board).lower() not in ('chessboard', 'charuco'):
        raise ValueError(f'{path}: calibrate.board 只能是 chessboard 或 charuco')
    task.calibrate.kind = str(task.calibrate.kind).strip().lower()
    task.calibrate.eye = str(task.calibrate.eye).strip().lower()
    task.calibrate.board = str(task.calibrate.board).strip().lower()
    if len(task.calibrate.pattern) != 2 or float(task.calibrate.square_size) <= 0:
        raise ValueError(f'{path}: calibrate.pattern 需要 [cols, rows]，square_size > 0')
    if not isinstance(task.collect.cameras, list):
        raise ValueError(f'{path}: collect.cameras 必须是数组')
    if any(not isinstance(item, dict) for item in task.collect.cameras):
        raise ValueError(f'{path}: collect.cameras 的每一项都必须是对象')
    if float(task.collect.fps) <= 0:
        raise ValueError(f'{path}: collect.fps 必须大于 0')
    if not isinstance(task.collect.compress, bool):
        raise ValueError(f'{path}: collect.compress 必须是 true 或 false')
    if float(task.teach.rate) <= 0 or float(task.replay.rate) <= 0:
        raise ValueError(f'{path}: teach.rate / replay.rate 必须大于 0')
    return task


def print_configs() -> None:
    items = discover_configs()
    for directory in config_dirs():
        print(f'任务配置目录：{directory}')
    if not items:
        print('  (没有 .json 配置文件)')
        return
    for path in items:
        try:
            task = load_task_config(str(path))
        except Exception as exc:
            print(f'  {path.stem:24s}  读取失败：{exc}')
            continue
        print(f'  {path.stem:24s}  mode={task.mode:13s}  '
              f'robot={task.robot.get("kind", ""):12s}  task={task.task}')


# ----------------------------------------------------------------------- paths
def _path(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def task_dir(task: TaskConfig) -> Path:
    return _path(task.root) / task.slug


def trajectory_root(task: TaskConfig, override: str = '') -> Path:
    value = (override or task.collect.trajectory_root or '').strip()
    return _path(value) if value else task_dir(task) / 'trajectories'


def dataset_root(task: TaskConfig, override: str = '') -> Path:
    value = (override or task.collect.dataset_root or '').strip()
    return _path(value) if value else task_dir(task) / 'dataset'


def latest_trajectory(root: Path) -> str | None:
    items = list(root.glob('*/trajectory.json')) if root.exists() else []
    if not items:
        return None
    return max(items, key=lambda path: path.stat().st_mtime).parent.name


def _next_name(task: TaskConfig, root: Path) -> str:
    base = task.slug or 'task'
    for index in range(1, 1000):
        name = f'{base}_{index:02d}'
        if not (root / name).exists():
            return name
    raise RuntimeError(f'{root} 下同名轨迹过多')


def _collect_name(task: TaskConfig, root: Path, explicit: str | None) -> str:
    if explicit:
        if not (root / explicit).exists():
            raise FileNotFoundError(f'轨迹不存在：{root / explicit}')
        return explicit
    if task.name and (root / task.name).exists():
        return task.name
    latest = latest_trajectory(root)
    if latest is None:
        raise FileNotFoundError(f'{root} 下没有示教轨迹，请先运行 teach')
    return latest


# ------------------------------------------------------------------- workflows
def _robot_speed(profile: dict) -> float:
    options = profile.get('options', {}) or {}
    limits = profile.get('limits', {}) or {}
    return float(limits.get('max_velocity', options.get('max_joint_speed', 1.0)))


def _build_replayer(task: TaskConfig, robot, gripper, capture_fps: float = 0.0):
    return Replayer(
        robot, gripper, rate=float(task.replay.rate),
        max_joint_speed=_robot_speed(task.robot),
        close_settle_s=float(task.teach.gripper_close_time_s),
        open_settle_s=float(task.teach.gripper_open_time_s),
        smoothing=int(task.replay.smoothing),
        capture_fps=float(capture_fps),
        continuous=bool(task.replay.continuous),
        max_joint_accel=float(task.replay.max_joint_accel),
        max_joint_jerk=float(task.replay.max_joint_jerk),
        use_sdk_period=bool(task.replay.use_sdk_period),
    )


def _open_gripper(gripper) -> None:
    if gripper is None:
        return
    gripper.connect()
    starter = getattr(gripper, 'start_polling', None)
    if callable(starter):
        starter()


def _close_gripper(gripper) -> None:
    if gripper is None:
        return
    try:
        gripper.stop()
    except Exception:
        pass
    try:
        gripper.disconnect()
    except Exception:
        pass


def _load_trajectory_checked(task: TaskConfig, name: str):
    """Load a trajectory and reject nearly-static recordings."""
    directory = trajectory_root(task) / name
    try:
        trajectory = load_trajectory(directory)
    except Exception as exc:
        print(f'读不到轨迹 {directory}：{exc}')
        return None, 1
    motion = joint_motion(trajectory)
    if motion < 1e-4 and not task.teach.allow_static:
        print(f'轨迹关节最大变化量只有 {motion:.4f} rad，几乎是静止轨迹；'
              '请重新示教，或设置 teach.allow_static=true。')
        return None, 3
    return trajectory, 0


def run_teach(task: TaskConfig, name: str, mock: bool) -> int:
    """Drag teach on any backend that implements ``freedrive``."""
    root = trajectory_root(task)
    print(f'示教轨迹目录：{root}')
    robot = None
    gripper = None
    try:
        robot, gripper, _ = _build_runtime(task, mock, with_cameras=False)
        robot.connect()
        _open_gripper(gripper)
        if not task.teach.freedrive:
            print('teach.freedrive=false：请先在示教器开启 Freedrive，再开始拖动机械臂。',
                  flush=True)
        trajectory = keyboard_teach(
            robot, gripper, name=name, task=task.task,
            rate=float(task.teach.rate),
            max_duration=float(task.teach.max_duration),
            meta={'backend': task.robot.get('kind', ''),
                  'camera': 'none'},
            freedrive=bool(task.teach.freedrive),
            wait_feedback=str(task.teach.wait_feedback or ''),
        )
        if trajectory is None:
            return 2
        motion = joint_motion(trajectory)
        print(f'关节最大变化量：{motion:.4f} rad', flush=True)
        if motion < 1e-4 and not task.teach.allow_static:
            print('警告：整段轨迹关节角几乎没变化，未保存。'
                  '若确实只需要夹爪开合，请设置 teach.allow_static=true。')
            return 3
        save_trajectory(root / name, trajectory)
        print('已保存：', root / name)
        print('轨迹摘要：', summarize(trajectory), flush=True)
        return 0
    except Exception as exc:
        print(f'示教失败：{type(exc).__name__}: {exc}')
        return 1
    finally:
        _close_gripper(gripper)
        if robot is not None:
            try:
                robot.disconnect()
            except Exception:
                pass


def _confirm(yes: bool, message: str = '确认急停在手、工作区无人，开始回放？') -> bool:
    if yes:
        return True
    try:
        answer = input(f'{message} [y/N] ').strip().lower()
    except EOFError:
        return False
    return answer in ('y', 'yes')


def run_replay(task: TaskConfig, name: str, mock: bool, yes: bool,
               speed_scale: float | None = None) -> int:
    trajectory, code = _load_trajectory_checked(task, name)
    if trajectory is None:
        return code
    robot = None
    gripper = None
    try:
        robot, gripper, _ = _build_runtime(task, mock, with_cameras=False)
        robot.connect()
        _open_gripper(gripper)
        replayer = _build_replayer(task, robot, gripper)
        result = replayer.run(
            trajectory,
            speed_scale=float(task.replay.speed_scale if speed_scale is None
                              else speed_scale),
            allow_fast=bool(task.replay.allow_fast),
            confirm=(lambda text: _confirm(yes, text)),
            realtime=bool(task.replay.realtime),
        )
        print('回放结果：', result.as_dict(), flush=True)
        return 0 if not result.aborted else 4
    except Exception as exc:
        print(f'回放失败：{type(exc).__name__}: {exc}')
        if robot is not None:
            try:
                robot.stop_motion()
            except Exception:
                pass
        return 1
    finally:
        _close_gripper(gripper)
        if robot is not None:
            try:
                robot.disconnect()
            except Exception:
                pass


def run_export(task: TaskConfig, name: str) -> int:
    """Export an existing teach trajectory without moving the robot."""
    if task.collect.cameras:
        print('export 模式不会产生相机帧；带相机的数据请用 collect 模式回放采集。')
        return 2
    trajectory, code = _load_trajectory_checked(task, name)
    if trajectory is None:
        return code
    robot = build_robot(task.robot)
    try:
        schema = DatasetSchema.for_robot(
            robot.spec, cameras=(), fps=int(round(task.collect.fps)),
            repo_id=task.collect.repo_id or f'local/{task.slug}',
            name_style=task.collect.feature_names)
        collector = collect_trajectory(
            trajectory, schema, dataset_root(task), task=task.task,
            export=exporter_for(task.collect.export), fps=float(task.collect.fps))
        collector.close()
        if collector.export_errors:
            print(f'导出错误：{collector.export_errors}')
            return 2
        print(f'已保存 {collector.saved} 条 episode：{collector.dataset_root}')
        return 0 if collector.saved else 1
    finally:
        try:
            robot.disconnect()
        except Exception:
            pass


def _build_runtime(task: TaskConfig, mock: bool, with_cameras: bool = True):
    """Build whatever backend/gripper the task profile names."""
    gripper_cfg = dict(task.gripper)
    gripper_kind = str(gripper_cfg.get('kind', '')).strip().lower()
    robot_profile = dict(task.robot)
    options = dict(robot_profile.get('options', {}) or {})
    options.setdefault('has_gripper', gripper_kind not in ('none', 'disabled', ''))
    robot_profile['options'] = options
    robot = build_robot(robot_profile, mock=mock)

    gripper = None
    if gripper_kind not in ('none', 'disabled', ''):
        gripper_options = dict(gripper_cfg.get('options', {}) or {})
        if gripper_kind in ('agx', 'agx_gripper', 'piper', 'piper_gripper'):
            gripper_options.setdefault('force', 1.0)
            gripper_options.setdefault('close_time_s',
                                       float(task.teach.gripper_close_time_s))
            gripper_options.setdefault('open_time_s',
                                       float(task.teach.gripper_open_time_s))
            gripper_options.setdefault('ramp_hz',
                                       float(task.teach.gripper_ramp_hz))
        if gripper_kind in ('robotiq', 'robotiq_2f85'):
            gripper_options.setdefault(
                'ip', str((task.robot.get('options', {}) or {}).get('ip', '')))
        if mock and gripper_kind in ('robotiq', 'robotiq_2f85'):
            gripper_options['backend'] = 'mock'
        gripper = build_gripper(gripper_kind, robot=robot, options=gripper_options)

    cameras = []
    if with_cameras and task.collect.cameras:
        entries = task.collect.cameras
        if mock:
            entries = [{
                'kind': 'mock', 'name': entry.get('name', 'camera'),
                'width': int(entry.get('width', 640)),
                'height': int(entry.get('height', 480)),
                'fps': int(entry.get('fps', 30)),
                'depth': bool(entry.get('depth', False)),
            } for entry in entries]
        cameras = list(build_camera_hub(entries))
    return robot, gripper, cameras


def run_collect(task: TaskConfig, name: str, mock: bool = False, yes: bool = False,
                speed_scale: float | None = None,
                dataset_override: str = '') -> int:
    """Replay one teach trajectory and record state/action/camera frames."""
    trajectory, code = _load_trajectory_checked(task, name)
    if trajectory is None:
        return code
    if not _confirm(yes, '确认急停在手、工作区无人，开始回放采集？'):
        print('已取消采集。')
        return 0

    target = dataset_root(task, dataset_override)
    print(f'任务：{task.task}')
    print(f'模式：回放并采集 | 轨迹：{trajectory_root(task) / name}')
    print(f'数据集根目录：{target}')

    robot = None
    gripper = None
    collector = None
    started = False
    try:
        robot, gripper, cameras = _build_runtime(task, mock)
        robot.connect()
        _open_gripper(gripper)
        camera_specs = tuple(CameraSpec.from_camera(camera) for camera in cameras)
        schema = DatasetSchema.for_robot(
            robot.spec, cameras=camera_specs,
            fps=int(round(task.collect.fps)),
            repo_id=task.collect.repo_id or f'local/{task.slug}',
            name_style=task.collect.feature_names)
        collector = EpisodeCollector(
            target, schema, robot=robot, gripper=gripper, cameras=cameras,
            export=exporter_for(task.collect.export),
            fps=float(task.collect.fps),
            # The replay loop already calls the hook at the dataset rate.
            # Do not apply a second wall-clock rate limit here: it would drop
            # frames when replay runs with realtime=False (tests, previews).
            min_interval_s=0.0, compress=bool(task.collect.compress))
        collector.start(task.task, meta={
            'source': 'teach_replay',
            'trajectory': name,
            'mode': task.mode,
            'speed_scale': float(task.replay.speed_scale if speed_scale is None
                                 else speed_scale),
        })
        started = True

        def capture_hook(t, frames, command, width):
            # ``command`` is the joint target for the current replay tick;
            # ``width`` is the scheduled gripper target at the same time.
            action = schema.observation(command, width)
            collector.capture_state(action=action, timestamp=float(t))

        replayer = _build_replayer(task, robot, gripper,
                                   capture_fps=float(task.collect.fps))
        result = replayer.run(
            trajectory,
            speed_scale=float(task.replay.speed_scale if speed_scale is None
                              else speed_scale),
            allow_fast=bool(task.replay.allow_fast),
            capture_hook=capture_hook,
            realtime=bool(task.replay.realtime),
        )
        if result.aborted:
            collector.discard()
            print(f'采集中止：{result.reason}')
            return 2
        collector.stop(success=True)
        started = False
    except Exception as exc:
        print(f'采集失败：{type(exc).__name__}: {exc}')
        if collector is not None and started:
            try:
                collector.discard()
            except Exception:
                pass
        if robot is not None:
            try:
                robot.stop_motion()
            except Exception:
                pass
        return 1
    finally:
        if collector is not None:
            try:
                collector.close()
            except Exception as exc:
                print(f'数据集导出失败：{exc}')
        _close_gripper(gripper)
        if robot is not None:
            try:
                robot.disconnect()
            except Exception:
                pass

    if collector is not None:
        errors = list(collector.export_errors)
        if collector.saved:
            print(f'已保存 {collector.saved} 条 episode：{collector.dataset_root}')
        if errors:
            print(f'导出错误：{errors}')
            return 2
        if collector.saved:
            return 0
    print('没有生成 episode，请检查回放是否产生了采集帧。')
    return 2


def _next_session_name(task: TaskConfig, root: Path) -> str:
    prefix = (task.session.prefix or task.slug or 'session').strip()
    for index in range(1, 1000):
        name = f'{prefix}_{index:02d}'
        if not (root / name).exists():
            return name
    raise RuntimeError(f'{root} 下同名轨迹过多')


def _run_session_episode(task: TaskConfig, robot, gripper, collector,
                         name: str, speed_scale: float | None) -> bool:
    """Replay one saved trajectory while the long-lived collector is open."""
    trajectory, code = _load_trajectory_checked(task, name)
    if trajectory is None:
        return False
    if collector.active:
        collector.discard()
    collector.start(task.task, meta={
        'source': 'joycon_session',
        'trajectory': name,
        'speed_scale': float(task.replay.speed_scale if speed_scale is None
                             else speed_scale),
    })

    def capture_hook(t, frames, command, width):
        action = collector.schema.observation(command, width)
        collector.capture_state(action=action, timestamp=float(t))

    replayer = _build_replayer(task, robot, gripper,
                               capture_fps=float(task.collect.fps))
    try:
        result = replayer.run(
            trajectory,
            speed_scale=float(task.replay.speed_scale if speed_scale is None
                              else speed_scale),
            allow_fast=bool(task.replay.allow_fast),
            capture_hook=capture_hook,
            realtime=bool(task.replay.realtime),
        )
        if result.aborted:
            collector.discard()
            print(f'采集中止：{result.reason}', flush=True)
            return False
        collector.stop(success=True)
        print(f'已录制 episode：{collector.dataset_root}', flush=True)
        return True
    except Exception as exc:
        print(f'回放采集失败：{type(exc).__name__}: {exc}', flush=True)
        try:
            collector.discard()
        except Exception:
            pass
        try:
            robot.stop_motion()
        except Exception:
            pass
        return False


def run_master_slave(task: TaskConfig, mock: bool = False,
                     dataset_override: str = '') -> int:
    """Sample a leader/follower pair passively; nothing is commanded."""
    robot, _gripper, cameras = _build_runtime(task, mock, with_cameras=True)
    # Fail closed: a profile that still writes configuration frames would pull a
    # master-slave pair out of teach mode.  The option itself stays where
    # robot-specific options belong, i.e. in the profile.
    backend = getattr(robot, 'inner', robot)
    if getattr(getattr(backend, 'config', None), 'configure_on_connect', False):
        print('master_slave 需要只读连接：请在 robot.options 里设置 '
              '"configure_on_connect": false')
        return 2

    fps = float(task.master_slave.fps or task.collect.fps or 30.0)
    schema = DatasetSchema.for_robot(
        robot.spec, cameras=cameras, fps=int(round(fps)),
        repo_id=task.collect.repo_id or f'local/{task.slug}',
        name_style=task.collect.feature_names)
    root = dataset_root(task, dataset_override)
    print(f'主从采样（只读，不发送任何报文）：{task.task}  fps={fps:g}  '
          f'相机={schema.camera_names or "无"}  ->  {root}')
    try:
        robot.connect()
        stats = sample_episodes(
            robot, schema, root, cameras=cameras,
            export=exporter_for(task.collect.export), fps=fps, task=task.task,
            auto_record=task.master_slave.auto_record,
            sample_seconds=task.master_slave.sample_seconds,
            max_frames=int(task.collect.max_frames or 0),
            compress=bool(task.collect.compress),
        )
    finally:
        robot.disconnect()
    print(f'采样结束：episodes={stats["episodes"]} frames={stats["frames"]} '
          f'rate={stats.get("rate", 0.0):.1f} Hz saved={stats["saved"]} '
          f'skipped={stats["skipped"]} root={stats["dataset_root"]}')
    for error in stats.get('export_errors') or ():
        print(f'导出失败：{error}')
    return 0 if stats['saved'] else 1


def _calibrate_poses(robot, cfg: CalibrateOptions) -> list[list[float]]:
    """Explicit poses when configured, otherwise a seeded sweep around the start."""
    if cfg.poses:
        return [[float(value) for value in pose] for pose in cfg.poses]
    import numpy as _np
    state = robot.state()
    if state.q is None:
        raise DatasetError('robot has no joint feedback; cannot generate poses')
    base = _np.asarray(state.q, dtype=float)
    rng = _np.random.default_rng(int(cfg.pose_seed))
    count = max(3, int(cfg.pose_count))
    delta = max(0.02, float(cfg.pose_delta_rad))
    poses = []
    for candidate in range(count * 6):
        if len(poses) >= count:
            break
        offset = rng.uniform(-delta, delta, size=base.shape)
        offset[0] = 0.0                      # keep the base yaw
        poses.append((base + offset).tolist())
    return poses


def _run_calibrate_auto(robot, camera, detector, session, cfg: CalibrateOptions,
                        path, need: int, yes: bool) -> int:
    """Drive the arm through the poses, detect the board, then solve once."""
    if not yes:
        print('自动标定会移动机械臂：确认工作区无人、急停在手后加 --yes')
        return 2
    try:
        poses = _calibrate_poses(robot, cfg)
    except Exception as exc:
        print(f'无法生成标定位姿：{exc}')
        return 2
    print(f'自动标定：{len(poses)} 个位姿，速度 {cfg.speed_scale:g}，'
          f'每处停 {cfg.settle_s:g}s；目标 ≥{need} 组')
    try:
        robot.connect()
        # ``enable`` lives on the concrete backend; SafeRobot does not proxy it.
        backend = getattr(robot, 'inner', robot)
        if hasattr(backend, 'enable'):
            try:
                backend.enable()
            except Exception as exc:
                print(f'使能失败：{exc}')
                return 2
        if not camera.opened:
            camera.open()
        for index, pose in enumerate(poses, 1):
            if getattr(robot, 'protective_stop', lambda: False)():
                print('检测到保护性停止，终止自动标定')
                break
            try:
                robot.move_j(pose, speed=float(cfg.speed_scale), wait=True)
            except Exception as exc:
                print(f'[{index}/{len(poses)}] 运动失败：{exc}（跳过）')
                continue
            time.sleep(max(0.0, float(cfg.settle_s)))
            frame = camera.snapshot()
            if frame.rgb is None:
                print(f'[{index}/{len(poses)}] 无图像，跳过')
                continue
            if cfg.kind == 'intrinsics':
                observation = detector.detect(frame.rgb)
                if observation is not None:
                    session.views.append(observation.view)
                    session.images.append(frame.rgb)
            else:
                try:
                    observation = session.capture(image=frame.rgb)
                except Exception as exc:
                    print(f'[{index}/{len(poses)}] 采集失败：{type(exc).__name__}: {exc}')
                    continue
            if observation is None:
                print(f'[{index}/{len(poses)}] 没检测到棋盘，跳过')
                continue
            count = len(session.views) if cfg.kind == 'intrinsics' else len(session.base_tcp)
            print(f'[{index}/{len(poses)}] 采集 {count}/{need}')
            if count >= need:
                break
    except KeyboardInterrupt:
        print('中断')
    finally:
        try:
            camera.close()
        except Exception:
            pass
        try:
            robot.disconnect()
        except Exception:
            pass

    count = len(session.views) if cfg.kind == 'intrinsics' else len(session.base_tcp)
    if count < 3:
        print(f'样本不足（{count}），没有保存')
        return 1
    summary = session.save(path, camera_name=camera.name, apply=bool(cfg.apply))
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    print(f'已写入 {summary["path"]}')
    return 0


def run_calibrate(task: TaskConfig, mock: bool = False,
                  dataset_override: str = '', yes: bool = False) -> int:
    """Collect board observations from the profile and solve the calibration."""
    from .calibrate import CalibrationSession, ChessboardDetector

    cfg = task.calibrate
    robot, _gripper, cameras = _build_runtime(task, mock, with_cameras=True)
    if not cameras:
        print('calibrate 需要至少一个相机：请在 collect.cameras 里配置')
        return 2
    camera = None
    if cfg.camera:
        camera = next((item for item in cameras if item.name == cfg.camera), None)
    elif cameras:
        camera = cameras[0]
    if camera is None:
        print(f'找不到相机 {cfg.camera!r}；可用：{", ".join(c.name for c in cameras)}')
        return 2

    if cfg.board == 'charuco':
        from .calibrate import CharucoDetector
        detector = CharucoDetector(
            squares=tuple(int(value) for value in cfg.squares),
            square_size=float(cfg.square_size),
            marker_size=float(cfg.marker_size),
            aruco_dict=str(cfg.aruco_dict),
            min_corners=int(cfg.min_corners),
        )
    else:
        from .calibrate import ChessboardDetector
        pattern = tuple(int(value) for value in cfg.pattern)
        detector = ChessboardDetector(pattern=pattern, square_size=float(cfg.square_size))
    session = CalibrationSession(robot=None if cfg.kind == 'intrinsics' else robot,
                                 camera=camera, detector=detector, mode=cfg.eye,
                                 camera_name=camera.name)
    path = Path(cfg.path) if cfg.path else task_dir(task) / 'calibration.json'
    need = max(3, int(cfg.min_samples))

    if cfg.auto:
        return _run_calibrate_auto(robot, camera, detector, session, cfg, path, need, yes)

    print(f'标定：camera={camera.name}  kind={cfg.kind}  eye={cfg.eye}  board={cfg.board}')
    if cfg.board == 'charuco':
        print(f'ChArUco {cfg.squares[0]}x{cfg.squares[1]}，方格 {cfg.square_size * 1000:.1f} mm，'
              f'字典 {cfg.aruco_dict}，目标 ≥{need} 组')
    else:
        print(f'棋盘 {cfg.pattern[0]}x{cfg.pattern[1]}，方格 {cfg.square_size * 1000:.1f} mm，'
              f'目标 ≥{need} 组')
    print('按键：空格=采集一帧，s=求解并保存，q=退出（不发送任何运动指令）')
    keys = KeyReader().start()
    try:
        if cfg.kind != 'intrinsics':
            robot.connect()
        if not camera.opened:
            camera.open()
        # The first snapshot right after open() is often empty; wait for a real
        # frame so the first key press is not wasted.
        warmed = False
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if camera.snapshot().rgb is not None:
                warmed = True
                break
            time.sleep(0.05)
        print('相机就绪' if warmed else '相机 5 秒内没有出图，检查连接/曝光',
              f'  内参 {"有" if camera.intrinsics is not None else "无"}')
        while True:
            key = keys.get()
            if key == ' ':
                frame = camera.snapshot()
                if frame.rgb is None:
                    print('相机没有帧，稍后重试')
                    continue
                if cfg.kind == 'intrinsics':
                    observation = detector.detect(frame.rgb)
                    if observation is None:
                        print('没检测到棋盘')
                        continue
                    session.views.append(observation.view)
                    session.images.append(frame.rgb)
                    print(f'采集 {len(session.views)}/{need} 组视角')
                else:
                    observation = session.capture(image=frame.rgb)
                    if observation is None:
                        print('没检测到棋盘')
                        continue
                    print(f'采集 {len(session.base_tcp)}/{need} 组位姿，'
                          f'视角 {len(session.views)}')
            elif key == 's':
                count = len(session.views) if cfg.kind == 'intrinsics' else len(session.base_tcp)
                if count < 3:
                    print(f'样本太少（{count}），至少需要 3 组，建议 ≥{need}')
                    continue
                summary = session.save(path, camera_name=camera.name, apply=bool(cfg.apply))
                print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
                print(f'已写入 {summary["path"]}')
                break
            elif key == 'q':
                print('退出（未保存）')
                break
            time.sleep(0.02)
    except KeyboardInterrupt:
        pass
    finally:
        keys.stop()
        try:
            camera.close()
        except Exception:
            pass
        try:
            robot.disconnect()
        except Exception:
            pass
    return 0 if path.exists() else 1


def run_session(task: TaskConfig, mock: bool = False) -> int:
    """Persistent Joy-Con session: B teach, A save, X×2 replay+collect.

    The robot, gripper and cameras stay connected for the whole session, so the
    same process can keep recording episodes without reconnecting hardware.
    ``Y`` twice removes the latest teach trajectory; ``Home`` prints status.
    """
    root = trajectory_root(task)
    target = dataset_root(task)
    print(f'Joy-Con 常驻会话：{task.task}')
    print(f'轨迹目录：{root}')
    print(f'数据集根目录：{target}')
    print('B 开始示教 | A 结束保存 | X×2 回放并采集 | Y×2 删除上一条轨迹 | '
          'Home 查看状态 | Ctrl+C 退出', flush=True)

    robot = None
    gripper = None
    collector = None
    hub = None
    last_name = latest_trajectory(root)
    confirm_x_until = 0.0
    confirm_y_until = 0.0
    try:
        robot, gripper, cameras = _build_runtime(task, mock)
        robot.connect()
        _open_gripper(gripper)
        camera_specs = tuple(CameraSpec.from_camera(camera) for camera in cameras)
        schema = DatasetSchema.for_robot(
            robot.spec, cameras=camera_specs,
            fps=int(round(task.collect.fps)),
            repo_id=task.collect.repo_id or f'local/{task.slug}',
            name_style=task.collect.feature_names)
        collector = EpisodeCollector(
            target, schema, robot=robot, gripper=gripper, cameras=cameras,
            export=exporter_for(task.collect.export),
            fps=float(task.collect.fps), min_interval_s=0.0,
            compress=bool(task.collect.compress))
        hub = ButtonHub()
        hub.start()
        while True:
            try:
                button, _stamp = hub.events.get(timeout=0.1)
            except queue.Empty:
                continue
            now = time.monotonic()
            if button == 'b':
                name = _next_session_name(task, root)
                confirm_x_until = confirm_y_until = 0.0
                # Do not let a stale A from the previous teach stop this one.
                while True:
                    try:
                        hub.events.get_nowait()
                    except queue.Empty:
                        break
                trajectory = joycon_teach(
                    robot, gripper, hub, name=name, task=task.task,
                    rate=float(task.teach.rate),
                    max_duration=float(task.teach.max_duration),
                    meta={'backend': task.robot.get('kind', ''),
                          'camera': 'none', 'source': 'joycon_session'},
                    freedrive=bool(task.teach.freedrive),
                    wait_feedback=str(task.teach.wait_feedback or ''),
                )
                if trajectory is None:
                    print('本次示教已放弃。', flush=True)
                    continue
                motion = joint_motion(trajectory)
                if motion < 1e-4 and not task.teach.allow_static:
                    print(f'轨迹几乎静止（{motion:.4f} rad），未保存。', flush=True)
                    continue
                save_trajectory(root / name, trajectory)
                last_name = name
                print(f'已保存轨迹：{root / name}', flush=True)
                if task.session.auto_replay:
                    _run_session_episode(task, robot, gripper, collector, name,
                                         task.session.replay_speed_scale)
            elif button == 'x':
                if last_name is None:
                    last_name = latest_trajectory(root)
                if last_name is None:
                    print('还没有可回放的轨迹。', flush=True)
                    continue
                if task.session.confirm_replay and now > confirm_x_until:
                    confirm_x_until = now + float(task.session.double_press_timeout_s)
                    print(f'再按一次 X，回放并采集 {last_name}。', flush=True)
                    continue
                confirm_x_until = 0.0
                _run_session_episode(task, robot, gripper, collector, last_name,
                                     task.session.replay_speed_scale)
            elif button == 'y':
                if last_name is None:
                    last_name = latest_trajectory(root)
                if last_name is None:
                    print('没有可删除的示教轨迹。', flush=True)
                    continue
                if now > confirm_y_until:
                    confirm_y_until = now + float(task.session.double_press_timeout_s)
                    print(f'再按一次 Y，删除轨迹 {last_name}。', flush=True)
                    continue
                confirm_y_until = 0.0
                shutil.rmtree(root / last_name, ignore_errors=True)
                print(f'已删除轨迹：{root / last_name}', flush=True)
                last_name = latest_trajectory(root)
            elif button == 'r':
                home = (task.robot.get('options', {}) or {}).get('home_joints')
                if home:
                    print('缓慢回默认姿态……', flush=True)
                    robot.move_j(home, speed=0.2, accel=0.25, wait=True)
            elif button == 'home':
                print(state_line(robot, gripper), flush=True)
                print(f'当前上一条轨迹：{last_name or "(none)"}', flush=True)
    except KeyboardInterrupt:
        print('\n结束 Joy-Con 会话。', flush=True)
        return 0
    except Exception as exc:
        print(f'Joy-Con 会话失败：{type(exc).__name__}: {exc}', flush=True)
        return 1
    finally:
        if hub is not None:
            hub.stop()
        if collector is not None:
            try:
                collector.close()
            except Exception as exc:
                print(f'数据集导出失败：{exc}', flush=True)
        _close_gripper(gripper)
        if robot is not None:
            try:
                robot.disconnect()
            except Exception:
                pass


# ------------------------------------------------------------------- dispatch
def print_plan(task: TaskConfig, mode: str, name: str | None,
               dataset_override: str = '') -> None:
    root = trajectory_root(task)
    print(f'配置文件：{task.source_path}')
    print(f'任务：{task.task}  (slug={task.slug})')
    print(f'模式：{mode}')
    print(f'机器人：{task.robot.get("kind")}'
          f'  name={task.robot.get("name", "")}'
          f'  max_joint_speed={_robot_speed(task.robot)} rad/s')
    print(f'夹爪：{task.gripper.get("kind", "")}')
    print(f'轨迹目录：{root}')
    print(f'轨迹名称：{name or "(自动生成/使用最新)"}')
    print(f'数据集根目录：{dataset_root(task, dataset_override)}')
    print(f'相机：{len(task.collect.cameras)} 个'
          + (f" ({', '.join(str(item.get('name', 'camera')) for item in task.collect.cameras)})"
             if task.collect.cameras else '（仅关节/夹爪）'))
    print(f'示教：{task.teach.rate} Hz，最长 {task.teach.max_duration} s')
    print(f'回放：speed_scale={task.replay.speed_scale}，'
          f'smoothing={task.replay.smoothing}，'
          f'continuous={task.replay.continuous}')
    if mode == 'session':
        print(f'Joy-Con 会话：prefix={task.session.prefix or task.slug}，'
              f'X 确认={task.session.confirm_replay}，'
              f'自动回放={task.session.auto_replay}')
        print('按键：B 示教 / A 保存 / X×2 回放采集 / Y×2 删除轨迹')
    if mode == 'master_slave':
        duration = ('不限时长' if task.master_slave.sample_seconds <= 0
                    else f'{task.master_slave.sample_seconds:g} s 后自动结束')
        cap = ('单条不限帧数' if not task.collect.max_frames
               else f'单条上限 {task.collect.max_frames} 帧')
        start = '启动即录' if task.master_slave.auto_record else '按空格开始'
        print(f'主从采样：fps={task.master_slave.fps:g}，{start}，{duration}，{cap}')
        print('按键：空格 结束并保存（再按开始下一条），x 丢弃当前，q 退出')
    if mode == 'calibrate':
        cfg = task.calibrate
        target = Path(cfg.path) if cfg.path else task_dir(task) / 'calibration.json'
        print(f'标定：kind={cfg.kind} eye={cfg.eye} camera={cfg.camera or "(第一个)"}')
        if cfg.board == 'charuco':
            print(f'ChArUco {cfg.squares[0]}x{cfg.squares[1]}，方格 '
                  f'{cfg.square_size * 1000:.1f} mm，字典 {cfg.aruco_dict}，'
                  f'目标 ≥{max(3, cfg.min_samples)} 组')
        else:
            print(f'棋盘 {cfg.pattern[0]}x{cfg.pattern[1]}，方格 '
                  f'{cfg.square_size * 1000:.1f} mm，目标 ≥{max(3, cfg.min_samples)} 组')
        print(f'输出：{target}')
        print('按键：空格 采集一帧，s 求解并保存，q 退出')
        if cfg.auto:
            poses = len(cfg.poses) if cfg.poses else max(3, int(cfg.pose_count))
            print(f'自动扫描：{poses} 个位姿，速度 {cfg.speed_scale:g}，'
                  f'停止 {cfg.settle_s:g}s（需要 --yes）')


def execute(task: TaskConfig, mode: str | None = None, name: str | None = None,
            mock: bool = False, yes: bool = False, force: bool = False,
            speed_scale: float | None = None,
            dataset_override: str = '', dry_run: bool = False,
            auto_record: bool | None = None,
            sample_seconds: float | None = None) -> int:
    mode = (mode or task.mode).strip().lower()
    if mode not in MODES:
        print(f'未知模式 {mode!r}，可选：{", ".join(MODES)}')
        return 2
    if auto_record is not None:
        task.master_slave.auto_record = bool(auto_record)
    if sample_seconds is not None:
        task.master_slave.sample_seconds = float(sample_seconds)
    if mode == 'session':
        if speed_scale is not None:
            task.session.replay_speed_scale = float(speed_scale)
        if dry_run:
            print_plan(task, mode, task.session.prefix or task.slug, dataset_override)
            return 0
        return run_session(task, mock=mock)
    if mode == 'master_slave':
        if dry_run:
            print_plan(task, mode, None, dataset_override)
            return 0
        return run_master_slave(task, mock=mock, dataset_override=dataset_override)
    if mode == 'calibrate':
        if dry_run:
            print_plan(task, mode, None, dataset_override)
            return 0
        return run_calibrate(task, mock=mock, dataset_override=dataset_override,
                             yes=yes)
    root = trajectory_root(task)
    if mode in ('teach', 'teach_collect'):
        resolved = name or task.name or _next_name(task, root)
        if (root / resolved).exists() and not force and not dry_run:
            print(f'轨迹已存在：{root / resolved}；换一个 --name，或加 --force 覆盖。')
            return 2
    else:
        try:
            resolved = _collect_name(task, root, name)
        except FileNotFoundError as exc:
            if not dry_run:
                print(exc)
                return 2
            # A dry run should still show where collect/replay would write,
            # even before the first teach creates the trajectory directory.
            resolved = name or task.name or '(latest)'

    if dry_run:
        print_plan(task, mode, resolved, dataset_override)
        return 0

    if mode in ('teach', 'teach_collect'):
        code = run_teach(task, resolved, mock)
        if code != 0 or mode == 'teach':
            return code
        # Continue with replay + collection using the trajectory just saved.
        return run_collect(task, resolved, mock=mock, yes=yes,
                           speed_scale=speed_scale,
                           dataset_override=dataset_override)
    if mode == 'replay':
        return run_replay(task, resolved, mock, yes, speed_scale)
    if mode == 'collect':
        return run_collect(task, resolved, mock=mock, yes=yes,
                           speed_scale=speed_scale,
                           dataset_override=dataset_override)
    if mode == 'export':
        return run_export(task, resolved)
    return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='piper-task',
        description='按任务 JSON 启动 Piper X 示教 / 回放 / 采集')
    parser.add_argument('task', nargs='?', help='任务配置名，或 JSON 路径')
    parser.add_argument('action', nargs='?', choices=MODES,
                        help='覆盖配置文件里的 mode')
    parser.add_argument('--config', help='显式指定任务 JSON')
    parser.add_argument('--list', action='store_true', help='列出可用任务')
    parser.add_argument('--name', help='覆盖轨迹名称')
    parser.add_argument('--force', action='store_true', help='允许覆盖已有示教轨迹')
    parser.add_argument('--speed-scale', type=float, default=None,
                        help='覆盖回放速度倍率')
    parser.add_argument('--dataset-root', default='', help='覆盖数据集根目录')
    parser.add_argument('--yes', action='store_true', help='跳过回放安全确认')
    parser.add_argument('--mock', action='store_true', help='离线 mock，不接硬件')
    parser.add_argument('--auto-record', action='store_true',
                        help='master_slave：启动即开始录制，不等按键')
    parser.add_argument('--sample-seconds', type=float, default=None,
                        help='master_slave：采样时长（0 = 一直采到按 q）')
    parser.add_argument('--dry-run', action='store_true', help='只打印解析后的计划，不动机器人')
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.list:
        print_configs()
        return 0
    if args.config and args.task in MODES and args.action is None:
        # ``run.bat --config task.json collect``: the first positional is
        # the action, not a task name.
        args.action, args.task = args.task, None
    if not args.task and not args.config:
        parser.print_help()
        return 0
    try:
        task = load_task_config(args.config or args.task)
    except Exception as exc:
        print(f'加载任务配置失败：{exc}')
        return 2
    try:
        return execute(task, mode=args.action, name=args.name, mock=args.mock,
                       yes=args.yes, force=args.force, speed_scale=args.speed_scale,
                       dataset_override=args.dataset_root, dry_run=args.dry_run,
                       auto_record=args.auto_record or None,
                       sample_seconds=args.sample_seconds)
    except Exception as exc:
        print(f'任务执行失败：{type(exc).__name__}: {exc}')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
