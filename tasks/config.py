"""Task config discovery and loading."""
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
from ..runtime.profiles import build_camera_hub, build_gripper, build_robot
from ..teach import (Replayer, KeyReader, joycon_teach, joint_motion, keyboard_teach,
                     load_trajectory, save_trajectory, state_line, summarize)
from ..teleop import ButtonHub

from .options import TaskConfig



from . import constants
from .constants import DEFAULT_ROOT, MODES, REPO_ROOT  # noqa: F401

# Re-exported for importers; path lookups below read ``constants.ROOT`` so a
# test or host can redirect the workspace root at runtime.
ROOT = constants.ROOT

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
    # Project configs are authoritative; workspace-level configs stay as a
    # fallback for older tasks that have not been moved into this package.
    code_root = REPO_ROOT
    dirs = [code_root / 'configs' / 'tasks', constants.ROOT / 'configs' / 'tasks']
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
    root = constants.ROOT / DEFAULT_ROOT / 'tasks'
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
    if float(task.hand_demo.fps) < 0:
        raise ValueError(f'{path}: hand_demo.fps 不能为负（0 = 使用 collect.fps）')
    if not isinstance(task.collect.record_state_action, bool):
        raise ValueError(f'{path}: collect.record_state_action 必须是 true 或 false')
    if task.mode == 'hand_demo' and task.collect.record_state_action:
        raise ValueError(f'{path}: hand_demo 需要 collect.record_state_action=false')
    if str(task.hand_demo.input).strip().lower() not in ('keyboard', 'joycon'):
        raise ValueError(f'{path}: hand_demo.input 只能是 keyboard 或 joycon')
    task.hand_demo.input = str(task.hand_demo.input).strip().lower()
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
