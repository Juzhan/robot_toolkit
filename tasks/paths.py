"""Output directories and episode naming."""
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

from . import constants
from .options import TaskConfig


def _path(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (constants.ROOT / path).resolve()


def task_dir(task: TaskConfig) -> Path:
    return _path(task.root) / task.slug


def trajectory_root(task: TaskConfig, override: str = '') -> Path:
    value = (override or task.collect.trajectory_root or '').strip()
    return _path(value) if value else task_dir(task) / 'trajectories'


def dataset_root(task: TaskConfig, override: str = '') -> Path:
    value = (override or task.collect.dataset_root or '').strip()
    return _path(value) if value else task_dir(task) / 'dataset'


def lerobot_root(task: TaskConfig) -> Path:
    value = (task.collect.lerobot_root or '').strip()
    return _path(value) if value else task_dir(task) / 'lerobot'


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
def _next_session_name(task: TaskConfig, root: Path) -> str:
    prefix = (task.session.prefix or task.slug or 'session').strip()
    for index in range(1, 1000):
        name = f'{prefix}_{index:02d}'
        if not (root / name).exists():
            return name
    raise RuntimeError(f'{root} 下同名轨迹过多')


