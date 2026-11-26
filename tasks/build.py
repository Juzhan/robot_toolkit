"""Shared construction helpers for robots, grippers, cameras and runs."""
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

from .options import TaskConfig
from .paths import (dataset_root, trajectory_root,
                     _next_name, _next_session_name)


def _exporter(task: TaskConfig):
    from ..collect.export import isolated_exporter
    name = str(task.collect.export).strip().lower()
    if task.collect.isolated_export and name == 'lerobot':
        return isolated_exporter(name)
    return exporter_for(name)


def _delete_last_episode(task: TaskConfig, collector) -> str:
    """Remove the newest exported episode; refuses while exports are pending."""
    from ..collect.dataset_edit import delete_last_isolated
    if collector.active:
        return '正在录制，不能删除'
    if collector.recorder.status().queued:
        return '还有数据在导出，导出完成后再删'
    try:
        result = delete_last_isolated(collector.dataset_root)
    except Exception as exc:
        return f'删除失败：{exc}'
    return f'已删除第 {result["deleted"] + 1} 条，数据集剩 {result["episodes"]} 条'


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


def _confirm(yes: bool, message: str = '确认急停在手、工作区无人，开始回放？') -> bool:
    if yes:
        return True
    try:
        answer = input(f'{message} [y/N] ').strip().lower()
    except EOFError:
        return False
    return answer in ('y', 'yes')


def _camera_entries(task: TaskConfig, mock: bool) -> list[dict]:
    """Camera entries from ``collect.cameras``; ``--mock`` swaps in MockCamera."""
    entries = list(task.collect.cameras)
    if mock:
        entries = [{
            'kind': 'mock', 'name': entry.get('name', 'camera'),
            'width': int(entry.get('width', 640)),
            'height': int(entry.get('height', 480)),
            'fps': int(entry.get('fps', 30)),
            'depth': bool(entry.get('depth', False)),
        } for entry in entries]
    return entries


def _build_cameras(task: TaskConfig, mock: bool = False) -> list:
    entries = _camera_entries(task, mock)
    return list(build_camera_hub(entries)) if entries else []


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

    cameras = _build_cameras(task, mock) if with_cameras else []
    return robot, gripper, cameras


