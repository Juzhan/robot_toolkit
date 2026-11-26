"""Dataset export mode."""
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

from .build import _load_trajectory_checked
from .paths import dataset_root


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


