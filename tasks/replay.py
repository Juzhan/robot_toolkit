"""Trajectory replay mode."""
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

from .build import (_build_replayer, _build_runtime, _confirm,
                        _load_trajectory_checked, _open_gripper, _close_gripper)


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


