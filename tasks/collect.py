"""Collect mode."""
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

from .build import _exporter
from .options import TaskConfig
from .paths import lerobot_root

from .build import (_build_cameras, _build_replayer, _build_runtime,
                        _confirm, _load_trajectory_checked, _open_gripper,
                        _close_gripper)
from .paths import (_collect_name, dataset_root, latest_trajectory,
                    trajectory_root)


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
            export=_exporter(task), dataset_root=lerobot_root(task),
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


