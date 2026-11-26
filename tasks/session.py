"""Interactive teach/replay collection session."""
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
from .paths import lerobot_root, trajectory_root

from .build import (_build_cameras, _build_replayer, _build_runtime,
                        _delete_last_episode, _load_trajectory_checked,
                        _open_gripper, _close_gripper)
from .paths import (dataset_root, latest_trajectory, _next_session_name)


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
    print(f'LeRobot 数据集：{lerobot_root(task)}（跨会话追加）')
    print('B 开始示教 | A 结束保存 | X×2 回放并采集 | Y×2 删除上一条轨迹 | '
          '+×2 删除数据集最后一条 | Home 查看状态 | Ctrl+C 退出', flush=True)

    robot = None
    gripper = None
    collector = None
    hub = None
    last_name = latest_trajectory(root)
    from ..teleop.announce import Announcer
    from ..teleop.session_input import DoublePress
    voice = Announcer(task.session.voice)
    confirm = DoublePress(task.session.double_press_timeout_s)
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
            fps=float(task.collect.fps), min_interval_s=0.0,
            compress=bool(task.collect.compress))
        hub = ButtonHub()
        hub.start()
        voice.say('示教会话就绪，按 B 开始示教')
        while True:
            try:
                button, _stamp = hub.events.get(timeout=0.1)
            except queue.Empty:
                continue
            if button == 'b':
                name = _next_session_name(task, root)
                confirm.reset()
                voice.say('开始示教')
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
                    voice.say('示教已放弃')
                    continue
                motion = joint_motion(trajectory)
                if motion < 1e-4 and not task.teach.allow_static:
                    print(f'轨迹几乎静止（{motion:.4f} rad），未保存。', flush=True)
                    voice.say('轨迹几乎静止，未保存')
                    continue
                save_trajectory(root / name, trajectory)
                last_name = name
                print(f'已保存轨迹：{root / name}', flush=True)
                voice.say('轨迹已保存')
                if task.session.auto_replay:
                    _run_session_episode(task, robot, gripper, collector, name,
                                         task.session.replay_speed_scale)
            elif button == 'x':
                if last_name is None:
                    last_name = latest_trajectory(root)
                if last_name is None:
                    print('还没有可回放的轨迹。', flush=True)
                    continue
                if task.session.confirm_replay and not confirm.confirm('x'):
                    print(f'再按一次 X，回放并采集 {last_name}。', flush=True)
                    voice.say('再按一次 X 回放采集')
                    continue
                voice.say('开始回放采集')
                _run_session_episode(task, robot, gripper, collector, last_name,
                                     task.session.replay_speed_scale)
            elif button == 'y':
                if last_name is None:
                    last_name = latest_trajectory(root)
                if last_name is None:
                    print('没有可删除的示教轨迹。', flush=True)
                    continue
                if not confirm.confirm('y'):
                    print(f'再按一次 Y，删除轨迹 {last_name}。', flush=True)
                    voice.say('再按一次 Y 删除')
                    continue
                shutil.rmtree(root / last_name, ignore_errors=True)
                print(f'已删除轨迹：{root / last_name}', flush=True)
                voice.say('轨迹已删除')
                last_name = latest_trajectory(root)
            elif button == 'r':
                home = (task.robot.get('options', {}) or {}).get('home_joints')
                if home:
                    print('缓慢回默认姿态……', flush=True)
                    robot.move_j(home, speed=0.2, accel=0.25, wait=True)
            elif button == 'plus':
                if not confirm.confirm('plus'):
                    print('再按一次 +，删除数据集最后一条。', flush=True)
                    voice.say('再按一次删除上一条')
                    continue
                message = _delete_last_episode(task, collector)
                print(message, flush=True)
                voice.say(message)
            elif button == 'home':
                from ..link_check import run_link_check
                print(state_line(robot, gripper), flush=True)
                print(f'当前上一条轨迹：{last_name or "(none)"}', flush=True)
                _ok, summary = run_link_check(robot, gripper, cameras,
                                              log=lambda text: print(text, flush=True))
                voice.say(summary)
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
