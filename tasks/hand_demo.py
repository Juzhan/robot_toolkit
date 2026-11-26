"""Vision-only human demonstration recording."""
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

from ..collect.hand_demo import record_hand_demos
from ..collect.speech import speak
from ..core.conventions import RobotSpec
from .options import TaskConfig

from .build import _build_cameras, _build_runtime
from .paths import (dataset_root, _collect_name)


class _JoyConKeys:
    """Adapt Joy-Con buttons to hand-demo keys, matching session semantics.

    In ``session`` mode B starts and A saves; here B is ignored while recording
    and A is ignored while idle, so a single button never changes meaning.  Y
    cancels the current clip and Plus deletes the newest episode, like session
    mode.  X and R have no counterpart in a vision-only session and are ignored.
    Home reports status; Minus exits (unused in session mode).
    """

    def __init__(self, hub):
        self.hub = hub
        self.active = False

    def sync(self, active: bool) -> None:
        self.active = bool(active)

    def get(self):
        try:
            button, _stamp = self.hub.events.get_nowait()
        except queue.Empty:
            return None
        if button == 'b':
            return None if self.active else ' '
        if button == 'a':
            return ' ' if self.active else None
        if button == 'y':
            return 'cancel'
        if button == 'plus':
            return 'delete'
        if button == 'home':
            return 'status'
        if button == 'minus':
            return 'q'
        return None


def run_hand_demo(task: TaskConfig, mock: bool = False,
                  dataset_override: str = '') -> int:
    """Record camera-only human demonstrations; no robot is built or moved.

    The flow and dataset format match robot collection (same EpisodeCollector,
    same LeRobot export), with ``collect.record_state_action=false`` switching
    off the state/action modality.  ``hand_demo.input`` selects keyboard or
    Joy-Con; both can record any number of consecutive clips.
    """
    if not task.collect.cameras:
        print('hand_demo 需要至少一个相机：请在 collect.cameras 里配置')
        return 2
    cameras = _build_cameras(task, mock)
    if not cameras:
        print('hand_demo 没有可用的相机')
        return 2
    fps = float(task.hand_demo.fps or task.collect.fps or 30.0)
    root = dataset_root(task, dataset_override)
    tracker_name = str(task.hand_demo.tracker or '').strip()
    input_kind = str(task.hand_demo.input or 'keyboard').strip().lower()
    spec = RobotSpec(name='hand_demo', joint_names=(), gripper=None)
    camera_specs = tuple(CameraSpec.from_camera(camera) for camera in cameras)
    schema = DatasetSchema.for_robot(
        spec, cameras=camera_specs, fps=int(round(fps)),
        repo_id=task.collect.repo_id or f'local/{task.slug}',
        name_style=task.collect.feature_names,
        include_state_action=False)
    print(f'人手演示采集（仅视频）：{task.task}  fps={fps:g}  '
          f'相机={[camera.name for camera in cameras]}  '
          f'输入={input_kind}  语音={"开" if task.hand_demo.voice else "关"}  '
          f'导出={task.collect.export}  ->  {root}')
    if tracker_name:
        print(f'hand_demo.tracker={tracker_name!r}：手部跟踪接口已预留但尚未实现，'
              '本次只录视频。')
    hub = None
    keys = None
    if input_kind == 'joycon':
        hub = ButtonHub()
        hub.start()
        keys = _JoyConKeys(hub)
        print('Joy-Con：B=开始，A=结束保存，Y=取消当前（按两次），'
              '+=删除上一条（按两次），Home=状态，-=退出（X/R 无操作）', flush=True)
    else:
        print('键盘：空格=结束并保存（再按开始下一条），x=丢弃当前，'
              'd=删除上一条（按两次），q=退出', flush=True)
    announce = None
    if task.hand_demo.voice:
        intro = ('就绪：B 开始录制，A 结束保存，Y 取消当前，'
                 '加号键删除上一条，Home 检查状态，减号键退出'
                 if input_kind == 'joycon' else
                 '就绪：空格开始或结束保存，X 丢弃，D 删除上一条，Q 退出')
        # Old session announces the control map when it becomes idle, not when
        # the first recording starts.
        speak(intro)
        announce = speak
    try:
        stats = record_hand_demos(
            cameras, root, schema, fps=fps, task=task.task,
            auto_record=bool(task.hand_demo.auto_record),
            sample_seconds=float(task.hand_demo.sample_seconds),
            max_frames=int(task.collect.max_frames or 0),
            export=exporter_for(task.collect.export),
            tracker=None, keys=keys, announce=announce,
        )
    finally:
        if hub is not None:
            hub.stop()
    print(f'采集结束：episodes={stats["episodes"]} frames={stats["frames"]} '
          f'rate={stats.get("rate", 0.0):.1f} Hz saved={stats["saved"]} '
          f'skipped={stats["skipped"]} root={stats["dataset_root"]}')
    for error in stats.get('export_errors') or ():
        print(f'导出失败：{error}')
    return 0 if stats['saved'] else 1


