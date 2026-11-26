"""Safety envelope capture."""
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
from .paths import task_dir

from .build import _build_runtime, _confirm
from .paths import dataset_root


def run_safety(task: TaskConfig, mock: bool = False) -> int:
    """Capture a reusable safety envelope while the operator drags the arm."""
    cfg = task.safety
    robot, _gripper, _cameras = _build_runtime(task, mock, with_cameras=False)
    path = Path(cfg.path) if cfg.path else task_dir(task) / 'safety_envelope.json'
    live_path = path.parent / 'live_state.json'
    keys = KeyReader().start()
    samples = []
    started = time.monotonic()
    try:
        robot.connect()
        if cfg.freedrive:
            robot.freedrive(True)
        if cfg.wait_feedback:
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline:
                if robot.state().extra.get('feedback', '') == cfg.wait_feedback:
                    break
                time.sleep(0.05)
            else:
                print(f'未收到 {cfg.wait_feedback!r} 反馈，未开始采集。')
                return 2
        print('安全包络采集：请手动拖动机械臂覆盖允许工作区域。', flush=True)
        print('按 s/回车保存，q 放弃；急停保持在手边。', flush=True)
        period = 1.0 / max(1.0, float(cfg.rate))
        next_sample = time.monotonic()
        while time.monotonic() - started < float(cfg.max_duration):
            now = time.monotonic()
            if now >= next_sample:
                state = robot.state()
                live_path.parent.mkdir(parents=True, exist_ok=True)
                live_path.write_text(json.dumps({
                    'ok': state.q is not None,
                    'q': None if state.q is None else state.q.tolist(),
                    'tcp': None if state.tcp is None else state.tcp.tolist(),
                    'protective_stop': bool(state.protective_stop),
                    'extra': state.extra,
                }, ensure_ascii=False), encoding='utf-8')
                if state.q is not None and state.tcp is not None:
                    samples.append((state.q.copy(), state.tcp.copy()))
                next_sample = now + period
            key = keys.get()
            if key in ('s', '\r', '\n'):
                break
            if key in ('q', '\x1b'):
                print('放弃安全包络采集。', flush=True)
                return 2
            time.sleep(0.01)
        if len(samples) < max(3, int(cfg.min_samples)):
            print(f'样本不足：{len(samples)}，至少需要 {max(3, int(cfg.min_samples))} 个。')
            return 3
        envelope = SafetyEnvelope.from_samples(samples, cfg.margin_rad, cfg.tcp_margin_m)
        envelope.save(path)
        print(json.dumps(envelope.to_dict(), ensure_ascii=False, indent=2))
        print(f'已保存安全包络：{path}')
        return 0
    finally:
        keys.stop()
        if cfg.freedrive:
            try:
                robot.freedrive(False)
            except Exception:
                pass
        robot.disconnect()


