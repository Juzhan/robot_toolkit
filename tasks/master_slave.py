"""Hardware-linked leader/follower sampling."""
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
from .paths import lerobot_root

from .build import (_build_cameras, _build_runtime, _delete_last_episode,
                        _exporter, _robot_speed)
from .paths import dataset_root


def _master_slave_session(task: TaskConfig, robot, cameras, schema, root, fps) -> int:
    """Button session over a passive leader/follower pair (Joy-Con or keyboard).

    B starts, A saves, Y twice discards, Home prints status; recording refuses
    to start while the leader is silent. Exports run after the session so they
    never compete with the capture loop.
    """
    from ..collect.passive import capture_leader_frame, leader_label, read_leader
    from ..teleop.announce import Announcer
    from ..teleop.session_input import DoublePress, SessionInput

    opts = task.master_slave
    voice = Announcer(opts.voice)
    confirm = DoublePress(task.session.double_press_timeout_s)
    collector = EpisodeCollector(
        root, schema, robot=robot, cameras=cameras,
        export=_exporter(task), dataset_root=lerobot_root(task), fps=fps,
        min_interval_s=0.0, max_frames=int(task.collect.max_frames or 0),
        compress=bool(task.collect.compress))
    stats = dict(saved=0, discarded=0)
    ep = {}
    ui = dict(line=False, dropped=0)

    def say(text, speech=None):
        print(('\n' if ui['line'] else '') + text, flush=True)
        ui['line'] = False
        if speech:
            voice.say(speech)

    def line(text):
        # Wide CJK text overflows fixed-width padding; clear to end of line instead.
        print('\r' + text + '\x1b[K', end='', flush=True)
        ui['line'] = True

    def episodes_in_dataset():
        info = collector.dataset_root / 'meta' / 'info.json'
        try:
            return int(json.loads(info.read_text(encoding='utf-8'))['total_episodes'])
        except Exception:
            return 0

    def status():
        leader = read_leader(robot, opts.leader_timeout_s)
        grip = f'{leader["gripper"] * 1000:.1f}mm' if leader['gripper_ok'] else '无'
        say(f'{state_line(robot)}\n主臂：{leader_label(leader)}，夹爪指令 {grip}\n'
            f'本次：保存 {stats["saved"]} 条，丢弃 {stats["discarded"]} 条，导出中 '
            f'{collector.recorder.status().queued} 条\n'
            f'数据集：{collector.dataset_root}（共 {episodes_in_dataset()} 条）',
            f'已保存 {stats["saved"]} 条，{leader_label(leader)}')

    inputs = None
    try:
        if collector.hub is not None:
            for camera in collector.hub:
                if not camera.opened:
                    camera.open()
            if not collector.hub.warmup():
                say('警告：相机 8 秒内未全部出图', '相机未就绪')
        inputs = SessionInput(joycon=opts.joycon).start()
        say(f'主从采集会话：{task.task}\n数据集：{collector.dataset_root}（已有 {episodes_in_dataset()} 条，本次追加）\n'
            'B 开始录制 | A 结束保存 | Y×2 丢弃当前 | +×2 删除上一条 | Home 查看状态 | Ctrl+C 退出\n'
            f'键盘同样可用：b / a / y / d / h，q 空闲时退出；语音{"开" if voice.available else "关"}',
            '主从采集就绪，按 B 开始录制')
        status()
        period = 1.0 / max(fps, 1.0)
        next_at = next_line = time.monotonic()
        exported = errors = 0
        while True:
            name = inputs.get(timeout=max(0.0, next_at - time.monotonic()))
            if inputs.dropped > ui['dropped']:
                ui['dropped'] = inputs.dropped
                say('程序卡顿期间的按键已忽略，请重新按。', '按键已忽略，请重按')
            if name == 'b':
                leader = read_leader(robot, opts.leader_timeout_s)
                if collector.active:
                    say('正在录制：A 保存，Y 按两次丢弃。', '正在录制')
                elif not leader['ok']:
                    say('未收到主臂指令：动一下主臂再按 B。', '未收到主臂指令')
                else:
                    confirm.reset()
                    collector.start(task.task, meta={'source': 'master_slave',
                                                      'leader_gripper_command': leader['gripper_ok']})
                    ep.update(started=time.monotonic(), leader=0, follower_gripper=0, no_leader=0)
                    say(f'● 开始录制第 {stats["saved"] + 1} 条', '开始录制')
            elif name == 'a':
                if not collector.active:
                    say('当前没有在录制：按 B 开始。', '没有在录制')
                elif collector.recorder.frames < 2:
                    say('这条太短，继续录或 Y 按两次丢弃。', '太短')
                else:
                    frames = collector.recorder.frames
                    seconds = time.monotonic() - ep['started']
                    collector.stop(success=True)
                    stats['saved'] += 1
                    grip = ('主臂指令' if not ep['follower_gripper'] else
                            f'从臂回退 {ep["follower_gripper"]} 帧')
                    say(f'■ 第 {stats["saved"]} 条已保存：{frames} 帧，{seconds:.1f}s，'
                        f'{frames / seconds:.1f} Hz；夹爪动作 {grip}；主臂缺失 {ep["no_leader"]} 帧',
                        f'已保存第 {stats["saved"]} 条')
            elif name == 'y':
                if not collector.active:
                    say('当前没有在录制。', '没有在录制')
                elif confirm.confirm('y'):
                    collector.discard()
                    stats['discarded'] += 1
                    say('已丢弃当前这条。', '已丢弃')
                else:
                    say('再按一次 Y 丢弃当前这条。', '再按一次丢弃')
            elif name == 'home':
                status()
                if not collector.active:
                    from ..runtime.link_check import run_link_check
                    _ok, summary = run_link_check(robot, cameras=cameras, log=say)
                    voice.say(summary)
            elif name == 'plus':
                if collector.active:
                    say('正在录制，不能删除。', '正在录制')
                elif confirm.confirm('plus'):
                    say('正在删除数据集最后一条……')
                    say(_delete_last_episode(task, collector), '删除完成')
                else:
                    say(f'再按一次 +（键盘 d），删除数据集最后一条（第 {episodes_in_dataset()} 条）。',
                        '再按一次删除上一条')
            elif name == 'quit':
                if collector.active:
                    say('正在录制：先 A 保存或 Y×2 丢弃。', '正在录制')
                else:
                    break
            now = time.monotonic()
            if now < next_at:
                continue
            next_at = max(next_at + period, now - period)
            if collector.active:
                if collector.recorder.phase == 'error':
                    say(f'录制出错：{collector.recorder.error}，已丢弃', '录制出错')
                    collector.discard()
                    stats['discarded'] += 1
                    continue
                try:
                    ep[capture_leader_frame(robot, collector, schema,
                                            max_age_s=opts.leader_timeout_s)] += 1
                except DatasetError as exc:
                    say(f'跳过一帧：{exc}')
            if collector.saved > exported:
                exported = collector.saved
                say(f'导出完成：数据集共 {episodes_in_dataset()} 条', '导出完成')
            if len(collector.export_errors) > errors:
                for error in collector.export_errors[errors:]:
                    say(f'导出失败：{error}', '导出失败，请看屏幕')
                errors = len(collector.export_errors)
            if now >= next_line:
                next_line = now + 1.0
                leader = read_leader(robot, opts.leader_timeout_s)
                arm = leader_label(leader)
                queued = collector.recorder.status().queued
                if collector.active:
                    seconds = now - ep['started']
                    frames = collector.recorder.frames
                    line(f'● 录制中 {seconds:5.1f}s {frames:5d} 帧 '
                         f'{frames / seconds if seconds > .5 else 0:4.1f} Hz | {arm}'
                         + (f' | 主臂缺失 {ep["no_leader"]}' if ep['no_leader'] else ''))
                else:
                    line(f'○ 待命（B 开始） | {arm} | 本次已保存 {stats["saved"]} 条'
                         + (f' | 导出中 {queued}' if queued else ''))
    except KeyboardInterrupt:
        pass
    finally:
        if collector.active:
            collector.discard()
            stats['discarded'] += 1
            say('中断：当前这条未保存。', '中断，未保存')
        if inputs is not None:
            inputs.stop()
        queued = collector.recorder.status().queued
        if queued:
            say(f'等待后台导出 {queued} 条……（原始数据已在磁盘上）', f'正在导出 {queued} 条')
        collector.close(timeout=3600.0)
        for error in collector.export_errors[errors:]:
            say(f'导出失败：{error}')
        say(f'会话结束：保存 {stats["saved"]} 条，丢弃 {stats["discarded"]} 条，'
            f'导出成功 {collector.saved} 条；数据集共 {episodes_in_dataset()} 条', '采集结束')
    return 0


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
    if task.master_slave.interactive and task.master_slave.sample_seconds <= 0:
        try:
            robot.connect()
            return _master_slave_session(task, robot, cameras, schema, root, fps)
        finally:
            robot.disconnect()
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


