"""Passive episode sampling driven by an external leader input device.

Nothing is commanded: the backend is expected to be connected read-only, so the
loop only reads the robot state and any leader joints the backend exposes and
writes state/action/image frames.  Shared by ``examples.piper_collect`` and the
config-driven ``task_runner``.
"""
from __future__ import annotations

import time

from ..errors import DatasetError
from ..teach.console import KeyReader
from .schema import DatasetSchema
from .session import EpisodeCollector


def sample_episodes(robot, schema: DatasetSchema, root, *, cameras=(), export=None,
                    fps: float = 15.0, task: str = 'robot task',
                    auto_record: bool = False, sample_seconds: float = 0.0,
                    max_frames: int = 0, compress: bool = False, keys=None,
                    printer=print) -> dict:
    """Record episodes until ``q``/Ctrl-C.

    Keys: space (or ``s``) stops and saves the current episode (pressing it again
    starts the next one), ``x`` discards it, ``q`` quits.

    Recording starts as soon as the loop begins (``auto_record``, the default),
    so one command collects one demonstration like the UR5 flow; nothing is
    bounded unless ``sample_seconds`` or ``max_frames`` are set explicitly.
    """
    collector = EpisodeCollector(
        root, schema, robot=robot,
        cameras=() if not cameras else cameras,
        export=export,
        fps=fps,
        # The loop paces itself; a second throttle would halve the real rate.
        min_interval_s=0.0,
        max_frames=max_frames,
        compress=compress,
    )
    own_keys = keys is None
    if own_keys:
        keys = KeyReader().start()
    stats = {'episodes': 0, 'frames': 0, 'seconds': 0.0, 'saved': 0,
             'skipped': 0, 'dataset_root': collector.dataset_root}
    state = {'started_at': 0.0, 'deadline': None}

    def begin():
        collector.start(task, meta={
            'source': 'master_slave',
            'leader_present': robot.leader_joint_angles() is not None,
        })
        state['started_at'] = time.monotonic()
        if sample_seconds > 0 and state.get('deadline') is None:
            # Opening/warming up cameras can take seconds; start the window only
            # once recording actually begins.
            state['deadline'] = time.monotonic() + sample_seconds
        printer('\n开始录制', flush=True)

    def finish(success: bool = True):
        frames = collector.recorder.frames
        collector.stop(success=success)
        stats['episodes'] += 1
        stats['frames'] += frames
        stats['seconds'] += max(0.0, time.monotonic() - state['started_at'])
        printer(f'\nepisode {stats["episodes"]} {"保存" if success else "丢弃"}：{frames} 帧',
                flush=True)

    printer(f'相机：{schema.camera_names or "无"}  目标 {fps:g} Hz')
    try:
        if auto_record or sample_seconds > 0:
            # A configured duration implies hands-free recording.
            begin()
        period = 1.0 / max(fps, 1.0)
        next_at = time.monotonic()
        while True:
            deadline = state.get('deadline')
            if deadline is not None and time.monotonic() >= deadline:
                break
            key = keys.get()
            if key in (' ', 's'):
                if collector.active:
                    finish(True)
                else:
                    begin()
            elif key == 'x' and collector.active:
                finish(False)
            elif key == 'q':
                break
            sample = robot.state()
            if sample.q is None:
                time.sleep(0.02)
                continue
            if collector.active:
                grip = sample.extra.get('gripper_width')
                leader = robot.leader_joint_angles()
                action = None if leader is None else schema.observation(leader, grip)
                try:
                    collector.capture_state(state=sample, action=action)
                except DatasetError as exc:
                    stats['skipped'] += 1
                    if stats['skipped'] == 1:
                        printer(f'\n跳过没有图像的帧：{exc}', flush=True)
            next_at += period
            delay = next_at - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_at = time.monotonic()
    except KeyboardInterrupt:
        pass
    finally:
        if own_keys:
            keys.stop()
        if collector.active:
            try:
                finish(True)
            except Exception as exc:
                printer(f'\n保存失败：{type(exc).__name__}: {exc}', flush=True)
        collector.close()
        stats['dataset_root'] = collector.dataset_root
        stats['saved'] = collector.saved
        stats['export_errors'] = list(collector.export_errors)
        stats['rate'] = stats['frames'] / stats['seconds'] if stats['seconds'] > 0 else 0.0
    return stats
