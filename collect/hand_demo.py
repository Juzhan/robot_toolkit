"""Camera-only collection of human demonstrations.

Uses the same ``DatasetSchema`` / ``EpisodeCollector`` / ``export_lerobot``
pipeline as robot collection, with ``include_state_action=False``.  No
``observation.state`` or ``action`` column is written; LeRobot features contain
only ``observation.images.*``.

Hand tracking is reserved as an extension point.  A :class:`HandTracker` is
offered every recorded frame and its payload is stored as ``tracker_*`` extra
fields in the spool.  Pass ``tracker=None`` (the default) for video only.

.. note::
   ``HandDemoCollector`` and the mp4 sink below are the first lightweight
   implementation and are no longer exported or used by ``record_hand_demos``.
   They are kept only for old callers; new code uses ``EpisodeCollector``.
"""
from __future__ import annotations

import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Protocol

import numpy as np

from ..camera import CameraHub
from ..errors import DatasetError
from ..teach.console import KeyReader
from .export import delete_last_episode
from .recorder import write_json
from .schema import CameraSpec, DatasetSchema
from .session import EpisodeCollector


#: Two Y presses within this window delete the newest saved clip (old session
#: used the same 8 s confirmation window).
DELETE_CONFIRM_S = 8.0


def _dataset_episode_count(dataset_root) -> int:
    """Actual episode count on disk, independent of background export."""
    root = Path(dataset_root)
    info = root / 'meta' / 'info.json'
    if info.exists():
        try:
            return int(json.loads(info.read_text(encoding='utf-8'))
                       .get('total_episodes', 0) or 0)
        except Exception:
            return 0
    manifest = root / 'episodes.jsonl'
    if manifest.exists():
        return sum(1 for line in manifest.read_text(encoding='utf-8').splitlines()
                   if line.strip())
    return 0


class HandTracker(Protocol):
    """Optional per-frame tracker channel (e.g. hand landmarks).

    ``read`` receives the episode-relative timestamp and the RGB frames of all
    cameras.  Return a JSON-serialisable mapping to append one row to
    ``tracking.jsonl``, or ``None`` to skip the frame.  Trackers are called from
    the recording loop, so they should return quickly or fan out internally.
    """

    name: str

    def read(self, timestamp: float, images: dict[str, np.ndarray]) -> dict | None:
        ...


def _jsonable(value):
    """Convert tracker payloads (numpy included) to plain JSON values."""
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _opencv():
    try:
        import cv2
    except ImportError:
        return None
    return cv2


_CODEC_ALIASES = {'h264': 'avc1', 'h265': 'hev1', 'mjpg': 'MJPG'}


def _fourcc(cv2, codec: str):
    value = _CODEC_ALIASES.get(str(codec).strip().lower(), str(codec).strip())
    value = (value or 'mp4v').ljust(4)[:4]
    return cv2.VideoWriter_fourcc(*value)


class _Mp4Sink:
    kind = 'mp4'

    def __init__(self, path: Path, width: int, height: int, fps: float, codec: str):
        cv2 = _opencv()
        if cv2 is None:
            raise DatasetError('writing mp4 videos requires `pip install opencv-python`')
        self._cv2 = cv2
        self.width, self.height = int(width), int(height)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = cv2.VideoWriter(str(path), _fourcc(cv2, codec),
                                       float(fps), (self.width, self.height))
        if not self._writer.isOpened():
            self._writer.release()
            raise DatasetError(f'cannot open video writer for {path.name} '
                               f'({self.width}x{self.height} @ {fps:g} fps)')

    def write(self, rgb: np.ndarray) -> None:
        if rgb.shape[1] != self.width or rgb.shape[0] != self.height:
            rgb = self._cv2.resize(rgb, (self.width, self.height),
                                   interpolation=self._cv2.INTER_AREA)
        self._writer.write(self._cv2.cvtColor(rgb, self._cv2.COLOR_RGB2BGR))

    def close(self) -> None:
        self._writer.release()


class _NpzFrameSink:
    """Fallback for machines without OpenCV: one ``.npz`` per frame."""

    kind = 'npz-frames'

    def __init__(self, directory: Path, width: int, height: int, fps: float,
                 codec: str):
        self.width, self.height = int(width), int(height)
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.index = 0

    def write(self, rgb: np.ndarray) -> None:
        if rgb.shape[1] != self.width or rgb.shape[0] != self.height:
            rows = np.linspace(0, rgb.shape[0] - 1, self.height).astype(int)
            columns = np.linspace(0, rgb.shape[1] - 1, self.width).astype(int)
            rgb = rgb[rows][:, columns]
        np.savez(self.directory / f'{self.index:06d}.npz', rgb=rgb)
        self.index += 1

    def close(self) -> None:
        pass


def _create_sink(take: Path, name: str, width: int, height: int, fps: float,
                 codec: str):
    """Return ``(sink, path relative to the take)`` for one camera."""
    if _opencv() is not None:
        relative = f'video/{name}.mp4'
        return _Mp4Sink(take / relative, width, height, fps, codec), relative
    relative = f'video/{name}'
    return _NpzFrameSink(take / relative, width, height, fps, codec), relative


class HandDemoCollector:
    """Record video-only human-demo episodes.

    Cameras are opened lazily by :meth:`start`; every episode is one take with
    one video stream per camera.  No state or action vector is written.  The
    optional ``tracker`` hook adds a ``tracking.jsonl`` channel but is unused by
    default.
    """

    def __init__(self, root, cameras=(), fps: float = 30.0, max_frames: int = 0,
                 codec: str = 'mp4v', tracker: HandTracker | None = None,
                 clock=time.monotonic):
        specs = tuple(CameraSpec.from_camera(camera) for camera in cameras)
        if not specs:
            raise ValueError('hand demo collection needs at least one camera')
        self.hub = CameraHub(cameras)
        self.spec_cameras = specs
        if float(fps) <= 0:
            raise ValueError('fps must be greater than 0')
        self.fps = float(fps)
        self.max_frames = max(0, int(max_frames))
        self.codec = str(codec or 'mp4v')
        self.tracker = tracker
        self.tracking_name = (getattr(tracker, 'name', type(tracker).__name__)
                              if tracker is not None else '')
        self.clock = clock
        self.format = 'mp4' if _opencv() is not None else 'npz-frames'
        self.session = Path(root) / (datetime.now().strftime('%Y%m%d_%H%M%S') + '_' +
                                     uuid.uuid4().hex[:6])
        self.dataset_root = self.session / 'dataset'
        self.session.mkdir(parents=True, exist_ok=True)
        self.phase = 'idle'
        self.take: Path | None = None
        self.frames = 0
        self.skipped = 0
        self.saved = 0
        self.tracker_errors = 0
        self.tracking_rows = 0
        self.export_errors: list[str] = []
        self.limit_reached = False
        self._sinks: dict[str, object] = {}
        self._files: dict[str, str] = {}
        self._sizes: dict[str, tuple[int, int]] = {}
        self._frames_file = None
        self._tracking_file = None
        self._tracking_error = ''
        self._started_at = 0.0
        self._closed = False

    # ----------------------------------------------------------------- status
    @property
    def active(self) -> bool:
        return self.phase == 'recording'

    # ---------------------------------------------------------------- control
    def start(self, task: str, meta: dict | None = None) -> Path:
        if self.phase != 'idle':
            raise RuntimeError('finish or discard the current episode first')
        task = (task or '').strip()
        if not task:
            raise ValueError('task description must not be empty')
        if len(task) > 300:
            raise ValueError('task description must be at most 300 characters')
        for camera in self.hub:
            if not camera.opened:
                camera.open()
        self.hub.warmup()
        self.take = self.session / ('take_' + uuid.uuid4().hex[:10])
        self.take.mkdir()
        self.frames = 0
        self.skipped = 0
        self.limit_reached = False
        self.tracker_errors = 0
        self.tracking_rows = 0
        self._tracking_error = ''
        self._sinks = {}
        self._files = {}
        self._sizes = {}
        self._frames_file = (self.take / 'frames.jsonl').open('w', encoding='utf-8')
        self._tracking_file = ((self.take / 'tracking.jsonl').open('w', encoding='utf-8')
                               if self.tracker is not None else None)
        self._started_at = self.clock()
        self.phase = 'recording'
        metadata = dict(meta or {})
        metadata.update(task=task, fps=self.fps, status='recording',
                        created=datetime.now().isoformat(timespec='seconds'),
                        frames=0, kind='hand_demo')
        write_json(self.take / 'take.json', metadata)
        return self.take

    def capture(self, frames=None, timestamp: float | None = None) -> bool:
        """Write one frame from every camera; ``False`` means it was skipped."""
        if self.phase != 'recording' or self.take is None:
            return False
        if self.max_frames > 0 and self.frames >= self.max_frames:
            self.limit_reached = True
            return False
        if frames is None:
            frames = self.hub.snapshot(strict=False)
        images: dict[str, np.ndarray] = {}
        for camera in self.spec_cameras:
            frame = frames.get(camera.name)
            rgb = None if frame is None else frame.rgb
            if rgb is None:
                self.skipped += 1
                return False
            images[camera.name] = np.ascontiguousarray(rgb)
        now = self.clock()
        relative_t = now - self._started_at if timestamp is None else float(timestamp)
        for camera in self.spec_cameras:
            sink = self._sinks.get(camera.name)
            if sink is None:
                rgb = images[camera.name]
                height, width = int(rgb.shape[0]), int(rgb.shape[1])
                sink, relative = _create_sink(self.take, camera.name, width,
                                              height, self.fps, self.codec)
                self._sinks[camera.name] = sink
                self._files[camera.name] = relative
                self._sizes[camera.name] = (width, height)
            sink.write(images[camera.name])
        index = self.frames
        self.frames += 1
        if self._frames_file is not None:
            self._frames_file.write(json.dumps(
                {'frame': index, 't': round(relative_t, 6)}) + '\n')
            if index % 30 == 29:
                self._frames_file.flush()
        self._track(index, relative_t, images)
        return True

    def stop(self, success: bool | None = True) -> dict | None:
        """Close the videos and archive the take; returns the export summary."""
        if self.phase != 'recording' or self.take is None:
            raise RuntimeError('no episode is being recorded')
        take = self.take
        self._close_streams()
        frames = self.frames
        if frames < 2:
            self._finish_take(take, 'discarded', success=success)
            self._reset()
            raise DatasetError('episode is too short (need at least two frames)')
        duration = max(0.0, self.clock() - self._started_at)
        meta = json.loads((take / 'take.json').read_text(encoding='utf-8'))
        meta.update(status='exporting', success=success, frames=frames,
                    duration_s=round(duration, 3), skipped=self.skipped,
                    tracker_errors=self.tracker_errors,
                    video={
                        'format': self.format,
                        'fps': self.fps,
                        'codec': self.codec if self.format == 'mp4' else '',
                        'cameras': [{
                            'name': camera.name,
                            'width': self._sizes[camera.name][0],
                            'height': self._sizes[camera.name][1],
                            'file': self._files[camera.name],
                        } for camera in self.spec_cameras],
                    },
                    tracking=({
                        'name': self.tracking_name,
                        'file': 'tracking.jsonl',
                        'frames': self.tracking_rows,
                    } if self.tracker is not None else None),
                    contract=self._contract(meta.get('task', '')))
        write_json(take / 'take.json', meta)
        try:
            raise DatasetError('legacy HandDemoCollector is no longer supported; '
                               'use EpisodeCollector with a vision-only schema')
        except Exception as exc:
            meta.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            write_json(take / 'take.json', meta)
            self.export_errors.append(f'{take.name}: {exc}')
            summary = None
        finally:
            self._reset()
        return summary

    def discard(self) -> None:
        if self.phase != 'recording' or self.take is None:
            raise RuntimeError('no episode is being recorded')
        take = self.take
        self._close_streams()
        self._finish_take(take, 'discarded', success=False)
        self._reset()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.phase == 'recording':
            try:
                if self.frames >= 2:
                    self.stop(success=True)
                else:
                    self.discard()
            except Exception:
                self._reset()
        self.hub.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
        return False

    # --------------------------------------------------------------- tracking
    def _track(self, index: int, timestamp: float, images: dict[str, np.ndarray]) -> None:
        if self.tracker is None or self._tracking_file is None or self._tracking_error:
            return
        try:
            payload = self.tracker.read(timestamp, images)
        except Exception as exc:
            self.tracker_errors += 1
            self._tracking_error = f'{type(exc).__name__}: {exc}'
            return
        if payload is None:
            return
        row = {'frame': index, 't': round(timestamp, 6)}
        if isinstance(payload, dict):
            row.update(_jsonable(payload))
        else:
            row['value'] = _jsonable(payload)
        self._tracking_file.write(json.dumps(row, ensure_ascii=False) + '\n')
        self.tracking_rows += 1
        if index % 30 == 29:
            self._tracking_file.flush()

    # ---------------------------------------------------------------- helpers
    def _contract(self, task: str) -> dict:
        return {
            'toolkit': 'robot_toolkit',
            'kind': 'hand_demo_video',
            'task': task,
            'state_action': None,
            'description': 'video-only human demonstration: no robot state or action',
            'fps': self.fps,
            'videos': {
                'format': self.format,
                'codec': self.codec if self.format == 'mp4' else None,
                'layout': 'one video stream per camera',
            },
            'cameras': [{'name': camera.name,
                         'width': self._sizes[camera.name][0],
                         'height': self._sizes[camera.name][1],
                         'file': self._files[camera.name]}
                        for camera in self.spec_cameras],
            'frame_timestamps': 'frames.jsonl: {"frame": int, "t": seconds from episode start}',
            'tracking': ({'name': self.tracking_name,
                          'file': 'tracking.jsonl',
                          'channels': 'reserved; empty when no tracker is attached'}
                         if self.tracker is not None else None),
        }

    def _finish_take(self, take: Path, status: str, **extra) -> None:
        try:
            meta = json.loads((take / 'take.json').read_text(encoding='utf-8'))
            meta.update(status=status, frames=self.frames, **extra)
            write_json(take / 'take.json', meta)
        except Exception:
            pass

    def _close_streams(self) -> None:
        for sink in self._sinks.values():
            try:
                sink.close()
            except Exception as exc:
                self.export_errors.append(f'video writer: {exc}')
        self._sinks = {}
        for handle in (self._frames_file, self._tracking_file):
            if handle is not None:
                try:
                    handle.close()
                except Exception:
                    pass
        self._frames_file = None
        self._tracking_file = None

    def _reset(self) -> None:
        self.phase = 'idle'
        self.take = None
        self.frames = 0
        self.skipped = 0
        self.limit_reached = False
        self.tracking_rows = 0
        self._sinks = {}
        self._files = {}
        self._sizes = {}
        self._tracking_error = ''


def record_hand_demos(cameras, root, schema: DatasetSchema, *, fps: float | None = None,
                      task: str = 'human demonstration',
                      auto_record: bool = False, sample_seconds: float = 0.0,
                      max_frames: int = 0, export=None,
                      tracker: HandTracker | None = None, keys=None,
                      announce=None, printer=print) -> dict:
    """Record vision-only episodes through the shared ``EpisodeCollector``.

    ``schema`` must have ``include_state_action=False``.  Spooling, export and
    deletion are the same components used by robot collection; only the
    state/action modality is switched off.
    """
    if schema.include_state_action:
        raise ValueError('record_hand_demos needs a vision-only schema')
    if not cameras:
        raise ValueError('hand demo collection needs at least one camera')
    fps = float(fps or schema.fps)
    task = (task or 'human demonstration').strip()
    collector = EpisodeCollector(
        root, schema, robot=None, cameras=cameras, export=export,
        fps=fps, min_interval_s=0.0, max_frames=max_frames)
    own_keys = keys is None
    if own_keys:
        keys = KeyReader().start()
    stats = {'episodes': 0, 'frames': 0, 'seconds': 0.0, 'saved': 0,
             'skipped': 0, 'dataset_root': collector.dataset_root}
    state = {'started_at': 0.0, 'deadline': None, 'delete_until': 0.0,
             'cancel_until': 0.0}

    def say(text: str) -> None:
        if announce is None:
            return
        try:
            announce(text, preempt=True)
        except TypeError:
            announce(text)
        except Exception:
            pass

    def frame_hook(timestamp, images):
        if tracker is None:
            return None
        try:
            payload = tracker.read(timestamp, images)
        except Exception as exc:
            printer(f'\n手部跟踪失败：{type(exc).__name__}: {exc}', flush=True)
            return None
        if not isinstance(payload, dict):
            return None
        return {f'tracker_{key}': value for key, value in payload.items()}

    def begin():
        collector.start(task, meta={'source': 'hand_demo'})
        state['started_at'] = time.monotonic()
        if sample_seconds > 0 and state.get('deadline') is None:
            state['deadline'] = time.monotonic() + sample_seconds
        printer('\n开始录制人手演示（仅视频）', flush=True)
        say('开始录制')

    def finish(success: bool = True):
        frames = collector.recorder.frames
        index = stats['episodes']
        if not success:
            collector.discard()
            printer('\n已丢弃当前片段', flush=True)
            say('已丢弃')
            return
        try:
            collector.stop(success=True)
        except Exception as exc:
            printer(f'\n当前片段未保存：{type(exc).__name__}: {exc}', flush=True)
            say('片段未保存')
            return
        stats['episodes'] += 1
        stats['frames'] += frames
        stats['seconds'] += max(0.0, time.monotonic() - state['started_at'])
        printer(f'\nepisode {index} 录制完成，正在导出：{frames} 帧 -> '
                f'{collector.dataset_root}', flush=True)
        say(f'第 {index + 1} 条录制完成')

    printer(f'仅视频人手演示：相机={collector.hub.names if collector.hub else []}  '
            f'fps={fps:g}  导出={getattr(export, "__name__", export) or "无"}')
    try:
        if auto_record or sample_seconds > 0:
            begin()
        period = 1.0 / max(float(fps), 1.0)
        next_at = time.monotonic()
        while True:
            deadline = state.get('deadline')
            if deadline is not None and time.monotonic() >= deadline:
                break
            # Lets an input adapter know whether a clip is running, so Joy-Con
            # B is start-only and A is save-only (the session-mode semantics).
            sync = getattr(keys, 'sync', None)
            if callable(sync):
                sync(collector.active)
            key = keys.get()
            if key in (' ', 's'):
                if collector.active:
                    finish(True)
                else:
                    begin()
            elif key == 'x' and collector.active:
                finish(False)
            elif key == 'cancel':
                now = time.monotonic()
                if not collector.active:
                    printer('\n当前没有录制，无法取消', flush=True)
                    say('当前没有录制')
                elif now > state.get('cancel_until', 0.0):
                    state['cancel_until'] = now + DELETE_CONFIRM_S
                    printer('\n再按一次 Y，丢弃当前片段', flush=True)
                    say('再按一次 Y，丢弃当前片段')
                else:
                    state['cancel_until'] = 0.0
                    finish(False)
            elif key in ('delete', 'd'):
                now = time.monotonic()
                if now > state.get('delete_until', 0.0):
                    try:
                        collector.recorder.wait_export()
                    except Exception:
                        pass
                    count = _dataset_episode_count(collector.dataset_root)
                    if count <= 0:
                        printer('\n没有可删除的已保存片段', flush=True)
                        say('没有可删除的片段')
                    else:
                        state['delete_until'] = now + DELETE_CONFIRM_S
                        printer(f'\n再按一次删除，删除最后一条（共 {count} 条）',
                                flush=True)
                        say(f'再按一次删除，共 {count} 条')
                else:
                    state['delete_until'] = 0.0
                    try:
                        collector.recorder.wait_export()
                        entry = delete_last_episode(collector.dataset_root)
                    except Exception as exc:
                        printer(f'\n删除失败：{type(exc).__name__}: {exc}', flush=True)
                        say('删除失败')
                    else:
                        if entry is None:
                            printer('\n没有可删除的已保存片段', flush=True)
                            say('没有可删除的片段')
                        else:
                            remaining = entry.get('remaining')
                            if remaining is None:
                                remaining = _dataset_episode_count(
                                    collector.dataset_root)
                            collector.recorder.saved = max(0, int(remaining))
                            stats['episodes'] = max(0, stats['episodes'] - 1)
                            printer(f'\n已删除 episode {entry.get("episode")}', flush=True)
                            say('已删除最后一条')
            elif key == 'q':
                break
            elif key == 'status':
                text = (f'{"录制中" if collector.active else "空闲"}，'
                        f'已保存 {collector.saved} 条')
                printer(f'\n状态：{text}  frames={collector.recorder.frames}', flush=True)
                say(text)
            if collector.active:
                if max_frames > 0 and collector.recorder.frames >= max_frames:
                    finish(True)
                    printer(f'达到单条上限 {max_frames} 帧，结束采集。', flush=True)
                    break
                captured = collector.capture_frame(frame_hook=frame_hook)
                if collector.recorder.phase == 'error':
                    printer(f'\n录制停止：{collector.recorder.error}', flush=True)
                    try:
                        collector.discard()
                    except Exception:
                        pass
                    break
                if not captured:
                    stats['skipped'] += 1
                    if stats['skipped'] == 1:
                        printer('\n相机暂时没有帧，跳过（检查连接/曝光）', flush=True)
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
        stats['rate'] = (stats['frames'] / stats['seconds']
                         if stats['seconds'] > 0 else 0.0)
        say('采集结束')
    return stats
