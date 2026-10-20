"""Asynchronous episode recorder: capture cheaply, spool to disk, export later.

The capture loop must never block on disk or on a dataset writer, because a
stalled capture silently changes the timing of a demonstration.  Frames go to
a queue and are written as ``.npz`` files by a worker thread; the dataset
export runs in a second thread (or later, from the manifest).

Frames are written uncompressed by default: deflating a 640x480 RGB + depth
frame costs ~50 ms on a laptop CPU (the spool queue then drains at ~20 fps and a
30 Hz episode aborts with "write queue is full") while saving only a few percent.
Pass ``compress=True`` when disk space matters more than sustained frame rate.
"""

from __future__ import annotations

import json
import queue
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from ..errors import DatasetError
from .schema import DatasetSchema


def write_json(path, value) -> None:
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


@dataclass
class EpisodeRecorderStatus:
    phase: str
    frames: int
    seconds: float
    saved: int
    error: str
    path: str
    fps: float
    export_mode: str
    queued: int
    processing: int
    export_errors: list

    def as_dict(self) -> dict:
        return dict(self.__dict__)


class EpisodeRecorder:
    def __init__(self, root, schema: DatasetSchema, export=None,
                 export_mode: str = 'background', fps: float | None = None,
                 max_frames: int = 6000, queue_max: int = 64,
                 compress: bool = False, clock=time.monotonic):
        """``max_frames`` caps one episode; ``0`` means no frame cap.

        ``compress=False`` (default) stores raw ``.npz`` frames so the spool can
        keep up with a 30 Hz camera; compression is ~20x slower per frame.
        """
        if export_mode not in ('background', 'deferred'):
            raise ValueError("export_mode must be 'background' or 'deferred'")
        self.root = Path(root)
        self.schema = schema
        self.export = export
        self.export_mode = export_mode
        self.fps = float(fps or schema.fps)
        self.max_frames = int(max_frames)
        self.queue_max = int(queue_max)
        self.compress = bool(compress)
        self.clock = clock
        self.session = self.root / (datetime.now().strftime('%Y%m%d_%H%M%S') + '_' +
                                    uuid.uuid4().hex[:6])
        self.dataset_root = self.session / 'dataset'
        self.session.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._queue: queue.Queue = queue.Queue()
        self._export_queue: queue.Queue = queue.Queue()
        self.phase = 'idle'
        self.frames = 0
        self.saved = 0
        self.error = ''
        self.take: Path | None = None
        self.pending: list[Path] = []
        self.processing = 0
        self.unfinished = 0
        self.export_errors: list[str] = []
        self._bad_takes: set[Path] = set()
        self._closed = False
        self._writer = threading.Thread(target=self._write_loop, daemon=True,
                                        name='episode-writer')
        self._writer.start()
        self._exporter = None
        if export is not None and export_mode == 'background':
            self._exporter = threading.Thread(target=self._export_loop, daemon=True,
                                              name='episode-exporter')
            self._exporter.start()

    # ----------------------------------------------------------------- status
    def status(self) -> EpisodeRecorderStatus:
        with self._lock:
            return EpisodeRecorderStatus(
                phase=self.phase, frames=self.frames,
                seconds=round(self.frames / max(self.fps, 1e-9), 2),
                saved=self.saved, error=self.error, path=str(self.dataset_root),
                fps=self.fps, export_mode=self.export_mode,
                queued=self.unfinished, processing=self.processing,
                export_errors=list(self.export_errors))

    # ---------------------------------------------------------------- control
    def start(self, task: str, meta: dict | None = None) -> Path:
        with self._lock:
            if self.phase != 'idle':
                raise RuntimeError('finish or discard the current episode first')
            task = (task or '').strip()
            if not task:
                raise ValueError('task description must not be empty')
            if len(task) > 300:
                raise ValueError('task description must be at most 300 characters')
            self.take = self.session / ('take_' + uuid.uuid4().hex[:10])
            self.take.mkdir()
            self.frames = 0
            self.error = ''
            self.phase = 'recording'
            metadata = dict(meta or {})
            metadata.update(task=task, fps=self.fps, schema=self.schema.to_dict(),
                            robot=self.schema.robot.name, status='recording',
                            success=None, frames=0,
                            created=datetime.now().isoformat(timespec='seconds'))
            write_json(self.take / 'take.json', metadata)
            return self.take

    def capture(self, observation_state, action, images=None, depth=None,
                extra=None, timestamp: float | None = None) -> bool:
        with self._lock:
            if self.phase != 'recording' or self.take is None:
                return False
            if self.max_frames > 0 and self.frames >= self.max_frames:
                self.error = 'maximum episode length reached'
                self.phase = 'error'
                return False
            if self._queue.qsize() >= self.queue_max:
                self.error = 'write queue is full; episode stopped'
                self.phase = 'error'
                return False
            state = np.asarray(observation_state, dtype=np.float32).reshape(-1)
            action_array = np.asarray(action, dtype=np.float32).reshape(-1)
            if state.size != self.schema.dim or action_array.size != self.schema.dim:
                raise DatasetError(f'state/action must have {self.schema.dim} values')
            sample = {
                'observation_state': state,
                'action': action_array,
                'timestamp': np.array([float(timestamp if timestamp is not None
                                             else self.clock())], dtype=np.float64),
            }
            if self.schema.require_images:
                missing = [camera.name for camera in self.schema.cameras
                           if images is None or images.get(camera.name) is None]
                if missing:
                    raise DatasetError('missing images for cameras: ' + ', '.join(missing))
            for camera in self.schema.cameras:
                image = None if images is None else images.get(camera.name)
                if image is not None:
                    if tuple(image.shape) != (camera.height, camera.width, camera.channels):
                        raise DatasetError(
                            f'camera {camera.name}: {image.shape} != '
                            f'{(camera.height, camera.width, camera.channels)}')
                    sample[f'{camera.name}_rgb'] = np.asarray(image, dtype=np.uint8)
                depth_map = None if depth is None else depth.get(camera.name)
                if depth_map is not None:
                    sample[f'{camera.name}_depth_mm'] = np.asarray(depth_map)
                    sample[f'{camera.name}_K'] = np.asarray(depth.get(f'{camera.name}_K', np.eye(3)))
                    sample[f'{camera.name}_T_world_camera'] = np.asarray(
                        depth.get(f'{camera.name}_T_world_camera', np.eye(4)))
            if extra:
                for key, value in extra.items():
                    sample[key] = np.asarray(value)
            index = self.frames
            self.frames += 1
            self._queue.put_nowait(('frame', self.take, (index, sample)))
            return True

    def save(self, success: bool | None = None) -> None:
        with self._lock:
            if self.phase not in ('recording', 'error'):
                raise RuntimeError('no episode is being recorded')
            if self.phase == 'error':
                raise DatasetError(self.error or 'recording failed')
            if self.frames < 2:
                raise DatasetError('episode is too short (need at least two frames)')
            take = self.take
            meta = json.loads((take / 'take.json').read_text(encoding='utf-8'))
            meta.update(success=success, status='spooling', frames=self.frames)
            write_json(take / 'take.json', meta)
            self.unfinished += 1
            self._queue.put_nowait(('save', take, None))
            self.phase = 'idle'
            self.take = None
            self.error = ''

    def discard(self) -> None:
        with self._lock:
            if self.phase not in ('recording', 'error'):
                raise RuntimeError('no episode is being recorded')
            take = self.take
            self._queue.put_nowait(('discard', take, None))
            self.phase = 'idle'
            self.take = None
            self.error = ''

    # ---------------------------------------------------------------- workers
    def _write_loop(self) -> None:
        while True:
            operation, take, payload = self._queue.get()
            if operation == 'stop':
                self._queue.task_done()
                return
            try:
                if operation == 'frame':
                    index, sample = payload
                    writer = np.savez_compressed if self.compress else np.savez
                    writer(take / f'{index:06d}.npz', **sample)
                elif operation == 'save':
                    meta = json.loads((take / 'take.json').read_text(encoding='utf-8'))
                    if take in self._bad_takes:
                        raise DatasetError('some frames failed to write')
                    meta['status'] = 'queued'
                    write_json(take / 'take.json', meta)
                    with self._lock:
                        self.pending.append(take)
                    self._schedule(take)
                elif operation == 'discard':
                    meta = json.loads((take / 'take.json').read_text(encoding='utf-8'))
                    meta['status'] = 'discarded'
                    write_json(take / 'take.json', meta)
            except Exception as exc:
                with self._lock:
                    self._bad_takes.add(take)
                    if take == self.take and self.phase == 'recording':
                        self.error = str(exc)
                        self.phase = 'error'
                    if operation == 'save':
                        self.unfinished -= 1
                        self.export_errors.append(f'{take.name}: {exc}')
                self._mark_failed(take, exc)
            finally:
                self._queue.task_done()

    def _mark_failed(self, take: Path, exc: Exception) -> None:
        try:
            path = take / 'take.json'
            meta = json.loads(path.read_text(encoding='utf-8'))
            meta.update(status='failed', error=str(exc))
            write_json(path, meta)
        except Exception:
            pass

    def _schedule(self, take: Path) -> None:
        if self.export is None:
            return
        if self.export_mode == 'background' and self._exporter is not None:
            self._export_queue.put_nowait(take)

    def _export_loop(self) -> None:
        while True:
            take = self._export_queue.get()
            try:
                self._run_export(take)
            finally:
                self._export_queue.task_done()

    def _run_export(self, take: Path) -> None:
        with self._lock:
            self.processing += 1
        try:
            meta = json.loads((take / 'take.json').read_text(encoding='utf-8'))
            meta['status'] = 'exporting'
            write_json(take / 'take.json', meta)
            if callable(self.export):
                self.export(take, self.dataset_root)
            else:
                self.export.export(take, self.dataset_root)
            meta['status'] = 'saved'
            write_json(take / 'take.json', meta)
            with self._lock:
                self.saved += 1
        except Exception as exc:
            self._mark_failed(take, exc)
            with self._lock:
                self.export_errors.append(f'{take.name}: {exc}')
        finally:
            with self._lock:
                self.processing = max(0, self.processing - 1)
                self.unfinished = max(0, self.unfinished - 1)

    def process_pending(self) -> int:
        """Run deferred exports synchronously; returns the number attempted."""
        with self._lock:
            takes = list(self.pending)
            self.pending.clear()
        for take in takes:
            self._run_export(take)
        return len(takes)

    def flush(self) -> None:
        """Wait until every captured frame is on disk."""
        self._queue.join()

    # ----------------------------------------------------------------- shutdown
    def close(self, wait_export: bool = True, timeout: float = 60.0) -> None:
        if self._closed:
            return
        self._closed = True
        if self.phase in ('recording', 'error'):
            try:
                self.save() if self.phase == 'recording' and self.frames >= 2 else self.discard()
            except Exception:
                pass
        self._queue.join()
        self._queue.put_nowait(('stop', None, None))
        if self._writer is not None:
            self._writer.join(timeout=2.0)
        if wait_export and self._exporter is not None:
            self._export_queue.join()
        elif self.export_mode == 'deferred':
            self.process_pending()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
        return False
