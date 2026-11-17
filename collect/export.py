"""Turn a recorded take into a dataset.

Two exporters are provided:

``export_lerobot``
    Writes the standard LeRobot layout using LeRobot's own writer, including a
    depth side-car that mirrors the existing simulation datasets.
``export_manifest``
    Dependency-free fallback that appends to ``episodes.jsonl``.  Useful for
    dry runs, tests, and machines without LeRobot installed.
"""

from __future__ import annotations

import gc
import inspect
import json
import shutil
import time
import uuid
from pathlib import Path

import numpy as np

from ..core.conventions import CAMERA_FRAME_CONVENTION, DEPTH_ENCODING, EXTRINSIC_CONVENTION
from ..errors import DatasetError
from .schema import DatasetSchema


def _read_meta(take: Path) -> dict:
    return json.loads((take / 'take.json').read_text(encoding='utf-8'))


def _schema_from_meta(meta: dict, schema) -> DatasetSchema:
    if schema is not None:
        return schema
    if 'schema' not in meta:
        raise DatasetError('take.json has no schema and no schema was passed')
    return DatasetSchema.from_dict(meta['schema'])


def _frame_files(take: Path) -> list[Path]:
    files = sorted(path for path in take.glob('[0-9]*.npz') if path.name[:1].isdigit())
    if len(files) < 2:
        raise DatasetError(f'{take.name}: need at least two frames, found {len(files)}')
    return files


class FrameReader:
    """Read spooled rows, resolving ``<camera>_ref`` image references.

    A row that repeats an earlier camera frame carries no pixels, only the index
    of the row that stored them; ``read(i)`` returns every row complete (rgb and
    depth/K/T for each camera), so exporters see one image per row as before.
    """

    _PIXELS = ('rgb', 'depth_mm', 'K', 'T_world_camera')

    def __init__(self, files, cameras):
        self.files = list(files)
        self.cameras = [camera.name for camera in cameras]
        self._cache = {}          # camera -> (row index, {suffix: array})

    def _pixels(self, camera, row):
        cached = self._cache.get(camera)
        if cached is not None and cached[0] == row:
            return cached[1]
        with np.load(self.files[row], allow_pickle=False) as raw:
            if f'{camera}_rgb' not in raw:
                raise DatasetError(f'row {row} is referenced for {camera} but has no image')
            pixels = {suffix: raw[f'{camera}_{suffix}'] for suffix in self._PIXELS
                      if f'{camera}_{suffix}' in raw}
        self._cache[camera] = (row, pixels)
        return pixels

    def stored(self, index, camera) -> bool:
        """True if row ``index`` holds ``camera``'s pixels itself (not a reference)."""
        with np.load(self.files[index], allow_pickle=False) as raw:
            return f'{camera}_rgb' in raw

    def read(self, index) -> dict:
        with np.load(self.files[index], allow_pickle=False) as raw:
            row = {key: raw[key] for key in raw.files}
        for camera in self.cameras:
            key = f'{camera}_ref'
            if key not in row:
                continue
            source = int(row[key][0])
            if not 0 <= source <= index:
                raise DatasetError(f'row {index}: invalid {camera} reference {source}')
            if source != index:
                for suffix, value in self._pixels(camera, source).items():
                    row[f'{camera}_{suffix}'] = value
        return row


def _filter_kwargs(function, **kwargs) -> dict:
    """Pass only the keywords this LeRobot version understands.

    The API changed between 0.4 (``vcodec=``) and 0.6 (``use_videos=`` /
    ``rgb_encoder=``), and both are in use across machines.
    """
    try:
        parameters = inspect.signature(function).parameters
    except (TypeError, ValueError):
        return kwargs
    if any(parameter.kind is inspect.Parameter.VAR_KEYWORD
           for parameter in parameters.values()):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in parameters}


def _install_retry() -> None:

    import shutil as shutil_module

    if getattr(shutil_module.rmtree, '_toolkit_retry', False):
        return
    original = shutil_module.rmtree

    def rmtree_retry(path, *args, **kwargs):
        for attempt in range(12):
            try:
                return original(path, *args, **kwargs)
            except PermissionError:
                if attempt == 11:
                    raise
                import time
                time.sleep(0.5)

    rmtree_retry._toolkit_retry = True
    shutil_module.rmtree = rmtree_retry


def _rename_retry(source: Path, target: Path, attempts: int = 12) -> None:
    """Rename a directory even if Windows still holds a short-lived handle."""
    for attempt in range(attempts):
        try:
            source.rename(target)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            gc.collect()
            time.sleep(0.5)


def _swap_dataset(stage: Path, root: Path) -> None:
    backup = root.with_name(root.name + '_previous')
    if backup.exists():
        shutil.rmtree(backup, ignore_errors=True)
    if root.exists():
        _rename_retry(root, backup)
    try:
        _rename_retry(stage, root)
    except Exception:
        if root.exists():
            shutil.rmtree(root, ignore_errors=True)
        if backup.exists():
            _rename_retry(backup, root)
        raise
    # Keep the previous dataset for manual recovery, matching the existing
    # projects' behaviour.  The next export replaces this backup.


def export_lerobot(take, dataset_root, schema: DatasetSchema | None = None,
                   repo_id: str | None = None, video_backend: str = 'pyav',
                   validate: bool = True) -> dict:
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        raise ImportError('exporting to LeRobot requires `pip install lerobot`; '
                          'use export_manifest for a dependency-free fallback') from exc
    take = Path(take)
    root = Path(dataset_root)
    meta = _read_meta(take)
    schema = _schema_from_meta(meta, schema)
    repo_id = repo_id or schema.repo_id
    _install_retry()
    root.parent.mkdir(parents=True, exist_ok=True)
    files = _frame_files(take)
    if int(meta.get('frames', len(files))) != len(files):
        raise DatasetError('raw frame count does not match the saved intent')
    features = schema.features()
    stage = root.with_name(f'dataset_staging_{take.name}_{uuid.uuid4().hex[:6]}')
    if stage.exists():
        raise DatasetError(f'staging directory already exists: {stage}')
    try:
        if root.exists():
            shutil.copytree(root, stage)
            if hasattr(LeRobotDataset, 'resume'):
                # LeRobot >= 0.5 opens existing datasets read-only; resume()
                # returns a writer that can append another episode.
                dataset = LeRobotDataset.resume(
                    repo_id, **_filter_kwargs(LeRobotDataset.resume, root=stage,
                                              video_backend=video_backend))
            else:
                dataset = LeRobotDataset(repo_id, root=stage,
                                         video_backend=video_backend)
            dataset.vcodec = 'h264'
        else:
            dataset = LeRobotDataset.create(
                repo_id,
                **_filter_kwargs(LeRobotDataset.create, fps=schema.fps,
                                 features=features, root=stage,
                                 robot_type=schema.robot_type,
                                 video_backend=video_backend, vcodec='h264'))
        episode = dataset.num_episodes
        depth_dir = stage / 'depth' / f'episode_{episode:06d}'
        depth_dir.mkdir(parents=True, exist_ok=True)
        task = meta.get('task', '')
        reader = FrameReader(files, schema.cameras)
        depth_rows = []           # row -> sidecar file holding its depth
        for index in range(len(files)):
            raw = reader.read(index)
            frame = {
                'observation.state': np.asarray(raw['observation_state'], dtype=np.float32),
                'action': np.asarray(raw['action'], dtype=np.float32),
                'task': task,
            }
            if schema.include_time:
                value = float(raw['timestamp'][0]) if 'timestamp' in raw else index / schema.fps
                frame[schema.time_name] = np.array([value], dtype=np.float64)
            for camera in schema.cameras:
                key = f'{camera.name}_rgb'
                if key in raw:
                    frame[f'observation.images.{camera.name}'] = raw[key]
            dataset.add_frame(frame)
            with np.errstate(all='ignore'):
                if any(f'{camera.name}_depth_mm' in raw for camera in schema.cameras):
                    # Depth repeats with its camera frame: write it once (the row
                    # that first showed it) and map every row to that file.
                    refs = [int(raw[f'{camera.name}_ref'][0]) for camera in schema.cameras
                            if f'{camera.name}_ref' in raw]
                    source = min(refs) if refs and all(r == refs[0] for r in refs) else index
                    if source == index or not (depth_dir / f'{source:06d}.npz').exists():
                        payload = {'timestamp': raw['timestamp'] if 'timestamp' in raw
                                   else np.array([index / schema.fps])}
                        for camera in schema.cameras:
                            for suffix in ('depth_mm', 'K', 'T_world_camera'):
                                key = f'{camera.name}_{suffix}'
                                if key in raw:
                                    payload[key] = raw[key]
                        source = index
                        np.savez_compressed(depth_dir / f'{index:06d}.npz', **payload)
                    depth_rows.append(source)
        dataset.save_episode(parallel_encoding=False)
        dataset.finalize()
        (depth_dir / 'episode.json').write_text(json.dumps({
            **meta, 'episode_index': episode, 'frames': len(files),
            # Row i's depth is depth/episode_*/{depth_rows[i]:06d}.npz (shared by the
            # rows that show the same camera frame).
            'depth_rows': depth_rows,
            'depth_encoding': DEPTH_ENCODING,
            'extrinsic_convention': EXTRINSIC_CONVENTION,
            'camera_frame': CAMERA_FRAME_CONVENTION,
        }, ensure_ascii=False, indent=2), encoding='utf-8')
        contract_dir = stage / 'meta'
        contract_dir.mkdir(parents=True, exist_ok=True)
        (contract_dir / 'capture_contract.json').write_text(
            json.dumps(schema.contract(), ensure_ascii=False, indent=2), encoding='utf-8')
        if validate:
            check = LeRobotDataset(repo_id, root=stage, video_backend=video_backend)
            for index in (len(check) - len(files), len(check) - 1):
                sample = check[index]
                if tuple(sample['action'].shape) != (schema.dim,):
                    raise DatasetError('exported action shape is wrong')
                for camera in schema.cameras:
                    key = f'observation.images.{camera.name}'
                    if key in sample:
                        expected = (3, camera.height, camera.width)
                        if tuple(sample[key].shape) != expected:
                            raise DatasetError(f'exported {key} shape {sample[key].shape} != {expected}')
            summary = dict(episodes=check.num_episodes, frames=len(check), root=str(root))
            del sample, check
        else:
            summary = dict(episodes=dataset.num_episodes, frames=len(dataset), root=str(root))
        del dataset
        gc.collect()
        _swap_dataset(stage, root)
        return summary
    except Exception:
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)
        raise


def exporter_for(name: str):
    """Map a config value to the LeRobot writer or the manifest fallback."""
    return export_manifest if str(name).strip().lower() == 'manifest' else export_lerobot


def export_manifest(take, dataset_root, schema: DatasetSchema | None = None) -> dict:
    """Dependency-free fallback: index the take without converting the videos."""
    take = Path(take)
    root = Path(dataset_root)
    meta = _read_meta(take)
    schema = _schema_from_meta(meta, schema)
    files = _frame_files(take)
    root.mkdir(parents=True, exist_ok=True)
    entry = {
        'take': take.name,
        'task': meta.get('task', ''),
        'frames': len(files),
        'fps': schema.fps,
        'robot': schema.robot.name,
        'state_names': schema.state_names,
        'cameras': schema.camera_names,
        'success': meta.get('success'),
        'created': meta.get('created'),
        'source': str(take),
        'format': 'npz-frames',
    }
    manifest = root / 'episodes.jsonl'
    with manifest.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + '\n')
    (root / 'schema.json').write_text(json.dumps(schema.to_dict(), ensure_ascii=False, indent=2),
                                      encoding='utf-8')
    (root / 'capture_contract.json').write_text(
        json.dumps(schema.contract(), ensure_ascii=False, indent=2), encoding='utf-8')
    with manifest.open('r', encoding='utf-8') as handle:
        episodes = sum(1 for _ in handle)
    return dict(episodes=episodes, frames=len(files), root=str(root))


def export_episode(take, dataset_root, schema=None, prefer='lerobot') -> dict:
    """Try LeRobot and fall back to the manifest exporter."""
    if prefer == 'manifest':
        return export_manifest(take, dataset_root, schema)
    try:
        return export_lerobot(take, dataset_root, schema)
    except ImportError:
        return export_manifest(take, dataset_root, schema)
