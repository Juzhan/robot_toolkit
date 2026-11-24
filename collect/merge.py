"""Merge exported LeRobot v3 datasets without re-exporting or re-encoding.

LeRobot's ``aggregate_datasets`` copies source parquet/mp4 files as they are and
renumbers episode, frame and file indices, video offsets, tasks and stats. The
toolkit's depth side-car (``depth/episode_NNNNNN``) is outside LeRobot, so it is
renumbered here with the same episode offsets. ``verify_merge`` then compares
every merged row and the boundary video frames with the sources.

    python -m robot_toolkit.collect.merge OUT SRC1 SRC2 [...]
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def _meta(root):
    return json.loads((Path(root) / 'meta' / 'info.json').read_text(encoding='utf-8'))


def fix_episode_meta_pointers(root) -> int:
    """Point ``meta/episodes/{chunk,file}_index`` at the parquet each row is in.

    ``aggregate_datasets`` packs the episode metadata into new files but keeps
    the source pointers; loading ignores them, ``delete_episodes`` does not.
    Returns the number of rows corrected.
    """
    import pandas as pd
    fixed = 0
    for path in sorted((Path(root) / 'meta' / 'episodes').glob('chunk-*/file-*.parquet')):
        chunk, file = int(path.parent.name.split('-')[1]), int(path.stem.split('-')[1])
        table = pd.read_parquet(path)
        wrong = ((table['meta/episodes/chunk_index'] != chunk) |
                 (table['meta/episodes/file_index'] != file))
        if wrong.any():
            fixed += int(wrong.sum())
            table['meta/episodes/chunk_index'] = chunk
            table['meta/episodes/file_index'] = file
            temporary = path.with_suffix('.tmp')
            table.to_parquet(temporary, index=False)
            temporary.replace(path)
    return fixed


def merge_datasets(sources, output, repo_id: str = 'local/merged') -> dict:
    from lerobot.datasets.aggregate import aggregate_datasets
    sources = [Path(s) for s in sources]
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'{output} already exists; choose a new directory')
    infos = [_meta(s) for s in sources]
    for source, info in zip(sources, infos):
        if (source.parent / 'dataset_staging').exists() or any(source.parent.glob('dataset_staging_*')):
            raise RuntimeError(f'{source}: export still in progress')
    aggregate_datasets(repo_ids=[f'local/src{i}' for i in range(len(sources))],
                       aggr_repo_id=repo_id, roots=sources, aggr_root=output,
                       concatenate_videos=False, concatenate_data=False)
    fix_episode_meta_pointers(output)
    offset = 0
    for source, info in zip(sources, infos):
        for episode in range(info['total_episodes']):
            src = source / 'depth' / f'episode_{episode:06d}'
            if not src.exists():
                continue
            dst = output / 'depth' / f'episode_{episode + offset:06d}'
            shutil.copytree(src, dst)
            meta_path = dst / 'episode.json'
            meta = json.loads(meta_path.read_text(encoding='utf-8'))
            meta.update(episode_index=episode + offset, merged_from=dict(
                dataset=str(source.resolve()), episode_index=episode))
            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
        for episode in range(info['total_episodes']):
            for suffix in ('npz', 'json'):
                src = source / 'timestamps' / f'episode_{episode:06d}.{suffix}'
                if src.exists():
                    (output / 'timestamps').mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, output / 'timestamps' / f'episode_{episode + offset:06d}.{suffix}')
        offset += info['total_episodes']
    return dict(episodes=offset, frames=sum(i['total_frames'] for i in infos), root=str(output))


def verify_merge(sources, output) -> dict:
    """Row-by-row data check plus first/last video frame of every episode."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    merged = LeRobotDataset('local/merged', root=output)
    report = dict(episodes=merged.meta.total_episodes, frames=merged.meta.total_frames,
                  checked_rows=0, checked_video_frames=0, max_image_diff=0.0, depth_ok=0)
    episode_offset = frame_offset = 0
    if np.any(np.diff(np.asarray(merged.hf_dataset['index'])) != 1):
        raise AssertionError('merged global index is not contiguous')
    for source in sources:
        src = LeRobotDataset('local/src', root=source)
        for row in range(len(src)):
            a = src.hf_dataset[row]
            b = merged.hf_dataset[row + frame_offset]
            for key in ('observation.state', 'action'):
                if not np.array_equal(np.asarray(a[key]), np.asarray(b[key])):
                    raise AssertionError(f'{source} row {row}: {key} differs')
            if (int(b['episode_index']) != int(a['episode_index']) + episode_offset or
                    int(b['index']) != int(a['index']) + frame_offset or
                    int(b['frame_index']) != int(a['frame_index'])):
                raise AssertionError(f'{source} row {row}: index fields misaligned')
            report['checked_rows'] += 1
        for episode in range(src.meta.total_episodes):
            start = int(src.meta.episodes[episode]['dataset_from_index'])
            stop = int(src.meta.episodes[episode]['dataset_to_index'])
            for row in (start, stop - 1):
                a, b = src[row], merged[row + frame_offset]
                for key in src.meta.video_keys:
                    diff = float((a[key] - b[key]).abs().max())
                    report['max_image_diff'] = max(report['max_image_diff'], diff)
                    report['checked_video_frames'] += 1
            depth = Path(output) / 'depth' / f'episode_{episode + episode_offset:06d}' / 'episode.json'
            if depth.exists() and json.loads(depth.read_text())['episode_index'] == episode + episode_offset:
                report['depth_ok'] += 1
        episode_offset += src.meta.total_episodes
        frame_offset += len(src)
    if report['max_image_diff'] > 0:
        raise AssertionError(f'video frames differ (max {report["max_image_diff"]})')
    if _pointer_errors(output):
        raise AssertionError('episode metadata pointers do not match their files')
    return report


def _pointer_errors(root) -> int:
    import pandas as pd
    errors = 0
    for path in (Path(root) / 'meta' / 'episodes').glob('chunk-*/file-*.parquet'):
        chunk, file = int(path.parent.name.split('-')[1]), int(path.stem.split('-')[1])
        table = pd.read_parquet(path)
        errors += int(((table['meta/episodes/chunk_index'] != chunk) |
                       (table['meta/episodes/file_index'] != file)).sum())
    return errors


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('output')
    parser.add_argument('sources', nargs='+')
    parser.add_argument('--repo-id', default='local/merged')
    args = parser.parse_args(argv)
    print(merge_datasets(args.sources, args.output, args.repo_id), flush=True)
    print(verify_merge([Path(s) for s in args.sources], args.output), flush=True)


if __name__ == '__main__':
    main()
