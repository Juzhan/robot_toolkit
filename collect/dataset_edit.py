"""Remove the most recently exported episode from a LeRobot dataset.

Uses LeRobot's ``delete_episodes`` into a staging copy, carries the toolkit's
depth side-car and capture contract across (LeRobot rebuilds only its own
files), then swaps the staging copy in. The previous dataset is kept as
``<root>_previous`` for manual recovery.

    python -m robot_toolkit.collect.dataset_edit delete-last DATASET_ROOT
"""
from __future__ import annotations

import gc
import json
import shutil
import uuid
from pathlib import Path

from .export import _swap_dataset, dataset_lock


def delete_last(root) -> dict:
    from lerobot.datasets.dataset_tools import delete_episodes
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    root = Path(root)
    info = root / 'meta' / 'info.json'
    if not info.exists():
        raise RuntimeError(f'{root}: no dataset')
    total = int(json.loads(info.read_text(encoding='utf-8'))['total_episodes'])
    if total <= 0:
        raise RuntimeError(f'{root}: dataset is empty')
    last = total - 1
    if total == 1:
        backup = root.with_name(f'{root.name}_deleted_{uuid.uuid4().hex[:6]}')
        root.rename(backup)
        return dict(deleted=last, episodes=0, backup=str(backup))
    stage = root.with_name(f'dataset_staging_delete_{uuid.uuid4().hex[:6]}')
    dataset = LeRobotDataset('local/edit', root=root)
    try:
        delete_episodes(dataset, [last], output_dir=stage, repo_id=dataset.repo_id)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    finally:
        del dataset
        gc.collect()
    for entry in (root / 'depth').glob('episode_*'):
        if entry.name != f'episode_{last:06d}':
            shutil.copytree(entry, stage / 'depth' / entry.name)
    for entry in (root / 'timestamps').glob('episode_*'):
        if not entry.name.startswith(f'episode_{last:06d}.'):
            (stage / 'timestamps').mkdir(parents=True, exist_ok=True)
            shutil.copy2(entry, stage / 'timestamps' / entry.name)
    contract = root / 'meta' / 'capture_contract.json'
    if contract.exists():
        shutil.copy2(contract, stage / 'meta' / contract.name)
    _swap_dataset(stage, root)
    remaining = json.loads(info.read_text(encoding='utf-8'))['total_episodes']
    return dict(deleted=last, episodes=int(remaining))


def delete_last_isolated(root, python=None) -> dict:
    """Run ``delete_last`` in a child process (it may re-encode video)."""
    import os
    import subprocess
    import sys
    package_parent = Path(__file__).resolve().parents[2]
    env = dict(os.environ, PYTHONUTF8='1', HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1')
    env['PYTHONPATH'] = os.pathsep.join(filter(None, [str(package_parent), env.get('PYTHONPATH', '')]))
    completed = subprocess.run([python or sys.executable, '-m', f'{__package__}.dataset_edit',
                                'delete-last', str(root)],
                               env=env, capture_output=True, text=True)
    if completed.returncode:
        tail = (completed.stderr or completed.stdout).strip().splitlines()[-1:] or ['']
        raise RuntimeError(f'delete failed: {tail[0]}')
    return json.loads(completed.stdout.strip().splitlines()[-1])


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('command', choices=('delete-last',))
    parser.add_argument('dataset_root')
    args = parser.parse_args(argv)
    with dataset_lock(args.dataset_root):
        print(json.dumps(delete_last(args.dataset_root), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
