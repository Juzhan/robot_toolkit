"""Compatibility entry point; the modes now live in :mod:`robot_toolkit.tasks`.

Kept so ``python -m robot_toolkit.task_runner``, ``run.py`` and existing imports
of ``robot_toolkit.task_runner`` keep working.
"""

from .tasks import *  # noqa: F401,F403
from .tasks import (DEFAULT_ROOT, MODES, ROOT, TaskConfig, _JoyConKeys,  # noqa: F401
                    dataset_root, execute, lerobot_root, load_task_config, main,
                    trajectory_root)

__all__ = ['DEFAULT_ROOT', 'MODES', 'ROOT', 'TaskConfig', 'dataset_root',
           'execute', 'lerobot_root', 'load_task_config', 'main',
           'trajectory_root']
