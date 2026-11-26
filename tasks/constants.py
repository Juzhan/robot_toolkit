"""Paths and mode names shared by the task modules."""

from pathlib import Path

# In the host workspace the package lives at <root>/projects/robot_toolkit,
# while a standalone clone keeps the package at <repo>/robot_toolkit/.
REPO_ROOT = Path(__file__).resolve().parents[1]
_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
ROOT = (Path(__file__).resolve().parents[3]
        if (Path(__file__).resolve().parents[3] / 'configs').exists()
        else _PACKAGE_ROOT)
DEFAULT_ROOT = 'outputs/robot_tasks'
MODES = ('teach', 'replay', 'collect', 'teach_collect', 'export', 'session',
         'master_slave', 'hand_demo', 'calibrate', 'safety', 'agent')
