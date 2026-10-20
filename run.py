#!/usr/bin/env python3
"""Cross-platform launcher for the robot task runner.

Usage (any OS):

    python run.py --list
    python run.py stack_the_cube --dry-run
    python run.py stack_the_cube teach

The robot is chosen by ``configs/tasks/<name>.json``; this file only bootstraps
the package path and current working directory so ``python run.py`` works from
PowerShell, bash, zsh and CI without setting PYTHONPATH manually.  It works both
when it sits next to the package (``projects/robot_toolkit/run.py``) and when it
sits at the workspace root (``RobotTeleopSimWindows/run.py``).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _import_runner():
    # Package layout: <root>/robot_toolkit/task_runner.py  -> ``robot_toolkit``
    # Workspace layout: <root>/projects/robot_toolkit/...  -> ``projects.robot_toolkit``
    from robot_toolkit.task_runner import main as runner
    return runner

def main(argv=None) -> int:
    # Console output is Chinese; make it independent of the terminal code page
    # so ``python run.py`` matches what ``run.bat`` gets from ``-X utf8``.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass
    for entry in (ROOT, ROOT.parent, ROOT.parent.parent):
        if str(entry) not in sys.path:
            sys.path.insert(0, str(entry))
    os.chdir(ROOT)
    try:
        runner = _import_runner()
    except Exception as exc:
        print(f'无法加载 robot_toolkit：{type(exc).__name__}: {exc}', file=sys.stderr)
        print('请确认当前 Python 已安装 numpy，并设置 ROBOT_TOOLKIT_PYTHON 指向正确解释器。',
              file=sys.stderr)
        return 2
    return int(runner(argv) or 0)


if __name__ == '__main__':
    raise SystemExit(main())
