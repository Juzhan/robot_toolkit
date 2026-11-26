#!/usr/bin/env python3
"""Cross-platform launcher for the robot task runner.

Usage (any OS):

    python run.py --list
    python run.py stack_the_cube --dry-run
    python run.py stack_the_cube teach

The robot is chosen by ``configs/tasks/<name>.json``; this file only bootstraps
the package path and current working directory so ``python run.py`` works from
PowerShell, bash, zsh and CI without setting PYTHONPATH manually.

Interpreter selection, same order on every OS:

1. ``ROBOT_TOOLKIT_PYTHON`` in the environment;
2. a ``.robot_toolkit_python`` file in this directory or one of its parents,
   containing the interpreter path (gitignored, so it stays machine-specific);
3. the interpreter already running this file.

When 1 or 2 names a different interpreter, the script re-executes itself with it.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
_REEXEC_FLAG = '_ROBOT_TOOLKIT_REEXEC'


def _selected_python():
    """Interpreter from the environment or a pointer file, resolved to a path."""
    value = os.environ.get('ROBOT_TOOLKIT_PYTHON')
    if not value:
        for base in (ROOT, ROOT.parent, ROOT.parent.parent):
            pointer = base / '.robot_toolkit_python'
            if pointer.is_file():
                value = pointer.read_text(encoding='utf-8-sig').strip()
                break
    if not value:
        return None
    if Path(value).exists():
        return value
    return shutil.which(value)


def _reexec_with_selected_python() -> None:
    """Honour ROBOT_TOOLKIT_PYTHON / .robot_toolkit_python before importing."""
    if os.environ.get(_REEXEC_FLAG):
        return
    target = _selected_python()
    if not target:
        return
    try:
        same = Path(target).resolve() == Path(sys.executable).resolve()
    except OSError:
        same = False
    if same:
        return
    os.environ[_REEXEC_FLAG] = '1'
    try:
        os.execv(target, [target, '-X', 'utf8', str(Path(__file__).resolve()), *sys.argv[1:]])
    except OSError:
        os.environ.pop(_REEXEC_FLAG, None)


def _import_runner():
    # Package layout: <root>/robot_toolkit/task_runner.py  -> ``robot_toolkit``
    # Workspace layout: <root>/projects/robot_toolkit/...  -> ``projects.robot_toolkit``
    from robot_toolkit.task_runner import main as runner
    return runner

def main(argv=None) -> int:
    _reexec_with_selected_python()
    # Console output is Chinese; make it independent of the terminal code page.
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
