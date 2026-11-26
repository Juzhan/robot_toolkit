"""``python -m robot_toolkit`` is the same task CLI as ``run.py``."""

from .tasks.cli import main

if __name__ == '__main__':
    raise SystemExit(main())
