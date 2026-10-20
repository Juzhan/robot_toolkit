"""Make the toolkit importable both standalone and inside the host workspace."""

import sys
from pathlib import Path

HERE = Path(__file__).resolve()
PACKAGE_ROOT = HERE.parents[1]          # .../robot_toolkit (repository root)
PACKAGE_PARENT = HERE.parents[2]        # .../projects in the host workspace
WORKSPACE_ROOT = HERE.parents[3]        # host workspace root, if present

for path in (PACKAGE_PARENT, PACKAGE_ROOT, WORKSPACE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
