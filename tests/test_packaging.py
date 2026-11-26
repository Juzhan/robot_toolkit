"""Keeps the hand-written package list in step with the directories on disk."""

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHIPPED = {'robot_toolkit', 'robot_toolkit.adapters', 'robot_toolkit.calibrate',
           'robot_toolkit.camera', 'robot_toolkit.collect', 'robot_toolkit.control',
           'robot_toolkit.core', 'robot_toolkit.examples', 'robot_toolkit.gripper',
           'robot_toolkit.robot', 'robot_toolkit.teach', 'robot_toolkit.teleop',
           'robot_toolkit.agent', 'robot_toolkit.agent_rpc',
           'robot_toolkit.tasks'}


def _declared() -> set:
    data = tomllib.loads((ROOT / 'pyproject.toml').read_text(encoding='utf-8'))
    return set(data['tool']['setuptools']['packages'])


def _on_disk() -> set:
    """Every directory with an __init__.py, minus the test suite."""
    found = {'robot_toolkit'} if (ROOT / '__init__.py').exists() else set()
    for directory in ROOT.iterdir():
        if not directory.is_dir() or not (directory / '__init__.py').exists():
            continue
        if directory.name in {'tests', 'build', 'outputs'}:
            continue
        found.add(f'robot_toolkit.{directory.name}')
    return found


def test_every_package_is_declared():
    missing = sorted(_on_disk() - _declared())
    # A package that is not declared is silently left out of a wheel build.
    assert not missing, f'packages missing from pyproject.toml: {missing}'


def test_no_declared_package_is_gone():
    stale = sorted(_declared() - _on_disk())
    assert not stale, f'pyproject.toml declares packages that no longer exist: {stale}'


def test_shipped_set_matches_declaration():
    assert _declared() == SHIPPED
