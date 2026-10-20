"""Adapters that expose the existing project-specific drivers as toolkit backends.

Nothing here imports the legacy modules eagerly: pass an already constructed
object, which keeps the toolkit independent from the applications around it.
"""

from .legacy_ur import LegacyUrRobot

__all__ = ['LegacyUrRobot']
