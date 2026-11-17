"""Keep Python's cyclic garbage collector out of real-time loops.

A full collection pauses every thread of the process (25-50 ms measured by Flash
ALOHA in a 100 Hz recorder; it recurs at the same allocation counts). Processes
that stream commands or camera frames call ``pause_gc`` once streaming starts:
existing objects are frozen and automatic collection stays off. Cyclic garbage is
reclaimed only by ``collect_idle`` at points where nothing is streaming.
"""

from __future__ import annotations

import gc
import threading

_lock = threading.Lock()
_paused = False


def pause_gc() -> None:
    """Disable automatic collection for this process (idempotent, cheap to repeat)."""
    global _paused
    with _lock:
        if not _paused:
            gc.freeze()        # constant time: no collection right before streaming
            _paused = True
        gc.disable()           # also undoes a library that re-enabled it


def collect_idle(resume: bool = False) -> int:
    """Collect cyclic garbage now (call only while nothing is streaming).

    With ``resume`` automatic collection is re-enabled afterwards.
    """
    global _paused
    with _lock:
        gc.unfreeze()
        collected = gc.collect()
        if resume:
            _paused = False
            gc.enable()
        else:
            gc.freeze()
        return collected


def gc_paused() -> bool:
    return _paused and not gc.isenabled()
