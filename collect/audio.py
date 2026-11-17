"""Cross-platform short audio cues.

Ported from ``projects/ur5e_real_collect/audio.py``; the WAV assets live in
``robot_toolkit/assets/audio``.

Windows uses ``winsound``; macOS uses ``afplay``/``osascript``; Linux uses
``paplay``/``aplay`` when available.  All failures are swallowed: audio cues are
nice-to-have and must never stop a robot session.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

AUDIO_DIR = Path(__file__).resolve().parents[1] / 'assets' / 'audio'


def _popen(cmd):
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def play_file(path, block=False):
    """Play a WAV/MP3 file; return True when a player was started."""
    path = Path(path)
    if not path.exists():
        return False
    try:
        if sys.platform == 'win32':
            import winsound
            if block:
                winsound.PlaySound(str(path), winsound.SND_FILENAME)
            else:
                winsound.PlaySound(str(path), winsound.SND_FILENAME | winsound.SND_ASYNC)
            return True
        if sys.platform == 'darwin':
            proc = _popen(['afplay', str(path)])
        else:
            player = shutil.which('paplay') or shutil.which('aplay')
            if player is None:
                return False
            proc = _popen([player, str(path)])
        if block:
            try:
                proc.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                proc.terminate()
        return True
    except Exception:
        return False


def play_file_async(path):
    return play_file(path, block=False)


def beep(frequency=880, duration_ms=150, block=True):
    """Beep without requiring a WAV file."""
    try:
        if sys.platform == 'win32':
            import winsound
            winsound.Beep(int(frequency), int(duration_ms))
            return True
        if sys.platform == 'darwin':
            proc = _popen(['osascript', '-e', 'beep'])
        else:
            return False
        if block:
            try:
                proc.wait(timeout=max(1.0, duration_ms / 1000.0 + 1.0))
            except subprocess.TimeoutExpired:
                proc.terminate()
        return True
    except Exception:
        return False
