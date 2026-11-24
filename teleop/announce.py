"""Spoken prompts for operator sessions (Joy-Con teach, master-slave collect).

Uses the system ``spd-say`` without blocking the control loop; when it is not
installed or voice is disabled, ``say`` does nothing.
"""
from __future__ import annotations

import shutil
import subprocess


class Announcer:
    def __init__(self, enabled: bool = True, language: str = 'zh', command: str = 'spd-say'):
        self.language = language
        self.path = shutil.which(command) if enabled else None

    @property
    def available(self) -> bool:
        return self.path is not None

    def say(self, text: str) -> None:
        if self.path is None or not text:
            return
        try:
            subprocess.Popen([self.path, '-l', self.language, text],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError:
            self.path = None
