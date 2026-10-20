"""Teaching: record, store, edit and replay demonstrations."""

from .console import KeyReader, joycon_teach, keyboard_teach, state_line
from .editor import add_marker, mark_dwells, remove_marker, retime, summarize, trim
from .recorder import TeachRecorder, TeachRecorderConfig
from .replay import ReplayPlan, ReplayResult, Replayer, prepare_playback
from .trajectory import (Marker, Trajectory, find_dwell_candidates, find_still_runs,
                         joint_motion, load_trajectory, save_trajectory)

__all__ = [
    'add_marker',
    'mark_dwells',
    'remove_marker',
    'retime',
    'summarize',
    'trim',
    'TeachRecorder',
    'TeachRecorderConfig',
    'KeyReader',
    'keyboard_teach',
    'joycon_teach',
    'joint_motion',
    'state_line',
    'ReplayPlan',
    'ReplayResult',
    'Replayer',
    'prepare_playback',
    'Marker',
    'Trajectory',
    'find_dwell_candidates',
    'find_still_runs',
    'load_trajectory',
    'save_trajectory',
]
