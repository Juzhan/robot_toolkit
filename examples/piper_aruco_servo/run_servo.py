"""Cross-platform launcher for the ArUco visual servo with the tuned settings.

    python run_servo.py                      # PBVS hand following (moves the arm!)
    python run_servo.py --obstacles          # same, with the example desk obstacles
    python run_servo.py --obstacles my.json  # your own obstacle file
    python run_servo.py --serial 1234 --lead .04   # any probe option overrides the preset
    python run_servo.py --read-only          # no enable, no motion: detection check only

Logs go to ``<repo>/outputs/aruco_servo/servo_<time>.jsonl`` (git-ignored).
"""

import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

import piper_aruco_real_probe as probe  # noqa: E402

# Tuned on the Piper X + wrist D435 (2026-11). Later command-line options win.
PRESET = [
    '--execute', '--yes',
    '--serial', '263022075825',
    '--fps', '60', '--exposure-ms', '8', '--camera-gain', '96',
    '--motion', 'servo', '--control', 'pbvs', '--marker-id', '0', '--distance', '.30',
    '--control-hz', '100', '--lead', '.08',
    '--reference-speed', '1.2', '--reference-accel', '4', '--reference-jerk', '30',
    '--camera-latency', '.03',
    '--ibvs-gain', '2.0', '--ibvs-linear-speed', '.22', '--ibvs-angular-speed', '.8',
    '--ibvs-feedforward', '.9', '--ibvs-ff-window', '.15', '--ibvs-ff-deadband', '.02',
    '--ibvs-ff-max', '.3', '--ibvs-stop-px', '3', '--ibvs-resume-px', '6',
    '--ibvs-timeout', '.2', '--ibvs-duration', '0',
    '--speed-percent', '100', '--steps', '0', '--hold',
]
OBSTACLES = REPO / 'configs/obstacles/piper_desk_example.json'


def build_argv(argv):
    argv = list(argv)
    extra = []
    if '--obstacles' in argv:
        index = argv.index('--obstacles')
        value = argv[index + 1] if index + 1 < len(argv) and not argv[index + 1].startswith('--') \
            else None
        del argv[index:index + (2 if value else 1)]
        extra = ['--obstacles', value or str(OBSTACLES)]
    preset = list(PRESET)
    if '--read-only' in argv or '--dry-run' in argv:
        preset = [a for a in preset if a not in ('--execute', '--yes')]
    if '--log' not in argv:
        log = REPO / 'outputs' / 'aruco_servo' / time.strftime('servo_%Y%m%d_%H%M%S.jsonl')
        extra += ['--log', str(log)]
    return preset + extra + argv


if __name__ == '__main__':
    raise SystemExit(probe.main(build_argv(sys.argv[1:])))
