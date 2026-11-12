"""Check the unified servo stream on a Piper / Piper X.

Read-only by default: connects without enabling the motors, reads the joints,
builds the motion and prints every pre-motion check (speed, acceleration,
jerk, joint bounds, start distance).  Nothing is commanded.

With ``--execute`` (and a typed confirmation, unless ``--yes``) it streams
the motion through ``control.run_trajectory`` behind the profile's safety
guard, then prints the stream statistics:

- default: a small out-and-back move of one joint (``--joint``,
  ``--amplitude``) as a min-jerk profile;
- ``--plan FILE.npz``: a planned trajectory (``q``: N x 6 at ``dt`` seconds,
  e.g. a robokit plan retimed with ``control.retime``).  It must start within
  ``--start-tolerance`` of the measured joints.

Example::

    python -m robot_toolkit.examples.piper_stream_check                 # dry run
    python -m robot_toolkit.examples.piper_stream_check --execute       # moves
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from ..control import (TrajectoryLimits, check_trajectory, min_jerk, run_trajectory,
                       trajectory_peaks)
from ..profiles import build_robot
from ..robot import AgxPiperConfig


def _motion(args, start: np.ndarray, dt: float) -> np.ndarray:
    if args.plan:
        data = np.load(args.plan)
        if 'dt' in data.files and abs(float(data['dt']) - dt) > 1e-9:
            raise ValueError(f"plan dt {float(data['dt'])} differs from the stream dt {dt}")
        return np.asarray(data['q'], dtype=float)
    away = start.copy()
    away[args.joint] += args.amplitude
    out = min_jerk(start, away, dt, args.speed, args.accel)
    back = min_jerk(away, start, dt, args.speed, args.accel)
    return np.concatenate([out, np.repeat(away[None], int(0.5 / dt), axis=0), back[1:]])


def main(argv=None) -> int:
    defaults = AgxPiperConfig()
    parser = argparse.ArgumentParser(description='Piper servo stream check (dry run by default)')
    parser.add_argument('--model', default='piper_x',
                        choices=['piper', 'piper_h', 'piper_l', 'piper_x'])
    parser.add_argument('--interface', default='auto')
    parser.add_argument('--channel', default='auto')
    parser.add_argument('--firmware', default='auto',
                        choices=['auto', 'default', 'v183', 'v188', 'v189'])
    parser.add_argument('--joint', type=int, default=5, help='joint index 0-5 for the default move')
    parser.add_argument('--amplitude', type=float, default=0.15, help='rad, default move')
    parser.add_argument('--speed', type=float, default=0.3, help='peak joint speed, rad/s')
    parser.add_argument('--accel', type=float, default=0.6, help='peak joint accel, rad/s^2')
    parser.add_argument('--plan', help='npz with q (N x 6) and dt instead of the default move')
    parser.add_argument('--max-velocity', type=float, default=defaults.max_joint_speed)
    parser.add_argument('--max-acceleration', type=float, default=1.2)
    parser.add_argument('--max-jerk', type=float, default=8.0, help='rad/s^3')
    parser.add_argument('--start-tolerance', type=float, default=0.02)
    parser.add_argument('--execute', action='store_true', help='actually move the arm')
    parser.add_argument('--yes', action='store_true', help='skip the confirmation prompt')
    parser.add_argument('--mock', action='store_true', help='offline mock (no CAN)')
    args = parser.parse_args(argv)
    if not 0 <= args.joint < 6:
        parser.error('--joint must be 0-5')

    dt = 1.0 / defaults.command_hz
    options = {'model': args.model, 'interface': args.interface, 'channel': args.channel,
               'firmware': args.firmware, 'max_joint_speed': args.max_velocity,
               'enable_on_connect': bool(args.execute), 'has_gripper': False,
               'ik_backend': 'none', 'ik_warmup': False}  # joint-space only: no IK
    robot = build_robot({'kind': 'agx_piper', 'name': args.model, 'options': options,
                         'limits': {'max_velocity': args.max_velocity}, 'safety': True},
                        mock=args.mock)
    limits = TrajectoryLimits(args.max_velocity, args.max_acceleration, args.max_jerk)
    try:
        robot.connect()
        state = robot.state()
        if state.q is None:
            print('no joint state; check the CAN link', flush=True)
            return 1
        start = np.asarray(state.q, dtype=float)
        print('measured q:', np.round(start, 4).tolist(), flush=True)
        q = _motion(args, start, dt)
        peaks = {k: np.round(v, 3).tolist() for k, v in trajectory_peaks(q, dt).items()}
        gap = float(np.max(np.abs(q[0] - start)))
        print(json.dumps({'samples': len(q), 'duration_s': round((len(q) - 1) * dt, 2), 'dt': dt,
                          'start_gap_rad': round(gap, 4), 'peaks': peaks}), flush=True)
        try:
            check_trajectory(q, dt, limits, defaults.limits_lower, defaults.limits_upper)
        except ValueError as exc:
            print('REJECTED:', exc, flush=True)
            return 2
        if gap > args.start_tolerance:
            print(f'REJECTED: starts {gap:.4f} rad from the measured joints', flush=True)
            return 2
        print('checks passed', flush=True)
        if not args.execute:
            print('dry run: nothing was commanded (add --execute to move)', flush=True)
            return 0
        if not args.yes:
            answer = input('确认急停在手、工作区无人，执行？[y/N] ').strip().lower()
            if answer not in ('y', 'yes'):
                print('取消。')
                return 0
        result = run_trajectory(robot, q, dt, limits=None, start_tolerance=args.start_tolerance,
                                tolerance=0.01)
        print(json.dumps(result.as_dict(), default=float), flush=True)
        return 0 if result.ok else 3
    finally:
        try:
            robot.disconnect()
        except Exception:
            pass


if __name__ == '__main__':
    raise SystemExit(main())
