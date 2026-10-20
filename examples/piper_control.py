"""Read or move an AgileX Piper / Piper X with the Python SDK.

Read-only by default.  Every motion command requires ``--yes``.

Examples (Windows, AgileX CAN module)::

    python -m projects.robot_toolkit.examples.piper_control
    python -m projects.robot_toolkit.examples.piper_control --enable
    python -m projects.robot_toolkit.examples.piper_control --move-j 0 1.1 -1.1 0 0.7 0 --yes
    python -m projects.robot_toolkit.examples.piper_control --move-p 0.10 0.02 0.25 0 1.57 0 --yes
    python -m projects.robot_toolkit.examples.piper_control --open --yes
    python -m projects.robot_toolkit.examples.piper_control --close --yes
    python -m projects.robot_toolkit.examples.piper_control --disable --yes
    python -m projects.robot_toolkit.examples.piper_control --estop --yes

The arm holds its last position when the script exits.  ``--estop`` uses the
Piper electronic emergency stop; use the physical e-stop for a real emergency.
"""
from __future__ import annotations

import argparse
import json

from ..robot import AgxPiperConfig, AgxPiperRobot


def _json_default(value):
    if hasattr(value, 'tolist'):
        return value.tolist()
    if hasattr(value, 'item'):
        return value.item()
    return str(value)


def _state_dict(robot: AgxPiperRobot) -> dict:
    state = robot.state()
    return {
        'q': None if state.q is None else state.q.tolist(),
        'qd': None if state.qd is None else state.qd.tolist(),
        'tcp': None if state.tcp is None else state.tcp.tolist(),
        'protective_stop': state.protective_stop,
        'extra': state.extra,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description='Piper X read/move helper')
    parser.add_argument('--model', default='piper_x',
                        choices=['piper', 'piper_h', 'piper_l', 'piper_x'])
    parser.add_argument('--interface', default='auto')
    parser.add_argument('--channel', default='auto')
    parser.add_argument('--firmware', default='auto',
                        choices=['auto', 'default', 'v183', 'v188', 'v189'])
    parser.add_argument('--speed-percent', type=int, default=30)
    parser.add_argument('--no-gripper', action='store_true')
    parser.add_argument('--enable', action='store_true', help='enable motors, no motion')
    parser.add_argument('--move-j', nargs=6, type=float, metavar=('J1', 'J2', 'J3', 'J4', 'J5', 'J6'),
                        help='move to absolute joint angles (rad)')
    parser.add_argument('--move-p', nargs=6, type=float, metavar=('X', 'Y', 'Z', 'RX', 'RY', 'RZ'),
                        help='point-to-point TCP pose (m, rad)')
    parser.add_argument('--move-l', nargs=6, type=float, metavar=('X', 'Y', 'Z', 'RX', 'RY', 'RZ'),
                        help='linear TCP pose (m, rad)')
    parser.add_argument('--gripper', type=float, metavar='METRES', help='set gripper width')
    parser.add_argument('--open', action='store_true', help='open the gripper')
    parser.add_argument('--close', action='store_true', help='close the gripper')
    parser.add_argument('--estop', action='store_true', help='Piper electronic emergency stop')
    parser.add_argument('--disable', action='store_true',
                        help='disable motors; support the arm first, it may fall')
    parser.add_argument('--mock', action='store_true', help='offline mock (no CAN)')
    parser.add_argument('--yes', action='store_true', help='confirm a motion command')
    args = parser.parse_args(argv)

    motion = any([args.move_j is not None, args.move_p is not None,
                  args.move_l is not None, args.gripper is not None,
                  args.open, args.close, args.estop, args.disable])
    if motion and not args.yes:
        parser.error('motion commands require --yes (check clearance and keep the e-stop ready)')
    if sum(bool(value) for value in (args.open, args.close, args.estop, args.disable)) > 1:
        parser.error('choose only one of --open / --close / --estop / --disable')
    if sum(bool(value) for value in (args.move_j is not None, args.move_p is not None,
                                     args.move_l is not None, args.gripper is not None,
                                     args.open, args.close, args.estop,
                                     args.disable)) > 1:
        parser.error('choose one motion command per invocation')

    config = AgxPiperConfig(model=args.model, interface=args.interface,
                            channel=args.channel, firmware=args.firmware,
                            speed_percent=args.speed_percent,
                            enable_on_connect=bool(
                                args.enable or (motion and not (args.estop or args.disable))),
                            has_gripper=not args.no_gripper, mock=args.mock)
    robot = AgxPiperRobot(config, mock=args.mock)
    try:
        robot.connect()
        print('before:', json.dumps(_state_dict(robot), ensure_ascii=False,
                                   default=_json_default))
        if args.move_j is not None:
            robot.move_j(args.move_j, wait=True, timeout=60.0)
            print('move_j: reached', args.move_j)
        elif args.move_p is not None:
            robot.move_p(args.move_p, wait=True, timeout=60.0)
            print('move_p: reached', args.move_p)
        elif args.move_l is not None:
            robot.move_l(args.move_l, wait=True, timeout=60.0)
            print('move_l: reached', args.move_l)
        elif args.open:
            robot.open_gripper(wait=True)
            print('gripper: open')
        elif args.close:
            robot.close_gripper(wait=True)
            print('gripper: close')
        elif args.gripper is not None:
            robot.move_gripper(args.gripper, wait=True)
            print('gripper:', args.gripper)
        elif args.estop:
            robot.emergency_stop()
            print('emergency_stop: sent; recover on the pendant/controller')
        elif args.disable:
            ok = robot.disable()
            print('disable:', 'ok' if ok else 'not acknowledged; check the controller')
            if not ok:
                return 1
        print('after :', json.dumps(_state_dict(robot), ensure_ascii=False,
                                   default=_json_default))
        return 0
    finally:
        robot.disconnect()


if __name__ == '__main__':
    raise SystemExit(main())
