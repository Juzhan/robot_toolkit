"""Drive UR5e or Piper X with the same Joy-Con Cartesian teleop loop.

The point of this example is the switch at the bottom:

    --robot piper_x   -> AgxPiperRobot  + controller-side move_p (20 Hz)
    --robot ur_rtde   -> UrRtdeRobot    + RTDE servo_l (50 Hz)

Everything above that switch (Joy-Con source, Cartesian mapping, safety
wrapper, stale-input handling) is identical.

Examples:
    # Piper X, real hardware; keep the e-stop in hand
    python -m projects.robot_toolkit.examples.joycon_teleop --robot piper_x --yes

    # UR5e at its RTDE address
    python -m projects.robot_toolkit.examples.joycon_teleop --robot ur_rtde --ip 192.168.1.10 --yes

    # No hardware: scripted xyz command, useful for checking the loop
    python -m projects.robot_toolkit.examples.joycon_teleop --robot mock --duration 2
"""
from __future__ import annotations

import argparse

from ..teleop import TeleopSessionConfig, build_teleop_session


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description='Generic Joy-Con Cartesian teleop')
    parser.add_argument('--robot', default='piper_x', choices=['piper_x', 'ur_rtde', 'mock'])
    parser.add_argument('--input', default='joycon', choices=['joycon', 'keyboard'])
    parser.add_argument('--ip', default='192.168.1.10', help='UR RTDE IP')
    parser.add_argument('--duration', type=float, default=60.0,
                        help='stop after N seconds (0 = until Ctrl+C)')
    parser.add_argument('--rate', type=float, default=50.0)
    parser.add_argument('--move-p-hz', type=float, default=20.0)
    parser.add_argument('--stale-s', type=float, default=0.4)
    parser.add_argument('--speed', type=float, default=0.08, help='m/s at full stick')
    parser.add_argument('--smoothing', type=float, default=0.2,
                        help='0=raw, 1=no movement; 0.1-0.3 is a good start')
    parser.add_argument('--linear-accel', type=float, default=0.6,
                        help='m/s^2; 0 disables acceleration limiting')
    parser.add_argument('--angular-accel', type=float, default=1.0,
                        help='rad/s^2; 0 disables acceleration limiting')
    parser.add_argument('--speed-percent', type=int, default=20)
    parser.add_argument('--max-joint-speed', type=float, default=1.0)
    parser.add_argument('--gripper-close-time', type=float, default=0.8)
    parser.add_argument('--gripper-open-time', type=float, default=0.5)
    parser.add_argument('--gripper-button', default='b',
                        choices=['b', 'zr', 'x', 'y', 'r'],
                        help='B avoids JoyconRobotics\' own ZR gripper toggle')
    parser.add_argument('--enable-button', default='y', choices=['y', 'a', 'plus'])
    parser.add_argument('--calibrate', action='store_true',
                        help='run Joy-Con horizontal gyro calibration before teleop')
    parser.add_argument('--mock', action='store_true', help='use scripted input, no hardware')
    parser.add_argument('--yes', action='store_true', help='confirm real robot motion')
    args = parser.parse_args(argv)
    if args.mock and args.robot != 'mock':
        parser.error('offline teleop currently uses --robot mock; '
                     'the Piper/UR mock backends have no Cartesian model')
    mock = bool(args.mock or args.robot == 'mock')
    if not mock and not args.yes:
        parser.error('real robot motion requires --yes; keep the physical e-stop ready')

    session = build_teleop_session(TeleopSessionConfig(
        robot=args.robot,
        input=args.input,
        ip=args.ip,
        mock=mock,
        speed_percent=args.speed_percent,
        max_joint_speed=args.max_joint_speed,
        translation_speed=args.speed,
        smoothing=args.smoothing,
        linear_acceleration=args.linear_accel,
        angular_acceleration=args.angular_accel,
        rate_hz=args.rate,
        move_p_hz=args.move_p_hz,
        stale_s=args.stale_s,
        max_duration_s=args.duration,
        gripper_close_time_s=args.gripper_close_time,
        gripper_open_time_s=args.gripper_open_time,
        gripper_button=args.gripper_button,
        enable_button=args.enable_button,
        calibrate=args.calibrate,
    ))
    try:
        if args.input == 'keyboard':
            print('W/S/A/D/R/F 平移，方向键/Q/E 旋转，Space 夹爪，Y/Enter 使能，Esc 退出。',
                  flush=True)
        else:
            print(f'{args.enable_button.upper()} 使能/暂停；'
                  f'{args.gripper_button.upper()} 切换夹爪；Ctrl+C 退出。', flush=True)
        result = session.run()
        print('teleop:', result.as_dict(), flush=True)
        return 0 if not result.aborted else 2
    finally:
        session.close()


if __name__ == '__main__':
    raise SystemExit(main())
