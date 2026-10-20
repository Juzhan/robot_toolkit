"""Piper X keyboard teach + optional replay, without cameras.

Flow:
  1. 机器人进入 leader 零力拖动（set_leader_mode）。
  2. 人以 50 Hz 采样拖动轨迹。
  3. 空格切换夹爪开/合，并记录 close/open 标记。
  4. s 或回车结束并保存；q/Esc 放弃。
  5. 加 --replay 可在保存后按同一轨迹回放。

No camera and no dataset are involved; this is a joint trajectory recorder.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ..profiles import build_gripper, build_robot
from ..teach import (Replayer, joint_motion, keyboard_teach, load_trajectory,
                     save_trajectory, summarize)


def _gripper_options(args) -> dict:
    return {
        'force': 1.0,
        'close_time_s': args.gripper_close_time,
        'open_time_s': args.gripper_open_time,
        'ramp_hz': args.gripper_ramp_hz,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description='Piper X keyboard teach/replay (no camera)')
    parser.add_argument('--name', default='piper_teach_01', help='trajectory name')
    parser.add_argument('--output', default='outputs/piper_teach', help='trajectory directory')
    parser.add_argument('--task', default='', help='task text stored in trajectory metadata')
    parser.add_argument('--model', default='piper_x',
                        choices=['piper', 'piper_h', 'piper_l', 'piper_x'])
    parser.add_argument('--interface', default='auto')
    parser.add_argument('--channel', default='auto')
    parser.add_argument('--firmware', default='auto',
                        choices=['auto', 'default', 'v183', 'v188', 'v189'])
    parser.add_argument('--speed-percent', type=int, default=20)
    parser.add_argument('--rate', type=float, default=50.0)
    parser.add_argument('--max-joint-speed', type=float, default=1.5,
                        help='replay/safety joint speed limit in rad/s')
    parser.add_argument('--smoothing', type=int, default=9,
                        help='trajectory smoothing window in samples')
    parser.add_argument('--max-duration', type=float, default=300.0)
    parser.add_argument('--no-gripper', action='store_true')
    parser.add_argument('--gripper-close-time', type=float, default=0.8,
                        help='seconds to ramp the gripper closed (0 = SDK default)')
    parser.add_argument('--gripper-open-time', type=float, default=0.5,
                        help='seconds to ramp the gripper open (0 = SDK default)')
    parser.add_argument('--gripper-ramp-hz', type=float, default=50.0,
                        help='target update rate while ramping the gripper')
    parser.add_argument('--replay', action='store_true', help='replay after saving')
    parser.add_argument('--replay-only', action='store_true',
                        help='skip teaching; load output/name and replay it')
    parser.add_argument('--allow-static', action='store_true',
                        help='save even if the joints hardly moved')
    parser.add_argument('--speed-scale', type=float, default=0.3)
    parser.add_argument('--yes', action='store_true', help='confirm replay without prompting')
    parser.add_argument('--mock', action='store_true', help='offline mock (no CAN)')
    args = parser.parse_args(argv)

    options = {
        'model': args.model,
        'interface': args.interface,
        'channel': args.channel,
        'firmware': args.firmware,
        'speed_percent': args.speed_percent,
        'max_joint_speed': args.max_joint_speed,
        'enable_on_connect': True,
        'has_gripper': not args.no_gripper,
    }
    directory = Path(args.output) / args.name
    if args.replay_only:
        try:
            trajectory = load_trajectory(directory)
        except Exception as exc:
            print(f'读不到轨迹 {directory}: {exc}', flush=True)
            return 1
        motion = joint_motion(trajectory)
        if motion < 1e-4 and not args.allow_static:
            print(f'轨迹关节最大变化量只有 {motion:.4f} rad，几乎是静止轨迹。',
                  flush=True)
            print('这通常是旧版本在 leader 模式下读错反馈保存的空轨迹；'
                  '请重新示教，或加 --allow-static 强制回放。', flush=True)
            return 3
        if not args.yes:
            answer = input('确认急停在手、工作区无人，回放？[y/N] ').strip().lower()
            if answer not in ('y', 'yes'):
                print('取消回放。')
                return 0
        return _replay(trajectory, args, options)

    robot = build_robot({'kind': 'agx_piper', 'name': args.model,
                         'options': options,
                         'limits': {'max_velocity': args.max_joint_speed},
                         'safety': True}, mock=args.mock)
    gripper = (None if args.no_gripper
               else build_gripper('agx', robot=robot, options=_gripper_options(args)))
    try:
        robot.connect()
        if gripper is not None:
            gripper.connect()
        trajectory = keyboard_teach(
            robot, gripper, name=args.name, task=args.task,
            rate=args.rate, max_duration=args.max_duration,
            meta={'backend': 'agx_piper', 'camera': 'none'},
            freedrive=True, wait_feedback='leader')
    except Exception as exc:
        print(f'\n示教失败：{type(exc).__name__}: {exc}', flush=True)
        return 1
    finally:
        if gripper is not None:
            try:
                gripper.stop()
            except Exception:
                pass
        try:
            robot.disconnect()
        except Exception:
            pass

    if trajectory is None:
        return 2

    motion = joint_motion(trajectory)
    print(f'关节最大变化量：{motion:.4f} rad', flush=True)
    if motion < 1e-4:
        print('警告：整段轨迹关节角几乎没变化，leader 反馈可能仍未生效，'
              '或示教时没有拖动机械臂。', flush=True)
        if not args.allow_static:
            print('未保存。若确实只需要夹爪开合，请加 --allow-static 后重试。',
                  flush=True)
            return 3

    save_trajectory(directory, trajectory)
    print('已保存：', directory)
    print('轨迹摘要：', summarize(trajectory), flush=True)

    if args.replay:
        if not args.yes:
            answer = input('确认急停在手、工作区无人，回放？[y/N] ').strip().lower()
            if answer not in ('y', 'yes'):
                print('取消回放。')
                return 0
        return _replay(trajectory, args, options)
    return 0


def _replay(trajectory, args, options) -> int:
    replay_robot = build_robot({'kind': 'agx_piper', 'name': args.model,
                                'options': options,
                                'limits': {'max_velocity': args.max_joint_speed},
                                'safety': True}, mock=args.mock)
    replay_gripper = (None if args.no_gripper
                      else build_gripper('agx', robot=replay_robot,
                                         options=_gripper_options(args)))
    try:
        replay_robot.connect()
        if replay_gripper is not None:
            replay_gripper.connect()
        replayer = Replayer(replay_robot, replay_gripper, rate=args.rate,
                            max_joint_speed=args.max_joint_speed,
                            close_settle_s=args.gripper_close_time,
                            open_settle_s=args.gripper_open_time,
                            smoothing=args.smoothing)
        result = replayer.run(trajectory, speed_scale=args.speed_scale, confirm=None)
        print('回放结果：', result.as_dict(), flush=True)
        return 0 if not result.aborted else 4
    except Exception as exc:
        print(f'回放失败：{type(exc).__name__}: {exc}', flush=True)
        return 1
    finally:
        if replay_gripper is not None:
            try:
                replay_gripper.stop()
            except Exception:
                pass
        try:
            replay_robot.disconnect()
        except Exception:
            pass


if __name__ == '__main__':
    raise SystemExit(main())
