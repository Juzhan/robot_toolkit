"""Collect LeRobot episodes from a Piper teach trajectory, master-slave teach or Joy-Con teleop.

Three sources:

* ``--source master-slave``: the leader/follower pair already runs its own teach
  mode; this only *reads* follower state + the leader's broadcast joints and
  records them with the camera images.  Nothing is commanded.
* ``--source trajectory``: turn one trajectory saved by ``piper_teach`` into a
  robot-only LeRobot dataset (state -> next state).
* ``--source teleop``: run the shared Joy-Con teleop loop and record live
  state/action plus an optional camera JSON configuration.

Camera JSON is a list accepted by ``profiles.build_camera_hub``, for example::

    [
      {"kind": "opencv", "name": "wrist_camera", "width": 640,
       "height": 480, "fps": 30, "depth": false},
      {"kind": "realsense", "name": "external_camera", "width": 640,
       "height": 480, "fps": 30, "depth": true}
    ]

The exporter is LeRobot's official writer by default; use ``--export manifest``
when LeRobot is not installed.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from ..collect import (CameraSpec, DatasetSchema, EpisodeCollector,
                       collect_trajectory, exporter_for)
from ..collect.passive import sample_episodes
from ..core.conventions import RobotSpec
from ..profiles import build_camera_hub
from ..robot import AgxPiperConfig, AgxPiperRobot
from ..teach import load_trajectory
from ..teleop import TeleopSessionConfig, build_teleop_session


def _cameras(path):
    if not path:
        return (), None
    # utf-8-sig: Windows editors/PowerShell often prepend a BOM.
    entries = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    hub = build_camera_hub(entries)
    return tuple(CameraSpec.from_camera(camera) for camera in hub), hub


def _trajectory_mode(args):
    if args.cameras_json:
        raise SystemExit('trajectory export has no camera frames; '
                         'use --source teleop for camera datasets')
    trajectory = load_trajectory(Path(args.trajectory_root) / args.name)
    schema = DatasetSchema.for_robot(
        RobotSpec.piper_x(), cameras=(), fps=args.fps, repo_id=args.repo_id,
        name_style=args.feature_names,
    )
    collector = collect_trajectory(
        trajectory, schema, args.dataset_root, task=args.task,
        export=exporter_for(args.export), fps=args.fps,
    )
    collector.close()
    if collector.export_errors:
        print(f'trajectory dataset: export errors={collector.export_errors}')
        return 2
    print(f'trajectory dataset: saved={collector.saved} root={collector.dataset_root}')
    return 0 if collector.saved else 1


def _teleop_mode(args):
    camera_specs, cameras = _cameras(args.cameras_json)
    schema = DatasetSchema.for_robot(
        RobotSpec.piper_x(), cameras=camera_specs, fps=args.fps,
        repo_id=args.repo_id, name_style=args.feature_names,
    )
    session = build_teleop_session(TeleopSessionConfig(
        robot='mock' if args.mock else 'piper_x',
        input=args.input,
        mock=args.mock,
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
        calibrate=args.calibrate,
    ))
    collector = EpisodeCollector(
        args.dataset_root, schema, robot=session.robot, gripper=session.gripper,
        cameras=() if cameras is None else cameras,
        export=exporter_for(args.export), fps=args.fps,
        min_interval_s=1.0 / max(args.fps, 1e-9),
        compress=bool(args.compress),
    )
    started = [False]

    def on_sample(state, target):
        if not started[0]:
            collector.start(args.task, meta={'source': 'joycon_teleop'})
            started[0] = True
        collector.capture_state(state=state)

    result = None
    try:
        result = session.run(on_sample=on_sample)
    finally:
        if started[0] and collector.active:
            try:
                collector.stop(success=bool(result is not None and not result.aborted))
            except Exception:
                pass
        collector.close()
        session.close()
    print('teleop collection:', result.as_dict() if result is not None else 'interrupted')
    print(f'dataset root: {collector.dataset_root}')
    return 0 if started[0] and not collector.export_errors else 2


def _master_slave_mode(args) -> int:
    """Sample a leader/follower pair that already runs its own teach mode.

    Nothing is commanded: the backend connects with ``configure_on_connect=False``
    so no ``0x151`` mode frame hits the shared bus, and the follower state plus
    the leader's broadcast joints are recorded as state/action.
    """
    camera_specs, cameras = _cameras(args.cameras_json)
    schema = DatasetSchema.for_robot(
        RobotSpec.piper_x(), cameras=camera_specs, fps=args.fps,
        repo_id=args.repo_id, name_style=args.feature_names,
    )
    robot = AgxPiperRobot(AgxPiperConfig(
        model='piper_x', interface=args.interface, channel=args.channel,
        firmware=args.firmware,
        enable_on_connect=False, configure_on_connect=False, mock=bool(args.mock),
    ))
    if args.mock:
        print('mock 模式：没有真实手臂/相机，只验证采样链路')
    print('主从示教采样（全程不发送报文）：空格=开始/结束一条 episode，x=丢弃，q=退出')
    try:
        robot.connect()
        # The sampling window starts once the link is ready: connect() itself
        # can take a couple of seconds.
        stats = sample_episodes(
            robot, schema, args.dataset_root, cameras=() if cameras is None else cameras,
            export=exporter_for(args.export), fps=args.fps, task=args.task,
            auto_record=args.auto_record, sample_seconds=args.sample_seconds,
            compress=bool(args.compress),
        )
    finally:
        robot.disconnect()
    print(f'采样结束：episodes={stats["episodes"]} frames={stats["frames"]} '
          f'rate={stats.get("rate", 0.0):.1f} Hz saved={stats["saved"]} '
          f'skipped={stats["skipped"]} root={stats["dataset_root"]}')
    for error in stats.get('export_errors') or ():
        print(f'导出失败：{error}')
    return 0 if stats['saved'] else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description='Collect a LeRobot dataset on Piper X')
    parser.add_argument('--source', default='teleop',
                        choices=['teleop', 'trajectory', 'master-slave'])
    parser.add_argument('--input', default='joycon', choices=['joycon', 'keyboard'])
    parser.add_argument('--task', default='piper task')
    parser.add_argument('--dataset-root', default='outputs/piper_dataset')
    parser.add_argument('--repo-id', default='local/piper_dataset')
    parser.add_argument('--fps', type=float, default=10.0)
    parser.add_argument('--export', default='lerobot', choices=['lerobot', 'manifest'])
    parser.add_argument('--feature-names', default='unit', choices=['unit', 'pos'],
                        help='unit: joint_1.rad / piper_gripper.width_m; '
                             'pos: LeRobot 风格 joint_1.pos / gripper.pos（仅改标签）')
    parser.add_argument('--cameras-json', default='', help='JSON list of camera entries')
    parser.add_argument('--interface', default='auto', help='auto / agx_cando / socketcan / slcan')
    parser.add_argument('--channel', default='auto', help='auto / 0 / can0 / /dev/ttyACM0')
    parser.add_argument('--sample-seconds', type=float, default=0.0,
                        help='master-slave 采样时长（0 = 一直采到按 q）')
    parser.add_argument('--firmware', default='default',
                        choices=['auto', 'default', 'v183', 'v188', 'v189'],
                        help='master-slave 采样不需要固件探测，默认 default 以省掉 2s 等待')
    parser.add_argument('--auto-record', action='store_true',
                        help='master-slave 模式启动即开始录制（无需按键）')
    parser.add_argument('--compress', action='store_true',
                        help='压缩 spool 的 .npz 帧（更省盘，但每帧慢约 20 倍）')
    parser.add_argument('--name', default='piper_02', help='piper_teach trajectory name')
    parser.add_argument('--trajectory-root', default='outputs/piper_teach')
    parser.add_argument('--duration', type=float, default=60.0)
    parser.add_argument('--rate', type=float, default=50.0)
    parser.add_argument('--move-p-hz', type=float, default=20.0)
    parser.add_argument('--stale-s', type=float, default=0.4)
    parser.add_argument('--speed', type=float, default=0.08)
    parser.add_argument('--smoothing', type=float, default=0.2)
    parser.add_argument('--linear-accel', type=float, default=0.6)
    parser.add_argument('--angular-accel', type=float, default=1.0)
    parser.add_argument('--speed-percent', type=int, default=20)
    parser.add_argument('--max-joint-speed', type=float, default=1.0)
    parser.add_argument('--gripper-close-time', type=float, default=0.8)
    parser.add_argument('--gripper-open-time', type=float, default=0.5)
    parser.add_argument('--gripper-button', default='b', choices=['b', 'zr', 'x', 'y', 'r'])
    parser.add_argument('--calibrate', action='store_true')
    parser.add_argument('--mock', action='store_true')
    parser.add_argument('--yes', action='store_true', help='confirm real motion')
    args = parser.parse_args(argv)
    if args.source == 'teleop' and not args.mock and not args.yes:
        parser.error('real Piper teleop collection requires --yes')
    if args.source == 'trajectory':
        return _trajectory_mode(args)
    if args.source == 'master-slave':
        return _master_slave_mode(args)
    return _teleop_mode(args)


if __name__ == '__main__':
    raise SystemExit(main())
