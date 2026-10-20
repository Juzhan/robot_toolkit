"""Collect LeRobot episodes from a Piper teach trajectory or Joy-Con teleop.

Two sources:

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
from pathlib import Path

from ..collect import (CameraSpec, DatasetSchema, EpisodeCollector,
                       collect_trajectory, exporter_for)
from ..core.conventions import RobotSpec
from ..profiles import build_camera_hub
from ..teach import load_trajectory
from ..teleop import TeleopSessionConfig, build_teleop_session


def _cameras(path):
    if not path:
        return (), None
    entries = json.loads(Path(path).read_text(encoding='utf-8'))
    hub = build_camera_hub(entries)
    return tuple(CameraSpec.from_camera(camera) for camera in hub), hub


def _trajectory_mode(args):
    if args.cameras_json:
        raise SystemExit('trajectory export has no camera frames; '
                         'use --source teleop for camera datasets')
    trajectory = load_trajectory(Path(args.trajectory_root) / args.name)
    schema = DatasetSchema.for_robot(
        RobotSpec.piper_x(), cameras=(), fps=args.fps, repo_id=args.repo_id,
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
        repo_id=args.repo_id,
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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description='Collect a LeRobot dataset on Piper X')
    parser.add_argument('--source', default='teleop', choices=['teleop', 'trajectory'])
    parser.add_argument('--input', default='joycon', choices=['joycon', 'keyboard'])
    parser.add_argument('--task', default='piper task')
    parser.add_argument('--dataset-root', default='outputs/piper_dataset')
    parser.add_argument('--repo-id', default='local/piper_dataset')
    parser.add_argument('--fps', type=float, default=10.0)
    parser.add_argument('--export', default='lerobot', choices=['lerobot', 'manifest'])
    parser.add_argument('--cameras-json', default='', help='JSON list of camera entries')
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
    return _teleop_mode(args)


if __name__ == '__main__':
    raise SystemExit(main())
