"""End-to-end offline example: teleop -> teach -> replay -> collect.

Run from the repository root:

    python -m projects.robot_toolkit.examples.mock_pipeline --out outputs/mock_pipeline

It uses a first-order joint model with trivial forward/inverse kinematics, so it
runs anywhere NumPy is installed and never touches hardware.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ..camera import MockCamera
from ..collect import CameraSpec, DatasetSchema, EpisodeRecorder, export_manifest
from ..core.conventions import GripperSpec, RobotSpec
from ..core.mathutils import rotvec_to_matrix
from ..gripper import MockGripper
from ..robot import MockRobot
from ..teach import Replayer, TeachRecorder, save_trajectory, summarize
from ..teleop import CartesianMappingConfig, CartesianTeleop, TeleopCommand


def kinematics():
    scale = 0.1
    offset = np.array([0.3, 0.0, 0.3])

    def forward(joints):
        joints = np.asarray(joints, dtype=float)
        return np.array([offset[0] + scale * joints[0],
                         offset[1] + scale * joints[1],
                         offset[2] + scale * joints[2], 0.0, 0.0, 0.0])

    def inverse(pose):
        joints = np.zeros(6)
        joints[:3] = (np.asarray(pose, dtype=float)[:3] - offset) / scale
        return joints

    return forward, inverse


def run(output: Path) -> dict:
    forward, inverse = kinematics()
    spec = RobotSpec(name='mock_arm', joint_names=RobotSpec.generic(6).joint_names,
                     gripper=GripperSpec(name='mock_gripper', max_width=0.085))
    robot = MockRobot(spec, forward_kinematics=forward, inverse_kinematics=inverse)
    gripper = MockGripper(name=spec.gripper.name, max_width=spec.gripper.max_width)

    mapping = CartesianTeleop(CartesianMappingConfig(), spec.gripper)
    state = robot.state()
    mapping.reset(state.tcp[:3], rotvec_to_matrix(state.tcp[3:]), 0.085, enabled=True)

    recorder = TeachRecorder(robot, gripper, rate=60.0)
    recorder.start()
    dt = 1.0 / 50.0
    for step in range(int(2.0 / dt)):
        elapsed = step * dt
        xyz = (1.0, 0.0, 0.0) if elapsed < 0.7 else ((0.0, 1.0, 0.0) if elapsed < 1.3
                                                     else (0.0, 0.0, -1.0))
        command = TeleopCommand(t=elapsed, xyz=xyz, grip=-1.0 if 0.8 <= elapsed < 1.4 else 0.0,
                                enable=True if step == 0 else None)
        sample = robot.state()
        target = mapping.update(command, dt, measured_position=sample.tcp[:3],
                                measured_rotation=rotvec_to_matrix(sample.tcp[3:]))
        robot.servo_l(target.pose, dt)
        gripper.move(target.gripper)
        if abs(elapsed - 1.42) < dt / 2:
            recorder.mark('close', label='grasp')
        time.sleep(dt)
    trajectory = recorder.stop()
    save_trajectory(output / 'trajectories', trajectory)

    playback_robot = MockRobot(spec, forward_kinematics=forward, inverse_kinematics=inverse)
    playback_gripper = MockGripper(name=spec.gripper.name, max_width=spec.gripper.max_width)
    replay = Replayer(playback_robot, playback_gripper, rate=125.0).run(
        trajectory, speed_scale=1.0, realtime=False)

    schema = DatasetSchema(robot=spec, cameras=(CameraSpec('wrist_camera', 64, 48),), fps=10)
    camera = MockCamera('wrist_camera', 64, 48, depth=True).open()
    collector = EpisodeRecorder(output / 'collect', schema, export=export_manifest)
    collector.start('offline example', meta={'teleoperator': 'scripted'})
    for index in range(20):
        frame = camera.read()
        sample = playback_robot.state()
        vector = schema.observation(sample.q, playback_gripper.width)
        collector.capture(vector, vector, images={camera.name: frame.rgb},
                          depth={camera.name: frame.depth,
                                 f'{camera.name}_K': frame.intrinsics,
                                 f'{camera.name}_T_world_camera': np.eye(4)},
                          timestamp=index / schema.fps)
    collector.save(success=True)
    collector.close()
    camera.close()

    summary = {'trajectory': summarize(trajectory), 'replay': replay.as_dict(),
               'saved_episodes': collector.saved}
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                         encoding='utf-8')
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='outputs/mock_pipeline')
    args = parser.parse_args(argv)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    summary = run(output)
    print(json.dumps({'output': str(output),
                      'samples': summary['trajectory']['samples'],
                      'replay_frames': summary['replay']['frames'],
                      'episodes': summary['saved_episodes']}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
