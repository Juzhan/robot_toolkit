"""Unified command line: internal checks, demos and calibration helpers.

    python -m projects.robot_toolkit.cli info
    python -m projects.robot_toolkit.cli doctor
    python -m projects.robot_toolkit.cli selftest
    python -m projects.robot_toolkit.cli demo --out outputs/robot_toolkit_demo
    python -m projects.robot_toolkit.cli calib-handeye --demo
    python -m projects.robot_toolkit.cli calib-intrinsics --synthetic
    python -m projects.robot_toolkit.cli calib-show path/to/calibration.json

Nothing in the demo touches hardware.
"""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from . import __version__
from .calibrate import CalibrationStore, estimate_intrinsics, solve_hand_eye
from .calibrate.intrinsics import project_points
from .collect import (CameraSpec, DatasetSchema, EpisodeRecorder, export_manifest)
from .core.conventions import GripperSpec, RobotSpec
from .core.mathutils import (invert_transform, matrix_to_pose, pose_to_matrix,
                             rotvec_to_matrix)
from .gripper import MockGripper, RobotiqConfig
from .robot import MockRobot
from .teleop import CartesianMappingConfig, CartesianTeleop, TeleopCommand

ROOT = Path(__file__).resolve().parents[2]


def _json_default(value):
    if hasattr(value, 'tolist'):
        return value.tolist()
    if hasattr(value, 'item'):
        return value.item()
    return str(value)


# --------------------------------------------------------------------- helpers
def _random_transform(rng, translation_scale=0.4) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, :3] = rotvec_to_matrix(rng.normal(0.0, 0.6, 3))
    matrix[:3, 3] = rng.uniform(-translation_scale, translation_scale, 3)
    return matrix


def synthetic_hand_eye(mode: str = 'eye_in_hand', samples: int = 12, seed: int = 3):
    rng = np.random.default_rng(seed)
    truth = _random_transform(rng, 0.08)
    truth[:3, 3] = [0.05, -0.03, 0.09]
    base_tcp, cam_target = [], []
    if mode == 'eye_in_hand':
        fixed_target = _random_transform(rng, 0.05)
        fixed_target[:3, 3] = [0.0, 0.0, 0.5]
        for _ in range(samples):
            robot = _random_transform(rng, 0.15)
            robot[:3, 3] += [0.3, 0.0, 0.3]
            base_tcp.append(robot)
            cam_target.append(invert_transform(robot @ truth) @ fixed_target)
    else:
        target_in_tcp = _random_transform(rng, 0.05)
        for _ in range(samples):
            robot = _random_transform(rng, 0.15)
            robot[:3, 3] += [0.3, 0.0, 0.3]
            base_tcp.append(robot)
            cam_target.append(invert_transform(truth) @ robot @ target_in_tcp)
    return solve_hand_eye(mode, base_tcp, cam_target), truth


def synthetic_intrinsics(image_size=(640, 480), views: int = 16, seed: int = 7):
    rng = np.random.default_rng(seed)
    K = np.array([[620.0, 1.5, 322.0], [0.0, 615.0, 238.0], [0.0, 0.0, 1.0]])
    columns, rows, square = 8, 6, 0.025
    grid = np.zeros((columns * rows, 3))
    grid[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2) * square
    samples = []
    while len(samples) < views:
        rotation = rotvec_to_matrix(rng.normal(0.0, 0.35, 3))
        translation = np.array([rng.uniform(-0.05, 0.05), rng.uniform(-0.04, 0.04),
                                rng.uniform(0.35, 0.6)])
        points = grid @ rotation.T + translation
        if np.any(points[:, 2] <= 0.1):
            continue
        image = project_points(grid, K, None, rotation, translation)
        if (np.any(image[:, 0] < 5) or np.any(image[:, 0] > image_size[0] - 5) or
                np.any(image[:, 1] < 5) or np.any(image[:, 1] > image_size[1] - 5)):
            continue
        samples.append({'object_points': grid, 'image_points': image})
    return estimate_intrinsics(samples, image_size=image_size), K, samples


def _mock_kinematics():
    scale = 0.1
    offset = np.array([0.3, 0.0, 0.3])

    def forward(joints):
        joints = np.asarray(joints, dtype=float)
        return np.array([offset[0] + scale * joints[0],
                         offset[1] + scale * joints[1],
                         offset[2] + scale * joints[2], 0.0, 0.0, 0.0])

    def inverse(pose):
        pose = np.asarray(pose, dtype=float)
        joints = np.zeros(6)
        joints[:3] = (pose[:3] - offset) / scale
        return joints

    return forward, inverse


# ----------------------------------------------------------------------- info
def cmd_info(args) -> int:
    spec = RobotSpec.ur5e()
    print(f'robot_toolkit {__version__}')
    print(f'contract dim : {spec.dim}')
    print(f'state/action : {spec.state_names}')
    print(f'gripper      : {spec.gripper}')
    print('backends     : mock, ur_rtde, agx_piper, mujoco')
    print('grippers     : mock, robotiq (mock/urscript/urcap_socket/tool_modbus/modbus_rtu), agx')
    print('cameras      : mock, opencv, realsense')
    print('piper_x      :', RobotSpec.piper_x().state_names)
    return 0


def cmd_doctor(args) -> int:
    modules = ['numpy', 'scipy', 'cv2', 'mujoco', 'lerobot', 'ur_rtde', 'pyrealsense2',
               'pymodbus', 'can', 'pyAgxArm']
    for name in modules:
        try:
            module = importlib.import_module(name)
            version = getattr(module, '__version__', 'unknown')
        except Exception as exc:
            version = f'MISSING ({type(exc).__name__})'
        print(f'{name:14s} {version}')
    print(f'python       {sys.version.split()[0]}')
    print(f'platform     {platform.system()} {platform.release()} ({platform.machine()})')
    print(f'robotiq port  {RobotiqConfig().serial_port}')
    print('piper CAN    Windows agx_cando/0; Linux socketcan/can0; macOS slcan/ttyACM0')
    return 0


# -------------------------------------------------------------------- selftest
def _check(results, name, function):
    try:
        detail = function()
        results.append({'check': name, 'ok': True, 'detail': detail})
        print(f'[ok]   {name}')
    except Exception as exc:
        results.append({'check': name, 'ok': False, 'detail': f'{type(exc).__name__}: {exc}'})
        print(f'[FAIL] {name}: {type(exc).__name__}: {exc}')


def cmd_selftest(args) -> int:
    results = []

    def pose_roundtrip():
        pose = np.array([0.1, -0.2, 0.3, 1.2, -0.4, 2.7])
        recovered = matrix_to_pose(pose_to_matrix(pose))
        return float(np.max(np.abs(recovered - pose)))

    def hand_eye(mode):
        result, truth = synthetic_hand_eye(mode)
        error = float(np.max(np.abs(result.transform - truth)))
        if error > 1e-6:
            raise AssertionError(f'{mode} error {error:.2e}')
        return error

    def intrinsics():
        result, truth, _ = synthetic_intrinsics()
        relative = float(max(abs(result.K[0, 0] / truth[0, 0] - 1),
                             abs(result.K[1, 1] / truth[1, 1] - 1)))
        if relative > 0.02:
            raise AssertionError(f'focal length error {relative:.2%}')
        return relative

    def trajectory():
        from .teach import Trajectory, load_trajectory, save_trajectory
        t = np.arange(0.0, 1.0, 0.01)
        with tempfile.TemporaryDirectory() as folder:
            trajectory_in = Trajectory(t=t, q=np.sin(t)[:, None], grip=np.zeros(len(t)))
            save_trajectory(folder, trajectory_in)
            loaded = load_trajectory(folder)
            return float(np.max(np.abs(loaded.q - trajectory_in.q)))

    def control_loop():
        forward, inverse = _mock_kinematics()
        robot = MockRobot(RobotSpec.generic(6, 'selftest'), forward_kinematics=forward,
                          inverse_kinematics=inverse)
        teleop = CartesianTeleop(CartesianMappingConfig(), GripperSpec())
        state = robot.state()
        teleop.reset(state.tcp[:3], rotvec_to_matrix(state.tcp[3:]), 0.05, enabled=True)
        target = teleop.update(TeleopCommand(t=0.0, xyz=(1, 0, 0)), 0.1)
        if not robot.servo_l(target.pose, 0.1):
            raise AssertionError('servo_l failed')
        return float(np.linalg.norm(robot.state().q))

    def recorder():
        from .collect import export_manifest
        schema = DatasetSchema(cameras=(CameraSpec('wrist_camera', 32, 24),))
        with tempfile.TemporaryDirectory() as folder:
            recorder_ = EpisodeRecorder(folder, schema, export=export_manifest)
            recorder_.start('selftest')
            image = np.zeros((24, 32, 3), dtype=np.uint8)
            for index in range(4):
                recorder_.capture(np.zeros(schema.dim), np.zeros(schema.dim),
                                  images={'wrist_camera': image})
            recorder_.save(success=True)
            recorder_.close()
            return (Path(folder) / recorder_.session / 'dataset' / 'episodes.jsonl').exists()

    _check(results, 'pose roundtrip', pose_roundtrip)
    _check(results, 'hand-eye eye_in_hand', lambda: hand_eye('eye_in_hand'))
    _check(results, 'hand-eye eye_to_hand', lambda: hand_eye('eye_to_hand'))
    _check(results, 'intrinsics (Zhang)', intrinsics)
    _check(results, 'trajectory roundtrip', trajectory)
    _check(results, 'mock control loop', control_loop)
    _check(results, 'episode recorder', recorder)
    failed = [item for item in results if not item['ok']]
    print(f'\n{len(results) - len(failed)}/{len(results)} checks passed')
    if args.json:
        Path(args.json).write_text(json.dumps(results, ensure_ascii=False, indent=2),
                                   encoding='utf-8')
    return 1 if failed else 0


# ----------------------------------------------------------------------- piper
def cmd_piper_check(args) -> int:
    from .robot import AgxPiperConfig, AgxPiperRobot

    config = AgxPiperConfig(
        model=args.model,
        interface=args.interface,
        channel=args.channel,
        firmware=args.firmware,
        speed_percent=args.speed_percent,
        enable_on_connect=False,
        has_gripper=not args.no_gripper,
        mock=bool(args.mock),
    )
    robot = AgxPiperRobot(config, mock=bool(args.mock))
    try:
        robot.connect()
        state = robot.state()
        payload = robot.describe()
        payload['state'] = {
            'q': None if state.q is None else state.q.tolist(),
            'qd': None if state.qd is None else state.qd.tolist(),
            'tcp': None if state.tcp is None else state.tcp.tolist(),
            'protective_stop': state.protective_stop,
            'extra': state.extra,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default))
        return 0 if state.q is not None else 1
    finally:
        robot.disconnect()


# ----------------------------------------------------------------------- demo
def cmd_demo(args) -> int:
    output = Path(args.out) if args.out else ROOT / 'outputs' / 'robot_toolkit_demo'
    output.mkdir(parents=True, exist_ok=True)
    forward, inverse = _mock_kinematics()
    spec = RobotSpec.generic(6, 'mock_arm')
    spec = RobotSpec(name='mock_arm', joint_names=spec.joint_names,
                     gripper=GripperSpec(name='mock_gripper', max_width=0.085))
    robot = MockRobot(spec, forward_kinematics=forward, inverse_kinematics=inverse)
    gripper = MockGripper(name=spec.gripper.name, max_width=spec.gripper.max_width)
    from .teach import Replayer, TeachRecorder, TeachRecorderConfig, save_trajectory, summarize

    mapping = CartesianTeleop(CartesianMappingConfig(), spec.gripper)
    state = robot.state()
    mapping.reset(state.tcp[:3], rotvec_to_matrix(state.tcp[3:]), 0.085, enabled=True)

    recorder = TeachRecorder(robot, gripper, TeachRecorderConfig(rate=60.0))
    recorder.start()
    rate = 50.0
    dt = 1.0 / rate
    steps = int(3.0 * rate)
    for step in range(steps):
        elapsed = step * dt
        xyz = (0.0, 0.0, 0.0)
        if elapsed < 0.8:
            xyz = (1.0, 0.0, 0.0)
        elif elapsed < 1.4:
            xyz = (0.0, 0.0, -1.0)
        elif elapsed < 2.0:
            xyz = (0.0, 1.0, 0.0)
        grip = -1.0 if 1.0 <= elapsed < 1.6 else 0.0
        command = TeleopCommand(t=elapsed, xyz=xyz, grip=grip,
                                enable=(True if step == 0 else None))
        sample = robot.state()
        target = mapping.update(command, dt, measured_position=sample.tcp[:3],
                                measured_rotation=rotvec_to_matrix(sample.tcp[3:]))
        robot.servo_l(target.pose, dt)
        gripper.move(target.gripper)
        if abs(elapsed - 1.62) < dt / 2:
            recorder.mark('close', label='grasp')
        if abs(elapsed - 2.02) < dt / 2:
            recorder.mark('open', label='release')
        time.sleep(dt)
    trajectory = recorder.stop()
    trajectory_dir = output / 'trajectories'
    save_trajectory(trajectory_dir, trajectory)
    print('teach:', json.dumps(summarize(trajectory), ensure_ascii=False))

    playback_robot = MockRobot(spec, forward_kinematics=forward, inverse_kinematics=inverse)
    playback_gripper = MockGripper(name=spec.gripper.name, max_width=spec.gripper.max_width)
    replayer = Replayer(playback_robot, playback_gripper, rate=125.0)
    result = replayer.run(trajectory, speed_scale=0.9, realtime=False)
    print('replay:', json.dumps(result.as_dict(), ensure_ascii=False))

    schema = DatasetSchema(robot=spec, cameras=(CameraSpec('wrist_camera', 64, 48),), fps=10,
                           include_time=True)
    from .camera import MockCamera
    camera = MockCamera('wrist_camera', 64, 48, depth=True).open()
    collector = EpisodeRecorder(output / 'collect', schema, export=export_manifest)
    collector.start('demo pick and place', meta={'teleoperator': 'scripted'})
    for index in range(30):
        frame = camera.read()
        sample = playback_robot.state()
        state_vector = schema.observation(sample.q, playback_gripper.width)
        collector.capture(state_vector, state_vector,
                          images={camera.name: frame.rgb},
                          depth={camera.name: frame.depth,
                                 f'{camera.name}_K': frame.intrinsics,
                                 f'{camera.name}_T_world_camera': np.eye(4)},
                          timestamp=index / schema.fps)
    collector.save(success=True)
    collector.close()
    camera.close()
    summary = {
        'trajectory': summarize(trajectory),
        'replay': result.as_dict(),
        'collector': collector.status().as_dict(),
        'output': str(output),
    }
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                         encoding='utf-8')
    print(json.dumps({'output': str(output), 'frames': trajectory.meta.get('samples'),
                      'replay_frames': result.frames, 'episode_saved': collector.saved},
                     ensure_ascii=False))
    return 0


# ---------------------------------------------------------------- calibration
def cmd_calib_handeye(args) -> int:
    if not args.demo:
        print('only --demo is implemented in the toolkit CLI; '
              'call solve_hand_eye() with your recorded poses for real data')
        return 2
    payload = {}
    for mode in ('eye_in_hand', 'eye_to_hand'):
        result, truth = synthetic_hand_eye(mode, samples=args.samples, seed=args.seed)
        payload[mode] = {'result': result.to_dict(),
                         'truth_error': float(np.max(np.abs(result.transform - truth)))}
        print(f'{mode}: rms_rot={result.rotation_rms:.2e} rms_trans={result.translation_rms:.2e} '
              f'truth_error={payload[mode]["truth_error"]:.2e}')
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                  encoding='utf-8')
    return 0


def cmd_calib_intrinsics(args) -> int:
    if not args.synthetic:
        print('only --synthetic is implemented in the toolkit CLI; '
              'call calibrate_checkerboard() or estimate_intrinsics() for real images')
        return 2
    result, truth, views = synthetic_intrinsics(seed=args.seed)
    payload = {'result': result.to_dict(),
               'truth': {'K': truth.tolist()},
               'focal_error': [float(result.K[0, 0] / truth[0, 0] - 1),
                               float(result.K[1, 1] / truth[1, 1] - 1)],
               'principal_error_px': [float(result.K[0, 2] - truth[0, 2]),
                                      float(result.K[1, 2] - truth[1, 2])]}
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                  encoding='utf-8')
    return 0


def cmd_calib_show(args) -> int:
    store = CalibrationStore(args.path)
    print(json.dumps(store.summary(), ensure_ascii=False, indent=2))
    return 0


# ----------------------------------------------------------------------- main
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='robot_toolkit',
                                     description='Robot-agnostic teleop / teach / collect / calib toolkit')
    parser.add_argument('--version', action='version', version=__version__)
    subparsers = parser.add_subparsers(dest='command')

    info = subparsers.add_parser('info', help='print the shared contract')
    info.set_defaults(func=cmd_info)

    doctor = subparsers.add_parser('doctor', help='show installed optional dependencies')
    doctor.set_defaults(func=cmd_doctor)

    selftest = subparsers.add_parser('selftest', help='run dependency-free internal checks')
    selftest.add_argument('--json', help='write the check results to this file')
    selftest.set_defaults(func=cmd_selftest)

    piper = subparsers.add_parser('piper-check', help='read Piper X state over CAN (no motion)')
    piper.add_argument('--model', default='piper_x',
                       choices=['piper', 'piper_h', 'piper_l', 'piper_x'])
    piper.add_argument('--interface', default='auto', help='auto / agx_cando / socketcan / slcan')
    piper.add_argument('--channel', default='auto', help='auto / 0 / can0 / /dev/ttyACM0')
    piper.add_argument('--firmware', default='auto', choices=['auto', 'default', 'v183', 'v188', 'v189'])
    piper.add_argument('--speed-percent', type=int, default=50)
    piper.add_argument('--no-gripper', action='store_true')
    piper.add_argument('--mock', action='store_true', help='exercise the backend without CAN')
    piper.set_defaults(func=cmd_piper_check)

    demo = subparsers.add_parser('demo', help='run an offline teach/replay/collect demo')
    demo.add_argument('--out', help='output directory')
    demo.set_defaults(func=cmd_demo)

    handeye = subparsers.add_parser('calib-handeye', help='hand-eye calibration')
    handeye.add_argument('--demo', action='store_true', help='use synthetic pose pairs')
    handeye.add_argument('--samples', type=int, default=12)
    handeye.add_argument('--seed', type=int, default=3)
    handeye.add_argument('--out')
    handeye.set_defaults(func=cmd_calib_handeye)

    intrinsics = subparsers.add_parser('calib-intrinsics', help='camera intrinsics calibration')
    intrinsics.add_argument('--synthetic', action='store_true')
    intrinsics.add_argument('--seed', type=int, default=7)
    intrinsics.add_argument('--out')
    intrinsics.set_defaults(func=cmd_calib_intrinsics)

    show = subparsers.add_parser('calib-show', help='summarize a calibration store')
    show.add_argument('path')
    show.set_defaults(func=cmd_calib_show)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, 'command', None):
        parser.print_help()
        return 0
    return int(args.func(args) or 0)


if __name__ == '__main__':
    raise SystemExit(main())
