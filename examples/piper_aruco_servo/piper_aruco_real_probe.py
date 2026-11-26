"""Real-robot ArUco PBVS/IBVS probe: read-only by default, explicit motion only.

This is the real-hardware counterpart of the Isaac visual-servo scene. It uses
the existing robot_toolkit Piper backend, RealSense driver and the saved
``tcp_camera`` calibration. It does not import Isaac and does not use ROS.

Safety defaults:
* no motion unless ``--execute --yes`` is supplied;
* ``--read-only`` and ``--dry-run`` connect without enabling or moving;
* every motion step is a small Cartesian target with a per-step clamp;
* the arm is left enabled and holding on exit; use the physical e-stop.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[2]          # the robot_toolkit repository
if __package__ in (None, '') and str(REPO.parent) not in sys.path:
    sys.path.insert(0, str(REPO.parent))            # run from a plain checkout, uninstalled

from robot_toolkit.core.realtime import collect_idle, pause_gc  # noqa: E402

# Hand-eye result of the Piper X wrist D435 (eye-in-hand, 2026-10-20); recalibrate
# if the camera mount changes.
DEFAULT_CALIBRATION = REPO / 'configs/calibration/piper_x_wrist_d435.json'


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--read-only', action='store_true',
                      help='connect read-only, detect the marker, do not move (default)')
    mode.add_argument('--dry-run', action='store_true',
                      help='read robot/camera, compute target and first step, do not move')
    mode.add_argument('--execute', action='store_true',
                      help='enable and execute small PBVS steps; requires --yes')
    parser.add_argument('--yes', action='store_true',
                        help='confirm real motion for --execute')
    parser.add_argument('--list-cameras', action='store_true',
                        help='list RealSense devices and exit')

    parser.add_argument('--calibration', type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument('--camera-name', default='wrist_camera')
    parser.add_argument('--serial', default='', help='RealSense serial; empty = first device')
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--exposure-ms', type=float, default=None,
                        help='manual colour exposure in ms (short = less motion blur); default auto')
    parser.add_argument('--camera-gain', type=float, default=None,
                        help='colour sensor gain 0-128 (raise with short exposure)')

    parser.add_argument('--model', default='piper_x',
                        choices=['piper', 'piper_h', 'piper_l', 'piper_x'])
    parser.add_argument('--interface', default='auto')
    parser.add_argument('--channel', default='auto')
    parser.add_argument('--speed-percent', type=int, default=10,
                        help='Piper controller speed percentage; keep low for the first test')

    parser.add_argument('--dictionary', default='DICT_4X4_50')
    parser.add_argument('--marker-length', type=float, default=.06)
    parser.add_argument('--distance', type=float, default=.30,
                        help='fixed desired camera-marker distance in m; 0 = use current distance')
    parser.add_argument('--roll-deg', type=float, default=0.)
    parser.add_argument('--target-tolerance-m', type=float, default=.003)
    parser.add_argument('--target-tolerance-rad', type=float, default=.02)
    parser.add_argument('--max-target-offset-m', type=float, default=.12,
                        help='warn (do not refuse) when the first target is farther than this')
    parser.add_argument('--max-target-offset-rad', type=float, default=.50)
    parser.add_argument('--allow-large', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--motion', choices=['step', 'servo'], default='step',
                        help='step = native move_p per cycle; servo = streamed servo_j')
    parser.add_argument('--control', choices=['pbvs', 'ibvs'], default='pbvs')
    parser.add_argument('--marker-id', type=int, default=0)
    parser.add_argument('--ibvs-gain', type=float, default=.8)
    parser.add_argument('--ibvs-joint-speed', type=float, default=None,
                        help='legacy alias for --reference-speed; conflicting values are rejected')
    parser.add_argument('--ibvs-linear-speed', type=float, default=.03)
    parser.add_argument('--ibvs-angular-speed', type=float, default=.2)
    parser.add_argument('--ibvs-feedforward', type=float, default=0.,
                        help='IBVS: fraction of the estimated marker velocity fed forward (0 = off)')
    parser.add_argument('--ibvs-ff-window', type=float, default=.15,
                        help='s of marker positions fitted for the velocity estimate')
    parser.add_argument('--ibvs-ff-deadband', type=float, default=.02,
                        help='m/s; marker speed below this is treated as still')
    parser.add_argument('--ibvs-ff-max', type=float, default=.3,
                        help='m/s cap of the feedforward velocity')
    parser.add_argument('--ibvs-stop-px', type=float, default=3.)
    parser.add_argument('--ibvs-resume-px', type=float, default=6.)
    parser.add_argument('--ibvs-timeout', type=float, default=.2)
    parser.add_argument('--ibvs-duration', type=float, default=30.,
                        help='IBVS run time in s; 0 = until Ctrl-C')
    parser.add_argument('--step-m', type=float, default=.005,
                        help='maximum Cartesian step per control iteration (step mode)')
    parser.add_argument('--step-rad', type=float, default=.02,
                        help='maximum rotation step per control iteration (step mode)')
    parser.add_argument('--steps', type=int, default=20,
                        help='step mode: move_p cycles; servo mode: camera frames with the '
                             'marker (frames without it do not count); 0 = until Ctrl-C')
    parser.add_argument('--hold', action='store_true',
                        help='keep running after reaching the target; move the marker by hand to test tracking')
    parser.add_argument('--period', type=float, default=.20,
                        help='seconds between Cartesian step targets (step mode)')
    parser.add_argument('--control-hz', type=float, default=100.,
                        help='servo mode: joint reference/servo_j stream rate')
    parser.add_argument('--reference-speed', type=float, default=None,
                        help='shared PBVS/IBVS max joint speed in rad/s (default: 0.5)')
    parser.add_argument('--reference-accel', type=float, default=1.0,
                        help='max joint reference acceleration in rad/s^2')
    parser.add_argument('--reference-jerk', type=float, default=6.0,
                        help='max joint reference jerk in rad/s^3')
    parser.add_argument('--camera-latency', type=float, default=.04,
                        help='host clock: seconds from exposure to host receipt, used to pick the '
                             'arm pose of a frame (also the fallback without hardware timestamps)')
    parser.add_argument('--camera-clock', choices=['hardware', 'host'], default='hardware',
                        help='hardware: per-frame capture time from the RealSense timestamp '
                             '(mid-exposure); host: receipt time minus --camera-latency')
    parser.add_argument('--marker-alpha', type=float, default=.3,
                        help='marker pose low-pass weight per frame (1 = no filtering)')
    parser.add_argument('--goal-deadband', type=float, default=.002,
                        help='rad; ignore joint-goal changes smaller than this')
    parser.add_argument('--target-deadband-m', type=float, default=.004,
                        help='re-target only when the TCP target moves more than this')
    parser.add_argument('--target-deadband-rad', type=float, default=.03)
    parser.add_argument('--target-gain', type=float, default=.6,
                        help='fraction of each target correction applied per update')
    parser.add_argument('--lock-target-m', type=float, default=.004,
                        help='enter target lock when position error drops below this')
    parser.add_argument('--unlock-target-m', type=float, default=.006,
                        help='leave target lock when position error rises above this')
    parser.add_argument('--lock-pixel', type=float, default=8.,
                        help='enter target lock when pixel error is below this')
    parser.add_argument('--unlock-pixel', type=float, default=15.,
                        help='leave target lock when pixel error rises above this')
    parser.add_argument('--settle-time', type=float, default=.25,
                        help='s the arm must be still before a frame updates the marker')
    parser.add_argument('--settle-before-update', action='store_true',
                        help='PBVS: only use frames while the arm is still (look-then-move, stop-and-go)')
    parser.add_argument('--track-while-moving', action='store_true',
                        help=argparse.SUPPRESS)  # now the default; kept for old launchers
    parser.add_argument('--pbvs-orientation', choices=['fixed', 'marker'], default='fixed',
                        help='PBVS target orientation: fixed = keep the start orientation and only '
                             'centre the marker (stable); marker = face the marker normal (6D, the '
                             'PnP normal wobbles with the viewpoint and the target swings)')
    parser.add_argument('--lead', type=float, default=0.,
                        help='s of marker-velocity prediction added to the target')
    parser.add_argument('--lead-deadband', type=float, default=.05,
                        help='PBVS: m/s; slower marker motion gets no lead (no overshoot at stops)')
    parser.add_argument('--lead-max-m', type=float, default=.015,
                        help='PBVS: cap on the lead offset; it is what overshoots when the hand stops')
    parser.add_argument('--reacquire-gap', type=float, default=.15,
                        help='s without detection after which the marker filter restarts')
    parser.add_argument('--reacquire-frames', type=int, default=3,
                        help='PBVS: consecutive consistent detections needed to (re)start tracking')
    parser.add_argument('--max-marker-speed', type=float, default=.8,
                        help='PBVS: m/s; a detection implying a faster marker jump is rejected')
    parser.add_argument('--max-reach-m', type=float, default=.35,
                        help='PBVS: TCP target is clamped to this radius around the start TCP')
    parser.add_argument('--orientation-weight', type=float, default=.3,
                        help='position-only: IK weight of keeping the start orientation')
    parser.add_argument('--max-orientation-drift', type=float, default=.6,
                        help='rad the IK may leave the target orientation when position is met')
    parser.add_argument('--log', type=Path, default=None,
                        help='servo mode: per-frame JSONL log')
    parser.add_argument('--ik-iterations', type=int, default=15)
    parser.add_argument('--servo-damping', type=float, default=.05)
    parser.add_argument('--position-only', action='store_true',
                        help='keep the start TCP orientation; only centre the marker at the distance')
    parser.add_argument('--timeout', type=float, default=3.,
                        help='move_p arrival timeout per step (step mode)')
    parser.add_argument('--obstacles', type=Path, default=None,
                        help='JSON of virtual obstacles (robot_toolkit.core.collision); the arm, '
                             'gripper and wrist camera spheres must keep their margin_m from them')
    parser.add_argument('--collision-clearance', type=float, default=.015,
                        help='m beyond margin_m a new goal/velocity must keep; the servo stream '
                             'aborts if the commanded reference ever gets inside margin_m')
    parser.add_argument('--watch-clearance', action='store_true',
                        help='read-only (no enable): print the closest obstacle while you move the '
                             'arm by hand; Ctrl-C to quit')

    args = parser.parse_args(argv)
    if args.ibvs_joint_speed is not None:
        if args.reference_speed is not None and args.reference_speed != args.ibvs_joint_speed:
            parser.error('--ibvs-joint-speed is an alias of --reference-speed; use one value')
        args.reference_speed = args.ibvs_joint_speed
    if args.reference_speed is None:
        args.reference_speed = .5
    args.ibvs_joint_speed = args.reference_speed
    if args.execute and not args.yes:
        parser.error('--execute requires --yes; keep the physical e-stop ready')
    if args.control == 'ibvs':
        if args.motion != 'servo' or args.position_only:
            parser.error('IBVS requires --motion servo without --position-only (four-corner 6D control)')
        for name in ('distance', 'marker_length', 'control_hz', 'fps',
                     'reference_speed', 'reference_accel', 'reference_jerk', 'servo_damping',
                     'ibvs_gain', 'ibvs_joint_speed', 'ibvs_linear_speed', 'ibvs_angular_speed',
                     'ibvs_stop_px', 'ibvs_resume_px', 'ibvs_timeout'):
            value = getattr(args, name)
            if not np.isfinite(value) or value <= 0:
                parser.error(f'{name} must be finite and positive for IBVS')
        for name in ('steps', 'ibvs_duration', 'ibvs_feedforward', 'ibvs_ff_deadband'):
            value = getattr(args, name)
            if not np.isfinite(value) or value < 0:
                parser.error(f'{name} must be finite and nonnegative for IBVS (0 = no limit)')
        if args.ibvs_feedforward > 1.5 or not args.ibvs_ff_window > 0 or not args.ibvs_ff_max > 0:
            parser.error('IBVS feedforward must be <= 1.5 with positive window and cap')
        if args.ibvs_resume_px <= args.ibvs_stop_px or args.distance <= .02:
            parser.error('IBVS requires resume-px > stop-px and distance > .02 m')
        if not np.isfinite(args.camera_latency) or args.camera_latency < 0:
            parser.error('camera-latency must be finite and nonnegative')
        if not 1 <= args.speed_percent <= 100:
            parser.error('speed-percent must be between 1 and 100')
        if not np.isfinite(args.roll_deg):
            parser.error('roll-deg must be finite')
    args.track_while_moving = not args.settle_before_update
    if (args.reacquire_frames < 1 or not args.max_marker_speed > 0
            or not args.max_reach_m > 0 or not args.reacquire_gap > 0
            or args.lead < 0 or args.lead_deadband < 0 or args.lead_max_m < 0):
        parser.error('reacquire-frames >= 1; max-marker-speed, max-reach-m, reacquire-gap > 0; '
                     'lead, lead-deadband, lead-max-m >= 0')
    if args.collision_clearance < 0:
        parser.error('--collision-clearance must be >= 0')
    if args.watch_clearance and (args.execute or args.obstacles is None):
        parser.error('--watch-clearance is read-only and needs --obstacles (no --execute)')
    args.collision = None  # CollisionGuard, built in main from --obstacles
    if args.control != 'ibvs' and args.pbvs_orientation == 'fixed':
        args.position_only = True
    if not (args.read_only or args.dry_run or args.execute):
        args.read_only = True
    return args


def make_detector(args):
    import cv2

    dictionary_id = getattr(cv2.aruco, args.dictionary, None)
    if dictionary_id is None:
        raise ValueError(f'unknown ArUco dictionary {args.dictionary!r}')
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    half = args.marker_length / 2.
    object_points = np.array([[-half, half, 0.], [half, half, 0.],
                              [half, -half, 0.], [-half, -half, 0.]], np.float32)
    return cv2, cv2.aruco.ArucoDetector(dictionary, parameters), object_points


def detect_marker(cv2, detector, object_points, rgb, intrinsics, distortion, marker_id=None):
    if rgb is None or intrinsics is None:
        return None
    gray = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2GRAY)
    corners, identifiers, _ = detector.detectMarkers(gray)
    if identifiers is None or not len(identifiers):
        return None
    scores = [float(cv2.contourArea(item.reshape(-1, 2).astype(np.float32)))
              for item in corners]
    candidates = [index for index, identifier in enumerate(identifiers.reshape(-1))
                  if marker_id is None or identifier == marker_id]
    if not candidates:
        return None
    chosen = max(candidates, key=lambda index: scores[index])
    image_points = corners[chosen].reshape(4, 2).astype(np.float32)
    coeffs = None if distortion is None or not np.any(distortion) else np.asarray(distortion, float)
    ok, rvec, tvec = cv2.solvePnP(object_points, image_points, np.asarray(intrinsics, float),
                                  coeffs, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        return None
    rotation, _ = cv2.Rodrigues(rvec)
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = np.asarray(tvec).reshape(3)
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, np.asarray(intrinsics, float),
                                     coeffs)
    if not (np.isfinite(transform).all() and np.isfinite(projected).all()):
        return None
    reprojection = float(np.mean(np.linalg.norm(
        projected.reshape(4, 2) - image_points, axis=1)))
    center, _ = cv2.projectPoints(np.zeros((1, 3), np.float32), rvec, tvec,
                                  np.asarray(intrinsics, float), coeffs)
    center = center.reshape(2)
    pixel_error = float(np.linalg.norm(center - [intrinsics[0, 2], intrinsics[1, 2]]))
    return {'T_camera_marker': transform, 'corners': image_points,
            'center_uv': center, 'pixel_error_px': pixel_error,
            'reprojection_error_px': reprojection, 'area': scores[chosen]}


def desired_camera_pose(T_base_marker, current_camera, distance, roll_deg):
    """Look back at the marker from its +Z side, at a given distance."""
    marker = np.asarray(T_base_marker, float)
    normal = marker[:3, 2].copy()
    normal /= max(np.linalg.norm(normal), 1e-9)
    if distance <= 0:
        distance = float(np.linalg.norm(current_camera[:3, 3] - marker[:3, 3]))
    position = marker[:3, 3] + distance * normal

    z_axis = -normal
    x_axis = marker[:3, 0].copy()
    x_axis -= np.dot(x_axis, z_axis) * z_axis
    if np.linalg.norm(x_axis) < 1e-6:
        x_axis = np.array([1., 0., 0.]) - z_axis[0] * z_axis
    x_axis /= np.linalg.norm(x_axis)
    x_axis = Rotation.from_rotvec(z_axis * np.radians(roll_deg)).as_matrix() @ x_axis
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)

    pose = np.eye(4)
    pose[:3, :3] = np.column_stack((x_axis, y_axis, z_axis))
    pose[:3, 3] = position
    return pose, float(distance)


def pose_vector(transform):
    return np.r_[transform[:3, 3], Rotation.from_matrix(transform[:3, :3]).as_rotvec()]


def pose_error(current, desired):
    position = float(np.linalg.norm(desired[:3, 3] - current[:3, 3]))
    rotation = float(Rotation.from_matrix(
        desired[:3, :3] @ current[:3, :3].T).magnitude())
    return position, rotation


def clamp_step(current, desired, step_m, step_rad):
    delta = desired[:3, 3] - current[:3, 3]
    length = np.linalg.norm(delta)
    if length > step_m:
        delta *= step_m / length
    rotvec = Rotation.from_matrix(desired[:3, :3] @ current[:3, :3].T).as_rotvec()
    angle = np.linalg.norm(rotvec)
    if angle > step_rad:
        rotvec *= step_rad / angle
    command = np.eye(4)
    command[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix() @ current[:3, :3]
    command[:3, 3] = current[:3, 3] + delta
    return command


def pose_matrix(pose):
    pose = np.asarray(pose, float).reshape(6)
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_rotvec(pose[3:]).as_matrix()
    transform[:3, 3] = pose[:3]
    return transform


def numerical_jacobian(robot, joints, epsilon=1e-4):
    """TCP geometric Jacobian in the robot base frame via six local FK calls."""
    joints = np.asarray(joints, float).reshape(-1)
    base = robot.forward_kinematics(joints)
    base_rotation = Rotation.from_rotvec(base[3:]).as_matrix()
    jacobian = np.zeros((6, joints.size))
    for index in range(joints.size):
        shifted = joints.copy()
        shifted[index] += epsilon
        pose = robot.forward_kinematics(shifted)
        jacobian[:3, index] = (pose[:3] - base[:3]) / epsilon
        rotation = Rotation.from_rotvec(pose[3:]).as_matrix()
        jacobian[3:, index] = Rotation.from_matrix(
            rotation @ base_rotation.T).as_rotvec() / epsilon
    return jacobian


def solve_ik(robot, seed, target, iterations, damping, lower, upper, orientation_weight=1.):
    """Damped least-squares IK to an absolute TCP pose, warm-started from ``seed``."""
    q = np.asarray(seed, float).copy()
    lower = np.asarray(lower, float)[:q.size]
    upper = np.asarray(upper, float)[:q.size]
    weights = np.r_[np.ones(3), np.full(3, orientation_weight)]
    for _ in range(max(1, int(iterations))):
        current = pose_matrix(robot.forward_kinematics(q))
        error = np.r_[target[:3, 3] - current[:3, 3],
                      Rotation.from_matrix(target[:3, :3] @ current[:3, :3].T).as_rotvec()]
        if np.linalg.norm(error[:3]) < 2e-4 and np.linalg.norm(error[3:]) < 2e-3:
            break
        jacobian = weights[:, None] * numerical_jacobian(robot, q)
        step = jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T + damping ** 2 * np.eye(6), weights * error)
        q = np.clip(q + np.clip(step, -.1, .1), lower, upper)
    residual = pose_error(pose_matrix(robot.forward_kinematics(q)), target)
    return q, residual


def image_jacobian(points, depths):
    points = np.asarray(points, float).reshape(4, 2)
    depths = np.asarray(depths, float).reshape(4)
    if not np.isfinite(points).all() or not np.isfinite(depths).all() or np.any(depths <= .02):
        raise ValueError('invalid IBVS image points or depths')
    rows = []
    for (horizontal, vertical), depth in zip(points, depths):
        rows.extend([
            [-1 / depth, 0, horizontal / depth, horizontal * vertical,
             -(1 + horizontal ** 2), vertical],
            [0, -1 / depth, vertical / depth, 1 + vertical ** 2,
             -horizontal * vertical, -horizontal],
        ])
    return np.asarray(rows)


def camera_jacobian(tcp_jacobian, base_tcp, tcp_camera):
    offset = base_tcp[:3, :3] @ tcp_camera[:3, 3]
    cross = np.cross(offset, np.eye(3)).T
    camera_rotation = (base_tcp[:3, :3] @ tcp_camera[:3, :3]).T
    return np.vstack((camera_rotation @ (tcp_jacobian[:3] - cross @ tcp_jacobian[3:]),
                      camera_rotation @ tcp_jacobian[3:]))


def ibvs_features(cv2, detection, object_points, intrinsics, distortion, distance, roll_deg):
    if not np.isfinite(distance) or distance <= .02:
        raise ValueError('IBVS needs a fixed positive camera-marker distance')
    points = cv2.undistortPoints(
        np.asarray(detection['corners'], float).reshape(4, 1, 2),
        np.asarray(intrinsics, float), distortion).reshape(4, 2)
    desired_rotation = Rotation.from_euler('z', -roll_deg, degrees=True).as_matrix() @ np.diag([1., -1., -1.])
    desired_corners = np.asarray(object_points) @ desired_rotation.T + [0, 0, distance]
    desired = desired_corners[:, :2] / desired_corners[:, 2:]
    measured = detection['T_camera_marker']
    depths = (np.asarray(object_points) @ measured[:3, :3].T + measured[:3, 3])[:, 2]
    interaction = image_jacobian(points, depths)
    error = (points - desired).reshape(8)
    pixel_delta = (points - desired) @ np.asarray(intrinsics)[:2, :2].T
    pixel_rms = float(np.sqrt(np.mean(np.sum(pixel_delta ** 2, axis=1))))
    return error, interaction, pixel_rms


def ibvs_joint_velocity(error, interaction, camera_joints, gain, damping,
                        max_joint_speed, max_linear_speed, max_angular_speed, feedforward=None):
    """Joint velocity from the feature error plus an optional camera-frame twist feedforward."""
    combined = interaction @ camera_joints
    identity = damping ** 2 * np.eye(combined.shape[1])
    velocity = -gain * np.linalg.solve(combined.T @ combined + identity, combined.T @ error)
    if feedforward is not None and np.any(feedforward):
        velocity = velocity + np.linalg.solve(camera_joints.T @ camera_joints + identity,
                                              camera_joints.T @ feedforward)
    if not np.isfinite(velocity).all():
        raise ValueError('nonfinite IBVS joint velocity')
    camera_velocity = camera_joints @ velocity
    scale = max(1., float(np.max(np.abs(velocity))) / max_joint_speed,
                float(np.linalg.norm(camera_velocity[:3])) / max_linear_speed,
                float(np.linalg.norm(camera_velocity[3:])) / max_angular_speed)
    return velocity / scale


class MarkerVelocity:
    """Base-frame marker velocity from a least-squares line over a short window."""

    def __init__(self, window, deadband, max_speed, reset_gap):
        self.window, self.deadband = window, deadband
        self.max_speed, self.reset_gap = max_speed, reset_gap
        self.samples = deque()

    def reset(self):
        self.samples.clear()

    def update(self, stamp, position):
        if self.samples and stamp - self.samples[-1][0] > self.reset_gap:
            self.samples.clear()
        self.samples.append((stamp, np.asarray(position, float)))
        while stamp - self.samples[0][0] > self.window:
            self.samples.popleft()
        times = np.array([item[0] for item in self.samples])
        if len(times) < 4 or times[-1] - times[0] < self.window / 2:
            return np.zeros(3)
        positions = np.array([item[1] for item in self.samples])
        times = times - times.mean()
        velocity = times @ (positions - positions.mean(axis=0)) / (times @ times)
        speed = float(np.linalg.norm(velocity))
        if speed <= self.deadband:
            return np.zeros(3)
        # Soft deadband: no step when the estimate crosses it.
        return velocity * min(speed - self.deadband, self.max_speed) / speed


class ServoStream(threading.Thread):
    """Sole owner of robot I/O during servo mode: state polling and servo_j at a fixed rate."""

    def __init__(self, robot, q0, rate_hz, speed, accel, jerk, lower, upper,
                 collision=None, collision_clearance=0.):
        from robot_toolkit.core.reference import ContinuousReference

        super().__init__(daemon=True, name='piper-servo')
        self.robot = robot
        self.dt = 1. / rate_hz
        # The toolkit fallback limit-cycles around a static goal (~1 s period); require Ruckig.
        self.reference = ContinuousReference(q0, dt=self.dt, speed=speed, acceleration=accel,
                                             jerk=jerk, lower=lower, upper=upper,
                                             backend='ruckig')
        self.lock = threading.Lock()
        self.goal = np.asarray(q0, float).copy()
        self.history = deque(maxlen=int(2 * rate_hz))
        self.latest = None
        self.reference_q = np.asarray(q0, float).copy()
        self.reference_v = np.zeros_like(self.reference_q)
        self.stop_event = threading.Event()
        self.error = None
        self.collision = collision
        self.send_gaps = deque(maxlen=int(rate_hz * 10))   # s between servo_j calls
        self._last_send = None
        self.brake = None
        if collision is not None:
            from robot_toolkit.control.guarded import CollisionBrake

            self.brake = CollisionBrake(collision, accel, jerk, self.dt, collision_clearance)

    @property
    def collision_brakes(self):
        return 0 if self.brake is None else self.brake.brakes

    def reference_position(self):
        with self.lock:
            return self.reference_q.copy()

    def set_goal(self, goal):
        with self.lock:
            self.goal = np.asarray(goal, float).copy()

    def get_goal(self):
        with self.lock:
            return self.goal.copy()

    def still(self, joint_tolerance=.02, speed_tolerance=.01):
        """Reference stopped at the goal and the measured arm caught up with it."""
        with self.lock:
            if self.latest is None:
                return False
            lag = np.max(np.abs(self.latest.q - self.reference_q))
            settled = np.max(np.abs(self.reference_q - self.goal)) < joint_tolerance
            return bool(settled and lag < joint_tolerance
                        and np.max(np.abs(self.reference_v)) < speed_tolerance)

    def state_at(self, stamp):
        """Measured state closest to a host ``perf_counter`` time."""
        with self.lock:
            samples = list(self.history)
        if not samples:
            return None
        return min(samples, key=lambda item: abs(item[0] - stamp))[1]

    def run(self):
        deadline = time.perf_counter()
        try:
            while not self.stop_event.is_set():
                state = self.robot.state()
                if state.q is not None and state.tcp is not None:
                    with self.lock:
                        self.history.append((time.perf_counter(), state))
                        self.latest = state
                q_ref = self.reference.step(self.get_goal())
                if self.brake is not None:
                    # Hard floor raises (the main loop's finally stops the arm); a motion
                    # whose stop would enter the clearance band is redirected to the
                    # farthest clear stopping point until the vision loop's next gated goal.
                    stop = self.brake.brake_goal(q_ref, self.reference.v, self.reference.a,
                                                 self.get_goal())
                    if stop is not None:
                        self.set_goal(stop)
                with self.lock:
                    self.reference_q = q_ref.copy()
                    self.reference_v = self.reference.v.copy()
                self.robot.servo_j(q_ref, self.dt)
                sent = time.perf_counter()
                if self._last_send is not None:
                    self.send_gaps.append(sent - self._last_send)
                self._last_send = sent
                deadline += self.dt
                remaining = deadline - time.perf_counter()
                if remaining > 0:
                    time.sleep(remaining)
                else:
                    deadline = time.perf_counter()
        except Exception as exc:  # surfaced by the vision loop
            self.error = exc


class IbvsStream(ServoStream):
    def __init__(self, robot, state, args, lower, upper):
        from ruckig import ControlInterface, InputParameter, OutputParameter, Ruckig

        speed = args.reference_speed
        super().__init__(robot, state.q, args.control_hz, speed, args.reference_accel,
                         args.reference_jerk, lower, upper, getattr(args, 'collision', None),
                         getattr(args, 'collision_clearance', 0.))
        self.otg = Ruckig(len(state.q), self.dt)
        self.inputs = InputParameter(len(state.q))
        self.outputs = OutputParameter(len(state.q))
        self.inputs.control_interface = ControlInterface.Velocity
        self.inputs.current_position = list(state.q)
        self.inputs.current_velocity = [0.] * len(state.q)
        self.inputs.current_acceleration = [0.] * len(state.q)
        self.inputs.max_velocity = [speed] * len(state.q)
        self.inputs.max_acceleration = [args.reference_accel] * len(state.q)
        self.inputs.max_jerk = [args.reference_jerk] * len(state.q)
        self.inputs.target_acceleration = [0.] * len(state.q)
        self.velocity = np.zeros_like(state.q)
        self.command_time = time.perf_counter()
        self.command_timeout = args.ibvs_timeout
        self.lower = np.asarray(lower, float)
        self.upper = np.asarray(upper, float)
        self.speed = speed
        self.block_low = np.zeros(len(state.q), bool)
        self.block_high = np.zeros(len(state.q), bool)
        self.reference_stopped = False
        self.reference_a = np.zeros_like(state.q)
        self.commands_sent = 0
        self.last_command_time = None
        self.command_timed_out = False

    def telemetry(self):
        with self.lock:
            measured = None if self.latest is None else self.latest.q
            return {
                'reference_stopped': self.reference_stopped,
                'reference_q': self.reference_q.tolist(),
                'reference_v': self.reference_v.tolist(),
                'reference_a': self.reference_a.tolist(),
                'measured_q': None if measured is None else measured.tolist(),
                'joint_error_max_rad': (None if measured is None else
                                        float(np.max(np.abs(self.reference_q - measured)))),
                'commands_sent': self.commands_sent,
                'last_command_t': self.last_command_time,
                'command_timed_out': self.command_timed_out,
            }

    def set_velocity(self, velocity):
        velocity = np.asarray(velocity, float)
        if velocity.shape != self.velocity.shape or not np.isfinite(velocity).all():
            raise ValueError('invalid IBVS velocity command')
        with self.lock:
            self.velocity = np.clip(velocity, -self.speed, self.speed)
            self.command_time = time.perf_counter()

    def braking_margin(self, direction):
        """Per-joint distance the reference needs to stop when moving toward ``direction``.

        Uses the current reference velocity/acceleration, not the speed cap, so a
        slow joint can approach its limit; the hard limit check still aborts.
        """
        accel_limit = np.asarray(self.inputs.max_acceleration)
        jerk_limit = np.asarray(self.inputs.max_jerk)
        speed = np.maximum(direction * np.asarray(self.inputs.current_velocity), 0.)
        accel = np.maximum(direction * np.asarray(self.inputs.current_acceleration), 0.)
        # Speed gained while the acceleration toward the limit ramps down to zero.
        speed = speed + accel ** 2 / (2 * jerk_limit)
        return (speed ** 2 / (2 * accel_limit) + speed * (accel_limit + accel) / jerk_limit
                + speed * self.dt + .02)

    def still(self, joint_tolerance=.02, speed_tolerance=.01):
        with self.lock:
            return bool(self.latest is not None
                        and np.max(np.abs(self.latest.q - self.reference_q)) < joint_tolerance
                        and np.max(np.abs(self.reference_v)) < speed_tolerance)

    def run(self):
        from ruckig import Result

        deadline = time.perf_counter()
        feedback_time = deadline
        feedback_stamp = None
        try:
            while not self.stop_event.is_set():
                now = time.perf_counter()
                state = self.robot.state()
                if (not state.connected or state.q is None or not np.isfinite(state.q).all()
                        or state.protective_stop or getattr(state, 'fault', False)):
                    raise RuntimeError('IBVS robot feedback unavailable or faulted')
                if state.t != feedback_stamp:
                    feedback_stamp, feedback_time = state.t, now
                if now - feedback_time > self.command_timeout:
                    raise RuntimeError('IBVS robot feedback stale')
                with self.lock:
                    self.latest = state
                    self.history.append((now, state))
                    velocity = self.velocity.copy()
                    timed_out = now - self.command_time > self.command_timeout
                    if timed_out:
                        velocity[:] = 0.
                current = np.asarray(self.inputs.current_position)
                if np.max(np.abs(current - state.q)) > .10:
                    raise RuntimeError('IBVS reference exceeds measured joint tracking limit')
                # Hysteresis: a joint stopped at its braking band stays stopped
                # instead of creeping on and off at the band edge.
                room_low = np.minimum(current, state.q) - self.lower
                room_high = self.upper - np.maximum(current, state.q)
                margin_low, margin_high = self.braking_margin(-1), self.braking_margin(1)
                self.block_low = (room_low < margin_low) | (
                    self.block_low & (room_low < margin_low + .01))
                self.block_high = (room_high < margin_high) | (
                    self.block_high & (room_high < margin_high + .01))
                velocity[(self.block_low & (velocity < 0))
                         | (self.block_high & (velocity > 0))] = 0.
                if self.brake is not None and np.any(velocity):
                    velocity *= self.brake.speed_scale(current, velocity,
                                                       self.inputs.current_acceleration)
                self.inputs.target_velocity = velocity.tolist()
                result = self.otg.update(self.inputs, self.outputs)
                if result not in (Result.Working, Result.Finished):
                    raise RuntimeError(f'IBVS Ruckig error: {result}')
                command = np.asarray(self.outputs.new_position)
                if np.any(command < self.lower) or np.any(command > self.upper):
                    raise RuntimeError('IBVS joint limit reached')
                if self.collision is not None:
                    self.collision.check_position(command)
                stopped = bool(not np.any(velocity) and result == Result.Finished
                           and np.max(np.abs(self.outputs.new_velocity)) < 1e-9
                           and np.max(np.abs(self.outputs.new_acceleration)) < 1e-9)
                send_command = not stopped or not self.reference_stopped
                if send_command:
                    if not self.robot.servo_j(command, self.dt):
                        raise RuntimeError('IBVS servo command rejected')
                sent_at = time.perf_counter() if send_command else None
                self.outputs.pass_to_input(self.inputs)
                with self.lock:
                    self.reference_q = command.copy()
                    self.reference_v = np.asarray(self.outputs.new_velocity).copy()
                    self.reference_a = np.asarray(self.outputs.new_acceleration).copy()
                    self.reference_stopped = stopped
                    self.command_timed_out = timed_out
                    if send_command:
                        self.commands_sent += 1
                        self.last_command_time = sent_at
                deadline += self.dt
                if deadline < time.perf_counter():
                    deadline = time.perf_counter()
                self.stop_event.wait(max(0., deadline - time.perf_counter()))
        except Exception as exc:
            self.error = exc
        finally:
            if not self.robot.stop_motion() and self.error is None:
                self.error = RuntimeError('IBVS stop/hold command failed; check physical emergency stop')


def ibvs_command(args, robot, state, detection, cv2, object_points, frame, tcp_camera,
                 marker_velocity=None, stamp=None):
    """Returns (joint velocity, corner rms px, base-frame marker velocity fed forward)."""
    if detection['reprojection_error_px'] > 2.:
        raise ValueError('IBVS marker reprojection error exceeds 2 pixels')
    error, interaction, pixel_rms = ibvs_features(
        cv2, detection, object_points, frame.intrinsics, frame.distortion,
        args.distance, args.roll_deg)
    base_tcp = pose_matrix(robot.forward_kinematics(state.q))
    jacobian = camera_jacobian(numerical_jacobian(robot, state.q), base_tcp, tcp_camera)
    feedforward, marker_speed = None, np.zeros(3)
    if marker_velocity is not None and args.ibvs_feedforward > 0:
        base_camera = base_tcp @ tcp_camera
        marker = base_camera[:3, :3] @ detection['T_camera_marker'][:3, 3] + base_camera[:3, 3]
        marker_speed = args.ibvs_feedforward * marker_velocity.update(
            frame.t if stamp is None else stamp, marker)
        feedforward = np.concatenate((base_camera[:3, :3].T @ marker_speed, np.zeros(3)))
    velocity = ibvs_joint_velocity(error, interaction, jacobian, args.ibvs_gain,
                                   args.servo_damping, args.reference_speed,
                                   args.ibvs_linear_speed, args.ibvs_angular_speed, feedforward)
    return velocity, pixel_rms, marker_speed


def frame_capture_time(frame, args):
    """Mid-exposure time of ``frame`` on the perf_counter clock used by the streams."""
    if args.camera_clock == 'hardware':
        capture = (getattr(frame, 'meta', None) or {}).get('color', {}).get('capture_t')
        if capture is not None:
            return capture - (args.exposure_ms or 0.) / 2000.
    return frame.t - args.camera_latency


def run_ibvs(args, robot, camera, config, cv2, detector, object_points, tcp_camera, state):
    stream = IbvsStream(robot, state, args, config.limits_lower, config.limits_upper)
    marker_velocity = MarkerVelocity(args.ibvs_ff_window, args.ibvs_ff_deadband,
                                     args.ibvs_ff_max, args.ibvs_timeout)
    last_frame = None
    locked = False
    was_valid = True
    processed = 0
    start = time.perf_counter()
    log = None
    try:
        if args.log is not None:
            args.log.parent.mkdir(parents=True, exist_ok=True)
            log = args.log.open('w')
        stream.start()
        pause_gc()   # no collector pauses while streaming (collected again in finally)
        while ((not args.steps or processed < args.steps)
               and (not args.ibvs_duration or time.perf_counter() - start < args.ibvs_duration)):
            if stream.error is not None:
                raise RuntimeError('IBVS control thread failed') from stream.error
            frame = camera.snapshot()
            age = time.perf_counter() - frame.t
            if age < 0 or age > args.ibvs_timeout or frame.rgb is None:
                stream.set_velocity(np.zeros_like(state.q))
                stream.stop_event.wait(.003)
                continue
            if frame.t == last_frame:
                stream.stop_event.wait(.003)
                continue
            last_frame = frame.t
            stamp = frame_capture_time(frame, args)
            detection = detect_marker(cv2, detector, object_points, frame.rgb,
                                      frame.intrinsics, frame.distortion, args.marker_id)
            measured = stream.state_at(stamp)
            valid = detection is not None and measured is not None
            pixel_rms = None
            marker_speed = np.zeros(3)
            velocity = np.zeros_like(state.q)
            if valid:
                try:
                    velocity, pixel_rms, marker_speed = ibvs_command(
                        args, robot, measured, detection, cv2, object_points, frame, tcp_camera,
                        marker_velocity, stamp)
                except ValueError:
                    valid = False
                    velocity[:] = 0.
            if valid:
                locked = (pixel_rms < args.ibvs_resume_px if locked
                          else pixel_rms <= args.ibvs_stop_px)
                # A moving marker keeps the loop running even inside the pixel lock.
                if locked and not np.any(marker_speed):
                    velocity[:] = 0.
            else:
                locked = False
                marker_velocity.reset()
            stream.set_velocity(velocity)  # no marker: zero velocity, hold and wait
            processed += valid
            record = {'control': 'ibvs', 'frame': processed, 'frame_t': frame.t,
                      'capture_t': stamp, 'latency_s': time.perf_counter() - stamp,
                      'valid': valid, 'locked': locked, 'corner_rms_px': pixel_rms,
                      'qdot': velocity.tolist(), 'marker_ff_mps': marker_speed.tolist(),
                      'distance_m': args.distance,
                      'stream': stream.telemetry()}
            if log is not None:
                log.write(json.dumps(record) + '\n')
                log.flush()
            if valid and processed % 6 == 1:
                print(json.dumps(record), flush=True)
            elif not valid and was_valid:
                print('marker lost: holding, waiting for it to reappear', flush=True)
            was_valid = valid
            if locked and not args.hold and stream.still():
                break
        if stream.error is not None:
            raise RuntimeError('IBVS control thread failed') from stream.error
    finally:
        stream.stop_event.set()
        if stream.ident is not None:
            stream.join()
        collect_idle(resume=True)
        if log is not None:
            log.close()
    if stream.error is not None:
        raise RuntimeError('IBVS control thread failed') from stream.error


def servo_target(T_base_marker, start_tcp, tcp_camera, distance, roll_deg, position_only):
    """Absolute TCP target from a filtered marker pose."""
    if position_only:
        # Fixed start orientation; put the marker on the optical axis at the distance.
        rotation = start_tcp[:3, :3]
        camera_rotation = rotation @ tcp_camera[:3, :3]
        camera_position = T_base_marker[:3, 3] - distance * camera_rotation[:, 2]
        target = np.eye(4)
        target[:3, :3] = rotation
        target[:3, 3] = camera_position - rotation @ tcp_camera[:3, 3]
        return target
    desired_camera, _ = desired_camera_pose(T_base_marker, None, distance, roll_deg)
    return desired_camera @ np.linalg.inv(tcp_camera)


def filter_pose(previous, measured, alpha):
    if previous is None:
        return measured.copy()
    result = np.eye(4)
    result[:3, 3] = previous[:3, 3] + alpha * (measured[:3, 3] - previous[:3, 3])
    delta = Rotation.from_matrix(previous[:3, :3].T @ measured[:3, :3]).as_rotvec()
    result[:3, :3] = previous[:3, :3] @ Rotation.from_rotvec(alpha * delta).as_matrix()
    return result


def run_servo(args, robot, camera, config, cv2, detector, object_points, tcp_camera,
              state, target_distance):
    lower = np.asarray(config.limits_lower, float)
    upper = np.asarray(config.limits_upper, float)
    start_tcp = pose_matrix(robot.forward_kinematics(state.q))
    stream = ServoStream(robot, state.q, args.control_hz, args.reference_speed,
                         args.reference_accel, args.reference_jerk, lower, upper,
                         args.collision, args.collision_clearance)
    stream.start()
    pause_gc()   # no collector pauses while streaming (collected again in finally)
    blocked = 0
    block_warned = False
    marker = None
    lead_velocity = MarkerVelocity(.15, args.lead_deadband, np.inf, args.reacquire_gap)
    last_seen = None
    candidates = []
    rejected = 0
    reach_warned = False
    orientation_weight = args.orientation_weight if args.position_only else 1.
    goal_target = None
    still_since = None
    last_frame_t = None
    processed = 0
    reached_once = False
    locked = False
    last_print = 0.
    log = None
    if args.log is not None:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        log = args.log.open('w')
    try:
        while not args.steps or processed < args.steps:
            if stream.error is not None:
                raise RuntimeError(f'servo stream stopped: {stream.error}') from stream.error
            frame = camera.snapshot()
            if frame.rgb is None or frame.t == last_frame_t:
                time.sleep(.003)
                continue
            last_frame_t = frame.t
            stamp = frame_capture_time(frame, args)
            detection = detect_marker(cv2, detector, object_points, frame.rgb,
                                      frame.intrinsics, frame.distortion, args.marker_id)
            measured_state = stream.state_at(stamp)
            if (detection is None or measured_state is None
                    or detection['reprojection_error_px'] > 2.):
                continue  # no (trustworthy) marker: the stream holds the last goal; wait
            T_base_camera = pose_matrix(robot.forward_kinematics(measured_state.q)) @ tcp_camera
            measured_marker = T_base_camera @ detection['T_camera_marker']
            if last_seen is not None and stamp - last_seen > args.reacquire_gap:
                marker = None
                lead_velocity.reset()
            if marker is not None:
                # A pose jump faster than a hand can move is a wrong detection/PnP flip.
                jump = np.linalg.norm(measured_marker[:3, 3] - marker[:3, 3])
                if jump > .03 + args.max_marker_speed * (stamp - last_seen):
                    rejected += 1
                    continue
            else:
                # (Re)acquire only after several consecutive, mutually consistent detections.
                if candidates and (stamp - candidates[-1][0] > args.reacquire_gap or np.linalg.norm(
                        measured_marker[:3, 3] - candidates[-1][1]) > .02):
                    candidates = []
                candidates.append((stamp, measured_marker[:3, 3].copy()))
                if len(candidates) < args.reacquire_frames:
                    continue
                candidates = []
            last_seen = stamp
            processed += 1
            now = time.monotonic()
            if stream.still():
                still_since = still_since or now
            else:
                still_since = None
            settled = still_since is not None and now - still_since >= args.settle_time
            use_frame = settled or args.track_while_moving or marker is None
            if use_frame:
                marker = filter_pose(marker, measured_marker, args.marker_alpha)
            predicted = marker.copy()
            if args.lead > 0:
                # Least-squares velocity with a deadband: zero once the hand stops instead of
                # decaying through the stop, and a capped offset, so the goal barely overshoots.
                offset = lead_velocity.update(stamp, measured_marker[:3, 3]) * args.lead
                size = np.linalg.norm(offset)
                if size > args.lead_max_m:
                    offset *= args.lead_max_m / size
                predicted[:3, 3] += offset
            target = servo_target(predicted, start_tcp, tcp_camera, target_distance,
                                  args.roll_deg, args.position_only)
            reach = target[:3, 3] - start_tcp[:3, 3]
            if np.linalg.norm(reach) > args.max_reach_m:
                target[:3, 3] = start_tcp[:3, 3] + reach * args.max_reach_m / np.linalg.norm(reach)
                if not reach_warned:
                    print(f'target clamped to --max-reach-m {args.max_reach_m} around the start TCP',
                          flush=True)
                reach_warned = True
            updated = False
            ik_position = 0.
            if goal_target is None:
                change = (np.inf, np.inf)
            else:
                change = pose_error(goal_target, target)
            position, _ = pose_error(pose_matrix(robot.forward_kinematics(stream.latest.q)), target)
            # Refine a residual only once the arm has stopped: the measured TCP lags the command
            # by ~80 ms, so judging it mid-motion re-issues goals and makes the arm hunt.
            if use_frame and not locked and (change[0] > args.target_deadband_m
                                             or change[1] > args.target_deadband_rad
                                             or (stream.still() and position > args.lock_target_m)):
                gain = 1. if goal_target is None else args.target_gain
                blended = filter_pose(goal_target, target, gain)
                goal = stream.get_goal()
                candidate, (ik_position, ik_rotation) = solve_ik(
                    robot, goal, blended, args.ik_iterations, args.servo_damping,
                    lower, upper, orientation_weight)
                rotation_limit = args.max_orientation_drift if args.position_only else .05
                solved = ik_position <= .005 and ik_rotation <= rotation_limit
                # Partial solution: still move closer instead of freezing on an unreachable target.
                closer = ik_position < pose_error(
                    pose_matrix(robot.forward_kinematics(goal)), blended)[0] - .002
                if (solved or closer) and args.collision is not None:
                    # Go only as far along the joint path as the obstacles allow.
                    start = stream.reference_position()
                    fraction = args.collision.safe_fraction(start, candidate,
                                                            args.collision_clearance)
                    if fraction < 1.:
                        candidate = start + (candidate - start) * fraction
                        solved = False  # re-evaluate next frame: the hand may move away
                        blocked += 1
                        if not block_warned:
                            _, link, name = args.collision.clearance(candidate)
                            print(f'goal blocked by obstacle {name!r} ({link}); holding at the '
                                  'edge', flush=True)
                        block_warned = True
                    else:
                        block_warned = False
                if solved or closer:
                    if np.max(np.abs(candidate - goal)) > args.goal_deadband:
                        stream.set_goal(candidate)
                        updated = True
                        still_since = None
                    if solved:
                        goal_target = blended
            position, rotation = pose_error(
                pose_matrix(robot.forward_kinematics(stream.latest.q)), target)
            if not locked and position <= args.lock_target_m \
                    and detection['pixel_error_px'] <= args.lock_pixel:
                locked = True
            elif locked and (position >= args.unlock_target_m
                             or detection['pixel_error_px'] >= args.unlock_pixel):
                locked = False
                still_since = None
            reached = settled and not updated and change[0] <= args.target_deadband_m \
                and change[1] <= args.target_deadband_rad
            if reached and not reached_once:
                reached_once = True
                print(json.dumps({'frame': processed, 'reached': True,
                                  'position_error_m': round(position, 5),
                                  'rotation_error_rad': round(rotation, 5)}), flush=True)
                if not args.hold:
                    break
            if log is not None:
                log.write(json.dumps({
                    't': now, 'frame_t': frame.t, 'capture_t': stamp,
                    'latency_s': time.perf_counter() - stamp, 'frame': processed,
                    'settled': settled, 'used': use_frame, 'updated': updated,
                    'q': measured_state.q.tolist(), 'q_latest': stream.latest.q.tolist(),
                    'goal': stream.get_goal().tolist(),
                    'marker': marker[:3, 3].tolist(), 'target': target[:3, 3].tolist(),
                    'position_error_m': position, 'rotation_error_rad': rotation,
                    'pixel_error_px': detection['pixel_error_px'],
                    'reprojection_px': detection['reprojection_error_px'],
                    'rejected': rejected, 'blocked': blocked,
                    'servo_gap_max_ms': (round(max(stream.send_gaps) * 1000, 2)
                                         if stream.send_gaps else None),
                    'collision_brakes': stream.collision_brakes,
                    'clearance_m': (None if args.collision is None else
                                    args.collision.clearance(stream.reference_position())[0])})
                          + '\n')
            if now - last_print > .2:
                last_print = now
                print(json.dumps({
                    'frame': processed, 'position_error_m': round(position, 4),
                    'rotation_error_rad': round(rotation, 4),
                    'pixel_error_px': round(detection['pixel_error_px'], 1),
                    'locked': locked,
                    'settled': settled, 'goal_updated': updated,
                    'ik_residual_m': round(ik_position, 5),
                    'marker_in_base': marker[:3, 3].round(4).tolist()}), flush=True)
        else:
            print(f'processed {processed} frames', flush=True)
    finally:
        stream.stop_event.set()
        stream.join(timeout=1.)
        collect_idle(resume=True)
        if log is not None:
            log.close()


def load_calibration(path, camera_name):
    from robot_toolkit.calibrate.store import CalibrationStore

    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f'calibration not found: {path}')
    store = CalibrationStore(path)
    entry = store.cameras.get(camera_name)
    if entry is None:
        raise ValueError(f'camera {camera_name!r} not in {path}')
    if entry.tcp_camera is None:
        raise ValueError(f'camera {camera_name!r} has no tcp_camera (eye-in-hand) extrinsic')
    if entry.intrinsics is None:
        raise ValueError(f'camera {camera_name!r} has no intrinsics')
    return entry


class MarkerNotFound(RuntimeError):
    pass


def read_scene(robot, camera, cv2, detector, object_points, tcp_camera, attempts=1, marker_id=None):
    # Right after connect the flange-pose message can still be the all-zero default.
    for _ in range(100):
        state = robot.state()
        if state.q is not None and state.tcp is not None:
            T_base_tcp = pose_matrix(robot.forward_kinematics(state.q))
            if pose_error(T_base_tcp, pose_matrix(state.tcp))[0] < .005:
                break
        time.sleep(.02)
    else:
        raise RuntimeError('robot TCP feedback does not match forward kinematics')
    frame = None
    detection = None
    for _ in range(max(1, int(attempts))):
        frame = camera.snapshot()
        detection = detect_marker(cv2, detector, object_points, frame.rgb,
                                  frame.intrinsics, frame.distortion, marker_id)
        if detection is not None:
            break
        time.sleep(.05)
    if detection is None:
        if frame is None or frame.rgb is None:
            raise RuntimeError('wrist camera has not produced a frame')
        raise MarkerNotFound('ArUco marker not detected in the wrist image')
    T_base_camera = T_base_tcp @ tcp_camera
    T_base_marker = T_base_camera @ detection['T_camera_marker']
    return state, frame, detection, T_base_tcp, T_base_camera, T_base_marker


def report(args, state, detection, T_base_camera, T_base_marker, target_tcp, target_distance):
    position, rotation = pose_error(
        np.block([[Rotation.from_rotvec(state.tcp[3:]).as_matrix(),
                   np.asarray(state.tcp[:3])[:, None]], [np.zeros((1, 3)), 1.]]),
        target_tcp)
    record = {
        'step': None,
        'mode': 'execute' if args.execute else 'dry-run' if args.dry_run else 'read-only',
        'marker_pixel_error_px': round(detection['pixel_error_px'], 3),
        'marker_reprojection_error_px': round(detection['reprojection_error_px'], 3),
        'camera_marker_distance_m': round(target_distance, 4),
        'current_tcp': np.asarray(state.tcp).round(5).tolist(),
        'target_tcp': np.asarray(pose_vector(target_tcp)).round(5).tolist(),
        'target_position_error_m': round(position, 5),
        'target_rotation_error_rad': round(rotation, 5),
        'marker_in_base': T_base_marker[:3, 3].round(5).tolist(),
    }
    print(json.dumps(record, ensure_ascii=False), flush=True)
    return record, position, rotation


def check_collision_model(args, robot):
    """Refuse to run if the URDF used for collision disagrees with the SDK FK, or the
    start pose is already inside an obstacle margin."""
    state = robot.state()
    if state.q is None:
        raise RuntimeError('collision guard: no joint feedback')
    urdf = args.collision.robot.chain.link_poses(state.q)['link6'][:3, 3]
    sdk = pose_matrix(robot.forward_kinematics(state.q))[:3, 3]
    if np.linalg.norm(urdf - sdk) > .01:
        raise RuntimeError(f'collision URDF link6 {urdf.round(4).tolist()} differs from the SDK '
                           f'FK {sdk.round(4).tolist()} by > 10 mm; the guard would be wrong')
    distance, link, name = args.collision.clearance(state.q)
    print(json.dumps({'clearance_m': round(distance, 4), 'closest_link': link,
                      'obstacle': name}), flush=True)
    if args.execute and distance < args.collision.margin_m:
        raise RuntimeError(f'start pose: {link} is {distance*1000:.0f} mm from {name!r}, inside '
                           'margin_m; move the arm clear (or fix the obstacle file) first')


def watch_clearance(args, robot):
    print('move the arm by hand (drag/teach mode); Ctrl-C to quit', flush=True)
    margin = args.collision.margin_m
    while True:
        state = robot.state()
        if state.q is not None:
            distance, link, name = args.collision.clearance(state.q)
            flag = ('INSIDE MARGIN' if distance < margin else 'near' if
                    distance < margin + args.collision_clearance else 'ok')
            tcp = pose_matrix(robot.forward_kinematics(state.q))[:3, 3]
            print(f'{flag:13s} {distance*1000:7.1f} mm  {link:12s} -> {name:16s} '
                  f'tcp {np.round(tcp, 3).tolist()}', flush=True)
        time.sleep(.2)


def main(argv=None):
    args = parse_args(argv)
    if args.list_cameras:
        from robot_toolkit.camera.realsense import list_devices
        for name, serial in list_devices():
            print(f'{serial}\t{name}')
        return 0

    import cv2
    from robot_toolkit.camera.realsense import RealSenseCamera
    from robot_toolkit.robot import AgxPiperConfig, AgxPiperRobot

    entry = load_calibration(args.calibration, args.camera_name)
    tcp_camera = np.asarray(entry.tcp_camera, float)
    if args.obstacles is not None:
        from robot_toolkit.core.collision import CollisionGuard, d435_spheres
        # TCP == link6 (tcp_offset is zero here), so the camera rides on link6.
        args.collision = CollisionGuard.from_config(
            args.obstacles, attached={'link6': d435_spheres(tcp_camera)})
        print(f'collision guard: {len(args.collision.obstacles)} obstacles, '
              f'{len(args.collision.robot.radii)} robot spheres, margin '
              f'{args.collision.margin_m:.3f} m + clearance {args.collision_clearance:.3f} m',
              flush=True)
    cv2, detector, object_points = make_detector(args)

    config = AgxPiperConfig(
        model=args.model, interface=args.interface, channel=args.channel,
        speed_percent=args.speed_percent,
        enable_on_connect=bool(args.execute),
        configure_on_connect=bool(args.execute),
        has_gripper=False, ik_backend='none', ik_warmup=False,
        command_hz=0.)
    robot = AgxPiperRobot(config)
    camera = RealSenseCamera(args.camera_name, serial=args.serial, width=args.width,
                             height=args.height, fps=args.fps, depth=False,
                             exposure_ms=args.exposure_ms, gain=args.camera_gain)
    motion_started = False
    try:
        if args.execute and args.control != 'ibvs':
            motion_started = True
            robot.connect()
            print('connected: motion enabled at', args.speed_percent, '% speed', flush=True)
        else:
            robot.connect_read_only()
            print('connected: read-only; no enable, no motion', flush=True)
        if args.collision is not None:
            check_collision_model(args, robot)
        if args.watch_clearance:
            watch_clearance(args, robot)
            return 0
        camera.open()
        camera.set_intrinsics(entry.intrinsics, entry.distortion)
        print('camera:', camera.name, 'serial:', args.serial or '<first>', flush=True)
        waiting = False
        while True:
            try:
                state, frame, detection, T_base_tcp, T_base_camera, T_base_marker = read_scene(
                    robot, camera, cv2, detector, object_points, tcp_camera, attempts=40,
                    marker_id=args.marker_id if args.control == 'ibvs' else None)
                break
            except MarkerNotFound:
                if not args.execute:
                    raise
                if not waiting:
                    print('marker not visible: waiting for it (Ctrl-C to quit)', flush=True)
                waiting = True
        if args.control == 'ibvs':
            try:
                velocity, pixel_rms, _ = ibvs_command(
                    args, robot, state, detection, cv2, object_points, frame, tcp_camera)
            except ValueError as exc:
                if not args.execute:
                    raise
                print(f'IBVS preflight skipped this frame ({exc}); the servo loop waits for '
                      'a usable one', flush=True)
            else:
                print(json.dumps({'control': 'ibvs', 'corner_rms_px': pixel_rms,
                                  'distance_m': args.distance, 'qdot': velocity.tolist(),
                                  'marker_id': args.marker_id}), flush=True)
            if not args.execute:
                return 0
            measured = detection['T_camera_marker']
            offset = np.linalg.norm(measured[:3, 3] - [0, 0, args.distance])
            desired_rotation = Rotation.from_euler('z', -args.roll_deg, degrees=True).as_matrix() @ np.diag([1., -1., -1.])
            angular_offset = Rotation.from_matrix(desired_rotation @ measured[:3, :3].T).magnitude()
            if offset > args.max_target_offset_m or angular_offset > args.max_target_offset_rad:
                print(f'warning: marker starts {offset:.3f} m / {angular_offset:.3f} rad from the '
                      'goal view; following it anyway (speed/accel/jerk/joint limits apply)',
                      flush=True)
            IbvsStream(robot, state, args, config.limits_lower, config.limits_upper)
            motion_started = True
            if not robot.prepare_motion():
                raise RuntimeError('IBVS motion enable failed')
            robot.set_speed_percent(args.speed_percent)
            run_ibvs(args, robot, camera, config, cv2, detector, object_points, tcp_camera, state)
            return 0
        desired_camera, distance = desired_camera_pose(
            T_base_marker, T_base_camera, args.distance, args.roll_deg)
        target_distance = args.distance if args.distance > 0 else distance
        target_tcp = servo_target(T_base_marker, T_base_tcp, tcp_camera, target_distance,
                                  args.roll_deg, args.position_only)
        _, position_error, rotation_error = report(
            args, state, detection, T_base_camera, T_base_marker, target_tcp, target_distance)

        if not args.execute:
            if args.dry_run:
                if args.motion == 'servo':
                    goal, residual = solve_ik(
                        robot, state.q, target_tcp, args.ik_iterations, args.servo_damping,
                        config.limits_lower, config.limits_upper)
                    print(json.dumps({'servo_goal_joints': goal.round(5).tolist(),
                                      'joint_delta': (goal - state.q).round(5).tolist(),
                                      'ik_residual': [round(residual[0], 5),
                                                      round(residual[1], 5)]}),
                          flush=True)
                else:
                    command = clamp_step(T_base_tcp, target_tcp, args.step_m, args.step_rad)
                    print(json.dumps({'first_step_tcp': pose_vector(command).round(5).tolist()}),
                          flush=True)
            print('dry-run only; no robot motion was sent', flush=True)
            return 0

        if (position_error > args.max_target_offset_m
                or rotation_error > args.max_target_offset_rad):
            print(f'warning: target is {position_error:.3f} m / {rotation_error:.3f} rad from the '
                  'current TCP; following it anyway (speed/accel/jerk/joint limits apply)',
                  flush=True)

        if args.motion == 'servo':
            run_servo(args, robot, camera, config, cv2, detector, object_points, tcp_camera,
                      state, target_distance)
            return 0

        step = 0
        waiting = False
        while not args.steps or step < args.steps:
            try:
                state, frame, detection, T_base_tcp, T_base_camera, T_base_marker = read_scene(
                    robot, camera, cv2, detector, object_points, tcp_camera, attempts=2)
            except MarkerNotFound:
                if not waiting:
                    print('marker lost: holding, waiting for it to reappear', flush=True)
                waiting = True
                time.sleep(args.period)
                continue
            waiting = False
            step += 1
            target_tcp = servo_target(T_base_marker, T_base_tcp, tcp_camera, target_distance,
                                      args.roll_deg, args.position_only)
            current_tcp = T_base_tcp
            position, rotation = pose_error(current_tcp, target_tcp)
            print(json.dumps({'step': step, 'position_error_m': round(position, 5),
                              'rotation_error_rad': round(rotation, 5),
                              'pixel_error_px': round(detection['pixel_error_px'], 2)}),
                  flush=True)
            if (position <= args.target_tolerance_m
                    and rotation <= args.target_tolerance_rad):
                print('target reached; holding position', flush=True)
                break
            command = clamp_step(current_tcp, target_tcp, args.step_m, args.step_rad)
            started = time.monotonic()
            robot.move_p(pose_vector(command), wait=True, timeout=args.timeout)
            elapsed = time.monotonic() - started
            if elapsed < args.period:
                time.sleep(args.period - elapsed)
        else:
            print(f'stopped after {args.steps} steps without reaching tolerance', flush=True)
        return 0
    finally:
        try:
            if motion_started and robot.connected:
                robot.stop_motion()
        except Exception:
            pass
        try:
            camera.close()
        finally:
            robot.disconnect()
        print('closed; the arm is left enabled/holding if it was enabled', flush=True)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print('stopped by Ctrl-C', flush=True)
        raise SystemExit(130)
