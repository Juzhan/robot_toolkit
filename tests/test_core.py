"""Core math, conventions, safety and config tests (NumPy only)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest

from robot_toolkit.config import ConfigFile, build_dataclass, load_json, save_json
from robot_toolkit.core.conventions import RobotSpec
from robot_toolkit.core.mathutils import (
    average_transforms, compose_transforms, euler_to_matrix, invert_transform,
    matrix_to_pose, matrix_to_quat, orthonormalize, pose_error,
    pose_to_matrix, quat_to_matrix, rotvec_to_matrix, transform_points)
from robot_toolkit.core.rate import Rate
from robot_toolkit.core.safety import (JointLimits, SafetyGuard, Watchdog,
                                       WorkspaceBox)
from robot_toolkit.errors import SafetyViolation


@dataclass
class InnerConfig:
    value: int = 0


@dataclass
class OuterConfig:
    name: str = 'default'
    inner: InnerConfig = field(default_factory=InnerConfig)
    items: list = field(default_factory=list)


def test_pose_matrix_roundtrip_including_pi():
    rng = np.random.default_rng(0)
    poses = [np.zeros(6), np.array([0.2, -0.3, 0.4, np.pi, 0.0, 0.0])]
    poses += [np.r_[rng.normal(0, 0.4, 3), rng.normal(0, 2.0, 3)] for _ in range(20)]
    for pose in poses:
        recovered = matrix_to_pose(pose_to_matrix(pose))
        assert np.allclose(recovered[:3], pose[:3], atol=1e-9)
        assert np.allclose(rotvec_to_matrix(recovered[3:]), rotvec_to_matrix(pose[3:]), atol=1e-8)


def test_quaternion_roundtrip_both_orders():
    rotation = rotvec_to_matrix([0.3, -0.6, 1.1])
    for order in ('xyzw', 'wxyz'):
        quaternion = matrix_to_quat(rotation, order=order)
        assert np.isclose(np.linalg.norm(quaternion), 1.0)
        assert np.allclose(quat_to_matrix(quaternion, order=order), rotation, atol=1e-10)


def test_transform_composition_and_inverse():
    rng = np.random.default_rng(1)
    a = pose_to_matrix(np.r_[rng.normal(0, 0.2, 3), rng.normal(0, 1.0, 3)])
    b = pose_to_matrix(np.r_[rng.normal(0, 0.2, 3), rng.normal(0, 1.0, 3)])
    assert np.allclose(compose_transforms(a, b), a @ b)
    assert np.allclose(compose_transforms(a, invert_transform(a)), np.eye(4), atol=1e-10)
    point = rng.normal(0, 0.1, 3)
    transformed = transform_points(a, point)
    assert np.allclose(transform_points(invert_transform(a), transformed), point, atol=1e-10)


def test_pose_error():
    pose = np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.4])
    translation, rotation = pose_error(pose, pose.copy())
    assert translation == pytest.approx(0.0, abs=1e-12)
    assert rotation == pytest.approx(0.0, abs=1e-12)
    translation, rotation = pose_error(np.zeros(6), pose)
    assert translation == pytest.approx(np.linalg.norm(pose[:3]))
    assert rotation == pytest.approx(0.4)


def test_average_transforms_is_near_midpoint():
    base = pose_to_matrix([0.1, 0.0, 0.2, 0.0, 0.0, 0.0])
    left = base @ pose_to_matrix([0.0, 0.0, 0.0, 0.0, 0.0, 0.2])
    right = base @ pose_to_matrix([0.0, 0.0, 0.0, 0.0, 0.0, -0.2])
    average = average_transforms([left, right])
    translation, rotation = pose_error(matrix_to_pose(average),
                                       matrix_to_pose(base))
    assert translation < 1e-9
    assert rotation < 1e-9


def test_euler_to_matrix_matches_expected_order():
    # Extrinsic xyz: R = Rz @ Ry @ Rx
    angles = [0.2, -0.3, 0.4]
    expected = (rotvec_to_matrix([0, 0, angles[2]]) @
                rotvec_to_matrix([0, angles[1], 0]) @
                rotvec_to_matrix([angles[0], 0, 0]))
    assert np.allclose(euler_to_matrix(angles, 'xyz'), expected)


def test_orthonormalize_repairs_numerical_drift():
    rotation = rotvec_to_matrix([0.1, 0.2, 0.3])
    damaged = rotation + np.full((3, 3), 1e-3)
    repaired = orthonormalize(damaged)
    assert np.allclose(repaired.T @ repaired, np.eye(3), atol=1e-12)
    assert np.linalg.det(repaired) == pytest.approx(1.0)


def test_robot_spec_contract():
    spec = RobotSpec.ur5e()
    assert spec.dim == 7
    assert spec.state_names[0] == 'shoulder_pan.rad'
    assert spec.state_names[-1] == 'robotiq_2f85.width_m'
    assert spec.pack([0] * 6, 0.5)[-1] == pytest.approx(0.085)
    restored = RobotSpec.from_dict(spec.to_dict())
    assert restored == spec
    assert RobotSpec.generic(7).dof == 7
    assert RobotSpec.generic(7, with_gripper=False).dim == 7


def test_joint_limits_clamp_and_rate_limit():
    limits = JointLimits([-1.0, -1.0], [1.0, 1.0], max_velocity=0.5)
    assert np.allclose(limits.clamp([2.0, -2.0]), [1.0, -1.0])
    limited = limits.rate_limit([0.0, 0.0], [1.0, -1.0], dt=0.1)
    assert np.allclose(limited, [0.05, -0.05])
    assert limits.violates([0.0, 1.5])


def test_safety_guard_flags_clipping_and_rate_limit():
    guard = SafetyGuard(JointLimits([-1, -1], [1, 1], max_velocity=10.0), max_step_rad=0.2)
    decision = guard.evaluate([0.0, 0.0], [5.0, 0.0], dt=1.0)
    assert decision.clipped and decision.rate_limited
    assert np.allclose(decision.target, [0.2, 0.0])
    with pytest.raises(SafetyViolation):
        SafetyGuard(JointLimits([-1], [1], 10.0), reject_on_limit=True).check([0.0], [2.0], 1.0)


def test_workspace_box():
    box = WorkspaceBox([0, 0, 0], [1, 1, 1])
    assert np.allclose(box.clamp([2, -1, 0.5]), [1, 0, 0.5])
    assert box.contains([0.5, 0.5, 0.5])
    assert not box.contains([1.1, 0.5, 0.5])


def test_watchdog_with_fake_clock():
    now = [0.0]
    watchdog = Watchdog(1.0, clock=lambda: now[0])
    now[0] = 0.5
    assert not watchdog.expired()
    now[0] = 1.5
    assert watchdog.expired()
    watchdog.feed()
    assert not watchdog.expired()


def test_rate_counts_overruns():
    now = [0.0]
    rate = Rate(100.0, clock=lambda: now[0])
    rate._started = 0.0
    rate._next = 0.0
    now[0] = 0.05  # simulate a 50 ms iteration against a 10 ms period
    rate.wait()
    assert rate.overruns == 1
    assert rate.max_overrun == pytest.approx(0.04)


def test_config_dataclass_roundtrip(tmp_path):
    raw = {'name': 'ur5e', 'inner': {'value': 3}, 'items': [1, 2], 'unknown': True}
    path = tmp_path / 'config.json'
    save_json(path, raw)
    assert load_json(path) == raw
    config = build_dataclass(OuterConfig, raw)
    assert config.name == 'ur5e'
    assert config.inner.value == 3
    assert config.items == [1, 2]
    # Unknown keys are ignored so old files keep loading.
    store = ConfigFile(path, OuterConfig)
    assert store.exists()
    assert store.load().inner.value == 3

