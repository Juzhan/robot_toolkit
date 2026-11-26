import json
from pathlib import Path

import numpy as np
import pytest

from robot_toolkit.core.collision import (Box, CollisionGuard, Plane, PointCloud, Sphere,
                                          SphereRobot, UrdfChain, box_spheres, d435_spheres)
from robot_toolkit.safety.limits import JointLimits, SafetyGuard
from robot_toolkit.errors import SafetyViolation
from robot_toolkit.robot.mock import MockRobot
from robot_toolkit.robot.safety import SafeRobot

ASSETS = Path(__file__).resolve().parents[1] / 'assets' / 'piperx_assets'
URDF = ASSETS / 'assets' / 'piper_x.urdf'
HOME = np.array([0., .5, -.6, 0., .3, 0.])     # wrist forward, above the table


def piper_guard(obstacles=(), **kwargs):
    return CollisionGuard.from_config({'obstacles': list(obstacles), **kwargs})


def test_chain_matches_yourdfpy():
    yourdfpy = pytest.importorskip('yourdfpy')
    chain = UrdfChain(URDF)
    model = yourdfpy.URDF.load(str(URDF))
    rng = np.random.default_rng(1)
    for _ in range(5):
        q = rng.uniform(-1, 1, 6)
        model.update_cfg({f'joint{i+1}': v for i, v in enumerate(q)})
        poses = chain.link_poses(q)
        for link in chain.links:
            np.testing.assert_allclose(poses[link], model.get_transform(link, 'base_link'),
                                       atol=1e-9)


def test_spheres_cover_every_link_mesh():
    trimesh = pytest.importorskip('trimesh')
    data = json.loads((ASSETS / 'piper_x_spheres.json').read_text())
    for link, items in data['spheres'].items():
        if link == 'gripper_base':
            continue  # includes the folded fingers; checked by construction
        mesh = trimesh.load(URDF.parent / 'meshes' / f'{link}.stl')
        points = np.vstack([mesh.vertices, trimesh.sample.sample_surface(mesh, 2000, seed=3)[0]])
        centers = np.array([s['center'] for s in items])
        radii = np.array([s['radius'] for s in items])
        gap = (np.linalg.norm(points[:, None] - centers[None], axis=-1) - radii).min(axis=1)
        assert gap.max() < .004, link   # padding_m (1 cm by default) absorbs this


def test_primitive_distances():
    points = np.array([[0., 0., 1.], [0., 0., -1.], [2., 0., 0.]])
    np.testing.assert_allclose(Plane([0, 0, 0], [0, 0, 2]).distance(points), [1, -1, 0])
    np.testing.assert_allclose(Box([0, 0, 0], [2, 2, 2]).distance(points), [0, 0, 1])
    np.testing.assert_allclose(Box([0, 0, 0], [2, 2, 2]).distance(np.zeros((1, 3))), [-1])
    np.testing.assert_allclose(Sphere([0, 0, 0], .5).distance(points), [.5, .5, 1.5])
    np.testing.assert_allclose(PointCloud([[0, 0, 0]], .1).distance(points), [.9, .9, 1.9])
    assert PointCloud(np.zeros((0, 3))).distance(points).min() == np.inf


def test_box_spheres_cover_the_box():
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.]])
    spheres = box_spheres([.1, 0, 0], [.14, .035, .03], .02, rotation)
    corners = np.array([[x, y, z] for x in (-.07, .07) for y in (-.0175, .0175)
                        for z in (-.015, .015)]) @ rotation.T + [.1, 0, 0]
    centers = np.array([c for c, _ in spheres])
    radii = np.array([r for _, r in spheres])
    assert np.all((np.linalg.norm(corners[:, None] - centers[None], axis=-1) - radii).min(1)
                  <= 1e-9)
    assert radii.max() <= .02 + 1e-9
    assert len(d435_spheres(np.eye(4))) == len(spheres)


def test_table_plane_and_virtual_wall():
    guard = piper_guard([{'type': 'plane', 'name': 'table', 'point': [0, 0, 0],
                          'normal': [0, 0, 1]}])
    distance, link, name = guard.clearance(HOME)
    assert name == 'table' and distance > .05 and link != 'base_link'
    guard.check_path(HOME, HOME + [.5, 0, 0, 0, 0, 0])
    down = HOME + [0, 1.2, 0, 0, 0, 0]          # shoulder forward: gripper into the table
    with pytest.raises(SafetyViolation, match='table'):
        guard.check_position(down)
    with pytest.raises(SafetyViolation):
        guard.check_path(HOME, down)            # endpoint check alone would catch it too
    fraction = guard.safe_fraction(HOME, down)
    assert 0 < fraction < 1
    edge = HOME + (down - HOME) * fraction
    guard.check_path(HOME, edge)
    # Inside the extra clearance band: may back away, not go deeper.
    assert guard.safe_fraction(edge, HOME, clearance_m=.05) == 1.
    assert guard.safe_fraction(edge, down, clearance_m=.05) < .05

    # The gripper reaches x = 0.30 m at HOME; a wall face at 0.35 m.
    wall = piper_guard([{'type': 'box', 'name': 'wall', 'center': [.36, 0, .3],
                         'size': [.02, .6, .6]}])
    wall.check_position(HOME)
    with pytest.raises(SafetyViolation, match='wall'):
        wall.check_path(HOME, HOME + [0, .5, -.2, 0, 0, 0])   # reach forward into the wall


def test_ignore_links_and_dynamic_points():
    # The base sits on the table; ignored by default, enforced if asked.
    guard = piper_guard([{'type': 'plane', 'point': [0, 0, 0], 'normal': [0, 0, 1]}])
    guard.check_position(HOME)
    strict = piper_guard([{'type': 'plane', 'point': [0, 0, 0], 'normal': [0, 0, 1]}],
                         ignore_links=[])
    with pytest.raises(SafetyViolation, match='base_link'):
        strict.check_position(HOME)
    tcp = guard.robot.chain.link_poses(HOME)['link6'][:3, 3]
    guard.set_dynamic([PointCloud([tcp + [0, 0, .03]], .01)])
    with pytest.raises(SafetyViolation, match='point_cloud'):
        guard.check_position(HOME)
    guard.set_dynamic([])
    guard.check_position(HOME)


def test_attached_box_extends_the_robot():
    plain = piper_guard()
    tooled = piper_guard(attached=[{'link': 'link6', 'center': [0, 0, .25],
                                    'size': [.04, .04, .2]}])
    assert len(tooled.robot.radii) > len(plain.robot.radii)
    camera = CollisionGuard.from_config({}, attached={'link6': d435_spheres(np.eye(4))})
    assert len(camera.robot.radii) == len(plain.robot.radii) + len(d435_spheres(np.eye(4)))


def test_safe_robot_rejects_joint_targets_into_obstacles():
    inner = MockRobot()
    inner.connect()
    dof = inner.dof
    q0 = np.zeros(dof)
    q0[:6] = HOME
    inner.move_j(q0)
    chain_guard = piper_guard([{'type': 'plane', 'name': 'table', 'point': [0, 0, 0],
                                'normal': [0, 0, 1]}])

    class Mapped:  # the mock may have != 6 joints; collide on the first six
        def __getattr__(self, name):
            return getattr(chain_guard, name)

        def check_path(self, start, target, clearance_m=0.):
            chain_guard.check_path(np.asarray(start)[:6], np.asarray(target)[:6], clearance_m)

        def check_feedback(self, state):
            chain_guard.check_position(np.asarray(state.q)[:6])

    robot = SafeRobot(inner, SafetyGuard(JointLimits([-3.] * dof, [3.] * dof),
                                         max_step_rad=3.), collision=Mapped())
    assert robot.describe()['safety']['collision_guard'] is True
    target = q0.copy()
    target[1] += 1.2
    with pytest.raises(SafetyViolation, match='table'):
        robot.move_j(target)
    with pytest.raises(SafetyViolation):
        robot.servo_l(np.zeros(6), .01)


def test_invalid_config():
    with pytest.raises(Exception):
        piper_guard([{'type': 'cone'}])
    with pytest.raises(ValueError):
        CollisionGuard(SphereRobot(UrdfChain(URDF), {'link1': [([0, 0, 0], .1)]}), margin_m=-1)
    with pytest.raises(SafetyViolation):
        piper_guard().check_position([0.] * 5)
