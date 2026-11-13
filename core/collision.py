"""Sphere-model clearance against virtual obstacles and point clouds (NumPy only).

The robot is a set of padded spheres per URDF link; obstacles are planes
(half-spaces), boxes, spheres or a point cloud in the robot base frame.
``CollisionGuard`` mirrors ``WorkspaceGuard`` (check_position / check_path /
check_feedback) so it plugs into ``SafeRobot`` and the servo streams. It only
ever adds checks: a violation raises ``SafetyViolation``.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from .mathutils import euler_to_matrix
from ..errors import ConfigurationError, SafetyViolation


def _transform(xyz=(0., 0., 0.), rpy=(0., 0., 0.)):
    matrix = np.eye(4)
    matrix[:3, :3] = euler_to_matrix(rpy, order='xyz')
    matrix[:3, 3] = xyz
    return matrix


def _skew(axis):
    x, y, z = axis
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


class UrdfChain:
    """Forward kinematics of every link reachable through revolute/fixed joints."""

    def __init__(self, urdf_path, base_link='base_link', joint_names=None):
        root = ET.parse(str(urdf_path)).getroot()
        children = {}
        for joint in root.findall('joint'):
            kind = joint.get('type')
            if kind not in ('revolute', 'continuous', 'fixed'):
                continue  # prismatic fingers: fold them into the parent's spheres
            origin = joint.find('origin')
            xyz = [float(v) for v in (origin.get('xyz', '0 0 0') if origin is not None
                                      else '0 0 0').split()]
            rpy = [float(v) for v in (origin.get('rpy', '0 0 0') if origin is not None
                                      else '0 0 0').split()]
            axis = joint.find('axis')
            axis = np.array([float(v) for v in (axis.get('xyz') if axis is not None
                                                else '1 0 0').split()])
            parent = joint.find('parent').get('link')
            children.setdefault(parent, []).append({
                'name': joint.get('name'), 'fixed': kind == 'fixed', 'parent': parent,
                'origin': _transform(xyz, rpy), 'axis': axis / np.linalg.norm(axis),
                'child': joint.find('child').get('link')})
        self.base_link = base_link
        self.joints = []                     # topological order from the base
        stack = [base_link]
        while stack:
            for joint in children.get(stack.pop(), []):
                self.joints.append(joint)
                stack.append(joint['child'])
        movable = [j['name'] for j in self.joints if not j['fixed']]
        self.joint_names = list(joint_names or movable)
        missing = set(self.joint_names) - set(movable)
        if missing:
            raise ConfigurationError(f'URDF lacks joints {sorted(missing)}')
        self.links = [base_link] + [j['child'] for j in self.joints]
        index = {link: i for i, link in enumerate(self.links)}
        for joint in self.joints:
            joint['parent_index'] = index[joint['parent']]
            joint['child_index'] = index[joint['child']]
            joint['q_index'] = (self.joint_names.index(joint['name'])
                                if joint['name'] in self.joint_names else None)
            joint['K'] = _skew(joint['axis'])
            joint['K2'] = joint['K'] @ joint['K']

    def link_poses_batch(self, qs):
        """(N, links, 4, 4) poses for joint vectors ``qs`` (N, dof), links in ``self.links``."""
        qs = np.asarray(qs, float)
        if qs.ndim != 2 or qs.shape[1] != len(self.joint_names) or not np.isfinite(qs).all():
            raise SafetyViolation('invalid joint vector for collision FK')
        poses = np.empty((len(qs), len(self.links), 4, 4))
        poses[:, 0] = np.eye(4)
        for joint in self.joints:
            local = np.broadcast_to(joint['origin'], (len(qs), 4, 4)).copy()
            if not joint['fixed'] and joint['q_index'] is not None:
                angle = qs[:, joint['q_index'], None, None]
                rotation = np.eye(3) + np.sin(angle) * joint['K'] + (1 - np.cos(angle)) * joint['K2']
                local[:, :3, :3] = joint['origin'][:3, :3] @ rotation
            poses[:, joint['child_index']] = poses[:, joint['parent_index']] @ local
        return poses

    def link_poses(self, q):
        poses = self.link_poses_batch(np.asarray(q, float)[None])[0]
        return dict(zip(self.links, poses))


class SphereRobot:
    """Padded collision spheres attached to URDF links."""

    def __init__(self, chain: UrdfChain, spheres: dict, padding_m=0.):
        self.chain = chain
        self.padding_m = float(padding_m)
        names, centers, radii = [], [], []
        for link, items in spheres.items():
            if link not in chain.links:
                raise ConfigurationError(f'collision spheres reference unknown link {link!r}')
            for center, radius in items:
                names.append(link)
                centers.append(np.asarray(center, float).reshape(3))
                radii.append(float(radius) + padding_m)
        if not names:
            raise ConfigurationError('no collision spheres')
        self.links = np.array(names)
        self.link_index = np.array([chain.links.index(name) for name in names])
        self.local = np.array(centers)
        self.radii = np.array(radii)
        if np.any(self.radii <= 0):
            raise ConfigurationError('collision sphere radii must be positive')

    @classmethod
    def load(cls, urdf_path, spheres_path, padding_m=0., attached=None):
        data = json.loads(Path(spheres_path).read_text())
        spheres = {link: [(s['center'], s['radius']) for s in items]
                   for link, items in data['spheres'].items()}
        for link, items in (attached or {}).items():
            spheres.setdefault(link, []).extend(items)
        chain = UrdfChain(urdf_path, data.get('base_link', 'base_link'), data.get('joints'))
        return cls(chain, spheres, padding_m)

    def spheres_batch(self, qs):
        """World centres (N, spheres, 3) for joint vectors ``qs`` (N, dof)."""
        poses = self.chain.link_poses_batch(qs)[:, self.link_index]
        return np.einsum('nsij,sj->nsi', poses[..., :3, :3], self.local) + poses[..., :3, 3]

    def spheres(self, q):
        """World centres (spheres, 3) at joint vector ``q``."""
        return self.spheres_batch(np.asarray(q, float)[None])[0]


def box_spheres(center, size, radius, rotation=None):
    """Spheres exactly covering a box (e.g. a camera body) in a link frame.

    The box is cut into cells no larger than a cube inscribed in ``radius``; each
    cell gets the sphere through its corners.
    """
    center, size = np.asarray(center, float), np.asarray(size, float)
    rotation = np.eye(3) if rotation is None else np.asarray(rotation, float)
    counts = np.maximum(1, np.ceil(size / (2 * radius / np.sqrt(3))).astype(int))
    cell = size / counts
    axes = [(np.arange(n) + .5) * c - s / 2 for n, c, s in zip(counts, cell, size)]
    grid = np.stack(np.meshgrid(*axes, indexing='ij'), -1).reshape(-1, 3)
    cover = float(np.linalg.norm(cell)) / 2
    return [(center + rotation @ p, cover) for p in grid]


# ------------------------------------------------------------------ obstacles
class Plane:
    """Half-space obstacle: everything on the side opposite ``normal`` is solid."""

    def __init__(self, point, normal, name='plane', ignore_links=()):
        normal = np.asarray(normal, float)
        if normal.shape != (3,) or not np.linalg.norm(normal) > 0:
            raise ConfigurationError('plane normal must be a non-zero 3-vector')
        self.point = np.asarray(point, float).reshape(3)
        self.normal = normal / np.linalg.norm(normal)
        self.name, self.ignore_links = name, set(ignore_links)

    def distance(self, points):
        return (points - self.point) @ self.normal


class Box:
    def __init__(self, center, size, rpy=(0., 0., 0.), name='box', ignore_links=()):
        self.center = np.asarray(center, float).reshape(3)
        self.half = np.asarray(size, float).reshape(3) / 2
        if np.any(self.half <= 0):
            raise ConfigurationError('box size must be positive')
        self.rotation = euler_to_matrix(rpy, order='xyz')
        self.name, self.ignore_links = name, set(ignore_links)

    def distance(self, points):
        local = np.abs((points - self.center) @ self.rotation) - self.half
        outside = np.linalg.norm(np.maximum(local, 0.), axis=-1)
        return outside + np.minimum(np.max(local, axis=-1), 0.)


class Sphere:
    def __init__(self, center, radius, name='sphere', ignore_links=()):
        self.center = np.asarray(center, float).reshape(3)
        self.radius = float(radius)
        if not self.radius > 0:
            raise ConfigurationError('sphere radius must be positive')
        self.name, self.ignore_links = name, set(ignore_links)

    def distance(self, points):
        return np.linalg.norm(points - self.center, axis=-1) - self.radius


class PointCloud:
    """Occupied points (e.g. voxel centres from depth) with a per-point radius."""

    def __init__(self, points, radius=0.01, name='point_cloud', ignore_links=()):
        self.points = np.asarray(points, float).reshape(-1, 3)
        self.radius = float(radius)
        self.name, self.ignore_links = name, set(ignore_links)

    def distance(self, points):
        shape = np.shape(points)[:-1]
        points = np.reshape(points, (-1, 3))
        best = np.full(len(points), np.inf)
        for start in range(0, len(self.points), 1024):  # bounded memory
            chunk = self.points[start:start+1024]
            gap = np.linalg.norm(points[:, None, :] - chunk[None], axis=-1).min(axis=1)
            best = np.minimum(best, gap)
        return (best - self.radius).reshape(shape)


class DistanceGrid:
    """Sampled distance field (e.g. an ESDF built from depth) on an axis-aligned grid.

    ``distance[i, j, k]`` is the distance in metres from the centre of cell
    ``origin + (i, j, k) * voxel`` to the nearest obstacle (<= 0 inside). A query
    returns the nearest cell's value minus half a cell diagonal, a lower bound
    because a distance field is 1-Lipschitz. Points outside the grid get
    ``outside`` (inf: the map only covers the workspace it observes).
    """

    def __init__(self, origin, voxel, distance, outside=np.inf, name='depth',
                 ignore_links=(), stamp=None):
        self.origin = np.asarray(origin, float).reshape(3)
        self.voxel = float(voxel)
        self.values = np.asarray(distance, float)
        if self.values.ndim != 3 or not self.voxel > 0:
            raise ConfigurationError('distance grid needs a 3-D array and a positive voxel')
        self.outside = float(outside)
        self.name, self.ignore_links, self.stamp = name, set(ignore_links), stamp

    def distance(self, points):
        points = np.asarray(points, float)
        index = np.rint((points - self.origin) / self.voxel).astype(int)
        inside = np.all((index >= 0) & (index < self.values.shape), axis=-1)
        result = np.full(points.shape[:-1], self.outside)
        i = index[inside]
        result[inside] = self.values[i[:, 0], i[:, 1], i[:, 2]] - self.voxel * np.sqrt(3) / 2
        return result


def obstacle_from_dict(item):
    kind = item.get('type')
    common = {'name': item.get('name', kind), 'ignore_links': item.get('ignore_links', ())}
    if kind == 'plane':
        return Plane(item['point'], item['normal'], **common)
    if kind == 'box':
        return Box(item['center'], item['size'], item.get('rpy', (0., 0., 0.)), **common)
    if kind == 'sphere':
        return Sphere(item['center'], item['radius'], **common)
    raise ConfigurationError(f'unknown obstacle type {kind!r}')


# ---------------------------------------------------------------------- guard
class CollisionGuard:
    """Rejects configurations/paths whose spheres come within ``margin_m`` of an obstacle."""

    def __init__(self, robot: SphereRobot, obstacles=(), margin_m=0.02, step_rad=0.01,
                 ignore_links=('base_link',)):
        if not np.isfinite(margin_m) or margin_m < 0 or not step_rad > 0:
            raise ValueError('invalid collision margin or sampling step')
        self.robot = robot
        self.obstacles = list(obstacles)
        self.margin_m = float(margin_m)
        self.step_rad = float(step_rad)
        self.ignore_links = set(ignore_links)
        self.dynamic = []                   # replaced wholesale, e.g. by a depth thread
        self._masks = {}

    @classmethod
    def from_config(cls, config, base_dir=None, attached=None):
        """``config``: dict or JSON path.

        Keys: ``obstacles`` (list of plane/box/sphere dicts in the base frame),
        ``attached`` (boxes on links), ``padding_m`` (added to every robot sphere,
        default 0.01), ``margin_m`` (required clearance, default 0.02),
        ``ignore_links``, ``step_rad``, ``urdf``, ``spheres`` (default Piper X).
        ``attached`` (argument) adds ``{link: [(center, radius), ...]}``.
        """
        if not isinstance(config, dict):
            path = Path(config)
            config, base_dir = json.loads(path.read_text()), path.parent
        base = Path(base_dir or '.')
        package = Path(__file__).resolve().parents[1]

        def resolve(value, default):
            value = Path(value) if value else package / default
            return value if value.is_absolute() or value.exists() else base / value

        attached = {link: list(items) for link, items in (attached or {}).items()}
        for box in config.get('attached', []):   # e.g. a tool or bracket, in the link frame
            attached.setdefault(box['link'], []).extend(box_spheres(
                box['center'], box['size'], float(box.get('sphere_radius', 0.02)),
                euler_to_matrix(box.get('rpy', (0., 0., 0.)), order='xyz')))
        robot = SphereRobot.load(
            resolve(config.get('urdf'), 'assets/piperx_assets/assets/piper_x.urdf'),
            resolve(config.get('spheres'), 'assets/piperx_assets/piper_x_spheres.json'),
            float(config.get('padding_m', 0.01)), attached)
        return cls(robot, [obstacle_from_dict(o) for o in config.get('obstacles', [])],
                   float(config.get('margin_m', 0.02)), float(config.get('step_rad', 0.01)),
                   config.get('ignore_links', ['base_link']))

    def set_dynamic(self, obstacles):
        self.dynamic = list(obstacles)

    def _mask(self, obstacle):
        key = id(obstacle)
        if key not in self._masks or self._masks[key][0] is not obstacle:
            ignored = list(self.ignore_links | obstacle.ignore_links)
            self._masks[key] = (obstacle, np.flatnonzero(~np.isin(self.robot.links, ignored)))
        return self._masks[key][1]

    def _gaps(self, qs):
        centers = self.robot.spheres_batch(qs)
        for obstacle in self.obstacles + self.dynamic:
            index = self._mask(obstacle)
            if len(index):
                yield obstacle, index, (obstacle.distance(centers[:, index])
                                        - self.robot.radii[index])

    def clearances(self, qs):
        """Closest sphere/obstacle distance for each joint vector in ``qs`` (N, dof)."""
        best = np.full(len(qs), np.inf)
        for start in range(0, len(qs), 256):          # bounded memory
            chunk = np.asarray(qs[start:start+256], float)
            for _, _, gaps in self._gaps(chunk):
                best[start:start+len(chunk)] = np.minimum(best[start:start+len(chunk)],
                                                          gaps.min(axis=1))
        return best

    def clearance(self, q):
        """(distance_m, link, obstacle) of the closest sphere/obstacle pair."""
        best = (np.inf, None, None)
        for obstacle, index, gaps in self._gaps(np.asarray(q, float)[None]):
            closest = int(np.argmin(gaps[0]))
            if gaps[0, closest] < best[0]:
                best = (float(gaps[0, closest]), str(self.robot.links[index[closest]]),
                        obstacle.name)
        return best

    def check_position(self, q, clearance_m=0.):
        distance, link, obstacle = self.clearance(q)
        if distance < self.margin_m + clearance_m:
            raise SafetyViolation(f'{link} is {distance*1000:.0f} mm from {obstacle} '
                                  f'(needs {(self.margin_m + clearance_m)*1000:.0f} mm)')

    def _samples(self, start, target):
        start, target = np.asarray(start, float), np.asarray(target, float)
        if (start.ndim != 1 or target.shape != start.shape or
                not np.isfinite([start, target]).all()):
            raise SafetyViolation('invalid joint path')
        steps = max(1, int(np.ceil(np.max(np.abs(target - start)) / self.step_rad)))
        if steps > 10000:
            raise SafetyViolation('joint path exceeds sampling budget')
        return start, target, steps

    def _path(self, start, target):
        start, target, steps = self._samples(start, target)
        return start + (target - start) * (np.arange(steps + 1) / steps)[:, None], steps

    def check_path(self, start, target, clearance_m=0.):
        path, _ = self._path(start, target)
        bad = np.flatnonzero(self.clearances(path) < self.margin_m + clearance_m)
        if len(bad):
            self.check_position(path[bad[0]], clearance_m)   # raises with the details

    def safe_fraction(self, start, target, clearance_m=0.):
        """Largest sampled fraction of the straight joint path that stays clear.

        A start already inside ``margin_m + clearance_m`` may still move as long as
        it never gets closer than it already is (backing away is always allowed),
        but never below ``margin_m``.
        """
        path, steps = self._path(start, target)
        distance = self.clearances(path)
        limit = min(self.margin_m + clearance_m, max(distance[0] - 1e-4, self.margin_m))
        bad = np.flatnonzero(distance[1:] < limit)
        return 1. if not len(bad) else bad[0] / steps

    def check_feedback(self, state):
        if state.q is None:
            raise SafetyViolation('collision guard needs joint feedback')
        self.check_position(state.q)


def d435_spheres(link_camera, sphere_radius=0.02):
    """Spheres for a RealSense D435 body given the link -> colour-optical-frame pose.

    The body is 90 x 25 x 25 mm; the colour imager sits off-centre, so the box is
    widened to 140 mm along x to cover either mounting direction, and sits behind
    the lens (optical z points out of the camera).
    """
    link_camera = np.asarray(link_camera, float)
    center = link_camera[:3, :3] @ [0., 0., -.0125] + link_camera[:3, 3]
    return box_spheres(center, (.14, .035, .03), sphere_radius, link_camera[:3, :3])
