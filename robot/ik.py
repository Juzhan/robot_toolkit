"""Optional host-side IK adapter.

The toolkit itself stays NumPy-only.  This module lazily imports robokit/Warp
only when the solver is used, so importing ``robot_toolkit`` does not pull it in.

On 1000 random reachable Piper X targets (5 mm / 0.05 rad) ``RobokitIK``
solved 99.6% from a fixed home seed at about 1.4 ms per call.  Install it in
the environment that runs the robot (GPU recommended)::

    git clone https://github.com/openrobokit/robokit.git   # tested at 736f112
    pip install -e 'robokit[pt-cu128]'

Example::

    from robot_toolkit.robot import RobokitIK

    ik = RobokitIK(
        'robot_toolkit/assets/piperx_assets/assets/piper_x.urdf',
        target_link_name='link6',
        joint_names=['joint1', 'joint2', 'joint3',
                     'joint4', 'joint5', 'joint6'],
    )
    robot.set_inverse_kinematics(ik)
    q = robot.inverse_kinematics(pose, seed=robot.state().q)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..core.mathutils import matrix_to_quat, rotvec_to_matrix


def _require_robokit():
    try:
        import warp as wp
        from robokit.helpers.ik import IKSolver, presets
        from robokit.robo import Robot
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            'robokit IK is optional. Install robokit (Warp, CUDA recommended) '
            'in the environment that runs the robot.') from exc
    return wp, IKSolver, presets, Robot


class RobokitIK:
    """Solve ``[x, y, z, rx, ry, rz]`` IK with robokit's batched Warp solver.

    ``solve(pose, seed)`` returns the
    joints in ``joint_names`` order, or ``[]`` when the solution misses the
    tolerances.  The default ``basic`` preset optimizes a population of
    candidates around the seed and keeps the best, so it does not need a good
    seed; ``single_candidate`` refines from the seed only.
    """

    def __init__(self, urdf_path, target_link_name: str = 'link6',
                 joint_names=None, preset: str = 'basic',
                 device: str | None = None,
                 position_tolerance: float = 0.005,
                 rotation_tolerance: float = 0.05):
        self.urdf_path = str(Path(urdf_path).expanduser())
        self.target_link_name = target_link_name
        self.joint_names = None if joint_names is None else tuple(joint_names)
        self.preset = str(preset)
        self.device = device
        self.position_tolerance = float(position_tolerance)
        self.rotation_tolerance = float(rotation_tolerance)
        self._wp = None
        self._robot = None
        self._solver = None
        self._actuated_names = None
        self._indices = None

    def _load(self):
        if self._solver is not None:
            return
        wp, IKSolver, presets, Robot = _require_robokit()
        wp.config.quiet = True
        robot = Robot.load(self.urdf_path)
        if self.target_link_name not in robot.link_names:
            raise ValueError(f'target link {self.target_link_name!r} not found; '
                             f'links are {tuple(robot.link_names)}')
        names = tuple(robot.spec.actuated_joint_names)
        if self.joint_names is not None:
            missing = [name for name in self.joint_names if name not in names]
            if missing:
                raise ValueError(f'joint_names not actuated in URDF: {missing}; '
                                 f'actuated names are {names}')
        device = self.device or ('cuda' if wp.is_cuda_available() else 'cpu')
        self._solver = IKSolver(getattr(presets, self.preset), robot=robot,
                                target_links=self.target_link_name, device=device)
        self.device = device
        self._wp = wp
        self._robot = robot
        self._actuated_names = names
        self._indices = (list(range(len(names))) if self.joint_names is None
                         else [names.index(name) for name in self.joint_names])

    @property
    def actuated_names(self):
        self._load()
        return self._actuated_names

    def warmup(self, pose, seed=None) -> bool:
        """Build the Warp kernels/CUDA graph once; the first solve is slow."""
        try:
            self.solve(pose, seed=seed)
            return True
        except Exception:
            return False

    def solve(self, pose, seed=None):
        """Return joint angles for a target-link pose, or ``[]`` if IK fails."""
        self._load()
        wp = self._wp
        pose = np.asarray(pose, dtype=float).reshape(6)
        wxyz = matrix_to_quat(rotvec_to_matrix(pose[3:]), order='wxyz')
        target = np.concatenate([pose[:3], wxyz]).astype(np.float32)[None]
        full = np.zeros((1, len(self._actuated_names)), np.float32)
        if seed is not None:
            seed = np.asarray(seed, dtype=float).reshape(-1)
            if seed.size == len(self._indices):
                full[0, self._indices] = seed
            elif seed.size == full.shape[1]:
                full[0] = seed
            else:
                raise ValueError(f'seed has {seed.size} joints, expected '
                                 f'{len(self._indices)} or {full.shape[1]}')
        init = self._robot.state(q=wp.array(full, dtype=wp.float32, device=self.device))
        state = self._solver.solve_numpy(target, init_state=init)
        metrics = self._solver.compute_metrics_numpy(
            state, target, self.position_tolerance, self.rotation_tolerance)
        if not bool(metrics.success[0]):
            return []
        return state.q.numpy()[0, self._indices].astype(float)

    def __repr__(self) -> str:
        return (f'RobokitIK(urdf={self.urdf_path!r}, '
                f'target={self.target_link_name!r}, preset={self.preset!r})')
