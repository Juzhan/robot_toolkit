"""Optional host-side IK adapters.

The toolkit itself stays NumPy-only.  This module lazily imports PyRoki and
its JAX stack only when a solver is constructed, so importing
``robot_toolkit`` does not pull in JAX.

Install PyRoki from source (it is not a hard dependency)::

    git clone https://github.com/chungmin99/pyroki.git
    pip install -e pyroki

Example::

    from robot_toolkit.robot import PyrokiIK

    ik = PyrokiIK(
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


def _require_pyroki():
    try:
        import jax
        import jax.numpy as jnp
        import jaxlie
        import jaxls
        import pyroki as pk
        import yourdfpy
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            'PyRoki IK is optional. Install it with:\n'
            '  git clone https://github.com/chungmin99/pyroki.git\n'
            '  pip install -e pyroki\n'
            'It also needs jax, jaxlie, jaxls and yourdfpy.'
        ) from exc
    return jax, jnp, jaxlie, jaxls, pk, yourdfpy


class PyrokiIK:
    """Solve ``[x, y, z, rx, ry, rz]`` IK with PyRoki's JAX optimizer.

    Parameters
    ----------
    urdf_path:
        Robot URDF.  The model must contain the joints you want to command.
    target_link_name:
        Link whose pose is matched to the target.  If omitted, common names
        (``tcp``, ``tool0``, ``flange``, ``link6``, ``gripper_base``) are
        tried, then the last link in the URDF.
    joint_names:
        Optional output order for the returned joint vector.  Defaults to
        PyRoki's actuated-joint order.  A subset is allowed (for example the
        six arm joints of a URDF that also contains gripper joints); the
        omitted joints stay near PyRoki's default configuration.
    default_joint_cfg:
        Optional seed used as PyRoki's default configuration.
    """

    def __init__(self, urdf_path, target_link_name: str | None = None,
                 joint_names=None, default_joint_cfg=None,
                 pos_weight: float = 50.0, ori_weight: float = 10.0,
                 position_tolerance: float = 0.005,
                 rotation_tolerance: float = 0.05,
                 self_collision: bool = False,
                 collision_margin: float = 0.02,
                 collision_weight: float = 5.0,
                 linear_solver: str = 'dense_cholesky'):
        self.urdf_path = str(Path(urdf_path).expanduser())
        self.target_link_name = target_link_name
        self.joint_names = None if joint_names is None else tuple(joint_names)
        self.default_joint_cfg = default_joint_cfg
        self.pos_weight = float(pos_weight)
        self.ori_weight = float(ori_weight)
        self.position_tolerance = float(position_tolerance)
        self.rotation_tolerance = float(rotation_tolerance)
        self.self_collision = bool(self_collision)
        self.collision_margin = float(collision_margin)
        self.collision_weight = float(collision_weight)
        self.linear_solver = str(linear_solver)
        self._robot = None
        self._jnp = None
        self._jaxlie = None
        self._jaxls = None
        self._pk = None
        self._target_index = None
        self._actuated_names = None
        self._collision = None

    # ------------------------------------------------------------------ load
    def _load(self):
        if self._robot is not None:
            return
        _, jnp, jaxlie, jaxls, pk, yourdfpy = _require_pyroki()
        urdf = yourdfpy.URDF.load(self.urdf_path)
        kwargs = {}
        if self.default_joint_cfg is not None:
            kwargs['default_joint_cfg'] = np.asarray(
                self.default_joint_cfg, dtype=float)
        robot = pk.Robot.from_urdf(urdf, **kwargs)
        names = tuple(robot.links.names)
        target = self.target_link_name
        if target is None:
            target = next((candidate for candidate in
                           ('tcp', 'tool0', 'flange', 'link6', 'gripper_base')
                           if candidate in names), names[-1])
        if target not in names:
            raise ValueError(
                f'target link {target!r} not found; links are {names}')
        self.target_link_name = target
        self._target_index = names.index(target)
        self._actuated_names = tuple(robot.joints.actuated_names)
        if self.joint_names is not None:
            missing = [name for name in self.joint_names
                       if name not in self._actuated_names]
            if missing:
                raise ValueError(
                    f'joint_names not actuated in URDF: {missing}; '
                    f'actuated names are {self._actuated_names}')
        if self.self_collision:
            self._collision = pk.collision.RobotCollision.from_urdf(urdf)
        self._jnp = jnp
        self._jaxlie = jaxlie
        self._jaxls = jaxls
        self._pk = pk
        self._robot = robot

    @property
    def robot(self):
        self._load()
        return self._robot

    @property
    def actuated_names(self):
        self._load()
        return self._actuated_names

    # ----------------------------------------------------------------- solve
    def warmup(self, pose, seed=None) -> bool:
        """Compile the JAX problem once so the first real solve is fast.

        Call this after the robot is connected and the current TCP pose is
        known.  A failed IK result still warms the JAX compilation.
        """
        try:
            self.solve(pose, seed=seed)
            return True
        except Exception:
            return False

    def solve(self, pose, seed=None):
        """Return joint angles for a TCP pose, or ``[]`` if IK fails."""
        self._load()
        jnp = self._jnp
        jaxlie = self._jaxlie
        jaxls = self._jaxls
        pk = self._pk
        robot = self._robot

        pose = np.asarray(pose, dtype=float).reshape(6)
        position = jnp.asarray(pose[:3], dtype=jnp.float32)
        wxyz = matrix_to_quat(rotvec_to_matrix(pose[3:]), order='wxyz')
        target_pose = jaxlie.SE3.from_rotation_and_translation(
            jaxlie.SO3(jnp.asarray(wxyz, dtype=jnp.float32)), position)

        joint_var = robot.joint_var_cls(0)
        costs = [
            pk.costs.pose_cost_analytic_jac(
                robot, joint_var, target_pose,
                jnp.asarray(self._target_index, dtype=jnp.int32),
                pos_weight=self.pos_weight, ori_weight=self.ori_weight),
            pk.costs.rest_cost(
                joint_var, jnp.asarray(joint_var.default_factory()), 0.01),
            pk.costs.limit_constraint(robot, joint_var),
        ]
        if self._collision is not None:
            costs.append(pk.costs.self_collision_cost(
                robot, self._collision, joint_var,
                self.collision_margin, self.collision_weight))
        problem = jaxls.LeastSquaresProblem(
            costs=costs, variables=[joint_var]).analyze()

        solve_kwargs = {
            'verbose': False,
            'linear_solver': self.linear_solver,
            'trust_region': jaxls.TrustRegionConfig(lambda_initial=1.0),
        }
        if seed is not None:
            seed = np.asarray(seed, dtype=float).reshape(-1)
            full_seed = np.asarray(joint_var.default_factory(), dtype=float).copy()
            if (self.joint_names is not None
                    and seed.size == len(self.joint_names)):
                for name, value in zip(self.joint_names, seed):
                    full_seed[self._actuated_names.index(name)] = float(value)
            elif seed.size == robot.joints.num_actuated_joints:
                full_seed = seed
            else:
                raise ValueError(
                    f'seed has {seed.size} joints, expected '
                    f'{robot.joints.num_actuated_joints} or '
                    f'{len(self.joint_names) if self.joint_names else 0}')
            solve_kwargs['initial_vals'] = jaxls.VarValues.make(
                [joint_var.with_value(jnp.asarray(full_seed, dtype=jnp.float32))])
        try:
            solution = problem.solve(**solve_kwargs)
        except Exception:
            return []

        cfg = np.asarray(solution[joint_var], dtype=float).reshape(-1)
        actual = jaxlie.SE3(
            robot.forward_kinematics(jnp.asarray(cfg, dtype=jnp.float32))[
                self._target_index])
        error = (actual.inverse() @ target_pose).log()
        position_error = float(np.linalg.norm(np.asarray(error[:3])))
        rotation_error = float(np.linalg.norm(np.asarray(error[3:])))
        if (position_error > self.position_tolerance
                or rotation_error > self.rotation_tolerance):
            return []

        if self.joint_names is not None:
            indices = [self._actuated_names.index(name)
                       for name in self.joint_names]
            cfg = cfg[indices]
        return cfg

    def __repr__(self) -> str:
        return (f'PyrokiIK(urdf={self.urdf_path!r}, '
                f'target={self.target_link_name!r})')
