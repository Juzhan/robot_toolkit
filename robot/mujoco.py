"""Generic MuJoCo arm backend plus damped-least-squares IK.

This is the simulation counterpart of the real-robot adapter: both expose
``state`` / ``servo_j`` / ``servo_l`` / ``move_j`` through ``RobotBackend``.
The model is supplied by the caller, so the toolkit itself contains no robot
description and works for any Menagerie arm with a TCP site.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ..core.conventions import GripperSpec, RobotSpec
from ..core.mathutils import matrix_to_rotvec, rotvec_to_matrix
from ..errors import Aborted, NotSupported
from .base import Capabilities, RobotBackend, RobotState


def _require_mujoco():
    try:
        import mujoco
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError('MuJoCo backend requires `pip install mujoco`') from exc
    return mujoco


class DlsIk:
    """Damped least squares IK around a seed configuration.

    Running IK on the host (rather than inside the controller) keeps the math
    shared between MuJoCo scenes and any backend that lacks native Cartesian
    servo.  Joint limits and seed proximity are enforced on every iteration.
    """

    def __init__(self, model, site: str, joint_names, iterations: int = 40,
                 position_tolerance: float = 5e-5, rotation_tolerance: float = 1e-3,
                 damping: float = 1e-5, rotation_weight: float = 0.15,
                 max_step: float = 0.04):
        mujoco = _require_mujoco()
        self.mujoco = mujoco
        self.model = model
        self.site = model.site(site).id
        self.joint_names = tuple(joint_names)
        self.qpos_indices = np.array([model.joint(name).qposadr[0] for name in self.joint_names])
        self.dof_indices = np.array([model.joint(name).dofadr[0] for name in self.joint_names])
        self.lower = np.array([model.joint(name).range[0] for name in self.joint_names])
        self.upper = np.array([model.joint(name).range[1] for name in self.joint_names])
        self.iterations = int(iterations)
        self.position_tolerance = float(position_tolerance)
        self.rotation_tolerance = float(rotation_tolerance)
        self.damping = float(damping)
        self.rotation_weight = float(rotation_weight)
        self.max_step = float(max_step)
        self.scratch = mujoco.MjData(model)

    def solve(self, position, rotation_matrix, seed):
        mujoco = self.mujoco
        seed = np.asarray(seed, dtype=float)
        # Start from the seed without disturbing the caller's MjData.
        self.scratch.qpos[:] = 0.0
        self.scratch.qpos[self.qpos_indices] = seed
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        position = np.asarray(position, dtype=float)
        rotation_matrix = np.asarray(rotation_matrix, dtype=float)
        for _ in range(self.iterations):
            mujoco.mj_forward(self.model, self.scratch)
            error_position = position - self.scratch.site_xpos[self.site]
            current_rotation = self.scratch.site_xmat[self.site].reshape(3, 3)
            error_rotation = matrix_to_rotvec(rotation_matrix @ current_rotation.T)
            if (np.linalg.norm(error_position) < self.position_tolerance and
                    np.linalg.norm(error_rotation) < self.rotation_tolerance):
                break
            error = np.concatenate([error_position, error_rotation * self.rotation_weight])
            mujoco.mj_jacSite(self.model, self.scratch, jacp, jacr, self.site)
            jacobian = np.vstack([jacp[:, self.dof_indices],
                                  jacr[:, self.dof_indices] * self.rotation_weight])
            delta = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + np.eye(6) * self.damping, error)
            self.scratch.qpos[self.qpos_indices] = np.clip(
                self.scratch.qpos[self.qpos_indices] + np.clip(delta, -self.max_step, self.max_step),
                self.lower, self.upper)
        mujoco.mj_forward(self.model, self.scratch)
        error_position = float(np.linalg.norm(position - self.scratch.site_xpos[self.site]))
        error_rotation = float(np.linalg.norm(matrix_to_rotvec(
            rotation_matrix @ self.scratch.site_xmat[self.site].reshape(3, 3).T)))
        return self.scratch.qpos[self.qpos_indices].copy(), error_position, error_rotation


class MujocoRobot(RobotBackend):
    """MuJoCo arm driven by joint position actuators.

    Parameters
    ----------
    model:
        A ``mujoco.MjModel``, a path to an MJCF file, or an XML string.
    joint_names:
        Actuated joints, in contract order.
    tcp_site:
        Site that defines the tool centre point pose.
    arm_actuators:
        Actuator names in joint order; defaults to the first ``dof`` actuators.
    gripper_actuator:
        Optional actuator driven by ``set_gripper(width)``.
    """

    def __init__(self, model, spec: RobotSpec | None = None, joint_names=None,
                 tcp_site: str = 'tcp', arm_actuators=None, gripper_actuator: str | None = None,
                 home=None, substeps: int | None = None, gravity_compensation: bool = True,
                 clock=time.monotonic):
        mujoco = _require_mujoco()
        self.mujoco = mujoco
        if isinstance(model, (str, Path)):
            text = str(model)
            self.model = (mujoco.MjModel.from_xml_path(text) if Path(text).exists()
                          else mujoco.MjModel.from_xml_string(text))
        else:
            self.model = model
        self.data = mujoco.MjData(self.model)
        if joint_names is None:
            joint_names = [name for name in _actuated_joint_names(self.model)][:6]
        joint_names = tuple(joint_names)
        if spec is None:
            spec = RobotSpec(name='mujoco', joint_names=joint_names,
                             gripper=(None if gripper_actuator is None
                                      else GripperSpec(name=gripper_actuator)))
        super().__init__(spec, Capabilities(servo_j=True, servo_l=True, move_j=True,
                                            move_l=True, inverse_kinematics=True,
                                            freedrive=False,
                                            gripper=gripper_actuator is not None))
        self.joint_names = joint_names
        self.qpos_indices = np.array([self.model.joint(name).qposadr[0] for name in joint_names])
        self.dof_indices = np.array([self.model.joint(name).dofadr[0] for name in joint_names])
        self.actuator_ids = (np.array([self.model.actuator(name).id for name in arm_actuators])
                             if arm_actuators else np.arange(len(joint_names)))
        self.gripper_id = (self.model.actuator(gripper_actuator).id
                           if gripper_actuator else None)
        self.tcp_site = tcp_site
        self.clock = clock
        self.gravity_compensation = bool(gravity_compensation)
        self._t0 = clock()
        self.home = np.asarray(home, dtype=float) if home is not None else np.zeros(len(joint_names))
        self.ik = DlsIk(self.model, tcp_site, joint_names)
        self.substeps = substeps
        self.reset()

    # ------------------------------------------------------------------ scene
    def reset(self, joints=None) -> None:
        self.mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.qpos_indices] = self.home if joints is None else np.asarray(joints)
        self.data.ctrl[self.actuator_ids] = self.data.qpos[self.qpos_indices]
        self.mujoco.mj_forward(self.model, self.data)

    def _step(self, seconds: float) -> None:
        step = float(self.model.opt.timestep)
        count = self.substeps or max(1, int(round(seconds / step)))
        for _ in range(count):
            if self.gravity_compensation:
                # Plain position actuators droop under gravity, which makes a
                # simulated arm feel different from a real servo.  Cancelling
                # the bias forces restores pose hold without changing dynamics
                # the scene author intended.
                self.mujoco.mj_forward(self.model, self.data)
                self.data.qfrc_applied[:] = 0.0
                self.data.qfrc_applied[self.dof_indices] = self.data.qfrc_bias[self.dof_indices]
            self.mujoco.mj_step(self.model, self.data)
        if self.gravity_compensation:
            self.data.qfrc_applied[:] = 0.0

    # ----------------------------------------------------------------- state
    def state(self) -> RobotState:
        self.mujoco.mj_forward(self.model, self.data)
        position = self.data.site_xpos[self.model.site(self.tcp_site).id].copy()
        rotation = self.data.site_xmat[self.model.site(self.tcp_site).id].reshape(3, 3).copy()
        tcp = np.concatenate([position, matrix_to_rotvec(rotation)])
        q = self.data.qpos[self.qpos_indices].copy()
        qd = self.data.qvel[self.dof_indices].copy()
        return RobotState(t=self.clock() - self._t0, q=q, qd=qd, tcp=tcp,
                          protective_stop=False, connected=self.connected,
                          timestamp_clock='host_elapsed',
                          tcp_frame='world',
                          sources={'q': 'simulation', 'qd': 'simulation', 'tcp': 'forward_kinematics'},
                          extra={'gripper': self.gripper_width(), 'simulation_time': float(self.data.time)})

    def describe(self) -> dict:
        description = super().describe()
        description['tcp_frame'] = 'world'
        return description

    def tcp_pose(self) -> np.ndarray:
        return self.state().tcp

    def inverse_kinematics(self, pose, seed=None):
        if self._ik_solver is not None:
            return self._ik_solver.solve(pose, seed=seed)
        pose = np.asarray(pose, dtype=float).reshape(6)
        q0 = self.state().q if seed is None else np.asarray(seed, dtype=float)
        joints, position_error, rotation_error = self.ik.solve(
            pose[:3], rotvec_to_matrix(pose[3:]), q0)
        if position_error > 0.005 or rotation_error > 0.05:
            return []
        return joints.tolist()

    # ---------------------------------------------------------------- motion
    def servo_j(self, joints, dt: float) -> bool:
        target = np.asarray(joints, dtype=float)
        if target.shape != (len(self.joint_names),):
            raise ValueError(f'expected {len(self.joint_names)} joints, got {target.shape}')
        self.data.ctrl[self.actuator_ids] = target
        self._step(dt)
        return True

    def servo_l(self, pose, dt: float, max_position_error: float = 0.005,
                max_rotation_error: float = 0.05) -> bool:
        pose = np.asarray(pose, dtype=float)
        joints, position_error, rotation_error = self.ik.solve(
            pose[:3], rotvec_to_matrix(pose[3:]), self.state().q)
        if position_error > max_position_error or rotation_error > max_rotation_error:
            return False
        return self.servo_j(joints, dt)

    def move_j(self, joints, speed=None, accel=None, wait=True, rate_hz=100.0,
               timeout=None, abort_event=None) -> bool:
        return RobotBackend.move_j(self, joints, speed=speed, accel=accel, wait=wait,
                                   rate_hz=rate_hz, timeout=timeout, abort_event=abort_event)

    def wait_until_settled(self, target=None, timeout: float = 10.0,
                           rate_hz: float = 50.0, tolerance: float = 0.02,
                           abort_event=None) -> bool:
        """Keep stepping physics while waiting; a MuJoCo scene has no background clock."""
        if target is None:
            target = self.state().q
        target = np.asarray(target, dtype=float)
        dt = 1.0 / max(1.0, float(rate_hz))
        deadline = time.monotonic() + float(timeout)
        while time.monotonic() < deadline:
            if abort_event is not None and abort_event.is_set():
                raise Aborted('settle aborted')
            current = self.state().q
            if current is not None and float(np.max(np.abs(current - target))) <= tolerance:
                return True
            self.servo_j(target, dt)
            time.sleep(dt)
        return False

    def move_l(self, pose, speed=None, accel=None, wait=True, rate_hz=100.0) -> bool:
        joints, position_error, rotation_error = self.ik.solve(
            np.asarray(pose, dtype=float)[:3],
            rotvec_to_matrix(np.asarray(pose, dtype=float)[3:]),
            self.state().q)
        if position_error > 0.005 or rotation_error > 0.05:
            return False
        return self.move_j(joints, speed=speed, accel=accel, wait=wait, rate_hz=rate_hz)

    # --------------------------------------------------------------- gripper
    def set_gripper(self, width: float) -> None:
        if self.gripper_id is None or self.spec.gripper is None:
            raise NotSupported('this MuJoCo model has no gripper actuator')
        normalized = self.spec.gripper.clamp(width) / max(self.spec.gripper.max_width, 1e-9)
        self.data.ctrl[self.gripper_id] = 255.0 * (1.0 - normalized)

    def gripper_width(self):
        if self.spec.gripper is None or self.gripper_id is None:
            return None
        return self.spec.gripper.max_width * (1.0 - float(self.data.ctrl[self.gripper_id]) / 255.0)

    def snapshot(self):
        """Deep copy of scene state, for tests and deterministic reruns."""
        return {
            'qpos': self.data.qpos.copy(),
            'qvel': self.data.qvel.copy(),
            'ctrl': self.data.ctrl.copy(),
            'time': float(self.data.time),
        }

    def restore(self, snapshot) -> None:
        self.data.qpos[:] = snapshot['qpos']
        self.data.qvel[:] = snapshot['qvel']
        self.data.ctrl[:] = snapshot['ctrl']
        self.data.time = snapshot['time']
        self.mujoco.mj_forward(self.model, self.data)


def _actuated_joint_names(model):
    names = []
    for actuator in range(model.nu):
        joint = int(model.actuator_trnid[actuator, 0])
        if joint >= 0:
            names.append(model.joint(joint).name)
    return names
