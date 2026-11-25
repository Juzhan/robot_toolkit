"""UNI (uni-robot-agent) environment backend over a robot_toolkit ``Station``.

Only format translation lives here: world poses as metres + wxyz quaternions,
normalized gripper openings, StepReceipt recovery facts and calibrated frames.
Motion safety, stop observation and operator blocking belong to the Station.
Runs in an environment that has ``uni_robot_agent`` importable (its runtime
or main distribution); the planner connects with
``uni-robot-agent run --integration robot_toolkit --env-url ...``.

    python -m robot_toolkit.agent.uni ur5e_session --execute --run-id trial-001 \\
        --instruction "put the cube in the bowl" --output-dir ./env-runs/trial-001
"""

from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path

import numpy as np

from ..core.mathutils import matrix_to_quat, quat_to_matrix
from .station import Rejected, Station

INTEGRATION = 'robot_toolkit'


def _contracts():
    from uni_robot_agent.services.environment import contracts
    return contracts


def _pose(T):
    c = _contracts()
    return c.Pose(position=tuple(float(v) for v in T[:3, 3]),
                  quaternion=tuple(float(v) for v in matrix_to_quat(T[:3, :3], order='wxyz')))


def _matrix(pose) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = quat_to_matrix(pose.quaternion, order='wxyz')
    T[:3, 3] = pose.position
    return T


class UniBackend:
    """Implements ``services.environment.backend.EnvironmentBackend`` (stepwise motion)."""

    def __init__(self, station: Station, *, episode_id: str):
        c = _contracts()
        self.station, self.episode_id = station, episode_id
        self.arm = station.options.arm_name
        self.steps = 0
        self.ended = False
        self._outcome = None
        self._lock = threading.RLock()
        info = station.describe()
        metadata = {
            'robot_id': info['robot_id'], 'robot': info['robot'], 'backend': info['backend'],
            'physical_hardware': info['physical'], 'execute': info['execute'],
            'motion_execution': 'stepwise', 'finish_statuses': ['success', 'failure', 'give_up'],
            'stop_motion_kind': info['stop_motion_kind'], 'station_limits': info['limits'],
            'gripper': info['gripper'],
            'control_frequency_semantics': 'maximum_submission_rate_not_realtime_guarantee',
        }
        self.description = c.EnvironmentDescription(
            arms=[c.ArmDescription(
                name=self.arm, control_frame=f'{self.arm}_control_point',
                grasp_site_description=info['control_site'],
                rotation_radius_m=info['rotation_radius_m'], joint_names=info['joint_names'],
                joint_tracking_tolerance_rad=station.options.joint_tolerance_rad,
                control=c.ArmControlCapabilities(
                    commands=['pose', 'joints'],
                    gripper='fixed' if info['gripper'] is None else 'continuous'))],
            cameras=[c.CameraDescription(name=cam['name'], role=cam['role'], has_depth=cam['has_depth'],
                                         has_calibration=cam['has_calibration']) for cam in info['cameras']],
            control_frequency_hz=station.options.control_frequency_hz,
            max_episode_steps=station.options.max_episode_steps,
            # The service publishes one snapshot per request; periodic captures would repeat it.
            observation_interval_steps=station.options.max_episode_steps + 1,
            metadata=metadata)

    # ---------------------------------------------------------------- state
    def _state(self, station_state=None):
        c = _contracts()
        s = station_state or self.station.read()
        command = s.gripper if self.station.gripper_command is None else self.station.gripper_command
        arm = c.ArmState(control_pose=_pose(s.T_world_control), joints=[float(v) for v in s.q],
                         gripper=float(s.gripper), gripper_command=float(command),
                         metadata={'gripper_opening_measured': s.gripper_measured,
                                   'gripper_width_m': s.gripper_width, 'feedback_time': s.t,
                                   'timestamp_clock': s.timestamp_clock,
                                   'protective_stop': s.protective_stop, 'fault': s.fault})
        return c.EnvironmentState(arms={self.arm: arm}, state_version=self.steps, native_steps=self.steps,
                                  ended=self.ended, metadata={'physical_hardware': self.station.physical})

    def read_state(self):
        return self._state()

    # --------------------------------------------------------------- motion
    def validate_commands(self, commands, mode):
        c = _contracts()
        if not commands:
            raise c.RequestRejected('Empty command sequence')
        for command in commands:
            if not isinstance(command, (c.PoseCommand, c.JointCommand)) or set(command.arms) != {self.arm}:
                raise c.RequestRejected(f'Commands must name exactly the arm {self.arm!r}')

    def _receipt(self, receipt):
        c = _contracts()
        outcome = None if receipt.outcome == 'completed' else receipt.outcome
        diagnostics = {'reason': receipt.reason, **receipt.diagnostics}
        state = self._state(receipt.state) if receipt.state is not None else self._fallback_state()
        return c.StepReceipt(accepted=receipt.accepted, state=state, diagnostics=_jsonable(diagnostics),
                             execution_outcome=outcome, stop_confirmed=receipt.stop_confirmed,
                             requires_operator_recovery=receipt.requires_operator)

    def _fallback_state(self):
        state = self._state()
        state.metadata['feedback_fresh'] = False
        return state

    def _step(self, target):
        c = _contracts()
        if self.ended or self.steps >= self.station.options.max_episode_steps:
            raise c.RequestRejected('The episode has ended or exhausted its step budget')
        with self._lock:
            try:
                if isinstance(target, c.JointTarget):
                    receipt = self.station.move_joints(target.joints, gripper=target.gripper)
                else:
                    receipt = self.station.move_pose(_matrix(target.pose), gripper=target.gripper)
            except Rejected as error:
                raise c.RequestRejected(f'{error}; no command was sent') from error
            if receipt.outcome != 'none':
                self.steps += 1
            return self._receipt(receipt)

    def step_pose(self, command):
        self.validate_commands([command], 'direct')
        return self._step(command.arms[self.arm])

    def step_joints(self, command):
        self.validate_commands([command], 'direct')
        return self._step(command.arms[self.arm])

    def step_native(self, command):
        raise _contracts().RequestRejected('This environment has no native command channel')

    def preview_joints(self, commands):
        raise _contracts().RequestRejected('This environment has no joint preview')

    # ---------------------------------------------------------- observation
    def capture(self, cameras=None, *, high_resolution=False):
        from uni_robot_agent.services.environment.contracts import RequestRejected
        from uni_robot_agent.services.environment.snapshots import CameraFrame, CapturedObservation
        try:
            frames = self.station.capture(cameras)
            state = self._state()
        except Exception as error:
            raise RequestRejected(f'Observation failed ({type(error).__name__}: {error}); '
                                  'no physical action was executed') from error
        # UNI accepts calibration only as a complete pair.
        payload = {name: CameraFrame(rgb=frame.rgb, depth_m=frame.depth_m,
                                     intrinsic_K=frame.K if frame.T_world_optical is not None else None,
                                     T_world_from_optical=frame.T_world_optical if frame.K is not None else None,
                                     calibration={'source': frame.calibration_source, 'frame_time': frame.t})
                   for name, frame in frames.items()}
        return CapturedObservation(self.episode_id, self.steps, time.time(), state.model_dump(mode='json'),
                                   payload, terminated=self.ended)

    # ------------------------------------------------------------ lifecycle
    def finish(self, reason, status=None):
        with self._lock:
            if self._outcome is None:
                self.ended = True
                stop = self.station.stop() if self.station.physical else {'status': 'not_required'}
                # Physical task success is assessed outside this process.
                self._outcome = {'task_success': None, 'physical_task_evaluated': False,
                                 'completion_claim': status, 'reason': reason, 'stop': stop,
                                 'physical_hardware': self.station.physical,
                                 'robot_id': self.description.metadata['robot_id'],
                                 'operator_block': self.station.blocked,
                                 'retryable': False, 'native_steps': self.steps}
        return self._outcome

    def close(self):
        self.station.close()


def _jsonable(value):
    from ..robot.base import _json_value
    return _json_value(value)


# --------------------------------------------------------------------- serving
def build_service(station: Station, output_dir, *, episode_id: str, instruction: str):
    from uni_robot_agent.artifacts.run import write_json_atomic
    from uni_robot_agent.artifacts.store import ArtifactStore
    from uni_robot_agent.services.environment.contracts import EpisodeInfo
    from uni_robot_agent.services.environment.service import EnvironmentService
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    backend = UniBackend(station, episode_id=episode_id)
    write_json_atomic(output_dir / 'station.json', _jsonable(station.describe()))
    info = EpisodeInfo(episode_id=episode_id, task=instruction, integration=INTEGRATION,
                       environment=backend.description,
                       metadata={'instruction': instruction, 'skill_namespace': INTEGRATION,
                                 'robot_id': backend.description.metadata['robot_id']})
    service = EnvironmentService(backend, info, ArtifactStore(output_dir))
    service.initialize()
    return service


def serve(service, output_dir, *, host: str, port: int) -> None:
    """Loopback only, like the upstream physical integrations; tunnel for remote planners."""
    from contextlib import asynccontextmanager

    import uvicorn
    from fastapi import FastAPI
    from uni_robot_agent.artifacts.run import write_json_atomic
    from uni_robot_agent.services.environment.routes import build_router
    if host not in {'127.0.0.1', 'localhost'}:
        raise ValueError('Physical environment routes must bind loopback; use an SSH tunnel remotely')

    def stop(reason):
        write_json_atomic(Path(output_dir) / 'environment-outcome.json', service.backend.finish(reason))

    @asynccontextmanager
    async def lifespan(app):
        try:
            yield
        finally:
            try:
                stop('server_shutdown')
            finally:
                service.close()

    class Server(uvicorn.Server):
        def handle_exit(self, sig, frame):
            threading.Thread(target=stop, args=('operator_cancelled',), daemon=True).start()
            super().handle_exit(sig, frame)

    app = FastAPI(lifespan=lifespan)
    app.include_router(build_router(service))
    Server(uvicorn.Config(app, host=host, port=port)).run()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description='Serve one robot_toolkit station as a UNI environment')
    parser.add_argument('task', help='robot_toolkit task config name or path')
    parser.add_argument('--execute', action='store_true', help='allow motion (default: observation only)')
    parser.add_argument('--mock', action='store_true')
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--instruction', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=19320)
    args = parser.parse_args(argv)
    station = Station.from_task(args.task, mock=args.mock).open(execute=args.execute)
    try:
        service = build_service(station, args.output_dir, episode_id=args.run_id, instruction=args.instruction)
    except BaseException:
        station.close()
        raise
    serve(service, args.output_dir, host=args.host, port=args.port)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
