"""Run a user script against a local Piper X MuJoCo model, never hardware."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path
import runpy
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from PIL import Image

from robot_toolkit import profiles
from robot_toolkit.core.conventions import RobotSpec
from robot_toolkit.core.safety import JointLimits, SafetyGuard
from robot_toolkit.robot.mujoco import MujocoRobot
from robot_toolkit.robot.safety import SafeRobot


def load_model():
    assets = Path(__file__).resolve().parents[1] / 'assets/piperx_assets/assets'
    root = ET.parse(assets / 'piper_x.urdf').getroot()
    for element in list(root):
        if element.get('name', '').startswith('gripper'):
            root.remove(element)
    settings = ET.SubElement(root, 'mujoco')
    ET.SubElement(settings, 'compiler', meshdir=str(assets / 'meshes'),
                  strippath='true', fusestatic='false')
    spec = mujoco.MjSpec.from_string(ET.tostring(root, encoding='unicode'))
    scene = ET.fromstring(spec.to_xml())
    option = scene.find('option')
    if option is None:
        option = ET.SubElement(scene, 'option')
    option.set('timestep', '0.002')
    option.set('integrator', 'implicitfast')
    world = scene.find('worldbody')
    ET.SubElement(world, 'light', pos='0 -1 2', dir='0 0 -1', directional='true')
    ET.SubElement(world, 'geom', name='floor', type='plane', pos='0 0 -0.025',
                  size='2 2 0.1', rgba='0.82 0.84 0.86 1')
    contact = ET.SubElement(scene, 'contact')
    ET.SubElement(contact, 'exclude', body1='base_link', body2='link1')
    palette = ['0.20 0.55 0.49 1', '0.75 0.78 0.80 1', '0.28 0.30 0.32 1']
    for index, body in enumerate(scene.findall('.//body')):
        for geom in body.findall('geom'):
            geom.set('rgba', palette[index % len(palette)])
        if body.get('name') == 'link6':
            ET.SubElement(body, 'site', name='tcp', size='0.008', rgba='1 0.2 0.1 1')
    actuator = ET.SubElement(scene, 'actuator')
    for index in range(1, 7):
        joint = scene.find(f'.//joint[@name="joint{index}"]')
        joint.set('damping', '1')
        joint.set('armature', '0.02')
        ET.SubElement(actuator, 'position', name=f'joint{index}', joint=f'joint{index}',
                      kp='400', kv='40', forcelimited='true', forcerange='-100 100',
                      ctrllimited='true', ctrlrange=joint.get('range'))
    visual = scene.find('visual')
    if visual is None:
        visual = ET.SubElement(scene, 'visual')
    ET.SubElement(visual, 'global', offwidth='1280', offheight='900')
    return mujoco.MjModel.from_xml_string(ET.tostring(scene, encoding='unicode'))


class PreviewRobot(MujocoRobot):
    def __init__(self, model):
        self.viewer = None
        self.samples = []
        super().__init__(model, spec=RobotSpec.piper_x(),
                         joint_names=[f'joint{index}' for index in range(1, 7)],
                         arm_actuators=[f'joint{index}' for index in range(1, 7)],
                         home=[0, 0.6, -0.6, 0, 0, 0])

    def servo_j(self, joints, dt):
        if self.viewer is not None and not self.viewer.is_running():
            raise KeyboardInterrupt
        result = super().servo_j(joints, dt)
        self.samples.append(self.state().q.tolist())
        if self.viewer is not None:
            self.viewer.sync()
        return result


class RecordedRobot(SafeRobot):
    def __init__(self, inner, guard, output):
        super().__init__(inner, guard)
        self.output = output
        self.moves = []

    def move_j(self, joints, **kwargs):
        started = time.monotonic()
        result = super().move_j(joints, **kwargs)
        self.moves.append({'kind': 'move_j', 'success': bool(result),
                           'elapsed_s': time.monotonic() - started,
                           'max_error_rad': float(np.max(np.abs(self.state().q - joints)))})
        self.snapshot(f'move_{len(self.moves):02d}.png')
        return result

    def move_p(self, pose, **kwargs):
        started = time.monotonic()
        result = super().move_p(pose, **kwargs)
        from robot_toolkit.core.mathutils import pose_error
        translation, rotation = pose_error(self.state().tcp, pose)
        self.moves.append({'kind': 'move_p', 'success': bool(result),
                           'elapsed_s': time.monotonic() - started,
                           'translation_error_m': float(translation),
                           'rotation_error_rad': float(rotation)})
        self.snapshot(f'move_{len(self.moves):02d}.png')
        return result

    def snapshot(self, name):
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [0, 0, 0.25]
        camera.distance = 1.25
        camera.azimuth = 130
        camera.elevation = -22
        with mujoco.Renderer(self.inner.model, height=900, width=1280) as renderer:
            renderer.update_scene(self.inner.data, camera=camera)
            Image.fromarray(renderer.render()).save(self.output / name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--script', type=Path,
                        default=Path(__file__).resolve().parents[2] / 'try.py')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--repeat', action='store_true')
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).resolve().parents[2] / 'outputs/piperx_try_mujoco')
    args = parser.parse_args()
    if args.headless and args.repeat:
        parser.error('--repeat needs a viewer')
    args.output.mkdir(parents=True, exist_ok=True)
    inner = PreviewRobot(load_model())
    original_builder = profiles.build_robot
    captured = {}

    def simulation_builder(profile, **kwargs):
        if not isinstance(profile, dict) or profile.get('kind') != 'piper_x':
            raise ValueError('simulation launcher expects the Piper X profile in try.py')
        settings = profile.get('limits', {})
        ranges = inner.model.jnt_range[inner.dof_indices]
        velocity = settings.get('max_velocity')
        limits = (JointLimits(ranges[:, 0], ranges[:, 1]) if velocity is None else
                  JointLimits(ranges[:, 0], ranges[:, 1], velocity))
        robot = RecordedRobot(inner, SafetyGuard(
            limits, max_step_rad=settings.get('max_step_rad', 0.8)), args.output)
        captured['robot'] = robot
        captured['profile'] = profile
        return robot

    profiles.build_robot = simulation_builder
    viewer = None
    try:
        if not args.headless:
            import mujoco.viewer
            viewer = mujoco.viewer.launch_passive(inner.model, inner.data)
            viewer.cam.lookat[:] = [0, 0, 0.25]
            viewer.cam.distance = 1.25
            viewer.cam.azimuth = 130
            viewer.cam.elevation = -22
            inner.viewer = viewer
        runpy.run_path(str(args.script.resolve()), run_name='__main__')
        robot = captured['robot']
        report = {'script': str(args.script.resolve()), 'hardware_connected': False,
                  'model': 'bundled Piper X URDF, no gripper',
                  'profile': captured['profile'], 'moves': robot.moves,
                  'clamped_commands': robot.clamped_commands,
                  'simulation_samples': len(inner.samples),
                  'joint_span_rad': np.ptp(np.asarray(inner.samples), axis=0).tolist(),
                  'all_moves_succeeded': all(move['success'] for move in robot.moves)}
        (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2), flush=True)
        if not report['all_moves_succeeded']:
            raise SystemExit(1)
        while args.repeat and viewer.is_running():
            with contextlib.redirect_stdout(io.StringIO()):
                runpy.run_path(str(args.script.resolve()), run_name='__main__')
    except KeyboardInterrupt:
        pass
    finally:
        profiles.build_robot = original_builder
        if viewer is not None:
            viewer.close()


if __name__ == '__main__':
    main()