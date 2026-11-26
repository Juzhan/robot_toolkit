"""Drag-teach mode."""

from __future__ import annotations
from ..teach import joint_motion, keyboard_teach, save_trajectory, summarize
from .options import TaskConfig
from .build import _build_runtime, _open_gripper, _close_gripper
from .paths import trajectory_root


def run_teach(task: TaskConfig, name: str, mock: bool) -> int:
    """Drag teach on any backend that implements ``freedrive``."""
    root = trajectory_root(task)
    print(f'示教轨迹目录：{root}')
    robot = None
    gripper = None
    try:
        robot, gripper, _ = _build_runtime(task, mock, with_cameras=False)
        robot.connect()
        _open_gripper(gripper)
        if not task.teach.freedrive:
            print('teach.freedrive=false：请先在示教器开启 Freedrive，再开始拖动机械臂。',
                  flush=True)
        trajectory = keyboard_teach(
            robot, gripper, name=name, task=task.task,
            rate=float(task.teach.rate),
            max_duration=float(task.teach.max_duration),
            meta={'backend': task.robot.get('kind', ''),
                  'camera': 'none'},
            freedrive=bool(task.teach.freedrive),
            wait_feedback=str(task.teach.wait_feedback or ''),
        )
        if trajectory is None:
            return 2
        motion = joint_motion(trajectory)
        print(f'关节最大变化量：{motion:.4f} rad', flush=True)
        if motion < 1e-4 and not task.teach.allow_static:
            print('警告：整段轨迹关节角几乎没变化，未保存。'
                  '若确实只需要夹爪开合，请设置 teach.allow_static=true。')
            return 3
        save_trajectory(root / name, trajectory)
        print('已保存：', root / name)
        print('轨迹摘要：', summarize(trajectory), flush=True)
        return 0
    except Exception as exc:
        print(f'示教失败：{type(exc).__name__}: {exc}')
        return 1
    finally:
        _close_gripper(gripper)
        if robot is not None:
            try:
                robot.disconnect()
            except Exception:
                pass

