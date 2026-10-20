"""Robot-agnostic keyboard drag-teach.

The only robot-dependent action is ``freedrive(True/False)`` on the backend.
Everything else (sampling, gripper markers, trajectory storage) is shared by
Piper, UR and mock.
"""
from __future__ import annotations

import queue
import sys
import threading
import time

from .recorder import TeachRecorder, TeachRecorderConfig
from .trajectory import Trajectory


class KeyReader:
    """Read single keys on Windows and POSIX terminals."""

    def __init__(self):
        self.events = queue.Queue()
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name='teach-keyboard')
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def get(self):
        try:
            return self.events.get_nowait()
        except queue.Empty:
            return None

    def _run(self):
        if sys.platform == 'win32':
            self._run_windows()
        else:
            self._run_posix()

    def _run_windows(self):
        import msvcrt
        while not self._stop.is_set():
            if msvcrt.kbhit():
                key = msvcrt.getwch()
                if key in ('\x00', '\xe0'):
                    msvcrt.getwch()      # consume arrow/function key
                    continue
                self.events.put(key)
            time.sleep(0.02)

    def _run_posix(self):
        import select
        import termios
        import tty
        fd = sys.stdin.fileno()
        try:
            old = termios.tcgetattr(fd)
        except Exception:
            return
        try:
            tty.setcbreak(fd)
            while not self._stop.is_set():
                ready, _, _ = select.select([sys.stdin], [], [], 0.05)
                if ready:
                    key = sys.stdin.read(1)
                    if key:
                        self.events.put(key)
        finally:
            try:
                termios.tcsetattr(fd, termios.TCSADRAIN, old)
            except Exception:
                pass


def state_line(robot, gripper=None) -> str:
    state = robot.state()
    q = '?' if state.q is None else ' '.join(f'{value:+.3f}' for value in state.q)
    try:
        if gripper is not None:
            grip = f'{gripper.width:.3f}m'
        elif hasattr(robot, 'gripper_width'):
            grip = f'{robot.gripper_width:.3f}m'
        else:
            grip = '?'
    except Exception:
        grip = '?'
    feedback = state.extra.get('feedback', '')
    suffix = f' feedback={feedback}' if feedback else ''
    return f'q=[{q}] grip={grip}{suffix}'


def _wait_for_feedback(robot, expected: str, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + max(0.1, float(timeout))
    while time.monotonic() < deadline:
        if robot.state().extra.get('feedback', '') == expected:
            return True
        time.sleep(0.05)
    return False


def _enter_freedrive(robot) -> None:
    try:
        result = robot.freedrive(True)
    except Exception as exc:
        print(f'自动开启 Freedrive 失败（{exc}）；请在示教器手动开启后再拖动。',
              flush=True)
        return
    if result is False:
        print('自动 Freedrive 返回 False；请在示教器手动开启后再拖动。', flush=True)


def _toggle_gripper(recorder, gripper, closed: bool, label: str) -> bool:
    if gripper is None:
        recorder.mark('mark', label=f'{label}_no_gripper')
        print(f'（没有夹爪，只记录普通标记）', flush=True)
        return closed
    target = gripper.max_width if closed else 0.0
    action = 'open' if target > 0 else 'close'
    try:
        gripper.move(target, wait=False)
    except Exception as exc:
        print(f'夹爪命令失败：{exc}（继续示教）', flush=True)
        return closed
    stage = recorder.mark(action, width=target, label=f'{label}_toggle')
    print(f'{label} -> {action}  {target:.3f} m  t={stage:.2f}s', flush=True)
    return not closed


def keyboard_teach(robot, gripper, *, name: str, task: str = '',
                   rate: float = 50.0, max_duration: float = 300.0,
                   meta: dict | None = None, freedrive: bool = True,
                   wait_feedback: str = '') -> Trajectory | None:
    """Record a drag demonstration and return it; ``None`` means cancelled.

    The caller owns connect/disconnect and trajectory storage.  ``wait_feedback``
    is for backends such as Piper whose follower joint feedback lags leader
    mode: wait until ``state().extra['feedback'] == wait_feedback`` before
    starting the recorder.
    """
    metadata = dict(meta or {})
    metadata.update(name=name, task=task)
    keyboard = KeyReader()
    recorder = TeachRecorder(robot, gripper, TeachRecorderConfig(
        rate=float(rate), max_duration=float(max_duration), meta=metadata))
    trajectory = None
    aborted = False
    try:
        if freedrive:
            _enter_freedrive(robot)
        if wait_feedback and not _wait_for_feedback(robot, wait_feedback):
            raise RuntimeError(
                f'未收到 {wait_feedback!r} 反馈，不能开始示教；'
                '请检查后端/固件，或把 teach.wait_feedback 设为空。')
        recorder.start()
        keyboard.start()
        print('\n示教中：空格=夹爪开/合，s/回车=保存，q/Esc=放弃。', flush=True)
        print('请用手拖动机械臂；物理急停保持在手边。', flush=True)
        print(state_line(robot, gripper), flush=True)
        closed = False
        if gripper is not None:
            try:
                closed = gripper.width < 0.5 * gripper.max_width
            except Exception:
                closed = False
        while True:
            key = keyboard.get()
            if key is not None:
                if key in (' ',):
                    closed = _toggle_gripper(recorder, gripper, closed, '空格')
                elif key in ('s', '\r', '\n'):
                    print('结束示教，保存……', flush=True)
                    break
                elif key in ('q', '\x1b'):
                    print('放弃本次示教。', flush=True)
                    aborted = True
                    break
            time.sleep(0.02)
        trajectory = recorder.stop()
    except KeyboardInterrupt:
        aborted = True
        print('\nCtrl+C，放弃本次示教。', flush=True)
        try:
            trajectory = recorder.stop()
        except Exception:
            trajectory = None
    finally:
        keyboard.stop()
        if gripper is not None:
            try:
                gripper.stop()
            except Exception:
                pass
        try:
            if recorder.recording:
                recorder.stop()
        except Exception:
            pass
        if freedrive:
            try:
                robot.freedrive(False)
            except Exception as exc:
                print(f'退出拖动模式失败：{exc}', flush=True)
    return None if aborted else trajectory


def joycon_teach(robot, gripper, hub, *, name: str, task: str = '',
                 rate: float = 50.0, max_duration: float = 300.0,
                 meta: dict | None = None, freedrive: bool = True,
                 wait_feedback: str = '') -> Trajectory | None:
    """Record a drag demonstration driven by Joy-Con buttons.

    ``ZR``/``R`` toggle the gripper, ``A``/``+`` finish, ``Y``/``-`` cancel.
    The caller is responsible for starting the session only after ``B`` has
    been pressed; this function records immediately.
    """
    if hub is None:
        raise RuntimeError('Joy-Con 未连接：不能开始示教')
    metadata = dict(meta or {})
    metadata.update(name=name, task=task)
    recorder = TeachRecorder(robot, gripper, TeachRecorderConfig(
        rate=float(rate), max_duration=float(max_duration), meta=metadata))
    trajectory = None
    aborted = False
    try:
        if freedrive:
            _enter_freedrive(robot)
        if wait_feedback and not _wait_for_feedback(robot, wait_feedback):
            raise RuntimeError(
                f'未收到 {wait_feedback!r} 反馈，不能开始示教；'
                '请检查后端/固件，或把 teach.wait_feedback 设为空。')
        recorder.start()
        print('\nJoy-Con 示教中：ZR 切夹爪，A 结束保存，Y 放弃。', flush=True)
        print('请用手拖动机械臂；物理急停保持在手边。', flush=True)
        print(state_line(robot, gripper), flush=True)
        closed = False
        if gripper is not None:
            try:
                closed = gripper.width < 0.5 * gripper.max_width
            except Exception:
                closed = False
        while True:
            try:
                button, _stamp = hub.events.get(timeout=0.05)
            except queue.Empty:
                continue
            if button in ('zr', 'r'):
                closed = _toggle_gripper(recorder, gripper, closed, 'ZR')
            elif button in ('a', 'plus'):
                print('结束示教，保存……', flush=True)
                break
            elif button in ('y', 'minus'):
                print('放弃本次示教。', flush=True)
                aborted = True
                break
            elif button == 'home':
                print(state_line(robot, gripper), flush=True)
        trajectory = recorder.stop()
    except KeyboardInterrupt:
        aborted = True
        print('\nCtrl+C，放弃本次示教。', flush=True)
        try:
            trajectory = recorder.stop()
        except Exception:
            trajectory = None
    finally:
        if gripper is not None:
            try:
                gripper.stop()
            except Exception:
                pass
        try:
            if recorder.recording:
                recorder.stop()
        except Exception:
            pass
        if freedrive:
            try:
                robot.freedrive(False)
            except Exception as exc:
                print(f'退出拖动模式失败：{exc}', flush=True)
    return None if aborted else trajectory
