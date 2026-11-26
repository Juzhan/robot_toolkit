"""Board calibration mode."""

from __future__ import annotations
import json
import time
from pathlib import Path
from ..calibrate.options import CalibrateOptions
from ..safety.envelope import SafetyEnvelope
from ..teach import KeyReader
from .options import TaskConfig
from .paths import task_dir
from .build import _build_runtime


def _run_calibrate_auto(robot, camera, session, cfg: CalibrateOptions,
                        path, need: int, yes: bool) -> int:
    """Task-level gate (confirmation, safety envelope) around calibrate.auto."""
    if not yes:
        print('自动标定会移动机械臂：确认工作区无人、急停在手后加 --yes')
        return 2
    envelope = None
    if cfg.safety_path:
        try:
            envelope = SafetyEnvelope.load(cfg.safety_path)
        except Exception as exc:
            print(f'安全包络加载失败：{exc}')
            return 2
    elif cfg.require_safety_envelope and not cfg.pose_plan_path:
        print('自动标定必须配置 calibrate.safety_path')
        return 2
    from ..calibrate.auto import run_auto
    return run_auto(robot, camera, session, cfg, path, need, envelope=envelope)

def run_calibrate(task: TaskConfig, mock: bool = False,
                  dataset_override: str = '', yes: bool = False) -> int:
    """Collect board observations from the profile and solve the calibration."""
    from ..calibrate import CalibrationSession, ChessboardDetector

    cfg = task.calibrate
    if cfg.action not in ('solve', 'verify'):
        raise ValueError('calibrate.action must be solve or verify')
    if cfg.quality_protocol and cfg.kind != 'handeye':
        raise ValueError('quality_protocol currently requires kind=handeye')
    if cfg.frames_per_pose < 3:
        raise ValueError('frames_per_pose must be at least 3')
    if cfg.action == 'verify':
        if not cfg.quality_protocol or not Path(cfg.solution_path).is_file():
            raise ValueError('verify requires quality_protocol and an existing solution_path')
    if cfg.quality_protocol and cfg.apply:
        raise ValueError('review independent verification before applying calibration')
    robot, _gripper, cameras = _build_runtime(task, mock, with_cameras=True)
    if not cameras:
        print('calibrate 需要至少一个相机：请在 collect.cameras 里配置')
        return 2
    camera = None
    if cfg.camera:
        camera = next((item for item in cameras if item.name == cfg.camera), None)
    elif cameras:
        camera = cameras[0]
    if camera is None:
        print(f'找不到相机 {cfg.camera!r}；可用：{", ".join(c.name for c in cameras)}')
        return 2
    if cfg.action == 'verify':
        from ..calibrate.quality import validate_identity
        from ..calibrate.identity import calibration_identity
        validate_identity(json.loads(Path(cfg.solution_path).read_text()),
                          calibration_identity(camera, cfg))

    if cfg.board == 'charuco':
        from ..calibrate import CharucoDetector
        detector = CharucoDetector(
            squares=tuple(int(value) for value in cfg.squares),
            square_size=float(cfg.square_size),
            marker_size=float(cfg.marker_size),
            aruco_dict=str(cfg.aruco_dict),
            min_corners=int(cfg.min_corners),
        )
    else:
        from ..calibrate import ChessboardDetector
        pattern = tuple(int(value) for value in cfg.pattern)
        detector = ChessboardDetector(pattern=pattern, square_size=float(cfg.square_size))
    session = CalibrationSession(robot=None if cfg.kind == 'intrinsics' else robot,
                                 camera=camera, detector=detector, mode=cfg.eye,
                                 camera_name=camera.name)
    path = Path(cfg.path) if cfg.path else task_dir(task) / 'calibration.json'
    need = max(3, int(cfg.min_samples))
    if cfg.action == 'verify':
        need = max(8, int(cfg.verify_samples))
    elif cfg.quality_protocol and cfg.verify_after:
        need += max(8, int(cfg.verify_samples))
    if cfg.quality_protocol:
        # Each capture keeps its images, observations and reports; preserve old runs.
        from datetime import datetime
        path = path.parent / datetime.now().strftime('%Y%m%d_%H%M%S_%f') / path.name

    if cfg.auto:
        return _run_calibrate_auto(robot, camera, session, cfg, path, need, yes)

    print(f'标定：camera={camera.name}  kind={cfg.kind}  eye={cfg.eye}  board={cfg.board}')
    if cfg.board == 'charuco':
        print(f'ChArUco {cfg.squares[0]}x{cfg.squares[1]}，方格 {cfg.square_size * 1000:.1f} mm，'
              f'字典 {cfg.aruco_dict}，目标 ≥{need} 组')
    else:
        print(f'棋盘 {cfg.pattern[0]}x{cfg.pattern[1]}，方格 {cfg.square_size * 1000:.1f} mm，'
              f'目标 ≥{need} 组')
    print('按键：空格=采集一帧，s=求解并保存，q=退出（不发送任何运动指令）')
    keys = KeyReader().start()
    quality_views = []
    try:
        if cfg.kind != 'intrinsics':
            robot.connect()
        if not camera.opened:
            camera.open()
        # The first snapshot right after open() is often empty; wait for a real
        # frame so the first key press is not wasted.
        warmed = False
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if camera.snapshot().rgb is not None:
                warmed = True
                break
            time.sleep(0.05)
        print('相机就绪' if warmed else '相机 5 秒内没有出图，检查连接/曝光',
              f'  内参 {"有" if camera.intrinsics is not None else "无"}')
        while True:
            key = keys.get()
            if key == ' ':
                if cfg.quality_protocol:
                    from ..calibrate.protocol import capture_station
                    view = capture_station(session, frames=cfg.frames_per_pose)
                    if view is not None:
                        quality_views.append(view)
                    print(f'有效位姿 {len(quality_views)}/{need}')
                    continue
                frame = camera.snapshot()
                if frame.rgb is None:
                    print('相机没有帧，稍后重试')
                    continue
                if cfg.kind == 'intrinsics':
                    observation = detector.detect(frame.rgb)
                    if observation is None:
                        print('没检测到棋盘')
                        continue
                    session.views.append(observation.view)
                    session.images.append(frame.rgb)
                    print(f'采集 {len(session.views)}/{need} 组视角')
                else:
                    observation = session.capture(image=frame.rgb)
                    if observation is None:
                        print('没检测到棋盘')
                        continue
                    print(f'采集 {len(session.base_tcp)}/{need} 组位姿，'
                          f'视角 {len(session.views)}')
            elif key == 's':
                if cfg.quality_protocol:
                    from ..calibrate.protocol import finish
                    return finish(session, cfg, path, quality_views)
                count = len(session.views) if cfg.kind == 'intrinsics' else len(session.base_tcp)
                if count < 3:
                    print(f'样本太少（{count}），至少需要 3 组，建议 ≥{need}')
                    continue
                summary = session.save(path, camera_name=camera.name, apply=bool(cfg.apply))
                print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
                print(f'已写入 {summary["path"]}')
                break
            elif key == 'q':
                print('退出（未保存）')
                break
            time.sleep(0.02)
    except KeyboardInterrupt:
        pass
    finally:
        keys.stop()
        try:
            camera.close()
        except Exception:
            pass
        try:
            robot.disconnect()
        except Exception:
            pass
    return 0 if path.exists() else 1

