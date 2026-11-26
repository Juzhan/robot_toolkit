"""Command line entry: plan, dispatch and argument parsing."""

from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
from ..safety.envelope import SafetyEnvelope
from ..teach import KeyReader
from .options import TaskConfig
from .build import _build_runtime, _robot_speed
from .calibrate import run_calibrate
from .collect import run_collect
from .config import MODES, load_task_config, print_configs
from .export import run_export
from .hand_demo import run_hand_demo
from .master_slave import run_master_slave
from .replay import run_replay
from .paths import _collect_name, _next_name, dataset_root, lerobot_root, task_dir, trajectory_root
from .safety import run_safety
from .session import run_session
from .teach import run_teach


def print_plan(task: TaskConfig, mode: str, name: str | None,
               dataset_override: str = '') -> None:
    root = trajectory_root(task)
    print(f'配置文件：{task.source_path}')
    print(f'任务：{task.task}  (slug={task.slug})')
    print(f'模式：{mode}')
    if mode != 'hand_demo':
        print(f'机器人：{task.robot.get("kind")}'
              f'  name={task.robot.get("name", "")}'
              f'  max_joint_speed={_robot_speed(task.robot)} rad/s')
        print(f'夹爪：{task.gripper.get("kind", "")}')
        print(f'轨迹目录：{root}')
        print(f'轨迹名称：{name or "(自动生成/使用最新)"}')
    print(f'数据集根目录：{dataset_root(task, dataset_override)}')
    print(f'LeRobot 数据集（跨会话追加）：{lerobot_root(task)}  '
          f'导出：{"独立子进程" if task.collect.isolated_export else "采集进程内"}')
    print(f'相机：{len(task.collect.cameras)} 个'
          + (f" ({', '.join(str(item.get('name', 'camera')) for item in task.collect.cameras)})"
             if task.collect.cameras else '（仅关节/夹爪）'))
    if mode != 'hand_demo':
        print(f'示教：{task.teach.rate} Hz，最长 {task.teach.max_duration} s')
        print(f'回放：speed_scale={task.replay.speed_scale}，'
              f'smoothing={task.replay.smoothing}，'
              f'continuous={task.replay.continuous}')
    if mode == 'session':
        print(f'Joy-Con 会话：prefix={task.session.prefix or task.slug}，'
              f'X 确认={task.session.confirm_replay}，'
              f'自动回放={task.session.auto_replay}')
        print('按键：B 示教 / A 保存 / X×2 回放采集 / Y×2 删除轨迹')
    if mode == 'master_slave':
        duration = ('不限时长' if task.master_slave.sample_seconds <= 0
                    else f'{task.master_slave.sample_seconds:g} s 后自动结束')
        cap = ('单条不限帧数' if not task.collect.max_frames
               else f'单条上限 {task.collect.max_frames} 帧')
        start = '启动即录' if task.master_slave.auto_record else '按空格开始'
        if task.master_slave.interactive and task.master_slave.sample_seconds <= 0:
            print(f'主从采样：fps={task.master_slave.fps:g}，按键会话，{cap}')
            print('按键：B 开始录制 | A 结束保存 | Y×2 丢弃当前 | +×2 删除数据集最后一条 | '
                  'Home 查看状态 | Ctrl+C 退出（键盘 b / a / y / d / h，q 空闲时退出）')
        else:
            print(f'主从采样：fps={task.master_slave.fps:g}，{start}，{duration}，{cap}')
            print('按键：空格 结束并保存（再按开始下一条），x 丢弃当前，q 退出')
        print(f'主从采样：fps={task.master_slave.fps:g}，{start}，{duration}，{cap}')
        print('按键：空格 结束并保存（再按开始下一条），x 丢弃当前，q 退出')
    if mode == 'hand_demo':
        duration = ('不限时长' if task.hand_demo.sample_seconds <= 0
                    else f'{task.hand_demo.sample_seconds:g} s 后自动结束')
        cap = ('单条不限帧数' if not task.collect.max_frames
               else f'单条上限 {task.collect.max_frames} 帧')
        start = '启动即录' if task.hand_demo.auto_record else '按键开始'
        fps = float(task.hand_demo.fps or task.collect.fps or 30.0)
        print(f'人手演示（仅视频）：fps={fps:g}，{start}，'
              f'{duration}，{cap}，导出={task.collect.export}，'
              f'输入={task.hand_demo.input}，'
              f'语音={"开" if task.hand_demo.voice else "关"}')
        if task.hand_demo.tracker:
            print(f'手部跟踪接口已预留但尚未实现：{task.hand_demo.tracker!r}')
        if task.hand_demo.input == 'joycon':
            print('Joy-Con：B 开始，A 结束保存，Y 取消当前，+ 删除上一条，'
                  'Home 状态，- 退出（X/R 无操作）')
        else:
            print('按键：空格 结束并保存，x 丢弃当前，d 删除上一条（两次），q 退出')
    if mode == 'calibrate':
        cfg = task.calibrate
        target = Path(cfg.path) if cfg.path else task_dir(task) / 'calibration.json'
        print(f'标定：kind={cfg.kind} eye={cfg.eye} camera={cfg.camera or "(第一个)"}')
        if cfg.board == 'charuco':
            print(f'ChArUco {cfg.squares[0]}x{cfg.squares[1]}，方格 '
                  f'{cfg.square_size * 1000:.1f} mm，字典 {cfg.aruco_dict}，'
                  f'目标 ≥{max(3, cfg.min_samples)} 组')
        else:
            print(f'棋盘 {cfg.pattern[0]}x{cfg.pattern[1]}，方格 '
                  f'{cfg.square_size * 1000:.1f} mm，目标 ≥{max(3, cfg.min_samples)} 组')
        print(f'输出：{target}')
        print('按键：空格 采集一帧，s 求解并保存，q 退出')
        if cfg.auto and cfg.pose_plan_path:
            print(f'固定视角清单：{cfg.pose_plan_path}')
            print(f'每个位姿等待 {cfg.settle_s:g}s；采集失败时停止，禁止替换样本角色。')
        elif cfg.auto:
            poses = len(cfg.poses) if cfg.poses else max(3, int(cfg.pose_count))
            print(f'自动扫描：{poses} 个位姿，速度 {cfg.speed_scale:g}，'
                  f'停止 {cfg.settle_s:g}s（需要 --yes）')
    if mode == 'safety':
        cfg = task.safety
        target = Path(cfg.path) if cfg.path else task_dir(task) / 'safety_envelope.json'
        print(f'安全包络：人工 Freedrive 采样 {cfg.rate:g} Hz，最长 {cfg.max_duration:g}s')
        print(f'输出：{target}；最小样本 {cfg.min_samples}；关节余量 {cfg.margin_rad:g} rad')
        print('按键：s/回车保存，q 放弃')

def run_safety(task: TaskConfig, mock: bool = False) -> int:
    """Capture a reusable safety envelope while the operator drags the arm."""
    cfg = task.safety
    robot, _gripper, _cameras = _build_runtime(task, mock, with_cameras=False)
    path = Path(cfg.path) if cfg.path else task_dir(task) / 'safety_envelope.json'
    live_path = path.parent / 'live_state.json'
    keys = KeyReader().start()
    samples = []
    started = time.monotonic()
    try:
        robot.connect()
        if cfg.freedrive:
            robot.freedrive(True)
        if cfg.wait_feedback:
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline:
                if robot.state().extra.get('feedback', '') == cfg.wait_feedback:
                    break
                time.sleep(0.05)
            else:
                print(f'未收到 {cfg.wait_feedback!r} 反馈，未开始采集。')
                return 2
        print('安全包络采集：请手动拖动机械臂覆盖允许工作区域。', flush=True)
        print('按 s/回车保存，q 放弃；急停保持在手边。', flush=True)
        period = 1.0 / max(1.0, float(cfg.rate))
        next_sample = time.monotonic()
        while time.monotonic() - started < float(cfg.max_duration):
            now = time.monotonic()
            if now >= next_sample:
                state = robot.state()
                live_path.parent.mkdir(parents=True, exist_ok=True)
                live_path.write_text(json.dumps({
                    'ok': state.q is not None,
                    'q': None if state.q is None else state.q.tolist(),
                    'tcp': None if state.tcp is None else state.tcp.tolist(),
                    'protective_stop': bool(state.protective_stop),
                    'extra': state.extra,
                }, ensure_ascii=False), encoding='utf-8')
                if state.q is not None and state.tcp is not None:
                    samples.append((state.q.copy(), state.tcp.copy()))
                next_sample = now + period
            key = keys.get()
            if key in ('s', '\r', '\n'):
                break
            if key in ('q', '\x1b'):
                print('放弃安全包络采集。', flush=True)
                return 2
            time.sleep(0.01)
        if len(samples) < max(3, int(cfg.min_samples)):
            print(f'样本不足：{len(samples)}，至少需要 {max(3, int(cfg.min_samples))} 个。')
            return 3
        envelope = SafetyEnvelope.from_samples(samples, cfg.margin_rad, cfg.tcp_margin_m)
        envelope.save(path)
        print(json.dumps(envelope.to_dict(), ensure_ascii=False, indent=2))
        print(f'已保存安全包络：{path}')
        return 0
    finally:
        keys.stop()
        if cfg.freedrive:
            try:
                robot.freedrive(False)
            except Exception:
                pass
        robot.disconnect()

def execute(task: TaskConfig, mode: str | None = None, name: str | None = None,
            mock: bool = False, yes: bool = False, force: bool = False,
            speed_scale: float | None = None,
            dataset_override: str = '', dry_run: bool = False,
            auto_record: bool | None = None,
            sample_seconds: float | None = None,
            input_kind: str | None = None,
            voice: bool | None = None) -> int:
    mode = (mode or task.mode).strip().lower()
    if mode not in MODES:
        print(f'未知模式 {mode!r}，可选：{", ".join(MODES)}')
        return 2
    if input_kind is not None:
        task.hand_demo.input = str(input_kind).strip().lower()
    if voice is not None:
        task.hand_demo.voice = bool(voice)
    if auto_record is not None:
        task.master_slave.auto_record = bool(auto_record)
        task.hand_demo.auto_record = bool(auto_record)
    if sample_seconds is not None:
        task.master_slave.sample_seconds = float(sample_seconds)
        task.hand_demo.sample_seconds = float(sample_seconds)
    if mode == 'session':
        if speed_scale is not None:
            task.session.replay_speed_scale = float(speed_scale)
        if dry_run:
            print_plan(task, mode, task.session.prefix or task.slug, dataset_override)
            return 0
        return run_session(task, mock=mock)
    if mode == 'master_slave':
        if dry_run:
            print_plan(task, mode, None, dataset_override)
            return 0
        return run_master_slave(task, mock=mock, dataset_override=dataset_override)
    if mode == 'hand_demo':
        if dry_run:
            print_plan(task, mode, None, dataset_override)
            return 0
        return run_hand_demo(task, mock=mock, dataset_override=dataset_override)
    if mode == 'calibrate':
        if dry_run:
            print_plan(task, mode, None, dataset_override)
            return 0
        return run_calibrate(task, mock=mock, dataset_override=dataset_override,
                             yes=yes)
    if mode == 'safety':
        if dry_run:
            print_plan(task, mode, None, dataset_override)
            return 0
        return run_safety(task, mock=mock)
    if mode == 'agent':
        print('agent 模式由外部智能体框架驱动，例如 UNI：\n'
              f'  python -m robot_toolkit.agent.uni {task.slug} --run-id <id> --instruction <任务> '
              '--output-dir <目录> [--execute]')
        return 2
    root = trajectory_root(task)
    if mode in ('teach', 'teach_collect'):
        resolved = name or task.name or _next_name(task, root)
        if (root / resolved).exists() and not force and not dry_run:
            print(f'轨迹已存在：{root / resolved}；换一个 --name，或加 --force 覆盖。')
            return 2
    else:
        try:
            resolved = _collect_name(task, root, name)
        except FileNotFoundError as exc:
            if not dry_run:
                print(exc)
                return 2
            # A dry run should still show where collect/replay would write,
            # even before the first teach creates the trajectory directory.
            resolved = name or task.name or '(latest)'

    if dry_run:
        print_plan(task, mode, resolved, dataset_override)
        return 0

    if mode in ('teach', 'teach_collect'):
        code = run_teach(task, resolved, mock)
        if code != 0 or mode == 'teach':
            return code
        # Continue with replay + collection using the trajectory just saved.
        return run_collect(task, resolved, mock=mock, yes=yes,
                           speed_scale=speed_scale,
                           dataset_override=dataset_override)
    if mode == 'replay':
        return run_replay(task, resolved, mock, yes, speed_scale)
    if mode == 'collect':
        return run_collect(task, resolved, mock=mock, yes=yes,
                           speed_scale=speed_scale,
                           dataset_override=dataset_override)
    if mode == 'export':
        return run_export(task, resolved)
    return 2

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='piper-task',
        description='按任务 JSON 启动 Piper X 示教 / 回放 / 采集')
    parser.add_argument('task', nargs='?', help='任务配置名，或 JSON 路径')
    parser.add_argument('action', nargs='?', choices=MODES,
                        help='覆盖配置文件里的 mode')
    parser.add_argument('--config', help='显式指定任务 JSON')
    parser.add_argument('--list', action='store_true', help='列出可用任务')
    parser.add_argument('--name', help='覆盖轨迹名称')
    parser.add_argument('--force', action='store_true', help='允许覆盖已有示教轨迹')
    parser.add_argument('--speed-scale', type=float, default=None,
                        help='覆盖回放速度倍率')
    parser.add_argument('--dataset-root', default='', help='覆盖数据集根目录')
    parser.add_argument('--yes', action='store_true', help='跳过回放安全确认')
    parser.add_argument('--mock', action='store_true', help='离线 mock，不接硬件')
    parser.add_argument('--auto-record', action='store_true',
                        help='master_slave / hand_demo：启动即开始录制，不等按键')
    parser.add_argument('--sample-seconds', type=float, default=None,
                        help='master_slave / hand_demo：采样时长（0 = 一直采到按 q）')
    parser.add_argument('--input', choices=['keyboard', 'joycon'], default=None,
                        help='hand_demo：采集按键来源（默认用配置文件里的值）')
    parser.add_argument('--voice', action=argparse.BooleanOptionalAction, default=None,
                        help='hand_demo：开始/保存/丢弃时语音播报（--no-voice 关闭）')
    parser.add_argument('--dry-run', action='store_true', help='只打印解析后的计划，不动机器人')
    parser.add_argument('--verify-solution', help='标定任务：用指定 solution.json 采集独立验证位姿')
    parser.add_argument('--pose-plan', help='按固定的标定/验证姿态清单采样，不重新随机生成')
    return parser

def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.list:
        print_configs()
        return 0
    if args.config and args.task in MODES and args.action is None:
        # ``run.bat --config task.json collect``: the first positional is
        # the action, not a task name.
        args.action, args.task = args.task, None
    if not args.task and not args.config:
        parser.print_help()
        return 0
    try:
        task = load_task_config(args.config or args.task)
        if args.pose_plan:
            task.calibrate.pose_plan_path = args.pose_plan
        if args.verify_solution:
            if args.pose_plan:
                raise ValueError('--verify-solution and --pose-plan cannot be combined')
            task.calibrate.pose_plan_path = ''
            task.calibrate.action = 'verify'
            task.calibrate.quality_protocol = True
            task.calibrate.solution_path = args.verify_solution
            task.calibrate.verify_after = False
            task.calibrate.pose_seed += 10000
    except Exception as exc:
        print(f'加载任务配置失败：{exc}')
        return 2
    try:
        return execute(task, mode=args.action, name=args.name, mock=args.mock,
                       yes=args.yes, force=args.force, speed_scale=args.speed_scale,
                       dataset_override=args.dataset_root, dry_run=args.dry_run,
                       auto_record=args.auto_record or None,
                       sample_seconds=args.sample_seconds,
                       input_kind=args.input, voice=args.voice)
    except Exception as exc:
        print(f'任务执行失败：{type(exc).__name__}: {exc}')
        return 1

if __name__ == '__main__':
    raise SystemExit(main())
