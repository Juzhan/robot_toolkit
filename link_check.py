"""Home-button connection check: robot, leader, gripper, cameras, CAN link.

Read-only and backend-agnostic: only toolkit contracts are used (``state()``,
``leader_feedback()``, gripper width, camera snapshots), nothing is commanded.
A robot counts as online only when its feedback keeps updating, not merely
when a connection object exists. Returns ``(ok, summary)`` like the original
UR5 pipeline's check.
"""
from __future__ import annotations

import time
from pathlib import Path

CAMERA_LABELS = {'wrist_camera': '腕部相机', 'external_camera': '外部相机'}


def _can_link(robot):
    """``(name, operstate)`` of the SocketCAN interface, or None if not applicable."""
    inner = getattr(robot, 'inner', robot)
    config = getattr(inner, 'config', None)
    channel = str(getattr(config, 'channel', '') or '')
    interface = str(getattr(config, 'interface', '') or '')
    if channel in ('', 'auto') and hasattr(inner, '_channel'):
        channel = inner._channel()
    if interface in ('', 'auto') and hasattr(inner, '_interface'):
        interface = inner._interface()
    path = Path('/sys/class/net') / channel / 'operstate'
    if interface != 'socketcan' or not path.parent.exists():
        return None
    try:
        return channel, path.read_text().strip()
    except OSError:
        return channel, 'unknown'


def _rx_packets(channel):
    try:
        return int((Path('/sys/class/net') / channel / 'statistics' / 'rx_packets').read_text())
    except (OSError, ValueError):
        return None


def robot_status(robot, probe_s: float = 0.25) -> dict:
    first = robot.state()
    time.sleep(probe_s)
    second = robot.state()
    report = dict(online=False, q=second.q, protective_stop=None, fault=getattr(second, 'fault', None),
                  enabled=getattr(second, 'enabled', None), reason='')
    if second.q is None:
        report['reason'] = '没有关节反馈'
        return report
    updating = first.t != second.t or getattr(robot, 'mock', False)
    if not updating:
        report['reason'] = f'反馈 {probe_s:.2f}s 内没有更新'
        return report
    report['online'] = True
    report['protective_stop'] = (bool(second.protective_stop)
                                 if getattr(second, 'protective_stop_known', True) else None)
    return report


def leader_status(robot) -> dict | None:
    if not hasattr(robot, 'leader_feedback'):
        return None
    from .collect.passive import read_leader
    return read_leader(robot)


def gripper_width(robot, gripper=None):
    try:
        if gripper is not None:
            return float(gripper.width)
        if hasattr(robot, 'gripper_width'):
            return float(robot.gripper_width)
    except Exception:
        return None
    return None


def camera_status(cameras, stale_s: float = 0.5) -> list[dict]:
    """Snapshot freshness per camera plus RealSense enumeration (no stream grab)."""
    found = None
    if any(getattr(camera, 'serial', '') for camera in cameras):
        try:
            import pyrealsense2 as rs
            found = {str(device.get_info(rs.camera_info.serial_number))
                     for device in rs.context().query_devices()}
        except Exception:
            found = None
    rows = []
    for camera in cameras:
        row = dict(name=camera.name, label=CAMERA_LABELS.get(camera.name, camera.name),
                   serial=getattr(camera, 'serial', ''), ok=False, reason='')
        if found is not None and row['serial'] and row['serial'] not in found:
            row['reason'] = f'未识别（序列号 {row["serial"]}）'
        elif not getattr(camera, 'opened', False):
            row['reason'] = '未打开'
        else:
            frame = camera.snapshot()
            if frame.rgb is None:
                row['reason'] = '没有图像'
            elif frame.age() > stale_s:
                row['reason'] = f'图像 {frame.age():.1f}s 未更新'
            else:
                row['ok'] = True
        rows.append(row)
    return rows


def run_link_check(robot, gripper=None, cameras=(), log=print) -> tuple[bool, str]:
    log('连接检查开始')
    clauses, ok = [], True
    link = _can_link(robot)
    rx_before = _rx_packets(link[0]) if link is not None else None
    if link is not None:
        log(f'CAN 接口 {link[0]}：{link[1]}')
        if link[1] not in ('up', 'unknown'):
            clauses.append(f'CAN 接口 {link[0]} 未启用')
            ok = False
    robot_report = robot_status(robot)
    rx_after = _rx_packets(link[0]) if link is not None else None
    if rx_before is not None and rx_after is not None:
        log(f'CAN 收帧：探测期间新增 {rx_after - rx_before}')
        if rx_after == rx_before:
            clauses.append('CAN 总线无数据（机械臂未上电？）')
            ok = False
    if robot_report['online']:
        stop = '，保护停止' if robot_report['protective_stop'] else ''
        log(f'机器人：在线{stop}，使能 {robot_report["enabled"]}，故障 {robot_report["fault"]}，'
            f'关节 {[round(float(v), 3) for v in robot_report["q"]]}')
        clauses.append('机器人保护停止' if robot_report['protective_stop'] else '机器人在线')
        ok = ok and not robot_report['protective_stop']
    else:
        log(f'机器人：离线（{robot_report["reason"]}）')
        clauses.append('机器人离线')
        ok = False
    leader = leader_status(robot)
    if leader is not None:
        from .collect.passive import leader_label
        grip = f'，夹爪指令 {leader["gripper"] * 1000:.1f}mm' if leader['gripper_ok'] else ''
        log(f'主臂：{leader_label(leader)}{grip}')
        clauses.append(leader_label(leader))
        ok = ok and leader['ok']
    width = gripper_width(robot, gripper)
    if width is not None:
        log(f'夹爪：开度 {width * 1000:.1f}mm')
    rows = camera_status(cameras)
    for row in rows:
        log(f'{row["label"]}：{"正常" if row["ok"] else row["reason"]}')
    if rows:
        bad = [row['label'] for row in rows if not row['ok']]
        clauses.append('、'.join(bad) + '异常' if bad else f'{len(rows)} 台相机正常')
        ok = ok and not bad
    summary = '连接检查：' + '，'.join(clauses)
    log(summary)
    return ok, summary
