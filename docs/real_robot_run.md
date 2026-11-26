# 真机运行命令（Piper / UR5e）

本文件是人工执行手册。命令会在真机上产生运动的，已标注 **会动**。请先在现场确认急停可用、工作区无人、机械臂处于可控状态。

## 0. 前置检查（不动机器人）

```bash
cd /home/jz/ws
# 相机能否枚举
/home/jz/ws/RealSense-Stereo-Trial/.venv-linux/bin/python -c "import pyrealsense2 as rs; print([d.get_info(rs.camera_info.serial_number) for d in rs.context().devices])"
# 离线测试
cd /home/jz/ws/StrangeLoop && PYTHONPATH=/home/jz/ws:/home/jz/ws/StrangeLoop/src /home/jz/ws/RealSense-Stereo-Trial/.venv-linux/bin/python -m pytest -q tests/test_real_environment_adapter.py
```

## 1. Piper：只读观察（不动）

```bash
cd /home/jz/ws && /home/jz/ws/uni-robot-agent/.venv-lite/bin/python -m robot_toolkit.agent.uni robot_toolkit/configs/tasks/piper_agent.json --run-id piper-observe-$(date +%H%M%S) --instruction "observe scene" --output-dir /tmp/piper_observe_$(date +%H%M%S) --port 19360
```

## 2. Piper：StrangeLoop 适配器读取真机状态与像素换算（不动）

```bash
cd /home/jz/ws/StrangeLoop && PYTHONPATH=/home/jz/ws:/home/jz/ws/StrangeLoop/src timeout 90 /home/jz/ws/RealSense-Stereo-Trial/.venv-linux/bin/python - <<'PY'
from strangeloop.environment.adapters.real import RealEnvironmentAdapter
from strangeloop.environment.robot.geometry import pixel_to_world
paths = ["/home/jz/ws/outputs/robot_tasks/piper_calibrate_eye_to_hand/20261117_175617_987429/calibration.json",
         "/home/jz/ws/outputs/robot_tasks/piper_calibrate_auto/calibration.json"]
adapter = RealEnvironmentAdapter(task_config="piper_master_slave", mock=False, calibration_paths=paths)
try:
    obs = adapter.reset("piper_pick_red_cube", 0, "ee_delta")
    print("tcp_xyz:", [round(v, 3) for v in adapter.robot_state()["primary"]["eef_position_m"]])
    for name, cal in obs["cameras"].items():
        depth = obs["depth"][name]
        h, w = depth.shape
        points = []
        for u, v in ((w // 2, h // 2), (w // 4, 3 * h // 4), (3 * w // 4, 3 * h // 4)):
            p = pixel_to_world(depth, cal["intrinsics"], cal["pose"], float(u), float(v), radius=2)
            points.append([round(x, 3) for x in p["position_m"]] if p["valid"] else None)
        print(f"cam {name}: source={cal['pose']['source']} pixels_to_world={points}")
finally:
    adapter.close()
    print("closed: true")
PY
```

## 3. Piper：UNI 带动作执行（**会动**）

仅在确认观察结果正确、并且有明确动作目标时使用。

```bash
cd /home/jz/ws && /home/jz/ws/uni-robot-agent/.venv-lite/bin/python -m robot_toolkit.agent.uni robot_toolkit/configs/tasks/piper_agent.json --execute --run-id piper-run-$(date +%H%M%S) --instruction "<具体任务描述>" --output-dir /tmp/piper_run_$(date +%H%M%S) --port 19361
```

## 4. UR5e：只读观察（不动）

先确认 `robot_toolkit/configs/tasks/ur5e_agent.json` 中的 `ip` 与现场一致，且 `T_world_base` 已按场景确认（当前为单位矩阵）。

```bash
cd /home/jz/ws && /home/jz/ws/uni-robot-agent/.venv-lite/bin/python -m robot_toolkit.agent.uni robot_toolkit/configs/tasks/ur5e_agent.json --run-id ur5e-observe-$(date +%H%M%S) --instruction "observe scene" --output-dir /tmp/ur5e_observe_$(date +%H%M%S) --port 19362
```

## 5. UR5e：UNI 带动作执行（**会动**）

```bash
cd /home/jz/ws && /home/jz/ws/uni-robot-agent/.venv-lite/bin/python -m robot_toolkit.agent.uni robot_toolkit/configs/tasks/ur5e_agent.json --execute --run-id ur5e-run-$(date +%H%M%S) --instruction "<具体任务描述>" --output-dir /tmp/ur5e_run_$(date +%H%M%S) --port 19363
```

## 注意事项

- `--output-dir` 必须是不存在的新目录，否则服务启动失败。
- UR5e 相机序列号：wrist 846112070112，external 342522071753。标定文件为 `robot_toolkit/configs/calibration/ur5e_collect_cameras.json`。
- UR5e 的 `T_world_base` 为单位矩阵，真机前必须按现场修正。
- 适配器 `real.py` 目前默认 Piper 任务，用于 UR5e 前需确认其假设。
- StrangeLoop 的 `real.py`、`real_robot.yaml` 与测试仍未提交。
