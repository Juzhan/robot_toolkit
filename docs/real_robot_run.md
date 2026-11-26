# 真机运行：环境与命令

不用激活任何环境。用一个启动脚本固定解释器和路径，所有后端共用。

```bash
cd /home/jz/ws/robot_toolkit
./scripts/realrun.sh env          # 看当前解释器与硬件依赖
```

解释器固定为 `RealSense-Stereo-Trial/.venv-linux/bin/python`，脚本自行设置
`PYTHONPATH=/home/jz/ws:/home/jz/ws/uni-robot-agent`。只有需要换解释器时才用 `PY=... ./scripts/realrun.sh`。

## 为什么是这一个环境

| 环境 | pyrealsense2 | pyAgxArm | UR rtde | UNI 服务 | 用途 |
|---|---|---|---|---|---|
| `RealSense-Stereo-Trial/.venv-linux` | 有 | 有 | 缺 | 有（靠 PYTHONPATH） | 真机运行 |
| `uni-robot-agent/.venv-lite` | 缺 | 缺 | 缺 | 有（靠 PYTHONPATH） | 只能 mock |
| `StrangeLoop/.venv` | 缺 | 缺 | 缺 | 缺 | 只能跑离线测试 |

三个环境都没把 `robot_toolkit`、`uni_robot_agent` 装成包，全靠工作目录能导入到。
所以"换环境"实际是换解释器加补路径，脚本已经替你做完。

`ur_rtde` 在所有环境都缺，UR5e 真机连接会失败。需要时安装：

```bash
/home/jz/ws/RealSense-Stereo-Trial/.venv-linux/bin/python -m pip install ur_rtde
```

## 0. 前置检查（不动机器人）

```bash
cd /home/jz/ws/robot_toolkit
# 相机能否枚举
/home/jz/ws/RealSense-Stereo-Trial/.venv-linux/bin/python -c "import pyrealsense2 as rs; print([d.get_info(rs.camera_info.serial_number) for d in rs.context().devices])"
# 离线测试
./scripts/realrun.sh test
```

## 1. Piper：只读观察（不动）

```bash
cd /home/jz/ws/robot_toolkit
./scripts/realrun.sh observe piper_agent
```

## 2. Piper：StrangeLoop 适配器读取真机状态与像素换算（不动）

注意这里多用 `StrangeLoop/src` 一条路径，其余与脚本相同。

**当前跑不了**：远端 main 重构后 `strangeloop/environment/adapters/base.py` 已不存在，
未提交的 `real.py` 仍引用它，导入即报错。脚本的 `strangeloop` 子命令会先检查并直接告知这一点。

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

## 3. 真正让机械臂动：nudge（**会动**）

`execute` 不会执行任何动作，它只是把工作站当成 UNI 环境服务跑起来，等规划器客户端连上来
发 `/steps`。`--instruction` 只写进 episode 元数据，没有任何解析器去读它。
所以只跑 `execute` 时机械臂不会动，直到你 Ctrl+C。

要验证一个明确位移，直接用 nudge：沿世界坐标轴按厘米移动控制点，自动按 `max_step_m` 切分。

```bash
cd /home/jz/ws/robot_toolkit
./scripts/realrun.sh nudge piper_agent 0 0 3      # 往上 3 cm
MOCK=1 ./scripts/realrun.sh nudge piper_agent 0 0 3   # 离线验证切分与回执
```

参数顺序是 `nudge <task> [x] [y] [z]`，单位厘米，缺省 `0 0 3`。每步打印 `accepted` 与 `outcome`，
最后打印实际到达位置。

## 4. Piper：UNI 带动作执行（**会动**）

仅在确认观察结果正确、并且有明确动作目标时使用。脚本会先要求输入 `yes` 确认。
需要已有 UNI 规划器客户端连端口才会产生动作。

```bash
cd /home/jz/ws/robot_toolkit
INSTRUCTION="把红色方块放到目标区" ./scripts/realrun.sh execute piper_agent
```

## 5. UR5e：只读观察（不动）

先确认 `configs/tasks/ur5e_agent.json` 中的 `ip` 与现场一致，且 `T_world_base` 已按场景确认（当前为单位矩阵）。缺 `ur_rtde` 时脚本会直接报错退出，不会碰硬件。

```bash
cd /home/jz/ws/robot_toolkit
./scripts/realrun.sh observe ur5e_agent
./scripts/realrun.sh nudge ur5e_agent 0 0 3
```

## 6. UR5e：UNI 带动作执行（**会动**）

```bash
cd /home/jz/ws/robot_toolkit
INSTRUCTION="<具体任务描述>" ./scripts/realrun.sh execute ur5e_agent
```

`OUT` 必须是不存在的目录，未指定时用 `/tmp/<mode>_<时间戳>`；`PORT` 默认 19360。

## 注意事项

- UR5e 相机序列号：wrist `846112070112`，external `342522071753`，标定文件
  `configs/calibration/ur5e_collect_cameras.json`。
- UR5e 的 `T_world_base` 为单位矩阵，真机前必须按现场修正；`ip` 当前是 `192.168.1.50`。
- UR5e 要真机运行，先装 `ur_rtde`，否则脚本会在启动前报缺依赖。
- 适配器 `real.py` 默认 Piper 任务，用于 UR5e 前需确认其假设。
- StrangeLoop 的 `real.py`、`real_robot.yaml` 与测试仍未提交，且已跟当前 main 的接口脱节。
- `execute` 会真实驱动机械臂，只能由人在现场确认急停可用后执行。
