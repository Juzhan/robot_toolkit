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

## Windows：本机 UR5（192.168.1.50）

上面几节针对 Linux 的 `scripts/realrun.sh`。Windows 没有这个脚本，直接用 conda `base`
解释器，不要新建环境。一键装齐（仓库根目录）：

```powershell
pip install -e ".[ur,cameras,server,vision,control]"
python -m robot_toolkit.cli doctor        # 按功能核对，缺什么会给出对应 extra
```

实测依赖：

| 组件 | 版本 |
|---|---|
| Python | `D:\app\conda\python.exe` 3.13.5 |
| ur_rtde | 1.6.6（cp313 win_amd64 wheel） |
| pyrealsense2 | 2.59.2 |
| fastapi / uvicorn | 0.165 / 0.54 |
| opencv-python | 4.14.0（`vision` 可选依赖） |
| ruckig | 0.19.4（`control` 可选依赖，连续回放需要） |

### 网络（必须先做对）

- 有线走 USB 网卡（ASIX USB GbE），静态地址 `192.168.1.100`，掩码 `255.255.255.0`，**网关留空**。
- 机器人是 `192.168.1.50`。本机网卡**不能也设成 `.50`**：Windows 会把地址标成
  `AddressState=Duplicate`，`ping` 全部超时。
- Clash Verge / `verge-mihomo` 的 TUN 会劫持 `192.168.1.50`（`Find-NetRoute` 显示
  NextHop `198.18.0.1`），RTDE 报 `RuntimeError: read: End of file`。把
  `192.168.1.0/24` 设为直连，或临时关闭 TUN。

自检：

```powershell
Get-NetIPAddress -InterfaceAlias '以太网 3' -AddressFamily IPv4
Find-NetRoute -RemoteIPAddress 192.168.1.50   # InterfaceAlias 必须是物理网卡，不是 Meta
ping 192.168.1.50
```

### 只读验证（不动机械臂）

```powershell
@'
from rtde_receive import RTDEReceiveInterface
r = RTDEReceiveInterface("192.168.1.50", 125.0)
print("connected:", r.isConnected(), "mode:", r.getRobotMode(), "safety:", r.getSafetyMode())
print("q_deg:", [round(v * 57.29578, 2) for v in r.getActualQ()])
print("tcp:", [round(v, 4) for v in r.getActualTCPPose()])
print("speed_scaling:", r.getSpeedScaling())
r.disconnect()
'@ | python -
```

走项目自己的适配器：

```powershell
$env:PYTHONPATH='D:\ws\robot'
@'
import numpy as np
from robot_toolkit.task_runner import load_task_config
from robot_toolkit.profiles import build_robot
task = load_task_config("ur5e_agent")
robot = build_robot(task.robot, mock=False)
robot.connect_read_only()
state = robot.state()
print(np.round(np.degrees(state.q), 2).tolist())
print(np.round(state.tcp, 4).tolist())
robot.disconnect()
'@ | python -
```

### Windows 上踩过的坑

- **`speed_scaling = 0`**：示教器速度滑块在 0% 时控制器照样接受 `moveJ`/`servoJ`，
  只是机器人不动，很容易误判成"执行成功"。`robot/ur_rtde.py` 现在会在发送运动指令前
  读一次速度标度，为 0 时直接抛 `RobotUnavailable`；读不到不影响发送。
- **相机没插**：`Station.open()` 会等满 8 秒后抛 `cameras did not deliver frames`，
  只验证机械臂时不要走 `agent.uni`。
- **机型是 UR5（CB3，PolyScope 3.15.2）**，实测 `robot_mode=7`、`safety_mode=1`。
  仓库的 `RobotSpec.ur5e()` 只提供关节名，运动学由控制器解算，宿主侧不受影响；
  只有接 `robot/ik.py` 的外部 URDF IK 才需要区分 UR5 与 UR5e。
- `ur_rtde 1.6.6` 没有 `getTCPOffset()`，TCP 偏移看 `configs/tasks/ur5e_agent.json`
  的 `tcp_offset` 或示教器配置。

## 注意事项

- UR5e 相机序列号：wrist `846112070112`，external `342522071753`，标定文件
  `configs/calibration/ur5e_collect_cameras.json`。
- UR5e 的 `T_world_base` 为单位矩阵，真机前必须按现场修正；`ip` 当前是 `192.168.1.50`。
- UR5e 要真机运行，先装 `ur_rtde`，否则脚本会在启动前报缺依赖。
- 适配器 `real.py` 默认 Piper 任务，用于 UR5e 前需确认其假设。
- StrangeLoop 的 `real.py`、`real_robot.yaml` 与测试仍未提交，且已跟当前 main 的接口脱节。
- `execute` 会真实驱动机械臂，只能由人在现场确认急停可用后执行。
