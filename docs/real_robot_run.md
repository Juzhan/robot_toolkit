# 真机运行

本文件讲现场怎么把解释器、SDK 和配置对齐，以及一条命令验证机械臂能不能动。
所有命令都在**仓库根目录的父目录**执行（或先 `pip install -e .`）。

## 1. 解释器与依赖

不需要 activate。`run.py` 按下面的顺序选解释器，三个平台一致：

1. 环境变量 `ROBOT_TOOLKIT_PYTHON`；
2. `run.py` 同目录或上级目录的 `.robot_toolkit_python`（内容为解释器路径，已 gitignore）；
3. 当前运行 `run.py` 的解释器。

用 `python -m robot_toolkit.cli doctor` 核对，它会打印解释器、平台和每个 SDK 是否就绪：

```bash
python -m robot_toolkit.cli doctor
```

| 缺什么 | 影响 | 处理 |
|---|---|---|
| `rtde_control` | UR5e 连接直接报错 | `pip install -r requirements/ur5.txt` |
| `pyrealsense2` | 相机打不开 | `pip install -r requirements/cameras.txt` |
| `pyAgxArm` | Piper 连不上 CAN | `pip install -r requirements/piper.txt` |
| `fastapi` / `uvicorn` | agent 服务起不来 | `pip install -r requirements/server.txt` |

## 2. 只读检查（不动机器人）

相机是否枚举得到：

```bash
python -c "import pyrealsense2 as rs; print([d.get_info(rs.camera_info.serial_number) for d in rs.context().devices])"
```

UR 只读（RTDE，不发送任何运动）：

```python
from rtde_receive import RTDEReceiveInterface
r = RTDEReceiveInterface("192.168.1.50", 125.0)
print("connected:", r.isConnected(), "mode:", r.getRobotMode(), "safety:", r.getSafetyMode())
print("q_deg:", [round(v * 57.29578, 2) for v in r.getActualQ()])
print("tcp:", [round(v, 4) for v in r.getActualTCPPose()])
print("speed_scaling:", r.getSpeedScaling())
r.disconnect()
```

走项目自己的适配器读同一份状态：

```python
import numpy as np
from robot_toolkit.task_runner import load_task_config
from robot_toolkit.runtime.profiles import build_robot

task = load_task_config("ur5e_agent")
robot = build_robot(task.robot, mock=False)
robot.connect_read_only()
state = robot.state()
print(np.round(np.degrees(state.q), 2).tolist())
print(np.round(state.tcp, 4).tolist())
robot.disconnect()
```

## 3. 验证位移：nudge

`nudge` 沿世界坐标轴按厘米移动控制点，自动按任务的 `max_step_m` 切分，每步打印回执。

先 `--mock` 只看计划：

```bash
python -m robot_toolkit.cli nudge ur5e_agent 0 0 3 --mock
# start xyz_m [0.0, 0.0, 0.0]
# delta_m [0.0, 0.0, 0.03] steps 2 (max 0.0285 m per step)
# mock: nothing sent to hardware
```

真机必须显式确认（**会让机械臂动**，先确认工作区无人、急停在手）：

```bash
python -m robot_toolkit.cli nudge ur5e_agent 0 0 3 --yes
# step 1/2 accepted=True outcome=completed
# step 2/2 accepted=True outcome=completed
# end xyz_m [...]
```

参数顺序是 `nudge <task> [x] [y] [z]`，单位厘米，缺省 `0 0 3`。
不加 `--mock` 也不加 `--yes` 时只打印提示并退出，不会发指令。

## 4. agent 服务

`python -m robot_toolkit.agent.uni` 把一个工作站包装成环境服务：

```bash
python -m robot_toolkit.agent.uni ur5e_agent \
  --run-id ur5e-observe-1 --instruction "observe scene" --output-dir /tmp/ur5e-observe
```

- 默认只观察：`--execute` 才允许运动；
- `--instruction` 只写进 episode 元数据，没有解析器会去执行它；
- 服务自身不会产生动作，要等规划器客户端连端口发 `/steps`；
- `--output-dir` 必须是不存在的新目录。

## 5. UR5e 上机前核对

`configs/tasks/ur5e_agent.json`：

- `robot.options.ip`（当前 `192.168.1.50`）与 `gripper.options.urcap_port`（63352）对应现场设备；
- 两个相机 `serial`：腕部 `846112070112`、外部 `342522071753`；
- `agent.T_world_base` 仍是单位矩阵，需要按场景标定后填入。

标定在 `configs/calibration/ur5e_collect_cameras.json`（腕部 eye-in-hand、外部 eye-to-hand）。

## 6. Windows 现场笔记

依赖一次装齐（仓库根目录）：

```powershell
pip install -e ".[ur,cameras,server,vision,control]"
python -m robot_toolkit.cli doctor
```

### 网络（必须先做对）

- 有线走 USB 网卡，静态地址 `192.168.1.100`，掩码 `255.255.255.0`，**网关留空**；
- 机器人是 `192.168.1.50`，本机网卡**不能也设成 `.50`**：Windows 会标成
  `AddressState=Duplicate`，`ping` 全部超时；
- Clash Verge / `verge-mihomo` 的 TUN 会劫持 `192.168.1.50`（`Find-NetRoute`
  显示 NextHop `198.18.0.1`），RTDE 报 `RuntimeError: read: End of file`。
  把 `192.168.1.0/24` 设为直连，或临时关闭 TUN。

自检：

```powershell
Get-NetIPAddress -InterfaceAlias '以太网 3' -AddressFamily IPv4
Find-NetRoute -RemoteIPAddress 192.168.1.50   # InterfaceAlias 必须是物理网卡
ping 192.168.1.50
```

### 踩过的坑

- **`speed_scaling = 0`**：示教器速度滑块在 0% 时控制器照样接受 `moveJ`/`servoJ`，
  只是机器人不动，很容易误判成执行成功。`robot/ur_rtde.py` 会在发送运动指令前读一次
  速度标度，为 0 时抛 `RobotUnavailable`；读不到不影响发送。
- **相机没插**：`Station.open()` 会等满 8 秒后抛 `cameras did not deliver frames`，
  只验证机械臂时不要走 agent 服务。
