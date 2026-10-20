# Piper / Piper X（pyAgxArm）

> 独立发布仓库只包含 `projects/robot_toolkit`；文中提到的 `run.bat` / `run`
> 和 `configs/tasks` 是宿主工程里的集成示例，不在本仓库中。

`robot_toolkit` 里的 Piper 后端叫 `agx_piper`，直接使用 AgileX 官方
[`pyAgxArm`](https://github.com/agilexrobotics/pyAgxArm) Python SDK，不依赖 ROS2。
同一套 `RobotBackend` 契约可以驱动 mock、Piper 和 UR5e：

```text
[ j1..j6 (rad), gripper width (m) ]
```

## 安装

官方 SDK 从 GitHub 安装；Windows 还必须安装 AgileX CANDO 的 `python-can` 插件。

Windows（当前机器）：

```powershell
pip install python-can
pip install "git+https://github.com/agilexrobotics/pyAgxArm.git"
pip install "git+https://github.com/agilexrobotics/python-can-agx-cando.git"
```

网络不通时，下载两个仓库的 ZIP 后在本目录执行 `pip install path\to\repo`。

Linux：

```bash
sudo apt install can-utils
sudo ip link set can0 up type can bitrate 1000000
pip install python-can
pip install "git+https://github.com/agilexrobotics/pyAgxArm.git"
```

macOS：

```bash
sudo chmod 777 /dev/ttyACM0       # 实际设备以 ls /dev/tty.* 为准
pip install python-can
pip install "git+https://github.com/agilexrobotics/pyAgxArm.git"
```

平台默认值：

| 平台 | `interface` | `channel` |
| --- | --- | --- |
| Windows | `agx_cando` | `0` |
| Linux | `socketcan` | `can0` |
| macOS | `slcan` | `/dev/ttyACM0` |

## 只读自检（不动机械臂）

```powershell
# 真实 CAN；默认 auto 选择 Windows agx_cando/0
python -m projects.robot_toolkit.cli piper-check --model piper_x

# 不接硬件也能验证后端契约
python -m projects.robot_toolkit.cli piper-check --model piper_x --mock
```

输出包括固件、关节角、TCP、使能状态、夹爪宽度和 SDK 配置。`piper-check` 不会使能、
不会发运动指令。

## 控制

所有运动都必须显式加 `--yes`：

```powershell
# 使能（不运动）
python -m projects.robot_toolkit.examples.piper_control --enable

# 绝对关节角（rad），速度百分比默认 30
python -m projects.robot_toolkit.examples.piper_control \
  --move-j 0 1.1 -1.1 0 0.7 0 --speed-percent 20 --yes

# 笛卡尔点到点：TCP [x, y, z, rx, ry, rz] 旋转向量（m / rad）
python -m projects.robot_toolkit.examples.piper_control \
  --move-p 0.10 0.02 0.25 0 1.57 0 --yes

# 笛卡尔直线：输入与 move_p 相同的 6 维位姿
python -m projects.robot_toolkit.examples.piper_control \
  --move-l 0.12 0.02 0.25 0 1.57 0 --yes

# 夹爪
python -m projects.robot_toolkit.examples.piper_control --open --yes
python -m projects.robot_toolkit.examples.piper_control --close --yes

# 失能电机（关闭机械臂；先托住，失能后会掉电下坠）
python -m projects.robot_toolkit.examples.piper_control --disable --yes

# 软件电子急停（不是物理急停的替代）
python -m projects.robot_toolkit.examples.piper_control --estop --yes
```

Python API：

```python
from projects.robot_toolkit.robot import AgxPiperConfig, AgxPiperRobot

robot = AgxPiperRobot(AgxPiperConfig(
    model='piper_x',
    speed_percent=30,        # 先低速
    enable_on_connect=True,
))
robot.connect()
try:
    print('q  =', robot.state().q)
    print('tcp=', robot.state().tcp)
    robot.move_j([0.0, 1.1, -1.1, 0.0, 0.7, 0.0], wait=True, timeout=60)
    robot.open_gripper(wait=True)
finally:
    robot.disconnect()     # 不会失能；机械臂保持当前位置
```

需要统一安全层时用 profile：

```python
from projects.robot_toolkit.profiles import build_robot, build_gripper

robot = build_robot('piper_x')          # 默认包 SafeRobot + Piper 关节限位
robot.connect()
gripper = build_gripper('agx', robot=robot)
gripper.close(wait=True)
```

`build_robot('piper_x')` 的 `SafeRobot` 会用通用插值；需要 SDK 原生平滑
`move_j` 时直接用 `AgxPiperRobot`，或 profile 里设 `safety=False`。

## 键盘示教（无相机）

日常流程优先用上面的通用 `run.bat`；`piper_teach` 保留为 Piper 专用调试示例，
内部已经复用同一个 `keyboard_teach` 通用循环。

没有相机时，`piper_teach` 只采集**关节轨迹 + 夹爪宽度 + 开关标记**，不生成
LeRobot 图像数据集；数据保存为 `trajectory.npz` / `trajectory.json`。

```powershell
# 先做只读自检，并在示教前关闭 AgileX 上位机
python -m projects.robot_toolkit.cli piper-check --model piper_x

# 进入零力拖动，开始示教
python -m projects.robot_toolkit.examples.piper_teach --name piper_01 --speed-percent 20
```

示教时：

| 操作 | 作用 |
| --- | --- |
| 用手拖动机械臂 | 记录 50 Hz 关节轨迹 |
| 空格 | 夹爪开/合切换，并记录 `close` / `open` 标记 |
| `s` 或回车 | 结束并保存 |
| `q` 或 Esc | 放弃本次示教 |

夹爪默认用 **0.8 s 闭合 / 0.5 s 张开** 的宽度渐变，不再瞬间合上；可用
`--gripper-close-time 1.2`、`--gripper-open-time 0.8` 调慢，设成 `0` 则恢复
SDK 原始速度。回放时也使用同一组参数。

Piper 在 leader 拖动模式下，`get_joint_angles()` 可能一直是 follower 的旧值；
脚本会等待并读取 `get_leader_joint_angles()`，终端应显示 `feedback=leader`。
如果保存前提示“关节最大变化量只有 0.0000 rad”，说明没有采到拖动反馈，
不会保存；重新示教即可。旧版本保存的静止空轨迹也会被回放命令拒绝。

默认输出：

```text
outputs/piper_teach/piper_01/
  trajectory.npz
  trajectory.json
```

示教结束后立即回放（先提示确认急停）：

```powershell
python -m projects.robot_toolkit.examples.piper_teach --name piper_01 --replay --speed-scale 0.3
```

只回放已经保存的轨迹，不再重新示教：

```powershell
python -m projects.robot_toolkit.examples.piper_teach --name piper_01 --replay-only --speed-scale 0.3 --yes
```

回放前默认会要求确认；`--yes` 用于确认急停和工作区安全后跳过交互确认。
回放使用关节轨迹 `servo_j`，夹爪按标记开合；不要在有人的工作区第一次全速回放。

Piper 回放默认 `--max-joint-speed 1.5`，因此 `--speed-scale 1.0` 不再被
`1.0 rad/s` 的安全阈值直接拒绝；`--smoothing 9` 会对 50 Hz 轨迹做约
0.18 s 的窗口平滑。若仍然觉得抖，先试：

```powershell
python -m projects.robot_toolkit.examples.piper_teach --name piper_02 --replay-only --speed-scale 0.5 --smoothing 13 --yes
```

不要为了消除抖动盲目提高 `--max-joint-speed`；先降速度、增平滑。
Piper 后端还会关闭 SDK 每帧自动切换运动模式的行为，只在回放开始时设置一次
J 模式，减少 CAN 控制帧和停顿。
没有相机时不能导出**带图像**的 LeRobot 数据集；但可以用
`DatasetSchema.for_robot(..., cameras=())` + `EpisodeRecorder` 导出一个只含
`observation.state` / `action` 的 LeRobot 数据集。`piper_teach` 本身只保存
`trajectory.npz|json`，不会自动写 LeRobot 数据集。

把已录示教转成 LeRobot：

```powershell
python -m projects.robot_toolkit.examples.piper_collect --source trajectory --name piper_02
```

Joy-Con 采集（可带相机 JSON）：

```powershell
python -m projects.robot_toolkit.examples.piper_collect --source teleop --yes
python -m projects.robot_toolkit.examples.piper_collect --source teleop --cameras-json cameras.json --yes
```

`piper_collect` 和 `joycon_teleop` 复用同一个 `TeleopSession` / `TeleopRunner`，
采集只额外挂 `EpisodeCollector`，没有第二套控制循环。

## 按任务配置启动（通用 `run.bat`）

`run.bat` 读取 `configs/tasks/<任务名>.json`。配置文件里的 `robot` 决定用
Piper、UR5e、MuJoCo 还是 mock，`gripper` 决定夹爪后端；入口本身不判断机器人。

```powershell
# 看有哪些任务
run.bat --list

# 只看解析结果，不接硬件、不动机器人
run.bat stack_the_cube --dry-run

# 推荐：常驻 Joy-Con 会话
# B 开始示教 / A 保存 / ZR 切夹爪 / X×2 回放并采集 / Y×2 删除上一条
run.bat stack_the_cube

# 单次调试命令（不使用常驻会话）
run.bat stack_the_cube teach
run.bat stack_the_cube collect

# 不接硬件跑通整个流程
run.bat stack_the_cube collect --mock --yes
```

macOS / Linux 使用同一个 `run.py`，只是外层脚本不同：

```bash
chmod +x run
./run --list
./run stack_the_cube --dry-run
./run stack_the_cube collect --yes

# 任意平台也可以直接用 Python 入口
python3 run.py stack_the_cube --dry-run
```

解释器优先级：`ROBOT_TOOLKIT_PYTHON` → `.robot_toolkit_python` → `.venv/bin/python`
（Windows 还会尝试 `py` / `python` 和本机准备好的 `lerobot` 环境）。跨平台差异
已经收在 CAN / 串口 / 相机适配器里，任务 JSON 不需要按操作系统分支。

Piper 的 `stack_the_cube.json` 关键字段：

```json
{
  "slug": "stack_the_cube",
  "task": "stack the cube",
  "mode": "session",
  "session": {"prefix": "stack_the_cube", "confirm_replay": true},
  "robot": {
    "kind": "agx_piper",
    "name": "piper_x",
    "options": {"model": "piper_x", "speed_percent": 20,
                "max_joint_speed": 1.5},
    "limits": {"max_velocity": 1.5},
    "safety": true
  },
  "gripper": {"kind": "agx", "options": {"force": 1.0}},
  "teach": {"rate": 50.0, "allow_static": false,
            "freedrive": true, "wait_feedback": "leader",
            "gripper_close_time_s": 0.8, "gripper_open_time_s": 0.5},
  "replay": {"speed_scale": 0.3, "smoothing": 9,
             "rate": 50.0, "allow_fast": false},
  "collect": {"fps": 10.0, "export": "lerobot",
              "repo_id": "local/stack_the_cube",
              "cameras": []}
}
```

换成 UR5e 时只需要改 `robot` / `gripper`，入口和上层流程不变：

```json
"robot": {
  "kind": "ur_rtde",
  "name": "ur5e",
  "options": {"ip": "192.168.1.50", "max_joint_speed": 2.0},
  "limits": {"max_velocity": 2.0},
  "safety": true
},
"gripper": {
  "kind": "robotiq",
  "options": {"backend": "urcap_socket", "serial_port": "COM5"}
}
```

UR CB3 如果只能用示教器按钮进入 Freedrive，把 `teach.freedrive` 设为 `false`，
启动 `teach` 前手动开启 Free；Piper 则保留 `wait_feedback: "leader"`。

仓库自带一份 UR5 开门示例 `configs/tasks/open_the_door.json`（Robotiq URCap +
腕部/外部 RealSense）。它使用通用 runner 的常驻 Joy-Con 会话：

```powershell
run.bat open_the_door
```

会话里：

| 按键 | 作用 |
| --- | --- |
| `B` | 开始一次示教 |
| `A` | 结束并保存轨迹 |
| `ZR` | 夹爪闭合/张开，并记录 marker |
| `X` ×2 | 回放上一条轨迹并录制 episode |
| `Y` ×2 | 删除上一条示教轨迹 |
| `R` | 空闲时缓慢回默认姿态 |
| `Home` | 打印当前状态 |

数据在 `outputs/robot_tasks/open_the_door/`，不会混到旧的
`outputs/ur5e_real_collect/tasks/open_the_door/`。`session` 模式只负责
Joy-Con 常驻示教/采集；单次 `teach` / `collect --mock` 仍可用 `run.bat
open_the_door teach` 或 `collect` 显式调用。

这份 UR 配置启用了 `replay.continuous=true`（Ruckig/限幅连续参考）和
`require_fresh_measurement=true`，参数来自 Mac 真机部署验证过的控制版本。
Windows/Linux 的 125 Hz 回放使用 RTDE `initPeriod()/waitPeriod()`；macOS
自动退回主机时钟，避免 `ur_rtde 1.6.5` 的相对 `waitPeriod` 漂移。

`mode` 可选：

| mode | 行为 |
| --- | --- |
| `teach` | 只拖动示教，保存 `trajectory.npz/json` |
| `replay` | 只回放已有轨迹，不写数据集 |
| `collect` | 回放已有轨迹，同时写 LeRobot 数据集（可选相机图像） |
| `teach_collect` | 先示教，再回放采集 |
| `session` | 常驻 Joy-Con：B/A 示教，X×2 回放采集，可一直重复 |
| `export` | 不动机器人，只把已有轨迹导出为无图像数据集 |

数据默认落在：

```text
outputs/robot_tasks/stack_the_cube/
  trajectories/stack_the_cube_01/
  dataset/<时间戳>/dataset/
```

没有相机时，数据集只有 `observation.state` / `action`（关节 + 夹爪）；
要带图像，在 `collect.cameras` 里加相机配置，例如：

```json
"cameras": [
  {"kind": "realsense", "name": "wrist_camera", "serial": "",
   "width": 640, "height": 480, "fps": 30, "depth": false}
]
```

`collect` 会在回放的同时按 `collect.fps` 采样机器人状态、关节目标、夹爪目标
和相机帧；相机名必须与 `EpisodeCollector` 的 schema 一致。加新任务只需复制
一个 JSON，改 `slug`、`task`、`robot`、`gripper` 和相机即可。

> 之前看到的 `piper_02` 不是预置 demo：它是用
> `piper_teach --name piper_02` 现场示教生成的轨迹。现在任务模式会自动生成
> `stack_the_cube_01`、`stack_the_cube_02` 这样的名字。

## move_j / move_p / move_l 的输入

三者输入**不一样**：

| 接口 | 输入 | 说明 |
| --- | --- | --- |
| `move_j(joints)` | `[j1..j6]`，rad | 关节空间点到点，SDK 会平滑 |
| `move_p(pose)` | `[x,y,z,rx,ry,rz]`，m / rad | 工具库旋转向量；后端转成 SDK 的 RPY |
| `move_l(pose)` | 同 `move_p` | 笛卡尔直线；不要用它连续高频发点 |

`move_p`/`move_l` 和 `state().tcp` 都遵循工具库统一的**旋转向量**契约；
SDK 原始 RPY 放在 `state().extra['tcp_rpy']` 便于诊断。若配置了非零
`tcp_offset`，后端会先把 TCP 目标转成法兰目标再调用 SDK。

```python
pose = robot.state().tcp          # [x, y, z, rx, ry, rz] 旋转向量
target = pose.copy()
target[2] += 0.01                 # 先只抬高 1 cm
robot.move_p(target, wait=True)
```

## 关键点

- **固件**：`firmware='auto'` 会读 `get_firmware()` 后用
  `resolve_firmware_profile()` 重连；也可以显式写 `default`/`v183`/`v188`/`v189`。
- **速度**：SDK 用 `set_speed_percent(0..100)`，不是每个关节的 rad/s；
  `move_j` 的 `speed`/`accel` 参数不会传给 SDK。第一次把 `speed_percent` 设 10–30。
- **笛卡尔**：后端另外暴露 `move_p(pose)` / `move_l(pose)`，`pose` 为
  `[x,y,z,rx,ry,rz]` 旋转向量（m / rad），和 UR 的 `state().tcp` 保持一致；
  后端内部转成 SDK 的 RPY。使用前先 `set_tcp_offset()` 或在配置里填
  `tcp_offset`，否则按法兰位姿解释。
- **使能**：`connect()` 默认 `enable_on_connect=True`；`piper-check` 固定只读。
- **夹爪**：通过 `init_effector(AGX_GRIPPER)` 创建，单位米，0 闭合；
  夹持力用 `gripper_force_n`（默认 1 N）。
- **拖动示教**：`freedrive(True)` 调 `set_leader_mode()`（零力拖动），
  `freedrive(False)` 调 `set_follower_mode()`；和 UR 的 freedrive 语义不完全一样。
  leader 模式下关节可能因重力下坠，切换前先托住机械臂，并随时准备物理急停。
- **停止**：`stop_motion()` 把当前关节角重新下发为目标；真正的急停用物理急停。
  `emergency_stop()` 只发软件电子急停，之后要按控制器流程恢复。
- **断开**：`disconnect()` 不会 `disable()`，避免抬起的机械臂掉落。
- **关闭**：先降到安全姿态并托住机械臂，再 `disable()` 失能，最后 `disconnect()`
  断开 CAN；彻底断电用控制箱电源开关。SDK 的使能状态反馈有延迟，`disable()`
  会自动重试最多 3 秒。急停请按物理急停，`--estop` 只是软件电子急停。
- **CAN**：只支持 AgileX 官方 CAN 模块；Windows 用 `agx_cando`，不要把
  `socketcan`/`slcan` 的路径混用。

## 验证

```powershell
python -m pytest projects\robot_toolkit\tests -q
```

离线测试不需要 CAN/SDK；真实连接前先用 `piper-check` 确认
`is_ok`、关节反馈和固件正常，再做小角度 `--move-j`。
