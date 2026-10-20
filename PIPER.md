# Piper / Piper X（pyAgxArm）

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
python -m robot_toolkit.cli piper-check --model piper_x

# 不接硬件也能验证后端契约
python -m robot_toolkit.cli piper-check --model piper_x --mock
```

输出包括固件、关节角、TCP、使能状态、夹爪宽度和 SDK 配置。`piper-check` 以
`configure_on_connect=False` 连接：**不使能、不设速度、不设运动模式、不写 TCP 偏移，
全程不发送任何报文**。主从手臂并线时这一点很关键——早期的 `connect()` 会发 `0x151`
（`ctrl_mode=0x01` CAN 指令控制 + `MOVE J`），总线上两台手臂都会收到，会被拉出
主从示教、表现为拖不动。

## 总线扫描（只监听，不发送）

主从手臂并到同一个接线端子时，它们共用 **1 个 CAN 适配器 / 1 个 channel**，
所以适配器数量不代表手臂数量。要看总线上到底有几台手臂，用被动监听：

```powershell
# 监听 3 秒（默认），列出总线上出现的 CAN ID、帧率和估计在线手臂数
python -m robot_toolkit.cli piper-scan

# 指定 channel / 监听时长 / 落盘
python -m robot_toolkit.cli piper-scan --channel 0 --seconds 5 --json scan.json
```

判定规则（基于 Piper 固定仲裁 ID）：

| 观测 | 含义 |
| --- | --- |
| `0x155-0x157` | 主臂（leader）正在广播关节目标，主从示教模式在跑 |
| `0x251-0x256` / `0x261-0x266` | 从臂电机/驱动器高速与低速反馈 |
| `0x2A1-0x2A7` | 从臂状态、法兰位姿、关节角 |
| `0x2A8` | 夹爪反馈 |
| `0x1C0-0x1C3` | Revo2 灵巧手反馈（状态/手指位置/速度/电流） |

只有 `0x155-0x157` 与从臂反馈同时出现才算 2 台手臂。两台手臂共用同一组 CAN ID 时
不会出现额外 ID，而是**帧率叠加**（例如 `0x2A1` 从 200 Hz 变成约 400 Hz），`piper-scan`
会据此提示可能有第二台手臂。一条帧都收不到时，先查适配器枚举（`adapters` 字段）
和 CAN_H/CAN_L 线序、1 Mbps 波特率。`piper-scan` 全程不发送任何报文。

## 恢复主从示教（0x470）

如果两台手臂被切到了 CAN 指令控制模式（`0x2A1` 的 byte0 = `0x01`、byte2 = `0x01`
即 `MOVE J`），表现为拖不动、示教无效，按下面顺序恢复：

1. 先托住从臂，两台手臂断电，等 10 秒再上电——重启会清掉 `ctrl_mode`。
2. 上电后仍拖不动，就单独配置联动角色（`0x470` 是广播帧，两台并线都会收到，
   所以一次只给一台手臂上电）：

```powershell
# 只给主臂上电
python -m robot_toolkit.cli piper-linkage --role leader --yes
# 断电，只给从臂上电
python -m robot_toolkit.cli piper-linkage --role follower --yes
```

`0xFA` = 示教输入臂（Leader，零力拖动），`0xFC` = 运动输出臂（Follower）。该命令只发
一帧 `0x470`，不会像 `connect()` 那样顺带发 `0x151`。

3. 验证：`piper-scan` 应看到 `0x155-0x157`（主臂关节广播），`arms_estimate` 变成 2。

## 主从示教直接采样（不发送任何报文）

主从模式由手臂自己完成随动，宿主程序只负责**读**和**存**：

```powershell
# 空格=开始/结束一条 episode，x=丢弃，q=退出
python -m robot_toolkit.examples.piper_collect --source master-slave

# 带腕部相机 + 固定 5 秒时长（脚本化/无键盘时用 --auto-record）
python -m robot_toolkit.examples.piper_collect `
  --source master-slave --auto-record --sample-seconds 5 `
  --fps 15 --cameras-json cameras.json `
  --dataset-root outputs/piper_dataset --repo-id local/piper_task
```

`cameras.json` 形如：

```json
[{"kind": "opencv", "name": "wrist_camera", "width": 640, "height": 480, "fps": 30}]
```

要点：

- 该模式用 `configure_on_connect=False` 连接，**不发 `0x151`、不使能、不查固件**，
  主从状态不受影响；采样前后都可以用 `piper-scan` 确认 `0x155-0x157` 仍在。
- `observation.state` = 从臂实际关节 + 夹爪宽度；`action` = 主臂通过 `0x155-0x157`
  广播的关节角（没有主臂广播时退化为等于 observation）。没有仿真，不会额外 import
  不需要的 SDK。
- 采样节奏由循环本身按 `--fps` 控速，实测 15 Hz 目标下约 14.9 Hz。
- 腕部相机的实际出图帧率由相机自身决定；`fps` 低于相机帧率时会重复取最新帧。

## 用配置文件启动（推荐）
任务配置由 `task_runner` 驱动，一个 JSON 就够。仓库外的工作区里已放好
`configs/tasks/piper_master_slave.json`。日常直接用 `run.bat`
（跨平台等价 `python run.py`），它会选好解释器和包路径：

```powershell
run.bat --list                                   # 列出任务
run.bat piper_master_slave                       # 直接跑：空格开始/结束，q 退出（不限时长）
run.bat piper_master_slave master_slave --dry-run  # 只解析，不连硬件
```

默认和 UR5 的 `collect` 一样，是**一条命令就开录**：链路就绪后立即录制，按空格结束并保存
（再按空格录下一条），不设时长、不设帧数上限。限制全部是显式可选项：

| 字段 | 默认 | 说明 |
|---|---|---|
| `master_slave.auto_record` | `true` | 启动即录；设 `false` 改成按空格才开始 |
| `master_slave.sample_seconds` | `0` | `>0` 才启用「到点自动结束」 |
| `collect.max_frames` | `0` | `0` = 单条 episode 不限帧数 |
| `collect.compress` | `false` | spool 的 `.npz` 帧是否压缩。默认原始帧（~1.5 MB/帧、~3 ms/帧）；压缩只省约 5% 空间却慢 ~20 倍，30 Hz 会「write queue is full」中断 |

`run.bat` 的解释器优先级：`ROBOT_TOOLKIT_PYTHON` → 同目录或上级目录的
`.robot_toolkit_python` 文件 → PATH 里的 `py`/`python`。要用 LeRobot 导出就指向装了
`lerobot` 的环境。在工作区根目录放一份 `.robot_toolkit_python`，工作区根目录和仓库里的
`run.bat` 都能用到。
等价的模块命令是 `python -m robot_toolkit.task_runner <task> [mode]`。

关键字段：

```json
{
  "mode": "master_slave",
  "robot": {
    "kind": "agx_piper", "name": "piper_x",
    "options": {
      "interface": "auto", "channel": "auto", "firmware": "default",
      "enable_on_connect": false,      // 只读
      "configure_on_connect": false,   // 不写 0x151
      "has_gripper": true
    },
    "safety": false
  },
  "gripper": {"kind": "none"},
  "collect": {
    "repo_id": "local/piper_master_slave", "fps": 30.0, "export": "lerobot",
    "feature_names": "unit",
    "cameras": [{"kind": "realsense", "name": "wrist_camera",
                 "width": 640, "height": 480, "fps": 30, "depth": true}]
  },
  "master_slave": {"fps": 30.0}
}
```

`master_slave` 段只写 `fps` 就是操作员控制模式（空格开始/结束、q 退出、不限时长）。
`sample_seconds > 0` 会自动开录并在到点后收尾，仅供无人值守/测试；`collect.max_frames`
同理，`0` 表示单条 episode 不设帧数上限。`fps` 建议与相机帧率对齐（腕部相机 30 fps
→ 30 Hz）。

`master_slave` 模式会强制 `enable_on_connect=false` / `configure_on_connect=false`，
即使 JSON 里写反了也不会往总线写帧。`export` 用 `lerobot` 需要装有 LeRobot 的环境，
否则改成 `"manifest"`。

## 标定（棋盘格，配置驱动）

仓库自带 `configs/tasks/piper_calibrate.json`，`mode` 是 `calibrate`：

```powershell
run.bat piper_calibrate      # 空格=采集一帧，s=求解并保存，q=退出
```

```json
"calibrate": {
  "kind": "handeye",        // handeye | intrinsics | both
  "eye": "eye_in_hand",     // eye_in_hand | eye_to_hand（腕部相机用 eye_in_hand）
  "camera": "wrist_camera",
  "pattern": [9, 6],        // 棋盘内角点数，不是方格数
  "square_size": 0.025,     // 米
  "min_samples": 8,
  "path": "",               // 空 = outputs/robot_tasks/<slug>/calibration.json
  "apply": false
}
```

- 主从联动下摆位 = 拖主臂，程序只读从臂 TCP + 腕部相机，**不发运动指令**；
- `handeye` 需要 `robot.options.tcp_offset` 正确（否则外参偏一个固定量）；
- `intrinsics` 只采棋盘视角、不读机器人位姿；
- 输出为 `CalibrationStore` JSON，`python -m robot_toolkit.cli calib-show <path>` 可查看摘要；
- 标定板可用 `assets/board_charuco_300dpi.pdf`（ChArUco）或任意 9×6 棋盘，按实际内角点改 `pattern`。

需要**全自动扫描**时用 `configs/tasks/piper_calibrate_auto.json`（单臂、主机可控，会移动机械臂）：

```powershell
run.bat piper_calibrate_auto --yes
```

逐点 `move_j` → 停 `settle_s` → 检测棋盘 → 攒够样本自动求解；检测失败的位姿自动跳过。
主从硬件联动下**不能**用（发运动指令会打断联动），只能用上面的手动摆位版。

## 控制

所有运动都必须显式加 `--yes`：

```powershell
# 使能（不运动）
python -m robot_toolkit.examples.piper_control --enable

# 绝对关节角（rad），速度百分比默认 30
python -m robot_toolkit.examples.piper_control \
  --move-j 0 1.1 -1.1 0 0.7 0 --speed-percent 20 --yes

# 笛卡尔点到点：TCP [x, y, z, rx, ry, rz] 旋转向量（m / rad）
python -m robot_toolkit.examples.piper_control \
  --move-p 0.10 0.02 0.25 0 1.57 0 --yes

# 笛卡尔直线：输入与 move_p 相同的 6 维位姿
python -m robot_toolkit.examples.piper_control \
  --move-l 0.12 0.02 0.25 0 1.57 0 --yes

# 夹爪
python -m robot_toolkit.examples.piper_control --open --yes
python -m robot_toolkit.examples.piper_control --close --yes

# 失能电机（关闭机械臂；先托住，失能后会掉电下坠）
python -m robot_toolkit.examples.piper_control --disable --yes

# 软件电子急停（不是物理急停的替代）
python -m robot_toolkit.examples.piper_control --estop --yes
```

Python API：

```python
from robot_toolkit.robot import AgxPiperConfig, AgxPiperRobot

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
from robot_toolkit.profiles import build_robot, build_gripper

robot = build_robot('piper_x')          # 默认包 SafeRobot + Piper 关节限位
robot.connect()
gripper = build_gripper('agx', robot=robot)
gripper.close(wait=True)
```

`build_robot('piper_x')` 的 `SafeRobot` 会用通用插值；需要 SDK 原生平滑
`move_j` 时直接用 `AgxPiperRobot`，或 profile 里设 `safety=False`。

## 键盘示教（无相机）

`piper_teach` 是 Piper 专用调试示例，内部已经复用同一个 `keyboard_teach` 通用循环。

没有相机时，`piper_teach` 只采集**关节轨迹 + 夹爪宽度 + 开关标记**，不生成
LeRobot 图像数据集；数据保存为 `trajectory.npz` / `trajectory.json`。

```powershell
# 先做只读自检，并在示教前关闭 AgileX 上位机
python -m robot_toolkit.cli piper-check --model piper_x

# 进入零力拖动，开始示教
python -m robot_toolkit.examples.piper_teach --name piper_01 --speed-percent 20
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
python -m robot_toolkit.examples.piper_teach --name piper_01 --replay --speed-scale 0.3
```

只回放已经保存的轨迹，不再重新示教：

```powershell
python -m robot_toolkit.examples.piper_teach --name piper_01 --replay-only --speed-scale 0.3 --yes
```

回放前默认会要求确认；`--yes` 用于确认急停和工作区安全后跳过交互确认。
回放使用关节轨迹 `servo_j`，夹爪按标记开合；不要在有人的工作区第一次全速回放。

Piper 回放默认 `--max-joint-speed 1.5`，因此 `--speed-scale 1.0` 不再被
`1.0 rad/s` 的安全阈值直接拒绝；`--smoothing 9` 会对 50 Hz 轨迹做约
0.18 s 的窗口平滑。若仍然觉得抖，先试：

```powershell
python -m robot_toolkit.examples.piper_teach --name piper_02 --replay-only --speed-scale 0.5 --smoothing 13 --yes
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
python -m robot_toolkit.examples.piper_collect --source trajectory --name piper_02
```

Joy-Con 采集（可带相机 JSON）：

```powershell
python -m robot_toolkit.examples.piper_collect --source teleop --yes
python -m robot_toolkit.examples.piper_collect --source teleop --cameras-json cameras.json --yes
```

`piper_collect` 和 `joycon_teleop` 复用同一个 `TeleopSession` / `TeleopRunner`，
采集只额外挂 `EpisodeCollector`，没有第二套控制循环。

## 按任务 JSON 启动

`task_runner.py` 读取任务 JSON：`robot` 决定用 Piper、UR5e、MuJoCo 还是 mock，
`gripper` 决定夹爪后端；入口本身不判断机器人。任务 JSON 放在应用层，用
`--config <path>` 直接指定（或用 `ROBOT_TASK_CONFIG_DIR` 指定配置目录后按任务名
查找）。下面的 `python -m robot_toolkit.task_runner` 都可以换成更短的
`run.bat`（Linux/macOS 用 `python run.py`），参数完全一致：

```powershell
# 列出可用任务
run.bat --list

# 只看解析结果，不接硬件、不动机器人
run.bat --config task.json --dry-run

# 推荐：常驻 Joy-Con 会话
# B 开始示教 / A 保存 / ZR 切夹爪 / X×2 回放并采集 / Y×2 删除上一条
run.bat --config task.json session

# 单次调试命令（不使用常驻会话）
run.bat --config task.json teach
run.bat --config task.json collect

# 不接硬件跑通整个流程
run.bat --config task.json collect --mock --yes
```

任务 JSON 不需要按操作系统分支，跨平台差异已经收在 CAN / 串口 / 相机适配器里。

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

UR5 开门这类任务同样是一份任务 JSON（Robotiq URCap + 腕部/外部 RealSense），
用通用 runner 的常驻 Joy-Con 会话启动：

```powershell
python -m robot_toolkit.task_runner --config open_the_door.json session
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

数据在 `outputs/robot_tasks/open_the_door/`。`session` 模式只负责
Joy-Con 常驻示教/采集；单次 `teach` / `collect --mock` 可用
`python -m robot_toolkit.task_runner --config open_the_door.json teach` /
`collect` 显式调用。

这份 UR 配置启用了 `replay.continuous=true`（Ruckig/限幅连续参考）和
`require_fresh_measurement=true`，参数来自真机部署验证过的控制版本。
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
