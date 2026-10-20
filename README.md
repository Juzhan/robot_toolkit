# robot_toolkit — 机器人无关的遥操 / 示教 / 采集 / 标定工具库

这个包把仓库里分散在两个仿真工程和 `ur5e_real_collect` 里的通用能力抽出来，
做成一个**不针对具体机器人**的基础库。目标是：

> 一套契约，多种后端。仿真、UR5、Piper 以及以后新增的机械臂，共用同一套
> 遥操、示教、回放、采集、标定和数据集代码。

本包是新增目录，可以独立使用，也可以作为后续迁移的目标。旧代码通过
`adapters/legacy_ur.py` 就能接入新的示教/采集流程；`ur5e_real_collect` 只同步了
Mac 真机验证过的通讯/安全和语音修复，不改变原有 CLI 结构。

---

## 1. 设计原则

1. **一个契约**：`observation.state` / `action` 永远是
   `[关节1..关节N (rad), 夹爪开度 (m)]`，动作用绝对关节角，不用增量。
   这与现有工程已经落盘的 7 维数据集保持一致。
2. **接口只依赖能力**：上层只面向 `RobotBackend`、`GripperBackend`、`Camera`，
   从不判断“这是仿真还是真机”。
3. **安全是装饰器，不是后端特性**：`SafeRobot` 包住任何后端，统一做关节限位、
   单步限幅、速度限制。仿真和真机走完全相同的检查。
4. **依赖按需加载**：核心只依赖 NumPy；`mujoco`、`ur_rtde`、`pyrealsense2`、
   `opencv`、`lerobot` 都在真正用到时才 import。没有装 SDK 也能跑自检和离线演示。
5. **数据先落盘，再导出**：采集线程只写压缩帧，导出在后台线程完成；
   网络、编码、LeRobot 版本问题都不会破坏已经录到的原始数据。
6. **坐标系/单位写死在契约里**：位姿是 `xyz + rotvec(rad)`，深度是
   `uint16 mm, 0 无效`，外参是 `T_world_camera`（光学系 x 右 / y 下 / z 前）。

---

## 2. 目录结构

```text
projects/robot_toolkit/
├── __init__.py               # 公共导出 + 版本
├── cli.py                    # 统一命令行：selftest / demo / calib-* / doctor
├── task_runner.py            # 按任务 JSON（含 robot/gripper profile）启动示教/回放/采集
├── config.py                 # JSON + dataclass 递归配置
├── profiles.py               # 由配置构建 mock / ur_rtde / agx_piper / mujoco 后端
├── control/
│   ├── chunks.py             # 绝对动作块 + 本地 SequenceSource
│   └── continuous.py         # 连续参考、共享动作时钟、跟踪/夹爪门控
├── core/
│   ├── conventions.py        # RobotSpec / GripperSpec，7 维契约
│   ├── mathutils.py          # 位姿/旋转/四元数/齐次变换（纯 NumPy）│   ├── continuous.py         # Ruckig 可选 + 限幅回退的连续关节参考│   ├── safety.py             # JointLimits / SafetyGuard / WorkspaceBox / Watchdog
│   └── rate.py               # 定频循环 + 超时统计
├── robot/
│   ├── base.py               # RobotBackend / RobotState / Capabilities
│   ├── mock.py               # 一阶关节模型（可带 FK/IK），测试与演示用
│   ├── mujoco.py             # 通用 MuJoCo 机械臂后端 + DLS IK
│   ├── ur_rtde.py            # UR3/UR5/UR10（CB3 & e-Series）
│   ├── agx_piper.py          # AgileX Piper / Piper X（pyAgxArm）
│   └── safety.py             # SafeRobot 装饰器
├── gripper/
│   ├── base.py               # GripperBackend（统一用米）
│   ├── mock.py               # 有限速度的 MockGripper
│   ├── callback.py           # 任意读写函数 -> 夹爪接口
│   ├── robotiq.py            # Robotiq 2F：mock / URScript / URCap / Modbus
│   └── agx.py                # AgileX AgxGripper（通过 Piper 后端）
├── camera/
│   ├── base.py               # Camera / CameraFrame / ThreadedCamera
│   ├── mock.py               # 合成彩色 + 深度
│   ├── opencv.py             # USB 摄像头
│   ├── realsense.py          # RealSense 彩色 + 对齐深度
│   └── hub.py                # 多相机同步快照 + 过期检测
├── teleop/
│   ├── base.py               # TeleopCommand / TeleopBridge / ScriptedSource
│   ├── mapping.py            # 设备无关的笛卡尔映射 + 工作空间限位
│   ├── joycon.py             # Joy-Con 适配器（不直接连机器人）
│   ├── joycon_buttons.py     # Joy-Con 按键 hub（session / 示教）
│   └── runner.py             # servo_l / move_p 两种后端的通用控制循环
├── teach/
│   ├── trajectory.py         # 轨迹模型、平滑、重采样、停顿检测、夹爪事件
│   ├── recorder.py           # 拖动示教采样
│   ├── console.py            # 通用键盘拖动示教（任意后端）
│   ├── replay.py             # 变速回放、事件、急停/中止
│   └── editor.py             # 补标记、裁剪、变速、摘要
├── collect/
│   ├── schema.py             # 数据集契约 + LeRobot features
│   ├── recorder.py           # 异步 episode 录制（帧落盘 + 后台导出）
│   ├── observation.py        # CameraFrame -> images/depth
│   └── export.py             # LeRobot 导出 + 无依赖 manifest 导出
├── calibrate/
│   ├── intrinsics.py         # 纯 NumPy Zhang 标定（可选 OpenCV 精修）
│   ├── handeye.py            # 纯 NumPy Park-Martin AX=XB
│   ├── session.py            # 棋盘格/机器人位姿采集 + 求解 + 保存
│   └── store.py              # JSON 标定存储、旧配置导入/导出
├── adapters/
│   └── legacy_ur.py          # 包装现有 URRobot，作为迁移桥
├── examples/
│   ├── mock_pipeline.py      # 完整离线链路示例
│   ├── piper_control.py      # Piper / Piper X 真机读写示例
│   ├── piper_teach.py        # Piper 键盘示教 + 轨迹回放（无需相机）
│   ├── piper_collect.py      # Piper 遥控/轨迹 -> LeRobot 数据集
│   └── joycon_teleop.py      # 同一 Joy-Con 栈驱动 UR5e / Piper X
├── PIPER.md                  # Piper SDK 安装、CAN、控制、任务配置与安全
├── TELEOP.md                 # UR/Piper 共用遥操作分层与控制注意事项
├── requirements-control.txt  # 可选：与 Mac 部署一致的 Ruckig 参考
└── tests/                    # 纯 NumPy 测试
```

---

## 3. 跨平台边界

上层（`RobotBackend` / `TeleopRunner` / `TeachRecorder` / `EpisodeCollector` /
`CalibrationSession`）不按操作系统分支；但**底层跨平台不等于上层自动跨平台**，
平台差异必须收在适配器或配置里：

| 边界 | Windows | Linux | macOS |
| --- | --- | --- | --- |
| Piper CAN | `agx_cando` / `0` | `socketcan` / `can0` | `slcan` / `/dev/ttyACM0` |
| Piper SDK | `pyAgxArm` + `python-can-agx-cando` | `pyAgxArm` + SocketCAN | `pyAgxArm` + slcan |
| Robotiq 串口 | `COM5` 默认 | `/dev/ttyUSB0` 默认 | `/dev/ttyUSB0` 默认 |
| 编码器/相机 | OpenCV 使用 DSHOW | V4L2 | AVFoundation |
| Joy-Con | HID/蓝牙驱动 | HID/蓝牙驱动 | HID/蓝牙驱动 |
| 键盘示教 | `msvcrt` | `termios` | `termios` |
| LeRobot 视频 | FFmpeg/TorchCodec 需正确安装 | 同左 | 同左 |

`python -m projects.robot_toolkit.cli doctor` 会打印当前平台、可选依赖和默认串口。
上层的坐标系、单位、状态/动作维度和安全逻辑保持一致；跨平台验证应覆盖
`mock`、真实机器人、相机、手柄、键盘和 LeRobot 导出，而不是只看 Python 能否导入。

## 4. 统一契约

### 4.1 状态与动作

```python
RobotSpec.ur5e().state_names
# ['shoulder_pan.rad', 'shoulder_lift.rad', 'elbow.rad',
#  'wrist_1.rad', 'wrist_2.rad', 'wrist_3.rad', 'robotiq_2f85.width_m']
```

* `observation.state`：`t` 时刻的关节角 + 夹爪开度。
* `action`：同一控制周期算出的**绝对**目标，在下一次观测前执行。
* 夹爪统一用**米**：`0.0` 闭合，`max_width` 全开。Robotiq 的 0–255 寄存器
  只在 `gripper/robotiq.py` 内部转换。
* 角度统一用弧度，位姿旋转统一用旋转向量（Rodrigues），与 UR
  `getActualTCPPose` 一致。

### 4.2 相机与深度

| 项目 | 约定 |
| --- | --- |
| 彩色 | `HxWx3 uint8`，RGB 顺序 |
| 深度 | `HxW uint16`，毫米，`0` 无效 |
| 内参 | 3×3 `K` |
| 外参 | `T_world_camera`，相机光学系 x 右 / y 下 / z 前 |
| 时间 | 主机 `perf_counter()` 秒；多相机可用同一时钟比较新旧 |

### 4.3 数据集

`collect/schema.py` 是唯一事实来源，导出的 `capture_contract.json` 会写明：

```json
{
  "state_action": {"names": ["shoulder_pan.rad", "..."], "dim": 7,
                   "action": "absolute joint targets plus absolute gripper opening"},
  "depth": "uint16 millimeters, 0 invalid",
  "extrinsics": "T_world_camera, optical: x right, y down, z forward"
}
```

---

## 5. 快速开始

> 独立发布版只包含 `projects/robot_toolkit`；`run.bat` / `run` / `run.py` 和
> `configs/tasks` 属于宿主工程里的启动器示例，不在本仓库中。库本身可以直接用
> 下面的 `python -m projects.robot_toolkit...` 命令。

```powershell
# 不接硬件，验证核心数学、接口和录制链路
python -m projects.robot_toolkit.cli selftest

# 查看统一契约
python -m projects.robot_toolkit.cli info

# 查看已安装的可选 SDK
python -m projects.robot_toolkit.cli doctor

# Piper X 只读自检（真实 CAN；加 --mock 可不接硬件）
python -m projects.robot_toolkit.cli piper-check --model piper_x

# Piper X 控制示例（需要 --yes；安装/CAN/固件详见 PIPER.md）
python -m projects.robot_toolkit.examples.piper_control --enable

# Piper X 键盘拖动示教（无相机；空格切换夹爪，s 保存）
python -m projects.robot_toolkit.examples.piper_teach --name piper_01 --speed-percent 20

# Joy-Con 遥操；同一脚本换 --robot ur_rtde 即可驱动 UR5e
python -m projects.robot_toolkit.examples.joycon_teleop --robot piper_x --yes

# 离线全链路：脚本遥操 -> 示教录制 -> 回放 -> 采集 -> manifest 导出
python -m projects.robot_toolkit.cli demo --out outputs\robot_toolkit_demo

# 单元测试（纯 NumPy；LeRobot 相关测试按环境自动跳过）
python -m pytest projects\robot_toolkit\tests -q
```

跨平台启动时，机器人由 `configs/tasks/<任务名>.json` 的 `robot` / `gripper`
profile 决定：

```powershell
# Windows
run.bat --list
run.bat stack_the_cube --dry-run
```

```bash
# macOS / Linux
chmod +x run
./run --list
./run stack_the_cube --dry-run

# 任意平台直接走 Python
python3 run.py stack_the_cube --dry-run
```

`run.bat` / `run` / `run.py` 只是启动器；真正的任务、机器人、夹爪、相机和
输出目录都在 JSON 配置里，不会在脚本里按操作系统或机器人型号分叉。

当前自带的 `stack_the_cube`（Piper X）和 `open_the_door`（UR5e）的 `mode`
都是 `session`：启动后保持机器人、夹爪和相机连接，用 Joy-Con 反复示教/采集。

| 按键 | 作用 |
| --- | --- |
| `B` | 开始一次拖动示教 |
| `A` | 结束并保存轨迹 |
| `ZR` | 夹爪闭合/张开 |
| `X` ×2 | 回放上一条轨迹并录制一个 episode |
| `Y` ×2 | 删除上一条示教轨迹 |
| `R` | 空闲时缓慢回默认姿态 |
| `Home` | 打印当前状态 |

一次会话里可以连续做多条轨迹、连续采多个 episode，数据集会追加到同一个
LeRobot 数据集，不需要重启进程。`projects/ur5e_real_collect` 的
`ur5.bat session` 是独立的旧入口，不负责这套通用 `session` 流程。

`demo` 会生成：

```text
outputs/robot_toolkit_demo/
├── trajectories/trajectory.npz|json   # 示教轨迹
├── collect/<session>/dataset/          # 采集结果（episodes.jsonl + schema.json）
└── summary.json
```

---

## 6. 核心用法

### 6.1 同一个控制循环，驱动仿真或真机

```python
from projects.robot_toolkit.profiles import build_robot
from projects.robot_toolkit.robot import SafeRobot
from projects.robot_toolkit.core.safety import JointLimits, SafetyGuard

robot = build_robot('ur_rtde')          # 换成 'mock' 或 'mujoco' 即可
robot = SafeRobot(robot, SafetyGuard(JointLimits.symmetric(6, 3.14, 1.0)))

robot.connect()
state = robot.state()
print(state.q, state.tcp)
robot.move_j(home, speed=0.3)           # 任意后端都有；真机用原生 moveJ
robot.servo_j(target, dt=1 / 125)       # 回放 / 遥操的高频原语
robot.stop_motion()
robot.disconnect()
```

一个后端只要实现 `state()` 和 `servo_j()` 就能被所有上层功能使用；
`move_j`、`hold`、`wait_until_settled` 都有通用实现。

### 6.2 遥操

```python
from projects.robot_toolkit.core.mathutils import rotvec_to_matrix
from projects.robot_toolkit.teleop import (CartesianMappingConfig, CartesianTeleop,
                                           JoyConSource, TeleopRunner,
                                           TeleopRunnerConfig)

state = robot.state()
mapping = CartesianTeleop(CartesianMappingConfig(
    translation_speed=0.08,
    workspace_lower=(-0.4, -0.4, 0.05),
    workspace_upper=(0.5, 0.4, 0.4),
    smoothing=0.2,
    max_linear_acceleration=0.6,
    max_angular_acceleration=1.0,
), robot.spec.gripper)

source = JoyConSource(controller, gripper_max=robot.spec.gripper.max_width,
                      gripper_button='zr', enable_button='y')
source.calibrate()                      # 水平放置 2 秒，重置陀螺仪/偏航

runner = TeleopRunner(robot, gripper, source, mapping,
                      TeleopRunnerConfig(rate_hz=50, stale_s=0.4,
                                         move_p_hz=20,
                                         tracking_translation_m=0.03))
result = runner.run()
```

`TeleopRunner` 是两端共用的控制循环：

* UR5e 的 `capabilities.servo_l=True`，直接走 RTDE `servoL`；
* Piper X 没有 `servo_l`，走 `capabilities.move_p=True` 的 20 Hz 点到点流，
  由控制器做 IK；
* 映射、安全、输入过期停机、夹爪事件和示教/回放完全不区分机器人。

设备只产生 `TeleopCommand`（归一化 xyz / rpy / 绝对四元数 / 夹爪速率 /
使能按钮），映射、限位、安全、录制都与设备无关。`JoyConSource` 的
`enable_button` / `gripper_button` 可配置；`paused`（陀螺仪校准）会强制
禁用运动，避免校准抖动被当成指令。

### 6.3 示教与回放

```python
from projects.robot_toolkit.teach import TeachRecorder, Replayer

with TeachRecorder(robot, gripper, rate=125) as recorder:
    robot.freedrive(True)
    # ... 人拖动机械臂 ...
    recorder.mark('close', label='grasp')
    robot.freedrive(False)

trajectory = recorder.build()
recorder.save('outputs/my_task/trajectories')

result = Replayer(robot, gripper, rate=125, max_joint_speed=1.0).run(
    trajectory, speed_scale=0.3, abort_event=operator_stop)
```

* 标记：`mark / close / open / move / hold / phase`；
* 回放自动做平滑、重采样、起点缓慢接近、夹爪事件冻结保持；
* 超速直接拒绝，除非显式 `allow_fast=True`；
* `Replayer` 只依赖 `servo_j`，所以仿真和真机共用同一段回放代码。

### 6.4 连续动作流（来自 Mac 真机验证版本）

`projects/ur5e_real_collect` 的 Mac 部署版用 Ruckig 把 10 Hz 策略动作接到
125 Hz `servoJ` 上，并在动作块之间保持位置/速度/加速度连续；工具库把这部分
抽成了机器人无关的 `control` 层。

```python
from projects.robot_toolkit.control import (ContinuousConfig, SequenceSource,
                                            run_continuous_stream)

config = ContinuousConfig(
    control_hz=125, action_hz=10,
    max_joint_speed=0.8, max_joint_accel=1.2, max_joint_jerk=8.0,
)
result = run_continuous_stream(robot, gripper, source, config,
                               abort_event=operator_stop)
```

* `source.next_chunk(action_index)` 可以是 HTTP 策略客户端、录制轨迹或
  `SequenceSource`；网络请求在后台线程，不会阻塞 servo 循环。
* 共享动作时钟在机械臂跟不上时暂停推进，不会把后续动作“跳着播”。
* 夹爪动作要等参考位置和实测位置都进入门控范围才下发，避免边动边夹。
* `initPeriod()/waitPeriod()` 是 ur_rtde 的控制器专属接口，不是工具库的通用
  API。工具库统一暴露的是 `RobotBackend.init_period()/wait_period()`：UR 覆盖为
  RTDE 控制器周期；Mock / Piper / MuJoCo 没有控制器周期时返回 `None/False`，
  上层自动回退到 `time.perf_counter()` + `sleep()` 的宿主时钟。
* UR 在 Windows/Linux 使用控制器周期；macOS 保留实测稳定的主机时钟分支。
* `ContinuousReference` 默认自动使用 Ruckig；没安装时退化为限幅追踪器，
  仍保证速度/加速度/急动度不越界。需要与 Mac 部署完全一致时：
  `pip install -r projects/robot_toolkit/requirements-control.txt`。
* Robotiq URCap 可设置 `require_fresh_measurement=true`：实测宽度超过 1 秒
  未更新时直接报错，不再用最后的目标宽度冒充观测。

### 6.5 采集

```python
from projects.robot_toolkit.collect import (CameraSpec, DatasetSchema,
                                            EpisodeRecorder, export_lerobot)

# 方案 A：只有机器人，没有任何相机（状态/动作 7 维）
schema = DatasetSchema.for_robot(spec, cameras=(), fps=10,
                                 repo_id='local/my_piper_task')

# 方案 B：按实际硬件选择任意相机组合
cameras = (
    CameraSpec.from_camera(wrist_camera),       # 已打开的 Camera 对象
    CameraSpec('external_camera', 640, 480, depth=True),
)
schema = DatasetSchema.for_robot(spec, cameras=cameras, fps=10,
                                 repo_id='local/my_piper_cameras')

recorder = EpisodeRecorder('outputs/my_task/recordings', schema,
                           export=export_lerobot)   # 或 export_manifest
recorder.start('stack the cube')
recorder.capture(state_vector, action_vector, images=..., depth=...)
recorder.save(success=True)
recorder.close()
```

应用层可以直接用 `EpisodeCollector`，它会复用同一套 schema/recorder，并自动
读取相机快照：

```python
from projects.robot_toolkit.collect import EpisodeCollector

collector = EpisodeCollector(
    'outputs/my_task/dataset', schema,
    robot=robot, gripper=gripper,
    cameras=(wrist_camera, external_camera),   # 可以省略
    fps=10,
)
collector.start('stack the cube')
collector.capture_state()        # 自动抓取 robot.state + 相机帧
collector.stop(success=True)
collector.close()
```

已有的示教轨迹可以直接转成数据集：

```python
from projects.robot_toolkit.collect import collect_trajectory
collector = collect_trajectory(trajectory, schema, 'outputs/my_task/dataset',
                               task='stack the cube')
collector.close()

# 或用统一的 Piper 入口
# --source trajectory: 导出 piper_teach 的轨迹
# --source teleop:     Joy-Con 采集，支持 --cameras-json
python -m projects.robot_toolkit.examples.piper_collect --source teleop --cameras-json cameras.json --yes
```

等价的 JSON 配置可以直接交给 `DatasetSchema.from_dict()`：

```json
{
  "robot": {
    "name": "piper_x",
    "joint_names": ["j1", "j2", "j3", "j4", "j5", "j6"],
    "gripper": {"name": "piper_gripper", "max_width": 0.08, "min_width": 0.0}
  },
  "cameras": [],
  "fps": 10,
  "require_images": false,
  "repo_id": "local/my_piper_task"
}
```

* `DatasetSchema.for_robot(..., cameras=())` 表示 robot-only；
  `require_images` 会自动变成 `False`，`capture()` 不要求 `images`。
* 相机可以只选一个、两个或全部；LeRobot features 只包含实际声明的相机。
* JSON 配置中写 `"cameras": []` 明确表示无相机；不写 `cameras` 才使用默认三相机。
* 采集循环不阻塞：帧写入队列，写盘线程落 `.npz`；
* 导出线程/延迟导出二选一；失败不会破坏已保存的 episode；
* `export_lerobot` 使用 LeRobot 官方 writer，并保留深度 side-car；
  已适配 LeRobot 0.4（`vcodec=`）与 0.6（`resume()`、`use_videos=`）两代 API；
* 没装 LeRobot 时用 `export_manifest`，生成 `episodes.jsonl` + `schema.json`。

### 6.6 标定

```python
from projects.robot_toolkit.calibrate import (CalibrationStore, solve_hand_eye,
                                              estimate_intrinsics)

# 内参：视图 = {'object_points': Nx3(Z=0), 'image_points': Nx2}
intrinsics = estimate_intrinsics(views, image_size=(640, 480))

# 手眼：eye_in_hand -> T_tcp_camera；eye_to_hand -> T_base_camera
result = solve_hand_eye('eye_in_hand', base_tcp_poses, cam_target_poses)

store = CalibrationStore('outputs/my_task/calib/calibration.json')
store.set_intrinsics('wrist_camera', intrinsics.K, intrinsics.dist, (640, 480))
store.set_hand_eye('wrist_camera', 'eye_in_hand', result.transform, result.rotation_rms)
store.save()

store.apply(camera, base_tcp=robot.state().tcp)   # 写入 T_world_camera
```

两种解算都是纯 NumPy（Zhang / Park-Martin），没有 OpenCV 也能跑；
装了 OpenCV 时内参会自动精修。`CalibrationStore.import_legacy()` 可以直接
读取现有 `config.json` 的 `tcp_camera` / `base_camera`。

标定求解与存储不依赖机器人；`base_tcp` 统一是 `[x,y,z,rx,ry,rz]` 旋转向量。
Piper 的 `state().tcp` 已归一化到该约定，所以同一个 `CalibrationStore.apply()`
可以用于 UR5e 和 Piper。

通用采集会话把“机器人当前 TCP + 相机帧 + 标定板位姿”串起来：

```python
from projects.robot_toolkit.calibrate import (CalibrationSession,
                                              ChessboardDetector)

session = CalibrationSession(
    robot=robot,
    camera=wrist_camera,
    detector=ChessboardDetector(pattern=(9, 6), square_size=0.025),
    mode='eye_in_hand',
)
for pose in calibration_poses:
    robot.move_j(pose)
    session.capture()                 # 检测棋盘格 + 记录 T_base_tcp
result = session.solve_hand_eye()
session.save('outputs/calib/calibration.json', apply=True)
```

`ChessboardDetector` 是 OpenCV 棋盘格实现；ChArUco/ArUco 可以注入一个
返回 `BoardObservation` 的自定义 detector，或使用 `session.add_sample()`。
采集循环本身仍由应用控制，但机器人、相机、求解和存储均复用同一套接口。

---

## 7. 接入一台新机械臂

只需要一个文件，通常 30–80 行：

```python
from projects.robot_toolkit.robot.base import (Capabilities, RobotBackend,
                                               RobotState)
from projects.robot_toolkit.core.conventions import RobotSpec

class MyArm(RobotBackend):
    def __init__(self, sdk):
        super().__init__(RobotSpec(name='my_arm',
                                   joint_names=('j1', 'j2', 'j3', 'j4', 'j5', 'j6')),
                         Capabilities(servo_j=True, move_j=True, freedrive=False))
        self.sdk = sdk

    def connect(self):
        self.sdk.open()
        self._connected = True
        return self

    def state(self):
        q, qd, pose = self.sdk.read()
        return RobotState(t=self.sdk.timestamp(), q=q, qd=qd, tcp=pose,
                          protective_stop=self.sdk.has_fault())

    def servo_j(self, joints, dt):
        return self.sdk.write_joints(joints, dt)

    def stop_motion(self, deceleration=2.0):
        return self.sdk.quick_stop()
```

然后注册到 `profiles.build_robot()`（或直接实例化），即可使用全部遥操、
示教、回放、采集和标定能力。夹爪实现 `GripperBackend`（或直接用
`CallbackGripper` 包住 SDK 的读写函数），相机实现 `Camera.read()`。

Piper：`RobotSpec.piper_x()` 已经提供关节/夹爪命名；真机按上面的模板包一层
`piper_sdk`（CAN 读写）即可，仿真则直接复用 `MujocoRobot`，把模型里的
关节名和执行器名传进去。UR 与 MuJoCo 已经各有一个完整实现可参考。

---

## 8. 真机与仿真如何统一

| 能力 | 仿真（MuJoCo） | 真机（UR） |
| --- | --- | --- |
| 状态 | `MujocoRobot.state()`：qpos/qvel/site 位姿 | `UrRtdeRobot.state()`：RTDE `getActualQ/TCPPose` |
| 高频控制 | `servo_j` 写 `ctrl` 并步进物理 | `servo_j` 调 `servoJ` |
| 笛卡尔伺服 | `servo_l` 用 `DlsIk` 解算后 `servo_j` | `servo_l` 用控制器原生 `servoL` |
| 点位运动 | 通用插值或 `move_j` | 原生 `moveJ/moveL` |
| 拖动示教 | 可自定义（重力补偿） | `freedriveMode` / `teachMode`（CB3 回退） |
| 安全 | `SafeRobot` | 同一个 `SafeRobot` |

`MujocoRobot` 默认开启重力/科氏力补偿（`gravity_compensation=True`），
因为纯位置执行器在重力下会有稳态下垂，和真机伺服的手感不一致。
实测 UR5e 场景：`servo_l` 追踪误差约 0.15 mm，`move_j` 正常收敛。

差别只存在于后端内部；示教、回放、采集、遥操循环完全相同。现有
`ur5e_joycon_sim` 和 `ur5e_real_collect` 都可以逐步迁移到这一层
（见 `MIGRATION.md`）。

---

## 9. 与现有工程的关系

* **零侵入**：没有修改 `projects/ur5e_joycon_sim`、`projects/piper_phone_sim`、
  `projects/ur5e_real_collect` 的任何文件。
* **可直接复用**：`adapters/legacy_ur.py` 把现有 `URRobot` 包装成
  `RobotBackend`，即可用新的 `TeachRecorder` / `Replayer` / `EpisodeRecorder`。
* **数据兼容**：状态/动作命名、深度约定、`T_world_camera` 都与既有数据集一致；
  LeRobot 导出使用相同的 feature 结构，可直接进入 LeRobot / 扩散策略等训练流程。
* **迁移路径**：先替换“录制/回放/导出”，再替换“机器人/夹爪/相机驱动”，
  最后把 UI 与业务会话留在应用层。详见 [MIGRATION.md](./MIGRATION.md)。

---

## 10. 依赖

| 功能 | 必需 | 可选 |
| --- | --- | --- |
| 核心、示教、标定数学、manifest 导出 | NumPy | SciPy（本项目未使用） |
| MuJoCo 后端 | — | `mujoco` |
| UR 真机 | — | `ur_rtde` |
| RealSense | — | `pyrealsense2` |
| USB 相机 / 棋盘格检测 | — | `opencv-python` |
| LeRobot 导出 | — | `lerobot>=0.4.3` |
| Robotiq Modbus | — | `pymodbus` |

`python -m projects.robot_toolkit.cli doctor` 会列出本机实际可用的组件。

### 已验证组合

* 基础环境（NumPy 2.1）：68 个测试通过，`selftest` 7/7；
* MuJoCo 3.8.1 + 现有 UR5e Menagerie 场景：`servo_l` 误差 0.15 mm、`move_j` 收敛；
* LeRobot 0.6.1：单 episode 导出、追加第二个 episode、深度 side-car 均通过。

---

## 11. 安全

* 真机第一次回放务必 `speed_scale=0.3`，人守在急停旁；
* `Replayer` 会在超速时直接拒绝，除非显式 `allow_fast=True`；
* `SafeRobot` 同时做关节限位和单步限幅；`servo_l` 后会复核关节增量；
* `CameraHub.snapshot(strict=True)` 会拒绝过期帧，避免用旧图像采集；
* `TeachRecorder` / `EpisodeRecorder` 的失败只影响当前 episode，不会删除已保存数据。

仿真里验证过的映射、轨迹和数据集，迁移到真机前仍需完成：限位、急停、
TCP/负载、手眼外参、夹爪开度标定。
