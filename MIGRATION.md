# 从现有工程迁移到 robot_toolkit

本文件给出“哪个旧模块对应哪个新模块”，以及建议的分批迁移顺序。
现有代码可以继续原样运行；迁移是渐进的，不需要一次替换。

## 1. 模块对照

| 现有文件 | 新位置 | 说明 |
| --- | --- | --- |
| `ur5e_real_collect/config.py` | `config.py`, `profiles.py` | 通用 JSON + dataclass 配置；机器人/夹爪/相机用 profile 构建 |
| `ur5e_real_collect/robot.py` | `robot/ur_rtde.py` | 同一套 RTDE 逻辑，多一个统一接口；`mock=True` 可直接离线跑 |
| `ur5e_real_collect/robot.py`（旧代码直接复用） | `adapters/legacy_ur.py` | 不改旧驱动，包一层即可接入新流程 |
| `ur5e_real_collect/gripper.py` | `gripper/robotiq.py` | 宽度统一为米；mock / URScript / URCap / Modbus 后端保留 |
| `ur5e_real_collect/cameras.py` | `camera/*` | RealSense / OpenCV / Mock + `CameraHub` 同步与过期检测 |
| `ur5e_real_collect/trajectory.py` | `teach/trajectory.py` | 轨迹格式、平滑、重采样、停顿检测、夹爪事件完全对齐 |
| `ur5e_real_collect/teach.py` | `teach/recorder.py` | 采样与标记；键盘/语音等交互留在应用层 |
| `ur5e_real_collect/replay.py` | `teach/replay.py` | 只依赖 `RobotBackend`，仿真/真机共用 |
| `ur5e_real_collect/recording.py` | `collect/recorder.py` | 通用异步 episode 录制，不再硬编码路径和环境变量 |
| `ur5e_real_collect/export_dataset.py` | `collect/export.py` | feature 由 `DatasetSchema` 生成；深度 side-car 保留 |
| `ur5e_real_collect/camera_calib.py` | `calibrate/*` | Zhang 内参 + Park-Martin 手眼 + JSON 存储；算法改为纯 NumPy |
| `ur5e_real_collect/joycon_buttons.py` | `teleop/joycon.py` | `stick_axes` / `GripperToggle` / `EnableLatch` |
| `ur5e_real_collect/joycon_hub.py` | 应用层 | 手柄连接/重连策略属于具体 UI 会话 |
| `ur5e_real_collect/session.py`, `cli.py`、`eval_*`、`speech.py` | 应用层 | 业务流程、语音、评估协议不应进入通用库 |
| `ur5e_joycon_sim/control.py` | `robot/mujoco.py` + `teleop/mapping.py` | 通用 MuJoCo 后端 + 设备无关笛卡尔映射 |
| `ur5e_joycon_sim/model.py`, `piper_phone_sim/model.py` | 保留在应用 | 机器人描述/场景是具体机器人的资产 |
| `ur5e_joycon_sim/recording.py`, `export_dataset.py` | `collect/*` | 两个仿真工程几乎逐行重复的代码 |
| `ur5e_joycon_sim/runtime.py` | 应用层 | 证书/网卡等 Web 运行时工具 |
| `ur5e_joycon_sim/server.py`, `index.html`, `phone.html` | 应用层 | 前端与 HTTP 协议不属于机器人抽象 |
| `ur5e_joycon_sim/joycon_upstream.py` | `teleop/joycon.py` + 应用层 | 设备读数和姿态解算可复用，HTTP 上报留在应用 |

## 2. 建议迁移顺序

### 第 1 步：先换“数据侧”（风险最低）

1. `dataset schema`：用 `collect.DatasetSchema` 描述现有的 7 维 + 三相机；
2. `recorder`：用 `EpisodeRecorder` 替换两个仿真工程里重复的 `Recorder`；
3. `export`：用 `export_lerobot` 替换手写导出，保留旧路径参数；
4. 对比新旧导出：episode 数、帧数、feature 名称、深度目录必须一致。

这一步不碰机器人控制，可以随时回退。

### 第 2 步：换“机器人接口”

1. 真机：`LegacyUrRobot(existing_URRobot(...))`，先只用 `state()`；
2. 用 `TeachRecorder` 录一条轨迹，和旧 `teach.py` 录的轨迹对比 q/tcp/grip；
3. 用 `Replayer` 以 0.1 速空跑（不夹爪、不录像），确认轨迹一致；
4. 再切换 `UrRtdeRobot`（自带 freedrive / servo / 安全封装）。

### 第 3 步：换“仿真控制”

1. 用 `MujocoRobot` 加载现有模型（`joint_names` + `tcp_site`）；
2. 用 `CartesianTeleop` 替换 `control.py` 里的笛卡尔映射；
3. 保留 `model.py`（机器人描述）和前端不动；
4. 用 `collect.EpisodeRecorder` 替换录制/导出。

### 第 4 步：换“标定”

1. 用 `CalibrationStore.import_legacy()` 读取现有 `config.json`；
2. 新标定结果写入同一 JSON，再 `export_legacy()` 回写旧配置；
3. 对比同一场景的重投影误差和外参，确认没有坐标系数值漂移。

## 3. 兼容性检查清单

- [ ] 状态/动作维度仍是 7，顺序与 `RobotSpec.ur5e()` 一致；
- [ ] 夹爪单位仍是米，0 = 闭合；
- [ ] 深度仍是 `uint16 mm`，0 无效；
- [ ] 外参仍是 `T_world_camera`（光学系）；
- [ ] 回放速度上限、起点接近行为、夹爪冻结时间与旧实现一致；
- [ ] 数据集目录、`capture_contract.json`、深度 side-car 路径不变；
- [ ] 旧 `config.json` 仍可加载（新字段有默认值）。

## 4. 有意保留在应用层的东西

以下内容**不**进入通用库，因为它们依赖具体产品/UI/协议：

* HTTP/HTTPS 服务、页面、手机姿态映射；
* 语音播报、手柄配对与重连策略；
* 策略服务器协议（`/act`、action chunk、评分规则）；
* 具体任务流程、目录命名、评估统计；
* 具体机器人的 MJCF/URDF 与标定板摆放。

这些应用只需要调用库的接口，不应反向污染 `robot_toolkit`。
