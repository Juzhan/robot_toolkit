# 遥操作控制说明（UR5e / Piper X）

## 1. 哪些层是共用的

| 层 | UR5e | Piper X | 是否共用 |
| --- | --- | --- | --- |
| 机器人契约 | `UrRtdeRobot` | `AgxPiperRobot` | `RobotBackend` / `RobotState` |
| 安全层 | `SafeRobot` | `SafeRobot` | 是 |
| 夹爪契约 | `RobotiqGripper` | `AgxGripper` | `GripperBackend` |
| 输入设备 | Joy-Con | Joy-Con | `JoyConSource` / `TeleopCommand` |
| 笛卡尔映射 | `CartesianTeleop` | `CartesianTeleop` | 是 |
| 控制循环 | `TeleopRunner` + `servo_l` | `TeleopRunner` + `move_p` | 是 |
| 示教/回放 | `TeachRecorder` / `Replayer` | 同左 | 是 |
| 采集/导出 | `EpisodeRecorder` / `DatasetSchema` | 同左 | 是（相机可选，任务配置指定） |

后端差异只放在两个能力里：

* `capabilities.servo_l`：UR RTDE 支持 125 Hz 连续笛卡尔伺服；
* `capabilities.move_p`：Piper 由控制器做 IK，适合 20 Hz 点到点流。

`teleop/runner.py` 根据能力选择，不会判断机器人名字。

## 2. 参考 `joycon-robotics` 后补的控制点

早期参考实现 `joycon-robotics/joyconrobotics.py` 里有几件事做得比早期
`robot_toolkit.teleop` 完整：

1. **陀螺仪校准与偏航重置**
   - `reset_joycon()` 要求水平放置 2 秒，校准后 `reset_yaw()`。
   - 现在 `JoyConSource.calibrate()` / `reset_yaw()` 可直接调用；建议每次会话
     开始前执行。

2. **使能/暂停语义**
   - 原 `EnableButtons.update(y, paused)` 返回的是“当前 paused 状态”，
     应用层再取反。
   - 旧 `EnableLatch` 在 `CartesianTeleop(enabled=False)` 下第一次按 Y 可能
     发送 `enable=False`。
   - 现在 `EnableLatch` 保存自己的 enabled 状态：按 Y 真正切换；陀螺仪
     校准中强制禁用。

3. **输入新鲜度**
   - 原 `joycon_hub` 用报告时间/读取线程存活检测断线并重连。
   - 现在 `JoyConSource.fresh()` 优先使用 `joycon.report_received_at`；
     `TeleopRunner` 在超过 `stale_s` 没有新命令时调用 `stop_motion()`。
   - 配对/重连仍留在应用层（见 `MIGRATION.md`），避免把蓝牙策略塞进通用库。

4. **夹爪按钮**
   - 原 Joy-Con 默认用 ZR 切换夹爪，也支持 `change_down_to_gripper`。
   - `JoyConSource` 现在支持 `gripper_button='b'|'zr'|'x'|'y'|'r'`，不再硬编码 B。
   - 使用 `JoyconRobotics` 时建议用 B：它内部已经把 ZR 绑定到自己的
     `gripper_state`，上层再用 ZR 会双重切换。

5. **位姿契约**
   - 原 `JoyconRobotics.get_control()` 返回虚拟工作空间的绝对 Euler 位姿。
   - 工具库统一使用旋转向量；`AgxPiperRobot.state().tcp` 和
     `move_p/move_l` 已在后端边界完成 RPY ↔ rotvec 转换。
   - SDK 原始 RPY 保留在 `state().extra['tcp_rpy']`。

## 3. 还没有完全对齐的控制点

这些不是接口问题，而是安全/手感参数，建议在具体任务里明确：

* **工作空间**：`CartesianTeleop` 只限制位置，不限制姿态。Piper 的示例
  使用 `(0.12,-0.35,0.05) .. (0.45,0.35,0.35)`，实际夹爪长度/桌面高度
  必须重新确认。
* **加速度**：`CartesianMappingConfig` 现在支持
  `max_linear_acceleration` / `max_angular_acceleration`，示例使用
  `0.6 m/s²` / `1.0 rad/s²`。`max_position_step` 仍是最后的单步保护；
  Piper 走 `move_p` 时还要用 `tracking_translation_m` 看实际跟踪误差。
* **姿态低通**：原工程对 roll/pitch 做了低通滤波；当前主要靠
  `max_rotation_step` 和加速度限制，若手抖明显，应用层可以再给 Joy-Con
  姿态加一阶低通。
* **绝对位置模式**：原 `joycon-robotics` 把虚拟位置按毫米累加，适合
  小范围精细操作；当前 `JoyConSource` 用摇杆速度，擅长快速大范围移动。
  两种模式可以并存，但不要在一个任务里混用参考系。
* **命令频率**：Piper 的 `move_p` 由控制器重规划，20 Hz 左右比 50 Hz
  更稳；UR 的 `servoL` 才适合 50–125 Hz。`TeleopRunnerConfig.move_p_hz`
  就是为此存在。

## 4. 运行示例

```powershell
# Piper X，真机；急停放手边
python -m robot_toolkit.examples.joycon_teleop --robot piper_x --yes

# UR5e
python -m robot_toolkit.examples.joycon_teleop --robot ur_rtde --ip 192.168.1.10 --yes

# 不接硬件，检查 runner/映射
python -m robot_toolkit.examples.joycon_teleop --robot mock --duration 2

# 同样的遥操循环 + LeRobot 采集
python -m robot_toolkit.examples.piper_collect --source teleop --yes
python -m robot_toolkit.examples.piper_collect --source trajectory --name piper_02

# 按任务 JSON 启动：示教 -> 回放 -> 采集（可带相机）
python -m robot_toolkit.task_runner --config task.json
```

所有入口共用同一个装配层：

```text
build_teleop_session()
  -> TeleopSession
       -> RobotBackend + GripperBackend + JoyConSource
       -> CartesianTeleop
       -> TeleopRunner (唯一的控制循环)
```

`joycon_teleop` 只负责命令行参数和打印；`piper_collect` 只额外挂一个
`EpisodeCollector` 采样回调，不再复制机器人构建、映射或控制循环。

示教回放和 Joy-Con 遥操作是两条独立路径：

* `piper_teach`：键盘空格 + 拖动示教，采关节轨迹；
* `joycon_teleop`：Joy-Con 连续笛卡尔遥操作，不产生示教轨迹；
* `piper_collect --source trajectory`：把 `piper_teach` 轨迹转成数据集；
* `task_runner`：读取任务 JSON，`robot` / `gripper` 决定后端；
  按 `mode` 组织示教、回放和采集，回放时可按 `collect.fps` 同步记录相机帧。
