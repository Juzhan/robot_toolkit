# 速度限制修改同步说明

本文供另一端按现有文件自行同步修改，不提供新压缩包，也不要求整包覆盖。
以下路径以包含 `try.py` 和 `robot_toolkit/` 的目录为根目录。

## 一、对齐数值

| 项目 | 目标值 |
| --- | --- |
| 安全速度阈值 `max_velocity` | `6` rad/s |
| 执行速度 `MOVE_SPEED` | `1` |
| 执行加速度 `MOVE_ACCEL` | `1` |
| 模式 `USE_NATIVE` | `False` |
| Piper 控制器 `speed_percent` | `100` |
| 当前测试机器 `has_gripper` | `False` |

`6` 是本次要求对齐的软件阈值，不是厂家认证的最大速度。
执行速度和加速度是当前规划器的参数，不保证实测峰值恰好等于对应数值。

## 二、相对之前精简包，需要同步的改动

以下文件已与 `robot_toolkit_clean_20261023_192116.tar.gz` 比较。
若对方使用该包，只需按本节修改，不必重新同步整套运动实现。

### 1. `robot_toolkit/core/safety.py`

位置：`JointLimits` 类。

- 把 `max_velocity` 默认值从 `1.5` 改为 `6`。
- `symmetric()` 的 `max_velocity` 参数默认设为 `None`。
- `symmetric()` 未指定速度时，构造 `cls(lower, upper)`，继承类的默认值；
  指定速度时才构造 `cls(lower, upper, max_velocity)`。
- 保留现有 `SafetyGuard.evaluate()` 限幅逻辑，不取消安全保护。

目的是只保留一个默认速度入口，避免多个默认常量互相覆盖。

### 2. `robot_toolkit/profiles.py`

位置：`build_robot()` 中的 `if profile.safety:` 分支。

将原先的：

```python
velocity = profile.limits.get('max_velocity', 1.5)
guard = SafetyGuard(JointLimits(lower, upper, velocity),
                    max_step_rad=float(profile.limits.get('max_step_rad', 0.8)))
```

改为：

```python
velocity = profile.limits.get('max_velocity')
joint_limits = (JointLimits(lower, upper) if velocity is None
                else JointLimits(lower, upper, velocity))
guard = SafetyGuard(joint_limits,
                    max_step_rad=float(profile.limits.get('max_step_rad', 0.8)))
```

外层有配置时使用配置；没配置时继承 `JointLimits` 默认值，不再写死 `1.5`。

### 3. `try.py`

参数和机器人创建部分按以下内容对齐，后续逐行运动调用保持原结构：

```python
USE_NATIVE = False
MOVE_SPEED = 1
MOVE_ACCEL = 1
SAFETY_MAX_VELOCITY = 6

robot = build_robot({
    'kind': 'piper_x',
    'options': {'has_gripper': False},
    'limits': {'max_velocity': SAFETY_MAX_VELOCITY},
})
```

各次调用继续传入 `speed=MOVE_SPEED, accel=MOVE_ACCEL`，不删除这两个参数。
安全阈值现在暴露在应用外层，日常调整只改 `SAFETY_MAX_VELOCITY`，无需改库源码。
配置优先级：外层 `limits.max_velocity` > `JointLimits.max_velocity` 默认值。

当前脚本保留用户已注释的 `move_p`；最后一次实际调用仍是 `move_j(TARGET, ...)`。
无夹爪选项仅对应当前测试机器，有夹爪的另一端需按自身硬件配置。

### 4. `robot_toolkit/tests/test_robot.py`

- `test_default_safety_velocity_and_piper_speed`：不再断言默认值固定为 `1.5`；
  从 `JointLimits([-1], [1]).max_velocity[0]` 取得默认值，验证 profile 和
  `JointLimits.symmetric()` 与其一致。
- 步长断言改为默认速度乘以 `0.01`，不再写死 `0.015`。
- 显式覆盖测试保留 `max_velocity=0.4`，并补充 `symmetric()` 显式覆盖检查。

### 5. `robot_toolkit/PIPER.md`

位置：原生速度设置示例附近的安全阈值说明。
将固定的 `1.5 rad/s`、`0.015 rad` 默认描述改为引用
`JointLimits.max_velocity` 及外层 `limits.max_velocity` 配置，说明默认继承关系。

## 三、旧包已包含的速度相关修改

以下文件与上述精简包一致，不是本次新增修改。
如果对方版本更早，应检查这些实现是否已有：

| 文件位置 | 已有改动 |
| --- | --- |
| `robot_toolkit/robot/base.py` | 通用运动不再被错误禁用；插值结束仍经 `servo_j` 跟踪末点；插值和跟踪共用一次超时，超时停止并返回 `False`。 |
| `robot_toolkit/robot/safety.py` | 按每次调用的 `use_native` 选择路径；通用运动仍经过 `SafeRobot.servo_j()` 的安全限幅。 |
| `robot_toolkit/robot/agx_piper.py` | `AgxPiperConfig.speed_percent` 默认 `100`，不再默认 `50`；支持 `has_gripper=False`；原生 `speed` 比例换算与通用 SDK 发点路径。 |

## 四、没有改动的行为

- 通用规划仍使用余弦插值，时长取速度项和加速度项的较大值。
  某组数值下提高速度未改变时长，不代表速度参数整体无效。
- `max_velocity=6`、100 Hz 时，安全层限制目标领先实测位置最多 `0.06 rad`，
  同时受 `max_step_rad` 限制；不是实测速度的完整闭环保证。
- 默认 `max_step_rad=0.8`、Piper `command_hz=100`，本次不改。
- 当前安全层没有有效的加速度保护，`MOVE_ACCEL` 是规划参数。
- 原生控制器倍率不能直接等同于 rad/s；本次按 `USE_NATIVE=False` 对齐。
- 不同步采集图像、中间数据，不修改相机标定和机械臂目标点。

## 五、对方验证

先核对 `1/1/6` 的配置及 profile 传参，再运行离线测试：

```bash
cd robot_toolkit
python -m pytest tests/test_robot.py tests/test_agx_piper.py tests/test_workspace_guard.py -q
```

不要为了验证说明直接运行真机脚本。真机运动需要另行确认安装、负载和运动空间。