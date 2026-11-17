# 标定与安全模块复用

## 分层

| 层 | 模块 | 职责 |
|---|---|---|
| 数学与数据 | `calibrate/quality.py`、`identity.py` | 相机/板身份、求解、独立验证、重投影叠图；无机器人 SDK |
| 视角规划 | `calibrate/view_plan.py` | 通过 FK 回调和工作空间约束生成固定 train/verify 清单 |
| 通用安全 | `core/safety_envelope.py`、`workspace_guard.py` | 示教范围存储、启用轴约束、离散关节路径的 TCP 检查 |
| 采集编排 | `calibrate/protocol.py`、`plan_capture.py` | 新帧聚合、停稳检查、清单执行、数据与报告保存 |
| 设备适配 | `robot/*`、`camera/*` | SDK、协议、单位与坐标约定转换 |
| 任务配置 | `profiles.py`、`task_runner.py`、`configs/tasks/*` | 选择设备与参数；安全设置和标定是独立任务 |

数据约定：关节为 rad，TCP 为基坐标系下 `[x,y,z,rx,ry,rz]`，
平移 m、旋转向量 rad；图像点为 px，深度观测为 mm。
板尺寸和相机身份只通过 `calibration_identity()` 组装。

## 接入其他机械臂

实现 `RobotBackend` 的状态、运动与停止接口；将 SDK 差异留在适配器。
标定编排不解包 `SafeRobot.inner`，也不查找设备私有 `enable()`。

- `connect_read_only()`：明确保证不使能、不切运动模式的连接。默认抛出
  `NotSupported`，不会用普通 `connect()` 冒充只读；当前 Piper 已适配。
- `prepare_motion()`：完成预检后准备控制器，不发送目标姿态。
  需显式使能的设备在此实现；默认只检查已连接，返回成功。
- `forward_kinematics(q)`：返回相同 TCP 定义的六维姿态。缺失则拒绝空间规划，
  不应退化为无保护运动。当前 UR RTDE 尚未提供此适配。
- 通过 `SafeRobot` 和 profile 中的 `limits.workspace` 接入通用保护。
  相机遵循 `Camera` 的图像、时间戳、内参、尺寸和序列号接口。

现有 `piper_*` 配置是本机设备实例，不是通用默认配置；复用时复制并修改
机器人、相机身份、保存姿态及安全包络路径。不要复用另一台设备的安全区域。
固定清单需要同一安装、板位置及 FK/TCP 模型；更换后重新生成。
本地生成的 `outputs/calibration_plans/wrist_guided_v1/plan.json` 仍由当前
自动配置引用，未打包进代码；缺失时应生成新清单并显式传 `--pose-plan`，
不要为了运行而关闭安全保护。

## 采集与验收

保留本次协议：旧版 7×10 ChArUco、26 mm 方格、19.5 mm 标记、
DICT_5X5_100；27 个训练位姿、8 个独立验证位姿，每位姿 6 帧，停稳 3 秒。
固定清单失败即停止，不把验证位姿补进训练集。原始图像、观测、清单、
执行情况、解算参数和叠图按轮次保存；不自动应用新外参。
训练残差不能替代独立验证；深度检查与像素检查分开报告。

安全包络仅约束采样得到的 TCP 范围，不是完整机械臂/夹具碰撞检测。
当前本机只启用 Y 两侧和 Z 下界；FK 离散检查不能证明连续轨迹无碰撞。
`tcp_margin_m` 只作用于平移，不能加到旋转分量上。

## 离线测试

在包父目录可导入的位置运行，或设置 `PYTHONPATH=..`：

```bash
python -m unittest robot_toolkit.tests.test_calibration_contracts \
  robot_toolkit.tests.test_calibration_quality \
  robot_toolkit.tests.test_view_plan robot_toolkit.tests.test_workspace_guard
python -m pytest tests
python -m robot_toolkit.examples.calibration_regression /path/to/ur5.zip \
  --out outputs/new_regression
```

这些测试与归档回归不连接硬件。归档回归只读 JSON 数据，不执行归档里的代码。
已有采样、默认/关闭姿态和标定输出均属于实验数据，清理源码不删除或覆盖它们。
