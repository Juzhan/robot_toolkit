# requirements/

按用途拆分的依赖清单，内容和 `pyproject.toml` 的 extras 一一对应。两种装法等价，
选一种即可：

```bash
pip install -r requirements/ur5.txt          # 只装第三方依赖
pip install -e ".[ur]"                       # 同时把本包装成可导入
```

每个文件都用 `-r base.txt` 带上核心依赖，可以单独使用。

| 文件 | 对应 extra | 安装内容 | 用途 |
|---|---|---|---|
| `base.txt` | — | `numpy` | 离线自检、mock、示教回放、无相机采集 |
| `vision.txt` | `vision` | `opencv-python`、`scipy` | 标定、ChArUco、图像质量、位姿规划 |
| `control.txt` | `control` | `ruckig` | 连续动作流（`auto` / `ruckig`） |
| `ur5.txt` | `ur` | `ur_rtde` | UR3 / UR5 / UR10 真机 |
| `cameras.txt` | `cameras` | `pyrealsense2`、`opencv-python` | RealSense 深度与 USB 相机 |
| `piper.txt` | `piper` | `python-can`、`pyAgxArm`（Windows 另加插件） | Piper / Piper X |
| `sim.txt` | `sim` | `mujoco`、`Pillow`、`imageio` | MuJoCo 仿真示例 |
| `server.txt` | `server` | `fastapi`、`uvicorn` | `python -m robot_toolkit.agent.uni` |
| `dev.txt` | `dev` | `pytest` | 单元测试与 `cli selftest` |

常用组合：

```bash
# UR5e 真机 + 相机 + agent 服务 + 测试
pip install -r requirements/ur5.txt -r requirements/cameras.txt \
            -r requirements/server.txt -r requirements/dev.txt

# 仿真 + 规划演示
pip install -r requirements/sim.txt -r requirements/vision.txt
```

以下是 extras 里剩下的可选能力，用到时再装：

| extra | 安装内容 |
|---|---|
| `modbus` | `pymodbus`（Robotiq 经 Modbus） |
| `lerobot` | `lerobot`（数据集导出 / 合并 / 编辑） |
| `reconstruction` | `open3d`（点云与重建） |
| `joycon` | `hidapi`、`PyGLM`、`scipy` |
| `voice` | `comtypes`（仅 Windows 语音播报） |
| `ik` | `pyroki`（宿主 IK，需从源码装） |

装完先自查，缺什么会直接给出对应的文件或 extra：

```bash
python -m robot_toolkit.cli doctor
```
