"""Config-driven task modes, split out of the former ``task_runner`` module.

Each mode owns one module; ``cli`` parses the arguments and dispatches.
"""

from .build import (_build_cameras, _build_replayer, _build_runtime, _confirm,
                    _load_trajectory_checked)
from .calibrate import run_calibrate
from .cli import build_parser, execute, main, print_plan
from .collect import run_collect
from .config import (DEFAULT_ROOT, MODES, ROOT, config_dir, config_dirs,
                     discover_configs, load_task_config, print_configs,
                     resolve_config_path)
from .export import run_export
from .hand_demo import _JoyConKeys, run_hand_demo
from .master_slave import run_master_slave
from .options import (CalibrateOptions, CollectOptions, HandDemoOptions,
                      MasterSlaveOptions, ReplayOptions, SafetyOptions,
                      SessionOptions, TaskConfig, TeachOptions)
from .paths import dataset_root, lerobot_root, task_dir, trajectory_root
from .replay import run_replay
from .safety import run_safety
from .session import run_session
from .teach import run_teach
