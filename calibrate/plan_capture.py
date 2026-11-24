"""Execute a fixed view plan through the shared unattended executor."""
from .auto import run_auto


def run_plan(robot, camera, session, cfg, path):
    """Compatibility entry: ``cfg.pose_plan_path`` selects the saved plan."""
    need = cfg.min_samples + cfg.verify_samples
    return run_auto(robot, camera, session, cfg, path, need)
