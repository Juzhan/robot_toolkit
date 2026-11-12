# Piper ArUco visual servo (snapshot)

Snapshot of the real-robot ArUco PBVS/IBVS probe that is run from the workspace
root (`/home/jz/ws`), where `robot_toolkit` is importable:

    cd /home/jz/ws && ./servo.sh              # PBVS hand following
    cd /home/jz/ws && ./servo_obstacles.sh    # same, with virtual obstacles
    python piper_aruco_real_probe.py --watch-clearance \
        --obstacles robot_toolkit/configs/obstacles/piper_desk_example.json \
        --calibration outputs/robot_tasks/piper_calibrate_auto/calibration.json  # read-only

The working copies live in the workspace root; this directory records the
version that matches this commit. Tests: `python -m pytest test_piper_aruco_real_probe.py`
from the workspace root.
