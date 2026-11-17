# Piper X ArUco visual servo (wrist D435)

Hand-following with an ArUco marker: the arm keeps the wrist camera 0.30 m from the
marker and follows it (PBVS by default, IBVS optional). The real-robot run is always
started by a person; nothing here moves the arm unless `--execute --yes` is given,
which the launcher preset does.

## Files

| File | What |
| --- | --- |
| `piper_aruco_real_probe.py` | the servo program (all options: `--help`) |
| `run_servo.py` | cross-platform launcher with the tuned preset |
| `servo.sh`, `servo_obstacles.sh` | the same for bash (the second adds the desk obstacles) |
| `test_piper_aruco_real_probe.py` | offline tests (no robot, no camera) |
| `../../configs/calibration/piper_x_wrist_d435.json` | hand-eye calibration used by default |
| `../../configs/obstacles/piper_desk_example.json` | example virtual obstacles (table + wall) |

## Set up a machine (e.g. a laptop)

Python 3.10-3.12, from the directory that contains the `robot_toolkit` checkout:

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install numpy scipy "opencv-python>=4.8" ruckig pyrealsense2 pytest
pip install -r robot_toolkit/requirements-piper.txt     # pyAgxArm + python-can (+ Windows CAN plugin)
python -m pytest -q robot_toolkit/examples/piper_aruco_servo   # offline, expect all passed
```

Not needed here: robokit/Warp/CUDA (only the GPU planner and depth map backends),
MuJoCo (simulations), the external depth services.

CAN: on Linux bring the adapter up first (`sudo ip link set can0 up type can bitrate
1000000`); on Windows the AgileX CAN module uses `python-can-agx-cando`
(see `requirements-piper.txt`). Plug the wrist D435 into a USB 3 port; check it with
`python robot_toolkit/examples/piper_aruco_servo/run_servo.py --list-cameras`.

## Run

```bash
cd robot_toolkit/examples/piper_aruco_servo
python run_servo.py --read-only           # 1. no enable/motion: marker detection + pose check
python run_servo.py                       # 2. PBVS hand following (moves the arm)
python run_servo.py --obstacles           # 3. with the virtual table/wall
python piper_aruco_real_probe.py --watch-clearance \
    --obstacles ../../configs/obstacles/piper_desk_example.json   # read-only clearance readout
```

Override anything after the preset, e.g. `--serial <your camera>`, `--lead .04`,
`--control ibvs`. Ctrl+C stops; the arm is left enabled and holding.

Logs: `robot_toolkit/outputs/aruco_servo/servo_<time>.jsonl` (per frame: capture time
and latency, target, goal, errors, clearance, `servo_gap_max_ms`).

## Notes

- Camera timing: by default each frame's arm pose is taken at the RealSense hardware
  capture time (mid-exposure); `--camera-clock host` uses receipt time minus
  `--camera-latency` instead. Measured on a D435I: ~16.5 ms capture to host.
- Garbage collection is paused while streaming (`robot_toolkit.core.realtime`) and
  collected when the run ends.
- Safety kept on: joint limits, speed/accel/jerk caps, stale feedback, 0.1 rad
  tracking abort (IBVS), reach clamp, wrong-marker rejection, and with `--obstacles`
  the collision guard (2 cm margin, braking before the 1.5 cm band, hard stop inside).
