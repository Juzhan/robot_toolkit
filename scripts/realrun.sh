#!/usr/bin/env bash
# One launcher for every backend, so no environment ever needs activating.
set -euo pipefail

WS=${WS:-/home/jz/ws}
PY=${PY:-$WS/RealSense-Stereo-Trial/.venv-linux/bin/python}
export PYTHONPATH="$WS:$WS/uni-robot-agent${PYTHONPATH:+:$PYTHONPATH}"

die() { echo "error: $*" >&2; exit 1; }

task_path() {
  local name=$1
  [[ -f "$name" ]] && { echo "$name"; return; }
  local candidate="$WS/robot_toolkit/configs/tasks/${name%.json}.json"
  [[ -f "$candidate" ]] || die "task config not found: $name"
  echo "$candidate"
}

check_deps() {
  local path=$1
  "$PY" - "$path" <<'PY'
import importlib.util, json, sys
task = json.load(open(sys.argv[1]))
needed = {'pyrealsense2': 'cameras'} if any(c.get('kind') == 'realsense'
         for c in task.get('collect', {}).get('cameras', [])) else {}
robot = (task.get('robot') or {}).get('kind')
if robot == 'agx_piper':
    needed['pyAgxArm'] = 'Piper arm'
elif robot == 'ur_rtde':
    needed['rtde_control'] = 'UR arm (pip install ur_rtde)'
missing = [f'{m} ({why})' for m, why in needed.items()
           if importlib.util.find_spec(m) is None]
if missing:
    print('missing in ' + sys.executable + ':', file=sys.stderr)
    for line in missing:
        print('  ' + line, file=sys.stderr)
    sys.exit(1)
PY
}

uni() {
  local mode=$1; shift
  local task; task=$(task_path "$1"); shift
  check_deps "$task"
  local stamp; stamp=$(date +%Y%m%d_%H%M%S)
  local out=${OUT:-/tmp/${mode}_${stamp}}
  local args=(--run-id "${mode}-${stamp}" --port "${PORT:-19360}")
  args+=(--instruction "${INSTRUCTION:-view scene}")
  [[ $mode == execute ]] && args+=(--execute)
  echo "interpreter: $PY"
  echo "task:        $task"
  echo "output:      $out"
  exec "$PY" -m robot_toolkit.agent.uni "$task" "${args[@]}" --output-dir "$out" "$@"
}

case ${1:-help} in
  nudge)
    shift
    [[ $# -ge 1 ]] || die "usage: realrun.sh nudge <task> [x] [y] [z]  # centimetres"
    task_file=$(task_path "$1"); shift
    check_deps "$task_file"
    exec "$PY" - "$task_file" "${MOCK:-0}" "${1:-0}" "${2:-0}" "${3:-3}" <<'PY'
import numpy as np
import sys
from robot_toolkit.agent.station import Station

task, mock, dx, dy, dz = sys.argv[1], sys.argv[2] == '1', *(float(v) for v in sys.argv[3:6])
delta = np.array([dx, dy, dz]) / 100.0


def build(mock):
    if not mock:
        return Station.from_task(task).open(execute=True)
    # MockRobot needs forward/inverse kinematics before it reports TCP feedback.
    from robot_toolkit.agent import AgentOptions
    from robot_toolkit.gripper import MockGripper
    from robot_toolkit.robot import MockRobot
    from robot_toolkit.robot_platform import RobotPlatform
    robot = MockRobot(home=[0.3, 0.0, 0.3, 0.0, 0.0, 0.0],
                      forward_kinematics=lambda q: np.asarray(q, float))
    robot.set_inverse_kinematics(type('IK', (), {'solve': lambda self, pose, seed=None: list(pose)})())
    platform = RobotPlatform(robot, gripper=MockGripper(max_width=0.08, speed=10.0))
    return Station(platform, AgentOptions(max_step_m=0.03, position_tolerance_m=0.002),
                   physical=False).open(execute=True)


station = build(mock)
try:
    state = station.read()
    start = np.asarray(state.T_world_control, dtype=float).copy()
    limit = station.options.max_step_m * 0.95
    count = max(1, int(np.ceil(np.linalg.norm(delta) / limit)))
    print('start xyz_m', np.round(start[:3, 3], 4).tolist())
    print('delta_m', np.round(delta, 4).tolist(), 'steps', count)
    for i in range(1, count + 1):
        target = start.copy()
        target[:3, 3] = start[:3, 3] + delta * i / count
        receipt = station.move_pose(target, gripper=state.gripper)
        print(f'step {i}/{count} accepted={receipt.accepted} outcome={receipt.outcome}')
        station.read()
    print('end xyz_m', np.round(np.asarray(station.read().T_world_control)[:3, 3], 4).tolist())
finally:
    station.close()
PY
    ;;
  env)
    echo "interpreter: $PY"
    "$PY" - <<'PY'
import importlib.util
for m in ('numpy', 'pyrealsense2', 'pyAgxArm', 'rtde_control', 'fastapi', 'uni_robot_agent'):
    print(f'  {m:16} {"ok" if importlib.util.find_spec(m) else "missing"}')
PY
    ;;
  observe) shift; uni observe "$@" ;;
  execute)
    shift
    if [[ ${1:-} != --yes ]]; then
      read -r -p "About to command real arm motion. Type yes to continue: " reply
      [[ $reply == yes ]] || die "aborted"
      [[ ${1:-} == --yes ]] || set -- --yes "$@"
    fi
    shift
    uni execute "$@"
    ;;
  test)
    shift
    exec "$PY" -m pytest -q "$WS/robot_toolkit/tests/test_ur5e_agent_config.py" "$@"
    ;;
  strangeloop)
    shift
    export PYTHONPATH="$PYTHONPATH:$WS/StrangeLoop/src"
    "$PY" -c 'import strangeloop.environment.adapters.real' 2>/dev/null || \
      die "StrangeLoop adapter no longer imports: it targets adapters/base.py, absent in current main"
    exec "$PY" -m pytest -q "$WS/StrangeLoop/tests/test_real_environment_adapter.py" "$@"
    ;;
  *)
    cat <<'USAGE'
usage: realrun.sh <command> [args]

  env                     show the interpreter and its hardware dependencies
  nudge <task> [x] [y] [z] move the control point by centimetres (default 0 0 3)
                          and report each receipt; MOCK=1 for a dry run
  observe <task>          read-only UNI session, no motion
  execute <task>          UNI session that may move the arm (asks for confirmation)
  test                    run the robot_toolkit UR5e config test
  strangeloop             run the StrangeLoop adapter test (fails while that
                          adapter still targets the pre-refactor API)

task  is a config name (piper_agent, ur5e_agent) or a path to a task JSON.
env   OUT, PORT, INSTRUCTION override the defaults; PY overrides the interpreter.
USAGE
    ;;
esac
