#!/usr/bin/env bash
# servo.sh with the example desk obstacles. See run_servo.py.
exec "${PYTHON:-python}" "$(dirname "$0")/run_servo.py" --obstacles "$@"
