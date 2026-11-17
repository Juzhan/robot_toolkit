#!/usr/bin/env bash
# PBVS hand following with the tuned preset (moves the arm). See run_servo.py.
exec "${PYTHON:-python}" "$(dirname "$0")/run_servo.py" "$@"
