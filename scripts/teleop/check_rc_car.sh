#!/usr/bin/env bash
set -eo pipefail
T3_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$T3_ROOT"
source /opt/ros/humble/setup.bash
source mssr_ws/install/setup.bash
export PYTHONPATH="$T3_ROOT/mssr_ws/src/mssr_expert:$T3_ROOT/scripts/smores_ep/src${PYTHONPATH:+:$PYTHONPATH}"
exec python3 scripts/teleop/check_rc_car.py "$@"
