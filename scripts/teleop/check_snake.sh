#!/usr/bin/env bash
set -eo pipefail
T6_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$T6_ROOT"
source /opt/ros/humble/setup.bash
source mssr_ws/install/setup.bash
export PYTHONPATH="$T6_ROOT/mssr_ws/src/mssr_expert:$T6_ROOT/scripts/smores_ep/src${PYTHONPATH:+:$PYTHONPATH}"
exec python3 scripts/teleop/check_snake.py "$@"
