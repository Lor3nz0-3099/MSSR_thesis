#!/usr/bin/env bash
set -eo pipefail
source /opt/ros/humble/setup.bash
TELEOP_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${TELEOP_REPO_ROOT}/mssr_ws/src/mssr_expert${PYTHONPATH:+:${PYTHONPATH}}"
cd -- "${TELEOP_REPO_ROOT}"
exec python3 scripts/teleop/check_dualsense.py "$@"
