#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${SCRIPT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

# Isaac's Python launcher does not source the ROS bridge environment itself.
# Keep user-selected ROS_DISTRO/RMW values, otherwise select Isaac's Humble
# defaults for Ubuntu 22.04.
set +u
source /home/lorenzo/isaac/setup_ros_env.sh
set -u

cd "${PROJECT_ROOT}"
exec /home/lorenzo/isaac/python.sh -m smores_ep.cli "$@"
