#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${SCRIPT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

set +u
source /home/lorenzo/isaac/setup_ros_env.sh
set -u

cd "${PROJECT_ROOT}"
exec /home/lorenzo/isaac/python.sh -m smores_ep.multi_lift_cli "$@"
