#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${SCRIPT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

# The self-assembly runtime communicates through atomic JSON files. ROS 2 and
# rclpy stay in the external bridge process, so Isaac does not load its ROS
# extensions and the GUI is independent from the selected DDS implementation.
cd "${PROJECT_ROOT}"
exec /home/lorenzo/isaac/python.sh -m smores_ep.self_assembly_cli "$@"
