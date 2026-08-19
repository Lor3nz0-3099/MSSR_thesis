#!/usr/bin/env bash
set -euo pipefail

ISAAC_ROOT="/home/lorenzo/isaac"
USD_LIBS="${ISAAC_ROOT}/extscache/omni.usd.libs-1.0.3+6312fa25.lx64.r.cp312"
PHYSX_SCHEMA="${ISAAC_ROOT}/extscache/omni.usd.schema.physx-110.1.11+110.1.1.lx64.r.cp312.u7f4"

export PYTHONPATH="${USD_LIBS}:${PHYSX_SCHEMA}${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${USD_LIBS}/bin:${PHYSX_SCHEMA}/bin${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

"${ISAAC_ROOT}/kit/python/bin/python3" "$@"
