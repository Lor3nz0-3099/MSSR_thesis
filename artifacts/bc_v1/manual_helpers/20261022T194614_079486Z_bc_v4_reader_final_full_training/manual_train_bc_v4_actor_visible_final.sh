#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/lorenzo/MSSR_thesis
BC_PYTHON=/home/lorenzo/.local/share/mssr-bc-v1/venv/bin/python
G2="$ROOT/artifacts/bc_v1/sidecar/g2_v4_actor_visible_r1m"
F2="$ROOT/artifacts/bc_v1/normalization/f2_actor_visible_r1m_v1"
CONFIG="$ROOT/configs/bc_v1/training_action_set_v1.json"
PREFLIGHT="/home/lorenzo/MSSR_thesis/artifacts/bc_v1/manual_helpers/20261022T194614_079486Z_bc_v4_reader_final_full_training/launch_preflight.py"
if [[ $# -eq 0 || ( $# -eq 1 && "$1" == --help ) ]]; then
    printf '%s\n' 'Manual final BC-v4 actor-visible R1M training, FROM SCRATCH.' 'Requires exactly --run. NOT EXECUTED.'
    exit 0
fi
if [[ $# -ne 1 || "$1" != --run ]]; then
    printf '%s\n' 'Usage: bash manual_train_bc_v4_actor_visible_final.sh --run' >&2
    exit 64
fi
cd "$ROOT"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
"$BC_PYTHON" -B "$PREFLIGHT"
STAMP="$(date -u +%Y%m%dT%H%M%S_%NZ)_$$_bc_v4_actor_visible_r1m_scratch_final"
OUTPUT="$ROOT/artifacts/bc_v1/runs/$STAMP"
LOG_DIR="$ROOT/artifacts/bc_v1/manual_runs"
LOG="$LOG_DIR/$STAMP.log"
[[ ! -e "$OUTPUT" ]] || { printf 'Output collision: %s\n' "$OUTPUT" >&2; exit 1; }
mkdir -p "$LOG_DIR"
(set -o noclobber; : > "$LOG")
COMMAND=("$BC_PYTHON" -B -u -m drl.IL.bc_v1.train_action_set
    --config "$CONFIG" --g2 "$G2" --normalizers "$F2" --output "$OUTPUT" --run)
{
    printf '%s\n' 'FINAL BC-v4 TRAINING — FROM SCRATCH' 'PRETRAINED: NONE; RESUME: NONE; TEST: SEALED' "OUTPUT=$OUTPUT" "LOG=$LOG"
    "$BC_PYTHON" -B "$PREFLIGHT"
    printf 'EXECUTING:'; printf ' %q' "${COMMAND[@]}"; printf '\n'
    "${COMMAND[@]}"
    printf 'TRAINING COMPLETED SUCCESSFULLY\nOUTPUT=%s\nLOG=%s\n' "$OUTPUT" "$LOG"
} 2>&1 | tee -a "$LOG"
