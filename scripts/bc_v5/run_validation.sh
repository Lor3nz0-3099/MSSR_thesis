#!/usr/bin/env bash
# BC-v5 v8: validation-only evaluator, no train/test writes.
set -Eeuo pipefail
cd "$HOME/MSSR_thesis" || exit 1
BC_PY="$HOME/.local/share/mssr-bc-v1/venv/bin/python"
[[ -x "$BC_PY" ]] || { echo "[STOP] Python BC-v5 not found: $BC_PY" >&2; exit 2; }
[[ -f scripts/bc_v5/evaluate_checkpoint.py ]] || { echo '[STOP] Missing in-repo evaluator' >&2; exit 2; }
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
printf '[TEST] BC-v5 evaluation metrics and physical payload codecs\n'
"$BC_PY" -B -m pytest -q scripts/bc_v5/tests/test_evaluate_checkpoint.py
unset PYTEST_DISABLE_PLUGIN_AUTOLOAD
printf '\n[CHECK] Authenticating checkpoint/F2/G2/supervision (no samples opened)\n'
"$BC_PY" -B scripts/bc_v5/evaluate_checkpoint.py --check
printf '\n[RUN] Full unresampled VALIDATION at threshold 0.0\n'
export PYTHONUNBUFFERED=1
"$BC_PY" -B scripts/bc_v5/evaluate_checkpoint.py \
  --validation --threshold 0.0 --batch-size 32 --progress-every 25 \
  2>&1 | tee "$HOME/bc_v5_best_validation_eval.log"
