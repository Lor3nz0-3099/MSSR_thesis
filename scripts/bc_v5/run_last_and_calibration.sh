#!/usr/bin/env bash
# Evaluate epoch-20 last.pt and sweep the decoder logits on VALIDATION ONLY.
# No retraining, no TEST access, no decoder/GUI changes.
set -Eeuo pipefail
cd "$HOME/MSSR_thesis" || exit 1
BC_PY="$HOME/.local/share/mssr-bc-v1/venv/bin/python"
[[ -x "$BC_PY" ]] || { echo "[STOP] BC Python missing: $BC_PY" >&2; exit 2; }
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
printf '\n[TEST] Evaluator + decoder threshold calibration tests\n'
"$BC_PY" -B -m pytest -q \
    scripts/bc_v5/tests/test_evaluate_checkpoint.py \
    scripts/bc_v5/tests/test_decoder_calibration.py
unset PYTEST_DISABLE_PLUGIN_AUTOLOAD
printf '\n[CHECK] last.pt vs certified G2/F2/Stage5/Stage678\n'
"$BC_PY" -B scripts/bc_v5/evaluate_checkpoint.py --check --checkpoint last
printf '\n[1/2] Epoch 20: full VALIDATION metrics at decoder logit 0.0\n'
"$BC_PY" -B scripts/bc_v5/evaluate_checkpoint.py \
    --validation --checkpoint last --threshold 0.0 \
    --batch-size 32 --progress-every 25 \
    2>&1 | tee "$HOME/bc_v5_last_validation_eval.log"
printf '\n[COMPARE] BEST epoch 12 vs LAST epoch 20 at unchanged threshold 0.0\n'
"$BC_PY" -B scripts/bc_v5/compare_checkpoints.py
printf '\n[2/2] Epoch 20: single-pass VALIDATION decoder sweep (TEST sealed)\n'
"$BC_PY" -B scripts/bc_v5/calibrate_decoder.py \
    --checkpoint last --batch-size 32 --progress-every 25 --max-recall-drop 0.01 \
    2>&1 | tee "$HOME/bc_v5_last_decoder_calibration.log"
printf '\n[DONE] Compare best vs last. A decoder proposal was SAVED, NOT APPLIED to GUI.\n'
