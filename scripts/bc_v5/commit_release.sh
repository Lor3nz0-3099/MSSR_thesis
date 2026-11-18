#!/usr/bin/env bash
# BC-v5 v8 reproducible release: stage ONLY selected BC-v5 code and small evidence.
# Run from the local MSSR_thesis checkout. Does not run training or touch TEST.
set -Eeuo pipefail

ROOT="${HOME}/MSSR_thesis"
BRANCH='snake8-global-path-ik-recovery'
RUN='artifacts/bc_v1/runs/bc_v5_v8_primitive_first_20261117T143857Z_118314'
BASE='artifacts/bc_v1/evaluations'
BEST='bc_v5_v8_best_epoch12_validation_20261118T085925Z'
LAST='bc_v5_v8_last_epoch20_validation_20261118T092501Z'
CAL='bc_v5_v8_last_epoch20_decoder_calibration_20261118T092914Z'
BC_PY="${HOME}/.local/share/mssr-bc-v1/venv/bin/python"

fail() { printf '[STOP] %s\n' "$*" >&2; exit 2; }
cd "$ROOT" || fail "Repository non trovata: $ROOT"
[[ -d .git ]] || fail 'Non sei nella radice della repository Git'
[[ "$(git branch --show-current)" == "$BRANCH" ]] || fail "Branch inatteso: $(git branch --show-current)"
[[ -x "$BC_PY" ]] || fail "Venv BC-v5 assente: $BC_PY"
[[ -z "$(git diff --cached --name-only)" ]] || fail 'Ci sono file già in staging: NON li mescolo con il rilascio BC-v5.'
[[ -n "$(git remote get-url origin)" ]] || fail 'Remote origin non configurato'

for p in "$RUN/best.pt" "$RUN/last.pt" "$RUN/history.jsonl" "$RUN/identity.json" "$RUN/plan.json" "$RUN/status.json" \
         "$BASE/$BEST/identity.json" "$BASE/$BEST/metrics.json" \
         "$BASE/$LAST/identity.json" "$BASE/$LAST/metrics.json" \
         "$BASE/$CAL/identity.json" "$BASE/$CAL/calibration.json" \
         "configs/bc_v1/bc_v5_v8_training_bc_v4_hparams.json" \
         "drl/IL/bc_v1/bc_v5_train.py" "drl/IL/bc_v1/bc_v5_v8_sampling.py" \
         "scripts/bc_v5/evaluate_checkpoint.py" "scripts/bc_v5/calibrate_decoder.py"; do
  [[ -f "$p" ]] || fail "File obbligatorio mancante: $p"
done

# Small integrity check against the independent evaluation certificates.
"$BC_PY" - "$RUN" "$BASE/$BEST" "$BASE/$LAST" "$BASE/$CAL" <<'PY'
from pathlib import Path
import hashlib,json,sys
run,best,last,cal=map(Path,sys.argv[1:])
cfg=json.loads(Path('configs/bc_v1/bc_v5_v8_training_bc_v4_hparams.json').read_text())
expected={'lr':0.0003,'weight_decay':0.0001,'batch_size':64,'max_epochs':30,
          'patience':8,'min_delta':0.002,'seed':20261021,'threshold_logit':0.0}
assert all(cfg.get(k)==v for k,v in expected.items()), 'BC-v4 hyperparameters changed'
status=json.loads((run/'status.json').read_text())
assert status.get('phase')=='COMPLETE' and status.get('test_payload_access')==0, 'Run not complete or TEST used'
print('[PASS] Full training complete; BC-v4 hyperparameters, patience=8')
for name,p,epoch in [('best',best,12),('last',last,20)]:
    doc=json.loads((p/'identity.json').read_text())
    checkpoint=run/(name+'.pt')
    checksum=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert checksum==doc['checkpoint_sha256'], f'{name}.pt SHA mismatch'
    assert doc['checkpoint_epoch']==epoch and doc['rows']==23749 and doc['split']=='validation'
    assert doc['threshold_logit']==0.0 and doc['test_payload_access']==0
    print(f'[PASS] {name}.pt ep={epoch} sha={checksum[:16]} VALIDATION={doc["rows"]}')
    assert checkpoint.stat().st_size<45*1024**2, 'Unexpected huge checkpoint'
c=json.loads((cal/'identity.json').read_text())
assert c['checkpoint_sha256']==hashlib.sha256((run/'last.pt').read_bytes()).hexdigest()
assert c['validation_rows']==23749 and c['test_payload_access']==0
calibr=json.loads((cal/'calibration.json').read_text())
assert calibr['test_payload_access']==0 and calibr['suggested_threshold_logit']==1.0
assert calibr['gui_decoder_changed'] is False
print('[PASS] Calibration on last.pt, GUI unchanged, TEST sealed')
PY

export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
printf '\n[TEST] Evaluator + calibration\n'
"$BC_PY" -B -m pytest -q scripts/bc_v5/tests/test_evaluate_checkpoint.py scripts/bc_v5/tests/test_decoder_calibration.py
printf '\n[TEST] BC-v5 code and sampler\n'
"$BC_PY" -B -m pytest -q \
  drl/IL/tests/bc_v1/test_bc_v5_stage678_v8.py \
  drl/IL/tests/bc_v1/test_bc_v5_binary_integrity.py \
  drl/IL/tests/bc_v1/test_bc_v5_census_sampling.py \
  drl/IL/tests/bc_v1/test_bc_v5_temporal_optional_v8.py
unset PYTEST_DISABLE_PLUGIN_AUTOLOAD

printf '\n[REMOTE] Checking branch status\n'
git fetch --no-tags origin "+refs/heads/$BRANCH:refs/remotes/origin/$BRANCH"
git merge-base --is-ancestor "refs/remotes/origin/$BRANCH" HEAD || \
  fail 'Il remoto contiene commit non presenti localmente. Push fermato: sincronizzare senza force.'

# Release note is deliberately independent of bulky G1/G2; the fingerprints
# below were read from the evaluation identities, not recomputed from payloads.
mkdir -p docs
DOC='docs/bc_v5_v8_validation_release.md'
[[ ! -e "$DOC" ]] || fail "Documento già presente (evito sovrascrittura): $DOC"
"$BC_PY" - "$DOC" "$RUN" "$BASE/$BEST" "$BASE/$LAST" "$BASE/$CAL" <<'PY'
from pathlib import Path
import json,sys
out,run,best,last,cal=map(Path,sys.argv[1:])
b=json.loads((best/'identity.json').read_text()); l=json.loads((last/'identity.json').read_text())
c=json.loads((cal/'calibration.json').read_text())
text=f'''# BC-v5 v8 — validation checkpoint (18 November 2026)

## Experimental identity

- Repo branch: `snake8-global-path-ik-recovery`.
- Training run: `{run.as_posix()}`.
- Checkpoints: `best.pt` (epoch 12, early-stopping by validation loss), `last.pt` (epoch 20, patience 8/8).
- TRAIN/VALIDATION: 82,518 / 23,749 supervised rows; TEST is sealed (`test_payload_access = 0`).
- Batch size 64; 48 uniform + 16 primitive-first balanced TRAIN draws; no VALIDATION resampling.
- Base logit decoder threshold `0.0`. Validation calibration only: recommended logit `1.0` under <= 1 percentage point global recall-drop guard. GUI not changed.
- Reproducing runs needs the separately stored immutable Canonical v8, G1, F2, G2 and the local large Stage 5 integrity overlay; their big payloads are intentionally not in Git.

## Checkpoint and source fingerprint

| Item | Value |
|---|---|
| best.pt SHA-256 | `{b['checkpoint_sha256']}` |
| last.pt SHA-256 | `{l['checkpoint_sha256']}` |
| F2 fingerprint | `{l['f2_fingerprint']}` |
| G2 manifest fingerprint | `{l['g2_manifest_fingerprint']}` |
| Stage 678 fingerprint | `{l['stage678_fingerprint']}` |

## VALIDATION comparison (all 23,749 rows)

| Metric | best ep12 @ 0.0 | last ep20 @ 0.0 | last ep20 @ 1.0 |
|---|---:|---:|---:|
| Loss | 0.126336 | 0.129523 | 0.129523 |
| Precision | 0.750 | 0.815 | 0.854 |
| Recall | 0.997 | 0.989 | 0.981 |
| F1 | 0.856 | 0.894 | 0.913 |
| Exact-Set | 0.691 | 0.736 | 0.775 |

These are VALIDATION results only. Last ep20 was preferred for GUI *experiments*, not proven on TEST. A decoder sweep with 14 logit thresholds is recorded under `{cal.as_posix()}`; threshold 1.0 is a **proposal**, not a GUI default.

## Reports

- best metrics: `{best.as_posix()}`
- last metrics: `{last.as_posix()}`
- last decoder sweep: `{cal.as_posix()}`

Per-primitive precision/recall/F1, exact-set and physical/normalized regression MAE/RMSE, morphology/task breakdowns are in these versioned CSV/JSON reports. No TEST experiment has been run or selected. Any later TEST evaluation must freeze model and decoder first.
'''
out.write_text(text,encoding='utf-8')
print('[RELEASE] '+str(out))
PY

# Explicit allowlist for all source changes; never `git add -A` or blanket artifacts.
shopt -s nullglob
sources=(
  drl/IL/bc_v1/bc_v5*.py
  drl/IL/tests/bc_v1/test_bc_v5*.py
  configs/bc_v1/bc_v5*.json
  docs/bc_v5_v8_validation_release.md
)
while IFS= read -r -d '' item; do sources+=("$item"); done \
  < <(find scripts/bc_v5 -type f \( -name '*.py' -o -name '*.sh' -o -name '*.md' \) -print0)
for item in "${sources[@]}"; do
  [[ -f "$item" ]] || continue
  git add -- "$item"
done

# Force-add only selected tiny metadata in artifacts, not dataset payloads.
artefacts=(
 "$RUN/best.pt" "$RUN/last.pt" "$RUN/identity.json" "$RUN/plan.json" "$RUN/status.json" "$RUN/history.jsonl"
 'artifacts/bc_v1/splits/f1_v8_640/split.json'
 'artifacts/bc_v1/splits/f1_v8_640/audit.json'
 'artifacts/bc_v1/splits/f1_v8_640/episodes.json'
 'artifacts/bc_v1/splits/f1_v8_640/source_hashes.json'
 'artifacts/bc_v1/cache/g1_v8_640/manifest.json'
 'artifacts/bc_v1/normalization/f2_v8_actor_visible_r1m/normalizers.json'
 'artifacts/bc_v1/normalization/f2_v8_actor_visible_r1m/fit_context.json'
 'artifacts/bc_v1/sidecar/g2_v8_actor_visible_r1m/manifest.json'
 'artifacts/bc_v1/sidecar/g2_v8_actor_visible_r1m/SUCCESS.json'
 'artifacts/bc_v1/sidecar/g2_v8_bc_v5_stage11_full/manifest.json'
 'artifacts/bc_v1/sidecar/g2_v8_bc_v5_stage11_full/SUCCESS.json'
 'artifacts/bc_v1/audits/bc_v5_v8/stage3_actor_visible.json'
 'artifacts/bc_v1/audits/bc_v5_v8/stage11_full_report.json'
 'artifacts/bc_v1/audits/bc_v5_v8/stage678_readiness.json'
 'artifacts/bc_v1/audits/bc_v5_v8/temporal_requirement_removed.json'
)
for name in "$BEST" "$LAST" "$CAL"; do
  for p in "$BASE/$name"/*.json "$BASE/$name"/*.csv; do
    [[ -f "$p" ]] && artefacts+=("$p")
  done
done
for p in "${artefacts[@]}"; do
  [[ -f "$p" ]] || { echo "[SKIP] Optional metadata absent: $p"; continue; }
  sz=$(stat -c '%s' "$p")
  # Individual files capped below GitHub's 50 MB warning threshold.
  (( sz < 45*1024*1024 )) || fail "Unexpected large artifact, NOT staged: $p (${sz} bytes)"
  git add -f -- "$p"
done
# Stage5 is a full 106K-row overlay, often tens of MiB; intentionally leave local.

printf '\n[STAGED] Exact release inventory:\n'
git diff --cached --name-status
printf '\n[STAGED] Size summary:\n'
git diff --cached --stat
[[ -n "$(git diff --cached --name-only)" ]] || fail 'No changes staged; nothing to commit'
if git diff --cached --name-only | grep -Eq '^((datasets/)|(assets/)|(mssr_ws/)|(artifacts/bc_v1/sidecar/[^/]+/[^/]+/))'; then
  fail 'Prohibited directory detected in staging; no commit/push performed.'
fi
# Every artifact is explicitly selected. Verify neither the sealed TEST nor
# large data backing files slipped into the index.
if git diff --cached --name-only | grep -Ei '\.(sqlite|db|npz|npy|parquet|jsonl\.gz|zip|tar\.gz)$|/(test_payload|episode_payload)/'; then
  fail 'Unexpected heavy payload staged; no commit/push performed.'
fi

git diff --cached --check || fail 'git diff check failed'

printf '\n[COMMIT] Publishing verified BC-v5 validation release\n'
git commit -m "Release BC-v5 v8 training and validated checkpoint diagnostics"
printf '\n[PUSH] origin/%s\n' "$BRANCH"
git push origin "$BRANCH"
printf '\n[PASS] Commit and push completed. HEAD: '
git rev-parse --short HEAD
printf '[PASS] Remote tracking: '
git rev-parse --short "refs/remotes/origin/$BRANCH"
printf '[LOCAL] Uncommitted unrelated changes (if any):\n'
git status --short
