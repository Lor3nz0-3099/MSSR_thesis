#!/usr/bin/env python3
"""BC-v5 TEST readiness inspection: metadata only. NEVER opens TEST examples."""
import hashlib
import json
from pathlib import Path

root = Path.cwd()
split = json.loads((root/'artifacts/bc_v1/splits/f1_v8_640/split.json').read_text())
g2 = json.loads((root/'artifacts/bc_v1/sidecar/g2_v8_bc_v5_stage11_full/manifest.json').read_text())
fit = json.loads((root/'artifacts/bc_v1/normalization/f2_v8_actor_visible_r1m/fit_context.json').read_text())
run = root/'artifacts/bc_v1/runs/bc_v5_v8_primitive_first_20261117T143857Z_118314'
ck = run/'last.pt'
val_identity = json.loads((root/'artifacts/bc_v1/evaluations/bc_v5_v8_last_epoch20_validation_20261118T092501Z/identity.json').read_text())
actual_hash = hashlib.sha256(ck.read_bytes()).hexdigest()
expected = val_identity['checkpoint_sha256']
assert actual_hash == expected, 'STOP: checkpoint sha mismatch'
assert val_identity['checkpoint_epoch'] == 20, 'STOP: checkpoint epoch mismatch'
assert split['global_counts']['test'] == 102, 'STOP: unexpected holdout count'
assert len(split['splits']['test']) == 102, 'STOP: unexpected holdout membership count'
assert not (set(split['splits']['test']) & set(split['splits']['train'])), 'STOP: split overlap train/test'
assert not (set(split['splits']['test']) & set(split['splits']['validation'])), 'STOP: split overlap val/test'
assert g2['inventory']['test'] == [], 'STOP: the current G2 no longer has a sealed TEST inventory'
assert g2['split_counts']['test']['episodes'] == 0, 'STOP: TEST payload unexpectedly in current G2'
assert fit['TRAIN_only'] is True and fit['TEST_payload_access'] == 0, 'STOP: F2 fit leaks TEST'
assert g2['identity']['f2_fingerprint'] == val_identity['f2_fingerprint'], 'STOP: F2 fingerprint mismatch'
print('PASS: checkpoint last.pt epoch=20 SHA256 verified')
print('PASS: F2 normalization fitted on TRAIN only')
print('PASS: immutable F1-v8 holdout TEST = 102 episodes, disjoint from TRAIN and VALIDATION')
print('PASS: G2 Stage11 training/validation dataset has 0 TEST episodes (sealed)')
print('NEXT: build a NEW read-only TEST sidecar from the F1-v8 sealed holdout and fixed F2;')
print('      implement a separate one-shot TEST evaluator. Do not use --validation as TEST.')
print('      Decoder threshold=1.0 frozen from VALIDATION before TEST; no refitting/tuning.')
