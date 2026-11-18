# BC-v5 v8 — validation checkpoint (18 November 2026)

## Experimental identity

- Repo branch: `snake8-global-path-ik-recovery`.
- Training run: `artifacts/bc_v1/runs/bc_v5_v8_primitive_first_20261117T143857Z_118314`.
- Checkpoints: `best.pt` (epoch 12, early-stopping by validation loss), `last.pt` (epoch 20, patience 8/8).
- TRAIN/VALIDATION: 82,518 / 23,749 supervised rows; TEST is sealed (`test_payload_access = 0`).
- Batch size 64; 48 uniform + 16 primitive-first balanced TRAIN draws; no VALIDATION resampling.
- Base logit decoder threshold `0.0`. Validation calibration only: recommended logit `1.0` under <= 1 percentage point global recall-drop guard. GUI not changed.
- Reproducing runs needs the separately stored immutable Canonical v8, G1, F2, G2 and the local large Stage 5 integrity overlay; their big payloads are intentionally not in Git.

## Checkpoint and source fingerprint

| Item | Value |
|---|---|
| best.pt SHA-256 | `24e3542b49352739e527ea6d3290f5d4d3953d82aa2383825a36e68a6b5305d3` |
| last.pt SHA-256 | `38b98fd70ca104e248bd507a59d19973fcf40c64c07f46ed2471f36b8d6d2205` |
| F2 fingerprint | `04bec0c04f091f0a4dc9b55c0fb606abe56f27258cf2acaa2afa96d4efab97e8` |
| G2 manifest fingerprint | `75f604a8d33f81ce18736006e1c3be6545c21cfdd57d18e44d3be41bd1e6f0b3` |
| Stage 678 fingerprint | `5e21171a02959e55197bbdf34a8e7ef6b1bd2638458872f116cb2a877fc4d4d4` |

## VALIDATION comparison (all 23,749 rows)

| Metric | best ep12 @ 0.0 | last ep20 @ 0.0 | last ep20 @ 1.0 |
|---|---:|---:|---:|
| Loss | 0.126336 | 0.129523 | 0.129523 |
| Precision | 0.750 | 0.815 | 0.854 |
| Recall | 0.997 | 0.989 | 0.981 |
| F1 | 0.856 | 0.894 | 0.913 |
| Exact-Set | 0.691 | 0.736 | 0.775 |

These are VALIDATION results only. Last ep20 was preferred for GUI *experiments*, not proven on TEST. A decoder sweep with 14 logit thresholds is recorded under `artifacts/bc_v1/evaluations/bc_v5_v8_last_epoch20_decoder_calibration_20261118T092914Z`; threshold 1.0 is a **proposal**, not a GUI default.

## Reports

- best metrics: `artifacts/bc_v1/evaluations/bc_v5_v8_best_epoch12_validation_20261118T085925Z`
- last metrics: `artifacts/bc_v1/evaluations/bc_v5_v8_last_epoch20_validation_20261118T092501Z`
- last decoder sweep: `artifacts/bc_v1/evaluations/bc_v5_v8_last_epoch20_decoder_calibration_20261118T092914Z`

Per-primitive precision/recall/F1, exact-set and physical/normalized regression MAE/RMSE, morphology/task breakdowns are in these versioned CSV/JSON reports. No TEST experiment has been run or selected. Any later TEST evaluation must freeze model and decoder first.
