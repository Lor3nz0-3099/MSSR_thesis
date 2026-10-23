# Final BC-v4 official one-shot TEST assessment

A — OFFICIAL TEST EVALUATION COMPLETE; BC-v4 SUITABLE FOR MAPPO WARM START

Checkpoint: `/home/lorenzo/MSSR_thesis/artifacts/bc_v1/runs/20261022T232532_854802073Z_207309_bc_v4_actor_visible_r1m_scratch_final/best.pt`; epoch5; SHA256 `d90a73e9b5ba2ecb579e4fc8733066b1c55d9e60dafeb1d71c029c49a7c21c6d`.
Official TEST evaluation number=1.18536samples,12episodes,290batches; one model prediction per sample.
Threshold logit0.0; FP32 eager; TF32/AMP/compileOFF; eval/no_grad; no optimizer/backward.
Frozen membership metrics use exactly production VAL raw candidate identities before physical feasibility/claim validation. Exact-set ignores numeric values; matched normalized payload metrics evaluate values separately.

## A. Generalization

| Metric | VAL best epoch5 | TEST | TEST minus VAL |
|---|---:|---:|---:|
| exact_match | 0.51546437 | 0.482250755 | -0.0332136152 |
| f1 | 0.811132357 | 0.823001669 | +0.0118693116 |
| precision | 0.682554393 | 0.699787549 | +0.017233156 |
| predicted_cardinality | 10.3345049 | 10.3605956 | +0.0260907212 |
| recall | 0.999396229 | 0.998877666 | -0.000518563689 |
| total_loss | 0.0686852984 | 0.0775344902 | +0.00884919172 |

Task and episode dispersion is fully preserved in test_task_breakdown.json and test_episode_breakdown.json; current task names are retained without inferring assembly/reconfiguration labels from episode names.

## B. Action-set understanding

TEST precision=0.699788,recall=0.998878,F1=0.823002,Jaccard=0.775522.
Target cardinality mean=7.258362 [1,15]; predicted mean=10.360596 [1,21]; absolute cardinality error=3.106010.
INITIAL RANDOM VAL REFERENCE ONLY: F1=0.089772,exact-set=0,predicted cardinality=150.011554. This is not a random TEST baseline and no random TEST model was evaluated.

| Action | Support | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| align_faces | 85 | 74 | 333 | 11 | 0.181818 | 0.870588 | 0.300813 |
| dock | 81 | 73 | 152 | 8 | 0.324444 | 0.901235 | 0.477124 |
| gravity_settle | 3 | 3 | 229 | 0 | 0.012931 | 1.000000 | 0.025532 |
| pan_delta | 0 | 0 | 15 | 0 | 0.000000 | 0.000000 | 0.000000 |
| pan_target | 37 | 8 | 517 | 29 | 0.015238 | 0.216216 | 0.028470 |
| pan_velocity | 26756 | 26753 | 144 | 3 | 0.994646 | 0.999888 | 0.997260 |
| tilt_delta | 0 | 0 | 14 | 0 | 0.000000 | 0.000000 | 0.000000 |
| tilt_target | 10677 | 10590 | 40315 | 87 | 0.208035 | 0.991852 | 0.343932 |
| tilt_velocity | 0 | 0 | 15 | 0 | 0.000000 | 0.000000 | 0.000000 |
| undock | 22 | 14 | 633 | 8 | 0.021638 | 0.636364 | 0.041854 |
| wheel_velocity | 96880 | 96875 | 15287 | 5 | 0.863706 | 0.999948 | 0.926847 |

Support0 means no positive TEST evidence; inherited empty-set scoring does not prove that action learned.

## C. Dominant failure mode

FP=57654,FN=151; FP/sample=3.110380,FN/sample=0.008146.
Largest FP contributions: tilt_target:40315, wheel_velocity:15287, undock:633, pan_target:517, align_faces:333.
Normalized matched payload MAE=0.154387,RMSE=0.306697,valid scalar support=231101. Numeric statistics are conditional on matched actions and cannot compensate for FP/FN or establish physical task success.
Worst task exact-set: stairs=0.236239 (n=6722); best: navigation=0.989228 (n=3992).
Worst episode: expert:assembly:seed-006284=0.000000 (n=19); best: expert:rc_car:seed-006004=0.970696 (n=273).

## D. Exact-set interpretation

| FP+FN | Samples | Fraction |
|---|---:|---:|
| 0 | 8939 | 48.225076% |
| 1 | 1901 | 10.255719% |
| 2 | 1292 | 6.970220% |
| 3 | 764 | 4.121709% |
| 4+ | 5640 | 30.427277% |

Exact=8939; off-by-one=1901; at least2errors=7696.
This distinguishes exact membership and near misses from larger bundle errors; it does not equate any non-exact sample with random behavior.

| Pattern | Samples | Fraction |
|---|---:|---:|
| FN_only | 24 | 0.129478% |
| FP_only | 9469 | 51.084376% |
| both | 104 | 0.561070% |
| exact | 8939 | 48.225076% |

## E. Warm start

SUITABLE_FOR_MAPPO_WARM_START
The frozen actor is SUITABLE_FOR_MAPPO_WARM_START (classification A). Overall TEST generalization is close to VAL: F1 improves from0.811132 to0.823002, precision improves0.017233, recall changes-0.000519, exact-set decreases0.033214, and loss increases0.008849. These are aggregate observations, not statistical equivalence claims or task-balanced guarantees.

Meaningful learned action structure is supported by wheel_velocity F1=0.926847, pan_velocity F1=0.997260, and navigation F1=0.996537/exact-set=0.989228. Predicted cardinality10.360596 is far more selective than the INITIAL RANDOM VAL REFERENCE of150.011554; that reference is qualitative context, not a random TEST experiment. The warm-start verdict uses coherent action membership, payload behavior and held-out generalization together, not recall alone.

The dominant limitation is excessive false-positive actions:57654FP versus151FN. tilt_target accounts for40315FP and wheel_velocity for15287FP. Stairs contributes42886FP and has exact-set0.236239/F1=0.659672, versus navigation exact-set0.989228. Strong overall micro metrics therefore conceal substantial task variation and dominance by common resource commands. The largest episode has10384/18536samples; all12episode summaries are retained, with no averaging or filtering introduced after TEST.

Exact-set means8939exact samples(48.2251%),1901off-by-one(10.2557%), and7696with at least2membership errors(41.5192%), including5640with4+errors(30.4273%). Thus the non-exact remainder is not predominantly off-by-one. FP-only samples are9469; FN-only24; both104. Non-exact does not mean random, but multi-action overactivation is a material failure mode.

Rare action weaknesses remain explicit: ALIGN_FACES F1=0.300813(n=85), DOCK0.477124(n=81), UNDOCK0.041854(n=22), GRAVITY_SETTLE0.025532(n=3), and pan_target0.028470(n=37). pan_target matched normalized payload RMSE=2.337540 has only8matched scalar observations; global payload RMSE0.306697 must not hide this. The assembly expert episode has0/19exact sets. These findings do not establish reliable autonomous structural execution.

The actor is a materially structured initialization with reasonable aggregate held-out generalization, rather than a complete or uniformly strong imitation controller. MAPPO warm-start suitability is an offline judgment; no MAPPO, rollout, safety/feasibility validation or task-success experiment is performed here. The existing BC checkpoint, threshold and training remain frozen; no BC retraining, replacement or tuning is recommended from TEST.

The current task routing contains only button,gap,navigation,stairs. No assembly/reconfiguration task category is inferred from episode names or future morphology. The assembly episode is identifiable in the episode breakdown; its task routing is retained exactly as frozen.

This is an actor-initialization assessment, not evidence of autonomous execution success. No changes to BC checkpoint/threshold/training are made or recommended from TEST.

## Integrity and access

TEST wall time=1451.647s; throughput=12.769samples/s, metadata only; no profiling.
Model state SHA256 before/after identical: `80cdf41cadd12d4b6e9af825e2619c2e5bba32f26fd0b81ac08005b440d59ad3`. Checkpoint bytes unchanged.
Streaming G1 TEST episode loading and in-memory actor projection/tensorization reused frozen F2/current candidates; no persistent TEST sidecar. All12episode payload loaders called once. Hash authentication reads bytes separately from deserialization; there is one logical TEST iterator and no second inference/metric pass.
Pre-access incident: audit-local SyntaxError before Python execution/import/TEST access; corrected and syntax-checked. No execution claim, payload or prediction existed for that parser failure. The only actual evaluation completed once, with no inference retry.
Only audit-local scripts/documents/results created; all production/tests/config/source datasets/normalizers/run artifacts preserved. See preservation.json and git snapshots.
NO THRESHOLD TUNING. NO CHECKPOINT COMPARISON. NO RETRAINING. NO PRODUCTION CODE MODIFIED. OFFICIAL TEST RESULT FROZEN.
