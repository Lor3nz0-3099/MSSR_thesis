# POST-J BC-v1 performance diagnosis — recovered audit

Recovery closure: all completed measurements are retained in immutable `timings.json`; no GPU workload was relaunched. The operator profiler did not publish a recoverable result. This report is ready for review with that explicit gap; it does not claim completion of the original operator-level profiling requirement.

## A. Profiling environment

| Setting | Measured value |
| --- | --- |
| torch | 2.14.1+cu130 |
| cuda | 13.0 |
| gpu | NVIDIA GeForce RTX 4070 Laptop GPU |
| device | cuda:0 |
| dtype | float32 |
| amp | False |
| deterministic | False |
| seed | 20261006 |
| cpu_threads | 1 |

Real CUDA float32, no AMP, no compile; normal timing deterministic algorithms OFF. Ten full train-step warm-ups per case excluded. J: 100 eval forwards, 50 component steps, five unsplit production steps. Each remaining case: three unsplit production steps and three component steps; raw repeat counts in timings.json. Internal profiles: three eval repeats. Numerical guards use a separate deterministic verification context, excluded from timings. Script execution and logs were in /tmp and were lost at reboot.

## B. Files changed

Diagnostic/audit only:

- `timings.json`: original saved measurements, not modified during recovery.
- `summary.json`: machine-readable findings, calculations, verification and limitations.
- `report.md`: this report.
- `torch_profiler_top_ops.txt`: explicit unavailable-result notice, not fabricated operators.

No production/config/test/spec changes by this task. No production tests rerun; this closure performs arithmetic, read-only source checks and artifact validation only.

## C. J wall-time decomposition

| Part | Estimated seconds | % historical wall |
| --- | --- | --- |
| estimated_train_s | 1099.152 | 97.869 |
| estimated_periodic_forward_s | 21.969 | 1.956 |
| estimated_periodic_loss_s | 0.059 | 0.005 |
| estimated_periodic_I2_s | 0.462 | 0.041 |
| checkpoint_and_final_verification_inside_historical_timer_s | 0.000 | 0.000 |
| unattributed_difference_s | 1.449 | 0.129 |

Historical wall: 1123.090840 s. Fresh unsplit J step: 5.495762 s, versus historical 5.615454 s/step. The latter amortizes four periodic evaluations. `small_overfit.py` stops its elapsed timer BEFORE the extra final evaluation, checkpoint save and reload verification; those contribute zero to the recorded 1123.09 s, although they cost additional time outside that timer. The residual is a cross-run reconstruction difference, not a separately measured component.

## D. Loader / batch construction

| Component (ms) | J batch8 | Real batch64 mean |
| --- | --- | --- |
| Sampling | saved fixed selection; not timed | 0.025 |
| Sidecar fetch | 201.050 | 751.028 |
| Collate | 11.796 | 65.361 |
| CPU preparation | 212.846 | 816.390 |
| CPU→GPU wall | 3.014 | 6.943 |

Loader timings are single observations per batch, not repeated throughput confidence intervals. The J8 and b64_0 loader entries come from their last internal-profile reload; sampling uses the three separately saved sampler timings. Transfer uses synchronized boundaries. Pre-transfer synchronization wait is recorded separately in timings.json and may drain previous work/context initialization; it is not attributed to per-step data transfer. Sampler: 56 general + 8 structural, seed 20261006. No validation/test payloads used.

## E. Train-step decomposition

| Component (ms) | J8 mean | Batch64 pooled mean | % batch64 split wall |
| --- | --- | --- | --- |
| zero_grad | 0.264 | 0.474 | 0.002 |
| h1 | 20.576 | 38.909 | 0.136 |
| h2 | 5407.587 | 27972.595 | 97.881 |
| i1 | 14.671 | 161.647 | 0.566 |
| backward | 45.592 | 399.042 | 1.396 |
| clip | 1.427 | 2.151 | 0.008 |
| optimizer | 1.760 | 2.004 | 0.007 |
| diagnostics | 0.287 | 0.275 | 0.001 |
| full_wall_ms | 5492.805 | 28578.245 | 100.000 |

Synchronized component measurements include region boundaries; unsplit production `train_step` is the primary end-to-end baseline. Parent/component totals include small unattributed Python/timing overhead; do not force their sum to equal full wall.

| Unsplit production step | n | mean ms | median | p95 | min | max |
| --- | --- | --- | --- | --- | --- | --- |
| j8 | 5 | 5495.762 | 5474.204 | 5535.001 | 5467.755 | 5536.020 |
| b64_0 | 3 | 28764.032 | 28793.278 | 28820.921 | 28674.825 | 28823.992 |
| b64_1 | 3 | 28399.925 | 28492.386 | 28612.439 | 28081.612 | 28625.778 |
| b64_2 | 3 | 28136.379 | 28115.428 | 28174.828 | 28112.281 | 28181.428 |

All component distributions and raw observations (including p95/min/max) remain in timings.json; pooled batch64 distributions are also in summary.json. Three-repeat p95 values are descriptive, not confidence estimates.

| Path (ms) | J8 | Batch64 mean |
| --- | --- | --- |
| eval_forward | 5492.216 | 27837.975 |
| train_forward | 5451.316 | 27989.086 |
| train_forward_loss | 5466.385 | 28149.089 |
| production_train_step | 5495.762 | 28433.445 |
| split forward+loss+backward | 5488.427 | 28572.193 |

## F. H1 internal profile

| Region (host dispatch/wall ms) | J8 | b64_0 |
| --- | --- | --- |
| h1.validation | 2.444 | 2.522 |
| h1.encoder.module | 0.114 | 0.116 |
| h1.encoder.connector | 0.074 | 0.079 |
| h1.encoder.resource | 0.066 | 0.070 |
| h1.encoder.coordinate | 0.065 | 0.072 |
| h1.encoder.connection | 0.061 | 0.078 |
| h1.encoder.relation | 0.063 | 0.072 |
| aggregation@h1.encode_local | 0.104 | 0.106 |
| h1.resource_fusion | 0.074 | 0.078 |
| h1.environment | 13.497 | 25.760 |
| h1.hierarchy | 0.416 | 0.419 |
| h1.message.1 | 0.938 | 0.947 |
| h1.message.2 | 0.921 | 0.916 |
| h1.message.3 | 0.894 | 0.904 |
| h1.pooling_readout | 0.639 | 0.655 |
| h1.total | 20.849 | 33.301 |

Hierarchy includes resource/connector→module aggregation and module fusion; local aggregation is coordinate→resource. Environment encoding/validation dominates H1 itself, but H1 is minor relative to the complete step. Nested regions overlap and must not be added twice.

| Message block | Physical MLP J8 / B64 ms | Environment MLP J8 / B64 ms | All three aggregations J8 / B64 ms |
| --- | --- | --- | --- |
| 1 | 0.087 / 0.078 | 0.079 / 0.079 | 0.185 / 0.194 |
| 2 | 0.079 / 0.078 | 0.087 / 0.081 | 0.196 / 0.192 |
| 3 | 0.077 / 0.083 | 0.075 / 0.078 | 0.186 / 0.190 |

| Readout detail ms | J8 | b64_0 |
| --- | --- | --- |
| aggregation@h1.pooling_readout | 0.066 | 0.063 |
| h1.graph_projection | 0.034 | 0.038 |
| h1.graph_norm | 0.021 | 0.028 |
| h1.connector_context | 0.082 | 0.082 |
| h1.resource_context | 0.070 | 0.073 |

## G. H2 internal profile

| Region (host dispatch/wall ms) | J8 | b64_0 |
| --- | --- | --- |
| h2.resource | 4.486 | 27.549 |
| h2.structural_conditions | 55.585 | 275.768 |
| h2.structural_scorer | 0.078 | 0.076 |
| h2.argument_validation | 4899.101 | 25188.942 |
| h2.argument_selection_and_validation | 5023.325 | 25835.279 |
| h2.arg.numeric | 1.555 | 8.603 |
| h2.arg.reference | 0.098 | 0.400 |
| h2.arg.reference_set | 15.109 | 123.028 |
| h2.arg.boolean | 0.026 | 0.034 |
| h2.arg.category | 123.670 | 669.946 |
| h2.arguments | 5163.888 | 26637.401 |
| h2.scope | 245.399 | 1309.592 |
| h2.total | 5469.562 | 28250.504 |

Argument validation alone accounts for 89.57% / 89.16% of H2. Selection-and-validation includes validation; exclusive selection is their difference, not another additive full region. All-arguments includes numeric/reference/category stages. Structural scorer timings are asynchronous dispatch spans, NOT independently measured CUDA kernel busy times. CUDA event timeline spans in timings.json can include host idle gaps and must not be described as kernel execution time.

## H. Graph/candidate cardinality

| Case | B | Resource cand. | Groups | Structural cand. | Cand./group | Max/group | Max/sample | Arg rows | Ref-domain members | Scope members |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| j8 | 8 | 448 | 64 | 8330 | 130.156 | 225 | 1800 | 86880 | 562320 | 17044 |
| b64_0 | 64 | 3584 | 512 | 43270 | 84.512 | 225 | 1800 | 449262 | 2908352 | 89612 |
| b64_1 | 64 | 3584 | 512 | 42274 | 82.566 | 225 | 1800 | 438786 | 2840576 | 87620 |
| b64_2 | 64 | 3584 | 512 | 42014 | 82.059 | 209 | 1582 | 436056 | 2822896 | 87100 |
| structural_heavy | 8 | 448 | 64 | 12340 | 192.812 | 225 | 1800 | 129060 | 835200 | 25064 |
| resource_only | 8 | 448 | 64 | 4680 | 73.125 | 92 | 586 | 48492 | 313952 | 9744 |
| scale_1 | 1 | 56 | 8 | 586 | 73.250 | 92 | 586 | 6072 | 39312 | 1220 |
| scale_2 | 2 | 112 | 16 | 1172 | 73.250 | 92 | 586 | 12144 | 78624 | 2440 |
| scale_4 | 4 | 224 | 32 | 2344 | 73.250 | 92 | 586 | 24288 | 157248 | 4880 |
| scale_8 | 8 | 448 | 64 | 4688 | 73.250 | 92 | 586 | 48576 | 314496 | 9760 |
| scale_16 | 16 | 896 | 128 | 9376 | 73.250 | 92 | 586 | 97152 | 628992 | 19520 |
| scale_32 | 32 | 1792 | 256 | 18752 | 73.250 | 92 | 586 | 194304 | 1257984 | 39040 |
| scale_64 | 64 | 3584 | 512 | 37504 | 73.250 | 92 | 586 | 388608 | 2515968 | 78080 |

Graph inventory per case: each cell is total / mean-per-sample / max-per-sample. Counts reflect current SMORES data, not architectural hard-coded limits.

| Case | module | connector | resource | coordinate | connection | relation | environment_objects | environment_tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| j8 | 64 / 8.000 / 8 | 256 / 32.000 / 32 | 128 / 16.000 / 16 | 256 / 32.000 / 32 | 35 / 4.375 / 7 | 168 / 21.000 / 32 | 35 / 4.375 / 10 | 88 / 11.000 / 21 |
| b64_0 | 512 / 8.000 / 8 | 2048 / 32.000 / 32 | 1024 / 16.000 / 16 | 2048 / 32.000 / 32 | 411 / 6.422 / 7 | 1896 / 29.625 / 32 | 1065 / 16.641 / 41 | 3287 / 51.359 / 150 |
| b64_1 | 512 / 8.000 / 8 | 2048 / 32.000 / 32 | 1024 / 16.000 / 16 | 2048 / 32.000 / 32 | 417 / 6.516 / 7 | 1944 / 30.375 / 32 | 1033 / 16.141 / 41 | 3201 / 50.016 / 150 |
| b64_2 | 512 / 8.000 / 8 | 2048 / 32.000 / 32 | 1024 / 16.000 / 16 | 2048 / 32.000 / 32 | 417 / 6.516 / 7 | 1864 / 29.125 / 32 | 977 / 15.266 / 41 | 3057 / 47.766 / 150 |

| Case | module | connector | resource | coordinate | connection | relation | environment_objects | environment_tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| structural_heavy | 64 / 8.000 / 8 | 256 / 32.000 / 32 | 128 / 16.000 / 16 | 256 / 32.000 / 32 | 10 / 1.250 / 4 | 256 / 32.000 / 32 | 40 / 5.000 / 5 | 168 / 21.000 / 21 |
| resource_only | 64 / 8.000 / 8 | 256 / 32.000 / 32 | 128 / 16.000 / 16 | 256 / 32.000 / 32 | 56 / 7.000 / 7 | 248 / 31.000 / 32 | 97 / 12.125 / 16 | 263 / 32.875 / 46 |

The seven scaling cases repeat the same gap sample N times; their complete per-entity inventory is in timings.json. Resource and argument/reference/scope totals, means and maxima for EVERY case are stored there. These are candidate domains, not positive label counts. Group counts include empty groups (none empty in the two internal-profile examples).

## I. Structural vs non-structural

| Batch8 case | Candidates | Argument rows | H1 ms | H2 ms | I1 ms | Backward ms | Unsplit step ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| structural_heavy | 12340 | 129060 | 15.440 | 8019.241 | 17.011 | 42.087 | 8161.205 |
| resource_only | 4680 | 48492 | 23.173 | 3032.161 | 18.293 | 48.101 | 3144.761 |

Resource-only supervision still computes the current structural candidate domain. Runtime tracks candidate/argument cardinality, not number of positive structural labels. Do not interpret this evidence as authorization for label-driven pruning or removal of validation. Temporary optimizer steps did not save weights.

## J. Batch-size scaling

| B | Train forward ms/sample | Backward ms/sample | Full step ms/sample | Peak allocated MiB |
| --- | --- | --- | --- | --- |
| 1 | 395.163 | 14.191 | 419.848 | 44.753 |
| 2 | 385.214 | 8.923 | 398.895 | 56.861 |
| 4 | 389.698 | 6.502 | 398.496 | 81.356 |
| 8 | 379.096 | 5.275 | 391.074 | 129.989 |
| 16 | 381.614 | 4.661 | 390.073 | 226.828 |
| 32 | 378.474 | 4.397 | 389.612 | 421.123 |
| 64 | 382.603 | 5.649 | 388.250 | 810.686 |

Same real TRAIN gap sample (`expert:gap:seed-004303`, transition 299) repeated N times isolates graph/candidate cardinality. This homogeneous diagnostic is not the 56+8 production distribution and is not used for epoch estimates. Full-step time per sample changes little beyond B2; increasing batching alone does not resolve the serial host work.

## K. GPU / CPU utilization

| Case | CPU % (1 core=100) | GPU activity % | Memory activity % | Peak allocated MiB | Reserved MiB | Peak RSS MiB |
| --- | --- | --- | --- | --- | --- | --- |
| j8 | 100.036 | 20.033 | 0.067 | 198.606 | 256.000 | 1659.867 |
| b64_0 | 100.032 | 20.518 | 1.129 | 949.723 | 1074.000 | 1813.711 |
| b64_1 | 100.032 | 19.059 | 0.000 | 926.458 | 980.000 | 1970.578 |
| b64_2 | 100.032 | 19.892 | 0.518 | 923.512 | 980.000 | 1970.578 |

Host/Python and host-device scalar synchronization dominate; this is not evidence of a saturated GPU compute workload. NVIDIA utilization is sampled activity, not SM occupancy; nsys/ncu were unavailable and not installed. PyTorch peak allocations, allocator reserved memory and NVIDIA process/context memory are different quantities. For J8 NVIDIA reported about 443 MiB, versus about 1261 MiB for b64_0. Sampling is descriptive and may miss brief spikes.

## L. Torch profiler top operators

UNAVAILABLE after host interruption. No operator-level CPU/CUDA result was published. `torch_profiler_top_ops.txt` documents this gap. The profiler is not rerun in this closure. Top-20 rankings, CUDA kernel busy-time totals and operator-by-operator attribution cannot be truthfully supplied. Instrumented H1/H2 region timings remain valid evidence. The computer freeze occurred after the saved scaling results; timing alone does not establish that profiler memory usage caused it. No OOM/Xid/lockup evidence was found in the checked kernel journals.

## M. Synchronization findings

- H2 `_validate_arguments`: per-row `int(CUDA_tensor)`, CUDA pointer-slice `.tolist()`, boolean `.any()`, domain `.tolist()` and bound-reference checks. These execute before filtering learnable parameters.
- H2 `_scope`: per-candidate pointer `.tolist()` and Python `if mask.all()`.
- H2 `_structural_conditions`: per-candidate CUDA boolean reads; resource/category/parameter routing also converts scalar indices.
- H1 `_packed`, `_pointer`, `_indices`, `_environment`: finite/range/owner checks and `.tolist()`/scalar conversions. These explain some H1 host cost, not the seconds in H2.
- I1 loss contains support/range/scalar checks. `train_step` has finite checks, norm conversion and family statistics; measured diagnostic conversion is ~0.3 ms, not the dominant cost.
- I2 `.cpu()`/`.tolist()`/scalar accumulation is measured separately and occurs periodically in J.
- No relevant hot-path `.numpy()` was identified. `int`/`bool` can synchronize without an explicit `.item()` call.
- Explicit profiler synchronization occurs at timing boundaries; fresh unsplit production timings avoid that component instrumentation. Event timeline spans are elapsed device timelines, not summed kernel time. No double-counted sync subtraction or speculative sync overhead number is reported.

## N. Python-loop findings

| File / function | Loop domain | J8 iterations | b64_0 iterations | Measured relevance |
| --- | --- | --- | --- | --- |
| action_heads.py:_validate_arguments | All argument rows | 86880 | 449262 | 4899 / 25189 ms inclusive |
| action_heads.py:_arguments selection | All argument rows | 86880 | 449262 | 124 / 646 ms exclusive estimate |
| action_heads.py:_scope | All structural candidates | 8330 | 43270 | 245 / 1310 ms inclusive |
| action_heads.py:_structural_conditions | All structural candidates | 8330 | 43270 | 56 / 276 ms inclusive |
| action_heads.py:forward_resource | Resource candidates / payload routing | 448 | 3584 | 4.5 / 27.5 ms whole resource path |
| action_heads.py:_parameter_conditions and category branch | Selected learnable rows | catalog/domain dependent | catalog/domain dependent | 124 / 670 ms whole category stage |
| action_heads.py:reference-set branches | Selected sets plus their members | subset of 562320 ref members | subset of 2908352 ref members | 15 / 123 ms whole reference-set stage |
| model.py:_environment | Populated semantic geometry slots | 24 | 47 | 13.5 / 25.8 ms whole environment path |
| model.py:propagate | Message blocks | 3 | 3 | ~2.8 ms combined |
| losses.py / training.py | Supported labels, families and finite checks | small support relative to candidate domain | small support relative to candidate domain | I1 ~15 / 161 ms; family diagnostics <1 ms |

Where exact selected-kind loop counts were not persisted, the domain and measured region are reported without inventing counts. Loop source inspection confirms that nonlearnable fields are validated but do not receive trainable predictions. Reference-domain member totals include repeated memberships across candidate domains, not distinct nodes.

## O. I2 / evaluation cost

| Quantity | J8 ms | Batch64 mean ms |
| --- | --- | --- |
| i2 | 115.552 | 1532.425 |
| eval_forward | 5492.216 | 27837.975 |

J cadence: every 50 optimizer steps, four periodic evaluations over 200 steps. Estimated I2-only contribution: 0.462 s (0.041% of historical wall). Most periodic evaluation cost is another slow forward. I2 does not execute inside every train_step. End-of-run extra evaluation and reload are outside the historical timer.

## P. Current full-training estimate

| Scenario | sec/step | sec/epoch | min/epoch | 10 epochs h | 50 epochs h | 100 epochs h |
| --- | --- | --- | --- | --- | --- | --- |
| production_step_only | 28.433 | 34802.537 | 580.042 | 96.674 | 483.369 | 966.737 |
| current_with_loader_transfer_and_J_eval_every_50 | 29.847 | 36533.272 | 608.888 | 101.481 | 507.407 | 1014.813 |
| excluding_periodic_eval_and_measured_family_diagnostics | 29.257 | 35809.991 | 596.833 | 99.472 | 497.361 | 994.722 |

78302 TRAIN samples, approved batch64 sampler, 1224 steps/epoch. Primary estimates use the three actual sampler batches (nine unsplit steps), not scaling/J extrapolation. No loader overlap assumed. Current scenario adds measured loader/transfer and one representative TRAIN-batch evaluation every 50 steps, using eval-forward + split-I1 proxy + I2. This is an explicit cadence scenario, not an implemented full trainer. Second scenario removes only periodic evaluation and measured family diagnostics; mandatory forward checks and training semantics remain. No hypothetical vectorization speedup is included. Full validation-set evaluation, checkpoints, scheduling, thermal drift and epoch-wide candidate mix are not benchmarked; estimates may vary accordingly.

## Q. Bottleneck ranking

| Component | Classification | Evidence |
| --- | --- | --- |
| Loader fetch | MINOR | ~0.75 s vs ~28.43 s production step |
| Collate | MINOR | ~0.065 s per batch64 |
| CPU→GPU transfer | NEGLIGIBLE | ~0.007 s |
| H1 local encoders | NEGLIGIBLE | each <0.2 ms dispatch in internal cases |
| H1 hierarchy | NEGLIGIBLE | ~0.42 ms |
| H1 message passing | NEGLIGIBLE | ~2.8 ms across three blocks |
| H1 environment relations/encoding | MINOR | ~26 ms; dominates H1, not full step |
| H1 pooling | NEGLIGIBLE | ~0.66 ms |
| H2 resource heads | MINOR | ~28 ms b64_0 internal |
| H2 structural/argument path | DOMINANT | H2 ~98% split step; validation ~89% H2 |
| I1 | MINOR | ~162 ms batch64 |
| Backward | MINOR | ~399 ms batch64 |
| Gradient clipping | NEGLIGIBLE | ~2 ms |
| Optimizer | NEGLIGIBLE | ~2 ms |
| I2 metrics | MINOR | ~1.53 s/eval; amortized over 50 steps |
| H2 scalar synchronization / Python loops | DOMINANT | nested inside validation/scope; not an additional additive cost |
| Family logging / statistics | NEGLIGIBLE | ~0.3 ms |

## R. Root cause

1. Historical J 5.615 s/step amortizes four evaluations; fresh production training is 5.496 s/step.
2. H2 consumes 98.45% of the split J step; H1 takes ~21 ms and backward ~46 ms.
3. H2 validates 86880 argument rows in J8, including executor-owned nonlearnable fields.
4. `_validate_arguments` spends ~4.90 s on repeated Python loops and CUDA scalar/domain reads.
5. At real batch64 this becomes 449262 rows and ~25.19 s validation in b64_0.
6. Three real batch64 steps average ~28.43 s; CPU uses one core at ~100%, GPU activity averages ~20%.
7. Scope and category routing add smaller Python/scalar costs; neural structural scoring dispatch is tiny.
8. Larger batches leave homogeneous per-sample step cost near 0.39 s, so batching alone does not cure it.
9. I2, optimizer and logging do not explain the multi-second inner-loop cost.
10. Main issue: host-driven contract/routing work and synchronization inside H2, not GNN matrix compute.

## S. Optimization candidates

| Optimization (proposal only) | Evidence | Expected benefit | Semantic risk | Implementation risk |
| --- | --- | --- | --- | --- |
| Vectorize argument validation | ~89% of H2 in both internal cases | Targets dominant measured region; speedup unmeasured | Medium: preserve every shape/type/range/owner/bound check and failure | Medium |
| Bulk current-input routing metadata on host / immutable prevalidated routing | Same per-row CUDA scalar reads | Alternative/complement to first proposal; gains are not additive | Medium: identity/invalidation and fail-closed provenance must remain | Medium |
| Batch optional-scope classification and scoring | ~4.5% of H2 | Removes candidate-wise pointer/boolean reads; unmeasured gain | Medium: required vs optional semantics identical | Medium |
| Batch parameter/category routing | Category ~2.4% plus exclusive selection ~2.3% of b64_0 H2 | Smaller secondary target; unmeasured gain | Medium: preserve learnable flags, bindings and output order | Low–medium |

None implemented. Preserve all current candidate domains and full input validation, including learnable=False routing fields. No label-driven pruning, architecture change, AMP/compile proposal or sampler/precision change is justified by this task. Targeted RED/GREEN and numerical equivalence are required before any future optimization.

## T. Numerical correctness

| Internal case | Output tensors | Bitwise equal | Maximum absolute difference | All finite |
| --- | --- | --- | --- | --- |
| j8 | 35 | 35 | 0.0 | True |
| b64_0 | 35 | 35 | 0.0 | True |

Saved pre-interruption checks: instrumented vs uninstrumented equivalent YES. Deterministic verification was isolated from ordinary profiling. Training semantics changed NO. These guards were recovered, not rerun after reboot. No guard exists for a completed torch operator-profiler run because none was saved.

## U. Frozen artifacts

Fresh CPU-only verification using production `load_sources()` and read-only TRAIN `SidecarDataset` passed. Canonical/F1/normalizer file hashes, G1/F2/layout semantic identities and sidecar identity match approved pins. Sidecar complete=true, 119600 total / 78302 TRAIN. J checkpoint actual SHA256 and saved selection hash match its manifest; H1/H2/I1/I2 config hashes and checkpoint upstream identity also match.

| Identity | Current verified value |
| --- | --- |
| canonical_sha | 31765272b6ecd7506c54b25e214b75d7b1b28691803ddfc286dafcce8b4ec562 |
| f1_sha | f6759e6af89d732078ac656133d40236e1fd3cec8b43c92708907ed90ac965c7 |
| g1_fingerprint | 56533e91e1d19404ac22c3aae260466ed8c2ef1ca5cdb89476b7098036fb5d0c |
| f2_fingerprint | 890eca0b9f6d4deca2cac81d3a41bf535aa3d03ac16e0ff4b81cbc0ca163abf5 |
| normalizer_sha | ea8efbd9062acb8625e0618bcc5f86b975f20839bf3132a684a626c744e52003 |
| layout_fingerprint | e5b6a313062746a68bc5c1ba43cd200479af15f1b9d1e3cee24a1675a90deb49 |
| split_fingerprint | aee975a2f8b044ad3f57f331fc659d6cfa91d7c1e493b90b812634fb6364e81b |
| feature_schema_fingerprint | 43ab1b93c21b2645176e614ab1af7437f37163ec2b6047c0671d18a354b67d57 |
| robot_spec_fingerprint | c6e1c78d3fe5f8ede3c1d7e8ab5957799d0952580a171437db7ffcddc092ae19 |
| tensorizer_fingerprint | 61e64fb426197b8f1926afad1856067aa3bb07273a05010cde2734dc420f9fda |
| storage_fingerprint | 9a5ae40bf224ae53901937883622d2c26935a47def2c93dd5c9c5866e4465bca |
| sidecar_manifest | 7bb9f77df7eefe7c3b0246cea10383ec484274093514b59294ea7d752458052a |
| J checkpoint SHA256 | 84d88d4332c65bd7f7c57f024668d4eb021647272519884505d1df2768e65e47 |

Scope of evidence: current approved identity compatibility and checkpoint bytes are verified; this is not a fresh full scan of all multi-GiB dataset payloads. The original 163-file pre-profile stat snapshot and original all-file hash baseline were in /tmp and are unavailable. Therefore exhaustive byte-for-byte preservation across the interrupted phase is not claimed. A fresh recovery baseline checks all 136 production/config/docs/tests/J files through report publication.

## V. Git audit

Branch: `wip/t8-c05-human-teleop-20260921`. No staging, commit, push, reset, restore, clean or branch switch. Seven tracked historical dirty files remain (782 insertions / 1635 deletions); unrelated dirty content is preserved during recovery. This task creates diagnostic artifacts only. Original pre-profile baseline was lost, so exact cross-interruption comparison of every untracked source file is unavailable. Recovery-phase hashes/status and diff-check are freshly verified below.

```text
 .../execution/parallel_assembly_executor.py        |   81 +-
 .../nodes/smores_parallel_self_assembly_node.py    |   14 +
 .../test/test_parallel_assembly_executor.py        |   41 +
 scripts/dataset_tools/build_expert_v1.py           |    3 +
 scripts/smores_ep/run_button_expert_to_ik.py       | 2013 +++++---------------
 .../src/smores_ep/control/button_ik_math.py        |   55 +
 .../tests/test_button_ik_source_contract.py        |  210 +-
 7 files changed, 782 insertions(+), 1635 deletions(-)
```

Final recovery audit PASS: 136 existing files unchanged, original timings hash unchanged, git diff --check exit 0, branch and tracked diff unchanged, 3 new audit-only files, no removed status entries. Staged paths: none. Git status has 368 entries (intentional dirty workspace); status SHA256: 75eedb2977d9d6b72b9bd3a2d7a8842e7baac2e87b295f2054122a0b0644e726.

?? artifacts/bc_v1/audits/post_j_performance_v1/report.md
?? artifacts/bc_v1/audits/post_j_performance_v1/summary.json
?? artifacts/bc_v1/audits/post_j_performance_v1/torch_profiler_top_ops.txt

## W. Remaining blockers

The measured throughput is the concrete practical blocker to affordable full training: roughly ten hours per epoch with current serial loader and J-like evaluation cadence. No numerical/checkpoint/identity failure was found. The full trainer/HPO remain outside this task. Missing torch-profiler operators and lost scratch baseline limit audit completeness; they do not invalidate recovered region timings, and no repeat profiling is launched here.

## X. Gate

Recovered diagnosis and measurements are ready for review with the operator-profile and baseline limitations disclosed. Original full requested profiling completeness: NO. No full BC-v1 training started.

POST-J PERFORMANCE PROFILE READY FOR REVIEW: YES
