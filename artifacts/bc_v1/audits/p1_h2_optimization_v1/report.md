# P1 — H2 bulk argument validation

One targeted production intervention only. No Step B, K1, full training, operator profiler, candidate pruning or upstream rebuild. Performance timing uses CUDA float32, AMP OFF, deterministic algorithms OFF, one CPU thread, seed 20261006. Equivalence alone uses the temporary deterministic verification context.

## A. Root cause confirmation

Legacy `_validate_arguments` repeats Python int/bool conversions, pointer/domain `.tolist()` and CUDA reductions per argument row. b64_0 has 449262 argument rows and 2908352 reference-domain memberships. Fresh focused probe: CUDA 25.925 s vs CPU 4.245 s (one warm-up + two observations). Original region profiling measured 25.189 s on CUDA. The measured issue is repeated host-device synchronization and Python routing, not neural scoring compute.

## B. Files changed

Production:

- `drl/IL/bc_v1/action_heads.py`

Tests:

- `drl/IL/tests/bc_v1/test_action_heads_bulk.py`
- `drl/IL/tests/bc_v1/h2_legacy_reference.py` (test-only pre-P1 validator oracle)

Audit only:

- `artifacts/bc_v1/audits/p1_h2_optimization_v1/baseline.json`
- `artifacts/bc_v1/audits/p1_h2_optimization_v1/step_a.json`
- `artifacts/bc_v1/audits/p1_h2_optimization_v1/postfix.json`
- `artifacts/bc_v1/audits/p1_h2_optimization_v1/equivalence.json`
- `artifacts/bc_v1/audits/p1_h2_optimization_v1/report.md`

One-off measurement scripts/logs live under `/tmp/mssr-p1-h2`; the original H2 source is retained in baseline.json for audit recovery. No other production/config file changed.

## C. Design of the fix

`_validate_arguments` delegates to a single bulk implementation. Shape/dtype/pointer checks run first. Catalog-sized action/kind lookup tensors check ALL current argument rows at once. Pointer expansion maps flat domain members to argument owners; bulk range and graph-ownership predicates validate references. Integer scatter accumulates matches to verify each fixed bound reference belongs to its own domain. Nonreference rows must still have empty domains and bound=-1.

Every batch-dependent predicate is recomputed every call, including learnable=False fields. No validation cache, stale identity, trusted-sidecar exception or routing plan is introduced. The existing RobotSpec construction validation remains the static check; no dynamic checks moved out of the forward. Learned tensors remain on CUDA; the optimized dynamic validation also runs on the current input device. Small catalog lookups contain semantic types/action indices, never learned physical IDs. No change to scope, category routing, structural conditions, H1, loss, optimizer or device transfer. AST/source comparison verifies every other legacy ActionHeads method is unchanged.

## D. Semantic invariants preserved

| Invariant | Result |
| --- | --- |
| Candidate count | Preserved |
| Candidate order | Preserved |
| Parameter order | Preserved |
| Reference domains | Preserved |
| Ownership | Preserved |
| Bindings / fixed connectors | Preserved |
| learnable flags | Preserved |
| Required / optional scope | Preserved |
| Structural groups = (sample, initiator) | Preserved |
| SETTLE endpoint-free | Preserved |
| Physical IDs routing only | Preserved |
| No supervision/target/stage/future used for acceleration | Preserved |

Six-case cardinalities agree with saved pre-P1 measurements; both implementations consume exactly the same current candidate tensors. Complete output-tree comparison includes routing, grouping and ordering. Validation creates temporary tensors and does not modify source inputs.

## E. RED → GREEN evidence

| Behavior / test | RED or characterization | GREEN |
| --- | --- | --- |
| test_bulk_validator_scalar_reads_are_bounded_not_per_argument_row | Observed RED: AssertionError Missing bulk H2 validator; 1 failed in 1.02 s | PASS on CPU and CUDA; bounded scalar counts independent of row multiplicity |
| All argument kinds, bound/nonlearnable fields, empty sets, optional/required scope, endpoint-free skill | 17 characterization tests PASS on legacy (15 invalid mutations included) | Retained in final targeted/full suites |
| Mutation after successful validation, nonlearnable routing | Legacy oracle rejects mutated kind; existing contract characterization | New path rechecks and rejects; no stale authorization |

The regression uses TorchDispatchMode to count `_local_scalar_dense` operations through the production validator entry, with 1 vs 8 copies of the same generic fixture. It is structural, not a wall-clock unit test. At most 32 scalar predicate reads are allowed per call, independent of argument rows. No seconds-long performance benchmark enters pytest.

## F. Invalid input equivalence

| Invalid case | Before | After | Fail closed |
| --- | --- | --- | --- |
| nonmonotonic_pointer | Rejected | Rejected | YES |
| pointer_out_of_range | Rejected | Rejected | YES |
| index_out_of_range | Rejected | Rejected | YES |
| wrong_owner | Rejected | Rejected | YES |
| nonreference_domain | Rejected | Rejected | YES |
| bound_outside_domain | Rejected | Rejected | YES |
| invalid_category_kind | Rejected | Rejected | YES |
| invalid_binding | Rejected | Rejected | YES |
| wrong_action_binding | Rejected | Rejected | YES |
| invalid_required_mask | Rejected | Rejected | YES |
| invalid_scope_member | Rejected | Rejected | YES |
| cross_graph_scope | Rejected | Rejected | YES |
| shape_mismatch | Rejected | Rejected | YES |
| dtype_mismatch | Rejected | Rejected | YES |
| missing_routing | Rejected | Rejected | YES |

Scope checks remain in structural conditions. The preserved test-only legacy validator supplies the argument rejection oracle. Missing routing remains an exception rather than a silent fallback. Existing I1 tests retain invalid categorical-label checks; H2 kind/category routing corruption is covered above. Valid domains with zero members remain valid where previously allowed.

## G. Forward equivalence

| Case | Output tensors | Bitwise equal | Max abs diff | assert_close | Candidate count |
| --- | --- | --- | --- | --- | --- |
| j8 | 35 | 35 | 0.0 | PASS | Same |
| b64_0 | 35 | 35 | 0.0 | PASS | Same |
| b64_1 | 35 | 35 | 0.0 | PASS | Same |
| b64_2 | 35 | 35 | 0.0 | PASS | Same |
| structural_heavy | 35 | 35 | 0.0 | PASS | Same |
| resource_only | 35 | 35 | 0.0 | PASS | Same |

All keys/shapes/dtypes and ordering match, using rtol=1e-6/atol=1e-7 with zero actual differences. Both branches use identical initial weights and immutable current inputs. Legacy full ActionHeads is instantiated from the pre-edit source snapshot, not approximated by a changed validator.

## H. Gradient / train-step equivalence

| Case | I1 old / new | All present gradients equivalent | All bitwise |
| --- | --- | --- | --- |
| j8 | 9.093771935 / 9.093771935 | True | True |
| b64_0 | 13.258290291 / 13.258290291 | True | True |

Nonzero finite representative gradients verified for H1 module encoder/fusion and H2 structural scorer, resource payload decoder and argument category scorer. All present parameter gradients were compared, beyond representatives.

Single train-mode J8 update with identical initial parameters, populated identical AdamW state and restored RNG (dropout 0.1): loss identical; 264/264 parameter-delta tensors and 774/774 optimizer-state tensors bitwise identical, including counters. No J checkpoint was used as an output or overwritten.

## I. Performance — `_validate_arguments`

| Case | Before ms | After ms | Speedup |
| --- | --- | --- | --- |
| j8 | 4899.101 | 1.033 | 4741.619x |
| b64_0 | 25188.942 | 4.941 | 5097.500x |

Old internal-validator timings exist only for J8 and b64_0; no before values are invented for other cases. Post-fix validators (five repeats each) for all six cases are stored in postfix.json. The fresh focused legacy b64_0 probe independently corroborates the old ~25.9 s scale. Gate >=10x is satisfied without removing validation.

## J. Performance — H2

| Case | Before ms | After ms | Speedup |
| --- | --- | --- | --- |
| j8 | 5407.587 | 584.465 | 9.252x |
| b64_0 | 28318.013 | 2918.220 | 9.704x |
| b64_1 | 28093.074 | 2857.822 | 9.830x |
| b64_2 | 27506.699 | 2848.692 | 9.656x |
| structural_heavy | 8019.241 | 852.216 | 9.410x |
| resource_only | 3032.161 | 346.469 | 8.752x |

These are synchronized component timings. Primary end-to-end conclusions use the unsplit production step below.

## K. Performance — full train step

| Case | Before ms | After mean ms | Median | Min | Max | Speedup |
| --- | --- | --- | --- | --- | --- | --- |
| j8 | 5495.762 | 682.264 | 664.780 | 662.766 | 727.290 | 8.055x |
| b64_0 | 28764.032 | 3564.963 | 3576.570 | 3471.913 | 3616.525 | 8.069x |
| b64_1 | 28399.925 | 3504.477 | 3532.686 | 3445.103 | 3550.375 | 8.104x |
| b64_2 | 28136.379 | 3493.443 | 3491.835 | 3485.841 | 3502.527 | 8.054x |
| structural_heavy | 8161.205 | 922.381 | 916.150 | 908.800 | 941.910 | 8.848x |
| resource_only | 3144.761 | 442.957 | 438.367 | 436.590 | 456.943 | 7.099x |

**Three real batch64 mean: 3.520961 s vs 28.433445 s = 8.075x speedup.** Five full-step warm-ups and five measured real production train_step calls per case; warm-up excluded. Final source is the same Step A implementation measured here; no further production optimization occurred. The <=4.0 s gate passes; stretch <=2.5 s does not. No Step B authorized by the conditional gate was needed.

| Batch64 | Fetch ms | Collate ms | Transfer ms | H1 ms | H2 ms | I1 ms | Backward ms | Clip ms | Optimizer ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| b64_0 | 824.940 | 66.480 | 7.232 | 36.550 | 2918.220 | 168.166 | 421.024 | 2.736 | 1.943 |
| b64_1 | 3641.289 | 68.992 | 7.215 | 36.199 | 2857.822 | 173.594 | 404.466 | 3.331 | 2.118 |
| b64_2 | 3254.536 | 59.454 | 7.164 | 37.492 | 2848.692 | 169.865 | 411.523 | 3.032 | 1.999 |

All measured observations and mean/median/min/max/p95 are retained in step_a.json and postfix.json. Five-repeat p95 is descriptive, not a confidence bound. Sampler indices replay the exact saved 56+8, seed-20261006 batches; no candidate/label filtering or new TRAIN census.

## L. Post-fix bottleneck

| Current b64_0 H2 region | Mean ms |
| --- | --- |
| _validate_arguments | 5.202 |
| _arguments | 1491.767 |
| _scope | 1295.732 |
| _structural_conditions | 294.338 |
| forward_resource | 28.465 |

Argument routing/prediction (~1.49 s inclusive) and scope (~1.30 s) now dominate H2; validation is only ~5 ms. These are nested region wall timings with synchronized boundaries, not CUDA kernel busy-time attribution. No torch operator profiler was run. Given the <=4 s mean gate, scope/category routing remain unchanged. Backward is now ~0.41 s, material relative to the shorter step.

## M. CPU/GPU utilization post-fix

| Case | CPU % (1 core=100) | GPU activity % | Peak allocated MiB | Peak reserved MiB | Peak RSS MiB |
| --- | --- | --- | --- | --- | --- |
| b64_0 | 100.038 | 25.500 | 954.719 | 1132.000 | 1831.816 |
| b64_1 | 100.039 | 18.167 | 931.129 | 1064.000 | 1860.785 |
| b64_2 | 100.026 | 30.444 | 927.704 | 1062.000 | 1860.785 |

Old batch64_0: CPU ~100%, sampled GPU activity ~20.5%, allocated ~950 MiB, reserved ~1074 MiB, peak RSS ~1814 MiB. New allocations remain ~928–955 MiB with process peak RSS <=1861 MiB. No large host routing copy/plan is stored. Remaining host loops still use one CPU core; sampled GPU activity is not SM occupancy. Sampling windows are short and vary by case; no utilization-based hardware speedup is claimed.

## N. New full-training estimate

| Scenario | sec/step | min/epoch | h/epoch | 10 epochs h | 50 epochs h | 100 epochs h |
| --- | --- | --- | --- | --- | --- | --- |
| train_step_only | 3.521 | 71.828 | 1.197 | 11.971 | 59.856 | 119.713 |
| loader_collate_transfer_included | 6.167 | 125.801 | 2.097 | 20.967 | 104.834 | 209.669 |

78302 TRAIN samples, batch64, 1224 steps/epoch. Includes only actual measured speedups. Loader+collate+transfer uses fresh once-per-batch observations: fetch 0.825 / 3.641 / 3.255 s, notably variable and higher than earlier warmed-profile observations; no loader behavior was modified. No overlap assumed. Evaluation/checkpoint/HPO orchestration costs are excluded. These three replayed batches are not an exhaustive epoch benchmark.

## O. K1 practicality

K1 throughput now practical: YES, with bounded trial budgets. The measured training step is ~8.08x faster and an epoch of optimizer work is ~1.20 h; the conservative current loader-inclusive estimate is ~2.10 h/epoch. This permits small tuning trials with explicit budgets, not an unbounded sweep. It does not imply that 8–15 long final-length runs would be inexpensive. K1 was not started.

## P. Targeted tests

Fresh command:

```text
/home/lorenzo/.local/share/mssr-bc-v1/venv/bin/python -B -m pytest drl/IL/tests/bc_v1/test_action_heads_bulk.py drl/IL/tests/bc_v1/test_action_heads.py drl/IL/tests/bc_v1/test_losses.py drl/IL/tests/bc_v1/test_model.py drl/IL/tests/bc_v1/test_training.py -q
```

163 passed in 13.05s. Process wall: 14.246 s.

Earlier Step A targeted run: 161 PASS in 23.91 s. Final run additionally includes CUDA scalar-read regression and mutated nonlearnable routing. Tensorization/batching were not changed.

## Q. Full BC-v1 regression

Fresh command:

```text
/home/lorenzo/.local/share/mssr-bc-v1/venv/bin/python -B -m pytest drl/IL/tests/bc_v1 -q
```

1616 passed, 1 skipped in 166.76s (0:02:46). Process wall: 168.159 s. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1. This is a new complete BC-v1 run, not a reused J result.

## R. Frozen artifacts

| Artifact | Verification |
| --- | --- |
| Canonical | UNCHANGED |
| F1 | UNCHANGED |
| G1 | UNCHANGED |
| F2 | UNCHANGED |
| Normalizer | UNCHANGED |
| Resolved layout | UNCHANGED |
| G2 sidecar | UNCHANGED |
| J checkpoint | UNCHANGED |

Production load_sources and read-only SidecarDataset verify approved SHA/semantic identities. Sidecar complete=true, 119600 samples / 78302 TRAIN. J checkpoint actual SHA256 matches 84d88d4332c65bd7f7c57f024668d4eb021647272519884505d1df2768e65e47. All 165 frozen file size/mtime snapshots are unchanged; no multi-GiB payload rebuild or census was run. Baseline source/config/docs/J hashes are unchanged except authorized action_heads.py.

## S. Git audit

Branch unchanged: `wip/t8-c05-human-teleop-20260921`. Historical dirty status and tracked diff preserved; the seven existing tracked changes are unchanged. New task source/tests are untracked in this intentionally dirty workspace, so their full source was reviewed and compared to baseline. `git diff --check` exit 0. Explicit --no-index --check of all three untracked source/test files also produced no whitespace diagnostics. No staging, commit, push, reset, restore, clean or branch switch.

Task file list is exact in section B. Production method comparison verifies the validator is the only region changed.

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

## T. Remaining issues

The <=2.5 s stretch target is not reached. Residual H2 argument routing/scope loops and variable loader fetch remain measured costs; no further optimization was performed because the mandatory gate passes. Full epoch/validation/checkpoint throughput has not been measured, and K1/full trainer remain outside this task. No semantic, gradient, invalid-input, regression or frozen-artifact failure remains.

P1 H2 OPTIMIZATION READY FOR REVIEW: YES
