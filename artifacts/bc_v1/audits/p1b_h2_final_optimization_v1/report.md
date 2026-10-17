# P1b — final H2 scope and argument routing pass

Formal performance gate PASS: 1.032370 s mean full production train_step on the three real batch64 cases; fresh pre-P1b baseline 3.338462 s. Exactly two production interventions. No K1/HPO/full training/operator profiler/upstream rebuild.

## A. Fresh baseline

| Case | Scope ms | Argument path ms | H2 ms | Full unsplit step ms |
| --- | --- | --- | --- | --- |
| b64_0 | 1062.639 | 1400.933 | 2798.126 | 3375.577 |
| b64_1 | 1059.657 | 1380.305 | 2702.855 | 3340.532 |
| b64_2 | 1046.289 | 1402.798 | 2693.852 | 3299.278 |

All six cases were measured afresh before production edits; five full-step warm-ups and five measured production steps per case, plus synchronized components and three direct scope/argument-path measurements. CUDA RTX4070 Laptop, PyTorch2.14.1+cu130/runtime13.0, float32, AMP OFF, deterministic timing OFF, seed20261006, CPUthreads1. Exact saved 56+8 sampler draws and J8 keys replayed; no new TRAIN census. Baseline contains source/hash/status snapshot and raw observations, not just historical P1 numbers.

## B. Files changed

Production: `drl/IL/bc_v1/action_heads.py`.

Tests: `drl/IL/tests/bc_v1/test_action_heads_routing.py`, `drl/IL/tests/bc_v1/h2_scope_routing_reference.py`. Existing P1 tests/oracle are unchanged.

Audit files in this directory: baseline.json, step_a_scope.json, step_b_routing.json, postfix.json, equivalence.json, equivalence_step_a.json, post_step_a_routing_breakdown.json, post_step_b_routing_breakdown.json, final_h2_breakdown.json, report.md. Scratch scripts/logs under /tmp/mssr-p1b-h2. No other production/config/spec changes.

## C. Step A — scope root cause

43,270 candidate scope visits in b64_0, 89,612 scope memberships. Each legacy candidate read its pointer slice with .tolist() and mask.all() as a Python boolean, including fixed/empty scopes. Only variable scopes require neural scoring. Hypothesis was recorded before patch: derive all variable-scope routing in bulk, retaining identical per-variable-scope neural call shapes/order and dropout RNG consumption.

## D. Step A — implementation

Current pointers expand to member owners; integer scatter counts optional members, masks select variable bindings and flat optional/required positions, and cumulative sums produce output pointers. Only three bounded host list transfers provide variable binding/boundary metadata. The remaining loop performs the same neural calls on the same per-variable-scope batches in the same order. No candidate/domain/duplicate member removal.

All existing pointer/range/graph-owner/mask checks remain in unchanged structural conditions before scope scoring; a missing variable-scope scorer still raises. Routing is recomputed each call. Invalid input paths remain fail-closed; no trusted-cache shortcut is used.

## E. Step A RED/GREEN

Characterization before: 11 passed, 2 deselected in 1.45s.

RED: AssertionError [51,408] scalar reads; expected <=16; 1 failed in 1.73s. The 51→408 scalar reads track candidate count.

GREEN: 70 passed in 6.49s. Persistent bounded-scalar tests pass on both CPU and CUDA; characterization covers empty/fixed/optional/required scopes, duplicates, multiple samples/initiators, candidate permutation and malformed scope routing.

## F. Scope performance

| Case | Fresh before ms | After Step A ms | Step A speedup | Final ms | Final speedup |
| --- | --- | --- | --- | --- | --- |
| j8 | 242.777 | 4.739 | 51.227x | 4.783 | 50.756x |
| b64_0 | 1062.639 | 38.749 | 27.424x | 37.802 | 28.111x |
| b64_1 | 1059.657 | 38.381 | 27.609x | 38.024 | 27.868x |
| b64_2 | 1046.289 | 38.153 | 27.423x | 37.356 | 28.009x |
| structural_heavy | 355.094 | 4.810 | 73.823x | 4.784 | 74.226x |
| resource_only | 141.011 | 4.795 | 29.405x | 4.863 | 28.998x |

## G. Post-Step-A bottleneck

| Case | Scope ms | Arguments ms | H2 ms | Full step ms |
| --- | --- | --- | --- | --- |
| b64_0 | 38.749 | 1345.629 | 1613.459 | 2199.888 |
| b64_1 | 38.381 | 1320.177 | 1582.865 | 2198.613 |
| b64_2 | 38.153 | 1321.219 | 1577.752 | 2195.134 |

Mean after scope-only: 2.197878 s, above 1.50 s. Scope-only equivalence was 35/35 bitwise tensors on all six cases before Step B. Argument path remained dominant.

| Measured routing region b64_0 | Post-A ms |
| --- | --- |
| selection_learnable_flags_and_bound_binding_dispatch | 632.104 |
| parameter_condition_numeric | 3.956 |
| numeric | 7.905 |
| parameter_condition_reference_set | 3.548 |
| reference_set | 114.890 |
| parameter_condition_reference | 0.028 |
| reference | 0.316 |
| parameter_condition_boolean | 0.024 |
| boolean | 0.043 |
| parameter_condition_category | 303.249 |
| category | 656.081 |

Selection, learnable flags and bound-binding dispatch share the same old loop and were timed jointly; no artificial per-predicate timings are invented. Per-kind regions include their parameter-condition subregion, which must not be added twice. Instrumentation numerical guard:35/35 bitwise, maxdiff0.

## H. Step B — routing root cause

After scope-only, joint selection/learnable/binding dispatch cost632ms and category processing656ms, including303ms parameter conditions. These dominated the ~1.4 s argument path. Reference-set ~115ms and numeric ~8ms were much smaller. Both main loops repeatedly converted current CUDA parameter/bound scalars to Python. Step B was started only after scope semantic/performance gates and the new breakdown passed.

## I. Step B — implementation

Bulk learnable/kind/bound masks select rows in original order. A small catalog lookup gathers parameter-conditioning indices in bulk. Category widths/prefix pointers expand row/category roles with repeat-interleave and integer offsets, preserving catalog category order exactly. No learned operation, neural layer, call shape/order or dropout sequence changes.

Numeric/reference/reference-set/boolean scoring and their small domain traversals remain unchanged; they reuse the bulk parameter-condition lookup. Static role maps derive from validated RobotSpec catalog metadata. All owner/binding/domain/topology-dependent routing derives from the current batch each call; no cache, supervision or physical-ID feature is introduced. P1 bulk validation is source-identical and still validates every argument, including learnable=False fields.

## J. Step B RED/GREEN

Characterization before: 4 passed, 15 deselected in 1.13s.

RED: AssertionError Missing bulk argument selection; 1 failed in 0.93s.

GREEN: 238 passed in 14.68s. Covers all eight generic argument kinds, bound/nonlearnable routing, required set members, empty set and single-reference domains, canonical category widths/order, plus existing parameter/candidate permutation tests. Structural bounded-scalar tests protect selection, parameter conditioning and category expansion without wall-time thresholds. ParameterSpec has no separate optional-parameter flag in the current ABI; existing bound/required domain semantics are preserved, not redefined.

## K. Argument routing performance

| Case | Fresh before ms | Final ms | Speedup |
| --- | --- | --- | --- |
| j8 | 268.660 | 17.395 | 15.445x |
| b64_0 | 1400.933 | 124.292 | 11.271x |
| b64_1 | 1380.305 | 124.319 | 11.103x |
| b64_2 | 1402.798 | 123.798 | 11.331x |
| structural_heavy | 391.734 | 17.919 | 21.861x |
| resource_only | 156.444 | 17.090 | 9.154x |

Post-Step-B detailed profile:

| Region b64_0 | Post-A ms | Final ms |
| --- | --- | --- |
| selection_learnable_flags_and_bound_binding_dispatch | 632.104 | 0.353 |
| parameter_condition_numeric | 3.956 | 0.136 |
| numeric | 7.905 | 4.043 |
| parameter_condition_reference_set | 3.548 | 0.113 |
| reference_set | 114.890 | 113.465 |
| parameter_condition_reference | 0.028 | 0.054 |
| reference | 0.316 | 0.330 |
| parameter_condition_boolean | 0.024 | 0.042 |
| boolean | 0.043 | 0.062 |
| parameter_condition_category | 303.249 | 0.596 |
| category | 656.081 | 5.048 |

The argument-path timer includes preserved P1 validation (~5ms); no semantics are skipped. Required/optional reference-set and boolean paths remain computed even without target support. No operator profiler was used.

## L. Forward equivalence

| Case | Output tensors | Bitwise equal | Max diff | assert_close | Candidate count | Scope membership count |
| --- | --- | --- | --- | --- | --- | --- |
| j8 | 35 | 35 | 0.0 | PASS | 8330 unchanged | 17044 unchanged |
| b64_0 | 35 | 35 | 0.0 | PASS | 43270 unchanged | 89612 unchanged |
| b64_1 | 35 | 35 | 0.0 | PASS | 42274 unchanged | 87620 unchanged |
| b64_2 | 35 | 35 | 0.0 | PASS | 42014 unchanged | 87100 unchanged |
| structural_heavy | 35 | 35 | 0.0 | PASS | 12340 unchanged | 25064 unchanged |
| resource_only | 35 | 35 | 0.0 | PASS | 4680 unchanged | 9744 unchanged |

Same keys, shapes, dtypes, order, grouping and current domains. Pre-P1b full ActionHeads is loaded from the before-edit source snapshot with identical weights. Deterministic verification only; no tolerances changed (rtol1e-6/atol1e-7), actual differences zero. Step A separately passed the same six-case check before Step B.

## M. Invalid input equivalence

| Invalid case | Before | After | Fail closed |
| --- | --- | --- | --- |
| nonmonotonic pointer | Rejected | Rejected | YES |
| pointer out of range | Rejected | Rejected | YES |
| index out of range | Rejected | Rejected | YES |
| wrong ownership | Rejected | Rejected | YES |
| wrong/nonreference domain | Rejected | Rejected | YES |
| bound outside domain | Rejected | Rejected | YES |
| invalid category kind | Rejected | Rejected | YES |
| invalid binding | Rejected | Rejected | YES |
| wrong action binding | Rejected | Rejected | YES |
| invalid required mask | Rejected | Rejected | YES |
| scope invalid member | Rejected | Rejected | YES |
| scope cross-graph | Rejected | Rejected | YES |
| shape mismatch | Rejected | Rejected | YES |
| dtype mismatch | Rejected | Rejected | YES |
| missing routing | Rejected | Rejected | YES |
| scope pointer | Rejected | Rejected | YES |
| scope required-mask shape/dtype | Rejected | Rejected | YES |
| scope routing missing | Rejected | Rejected | YES |

All P1 malformed corpus tests are retained. New scope-specific cases compare pre-P1b oracle and new complete structural path; empty/duplicate valid domains retain their original behavior. Mutated nonlearnable routing after successful validation is still rechecked.

## N. Gradient / optimizer equivalence

| Case | I1 old/new | Family losses | All present gradients |
| --- | --- | --- | --- |
| j8 | 9.093771935/9.093771935 | 9 equivalent | Bitwise identical |
| b64_0 | 13.258290291/13.258290291 | 9 equivalent | Bitwise identical |

All nine family losses/supports compared, including support-zero behavior. Supported representative H1 module encoder/fusion and H2 structural/payload/category gradients are finite, nonzero and bitwise equal.

J8 single train-mode step uses identical populated AdamW state, model state and RNG with dropout0.1 unchanged:264/264 parameter-delta tensors and774/774 optimizer-state tensors bitwise equal, including step counters, exp_avg and exp_avg_sq. J checkpoint remains read-only.

## O. Final H2 performance

| Case | P1 fresh ms | P1b ms | Speedup |
| --- | --- | --- | --- |
| j8 | 569.686 | 84.540 | 6.739x |
| b64_0 | 2798.126 | 478.259 | 5.851x |
| b64_1 | 2702.855 | 471.705 | 5.730x |
| b64_2 | 2693.852 | 470.873 | 5.721x |
| structural_heavy | 813.265 | 112.163 | 7.251x |
| resource_only | 330.013 | 61.890 | 5.332x |

## P. Final full-step performance

| Case | P1 fresh ms | P1b mean ms | Median | p95 | Min | Max | Speedup |
| --- | --- | --- | --- | --- | --- | --- | --- |
| j8 | 651.268 | 169.137 | 169.114 | 171.789 | 166.758 | 172.616 | 3.851x |
| b64_0 | 3375.577 | 1040.916 | 1039.804 | 1049.323 | 1033.920 | 1052.633 | 3.243x |
| b64_1 | 3340.532 | 1030.734 | 1029.210 | 1037.885 | 1025.462 | 1040.430 | 3.241x |
| b64_2 | 3299.278 | 1025.461 | 1024.754 | 1031.275 | 1021.010 | 1031.533 | 3.217x |
| structural_heavy | 899.568 | 187.064 | 186.633 | 189.282 | 185.526 | 189.687 | 4.809x |
| resource_only | 423.667 | 154.599 | 154.471 | 159.604 | 151.308 | 162.333 | 2.740x |

**Mean real b64_0/b64_1/b64_2: 1.032370 s**, fresh baseline 3.338462 s (3.234x). Historical P1 3.521 s is context only; fresh baseline is the primary comparison. Five warm-ups and10 measured unsplit real train_step calls per case (30 measured batch64 steps); synchronized CUDA completion, float32, AMP/determinism OFF. Gate1.50s passes; stretch1.00s does not. All raw observations are in postfix.json; p95 is descriptive for these small repeated-batch samples.

## Q. Final component breakdown

| Component | Batch64 mean ms | % unsplit step |
| --- | --- | --- |
| h1 | 33.930 | 3.287 |
| h2 | 473.612 | 45.876 |
| i1 | 149.686 | 14.499 |
| backward | 367.094 | 35.558 |
| clip | 2.492 | 0.241 |
| optimizer | 1.985 | 0.192 |

Components are separately synchronized measurements; minor differences versus unsplit mean reflect instrumentation/timing variation. Not all percentages must sum exactly to100.

H2 internal b64_0 on cached eval latents:

| Region | Mean ms |
| --- | --- |
| resource | 28.090 |
| structural_conditions | 289.472 |
| structural_scorer | 0.449 |
| argument_validation | 5.050 |
| argument_path_including_validation | 129.786 |
| scope | 35.615 |

Argument path includes validation. Scorer timing includes synchronized device completion; these are elapsed region times, not an operator/kernel trace.

## R. New bottleneck

H2 (~474ms) and backward (~367ms) are now comparable main compute costs; I1 is ~150ms. Inside H2 the unchanged structural conditions (~289ms) dominate, followed by argument/reference-set work (~130ms). Scope is ~36–38ms and argument validation ~5ms. No third major production patch is attempted.

## S. CPU/GPU/memory

| Case | CPU % (1 core=100) | GPU activity % | Peak allocated MiB | Peak reserved MiB | Peak RSS MiB |
| --- | --- | --- | --- | --- | --- |
| b64_0 | 100.045 | 27.091 | 957.045 | 1132.000 | 1828.250 |
| b64_1 | 100.040 | 65.182 | 932.807 | 1064.000 | 1853.469 |
| b64_2 | 100.041 | 62.545 | 929.521 | 1062.000 | 1853.469 |

P1 activity was18–30% and CPU~100% of one core. Current short sampled GPU windows vary27–65%; CPU remains~100% because remaining host work exists. Measured compute wall time is sharply reduced, not converted to a claim of saturated GPU occupancy. Peak allocations remain below1GiB; reserved~1.1GiB, RSS<1.9GiB. No huge host cache or domain-sized dense candidate matrix is introduced.

## T. Training estimate

Compute only (78302 TRAIN, batch64,1224steps/epoch):

| sec/step | 1 epoch min | 1 epoch h | 10 epochs h | 50 epochs h | 100 epochs h |
| --- | --- | --- | --- | --- | --- |
| 1.032 | 21.060 | 0.351 | 3.510 | 17.550 | 35.101 |

End-to-end including current measured loader/collate/transfer, no overlap:

| sec/step | 1 epoch min | 1 epoch h | 10 epochs h | 50 epochs h | 100 epochs h |
| --- | --- | --- | --- | --- | --- |
| 1.886 | 38.481 | 0.641 | 6.414 | 32.068 | 64.135 |

Loader observations:

| Case | Fetch ms | Collate ms | CPU→GPU ms |
| --- | --- | --- | --- |
| b64_0 | 769.112 | 62.007 | 7.485 |
| b64_1 | 786.542 | 59.015 | 7.136 |
| b64_2 | 804.626 | 58.823 | 7.127 |

Loader observations are once per replayed batch and currently warm (~0.77–0.80s fetch); prior P1 had0.825–3.641s variability. No loader change or overlap is assumed. These three batches do not characterize cold/random epoch-wide I/O; full validation/checkpoint/HPO orchestration costs are excluded. No hypothetical speedup is included.

## U. K1 practicality

K1 compute throughput practical: YES — 21.06 min/epoch, below the ~30min compute objective.

K1 end-to-end throughput practical: YES for bounded trial budgets under the measured loader scenario — 38.48 min/epoch. Loader variability remains a separate investigation; this is not a guarantee for a full cold epoch or long sweep. K1/HPO/full training were not started.

## V. Targeted tests

238 PASS /0 SKIP /0 FAIL in14.68s, fresh after Step B. Covers action_heads, P1 bulk, new routing/scope, model, losses, metrics, batching and training. Exact command saved in postfix.json. Step A:70PASS/6.49s. Performance benches live only in scratch scripts, not normal pytest.

## W. Full BC-v1 regression

Fresh run with PYTEST_DISABLE_PLUGIN_AUTOLOAD=1:

```text
/home/lorenzo/.local/share/mssr-bc-v1/venv/bin/python -B -m pytest drl/IL/tests/bc_v1 -q
```

1635 passed, 1 skipped in 155.17s (0:02:35). Process wall 156.419s; exit0.

## X. Frozen artifacts

| Artifact | Result |
| --- | --- |
| Canonical | UNCHANGED |
| F1 | UNCHANGED |
| G1 | UNCHANGED |
| F2 | UNCHANGED |
| Normalizer | UNCHANGED |
| Resolved tensor layout | UNCHANGED |
| G2 sidecar | UNCHANGED |
| J checkpoint | UNCHANGED |
| P1 audit artifacts | UNCHANGED |

Fresh production load_sources and read-only SidecarDataset verify approved file/semantic identities. Sidecar complete=true,119600samples /78302TRAIN. J checkpoint actual SHA matches the approved pin. All165 frozen file size/mtime snapshots are unchanged. All P1 audit files match pre-task SHA256; all147 preexisting source/config/docs/audit hashes are unchanged except authorized action_heads.py. No payload rebuild or exhaustive multi-GiB census was run.

## Y. Git audit

Branch `wip/t8-c05-human-teleop-20260921` unchanged. Only existing-file modification:action_heads.py. New tests and audit files listed in B. Historical dirty statuses and tracked diff preserved. git diff --check exit0; explicit no-index whitespace checks pass for all three untracked source/test files. No staging/commit/push/reset/restore/clean/switch.

AST/source check: only_scope,_arguments,_parameter_conditions changed; only_select_argument_rows,_category_routes added; constructor, neural architecture, P1 validator, resource head, structural conditions and reference scorer unchanged.

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

## Z. Remaining issues

Stretch <=1.00s is not reached (mean1.032s); no third major patch. Remaining measured compute includes structural-condition host work and backward. Deferred loader-throughput investigation remains separate because epoch-wide cold/random I/O is not characterized by these warm replayed batches. No semantic, gradient, invalid-input, full-regression or frozen-artifact failure remains.

P1b FINAL H2 PERFORMANCE PASS READY FOR REVIEW: YES
