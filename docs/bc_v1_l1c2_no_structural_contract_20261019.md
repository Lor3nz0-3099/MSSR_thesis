# BC-v1 L1c.2: structural decision contract v2

Approved scope (19 October 2026): implementation, synthetic tests and manual
commands only. No real G2 conversion, small-overfit, training, TEST or Isaac run
is performed by Codex in this change.

## Decision semantics

`a_t` composes resource decisions and structural decisions. They may coexist,
including on the same module when claims permit. No RESOURCE XOR STRUCTURAL.
Each current module has one structural group `(sample, initiator)`:

- its existing supervised structural binding remains the positive target;
- every other group receives `NO_STRUCTURAL`, independently of resource labels;
- the peer does not inherit the initiator's positive label;
- multiple supervised positives in one initiator group fail closed.

This is the user's explicit new annotation rule. Historical physical-stamp
buckets may contain coalesced sequential decisions; this change neither invents
intermediate physical states nor certifies their simultaneity. No labels or
historical executor state enter learned inputs.

`NO_STRUCTURAL` means do not start a new structural primitive for this initiator
now. It is not WAIT, a physical action, a duration, a claim, a cancellation,
release, reset, or keepalive. Existing executor goals continue autonomously.

## ABI and learning

The physical G2 input layout, RobotSpec/actions, H1 architecture, task input,
G/o/c whitelist, physical candidate enumeration and F2 remain unchanged.
The derived supervision schema is `mssr.bc_v1.structural_decisions.v2`:

- sparse positive `candidate/group/supervised` and argument/scope labels retained;
- dense `decision_group`, `decision_candidate`, `decision_supervised` added;
- `decision_candidate=-1` is NONE; nonnegative values index physical bindings;
- collate offsets physical indices and groups while preserving `-1`.

H2 config v2 adds one `no_structural_logits[g]` per current module. It competes
with the unchanged candidate scores in CE over `[NONE, candidates_g...]`.
NONE wins exact ties in deployment. It never becomes an executor goal.
A singleton NONE domain has finite zero CE. v1 remains available for frozen K1;
v2 loss rejects v1 predictions or missing/inconsistent dense targets.

Runtime claim checks and native admission remain authoritative. A conflicting
new goal is rejected without cancelling/replacing the old goal; disjoint goals
may run concurrently. Existing GRAVITY_SETTLE quarantine and L1c clock/stale
semantics are reused without a second ownership system. No task-based action
mask or raw-logit heuristic threshold is added.

## Prepared artifacts (not built)

`configs/bc_v1/no_structural_v2.json` declares the source pins and new output
`artifacts/bc_v1/sidecar/g2_v2_no_structural`. The converter reuses indexed G2-v1
physical tensors, preserving resource/positive targets and source sample keys.
Only TRAIN (78302) and VAL (22762) are materialized; TEST (18536) remains sealed.
Global inventory 119600 and positive total 1110 are inherited aggregates, not
claims of TEST processing. Certified TRAIN+VAL positives: ALIGN 418, DOCK 396,
UNDOCK 84, GRAVITY_SETTLE 21 (919 total). Actual group/NONE counts are emitted
post-build, with task/morphology/source/split breakdowns and code fingerprints.
Storage format remains sidecar v1; target ABI is explicitly v2 in its manifest
and identity. The physical tensor-layout fingerprint remains unchanged.

Manual entrypoints:

1. `python -B -m drl.IL.bc_v1.no_structural_prepare` then the same with `--verify`.
2. `python -B -m drl.IL.bc_v1.k2_no_structural small-overfit` (TRAIN only).
3. `python -B -m drl.IL.bc_v1.k2_no_structural train` (requires successful,
   identity-matched small-overfit report; optimizer TRAIN, selection VAL).

New configs are `action_heads_no_structural_v2.json` and
`training_no_structural_v2.json`. Batch64 stays fixed: 48 general +16 positive
sample draws replace 56+8. Draws are with replacement, positive sampling remains
episode-uniform, no physical sample duplication and no VAL oversampling.
Patience8; main K1 hyperparameters unchanged. All old K1 tensors warm-start
exactly; only the NONE scorer is newly initialized. K1 is never overwritten.

VAL reports NONE support/accuracy/precision/recall, positive support/binding
accuracy/activation recall, false/missed activation rates and per-operation
metrics. Positive metrics include misses caused by NONE. Zero denominators are
reported as null. Accuracy for NONE is its conditional correct rate (recall);
positive accuracy requires the exact physical binding. Teacher-conditioned
argument metrics retain their existing meaning.

The prepared selection gate rejects collapse (positive activation recall <0.5
or positive binding accuracy <half K1 VAL baseline 0.9095477387); this gate is
not a policy threshold. It selects by VAL objective among eligible candidates.
New outputs use `k2_no_structural_v1`; no TEST certification is implied.
A subsequent live smoke requires build, overfit, training and VAL review first.
