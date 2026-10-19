# BC-v1 delta spec — causal topology reconciliation v1.0.0

Date: 2026-10-10. Authority: approved causal-topology implementation request.
Status: **contract recorded; implementation BLOCKED at causal-boundary preflight**.
This document freezes the requested delta, not an alternative architecture.
No reconciler, provider, cache or official normalizer has been implemented or
published by this task. The historical architecture/plan remain unchanged.

## Approved contract

The policy remains `pi(G_t, o_t, c_t) -> a_t`. The delivered current topology
combines the published graph with physical disconnect completions confirmed
before the exact current decision. It does not infer topology from a label.

The generic core is planned at `drl/IL/bc_v1/causal_topology.py`. Its value types
use the existing generic connector references and immutable `StateGraph`:

```text
ConfirmedDisconnect(endpoint_a, endpoint_b, evidence_id, causal_order)
reconcile_topology(state_graph, confirmed_disconnects)
    -> ReconciledTopologyResult
```

For each causally admitted event, remove only the exact matching current
connection. Update the corresponding connector occupancy coherently. An
already absent connection or duplicate event is an idempotent no-op. A different
peer at either endpoint is a conflict: fail/diagnose without removing it.
The source graph is never mutated. No edges are added; module poses,
orientations, twists, resources, environment and task context are unchanged.
The core has no SMORES, task, morphology, stage, file, episode or transition
dependencies and imports without ROS/Isaac. A future live completion callback
can supply the same logical event type.

An offline SMORES evidence provider stays separate from the core and runs only
during prepare. It identifies the exact current primitive goal and anchors a
strict event-order prefix before that goal. It admits only successful physical
`UNDOCKED detached A from B` completions. A later completed `DOCKED attached`
before the decision invalidates affected pending disconnect evidence; it never
adds a graph edge. Missing goal identity and malformed, contradictory or
ambiguous endpoint evidence fail rather than guess. Log ordering, not invented
timestamps, determines the boundary.

Neither core nor provider uses future graph/rows/events, expert target endpoints,
control_state, stage or FSM to decide which edge to remove. Subsequent snapshots
are allowed only as diagnostic test evidence. Exact current goal identity is a
causal boundary, not authority to infer a detach from that goal's target.

Integration order is existing adapter typed state -> reconciled topology ->
existing PolicyInputs -> E1 -> E2. E1 mathematics and E2's compatible + free
endpoint rule remain unchanged. PolicyInputs and its whitelist are unchanged;
audit/evidence, lifecycle, stage, morphology/configuration goals and expert
actions do not become input features. Original topology, applied evidence and
derived topology belong exclusively to audit.

The implementation must identify this derivation as
`mssr.bc_v1.causal_topology.v1` with a deterministic fingerprint, included in
G1 cache identity together with hashes of runtime evidence actually read.
The fingerprint is not yet available because implementation is blocked.
RobotSpec, E2 schema, F1 membership and F2 normalization policy/engine stay pinned.

## Required gates and immutable assets

TDD must cover exact removal, immutability, idempotence, unrelated edges, peer
conflicts, causal prefix boundaries, redocking invalidation, missing/ambiguous
evidence, policy/audit isolation, offline/live equivalence, cardinality,
deterministic identity and hardware import isolation. All 20 frozen ALIGN cases
must become representable and the four healthy controls stay unchanged.
All 1110 structural targets must be retained and representable together in their
samples; no target changes or sample filtering are authorized. Production code
must not depend on the 20 regression IDs/indices.

Only after targeted GREEN: integrate into existing G1 and perform one complete
prepare. Immutable source is Canonical v6, manifest SHA256
`31765272b6ecd7506c54b25e214b75d7b1b28691803ddfc286dafcce8b4ec562`.
Target totals remain 73 BC episodes / 119600 samples, with F1 49/12/12 episodes
and 78302/22762/18536 samples. Seven expert button BC=0 episodes stay excluded.
Official F2 fit must access only 49 train episodes / 78302 samples and certify
complete train coverage. Reconciliation counts/edge removals must be audited by
split; unexpected cases beyond the 20 require a stop before completion.

No Canonical mutation/v7, runtime/recorder changes, G2/collate/sampler/DataLoader,
model, loss, optimizer, training/HPO, Isaac Lab installation, commit or push.
Future recorder work is documentation only: explicit ordered/timestamped
decision, completion, connect/disconnect and graph-publication/receipt events.

## Verified preflight incompatibility — mixed decision boundaries

Canonical episode `teleop-39e2f4b4cd684e9ea72734cc21baca94`, transition 1189,
physical stamp 157.85833333333332, aggregates source records 228–231. It supervises
UNDOCK `smores_05:BOTTOM <-> smores_06:LEFT` and new ALIGN
`smores_02:BOTTOM -> smores_01:TOP` in one current graph.

Runtime event order in the matched T15 log is:

1. line 3289: ACCEPTED UNDOCK02;
2. line 3290: ACCEPTED UNDOCK05;
3. line 3291: SUCCEEDED detach02 from06:RIGHT;
4. line 3292: SUCCEEDED detach05 from06:LEFT;
5. line 3294: ACCEPTED new REACH02 (the ALIGN decision).

With the unmodified adapter, candidate enumeration and G1 representability
validator, the strict prefix before UNDOCK05 keeps 7 edges: UNDOCK05 is
representable, ALIGN02 is not. The strict prefix before REACH02 includes both
detach completions, leaving 5 edges: ALIGN02 is representable, UNDOCK05 is not.
Keeping only detach02 leaves 6 edges and both targets representable, but this
intermediate prefix is neither decision's boundary: it is after UNDOCK05 was
accepted and before its completion, and omits a completion known before REACH02.
It cannot satisfy the approved exact pre-decision contract for both labels.

The stale-published-graph diagnosis for ALIGN remains valid. The new finding is
that this Canonical physical-stamp bucket contains sequential decision points.
One shared graph cannot be the exact reconciled pre-decision state of both.
Selecting a partial prefix to preserve labels or changing joint-sample semantics
is not part of this approved delta. Implementation and full prepare stop here;
resolving the decision-boundary incompatibility requires a separate architectural
decision. Evidence and reproducible probe are in
`artifacts/bc_v1/cache/g1_v1_causal_topology_blocked/`.
