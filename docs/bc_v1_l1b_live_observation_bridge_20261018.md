# BC-v1 L1b: consolidated runtime base and live observation bridge

Date: 2026-10-18. Scope: offline implementation and semantic certification for
current SMORES RC8 topology with task `navigation`. No simulator run, dispatcher,
scheduler, action application or L1c implementation.

## Source identities and consolidation

- Runtime base: `b50c543731e88ef09a18db7b68ad461056795749`.
- Official BC source: `9137627d64026cdbaa6e84aeb3d293492b93d2a4`.
- BC parent / merge base: `f06bc6bcb0fb84a8d029fb7ea382331f01feb218`.
- Worktree: `/home/lorenzo/MSSR_thesis_worktrees/bc-v1-live-runtime-integration`.
- Branch: `feature/bc-v1-live-runtime-integration`; no commit or staging.
- Original repository and old L1a worktree are read-only task sources.

The official commit has 133 SAFE ADD paths, zero SAFE UPDATE paths, one resolved
overlap and zero unresolved conflicts. All 133 additions were extracted from
Git objects, without copying the 88 manually copied old L1a baseline files.
The only overlap is `.gitignore`: retain the runtime version verbatim and append
the official BC ignore suffix. No active runtime behavior was overwritten.
Baseline imports plus the merged ignore file occupy 4,224,676 bytes; no episode
cache, SQLite, checkpoint, raw dataset, runtime log or training-run payload was
copied. Small versioned metadata already in the official commit is retained.

Only the following three L1a additions were ported from the old worktree, with
identical before/after SHA256:

| Path | SHA256, old = new |
|---|---|
| `drl/IL/bc_v1/action_decoder.py` | `cbd3a329bfea109621b0c0035519ed096f95558313bbb460cfaf8daf07bcf6ae` |
| `drl/IL/tests/bc_v1/test_action_decoder.py` | `7b635b7f78013762ba9b170e3824e84fe268c96b0b871a3af0dfebe151e4fa21` |
| `docs/bc_v1_l1a_decoder_temporal_contract_20261018.md` | `4f0b9d6a0e3e63e12552b25d320af056ded865d5818ac1180e7ce6b723581e3f` |

## Live source and observation model

The source is `scripts/smores_ep/src/smores_ep/isaac/state_graph_publisher.py`,
`StateGraphPublisher.publish()`. Acquire the combined `state_graph.json` payload:
its embedded state and graph have the same physical stamp and are written by
one temporary-file replacement. The separate `module_states.json` and
`robot_graph.json` files are individually atomic, but not an atomic pair.
No file reader or freshness policy is implemented in L1b.

Pose is the USD `body_link` world transform, quaternion XYZW. Linear velocity
is the publisher's finite difference of world translation. Angular velocity is
`(0, 0, wrapped_delta_yaw / dt)`, not a full PhysX angular-velocity measurement.
The existing publisher emits zero velocity on its first sample or invalid dt;
L1b preserves this observation model and never synthesizes missing measurements.
Actuator positions/rates come from current joint state: PAN retains its source
convention; the publisher already negates USD TILT to the logical convention.
There is no second sign flip, angle wrapping or unwrapping in the bridge.
Connections, contact candidates and world face geometry come from the current
docking state. Module type/capabilities come from the frozen RobotSpec, not
expert roles or physical IDs.

The adapter requires explicit finite poses, twists, all catalog actuator
measurements and connector geometry before invoking GraphBuilder, whose legacy
missing-value defaults must not be used to invent live state. It checks stamps,
duplicate IDs, state/graph physical agreement and attachment inventories; a
connected connector must reference a current physical edge. Target/assignment
edges cannot be supplied as current edges. Typed StateGraph entry also requires
complete current measurements, preventing a bypass through DTO construction.
Freshness, physical plausibility and simulator scheduling remain caller duties;
matching stamps and internal consistency do not prove an actually fresh sensor
sample or a mechanically realizable posture.

## Actual frozen offline path and live reuse

Historical source state/graph -> runtime
`mssr_expert.graph.graph_builder.GraphBuilder.build()` -> historical Canonical
role/root decoration -> `drl.IL.preprocessing.compact_robot_graph()` -> Canonical
current `graph_t` -> BC `SMORESAdapter.encode_state()` -> G1
`dataset.derive_physical_inputs()`:
causal topology reconciliation, where required -> D1 task/environment metadata
-> D2 `select_policy_inputs()` -> certified bridge resolution -> E1
`transform_inputs()` -> E2 `extract_features()` -> physical validation/current
candidate enumeration -> physical G1 cache.

F2 fitting has a separate train-only lifecycle. G2
`tensorization.tensorize_policy_sample()` applies those frozen normalizers,
then `batching.collate_policy_samples()` produces model inputs. L1b reuses these
functions; it does not reimplement normalization, tensorization, relative-pose
extraction or the feature schema.

Historical Canonical role/root decoration can reference a target morphology,
but it does not change the physical pose and its annotations are discarded by
D2. L1b does not reconstruct it. Historical Canonical `o_t` root-relative fields
are not blindly fed to the BC model: D1/D2 and E1 derive the approved physical
geometry. Historical source certification/provenance is audit/routing data,
not a trainable provenance feature. Certification can affect geometry validity,
which remains semantically relevant. Uncertified valid named-frame geometry is
rejected live instead of defaulting to WORLD or borrowing a demonstration's
certificate. WORLD identity uses the existing generic implicit identity rule.

An atomic current publication needs no historical bucket repair: L1b checks
current inventories instead. Historical causal-repair semantics are unchanged.

## Semantic feature map

`TRANSFORMABLE` means the same frozen functions produce the same learned values,
not an approximate alternative. No learned feature is UNKNOWN or MISMATCH.

| Feature | Training source/meaning | Live source/meaning | Frame | Unit | Transform | Learned | Parity |
|---|---|---|---|---|---|---|---|
| position | World body position, centroid-relative in E1 | Publisher body_link WORLD translation | WORLD -> robot_centroid | m | subtract observed centroid | YES | TRANSFORMABLE |
| orientation | XY Z W quaternion -> E1 rotation -> first two R6 columns | Publisher USD quaternion XYZW | WORLD axes | 1 | frozen quaternion normalization/R6 | YES | TRANSFORMABLE |
| linear velocity | World finite-difference body translation velocity | Publisher _velocity with same convention | WORLD axes, rotated only | m/s | frozen E1 rotation, no centroid-velocity subtraction | YES | EXACT |
| angular velocity | World-Z wrapped yaw-rate estimate, x=y=0 in this source | Publisher _velocity yaw estimator | WORLD axes, rotated only | rad/s | preserve recorded observation model | YES | EXACT |
| joints | Raw logical observed PAN, negated USD TILT; no new unwrap | Publisher actuator positions/rates | resource-local | rad; rad/s | no second sign flip or wrapping | YES | EXACT |
| wheels | Measured left/right wheel coordinates, not commands | Publisher actuator positions/rates | resource-local | rad; rad/s | preserve catalog coordinate order | YES | EXACT |
| connectivity | CURRENT physical docking/contact relations | Same snapshot graph + current attachments/contacts | topological | 1 | GraphBuilder -> compact -> BC adapter | YES | TRANSFORMABLE |
| connector geometry | Current world face poses -> body-local point/normal/tangent | Publisher docking face poses | WORLD -> module-body | m; 1 | frozen SMORESAdapter inverse body pose | YES | TRANSFORMABLE |
| edge geometry | Module B pose in endpoint A frame | Same current body poses | endpoint-A body | m; 1 | compact_robot_graph relative SE3 then E2 R6 | YES | TRANSFORMABLE |
| type/capability | RobotSpec catalog, not physical IDs or expert roles | Same pinned RobotSpec | intrinsic | 1 | existing E2 catalog encoding | YES | EXACT |
| task | navigation one-hot plus validity | Explicit current TaskContext.navigation | intrinsic | 1 | frozen G2 vocabulary | YES | EXACT |
| goal | Optional whitelisted world/map/stage-local XYZ, center or planar goal; E1 relative geometry | Explicit current TaskContext feature, never route-derived | source frame -> centroid/WORLD axes | m; rad | same bridge/E1/E2 representation | YES | TRANSFORMABLE |
| static geometry | Whitelisted physical cones/boxes/bounds, with intrinsic dimensions and unknown-height XY lines | Current task-owned physical geometry supplied as typed context | declared source frame -> frozen E1 | m; rad after E1 degree conversion | same D2/E1/E2, no nominal waypoints | YES | TRANSFORMABLE |
| navigation dynamic environment | No navigation-specific o_t channels in frozen D1; physical navigation objects live in c_static | Empty navigation ObservationBundle; robot dynamics remain in G_t | N/A | N/A | no new dynamic channel | NO | NOT USED |
| resource availability | Not a learned BC input | Not consumed by observation bridge | N/A | N/A | future dispatch contract only | NO | NOT USED |
| historical provenance/certification | Audit/routing only; certification affects physical validity, not a learned provenance feature | Current WORLD snapshot contract, explicit bridges for named frames | N/A | N/A | no historical source catalog lookup live | NO | NOT USED |

### Frame, ordering, masks and missingness

The actual frozen `frame_config_for()` uses the centroid of observed module
positions as origin, with WORLD axes. It does not select a minimum-ID root or
infer a robot heading. Module positions subtract this centroid; orientation
uses frozen quaternion normalization and the first two rotation-matrix columns
for R6. Twists rotate axes only: do not subtract centroid velocity. Named-source
geometry first uses its certified bridge to WORLD. Connector geometry is
converted WORLD -> module body once by SMORESAdapter; E1 preserves body-local
coordinates. Edge geometry is B-in-A and is not transformed twice.

Coordinate/capability/order definitions come from RobotSpec and frozen schema.
Current resource/connector state, feature validity, relationship endpoints,
normalization families and tensor order all reuse frozen E2/F2/G2. Missing
optional context remains absent/invalid with frozen masks; required current
robot measurements fail closed. XY cone positions remain unknown-height lines,
not invented z=0 points. Dimensions stay intrinsic; source degree angles use
frozen E1 conversion. Physical IDs are references for sorting, routing,
connectivity and decoding only; no learned physical-ID feature is introduced.

## Allowed and forbidden context

Only an explicit current `TaskContext(task_id='navigation', ...)` is certified
here. Frozen G2 navigation task one-hot is `[0, 1, 0, 0, 0]` plus its validity.
Optional goal geometry must already be a current task-owned feature admitted
by the frozen whitelist: e.g. a declared world XYZ goal becomes an E1
centroid-relative point; planar goals keep their partial-pose semantics.
Physical cones/boxes/bounds use the existing static object/relation encoding.
No seed, morphology goal or route-derived waypoint selection is introduced.

The navigation `ObservationBundle` is empty in the frozen D1 contract; robot
dynamics are in G_t and physical navigation objects are in c_static. This is
not permission to add new dynamic channels or discard robot dynamics. A future
runtime caller must supply allowed current task geometry; raw `course` metadata
may contain a future mission and is deliberately not forwarded wholesale.

Forbidden learned inputs: stage, FSM/control_state, configuration_goal,
current expert primitive/action/target, G_t+1, future topology, future route,
Nav2 route, cmd_vel, reward, success/outcome and provenance shortcuts. API
keyword injection is rejected; unknown typed context features are rejected;
extra raw expert/role/course metadata is stripped before graph construction.
Physical ID, stamp and SampleAudit are not learned columns. Empty supervision
is only the existing CachedSample ABI placeholder, not fabricated labels.

## Architecture and public boundary

```text
already-acquired combined current snapshot
-> encode_smores_snapshot(adapter, injected pure GraphBuilder)
-> compact_robot_graph -> SMORESAdapter -> LivePhysicalState
+ explicit current TaskContext / empty navigation ObservationBundle
+ explicit SourceFrameBridgeBatch / frozen configs and artifacts
-> prepare_live_observation()
-> strict PolicyInputs -> D2 -> E1 -> E2 -> validation/candidates
-> CachedSample with empty supervision
-> existing G2 tensorize (frozen F2 applied once) -> collate
-> H1-ready batch
```

`LivePhysicalState` and `LiveObservation` are frozen DTOs. Inputs are not mutated.
The module imports without Isaac, ROS, pxr, smores_ep or mssr_expert installed;
the snapshot adapter receives GraphBuilder as a dependency. No module count,
connector count or joint-name branching is added to the core typed path.
The SMORES publication adapter remains backend-specific by design.

## TDD and parity evidence

1. Before the production file existed: 21 new tests RED with the intended
   `Missing L1b live observation bridge` assertion; not an environment failure.
2. Minimal bridge: those 21 tests GREEN.
3. Typed-entry bypass tests first: pose/twist/coordinate cases all RED with
   `DID NOT RAISE ValueError`; a required-current-measurement gate made all three
   GREEN, reaching 24 tests.
4. Immutability and dependency-isolated import tests complete 26 L1b tests.

Hand-computed examples check centroid (2,2,0), module offset (-1,0,0), yaw +90
R6 (0,1,0,-1,0,0), unchanged world twists, B-in-A edge translation (0,-2,0),
goal offset (3,2,0) and unknown-height cone semantics. Nonfinite state, missing
measurements, mismatched stamps, future topology, forbidden context and unknown
named frames fail closed.

Golden source: SYNTHETIC publisher-shaped snapshot, 8 module references and
7 literal current RC8 connections. Its poses are a deterministic geometric
fixture, not a claim of a physically stable simulated RC8 posture. The same
physical snapshot is passed independently through the frozen offline
GraphBuilder/compact/`derive_physical_inputs` path with independently declared
D1 metadata and through L1b. Topology, typed PolicyInputs, node/edge/object and
relation FeatureViews, all learned tensors and masks, dtype/shape, normalized
tensors, tensor-layout/normalizer fingerprints and routing catalogs MATCH.
Tolerance is atol=0, rtol=0; maximum absolute learned-tensor delta is 0.
No VAL or TEST sample is required or accessed.

Renaming all module IDs and reversing input order preserves feature values and
validity after inverse-reference routing. Normalized module, connector,
resource, coordinate and connection rows differ only by the corresponding
permutation. No ID embedding is present. This verifies the bridge
representation, not a new certification of a simulator/controller.

## Verification record

| Stage | Passed | Failed | Skipped |
|---|---:|---:|---:|
| Official baseline imports/config/F2/G2/H1/H2/adapter | 337 | 0 | 7 |
| Exact L1a decoder port | 26 | 0 | 0 |
| Pertinent L1a regressions | 279 | 0 | 14 |
| Final pertinent L1b + baseline regressions | 541 | 0 | 27 |

Final suite files (under `drl/IL/tests/bc_v1/`): test_live_observation.py,
test_action_decoder.py, test_smores_adapter.py, test_normalization.py,
test_dataset.py, test_tensorization.py, test_batching.py, test_model.py,
test_features.py, test_policy_inputs.py, test_pipeline_validation.py,
test_source_frame_bridges.py, test_smores_source_frames.py; additionally
`drl/IL/test_canonical_dataset.py`. Skips concern missing external Canonical v6
fixtures in the isolated worktree; they were not regenerated or copied.

Command environment: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1
MKL_NUM_THREADS=1`, BC venv Python `-B -m pytest -p no:cacheprovider -q -rs`,
PYTHONPATH containing this worktree and its `drl/IL`. Logs and exact consolidation
inventories are in `/tmp/l1b_bc_v1/`, outside both source repositories.

### One optional frozen-checkpoint composition

Exactly one CPU forward, eval mode, `torch.inference_mode()`, on the synthetic
8-module current RC8-topology fixture. Original H1/H2 configs and frozen K1 step
6120 were loaded strictly. Synthetic FREE resource state and a wheels-only
candidate mask allow L1a semantic decoding without dispatch. Eight finite
wheel commands, zero invalid outputs. No accuracy or behavior evaluation.
Model parameters/buffers and checkpoint bytes remain identical before/after.
The mask is a test fixture, not a certified live feasibility provider.

### Counter limitation: existing regression autograd tests

Checkpoint backward calls: 0. Training runs: 0. Optimizer steps: 0.
F2 refits: 0. F1 TEST samples: 0. Isaac/ROS/Nav2 launches: 0.

The requested existing regression suites include
`test_backward_smoke_has_finite_nonzero_encoder_and_message_gradients` and
`test_raw_prediction_backward_reaches_heads_and_h1_without_bc_loss`, which call
backward on newly constructed synthetic models. Those tests ran during baseline
and L1a regression verification; the model backward test also ran in the final
regressions. Therefore the literal global requirement `backward calls = 0`
was NOT met. This is a procedural deviation, not a checkpoint update or a
training run, and cannot be represented as a zero global counter. No further
backward or model execution is needed to review L1b.

## Frozen artifacts and limits

K1 checkpoint SHA256 before/after:
`35737c025e7fa53492cb12e59ccf4c4f3f6210a6d6f0882c245bf1847975d06d`.
No frozen baseline algorithm/config, Canonical/F1/G1/G2 identity, F2 normalizer,
layout, model, loss or action vocabulary is changed by the bridge.
Final identity checks compare the original repository and old L1a status,
HEAD, diff, staging and source hashes against the initial snapshots, plus
baseline import and port hashes; see `/tmp/l1b_bc_v1/final_verification.json`.

Semantic certification is limited to the navigation contract and synthetic
RC8-topology fixtures. Other task dynamic observations and actual simulator
acquisition are not certified here. Current named frames require real certified
bridges from the caller; no certificate or historical evidence is synthesized.

Continuous wheel policy horizon remains uncertified. No arbitrary TTL, policy
query frequency, watchdog-as-horizon assumption, scheduling or executor
ownership logic has been introduced. Completion-driven L1a semantics remain
unchanged. The observation bridge is technically ready to review and L1c can
be designed, but autonomous execution remains blocked by the continuous-action
refresh/horizon contract. L1c is not implemented or run.

Because the literal zero-global-backward hard gate was violated by the existing
regression tests, the overall requested hard-gate report must remain NO even
though bridge tests and frozen-checkpoint integrity pass.

## Exact path classification

Every new-worktree status path belongs to the following inventories; no
additional runtime behavior file is modified.

| Path | Classification |
|---|---|
| `.gitignore` | official baseline: OVERLAP RESOLVED |
| `artifacts/bc_v1/audits/causal_repairs_v1.json` | official baseline: SAFE ADD |
| `artifacts/bc_v1/audits/j_small_overfit_v1/history.jsonl` | official baseline: SAFE ADD |
| `artifacts/bc_v1/audits/j_small_overfit_v1/selection.json` | official baseline: SAFE ADD |
| `artifacts/bc_v1/audits/j_small_overfit_v1/summary.json` | official baseline: SAFE ADD |
| `artifacts/bc_v1/audits/p0_loader_v1/inspection.json` | official baseline: SAFE ADD |
| `artifacts/bc_v1/audits/p0_loader_v1/requests.json` | official baseline: SAFE ADD |
| `artifacts/bc_v1/audits/p1_h2_optimization_v1/baseline.json` | official baseline: SAFE ADD |
| `artifacts/bc_v1/audits/p1_h2_optimization_v1/equivalence.json` | official baseline: SAFE ADD |
| `artifacts/bc_v1/audits/p1_h2_optimization_v1/postfix.json` | official baseline: SAFE ADD |
| `artifacts/bc_v1/audits/p1_h2_optimization_v1/report.md` | official baseline: SAFE ADD |
| `artifacts/bc_v1/audits/p1_h2_optimization_v1/step_a.json` | official baseline: SAFE ADD |
| `artifacts/bc_v1/audits/p1b_h2_final_optimization_v1/baseline.json` | official baseline: SAFE ADD |
| `artifacts/bc_v1/audits/p1b_h2_final_optimization_v1/equivalence.json` | official baseline: SAFE ADD |
| `artifacts/bc_v1/audits/p1b_h2_final_optimization_v1/equivalence_step_a.json` | official baseline: SAFE ADD |
| `artifacts/bc_v1/audits/p1b_h2_final_optimization_v1/final_h2_breakdown.json` | official baseline: SAFE ADD |
| `artifacts/bc_v1/audits/p1b_h2_final_optimization_v1/post_step_a_routing_breakdown.json` | official baseline: SAFE ADD |
| `artifacts/bc_v1/audits/p1b_h2_final_optimization_v1/post_step_b_routing_breakdown.json` | official baseline: SAFE ADD |
| `artifacts/bc_v1/audits/p1b_h2_final_optimization_v1/postfix.json` | official baseline: SAFE ADD |
| `artifacts/bc_v1/audits/p1b_h2_final_optimization_v1/report.md` | official baseline: SAFE ADD |
| `artifacts/bc_v1/audits/p1b_h2_final_optimization_v1/step_a_scope.json` | official baseline: SAFE ADD |
| `artifacts/bc_v1/audits/p1b_h2_final_optimization_v1/step_b_routing.json` | official baseline: SAFE ADD |
| `artifacts/bc_v1/audits/post_j_performance_v1/report.md` | official baseline: SAFE ADD |
| `artifacts/bc_v1/audits/post_j_performance_v1/summary.json` | official baseline: SAFE ADD |
| `artifacts/bc_v1/audits/post_j_performance_v1/torch_profiler_top_ops.txt` | official baseline: SAFE ADD |
| `artifacts/bc_v1/cache/g1_v1_optimized/manifest.json` | official baseline: SAFE ADD |
| `artifacts/bc_v1/checkpoints/j_small_overfit_v1/manifest.json` | official baseline: SAFE ADD |
| `artifacts/bc_v1/normalization/f2_v1_optimized/fit_context.json` | official baseline: SAFE ADD |
| `artifacts/bc_v1/normalization/f2_v1_optimized/normalizers.json` | official baseline: SAFE ADD |
| `artifacts/bc_v1/sidecar/g2_v1/manifest.json` | official baseline: SAFE ADD |
| `artifacts/bc_v1/splits/f1_v1/audit.json` | official baseline: SAFE ADD |
| `artifacts/bc_v1/splits/f1_v1/episodes.json` | official baseline: SAFE ADD |
| `artifacts/bc_v1/splits/f1_v1/report.md` | official baseline: SAFE ADD |
| `artifacts/bc_v1/splits/f1_v1/split.json` | official baseline: SAFE ADD |
| `configs/bc_v1/action_heads_v1.json` | official baseline: SAFE ADD |
| `configs/bc_v1/causal_topology.json` | official baseline: SAFE ADD |
| `configs/bc_v1/counts.v1.json` | official baseline: SAFE ADD |
| `configs/bc_v1/features.json` | official baseline: SAFE ADD |
| `configs/bc_v1/gnn_v1.json` | official baseline: SAFE ADD |
| `configs/bc_v1/loss_v1.json` | official baseline: SAFE ADD |
| `configs/bc_v1/metrics_v1.json` | official baseline: SAFE ADD |
| `configs/bc_v1/normalization.json` | official baseline: SAFE ADD |
| `configs/bc_v1/prepare.json` | official baseline: SAFE ADD |
| `configs/bc_v1/robot_spec.schema.json` | official baseline: SAFE ADD |
| `configs/bc_v1/sidecar_v1.json` | official baseline: SAFE ADD |
| `configs/bc_v1/smores_ep.robot.json` | official baseline: SAFE ADD |
| `configs/bc_v1/source_frames.json` | official baseline: SAFE ADD |
| `configs/bc_v1/task_context.json` | official baseline: SAFE ADD |
| `configs/bc_v1/tensor_layout.v1.json` | official baseline: SAFE ADD |
| `configs/bc_v1/training.json` | official baseline: SAFE ADD |
| `configs/bc_v1/training_baseline_v1.json` | official baseline: SAFE ADD |
| `configs/bc_v1/training_smoke_v1.json` | official baseline: SAFE ADD |
| `docs/bc_v1_align22_manual_audit_20261007.md` | official baseline: SAFE ADD |
| `docs/bc_v1_context_v1.0.1_20261008.md` | official baseline: SAFE ADD |
| `docs/bc_v1_joint_structural_audit_20261007.md` | official baseline: SAFE ADD |
| `docs/bc_v1_mm8_recoverability_audit_20261007.md` | official baseline: SAFE ADD |
| `docs/superpowers/plans/2026-10-06-bc-v1-implementation-v1.0.0.md` | official baseline: SAFE ADD |
| `docs/superpowers/plans/2026-10-11-bc-v1-coalesced-supervision-causal-topology-implementation-v1.0.0.md` | official baseline: SAFE ADD |
| `docs/superpowers/plans/2026-10-12-bc-v1-source-frame-bridges-implementation-v1.0.0.md` | official baseline: SAFE ADD |
| `docs/superpowers/specs/2026-10-06-bc-v1-architecture-v1.0.0.md` | official baseline: SAFE ADD |
| `docs/superpowers/specs/2026-10-10-bc-v1-causal-topology-v1.0.0.md` | official baseline: SAFE ADD |
| `docs/superpowers/specs/2026-10-11-bc-v1-coalesced-supervision-causal-topology-v1.0.0.md` | official baseline: SAFE ADD |
| `docs/superpowers/specs/2026-10-12-bc-v1-source-frame-bridges-v1.0.0.md` | official baseline: SAFE ADD |
| `drl/IL/bc_v1/__init__.py` | official baseline: SAFE ADD |
| `drl/IL/bc_v1/__main__.py` | official baseline: SAFE ADD |
| `drl/IL/bc_v1/action_heads.py` | official baseline: SAFE ADD |
| `drl/IL/bc_v1/batching.py` | official baseline: SAFE ADD |
| `drl/IL/bc_v1/bc_v1_contract.py` | official baseline: SAFE ADD |
| `drl/IL/bc_v1/candidates.py` | official baseline: SAFE ADD |
| `drl/IL/bc_v1/causal_topology.py` | official baseline: SAFE ADD |
| `drl/IL/bc_v1/dataset.py` | official baseline: SAFE ADD |
| `drl/IL/bc_v1/feature_schema.py` | official baseline: SAFE ADD |
| `drl/IL/bc_v1/features.py` | official baseline: SAFE ADD |
| `drl/IL/bc_v1/losses.py` | official baseline: SAFE ADD |
| `drl/IL/bc_v1/metrics.py` | official baseline: SAFE ADD |
| `drl/IL/bc_v1/model.py` | official baseline: SAFE ADD |
| `drl/IL/bc_v1/normalization.py` | official baseline: SAFE ADD |
| `drl/IL/bc_v1/normalization_prepare.py` | official baseline: SAFE ADD |
| `drl/IL/bc_v1/pipeline_validation.py` | official baseline: SAFE ADD |
| `drl/IL/bc_v1/policy_inputs.py` | official baseline: SAFE ADD |
| `drl/IL/bc_v1/requirements.txt` | official baseline: SAFE ADD |
| `drl/IL/bc_v1/sampling.py` | official baseline: SAFE ADD |
| `drl/IL/bc_v1/sidecar.py` | official baseline: SAFE ADD |
| `drl/IL/bc_v1/sidecar_build.py` | official baseline: SAFE ADD |
| `drl/IL/bc_v1/small_overfit.py` | official baseline: SAFE ADD |
| `drl/IL/bc_v1/smores_adapter.py` | official baseline: SAFE ADD |
| `drl/IL/bc_v1/smores_disconnect_evidence.py` | official baseline: SAFE ADD |
| `drl/IL/bc_v1/smores_source_frames.py` | official baseline: SAFE ADD |
| `drl/IL/bc_v1/source_frame_bridges.py` | official baseline: SAFE ADD |
| `drl/IL/bc_v1/split.py` | official baseline: SAFE ADD |
| `drl/IL/bc_v1/task_context.py` | official baseline: SAFE ADD |
| `drl/IL/bc_v1/temporal_supervision.py` | official baseline: SAFE ADD |
| `drl/IL/bc_v1/tensorization.py` | official baseline: SAFE ADD |
| `drl/IL/bc_v1/train_baseline.py` | official baseline: SAFE ADD |
| `drl/IL/bc_v1/trainer.py` | official baseline: SAFE ADD |
| `drl/IL/bc_v1/training.py` | official baseline: SAFE ADD |
| `drl/IL/tests/bc_v1/fixtures.py` | official baseline: SAFE ADD |
| `drl/IL/tests/bc_v1/fixtures/causal_topology.v1.json` | official baseline: SAFE ADD |
| `drl/IL/tests/bc_v1/g2_fixtures.py` | official baseline: SAFE ADD |
| `drl/IL/tests/bc_v1/h2_legacy_reference.py` | official baseline: SAFE ADD |
| `drl/IL/tests/bc_v1/h2_scope_routing_reference.py` | official baseline: SAFE ADD |
| `drl/IL/tests/bc_v1/test_action_heads.py` | official baseline: SAFE ADD |
| `drl/IL/tests/bc_v1/test_action_heads_bulk.py` | official baseline: SAFE ADD |
| `drl/IL/tests/bc_v1/test_action_heads_routing.py` | official baseline: SAFE ADD |
| `drl/IL/tests/bc_v1/test_batching.py` | official baseline: SAFE ADD |
| `drl/IL/tests/bc_v1/test_candidates.py` | official baseline: SAFE ADD |
| `drl/IL/tests/bc_v1/test_causal_topology.py` | official baseline: SAFE ADD |
| `drl/IL/tests/bc_v1/test_contract.py` | official baseline: SAFE ADD |
| `drl/IL/tests/bc_v1/test_dataset.py` | official baseline: SAFE ADD |
| `drl/IL/tests/bc_v1/test_dataset_counts.py` | official baseline: SAFE ADD |
| `drl/IL/tests/bc_v1/test_e2_integration.py` | official baseline: SAFE ADD |
| `drl/IL/tests/bc_v1/test_feature_schema.py` | official baseline: SAFE ADD |
| `drl/IL/tests/bc_v1/test_features.py` | official baseline: SAFE ADD |
| `drl/IL/tests/bc_v1/test_losses.py` | official baseline: SAFE ADD |
| `drl/IL/tests/bc_v1/test_metrics.py` | official baseline: SAFE ADD |
| `drl/IL/tests/bc_v1/test_model.py` | official baseline: SAFE ADD |
| `drl/IL/tests/bc_v1/test_normalization.py` | official baseline: SAFE ADD |
| `drl/IL/tests/bc_v1/test_normalization_prepare.py` | official baseline: SAFE ADD |
| `drl/IL/tests/bc_v1/test_pipeline_validation.py` | official baseline: SAFE ADD |
| `drl/IL/tests/bc_v1/test_policy_inputs.py` | official baseline: SAFE ADD |
| `drl/IL/tests/bc_v1/test_robot_spec.py` | official baseline: SAFE ADD |
| `drl/IL/tests/bc_v1/test_sampling.py` | official baseline: SAFE ADD |
| `drl/IL/tests/bc_v1/test_sidecar.py` | official baseline: SAFE ADD |
| `drl/IL/tests/bc_v1/test_sidecar_build.py` | official baseline: SAFE ADD |
| `drl/IL/tests/bc_v1/test_smores_adapter.py` | official baseline: SAFE ADD |
| `drl/IL/tests/bc_v1/test_smores_disconnect_evidence.py` | official baseline: SAFE ADD |
| `drl/IL/tests/bc_v1/test_smores_source_frames.py` | official baseline: SAFE ADD |
| `drl/IL/tests/bc_v1/test_source_frame_bridges.py` | official baseline: SAFE ADD |
| `drl/IL/tests/bc_v1/test_split.py` | official baseline: SAFE ADD |
| `drl/IL/tests/bc_v1/test_task_context.py` | official baseline: SAFE ADD |
| `drl/IL/tests/bc_v1/test_temporal_supervision.py` | official baseline: SAFE ADD |
| `drl/IL/tests/bc_v1/test_tensorization.py` | official baseline: SAFE ADD |
| `drl/IL/tests/bc_v1/test_trainer.py` | official baseline: SAFE ADD |
| `drl/IL/tests/bc_v1/test_training.py` | official baseline: SAFE ADD |
| `drl/IL/bc_v1/action_decoder.py` | exact L1a port |
| `drl/IL/tests/bc_v1/test_action_decoder.py` | exact L1a port |
| `docs/bc_v1_l1a_decoder_temporal_contract_20261018.md` | exact L1a port |
| `drl/IL/bc_v1/live_observation.py` | L1b production |
| `drl/IL/tests/bc_v1/test_live_observation.py` | L1b test |
| `docs/bc_v1_l1b_live_observation_bridge_20261018.md` | L1b documentation |
