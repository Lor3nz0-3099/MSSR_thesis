# L1a — pure action decoder and temporal decision boundary

Date: 2026-10-18. No Isaac execution, live observation bridge, controller,
continuous scheduler, training, or offline TEST evaluation is included.

## Result

ACTION DECODER READY: YES, subject to review of the explicit caller contracts.

TEMPORAL CONTRACT READY: NO. Historical command persistence is understood, but
there is no certified continuous BC decision horizon. This work deliberately
does not choose a frequency or turn a dead-man watchdog into a nominal horizon.

READY FOR L1b LIVE OBSERVATION BRIDGE: YES, as independent preparation work.
The continuous timing issue remains a blocker for an autonomous L1c driver.

## Repository boundary

Original runtime branch: `wip/t8-c05-human-teleop-20260921`.
Base HEAD: `b50c543731e88ef09a18db7b68ad461056795749`.
Evaluation branch: `feature/bc-v1-live-action-decoder`.
Worktree: `/home/lorenzo/MSSR_thesis_worktrees/bc-v1-live-action-decoder`.

The current runtime HEAD does not track BC-v1. To test against the actual
frozen BC contracts, 88 existing BC Python/config/test files were copied
byte-for-byte into this worktree, with a SHA inventory under
`/tmp/l1a_bc_v1/worktree_baseline.json`. None of the seven unrelated dirty
tracked files was copied. Two small F1 metadata fixtures (`episodes.json`,
`split.json`) were also copied for an existing normalization regression;
no dataset sample was copied. Original files and frozen artifacts were not
modified. These dependency copies are not new implementations.

## Temporal audit

Evidence paths below refer to the original runtime sources. No executor was
changed. Full raw streams or Canonical episodes were not scanned.

| Family | Dataset/runtime evidence | Ownership | Duration/completion | New decision point |
|---|---|---|---|---|
| Wheels | `primitives/file_channel.py:ActionFileChannel.commands`; teleop YAML; expert RC batch config | Current locomotion command source, subject to primitive claims | Last command retained until replacement; receipt-age watchdog rejects it after configured timeout (default 0.5 wall seconds) | Historical producer supplies next command; a continuous BC decision rule is NOT certified |
| PAN target | `IsaacPrimitiveExecutor.submit`, `_execute`, joint-target execution, `_finish` | Exclusive `internal_motion:<module>` | Joint tolerance/group rules or terminal failure/timeout/cancel; target may remain held after ownership release | Executor terminal event AND resource release, then fresh observation |
| PAN velocity | Action file's `pan_rate`/`pan_rate_rad_s` and `PAN_VELOCITY` drive mode | Same internal resource; primitive overlays can own it | Continuous command persistence/watchdog; not target completion | Continuous BC rule remains unresolved |
| TILT target | Behavior executor `_start_joint_target` path; teleop posture transport; primitive joint execution | Same exclusive internal resource as PAN; optional pushers/stabilizers add claims | Measured error inside tolerance, including coordination barrier, or terminal failure/timeout/cancel | Executor terminal event AND release; no same-resource new decision while BUSY |
| ALIGN | `IsaacPrimitiveExecutor._align_faces`, `_resources_for` | Mobile locomotion exclusive, target locomotion shared; TOP may additionally claim internal resource | Executor owns REACH/ALIGN/CLOCKING/APPROACH; terminal completion/failure/timeout | Terminal event and released claims; internal phase transitions are NOT new policy actions |
| DOCK | `_change_docking`, `_resources_for`, `_finish` | Both connector endpoints exclusive, mobile locomotion exclusive, target locomotion shared | Attach accepted/rejected, approach may run before final attach; timeout applies | Terminal event and release |
| UNDOCK | `_change_docking`, coordination handling, `_finish` | Both connector endpoints exclusive | Detach accepted/rejected, including coordination barrier and timeout | Terminal event and release |
| GRAVITY_SETTLE | `_gravity_settle`, `_resources_for` | Internal and locomotion resources of passive participants exclusive | Recorded/predicted physical `duration_s`; SUCCEEDED after elapsed duration or terminal failure | Terminal event and release; no invented peer/connector |

`TIMEOUT` is an existing terminal FAILED status with a timeout code; it is not
a new policy action or a new executor enum. Cancellation/rejection are terminal
too, but release/absence of ownership must still be confirmed.

### Continuous wheel horizon classification

**HELD_UNTIL_NEXT_COMMAND**.

The actual contract is persistence until the next producer publication,
bounded by a command-receipt watchdog. Evidence:

- `mssr_ws/src/mssr_expert/config/smores_teleop.yaml`: control 50 Hz,
  recording 10 Hz. The teleop timer uses the configured control rate.
- `scripts/smores_ep/run_rc_car_nav2_headless_batch.py`: behavior control
  10 Hz and log period 1. The morphology behavior node has a separate
  `/cmd_vel` freshness check; it must not become a BC controller.
- `ActionFileChannel.commands`: refreshes its monotonic receipt timestamp
  when a new file signature arrives, retains commands while fresh, returns
  no command after the timeout. This is NOT a promised action duration.
- TRAIN membership of `expert:rc_car:seed-005100` was verified using F1
  metadata before inspecting six compact behavior records. Their physical
  stamps were `44.425, 44.625, 44.625, 44.625, 44.625, 44.825`, including
  compact repeat spans. Sampling/physical-stamp spacing therefore cannot
  be equated to each producer decision or used as a per-action expiry.
- `preprocessing.py` transition duration is a state-transition interval;
  it does not establish an independent wheel-command lease.

The missing information is a reviewed runtime rule tying a continuous BC
decision to a certified horizon/refresh event. A 0.5 s watchdog, 0.1 s
recording interval, or 0.02 s teleop timer is not silently adopted for that
purpose. L1a implements no continuous clock, timer, TTL, or expiry scheduling.

### SET_TILT reference behavior

The morphology behavior executor creates a `PrimitiveGoalRequest`, records
the goal as active and waits for admission. Its configured joint timeout
defaults to 20 s; the generic `PrimitiveGoal` default is 30 s. Teleop posture
delivery supplies its own timeout and may retry transport admission of the
same goal. These are existing source-specific values, not a new BC timeout.

`IsaacPrimitiveExecutor.submit` claims `internal_motion:<module>` exclusively.
PAN and TILT therefore cannot independently own that module's internal
actuation. Additional pusher/stabilizer claims must be supplied by the runtime
adapter, including shared claims where appropriate. Conflicting exclusive
submissions are rejected with `RESOURCE_BUSY`; an already active goal ID is
not a new decision. Executor `step` advances execution; measured target error
and any coordination barrier determine success. `_finish` removes the active
goal and releases its claims. Some holding commands remain as a background
posture after the goal completes: retained posture is not a still-running
policy decision. Retransmitting the same request for delivery is distinct
from querying the policy again and creating a new goal.

## Decoder API and responsibilities

`action_decoder.decode_actions` takes only raw H2 predictions, causal G2
inputs/current routing, RobotSpec, already-loaded frozen F2, execution tags,
configured nonlearnable options, and ONLINE dispatch availability/claim
resolution. No metric accumulator or label values are consulted.

```text
raw H2
 -> current masked per-resource / per-initiator domain
 -> deterministic top-1 (first G2 position on exact ties)
 -> selected mode-conditioned payload / argument outputs
 -> inverse frozen F2
 -> physical ActionChoice + claims + execution semantic
```

The caller pins the frozen F2 artifact before passing it; the decoder also
checks its RobotSpec identity and the G2 normalizer fingerprint. No fitting
or artifact IO occurs inside the decoder. Categorical resource payloads retain
the existing catalog-index ABI; numeric values become physical units.
Declared numeric argument/resource parameter bounds are validated, never
silently clipped. Ambiguous resource-limit mapping fails closed.

`DispatchAvailability` carries explicit FREE/BUSY states and boolean resource
candidate/structural binding masks for each sample. Unknown ownership is
unavailable. Runtime masks certify current decision eligibility and physical
feasibility BEFORE selection; they are not Canonical supervision masks.
Execution tags are explicitly supplied by adapter/config, never inferred from
hard-coded PAN/TILT names in the generic decoder.

The mandatory backend `claim_resolver(ActionChoice)` must return ALL affected
current physical resource/connector claims and may perform backend-specific
validation. The generic decoder does not guess helper, magnetic, or shared
executor claims. Within one decode call it conservatively reserves proposed
claims, preventing conflicting outputs. A subsequent dispatcher must atomically
revalidate and acquire the real executor ownership; L1a is not that dispatcher.
The conservative FREE/BUSY view may suppress shared-compatible work; the
future runtime adapter must resolve that compatibility explicitly.

Nonlearnable options come only from `configured_arguments`; absent optional
options remain absent so existing executor defaults can apply. They never
come from teacher labels. Bound connector arguments use the existing ordered
`ActionChoice.connectors`; SETTLE stays endpoint-free. Scope and reference-set
predictions remain separate domains. Cross-domain hardware constraints must
be checked by the backend claim/admission adapter before execution.

For SMORES wheel velocity, the physical payload remains `(left, right)` in
rad/s. `SMORESAdapter.decode_action` produces `left_rad_s` and `right_rad_s`.
No conversion to `vx`, `yaw_rate`, or Nav2 command occurs here.

`DecodedAction.duration_s=None` describes the unresolved continuous lease.
It does not remove a delegated SETTLE action's own physical `duration_s`
parameter. No new WAIT/no-dispatch action is emitted when nothing is eligible.

## Future policy query interface (not a driver implementation)

```text
physics tick:
    advance existing executors / physics using their current ownership
    do not query BC merely because another physics tick occurred

executor terminal/release or episode start:
    determine eligible FREE resources from current runtime state
    if any actual decision resource is eligible:
        acquire fresh G_t/o_t AFTER prior execution
        build allowed current task/static context
        resolve pre-dispatch feasibility masks
        query frozen H1/H2 once
        decode proposals only for those eligible resources
        atomically revalidate/reserve claims and dispatch (future L1c)
    keep BUSY delegated resources untouched

continuous expiry/refresh:
    BLOCKED until an evidence-backed BC decision rule is approved
    no timer or arbitrary frequency is implemented in L1a
```

`physics_frequency != policy_decision_frequency`; executor update frequency
may equal physics frequency, but policy query frequency is event/horizon driven.
FREE resources do not imply inference on every physics tick: masks must describe
an actual decision point. The tests simulate repeated BUSY observations and
explicit ownership release; they do not certify a continuous expiry rule.

## Verification

- Initial decoder RED: 17 assertions `Missing L1a pure action decoder`.
- Additional RED: out-of-limit physical payload and invalid SETTLE endpoint
  each produced `DID NOT RAISE ValueError` before their respective checks.
- Unit/integration decoder fixtures: 26 passed, no dataset samples opened.
- Relevant regression across decoder, H2, F2, SMORES adapter, candidate domains,
  G2 tensorization, batching and H1: 279 passed, 14 skipped.
- Initial baseline had one missing external F1 metadata fixture; copying only
  the two metadata JSON files resolved it. Baseline: 87 passed, 1 skipped.
- Exactly one optional K1 synthetic CPU forward after GREEN: strict checkpoint
  load, original frozen F2, unchanged H1/H2, one finite physical wheel action,
  zero invalid actions, no accuracy evaluation. No VAL/TEST sample was used.
- Checkpoint SHA:
  `35737c025e7fa53492cb12e59ccf4c4f3f6210a6d6f0882c245bf1847975d06d`.
- F2 SHA:
  `ea8efbd9062acb8625e0618bcc5f86b975f20839bf3132a684a626c744e52003`.
- F2 fingerprint:
  `890eca0b9f6d4deca2cac81d3a41bf535aa3d03ac16e0ff4b81cbc0ca163abf5`.
- Temporary logs, copied-dependency inventories and the K1 synthetic result:
  `/tmp/l1a_bc_v1/`.

Remaining blockers: certified continuous decision rule; a live adapter providing
fresh snapshots, backend-validated claims and feasibility; the later exclusive
dispatcher/controller. No executor redesign was demonstrated as necessary.
No L1b/L1c implementation, commit, push, or merge was performed.
