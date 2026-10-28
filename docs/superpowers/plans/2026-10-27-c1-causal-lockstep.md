# C1 causal stepping and terminal authority

This contract extends the existing BC file ingress and native router. It is opt-in
with both `MSSR_BC_SESSION=<session>` and `MSSR_C1_LOCKSTEP=1`. Legacy BC lease
behavior is unchanged when lockstep is disabled. No physical smoke was launched
by this implementation task; physical evidence belongs to the C1 runner report.

Expert assembly runs continuously until the client explicitly arms the existing
`bc_request.json` mailbox. Arm is accepted only when native setup goals are no
longer active. Arm publishes an exact physical frame and holds physics. The
paused owner services mailboxes and sleeps without calling SimulationApp.update.

Every update supplies a distinct `command_id` of the existing generation form
`bc:<session>:<integer>`, including empty, native-goal-only, and rejected
proposals. Row generations and native goal IDs retain their existing meaning;
carried actuator generations do not need replacement on every decision.

Example request (usual row payloads omitted):

```json
{"kind":"update","session":"run-session","epoch":"owner-epoch",
 "command_id":"bc:run-session:1","state_stamp":12.0,
 "state_deadline":12345.9,"step_ticks":8,"continuous":[],"goals":[]}
```

`state_deadline` is the existing monotonic lease proof, refreshed immediately
before dispatch, strictly in the future and at most 1.01 seconds ahead of the
owner clock. `state_stamp` must equal the held simulation stamp. `step_ticks` is
an integer in [1,240], default 8, at the configured 240 Hz physics rate.

The owner admits through the existing executor and applies through the existing
native command router. Immediately after successful router application it records
`last_applied`, even for an empty/rejected actuator packet. Exactly the requested
number of SimulationManager ticks then run. Completion forces state publication
before publishing the existing `bc_status.json` status:

```json
{"last_completed":{"command_id":"bc:run-session:1","applied_stamp":12.0,
 "completed_stamp":12.033333333333333,"ticks":8,"outcome":"empty"}}
```

Outcomes are `accepted`, `empty`, or `rejected`; a BUSY recheck at the final drive
boundary also marks rejection. Rejection does not cancel an older native goal.
A partial proposal with any rejection reports `rejected`; details remain in the
existing rejection rows. The physics interval remains valid for rejected input.

After completion physics holds while policy inference runs; the completed lease
is cleared, so policy wall-clock latency does not cause spurious expiry. During
an in-flight interval, lease/source/proof/command-ID failure closes the physics
gate. The second gate immediately before simulation advancement also catches
revocation during the final drive boundary. Automatic rearm is prohibited.

`FileBackedIsaacControlStep(application_source=lambda env: status)` enables strict
causality. Call `state_reader(env)`, bind the dispatched ID with
`expect_action(env, command_id)`, then `physics_step()`, then `state_reader(env)`.
The adapter requires matching completed ID, `applied_stamp >= previous_stamp`,
`completed_stamp > applied_stamp`, and frame stamp exactly equal to the completed
stamp. A newer pre-application frame, an unrelated completion, or a frame after
the completed interval cannot satisfy the barrier. Cached next-state is reused
as the following previous-state. Reset clears the binding and cached episode.
Without application_source the earlier unmanaged freshness mode is preserved.

DirectMARLEnv revokes MARL authority independently for each environment on either
termination or truncation. Its terminal phase cannot reacquire authority until
reset launches a new Expert assembly. A physical barrier exception revokes all
active vector batch members and propagates the error without fabricating a
transition. Runtime context gates observe the same `handoff.marl_active` flag.

Verification: new tests were observed RED before implementation. Scoped
regression: 143 passed covering BC native admission/quarantine, causal frame
barrier, lifecycle/vectorization, physical interaction loop, vertical slice,
runtime context/binding, and C1 scenario wiring. No commit, push, dependency
installation, or Isaac launch was performed.
