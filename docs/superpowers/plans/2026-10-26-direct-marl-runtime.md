# Direct MARL Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Connect the validated structured MAPPO stack to the MSSR runtime through a thin DirectMARLEnv and a robot-specific runtime context provider, starting with deterministic Expert assembly of rc_car8 and an authority-safe handoff to MARL.

**Architecture:** `DirectMARLEnv` owns episode/reset/step/reward/termination. `DirectMARLRuntimeContextProvider` owns full-graph acquisition, actor/critic view separation, routing, hard masks and action execution through the existing adapter/runtime stack. A separate `DirectMARLHandoff` mirrors the teleoperation authority pattern: Expert success never establishes physical topology; MARL authority is granted only after an independent matching topology observation.

**Tech Stack:** Python, PyTorch, skrl 2.1, ROS2 Humble, Isaac Sim 6.0, existing MSSR Expert/SMORES runtime.

**Spec:** Approved conversation design, 2026-10-26; teleoperation authority pattern in `mssr_expert/teleop/state.py`, `coordinator.py`, `structural_macro.py`, and `smores_teleop_node.py`.

## Global Constraints

- Active morphologies remain `rc_car8`, `snake8`, `mobile_manipulator8`.
- First runtime slice uses N=8 and target `rc_car8`.
- Expert and MARL must never have actuator authority simultaneously.
- Expert `success=True` is insufficient for MARL handoff.
- Independent current topology verification must match the requested morphology.
- Failed or abnormally terminated Expert execution must never grant MARL authority.
- Actor input is post-visibility graph only.
- Critic input is current privileged pre-mask graph only; no future/oracle information.
- Full physical graph must be acquired once per decision context and forked into actor/critic views.
- Robot-specific semantics remain behind adapter/runtime boundaries; MAPPO remains robot-agnostic.
- Existing 110-test MARL regression must remain green.

## Review Focus

- Duplicate/stale Expert terminal events must not cause a second transition.
- A matching topology observed before Expert success must not grant MARL authority.
- A wrong/stale/None topology after Expert success must leave authority at NONE.
- Reset during any phase must revoke both Expert and MARL authority.
- Runtime action encoding must produce no MARL actuator command unless authority is exactly MARL.

---

### Task 1: Authority-safe Expert → MARL handoff

**Files:**
- Create: `drl/MARL/runtime_handoff.py`
- Create: `drl/MARL/tests/test_runtime_handoff.py`

**Interfaces:**
- Produces: `RuntimeAuthority`, `RuntimePhase`, `DirectMARLHandoff`.
- `DirectMARLHandoff.begin_expert(target_morphology: str) -> bool`
- `DirectMARLHandoff.observe_expert_terminal(success: bool) -> bool`
- `DirectMARLHandoff.observe_topology(name: str | None) -> bool`
- `DirectMARLHandoff.reset() -> None`

- [x] Write RED tests for authority ordering, successful terminal, topology verification, failure, duplicate terminal and reset.
- [x] Run focused tests and verify RED because runtime handoff module is absent.
- [x] Implement the minimal pure-Python state machine.
- [x] Run focused tests and full MARL regression.

### Task 2: DirectMARLRuntimeContextProvider

**Files:**
- Create: `drl/MARL/runtime_context.py`
- Create: `drl/MARL/tests/test_runtime_context.py`

**Interfaces:**
- Consumes: `DirectMARLHandoff`, actor visibility function, full-graph source, routing builder, hard-mask builder, action decoder/executor.
- Produces the already-approved MSSRMAPPO provider contract:
  - `current_action_contexts()`
  - `encode_env_actions(decisions)`
  - `current_critic_graphs()`

- [x] Test one full-graph acquisition per decision context.
- [x] Test critic receives the pre-mask graph.
- [x] Test actor receives only the post-visibility graph.
- [x] Test `critic_only` perturbation can affect critic view while actor view is unchanged.
- [x] Test MARL action execution is blocked unless authority is MARL.
- [x] Test routing/hard-mask snapshots remain aligned with each structured decision.
- [x] Implement through dependency injection first, then bind existing SMORES adapter/runtime components.
- [x] Run focused and full MARL regression.

#### Task 2B1: Real SMORES ActionSet/runtime binding

- [x] Bind the existing ActionSet decoder rather than creating a MARL-specific physical decoder.
- [x] Preserve all concurrently proposed ActionChoices in candidate order.
- [x] Reuse `assert_executable` for causal physical claims/conflict rejection.
- [x] Reuse `SMORESAdapter.decode_action` for robot-specific command translation.
- [x] Provide the real actor-visibility projector as an injectable runtime callable.
- [x] Keep runtime dispatch side effects outside this binding.

#### Task 2B2: Sampled MARLActionBundle -> ActionSet prediction

**Validated runtime seam:**
- `OnlineRolloutDecision.action` remains the true sampled structured action.
- `OnlineRolloutDecision.action_set_prediction_template` is the detached raw
  H2 output from the same forward.
- `SMORESMARLDecisionBridge` materializes and decodes that exact sample once.
- Semantic SMORES commands are staged through an injected dispatcher.
- The skrl env-action tensor is an explicit fixed-shape placeholder only;
  PPO reevaluation continues to use `StructuredRolloutSidecar`.
- No second actor forward, no second sampling, and no lossy action-vector
  serialization are introduced.


- [x] Convert sampled presence/payload values back into the existing ActionSet decoder schema.
- [x] Verify numeric payload values round-trip through frozen F2 normalization.
- [ ] Verify categorical/reference/scope samples decode to the sampled semantic choices.
- [ ] Verify multiple compatible present candidates survive as one ActionBundle.
- [ ] Verify hard-masked/absent candidates never enter the physical ActionBundle.

#### Task 2B3: Native SMORES dispatcher

- [x] Route `wheel_velocity` to native continuous locomotion commands.
- [x] Route `pan_velocity` to the native continuous PAN command.
- [x] Merge compatible wheel + PAN commands on the same module.
- [x] Submit structural actions through the existing primitive executor.
- [x] Replace, rather than latch, continuous commands at every policy decision.
- [x] Roll back primitive admissions from a partially rejected proposal.
- [x] Fail closed by clearing stale continuous commands on dispatch failure.
- [x] Compose staged MARL continuous commands with `PrimitiveExecutor.step()`
      and apply through `IsaacMultiModuleCommandRouter` through the validated
      native actuation bridge; DirectMARLEnv will own the call site.

### Task 2C: Altruistic APPO coordination objective

#### Task 2C1-A: Fixed-alpha causal altruism core

- [x] Build dynamic current-state neighbour sets `N_i,t` from radius `d`.
- [x] Compute `R_MF_i,t` as the average reward of current neighbours.
- [x] Define empty-neighbour behaviour explicitly as `R_MF_i,t = 0`
      (implementation choice; the source paper leaves this case undefined).
- [x] Implement exact fixed-alpha combination
      `A_AS_i = A_i + alpha_i * A_MF_i`.
- [x] Verify `alpha = 0` is exactly the selfish/individual advantage.
- [x] Support fixed personalized alpha values without making alpha learnable.
- [x] Preserve `team` as a separate MAPPO ablation mode.
- [ ] Integrate owner reward/value and mean-field reward/value into rollout GAE.
- [ ] Route selected altruistic advantages into the existing owner PPO loss.

#### Task 2C1-B: Fixed-alpha APPO value/GAE integration

- [x] Implement one generic GAE channel with explicit value bootstrap.
- [x] Compute `A_i` and owner critic return targets from `R_i` / `V_i`.
- [x] Compute `A_MF_i` and mean-field critic return targets independently
      from `R_MF_i` / `V_MF_i`.
- [x] Combine only the actor signal as
      `A_AS_i = A_i + alpha_i * A_MF_i`.
- [x] Keep owner and mean-field critic return targets separate.
- [x] Implement shared owner-conditioned `V_i(s)` critic.
- [x] Implement separate shared owner-conditioned `V_MF_i(s)` critic.
- [x] Store per-owner rewards and current neighbourhoods in rollout sidecar.
- [ ] Wire both critic losses and altruistic actor advantage into
      `MSSRMAPPO.coordinated_update()`.


Pending after 2C1-A. Requires separate owner/local and mean-field value
signals; `A_MF` must not be approximated as the average of neighbour
advantages.

#### Task 2C1-B2: Owner-conditioned critics

- [x] Encode the privileged graph once per critic.
- [x] Produce one value per module from
      `owner_h || mean(module_h) || max(module_h)`.
- [x] Share value-head parameters across owner IDs.
- [x] Preserve permutation equivariance under consistent owner-row permutation.
- [x] Support variable module count without parameter-shape changes.
- [x] Support multiple graph segments without cross-graph pooling.
- [x] Keep string/module identity mapping outside the critic.
- [x] Instantiate `V_i` and `V_MF_i` as independent networks with zero
      shared parameters.
- [x] Bind both critics to fresh repository-configured BCGNN encoders.
- [x] Verify repository factories reuse only `gnn_v1.json` architecture.
- [x] Verify owner and mean-field BCGNN encoders are independent fresh instances.
- [x] Verify actor, owner critic and mean-field critic have disjoint parameters.
- [x] Verify APPO critics are not warm-started from BC-v4 actor H1 weights.
- [x] Wire owner-ID -> current graph-row mapping in rollout sidecar.

#### Task 2C1-B4: Altruistic rollout sidecar

- [x] Require explicit `owner -> physical module_id` binding.
- [x] Resolve owner critic row from actual current graph module order.
- [x] Never infer a graph row from an owner/module name.
- [x] Store detached per-owner reward for each env/timestep.
- [x] Build causal current-state radius neighbourhoods.
- [x] Compute and store `R_MF_i,t` at rollout time.
- [x] Preserve time-major / environment-major sidecar alignment.
- [x] Support vectorized environments.
- [x] Feed this sidecar from the standard `record_transition() -> record_online_rollout_batch()` bridge.
- [x] Keep the existing team-reward `skrl Memory` path unchanged.
- [x] Supply APPO individual rewards through a separate runtime payload.
- [x] Record APPO rollout samples automatically with each online rollout batch.
- [x] Extend the existing PPO loss to accept either shared team advantage or per-owner advantage.
- [x] Preserve the exact team-MAPPO shared-advantage path.
- [x] Verify fixed-alpha altruistic advantage changes the real shared-actor optimizer update.
- [x] Convert recorded per-owner and mean-field rewards into time-major GAE trajectories.
- [x] Keep owner and mean-field critic return targets separate.
- [x] Build `A_i + alpha*A_MF_i` automatically from rollout data.
- [x] Verify recorded neighbour reward changes the real shared-actor PPO update.
- [x] Evaluate `V_i(s_t)` with the real owner-conditioned critic over recorded privileged graphs.
- [x] Evaluate `V_i^MF(s_t)` with the independent mean-field critic.
- [x] Resolve critic owner rows only from explicit rollout identity mapping.
- [x] Detach rollout value trajectories before GAE target construction.
- [x] Add explicit next-state owner-row mapping for final APPO bootstrap.
- [x] Evaluate `V_i(s_T+1)` from the real owner critic.
- [x] Evaluate `V_i^MF(s_T+1)` from the independent mean-field critic.
- [x] Use explicit next-state graph-row identity instead of reusing current-state ordering.
- [x] Consume it in `coordinated_update()` for altruistic GAE.
- [x] Reuse exactly the same PPO minibatch indexes for Memory, structured actions and altruistic rollout data.
- [x] Train the shared actor with per-owner `A_i + alpha*A_i^MF`.
- [x] Train the owner critic against owner returns.
- [x] Train the independent mean-field critic against mean-field returns.
- [x] Keep the global/team critic untouched in altruistic mode.
- [x] Preserve the existing team-MAPPO branch as the baseline objective.
- [x] Clear both rollout sidecars only after a successful update.
- [x] Fixed-alpha APPO offline integration complete.

#### Task 2C2: Personalized-alpha meta optimization

Pending after fixed-alpha APPO is validated. Implement the episode-scale
personalized-alpha optimization separately from the time-step PPO update.


**Goal:** Make conflict avoidance a learned coordination behaviour rather
than a runtime arbitration rule.

**Core distinction:**
- Hard masks remove only individually impossible / unavailable candidates.
- Candidates that are individually executable but mutually conflicting stay
  in the policy candidate universe.
- Candidate absence (`presence = 0`) is the implicit WAIT/yield decision.
- Runtime claim checking remains a fail-safe and never chooses a winner.

**Target objective:**
- Preserve per-owner policy/log-prob factorization.
- Introduce explicit altruistic advantage following the APPO motivation:
  `A_AS_i = A_i + alpha_i * A_MF_i`.
- Define the causal neighbour / mean-field benefit signal from current
  rollout information only.
- Begin with fixed/shared alpha as an ablation-safe implementation.
- Treat learnable/personalized alpha as a subsequent experiment rather
  than silently assuming it.
- Compare standard team-MAPPO vs altruistic APPO.

**Acceptance tests:**
- [ ] Two individually valid conflicting candidates are both visible to the actor.
- [ ] Neither candidate is hard-masked solely because the other exists.
- [ ] Presence sampling can express A-act/B-yield and B-act/A-yield.
- [ ] Runtime conflict rejection never selects or substitutes a winner.
- [ ] Per-agent/local advantage is computed causally.
- [ ] Mean-field/neighbour advantage excludes future information.
- [ ] `alpha = 0` reduces exactly to the non-altruistic owner advantage.
- [ ] Positive alpha changes the optimization signal when neighbours benefit.
- [ ] Inactive/hard-masked owners contribute no policy loss.
- [ ] Team-MAPPO baseline remains available as an explicit ablation.

### Task 3: Thin DirectMARLEnv and Expert assembly integration

### Task 3A — thin DirectMARLEnv authority lifecycle

- [x] Reset revokes previous authority before launching Expert.
- [x] Expert process launch happens before `EXPERT_ASSEMBLY` authority is claimed.
- [x] Expert success transitions only to `VERIFYING_TOPOLOGY`.
- [x] MARL requires an independent verified `rc_car8` topology observation.
- [x] Expert failure can never promote MARL.
- [x] Native MARL actuation is impossible without MARL authority.
- [x] Reset after MARL revokes MARL before the next Expert launch.
- [x] Vectorized environments keep handoff authority independent.
- [x] Bind the generic callbacks to the real SMORES Expert assembly runtime.
- [x] Bind the independent topology observer to the existing `configuration_assignment` detector.
- [ ] Bind `DirectMARLRuntimeContextProvider` and native actuation in the first Isaac vertical slice.

**Files:**
- Create: `drl/MARL/direct_env.py`
- Create: `drl/MARL/tests/test_direct_env.py`
- Modify only the minimal existing runtime entrypoint needed for the first Isaac slice.

**Interfaces:**
- Consumes: `DirectMARLHandoff`, `DirectMARLRuntimeContextProvider`, deterministic assembly Expert.
- Produces standard skrl multi-agent environment surfaces: `reset`, `state`, `step`.

- [ ] Test reset revokes all authority and begins deterministic assembly.
- [ ] Test Expert launch precedes Expert authority claim.
- [ ] Test Expert terminal success enters topology-verification phase, not MARL.
- [ ] Test verified `rc_car8` promotes to MARL.
- [ ] Test Expert failure/abnormal exit terminates the episode and cannot promote.
- [ ] Test Expert and MARL commands cannot coexist in one step.
- [ ] Test reset after MARL returns to the initial authority state.
- [ ] Bind the existing assembly Expert and topology detector.
- [ ] Run offline regression before any Isaac runtime gate.

### Task 4: First Isaac vertical slice

**Files:**
- Add/modify only the runtime entrypoint/config required for the first training smoke.

**Interfaces:**
- `N=8 disassembled → Expert assembly rc_car8 → topology verification → MARL locomotion → team reward → rollout → coordinated MAPPO update → reset`

- [ ] Run a bounded native smoke without training update.
- [ ] Verify authority/event trace.
- [ ] Run one bounded rollout through Memory + StructuredRolloutSidecar.
- [ ] Run one MAPPO update.
- [ ] Verify reset and second episode start.
- [ ] Only after native evidence, extend duration/reward/tuning work.


### Task 3B1 — real SMORES callback binding

- [x] Reuse the existing `mssr_smores_self_assembly_node`.
- [x] Reuse its structured `mssr.self_assembly_state.v1` terminal contract.
- [x] Reuse `SmoresSelfReconfigurationPlanner.configuration_assignment` for independent topology matching.
- [x] Preserve Expert-success and physical-topology verification as separate signals.
- [x] Plug the resulting callbacks directly into `DirectMARLEnv`.
- [x] Mirror `/mssr/expert/self_assembly/state` into the file-backed Isaac runtime.


### Task 3B2 — file-backed Expert/topology transport

- [x] Mirror `/mssr/expert/self_assembly/state` atomically to `self_assembly_state.json`.
- [x] Read Expert terminal state from the shared runtime directory without importing ROS into Isaac.
- [x] Read `robot_graph.json` and decode it through the repository's attributed-graph decoder.
- [x] Treat missing files as not-yet-observed, not success.
- [x] Fail explicitly on malformed runtime JSON.
- [x] Delete stale Expert terminal state before launching every new assembly episode.
- [x] Wire the Expert-state file through `smores_runtime.launch.py`.
- [ ] First real Isaac Expert -> topology verification -> MARL authority smoke.
