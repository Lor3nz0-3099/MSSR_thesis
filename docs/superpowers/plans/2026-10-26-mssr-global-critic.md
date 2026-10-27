# MSSR Global Critic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Implement the independent centralized MAPPO critic for MSSR, producing one scalar `V(s_global)` from the full current privileged graph for variable module count N.

**Architecture:** The critic has its own randomly initialized GNN encoder and shares no parameters with the BC-v4 actor. Its input contains every current-state feature visible to the actor plus all current `critic_only` features that are masked from the actor. Module embeddings are aggregated with permutation-invariant `mean || max` pooling and passed through a value MLP to produce one scalar per environment.

**Tech Stack:** Python, PyTorch 2.10, skrl 2.1, existing MSSR BC-v1 graph/tensorization contracts.

**Spec:** Approved in chat on 2026-10-26 as part of `docs/superpowers/plans/2026-10-26-mssr-mappo-vertical-slice.md`.

## Global Constraints

- Actor and critic encoders share zero trainable parameters.
- Critic initialization is random; no BC checkpoint warm-start.
- Critic input = actor-visible current state + current `critic_only` information.
- Critic must never receive future, post-action, or oracle information.
- Output is exactly one scalar global value per environment/timestep.
- Architecture must support variable N by construction.
- Initial validation N values: 1, 3, 8, 11.
- Pooling is permutation-invariant `mean || max`.
- A change confined to a valid `critic_only` feature must be able to change critic output while leaving actor input unchanged.
- Standard team reward and team GAE remain unchanged.

## Review Focus

- Empty or malformed module sets must fail explicitly rather than produce NaN from pooling.
- Variable N must not change parameter shapes.
- Module ordering permutations must not change the global value.
- `critic_only` information must be available to the critic but absent from actor-visible input.
- Actor and critic parameter identity sets must be disjoint.

---

### Task 1: Privileged critic input contract

**Files:**
- Modify/create MARL critic input adapter in `drl/MARL/`
- Test in `drl/MARL/tests/`

**Interfaces:**
- Consumes the current full graph/state before actor visibility masking, or an exactly equivalent reconstruction `actor_visible ∪ critic_only`.
- Produces the tensorized graph contract expected by the critic GNN.

- [ ] Add tests proving actor-visible features are preserved.
- [ ] Add tests proving `critic_only` features are restored only for critic input.
- [ ] Add a causality test rejecting future/post-action fields.
- [ ] Verify RED.
- [ ] Implement the minimal privileged input adapter.
- [ ] Verify GREEN.

### Task 2: Independent GlobalCritic model

**Files:**
- Create `drl/MARL/global_critic.py`
- Test `drl/MARL/tests/test_global_critic.py`

**Interfaces:**
- `GlobalCritic(...): skrl.models.torch.Model`
- `forward_global(graph) -> torch.Tensor`
- `act(inputs, role="value") -> (values, extras)`
- Output shape `(batch_size, 1)`.

- [ ] Add tests for N=1,3,8,11.
- [ ] Add test that actor/critic parameter IDs are disjoint.
- [ ] Add test that critic parameters are random-init and are not loaded from BC-v4.
- [ ] Verify RED.
- [ ] Implement independent graph encoder.
- [ ] Implement `mean || max` module pooling.
- [ ] Implement value MLP.
- [ ] Implement skrl `compute/act`.
- [ ] Verify GREEN.

### Task 3: Permutation invariance and privileged sensitivity

**Files:**
- Modify `drl/MARL/tests/test_global_critic.py`

**Interfaces:**
- Same physical global state under module permutation must return the same value within numerical tolerance.
- A perturbation to a current `critic_only` feature must be observable by the critic.

- [ ] Add permutation test.
- [ ] Add critic-only sensitivity test.
- [ ] Add actor-visible invariance test for the same critic-only perturbation.
- [ ] Verify RED/GREEN.

### Task 4: MSSRMAPPO integration

**Files:**
- Modify `drl/MARL/mappo.py`
- Modify tests in `drl/MARL/tests/test_mssr_mappo.py`

**Interfaces:**
- `compute_global_bootstrap_value()` uses the privileged critic state.
- `reevaluate_structured_minibatch()` evaluates the global critic once per minibatch.
- Only `critic_optimizer` owns critic parameters.
- Actor optimizer owns no critic parameters.

- [ ] Add optimizer ownership tests.
- [ ] Add one-value-per-environment test.
- [ ] Add coordinated-update gradient test.
- [ ] Verify RED.
- [ ] Wire GlobalCritic into existing coordinated update.
- [ ] Verify GREEN.

### Task 5: Full regression gate

- [ ] Run:
  `pytest drl/MARL/tests/test_mssr_mappo.py drl/MARL/tests/test_policy.py drl/MARL/tests/test_bc_v4_actor_wrapper.py drl/MARL/tests/test_global_critic.py -q`
- [ ] Require zero failures.
- [ ] Confirm N=1,3,8,11 coverage.
- [ ] Confirm actor/critic parameter sets are disjoint.
- [ ] Confirm no actor regression from privileged critic inputs.
