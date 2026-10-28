# C1 Vector APPO Existing-Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run C1 physical collection with multiple Isaac environments while reusing the existing vectorized MSSRMAPPO infrastructure and preserving fixed-alpha altruistic PPO semantics.

**Architecture:** Generalize only the C1 physical/runtime glue that currently fixes the batch to one environment. DirectMARLEnv, Phase1VerticalSliceRunner and MSSRMAPPO remain the canonical vector pipeline. APPO remains Altruistic PPO: owner and mean-field advantages drive the policy update; team reward remains compatibility/logging data and must not replace the altruistic path.

**Tech Stack:** Python, PyTorch, skrl MAPPO infrastructure, Isaac/Isaac Lab, ROS2 Humble, MSSRMAPPO.

**Spec:** Conversation-approved C1-A architecture, 2026-10-28.

## Global Constraints

- APPO means Altruistic PPO, not asynchronous PPO.
- Reuse MSSRMAPPO infrastructure; do not implement a second MARL learner.
- Fixed-alpha APPO advantage remains A_i + alpha * A_MF_i.
- Preserve owner rewards, mean-field neighborhood data, topology and module positions for every environment.
- Team reward may remain for SKRL compatibility/logging but must not replace APPO owner/mean-field learning.
- Physics remains 240 Hz.
- step_ticks remains unchanged unless explicitly benchmarked later.
- Expert assembly -> independent topology verification -> MARL authority remains mandatory per environment.
- Active morphologies remain rc_car8, snake8 and mobile_manipulator8.
- First functional vector smoke: 2 environments.
- First performance target: 8 environments.
- No commit or push without explicit approval.

## Review Focus

- Different envs must never share reward, topology, state, command or handoff data.
- One env terminating must not corrupt the rollout rows of another active env.
- AltruisticRolloutBatchPayload must contain one row per active physical env.
- Owner rewards must stay owner-specific; team rewards must not be silently broadcast as APPO advantages.
- Physical stepping must occur once for the whole active vector batch.

---

### Task 1: Vectorize C1 APPO rollout recording

**Files:**
- Modify: `drl/MARL/c1_runner.py`
- Test: `drl/MARL/tests/test_c1_vector_appo.py`

**Produces:**
- `record_learning_batch(args, learner, provider, step, scalar_rewards_by_env)`
- One AltruisticRolloutBatchPayload row per active env.
- One call to MSSRMAPPO.record_online_rollout_batch for the active vector batch.

- [ ] Write failing two-environment altruistic rollout test.
- [ ] Verify RED.
- [ ] Implement minimal batch recording glue.
- [ ] Verify targeted tests.
- [ ] Verify existing MSSRMAPPO tests.

### Task 2: Remove batch=1 assumptions from C1 runner

**Files:**
- Modify: `drl/MARL/c1_runner.py`
- Test: `drl/MARL/tests/test_c1_runner.py`
- Test: `drl/MARL/tests/test_c1_vector_appo.py`

**Produces:**
- No unconditional `transitions[0]`, `contexts[0]` or `decisions[0]` in vector collection.
- Scalar reward/provenance/episode accounting keyed by env index.

- [ ] Write RED tests for two active environments.
- [ ] Implement vector iteration over `step.active_env_indices`.
- [ ] Preserve asynchronous termination semantics.
- [ ] Verify C1 runner tests.

### Task 3: Generalize physical C1 backend

**Files:**
- Modify: `drl/MARL/c1_runner.py`
- Modify runtime launch/backend files only where required.
- Test relevant runtime/synchronizer tests.

**Produces:**
- N handoffs.
- N runtime directories/state channels.
- One DirectMARLEnv with `num_envs=N`.
- One shared physical vector step.

- [ ] RED with 2 envs.
- [ ] Generalize PhysicalEpisode/backend without duplicating MAPPO infrastructure.
- [ ] Verify independent handoff/topology/state per env.
- [ ] Verify one shared physics step.

### Task 4: Physical vector smoke

**Produces:**
- 2-env physical smoke.
- Then 8-env smoke.
- Same-domain environments in the first implementation.

- [ ] 2 env: Expert assembly and topology verification for both.
- [ ] Collect vector physical transitions.
- [ ] Confirm APPO owner + mean-field data for both.
- [ ] 8 env: collect at least 256 total transitions.

### Task 5: Throughput benchmark

**Produces:**
- Comparable 1 / 2 / 4 / 8 environment measurements.

- [ ] Measure transitions/s.
- [ ] Measure wall time including assembly.
- [ ] Measure GPU memory.
- [ ] Confirm no semantic change in reward/APPO.
- [ ] Stop before long C1-A training.

## Separate follow-up

PPO `ratio_mean=1`, `approx_kl=0`, `clip_fraction=0` inspection is a separate correctness task. Vectorization must not be used as evidence that PPO clipping is functioning.
