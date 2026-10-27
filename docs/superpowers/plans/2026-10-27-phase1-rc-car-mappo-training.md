# Phase 1 RC-Car8 MAPPO/APPO Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first curriculum environment for MARL training: seeded
RC-Car8 road navigation with static cones, raw reward terms, DirectMARLEnv
integration and a real Isaac vertical slice.

**Architecture:** Extend the already validated RC planar generator rather than
forking Expert geometry. Keep course generation, reward calculation and
DirectMARLEnv/runtime integration separate. Reward code emits raw terms first;
weights remain configuration, not embedded policy logic.

**Tech Stack:** Python 3.12, PyTorch, Isaac Sim 6.0, skrl 2.1.0, pytest,
existing SMORES-EP Isaac/runtime stack.

**Spec:** `docs/superpowers/specs/2026-10-27-phase1-rc-car-training.md`

## Global Constraints

- Active morphology in Phase 1 is `rc_car8`.
- 8 conceptual agents/modules.
- Expert assembly precedes MARL authority.
- Success requires all 8 module centres on the target platform.
- Cones are static rigid obstacles.
- No physical road side walls.
- No reverse-driving requirement in Phase 1.
- Reuse `rc_car_planar_obstacle_layout`.
- Actor must not receive generator-only oracle paths.
- Reward weights are not chosen before raw-term scale measurement.
- Runtime safety rejection remains independent of reward shaping.

## Review Focus

- Seed reproducibility.
- Target actually varies across seeds.
- No side-wall collider is introduced.
- Target success rejects even one module outside.
- Course geometry remains valid for the measured RC-Car8 footprint.

---

### Task 1: Phase-1 physical course contract

**Files:**
- Create: `scripts/smores_ep/src/smores_ep/isaac/phase1_training_course.py`
- Create: `scripts/smores_ep/tests/test_phase1_training_course.py`

**Produces:**
- `Phase1TrainingCourse`
- `OrientedPlatform`
- `Phase1Cone`
- `phase1_rc_car_training_course(seed)`

- [ ] RED tests for deterministic generation, varying target, static cones,
      no side walls, curved-road support and 8/8 target success.
- [ ] GREEN implementation reusing `rc_car_planar_obstacle_layout`.
- [ ] Focused regression.

### Task 2: Raw Phase-1 reward terms

**Files:**
- Create: `drl/MARL/phase1_reward.py`
- Create: `drl/MARL/tests/test_phase1_reward.py`

**Produces:**
- per-owner progress/inside/outside/event terms;
- team progress and 8/8 success;
- unweighted reward frame;
- explicit redundant-action signal input.

- [ ] RED reward tests.
- [ ] GREEN raw-term implementation.
- [ ] Measure numerical ranges without selecting weights.

### Task 3: DirectMARLEnv reward and termination bridge

**Files:**
- Modify current DirectMARLEnv/runtime provider files.
- Add focused MARL environment tests.

**Produces:**
- state -> action -> next state -> raw reward frame;
- collision/conflict/invalid event routing;
- configurable stagnation termination.

### Task 4: Isaac Phase-1 vertical slice

**Produces:**
- Expert assembly -> topology verify -> MARL;
- seeded Phase-1 physical course;
- BC-v4 warm-start actor;
- real action -> next state -> reward;
- persistent diagnostic log.

### Task 5: Training configuration

**Produces:**
- MAPPO/team baseline configuration;
- altruistic APPO configuration using the same Phase-1 environment;
- reward weights chosen from measured raw-term scales;
- first short training smoke before long campaigns.
