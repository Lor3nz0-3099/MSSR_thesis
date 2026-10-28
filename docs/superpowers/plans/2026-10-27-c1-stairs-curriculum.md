# C1 Stairs Curriculum Implementation Plan

> **For agentic workers:** execute with TDD: RED before production code.

**Goal:** Build the unseen procedural stairs distribution used by the
Snake8 component of C1 APPO multi-behavior robustification.

**Architecture:** The existing expert stair planner remains frozen and
continues to represent the BC-v4 demonstration domain. A separate pure
procedural generator produces APPO TRAIN/VALIDATION/TEST/CHALLENGE stair
courses as physical CourseBox geometry. Easy and medium remain uniform
ascending stairs; hard introduces both descent and internally non-uniform
stairs; challenge is non-uniform and alternates ascent/descent.

**Tech Stack:** Python 3.12, pytest, existing `CourseBox` physical contract.

**Spec:** C0/C1/C2/C3/C4 learning contract agreed in the thesis project.

## Global Constraints

- Active morphologies: rc_car8, snake8, mobile_manipulator8 only.
- BC-v4 remains frozen baseline and APPO warm-start.
- C1 stairs geometry must be unseen relative to expert demonstrations.
- Expert stair planner must not be modified.
- Hard includes descending and non-uniform stairs.
- No geometrically impossible negative support heights.
- TEST is sealed for later policy evaluation.
- No commit or push in this task.

## Task 1

Create:
`scripts/smores_ep/src/smores_ep/isaac/phase1_stairs_curriculum.py`

Test:
`scripts/smores_ep/tests/test_phase1_stairs_curriculum.py`

Public interfaces:

- `stair_curriculum_seeds(split: str) -> tuple[int, ...]`
- `stair_curriculum_identity(seed: int, *, split: str) -> StairCurriculumIdentity`
- `phase1_stairs_curriculum_course(seed: int, *, split: str) -> Phase1StairCourse`

Required progression:

- easy: ascent, uniform
- medium: ascent, uniform, larger envelope
- hard: ascent/descent + uniform/nonuniform
- challenge: ascent/descent + nonuniform

Completion:
all generator tests green, all 1000 TRAIN stages auditable, existing
obstacle-course tests remain green.


## Task 2 — Isaac Runtime Wiring

Goal:
materialize one Phase1StairCourse in Isaac and position loose modules on
the correct start platform before expert assembly.

Runtime contract:

- CLI:
  - `--phase1-stairs-course`
  - `--phase1-stairs-split`
  - `--phase1-stairs-seed`
- course selection is mutually exclusive with every other physical course.
- the procedural course replaces `/World/Ground`.
- ascent starts at nominal z=0.
- descent starts on the generated elevated platform.
- loose-module spawn XY/yaw remains the validated assembly layout.
- only spawn Z is translated by `course.start_height_m`.
- course observation comes from `Phase1StairCourse.to_observation()`.
- expert stair behavior is not used to define the APPO geometry.
- no reward changes in this task.
- no commit/push.
