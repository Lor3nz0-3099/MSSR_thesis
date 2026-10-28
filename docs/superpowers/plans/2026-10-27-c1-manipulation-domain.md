# C1 MobileManipulator8 Manipulation Domain Implementation Plan

> **For agentic workers:** implement task-by-task with strict RED -> GREEN.

**Goal:** add the fourth C1 APPO domain: MobileManipulator8 button manipulation on structurally unseen fixtures.

**Architecture:** expert self-assembly creates MobileManipulator8. A deterministic translation-only reposition places the already assembled robot at the validated manipulation-ready standoff. MARL begins from there and owns only the 5-DoF manipulation/button-press portion.

**Tech Stack:** Python 3.12, pytest, Isaac Sim 6.0, ROS2 Humble, existing SMORES-EP runtime.

## Global constraints

- C1 manipulation TRAIN must not reuse expert/BC button episodes.
- Existing `mobile_manipulator_button_test_course()` remains untouched.
- Existing physical fixture dimensions remain unchanged.
- Existing validated button height envelope remains unchanged.
- New structural novelty comes primarily from continuously rotated fixture normals, not trivial seed perturbations.
- Expert/deterministic reposition is setup, not part of the APPO manipulation action space.
- Manipulation-ready base standoff is 0.325 m.
- No reward weights until raw-scale logging.
- Success eventually comes from live physical button depression, not inferred proximity.
- Physics remains 240 Hz.
- No commit or push until explicitly requested.

## Split contract

- TRAIN: 20000..20999 = 1000
- VALIDATION: 21000..21119 = 120
- TEST: 22000..22199 = 200
- CHALLENGE: 22500..22529 = 30

TRAIN:
- easy: 250
- medium: 400
- hard: 350

VALIDATION:
- 40 easy
- 40 medium
- 40 hard

TEST:
- 60 easy
- 70 medium
- 70 hard

CHALLENGE:
- 30 challenge

## Difficulty

All button normals are non-cardinal so the MARL terrain is structurally
different from the expert's original +/-X and +/-Y fixture family.

Easy:
- nearest-cardinal angular offset: 8..18 deg
- button z: 0.115..0.155 m
- central XY region

Medium:
- angular offset: 18..32 deg
- button z: 0.105..0.168 m
- wider XY region

Hard:
- angular offset: 32..44 deg
- button z: 0.100..0.176 m
- full validated XY region

Challenge:
- angular offset: 40..45 deg
- alternating low/high extreme button heights
- full XY region

Every sample must retain enough platform clearance for the 0.34 m
loose-module assembly ring at the 1.20 m button-relative spawn standoff.

## Tasks

1. Pure procedural manipulation curriculum.
2. Isaac/config/CLI runtime wiring using existing button-spawn transform.
3. Visual + physical MobileManipulator8 assembly/reposition audit.
4. Raw manipulation reward from EE PRE geometry, normal-cone error,
   physical depression progress and success.
5. DirectMARLEnv-compatible transition evaluator + raw JSONL logger.
