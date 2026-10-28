# C1 Gap Curriculum Implementation Plan

**Goal:** Create an unseen procedural Snake8 gap distribution for C1 APPO.

**Architecture:** Keep the existing expert `CoplanarGapSpec` and expert
behavior untouched. Add a separate C1 course generator that produces two
coplanar physical banks separated by a real unsupported interval.

**Tech Stack:** Python, Isaac Sim CourseBox geometry, pytest.

## Contract

Splits:

- TRAIN: 16000..16999 — 1000 stages
- VALIDATION: 17000..17119 — 120 stages
- TEST: 18000..18199 — 200 stages
- CHALLENGE: 18500..18529 — 30 stages

TRAIN difficulty allocation:

- easy: 250
- medium: 400
- hard: 350

Difficulty is primarily controlled by:

1. gap width;
2. distance available before the near edge;
3. landing length;
4. small lateral displacement of the far bank.

No bank rotation is introduced in this first C1 version.

Physical safety constraint:

- maximum gap width = 0.178 m;
- this is the largest currently known physically solved Snake8 gap and is
  treated as a conservative cap until a later visual/physical audit expands it.

Ranges:

### Easy
- gap width: 0.120..0.145 m
- near edge X: 0.620..0.750 m
- landing length: 1.150..1.350 m
- landing lateral offset: 0

### Medium
- gap width: 0.140..0.160 m
- near edge X: 0.560..0.700 m
- landing length: 1.050..1.300 m
- landing lateral offset: ±0.025 m

### Hard
- gap width: 0.155..0.178 m
- near edge X: 0.520..0.640 m
- landing length: 0.950..1.200 m
- landing lateral offset: ±0.060 m

### Challenge
- gap width: 0.165..0.178 m
- near edge X: 0.500..0.600 m
- landing length: 0.850..1.050 m
- landing lateral offset: ±0.080 m

Common geometry:

- approach bank starts at X = -1.000 m;
- bank width = 1.200 m;
- bank top surface = Z = 0;
- bank thickness = 0.020 m;
- the unsupported interval is exactly `[near_edge_x, far_edge_x]`;
- initial eight-module radial assembly remains entirely on the approach bank;
- no success/reward/progress oracle is exposed to the actor observation.

## Follow-up

After generator validation:

1. wire the new course into Isaac;
2. visually audit representative stages;
3. verify Snake8 assembly and approach clearance;
4. implement raw gap reward;
5. add DirectMARLEnv-compatible gap transition evaluator;
6. calibrate raw reward scales before selecting APPO weights.
