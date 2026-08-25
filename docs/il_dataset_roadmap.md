# Deterministic expert and IL dataset roadmap

This document records the intended thesis progression after validating the
three initial eight-module morphologies.

## Separation of responsibilities

- A morphology behavior describes a reusable capability such as
  `crawl_stairs`, `cross_gap`, or `press_button`. It must be parameterized by
  live world observations and obstacle geometry, not specialized to one fixed
  course layout or terminated by arbitrary locomotion timers.
- A scenario generator chooses the course instance. It randomizes obstacle
  presence, order, dimensions and difficulty while retaining a seed and a
  feasibility check.
- A deterministic task expert observes the generated course, selects a
  capable morphology, plans any required reconfiguration, parameterizes the
  selected behavior and verifies its geometric outcome.
- The dataset logger records the complete observation, selected morphology,
  reconfiguration action, behavior name and parameters, low-level actions,
  progress observations, terminal outcome and scenario seed.

The current three-step stage is therefore a validation fixture for the
Snake8 `crawl_stairs` behavior, not the definition of that behavior.

## Dataset variation requirements

Generated episodes must vary independently across:

- obstacle presence or absence;
- obstacle order;
- geometric difficulty, including heights, widths, gaps and clearances;
- starting robot pose and admissible module arrangement;
- morphology and behavior sequences that remain feasible for the episode.

Training, validation and test partitions must use disjoint scenario seeds and
course layouts. Failed deterministic executions must remain available as
diagnostic rollouts but must not be labelled as successful expert
demonstrations for behavior cloning.

## Planned implementation order

1. Complete and validate the parameterized Snake8 stair behavior.
2. Complete the MobileManipulator8 behavior set and geometric button task.
3. Add further morphologies only with explicit capability metadata and
   verified transitions.
4. Implement one observation-driven deterministic task expert that selects
   morphologies and behaviors to reach the course goal.
5. Randomize feasible course episodes and generate a varied IL dataset.
6. Use behavior cloning to initialize a policy before subsequent learning or
   generalization stages.

The first successful `crawl_stairs_arch_wave` baseline is frozen in
`docs/validated_behaviors/snake8_crawl_stairs_arch_wave.md`.  Before step 2,
the isolated uniform-stair generator and headless evaluator exercise a small
conservative family of stairs.  This is a regression/robustness gate, not the
full randomized obstacle-course generator planned in step 5.
