# Deterministic expert and IL dataset roadmap

This document records the intended thesis progression after validating the
three initial eight-module morphologies.

## Separation of responsibilities

- A morphology behavior describes a reusable capability such as
  `crawl_stairs_arch_wave`, `gap_crossing`, or `press_button`. It must be parameterized by
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

The same separation now applies to Snake8 `gap_crossing`. The isolated
`coplanar_gap_v1` generator samples a reproducible conservative family of
equal-height, world-+X gaps and the headless evaluator records exact manifests
plus an independent terminal geometry check. These per-obstacle campaigns are
the first dataset stratum. A later course generator will compose variable
obstacle presence and order into complete task-achievement trajectories while
retaining every component seed and feasibility decision.

The first per-obstacle collection is curriculum gated and headless.  Stairs
and gaps each begin with the physically supported `robust` envelope, advance
to `intermediate` and then `challenging` only when the completed level reaches
the configured success-rate threshold, and write separate datasets.  Failed
rollouts remain in `all_transitions.jsonl`; only independently verified
successes are copied to `successful_transitions.jsonl` for BC/DAgger.

These records are graph trajectories, not pose-only trajectories.  Every
transition includes the attributed dynamic robot graph, the following graph
snapshot when available, complete current and target topology, connector
relations, module roles and assignment, per-module actuator state and
operational DoF interpretation, course geometry, behavior phase and expert
action.  The isolated stair and gap strata deliberately hold morphology
constant at Snake8.  The later complete-task stratum will add morphology
choice and reconfiguration-induced topology changes to the same graph-based
representation.

Environment conditioning is part of each transition as
`observation.environment`, not only a campaign-side manifest.  For the
isolated generators it contains Isaac world ground truth: coordinate frame,
scenario profile and seed, exact obstacle landmarks/dimensions, curriculum
difficulty and module geometry.  Later perception tensors or extracted
features should be stored alongside this oracle description so train-time
inputs can be selected without making the demonstrations irreproducible.

## Dataset-format design basis

The transition layout follows established robotics-learning conventions while
remaining JSONL and graph native:

- [DAgger (Ross, Gordon and Bagnell, 2011)](https://proceedings.mlr.press/v15/ross11a.html)
  motivates retaining expert labels on the state distribution that later
  learner rollouts actually visit, including recovery states;
- [RLDS](https://github.com/google-research/rlds) motivates explicit episode
  IDs and `is_first`/`is_last`/`is_terminal` step semantics plus environment
  configuration metadata;
- [robomimic](https://robomimic.github.io/docs/datasets/overview.html)
  motivates paired current and next observations instead of action-only logs;
- [Open X-Embodiment](https://robotics-transformer-x.github.io/) motivates
  task/instruction conditioning alongside multimodal observations;
- [Learning Modular Robot Control Policies](https://arxiv.org/abs/2105.10049)
  motivates preserving the module connection graph and local module state as
  first-class policy inputs rather than flattening one fixed morphology.

`task_context` and `environment` are policy-conditioning candidates.
`expert_annotation` is deliberately separate privileged supervision: it
contains FSM phase, active primitive and deterministic rationale and must not
silently become a deployment input unless an upstream task estimator can
produce the same signal.  `supervision` records label/execution source now so
future DAgger rollouts can distinguish learner actions, expert labels and
interventions.
