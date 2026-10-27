# MSSR MAPPO/APPO Phase 1 — RC-Car8 Training Specification

## Goal

Train the BC-v4-warm-started shared MARL actor to control an assembled
`rc_car8` from an assembly platform to a target platform through a seeded,
procedurally generated road containing static cone obstacles.

## Episode

1. Expert assembles `rc_car8` on the start platform.
2. Independent topology verification succeeds.
3. Authority passes to MARL.
4. MARL drives along the road and avoids static cones.
5. Success occurs only when all 8 module centres are inside the target
   platform.
6. Cone collision penalizes reward but does not immediately terminate.
7. Stagnation may later terminate the episode after a configurable timeout.

## Terrain

- Start platform.
- Target platform in a seed-dependent location.
- Physical road connecting them.
- No physical side walls.
- Road may curve.
- Static, non-movable cones are randomized by seed.
- The course must remain forward-only in Phase 1.
- Reverse-driving requirements are deferred to a later curriculum phase.
- Cone clearance is based on the measured physical RC-Car8 footprint plus
  explicit tolerance.
- Reuse the validated RC planar geometry rather than creating an unrelated
  generator.

## Reward architecture

Reward is decomposed into raw, independently logged terms before weights are
chosen.

Core Phase-1 terms:

- progress along road;
- progressive entry of modules onto target platform;
- terminal success when 8/8 modules are inside;
- outside-road penalty;
- cone-collision penalty;
- action-conflict penalty;
- invalid-action penalty;
- redundant/extra-action penalty;
- control-effort penalty;
- time cost;
- stagnation penalty.

Do not choose numerical weights until raw term scales have been measured.

Extra-action cost must penalize redundant task-irrelevant actions, not every
active action indiscriminately.

## Future reward composition

Assembly and reconfiguration share the same topological/geometric reward
family.

Composite episodes use:

    common core reward
    + reward terms for the currently active subtask

Inactive task reward families must not contribute.
