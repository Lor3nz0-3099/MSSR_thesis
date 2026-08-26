# Validated baseline: Snake8 `gap_crossing`

This record protects the first physically successful Snake8 flat-gap crossing
baseline from being lost during later mobile-manipulator and curriculum work.

## Provenance

- Validation date: 2026-08-26.
- Branch: `nav2-integration`.
- Final controller commit: `b8bb2d9` (`Preload Snake8 far-bank traction`).
- Development sequence retained in Git:
  - `d3212c3` replaces unreachable fixed-X goals with chord-feasible geometry;
  - `0921628` progressively releases the arch onto the far support;
  - `b8bb2d9` adds far-bank traction preload without changing physics.
- Implementation:
  `mssr_ws/src/mssr_expert/mssr_expert/behaviors/snake_gap_gait.py`.

## Physically validated fixture

The successful Isaac Sim 6.0 run used eight SMORES-EP modules assembled as
`snake8`, aligned approximately with world `+X`, and two coplanar banks:

- near edge: world `x = 0.550 m`;
- far edge: world `x = 0.750 m`;
- gap width: `0.200 m`;
- measured wheel radius: approximately `0.03106 m`;
- connected-chain spacing: approximately `0.078--0.082 m` during the run;
- seven rigid connected faces retained throughout the crossing.

The complete connected chain reached the far bank. During the terminal
transfer the tail was raised by the shortening arch while the landed anterior
section retained wheel contact through a shallow concave traction preload.

## Validated command

```bash
run_behavior \
  snake8 \
  gap-traction-preload-001 \
  gap_crossing \
  '{"approach_linear_m_s":0.050,"linear_m_s":0.040,"gap_profile_substeps":3,"far_bank_transition_links":1.0,"arch_clearance_wheel_radii":2.0,"landing_release_support_modules":3,"landing_release_ramp_links":1.0,"far_bank_traction_preload_wheel_radii":0.25,"gap_goal_tolerance_m":0.004}'
```

## What is dimension-parametric

The behavior does not embed the validated `0.200 m` width or either edge
coordinate. For each execution it reads from the live robot graph:

- near and far gap edges in world coordinates;
- measured connected-chain spacing;
- measured wheel radius;
- live TILT limits and initial heading.

Those observations determine the safe supports, admissible unsupported-link
count, arch span and amplitude, chord-feasible module positions, progressive
far-bank release, traction preload and every longitudinal completion goal.
Changing the declared coplanar gap width therefore regenerates the complete
program rather than scaling a recorded timer or replaying fixture coordinates.

## What was validated

- complete Snake8 self-assembly and morphology assignment;
- geometric near-edge approach without a locomotion timeout;
- chord-feasible head goals through the curved chain;
- a positive traveling backbone arch migrating from head to tail;
- all eight wheel pairs commanded in every traction phase;
- progressive far-bank landing after sufficient anterior support;
- automatic selection of the next landed module from chain geometry;
- shallow far-bank preload restoring anterior wheel traction;
- complete tail transfer and arrival of all eight modules on the far bank.

## Explicit limits

This result is dimension-parametric, not a claim of universal robustness. The
physically certified case remains the 200 mm fixture above. The deterministic
expert currently assumes:

- exactly eight modules in the `snake8` serial topology;
- coplanar, parallel banks and a gap perpendicular to world `+X`;
- gap metadata visible in the live robot graph;
- initial heading inside the configured admission limit;
- a width below the safe unsupported-span bound derived from live geometry;
- Isaac world poses as the simulated external-localization/Vicon source.

Wider or narrower conservative fixtures require a robustness campaign before
their successful episodes may be admitted to the behavior-cloning dataset.
Gaps above the computed five-link safe span are rejected rather than attempted.
Non-coplanar banks, oblique crossings and unknown gap heading remain future
generalization tasks for the observation-driven expert and learned policy.

## Regression rule

The command and fixture above must remain a named reference episode. Changes
to the gap gait are accepted only after unit tests pass and this complete GUI
or headless episode again places all eight connected modules on the far bank.
Do not replace geometric completion with a timer, relax the independent final
metric, or change contact physics to hide a gait regression.
