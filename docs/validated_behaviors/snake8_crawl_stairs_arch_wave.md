# Validated baseline: Snake8 `crawl_stairs_arch_wave`

This record protects the first physically successful Snake8 stair-climbing
baseline from being lost during later curriculum and morphology work.

## Provenance

- Validation date: 2026-08-25.
- Branch: `nav2-integration`.
- Final baseline commit: `70a3bdc` (`Complete Snake8 arch wave after tail lift`).
- Development sequence retained in Git:
  - `4c0076e` adds the separate experimental arch wave;
  - `0174120` adds upper-riser pre-lift;
  - `fc0b5ca` lands the head after each arch;
  - `70a3bdc` completes after the tail is geometrically lifted.
- Implementation:
  `mssr_ws/src/mssr_expert/mssr_expert/behaviors/snake_stair_gait.py`.
- The older `crawl_stairs` profile follower remains available and was not
  replaced.

## Physically validated fixture

The successful Isaac Sim 6.0 run used eight SMORES-EP modules assembled as
`snake8`, aligned approximately with world `+X`, and a uniform staircase:

- three risers;
- rise: `0.065 m` per riser;
- tread depth: `0.280 m`;
- first riser: world `x = 0.650 m`;
- top elevations: `0.065`, `0.130`, `0.195 m`;
- measured wheel radius: approximately `0.03106 m`;
- measured connected-chain spacing: approximately `0.07777 m`.

The observed terminal configuration placed the complete connected chain on
the upper deck.  The final correction in `70a3bdc` stops after the module next
to the tail is one wheel radius beyond the final riser; it does not demand an
additional traction-only distance while the last wheels have no useful
contact.

## Validated command

```bash
run_behavior snake8 stair-arch-reference-01 crawl_stairs_arch_wave \
  '{"riser_approach_linear_m_s":0.060,"riser_approach_tolerance_m":0.010,"linear_m_s":0.040,"profile_substeps":6,"transition_clearance_m":0.0065,"crawl_goal_tolerance_m":0.004}'
```

`upper_deck_advance_distance_m` is intentionally omitted.  Its arch-wave
default is zero after the geometrically verified tail lift.

## What was validated

- complete self-assembly followed by morphology assignment;
- live world-frame module feedback and live stair metadata;
- geometric approach to the first riser without a locomotion timer;
- repeated broad-arch/pre-lift/landing waves at later risers;
- all eight wheel pairs commanded during traction phases;
- progressive release of faces that would otherwise catch a tread edge;
- recurring head lift before the next riser while the tail completes the
  preceding transition;
- geometric terminal completion with all eight modules on the upper deck.

The same reference was revalidated headless on 2026-08-25 at real-time factor
`1.0`.  The command client returned success and the independent terminal
metric measured eight modules, seven connections and a minimum module-centre
height of `0.225039 m` (required minimum `0.211060 m`).

## Explicit limits

This single physical run does not prove universal stair robustness.  The
baseline assumes:

- exactly eight modules in the `snake8` serial topology;
- uniform rises and tread depth;
- staircase direction world `+X`;
- rise smaller than live link spacing;
- initial heading error below the behavior admission limit;
- Isaac ground-truth localization as the simulated Vicon equivalent.

The initial robustness campaign must stay inside the conservative sampling
envelope of 50--65 mm rise, 250--320 mm tread depth and 2--4 risers.  A seed is
successful only when the behavior reports success **and** an independent
robot-graph check finds eight modules, seven latched connections and every
module centre at the upper-deck elevation within the declared tolerance.

## Regression rule

The reference fixture and command above must remain a named batch episode.
Changes to the gait are accepted only after static/unit tests pass and the
reference headless episode remains successful.  Failed randomized episodes
are diagnostic rollouts, never successful behavior-cloning demonstrations.
The physics/ROS timing reference is `simulation_speed_factor=1.0`: a 2x
diagnostic run lost alignment during the first transition and therefore must
not be used to certify or collect demonstrations.
