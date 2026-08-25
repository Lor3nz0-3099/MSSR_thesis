# SMORES-EP eight-module obstacle-course runbook

This runbook keeps all eight physical modules in the connected robot through
the complete route. The implemented morphology sequence is:

```text
RC Car8 -> Snake8 -> MobileManipulator8 -> RC Car8
```

RC Car8 starts on a lower rear platform and climbs a physical ramp before the
gap platform. It then reconfigures to Snake8 while all modules remain on the
near platform. Snake8 spans the compact gap with its rigid serial train gait,
then supplies the articulated stair gait. This avoids the previous
Snake8-to-Bridge8 face-replacement transition at the gap edge.

All direct transitions among `snake8`, `bridge8`, `mobile_manipulator8` and
`rc_car8` are planner-tested with eight assigned modules and no reserves.

## Unified task-achievement node

`mssr_smores_obstacle_course_node` executes the full route in one ROS 2
process. It composes the existing parallel self-assembly executor,
self-reconfiguration executor and morphology behavior executor; no manual
handoff between behavior commands is required. Its capability policy selects
`RC Car8` for the initial ramp and final exit, `Snake8` for the gap and
stairs, and `MobileManipulator8` for the button.

Isaac is the authority for course geometry. Its state graph exports the gap
edges, stair heights and riser spacing, button center and exit pose. The node
uses live module poses with those landmarks: RC Car8 climbs the ramp and
stops before the gap, then reconfigures safely to Snake8. The Snake8 train
spans and clears the gap, verifies each stair-height progression, aligns the
manipulator at a button stand-off pose, and crosses the exit plane with RC
Car8. Snake8 also exposes the direct `straighten` behavior used by the
manual posture smoke test below.

Closed-loop approach phases run at `0.05 m/s` and the bridge crossing uses
`0.03 m/s`; these are below the configured morphology speed limits while
being substantially faster than the previous defaults.

The node verifies the button geometrically: the module assigned to the
`end_effector` role must be within `button_contact_radius_m` of the plunger
center. It verifies the finish only when at least
`goal_min_modules_past_exit` modules have crossed the Isaac-exported exit
plane. A failed or interrupted route is discarded; only a full successful
episode is appended to the JSONL dataset, retaining the task graph, selected
morphology and expert action for IL.

After starting Isaac and the file bridge as below, run:

```bash
cd ~/MSSR_thesis
source /opt/ros/humble/setup.bash
source mssr_ws/install/setup.bash
ros2 run mssr_expert mssr_smores_obstacle_course_node --ros-args \
  -p episode_id:=course-0001 \
  -p dataset_path:=$PWD/logs/datasets/course-0001.jsonl
```

Do not start `mssr_smores_morphology_behavior_node`,
`mssr_smores_self_assembly_node`, or
`mssr_smores_self_reconfiguration_node` for the same run: they would publish
competing action and primitive-goal streams. Restart Isaac after rebuilding so
its state publisher includes the course landmarks, rear platform and ramp.

### SMORES literature basis

The policy and execution assumptions refer to the SMORES material archived in
`references/`: `Design of the SMORES system.pdf`, `SMORES-EP.pdf`, `design and
characterization of the EP-Face Connector.pdf`,
`chao_smores_reconfiguration_2019.pdf`, `Accomplishing High-Level Tasks with
Modular Robots.pdf`, and `An Integrated System for Perception-Driven Autonomy
with Modular Robots.pdf`. The present implementation uses their design and
connector/reconfiguration concepts as deterministic simulation assumptions;
it does not claim a hardware validation beyond the existing backend evidence.

## Start the common runtime

Build once after pulling or editing the package:

```bash
cd ~/MSSR_thesis
source /opt/ros/humble/setup.bash
cd mssr_ws
colcon build --symlink-install --packages-select mssr_expert
cd ..
```

Terminal 1 — start one eight-module Isaac episode. Locomotion remains at
`4.0x`, while the deliberately exaggerated `8.0x` TILT profile supplies
18.4 Nm and a 96 Nm/rad hold for cantilevering the connected chain. Wheel
damping remains nominal and the `240 Hz` physics step keeps wheel-ground
contact stable during train locomotion:

```bash
cd ~/MSSR_thesis
bash scripts/smores_ep/run_self_assembly.sh \
  --module-count 8 \
  --obstacle-course \
  --performance \
  --physics-hz 240 \
  --actuator-effort-scale 4.0 \
  --tilt-effort-scale 8.0
```

Terminal 2 — keep the file bridge alive for the entire route:

```bash
cd ~/MSSR_thesis
source /opt/ros/humble/setup.bash
python3 ros2_bridge/mssr_file_bridge.py --publish-period 0.2
```

Terminal 3 — source the expert package:

```bash
cd ~/MSSR_thesis
source /opt/ros/humble/setup.bash
source mssr_ws/install/setup.bash
```

Terminal 4 — start one behavior node and leave it alive. It follows the
transient task graph published by each successful assembly/reconfiguration:

```bash
cd ~/MSSR_thesis
source /opt/ros/humble/setup.bash
source mssr_ws/install/setup.bash
ros2 run mssr_expert mssr_smores_morphology_behavior_node
```

### Isolated Snake8 stair stage

Before running the complete course, the stair gait can be validated on a
dedicated stage without Nav2, the gap, the button, or the task-achievement
node. Its reference preset contains three 65 mm risers with 280 mm treads.
The same generator also accepts an explicit uniform rise, tread depth, count
and first-riser position, or a reproducible conservative `stair_seed`. Start simulation,
file bridge and the single morphology behavior node together:

```bash
cd ~/MSSR_thesis/mssr_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch mssr_expert smores_runtime.launch.py \
  stair_test_course:=true \
  performance:=true
```

For example, a seeded conservative fixture is selected without changing the
behavior:

```bash
ros2 launch mssr_expert smores_runtime.launch.py \
  stair_test_course:=true \
  stair_seed:=17 \
  performance:=true
```

An explicit fixture can instead use `stair_rise_m:=0.055`,
`stair_depth_m:=0.310`, `stair_count:=4` and
`stair_first_riser_x_m:=0.700`.  Geometry and robot-graph metadata are built
from the same immutable specification.

The Isaac assembly executor uses 55 mm/s for collision-aware free-space
staging and 35 mm/s for the local connector-alignment arc.  The final magnetic
approach remains limited to 25 mm/s, preserving the validated contact gate.
These are command-speed changes only; actuator effort, damping, contact
material and physics frequency are unchanged.

Do not also pass `obstacle_course:=true`; the two physical stages are
mutually exclusive. Start only the Snake8 self-assembly expert in another
sourced terminal, using the command in section 1. After assembly, define
`run_behavior` as below. The isolated stage and full obstacle course now use
the same uniform geometry: 65 mm rise and 280 mm tread.

The preferred complete behavior is generated from the live geometry. Start
with Snake8 straight, aligned with world +X, and with its head close to the
first riser:

```bash
run_behavior snake8 stair-crawl-01 crawl_stairs \
  '{"riser_approach_linear_m_s":0.060,"riser_approach_tolerance_m":0.010,"linear_m_s":0.040,"profile_substeps":6,"transition_clearance_m":0.0065,"head_prelift_lookahead_m":0.080,"head_prelift_ramp_m":0.040,"head_hook_transfer_m":0.040,"head_overstep_clearance_m":0.010,"crawl_goal_tolerance_m":0.004,"upper_deck_advance_distance_m":0.080}'
```

The original profile follower above remains available.  The experimental
SMORES-inspired broad arch wave is a separate behavior, so it can be validated
without replacing the known baseline:

```bash
run_behavior snake8 stair-arch-01 crawl_stairs_arch_wave \
  '{"riser_approach_linear_m_s":0.060,"riser_approach_tolerance_m":0.010,"linear_m_s":0.040,"profile_substeps":6,"transition_clearance_m":0.0065,"crawl_goal_tolerance_m":0.004}'
```

`crawl_stairs_arch_wave` preserves `ARCH_00_*` at exactly the same TILT
targets as the validated first-riser `PROFILE_00_*` wave.  Starting with the
second riser it distributes one rise over two links, rather than concentrating
it in one nearly vertical link.  Halfway through every transfer it adds the
temporary vertical margin `arch_clearance_m`, then returns
to the geometry-derived settled angle at the endpoint.  Before each upper
riser, the terminal module is raised by a recurring v6/v7 hook and the hook is
then cross-faded into the broad v5/v7 arch.  The default lookahead is one
measured chain link plus the transition margin; its ramp is half a measured
link and its temporary arch clearance is 40% of the wheel radius.  These
defaults therefore scale with live module and stair metadata; each may still
be overridden for an experiment.  `ARCH_HEAD_GATE_01`, then `_02`, stop from
the live world-X pose of `snake_head` at each upcoming riser minus that
lookahead; the prelift is therefore not inferred from a timer or from the pose
of an internal edge module.  Once that head goal has crossed the riser, the
temporary v5/v7 preview is released: the ordinary arch migrates its descending
bend rearward, returns the head TILT to neutral and lowers its wheels onto the
new tread.  This landing release is repeated at every upper riser and prevents
the preview arch from accumulating on top of the moving chain profile.
Combined broad-arch and terminal-hook
targets are bounded by the smallest live TILT actuator limit, with a 0.03 rad
default safety margin, instead of relying on Isaac-side saturation.  All eight
wheel pairs remain commanded and every drive phase still terminates from a
live world-X goal; the arch wave introduces no locomotion timer.

The final `ARCH_TAIL_LIFT_COMPLETE` phase stops when the module adjacent to
the tail is one measured wheel radius beyond the last riser.  At that point
the final posture has already lifted the tail onto the top-deck elevation, so
the arch-wave completes without an extra traction-only run.  An optional
`upper_deck_advance_distance_m` greater than zero remains available for
experiments, but is deliberately disabled by default; ordinary train/Nav2
locomotion should perform any subsequent top-deck travel.

This is dimension-parametric within its declared model, not yet a universal
stair controller.  It currently requires an eight-module chain aligned with
uniform stairs in world +X and rejects non-uniform rises.  It reads the first
riser, tread depth, top heights, wheel radius and live link spacing instead of
embedding the coordinates of the test fixture.  Supporting variable rises,
curved approaches or an unknown stair heading requires a more general course
observation and is intentionally left explicit rather than hidden behind
fixture-specific constants.

The physically successful reference and its exact Git provenance are frozen
in `docs/validated_behaviors/snake8_crawl_stairs_arch_wave.md`.

### Isolated Snake8 gap stage

The isolated gap fixture has two coplanar banks separated by a real 200 mm
opening in world +X.  It is deliberately independent from the older Bridge8
timed behavior.  Start the complete GUI runtime with:

```bash
cd ~/MSSR_thesis/mssr_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch mssr_expert smores_runtime.launch.py \
  gap_test_course:=true \
  performance:=true
```

Assemble Snake8 with the command in section 1 and then execute:

```bash
run_behavior snake8 gap-crossing-01 gap_crossing \
  '{"approach_linear_m_s":0.050,"linear_m_s":0.040,"drawbridge_lift_angle_rad":1.20,"gap_goal_tolerance_m":0.004}'
```

`gap_crossing` has no locomotion timer.  It reads the two gap edges, live
module positions, measured chain spacing, wheel radius and TILT limits from
the robot graph.  Its closed-loop program is:

```text
RESTORE_GAP_NEUTRAL -> PRELIFT_HEAD_DRAWBRIDGE
-> LIFT_HEAD_DRAWBRIDGE -> STRAIGHTEN_HEAD_DRAWBRIDGE
-> ADVANCE_HEAD_PIVOT_TO_EDGE -> CONFORM_GAP_PROFILE_00
-> (CONFORM_GAP_PROFILE_N -> FOLLOW_GAP_PROFILE_N)*
-> LIFT_TAIL_DRAWBRIDGE
-> PULL_TAIL_TO_SAFE_LANDING -> LOWER_TAIL_ON_FAR_BANK -> CLEAR_FAR_EDGE
```

The planner computes how many terminal modules must be lifted from the live
gap width, wheel radius, link spacing and support margins.  It raises that
entire segment about one hinge by 1.20 rad by default: no downstream
counter-bend cancels the drawbridge slope.  For the nominal 200 mm gap, `v3`
is the fifth module from the head/right and receives positive TILT so that
`v4..v7` form the raised drawbridge.  A temporary positive pre-fold on `v4`
first selects the three-module head side of the otherwise symmetric flat
chain.  After `v3` reaches its positive drawbridge angle, `v4` returns to
neutral, leaving the four anterior modules in one rigid raised segment.  The
same rule is gap-parametric: narrower gaps select a more anterior pivot
directly, while wider admissible gaps migrate the pre-fold rearward through
as many adjacent hinges as required by the computed suspended-module count.
The default transfer angle is divided by that migration depth and remains
operator-overridable as `drawbridge_prelift_angle_rad`.  The tail lift uses
the spatially mirrored sign.  The grounded pivot is then driven
to the near edge before the rigid segment is curved upward across the opening,
with at least one supported module on each bank.  It is not flattened while
links remain over the opening.  Instead, a positive sine arch is fixed in the
world interval from the safe near support to the safe far support.  Before
each short geometric advance, the planner samples that arch at the translated
module positions and converts the non-negative center heights into distributed
TILT differences.  The material modules therefore follow the same upward
shape as it travels rearward through the chain.  The arch amplitude defaults
to one measured wheel radius plus the edge margin, while the spatial sampling
defaults to three substeps per measured link and is configurable with
`gap_profile_substeps`.  When the first front anchor is geometrically beyond
the far edge, the current arch transitions directly into the tail lift rather
than flattening over the opening.  The same number of tail modules is then
raised through the spatially mirrored hinge;
the front anchor is pulled far enough that lowering the tail leaves its last
wheel beyond the far edge.
The gait rejects gaps wider than its measured three-link span, a misaligned
chain, inconsistent world landmarks, or legacy duration parameters.  This is
the deterministic IL expert for equal-height banks; it is not yet a learned
general gap controller.

### Isolated MobileManipulator8 button stage with Nav2

This fixture contains only a continuous flat platform, a wall and its button.
Start from Snake8 exactly as requested; the common launch keeps Isaac, bridge
and morphology behavior node alive through the reconfiguration:

```bash
cd ~/MSSR_thesis/mssr_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch mssr_expert smores_runtime.launch.py \
  button_test_course:=true \
  performance:=true
```

In a second sourced terminal assemble Snake8 using section 1.  Stop only that
self-assembly process after completion, then run:

```bash
cd ~/MSSR_thesis
source /opt/ros/humble/setup.bash
source mssr_ws/install/setup.bash
ros2 run mssr_expert mssr_smores_self_reconfiguration_node --ros-args \
  -p source_graph_path:=auto \
  -p target_morphology:=mobile_manipulator8 \
  -p execution_id:=button-snake8-to-manip8-01 \
  -p episode_id:=button-snake8-to-manip8-01 \
  -p dataset_path:=$PWD/logs/datasets/button_manip8_reconfiguration.jsonl
```

Only after `Self-reconfiguration completed.`, start Nav2 in a third sourced
terminal. MobileManipulator8 now exports its own role-anchored `/odom` and
`odom -> base_link` frame, oriented from `arm_ground_drive` toward
`front_support`:

```bash
cd ~/MSSR_thesis/mssr_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch mssr_expert smores_nav2.launch.py
```

The stage exports the map-frame standoff pose `(x=0.85, y=0.275, yaw=pi/2)`.
Send it from a fourth sourced terminal:

```bash
ros2 action send_goal /navigate_to_pose nav2_msgs/action/NavigateToPose \
  '{pose: {header: {frame_id: map}, pose: {position: {x: 0.85, y: 0.275, z: 0.0}, orientation: {x: 0.0, y: 0.0, z: 0.70710678, w: 0.70710678}}}}' \
  --feedback
```

After Nav2 reports success, stop its residual velocity and exercise the arm:

```bash
run_behavior mobile_manipulator8 manip-stop-at-button-01 stop '{}'
run_behavior mobile_manipulator8 manip-press-01 press_button '{}'
run_behavior mobile_manipulator8 manip-release-01 release_button '{}'
```

Do not start Nav2 before reconfiguration completes: until a morphology with a
navigation profile is assigned, the behavior node intentionally has no valid
virtual base pose.

### Reproducible headless robustness batch

Build and source the workspace, then create a fresh output directory for each
campaign:

```bash
cd ~/MSSR_thesis
source /opt/ros/humble/setup.bash
source mssr_ws/install/setup.bash

python3 scripts/smores_ep/run_stair_headless_batch.py \
  --seeds 0:3 \
  --include-reference \
  --continue-on-failure \
  --output-dir logs/stair_headless_batch/campaign_001
```

Keep the default `--simulation-speed-factor 1.0` for certified rollouts.  A
2x diagnostic run changed the coupled ROS/physics timing and lost alignment;
headless mode still removes GUI/rendering overhead without changing simulated
time relative to the control loops.

Each episode starts Isaac without a GUI, assembles Snake8, executes the
validated arch wave, and stores `manifest.json`, process logs, the assembly
dataset and `result.json`.  A success requires both the terminal behavior
status and an independent final robot-graph check: eight module poses, seven
latched connections and all module centres at the top-deck elevation.  The
wall-clock limits are process-stall guards only; gait transitions continue to
use live geometric goals.  `--plan-only` generates manifests without starting
Isaac.  Failed episodes remain diagnostics and are not positive BC samples.

`crawl_stairs` reads module poses and Isaac stair landmarks from the robot
graph. It measures the connected-module pitch and derives the bend angle from
the actual rise. A riser becomes an inclined link followed by an opposite bend
back onto the tread. When the chain spans two risers, both bend pairs remain
active simultaneously. The terminal `v6/v7` TILT pair lifts only the head
after the preceding support wheel is completely beyond the tread edge; the
neck remains wheel-supported and keeps providing traction while the preceding
bend moves toward the tail. The trigger uses the world-X goal completed by the
previous traction microstep, rather than the future theoretical wave pose. The
head stays raised until its wheel reaches the next riser. The controller then
migrates the bend progressively from `v6/v7` to `v5/v6` over
`head_hook_transfer_m`, forming the next hook only after the head has support
available at the edge. `head_prelift_lookahead_m` bounds how late the lift may
start; on the 65 mm fixture, full support is the stricter condition and leaves
about 52 mm before nominal contact after including transition clearance and
goal tolerance. `head_prelift_ramp_m` completes the initial lift over 40 mm.
`head_overstep_clearance_m` temporarily adds 10 mm to the measured rise, so
the leading link reaches about 75 degrees and keeps the head underside above
the next tread edge.  The extra angle migrates one joint rearward and then
cross-fades back to the exact stair-height angle once the wheel has crossed.
Each one-link shift is divided into
`profile_substeps` posture/traction microsteps (six by default), approximating continuous
follow-the-leader motion without driving wheels while TILT joints move.

All locomotion phases are closed-loop rather than timed. After lifting the
head, the controller keeps the five grounded modules moving until the leading
wheel surface of the first elevated module (`v5`) reaches the first-riser
plane. That surface is reconstructed from its live world-X center and the
wheel radius exported by Isaac. Each subsequent traction phase targets the
live world-X position of the wheel transferring over the corresponding riser,
rather than accumulating a centroid displacement. During the middle of a
one-link transfer, `transition_clearance_m` temporarily advances the next
posture: the module beyond the edge straightens sooner while the bend moves
rearward onto the grounded chain. The default lead is 10% of the recognized
riser height; it returns to zero at both endpoint profiles, so it does not
accumulate along the stairs. The first-riser wave is left unchanged. From the
second riser onward, `upper_riser_edge_release_lead_m` advances only the
declining angle of the module leaving the upper tread edge. By default the
lead is derived from the measured module spacing: half one link minus the
existing transition clearance (about 32.4 mm for Snake8). Consequently that
module is horizontal by the midpoint of the transfer, clearing its BOTTOM
face without prematurely moving the two supporting bends. The correction
returns to the exact profile at both ends and repeats for all later risers.

Each traction phase drives the
union of wheels supporting the posture before and after the transition. This
includes wheels entering or leaving contact at a riser, preventing a passive
wheel from jamming against the tread edge while the grounded tail and leading
modules compress the chain. The upper-deck phase advances the centroid of all
eight modules by a geometric
distance (one measured link by default). Status messages report current X,
distance traveled, target and remaining error. There is no locomotion time
limit: a slow but progressing phase continues until its geometric goal is
reached. If any required live pose is temporarily unavailable, the controller
publishes no wheel commands and waits for feedback. Primitive joint timeouts
remain only as actuator-failure guards; they do not determine stair progress.
Admission is rejected if the chain differs from the +X stair direction by
more than `max_alignment_error_rad` (default 0.35 rad).

The following commands are retained for inspecting the individual legacy
postures and traction groups:

```bash
run_behavior snake8 stair-neutral-01 straighten '{}'
run_behavior snake8 stair-rise-01 lift_head '{}'
run_behavior snake8 stair-lifted-approach-01 approach_step_lifted \
  '{"riser_approach_linear_m_s":0.060,"riser_approach_duration_s":4.0}'
run_behavior snake8 stair-hook-01 hook_step '{}'
run_behavior snake8 stair-hook-advance-01 advance_hooked_front \
  '{"linear_m_s":0.030,"duration_s":3.0}'
run_behavior snake8 stair-shift-01 transfer_step_1 '{}'
run_behavior snake8 stair-shift-advance-01 advance_after_transfer_1 \
  '{"linear_m_s":0.030,"duration_s":3.0}'
run_behavior snake8 stair-shift-02 transfer_step_2 '{}'
run_behavior snake8 stair-shift-advance-02 advance_after_transfer_2 \
  '{"linear_m_s":0.030,"duration_s":3.0}'
run_behavior snake8 stair-shift-03 transfer_step_3 '{}'
run_behavior snake8 stair-shift-advance-03 advance_after_transfer_3 \
  '{"linear_m_s":0.030,"duration_s":3.0}'
run_behavior snake8 stair-reset-01 straighten '{}'
```

The initial lift shortens the horizontal projection of the robot. Therefore,
`approach_step_lifted` drives only the five modules still on the ground before
the hook is formed; the three vertical modules remain stopped. The square
profile then has a horizontal ground section, one nearly vertical 77.77 mm
link, and a horizontal two-module hook. The three transfer commands move that
profile toward the tail one link at a time. After the hook and after every
transfer, only the modules already supported by the upper tread advance: first
`v6..v7`, then `v5..v7`, `v4..v7`, and finally `v3..v7`. The legacy
single-riser timed program remains available only for comparison and is not
used by the KAIRO/Tanaka course policy:

```bash
run_behavior snake8 stair-pull-01 pull_over_step \
  '{"riser_approach_linear_m_s":0.060,"riser_approach_duration_s":4.0,"linear_m_s":0.030,"front_pull_duration_s":3.0,"transfer_pull_duration_s":3.0,"tread_advance_duration_s":5.0}'
```

Optional monitoring terminals:

```bash
ros2 topic echo /mssr/expert/self_reconfiguration/state std_msgs/msg/String
```

```bash
ros2 topic echo /mssr/morphology/status std_msgs/msg/String
```

Define this helper in any sourced ROS terminal. It delegates JSON encoding and
status matching to the validated ROS client. Every call blocks until the same
command ID reaches a terminal state, so a failed behavior cannot be mistaken
for a completed one. The client waits indefinitely by default because
geometric completion, rather than elapsed wall time, terminates obstacle
behaviors. Pass `--timeout-s N` directly to the client only when a bounded
operator-side wait is explicitly desired:

```bash
run_behavior() {
  local morphology="$1"
  local command_id="$2"
  local behavior="$3"
  local parameters="${4:-}"
  if [[ -z "$parameters" ]]; then
    parameters='{}'
  fi
  ros2 run mssr_expert mssr_smores_morphology_command_client \
    --morphology "$morphology" \
    --command-id "$command_id" \
    --behavior "$behavior" \
    --parameters-json "$parameters"
}
```

## 1. Assemble and test Snake8

Run in Terminal 3:

```bash
ros2 run mssr_expert mssr_smores_self_assembly_node --ros-args \
  -p target_graph_path:=$PWD/mssr_ws/src/mssr_expert/config/smores_snake8.json \
  -p execution_id:=course-snake8-assembly-01 \
  -p episode_id:=course-snake8-assembly-01 \
  -p dataset_path:=$PWD/logs/datasets/course_snake8_assembly.jsonl
```

After `Parallel self-assembly completed.`, test the neutral posture and the
restored train gait on the start platform. Positive Snake8 motion is toward
`snake_head` (`v7`):

```bash
run_behavior snake8 snake-straight-01 straighten '{}'
run_behavior snake8 snake-train-01 train '{"linear_m_s":0.025,"duration_s":6.0}'
```

`straighten` is a posture command, not a drive command. The behavior node
captures all eight TILT coordinates after assembly and restores those exact
neutral values in one coordinated barrier. It does not assume that the
loaded straight chain has zero-valued joints. `train` energizes all eight
wheel pairs in the same control cycle;
PAN and TILT remain under rigid position hold instead of becoming passive.
The live quaternion projection selects the correct local wheel sign for each
assigned module.

The stair gait is used after crossing the gap and returning to Snake8.

Stop only the self-assembly expert in Terminal 3 with `Ctrl-C`. Leave Isaac,
the bridge and the behavior node running.

## 2. Reconfigure Snake8 to Bridge8 and test the suspended span

```bash
ros2 run mssr_expert mssr_smores_self_reconfiguration_node --ros-args \
  -p source_graph_path:=auto \
  -p target_morphology:=bridge8 \
  -p execution_id:=course-snake8-to-bridge8-01 \
  -p episode_id:=course-snake8-to-bridge8-01 \
  -p dataset_path:=$PWD/logs/datasets/course_reconfiguration.jsonl
```

After `Self-reconfiguration completed.`, run the complete gap-crossing
program. The bridge now keeps the chain rigid and uses the internal joints as
a distributed lever: the rear half lifts the front span, the rear modules
push that lever forward, the span lands, and then the front half pulls the
tail clear. Wheel signs are projected from live body orientations so every
active module pushes in one world direction.

```bash
run_behavior bridge8 bridge-cross-01 cross_gap '{"linear_m_s":0.012,"approach_duration_s":4.0,"rear_push_duration_s":4.0,"front_transfer_duration_s":6.0,"rear_clear_duration_s":4.0}'
```

`cross_gap` includes the edge approach and both lift/land ramps. Use
`deploy_span` separately only when diagnosing the static front-lever posture.

The behavior commands all eight TILT joints through a coordinated barrier and
enforces this stopped sequence:

```text
APPROACH_EDGE -> LIFT_FRONT_PRELOAD -> LIFT_FRONT -> ADVANCE_REAR
              -> LOWER_FRONT -> LAND_FRONT -> TRANSFER_FRONT
              -> LIFT_REAR_PRELOAD -> LIFT_REAR -> CLEAR_REAR
              -> RETURN_FLAT
```

`APPROACH_EDGE` drives all eight modules as a train. `ADVANCE_REAR` drives the
rear half (`v0..v3`) while the front span is lifted. `TRANSFER_FRONT` and
`CLEAR_REAR` drive the front half (`v4..v7`) once that side has landed. Every
locomotion phase is stopped before the next coordinated tilt phase starts.
`RETURN_FLAT` is a required terminal barrier: success is not published until
all eight TILT targets have returned to zero, leaving the topology suitable
for the next Bridge8 -> Snake8 reconfiguration. The four durations are command
parameters because they depend on the near-bank offset, gap width and far-bank
clearance. Start with the low values above on flat ground, then calibrate them
against the actual gap geometry. The current version is open-loop;
obstacle-edge and support-contact events can later replace the timing gates
without changing the composite executor.

Stop the completed reconfiguration expert in Terminal 3.

## 3. Reconfigure Bridge8 back to Snake8 for the stairs

```bash
ros2 run mssr_expert mssr_smores_self_reconfiguration_node --ros-args \
  -p source_graph_path:=auto \
  -p target_morphology:=snake8 \
  -p execution_id:=course-bridge8-to-snake8-01 \
  -p episode_id:=course-bridge8-to-snake8-01 \
  -p dataset_path:=$PWD/logs/datasets/course_reconfiguration.jsonl
```

This transition retains six of seven connections and changes only the front
terminal docking relation. The preferred stair controller follows the full
three-riser profile in one command. It forms opposing bends at every active
riser, shifts those bends toward the tail, and advances only wheel groups that
remain supported before and after each shift. Every advance terminates from
the measured world positions rather than elapsed time:

```bash
run_behavior snake8 stair-straight-01 straighten '{}'
run_behavior snake8 stair-arch-01 crawl_stairs_arch_wave \
  '{"riser_approach_linear_m_s":0.060,"riser_approach_tolerance_m":0.010,"linear_m_s":0.040,"profile_substeps":6,"transition_clearance_m":0.0065,"crawl_goal_tolerance_m":0.004}'
```

The unified obstacle-course policy uses this same `crawl_stairs_arch_wave`
program once;
it no longer repeats the timed `lift_head -> hook_step -> pull_over_step`
sequence for each riser. Then stop the completed reconfiguration expert.

## 4. Reconfigure Snake8 to MobileManipulator8 and press a button

```bash
ros2 run mssr_expert mssr_smores_self_reconfiguration_node --ros-args \
  -p source_graph_path:=auto \
  -p target_morphology:=mobile_manipulator8 \
  -p execution_id:=course-snake8-to-manip8-01 \
  -p episode_id:=course-snake8-to-manip8-01 \
  -p dataset_path:=$PWD/logs/datasets/course_reconfiguration.jsonl
```

Approach the button, press and hold it, then release. `press_button` ends in
the contact posture; it does not retract before the simulated button can
register the contact:

```bash
run_behavior mobile_manipulator8 manip-approach-01 drive '{"linear_m_s":0.020,"yaw_rate_rad_s":0.0,"duration_s":2.0}'
run_behavior mobile_manipulator8 manip-press-01 press_button '{}'
run_behavior mobile_manipulator8 manip-release-01 release_button '{}'
run_behavior mobile_manipulator8 manip-retreat-01 drive '{"linear_m_s":-0.020,"yaw_rate_rad_s":0.0,"duration_s":2.0}'
```

The direct commands expose deterministic reach/hold/release postures.  In the
unified obstacle-course expert the preceding `button_standoff` phase aligns
the base to 25 mm, the press posture is held, and the grounded locomotors creep
at 12 mm/s until the live world position of the free TOP face is within 40 mm
of the plunger centre.  A cross-axis error above 50 mm stops the creep instead
of pushing blindly.  Only verified contact advances to release and retreat;
there is no locomotion timer in this closed-loop course path.

Stop the completed reconfiguration expert in Terminal 3.

## 5. Reconfigure to RC Car8 and drive to the exit

```bash
ros2 run mssr_expert mssr_smores_self_reconfiguration_node --ros-args \
  -p source_graph_path:=auto \
  -p target_morphology:=rc_car8 \
  -p execution_id:=course-manip8-to-rc8-01 \
  -p episode_id:=course-manip8-to-rc8-01 \
  -p dataset_path:=$PWD/logs/datasets/course_reconfiguration.jsonl
```

The RC Car8 final fold lowers all four supports in one coordinated group. Test
straight motion, turning and the final exit drive:

```bash
run_behavior rc_car8 rc8-forward-test-01 drive '{"linear_m_s":0.030,"yaw_rate_rad_s":0.0,"duration_s":2.0}'
run_behavior rc_car8 rc8-left-test-01 drive '{"linear_m_s":0.0,"yaw_rate_rad_s":0.25,"duration_s":1.5}'
run_behavior rc_car8 rc8-exit-01 drive '{"linear_m_s":0.050,"yaw_rate_rad_s":0.0,"duration_s":6.0}'
run_behavior rc_car8 rc8-stop-01 stop '{}'
```

For the alternative route beginning with RC Car8, assemble
`smores_rc_car8.json` first and request `target_morphology:=bridge8` or
`target_morphology:=snake8`; all remaining commands are unchanged.
