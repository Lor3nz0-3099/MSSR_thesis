# SMORES-EP control package

This package keeps the imported CAD visual separate from the generated physics
asset and from runtime scenarios.

## Coordinate convention

The runtime model follows ROS REP-103:

- `+X`: module forward direction, from the passive BOTTOM face toward TOP;
- `+Y`: left, toward the LEFT wheel;
- `+Z`: up;
- positive yaw: counter-clockwise around `+Z`.

The imported CAD uses a different frame. The Isaac stage builder performs the
frame conversion without modifying the source USD.

## Actuated degrees of freedom

- `left_wheel`: continuous;
- `right_wheel`: continuous;
- `pan`: continuous rotation of the TOP face around the module forward axis;
- `tilt`: rotation of the TOP carrier around the wheel axle, limited to
  `[-pi/2, pi/2]`.

The hardware pan/tilt differential is represented at its output coordinates.
The pure control module also exposes the two-motor mixing equations so the
internal mechanism is not confused with two unrelated direct-drive joints.

The CAD animation follows the physical gear grouping:

- outer diagonal: pinion 2 drives left outer gear 4 and pinion 4 drives right
  outer gear 2, together with the corresponding wheels;
- inner diagonal: pinions 1 and 3 drive inner gears 1 and 3;
- equal inner-gear directions produce tilt, opposite directions produce pan;
- every 9-tooth pinion counter-rotates relative to its 48-tooth gear with a
  visual ratio of `48:9`.

## ROS 2 interface

| Topic | Type | Units | Meaning |
|---|---|---|---|
| `/cmd_vel` | `geometry_msgs/msg/Twist` | m/s, rad/s | differential locomotion |
| `/smores_ep/pan_angle` | `std_msgs/msg/Float32` | rad | absolute pan target |
| `/smores_ep/tilt_angle` | `std_msgs/msg/Float32` | rad | absolute tilt target |
| `/smores_ep/docking_command` | `std_msgs/msg/String` | - | energize/de-energize a dock between two module IDs |
| `/mssr/primitives/goal` | `std_msgs/msg/String` (JSON) | SI | action-like behavioral primitive goal |
| `/mssr/primitives/cancel` | `std_msgs/msg/String` (JSON) | - | cancel a primitive goal by ID |
| `/mssr/primitives/status` | `std_msgs/msg/String` (JSON) | SI | lifecycle, feedback and terminal result |

Positive tilt raises the TOP face; negative tilt lowers it. The kinematic
scenario remains useful for inspecting CAD grouping and command signs. The
dynamic scenario does not correct the root pose: gravity, actuator reaction
torques and PhysX contacts determine whether the module remains supported,
tilts or falls.

The generated physics articulation is kept at:

```text
assets/smores-ep/usd_physics/smores_ep_physics_v1.usd
```

It contains five rigid links, four revolute joints, simple collision proxies,
the measured total mass, and four explicit docking frames named LEFT, RIGHT,
TOP, and BOTTOM. Internal self-collisions are disabled at articulation level,
while wheel, TOP, chassis-skid and ground collisions remain active. The
per-link mass split is an explicit engineering estimate because the references
only constrain the complete module mass and approximate center of mass.

All three rotating disks use explicit convex-hull contact shapes measured from
the CAD vertices. The two driving wheels share the same `62 mm` prototype;
rotated axis-aligned bounds must not be used to infer their diameter.
The passive rear edge has its own low-friction skid material (`0.03` static,
`0.02` dynamic), combined multiplicatively with the floor so it stabilizes the
module without behaving like a brake.

The actuator limits come from the SMORES reference: `1.2 N m` wheel,
`1.4 N m` pan, `2.3 N m` tilt and `23 RPM` no-load speed for each rotational
DoF. The same reference reports a maximum land speed of `1.1 body lengths/s`.
Applied to the `80 mm` SMORES-EP characteristic length, this is `0.088 m/s`
or `2.83 rad/s` at the measured `31.06 mm` CAD wheel radius.

Regenerate it with Isaac's USD libraries:

```bash
export PYTHONPATH="$PWD/scripts/smores_ep/src"
scripts/isaac_freebot/run_usd_tool.sh \
  scripts/smores_ep/tools/create_physics_asset.py
```

Verify gear signs and ground clearance at `-45`, `0`, and `+45` degrees:

```bash
export PYTHONPATH="$PWD/scripts/smores_ep/src"
scripts/isaac_freebot/run_usd_tool.sh \
  scripts/smores_ep/tools/check_kinematic_clearance.py
```

## Run

From the repository root:

```bash
bash scripts/smores_ep/run_dynamic.sh
```

This opens the fully dynamic Isaac Sim GUI and listens to ROS 2. For a
deterministic physical demonstration without ROS 2:

```bash
bash scripts/smores_ep/run_dynamic.sh \
  --headless --no-ros2 --demo --steps 1440
```

The visual-only diagnostic remains available with:

```bash
bash scripts/smores_ep/run_kinematic.sh
```

ROS 2 commands can be sent from another terminal:

```bash
source /opt/ros/humble/setup.bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

```bash
source /opt/ros/humble/setup.bash
ros2 topic pub --once /smores_ep/pan_angle std_msgs/msg/Float32 "{data: 1.5708}"
ros2 topic pub --once /smores_ep/tilt_angle std_msgs/msg/Float32 "{data: 0.7854}"
```

`/smores_ep/pan_angle` and `/smores_ep/tilt_angle` are absolute targets in
radians. A message on `/smores_ep/pan_delta` is instead added once to the
current pan target:

```bash
source /opt/ros/humble/setup.bash
ros2 topic pub --once /smores_ep/pan_delta std_msgs/msg/Float32 "{data: 1.5708}"
```

Repeating the command adds another `+90` degrees. Negative values rotate in
the opposite direction. The dynamic controller clamps tilt to
`[-pi/2, pi/2]`. Pan is controlled as an unwrapped continuous coordinate
through a bounded PhysX velocity drive, so its accumulated target may cross
any number of complete revolutions. Close the Isaac Sim window to stop the GUI
scenario.

## Two-module rigid docking

Start the docking scenario from the repository root:

```bash
bash scripts/smores_ep/run_docking.sh
```

It creates two physically identical, fully dynamic articulations. `active` is
connected to the direct teleoperation topics; `passive` has no direct teleop
source, but both IDs can receive goal-oriented primitive commands. A module
without a current command is dynamically towable, not kinematic or fixed.

The passive module is not fixed to the world. It initially faces the active
module with `active:TOP` approaching `passive:LEFT` and a `12 mm` clearance.
Use `/cmd_vel` to bring them into contact, then energize the contacting
EP-Faces:

```bash
source /opt/ros/humble/setup.bash
ros2 topic pub --once /smores_ep/docking_command \
  std_msgs/msg/String "{data: 'attach active passive'}"
```

The legacy command above asks the docking manager to check all 16 face pairs,
ignore occupied faces, and select the closest valid pair. Deterministic
planners should instead name both endpoints explicitly:

```bash
ros2 topic pub --once /smores_ep/docking_command \
  std_msgs/msg/String \
  "{data: 'attach active TOP passive LEFT'}"
```

Explicit detach uses the same endpoints:

```bash
ros2 topic pub --once /smores_ep/docking_command \
  std_msgs/msg/String \
  "{data: 'detach active TOP passive LEFT'}"
```

The manager accepts either form only when the frames are close and their
outward normals oppose each other. Square magnet arrays normally also require
compatible 90-degree clocking. For `BOTTOM` docking to a continuously rotating
`LEFT` or `RIGHT` disk, clocking is not a configuration constraint; this
matches the SMORES-EP topology model and avoids treating the disk's arbitrary
rolled angle as a failed planar alignment.
Because some USD docking markers are construction frames rather than rendered
contact planes, non-axial pairs allow up to `6.7 mm` normal separation.
`TOP<->BOTTOM` uses the stricter `1.5 mm` outer-plane contact gate. All pairs
allow at most `10 mm` of 3-D lateral marker offset and `8 deg` normal error;
clocking is limited to `10 deg` except for a `BOTTOM` face mating with a
continuously rotating lateral disk. Alignment and attach evaluate these same
limits, so reaching contact cannot leave the primitive idle behind a second
inconsistent gate.

On acceptance, Isaac creates a runtime `UsdPhysics.FixedJoint` between the two
rigid links. This is a real physical constraint: forces and motion pass
between the modules. It is not a root-pose correction or visual parenting.
The two docked links have mutual collision disabled at the joint, while all
other contacts remain physical.

De-energize the same module pair with:

```bash
source /opt/ros/humble/setup.bash
ros2 topic pub --once /smores_ep/docking_command \
  std_msgs/msg/String "{data: 'detach active passive'}"
```

The fixed joint is removed immediately and both modules continue as independent
dynamic bodies. Repeating `attach` after they have moved outside the capture
gate is safely rejected, with the closest face pair and its measured errors
printed in the Isaac terminal.

The docking scenario gives both identical articulations a payload-overdrive
profile. Its wheel, tilt and pan effort limits are three times the
nominal hardware values (`3.6`, `6.9`, and `4.2 N m`), and its implicit-PD tilt
and zero-speed wheel holding gains are increased so lifting torque moves the
payload instead of elastically folding or rolling the active module. The
An uncommanded module remains fully dynamic and backdrivable: LEFT/RIGHT
wheels, PAN and TILT receive zero stiffness and only small bearing damping.
Docking remains rigid at the magnetic face, but no motor joint is implicitly
held merely because that face is connected. A locomotion-only command enables
only LEFT/RIGHT; PAN and TILT remain passive. An explicit PAN or TILT primitive
is required to energize the internal differential.

For a repeatable initial pairing, the scenario accepts
`--initial-active-face` and `--initial-passive-face`. For example,
`--initial-active-face BOTTOM --initial-passive-face BOTTOM
--passive-yaw-deg 180 --initial-face-gap-mm 0` places the two CAD
`base-chassis` outer planes tangent and opposed.

Change the effort multiplier with:

```bash
bash scripts/smores_ep/run_docking.sh --actuator-effort-scale 2
```

The EP-Face reference reports an average bending failure moment of `1.8 N m`,
equivalent to 3.1 static modules in cantilever, and a practical limit of two
modules. This project instead specifies an enhanced SMORES-compatible physical
design requirement: tow attached payloads and lift a five-module minimum,
seven-module target chain. The overdrive gains, unbreakable runtime
`FixedJoint`, and optional support are recorded simulation fixtures used to
exercise that requirement; the capability itself is not simulation-only.

Torque alone cannot make one free module lift five equal modules: the reaction
moment tips the active module around its tire contact. Therefore the payload
scenario also creates an explicit anti-tip D6 support for the active
`body_link` when a docked structure receives its first non-zero tilt command.
Waiting until tilt is requested lets the chassis settle onto both CAD wheels
before roll and pitch are held. World `X`, `Y`, `Z`, and yaw remain free,
preserving differential-drive traction while the lifted structure moves.
Wheel, pan and tilt joints also remain actuated. The support is removed on
detach.

The D6 support also carries a force-limited yaw velocity drive. It follows
`cmd_vel.angular.z` with at most `6.9 N m` of steering assistance, while the
two wheels continue to receive their differential velocity targets. This
compensates for the deliberately exaggerated cantilever load without
kinematically overwriting the module pose.

Run the fully free-body version, useful for nominal-physics experiments, with:

```bash
bash scripts/smores_ep/run_docking.sh --no-active-ground-anchor
```

## Goal-oriented primitive interface

The two-module scenario also runs a goal-oriented executor for:

- `drive_to_pose`;
- `align_faces`;
- `assisted_align_faces`, where a third, rigidly attached helper drives the
  payload toward the target face;
- `dock` and `undock` with explicit endpoints;
- `set_pan`, `rotate_pan_by`, `set_tilt`, and `rotate_tilt_by`.

Goals have IDs, timeout, feedback, cancellation and terminal states
(`succeeded`, `failed`, `canceled`, or `rejected`). The transport currently
uses JSON over ROS `String`, while retaining action-server semantics. This
keeps the system ROS 2 `rclpy` process outside Isaac's Python environment.

Admission is based on physical resources. `locomotion` owns both wheels, so a
straight drive, a curve, and pure spin all exclude another wheel motion on the
same module. `internal_motion` owns the two coupled inner motors and executes
PAN or TILT, never both simultaneously. One wheel motion and one PAN or TILT
may run concurrently on the same module; goals on different modules may also
run concurrently. Connector resources are exclusive per named face.

Start the external bridge in a second terminal:

```bash
cd ~/MSSR_thesis
source /opt/ros/humble/setup.bash
python3 ros2_bridge/mssr_file_bridge.py
```

Inspect status in a third terminal:

```bash
source /opt/ros/humble/setup.bash
ros2 topic echo /mssr/primitives/status
```

Admission may be published as `mssr.primitive_status.v1`; periodic executor
snapshots use `mssr.primitive_status_batch.v1` with a `statuses` array so
concurrent goals keep independent feedback and terminal results.

Drive `active` to a world-frame planar pose:

```bash
ros2 topic pub --once /mssr/primitives/goal std_msgs/msg/String \
  "{data: '{\"schema_version\":\"mssr.primitive_goal.v1\",\"goal_id\":\"drive-001\",\"primitive\":\"drive_to_pose\",\"module_ids\":[\"active\"],\"parameters\":{\"x_m\":0.15,\"y_m\":0.0,\"yaw_rad\":0.0},\"timeout_s\":20.0}'}"
```

Align and then dock two explicit faces:

```bash
ros2 topic pub --once /mssr/primitives/goal std_msgs/msg/String \
  "{data: '{\"goal_id\":\"align-001\",\"primitive\":\"align_faces\",\"module_ids\":[\"active\",\"passive\"],\"parameters\":{\"face_a\":\"TOP\",\"face_b\":\"LEFT\"},\"timeout_s\":20.0}'}"

ros2 topic pub --once /mssr/primitives/goal std_msgs/msg/String \
  "{data: '{\"goal_id\":\"dock-001\",\"primitive\":\"dock\",\"module_ids\":[\"active\",\"passive\"],\"parameters\":{\"face_a\":\"TOP\",\"face_b\":\"LEFT\"},\"timeout_s\":5.0}'}"
```

Pan and tilt use radians, consistent with ROS:

```bash
ros2 topic pub --once /mssr/primitives/goal std_msgs/msg/String \
  "{data: '{\"goal_id\":\"pan-001\",\"primitive\":\"rotate_pan_by\",\"module_ids\":[\"active\"],\"parameters\":{\"delta_rad\":1.5708},\"timeout_s\":10.0}'}"

ros2 topic pub --once /mssr/primitives/goal std_msgs/msg/String \
  "{data: '{\"goal_id\":\"tilt-001\",\"primitive\":\"set_tilt\",\"module_ids\":[\"active\"],\"parameters\":{\"angle_rad\":0.7854},\"timeout_s\":10.0}'}"
```

Cancel a running goal:

```bash
ros2 topic pub --once /mssr/primitives/cancel std_msgs/msg/String \
  "{data: '{\"goal_id\":\"drive-001\"}'}"
```

At 20 Hz the scenario writes the canonical centralized observation to:

```text
logs/bridge/module_states.json
logs/bridge/robot_graph.json
logs/bridge/state_graph.json
```

Each graph node includes its current/target role, structured functional role,
four actuator states, four connector states, sensor capabilities and
simulation-fixture flags. Edges identify both module IDs, both face IDs,
clocking, connection state and rigid-joint semantics. Aligned contact
candidates are distinct from latched docking edges.

## Five-module chain lift

Start the dedicated multi-module scenario with:

```bash
bash scripts/smores_ep/run_multi_lift.sh
```

It creates six fully dynamic modules:

- `active`, the module subscribed to the direct motion topics;
- `chain_01` through `chain_05`, already joined in a horizontal chain by
  four rigid `TOP` (UP) to `BOTTOM` joints.

All six IDs are registered in the primitive executor and may be independently
controlled through `/mssr/primitives/goal`; “active” and “chain” are scenario
roles, not different robot implementations.

The CAD-derived root spacing is `77.77 mm`. `active:TOP` starts tangent to the
free `chain_01:BOTTOM` face but is not connected. Energize that contact with:

```bash
source /opt/ros/humble/setup.bash
ros2 topic pub --once /smores_ep/docking_command \
  std_msgs/msg/String "{data: 'attach active chain_01'}"
```

Then request a positive tilt to raise the chain:

```bash
ros2 topic pub --once /smores_ep/tilt_angle \
  std_msgs/msg/Float32 "{data: 1.5708}"
```

The default effort multiplier is `6`, giving the deliberately exaggerated
wheel, tilt and pan limits `7.2`, `13.8`, and `8.4 N m`. The active anti-tip
support engages only after `active` has docked and a non-zero tilt command is
received. The four pre-existing chain joints alone do not activate it.

Detach the active module without breaking the existing chain with:

```bash
ros2 topic pub --once /smores_ep/docking_command \
  std_msgs/msg/String "{data: 'detach active chain_01'}"
```

The chain length and effort scale are configurable:

```bash
bash scripts/smores_ep/run_multi_lift.sh \
  --chain-count 5 \
  --actuator-effort-scale 6
```

The default synchronized controller and PhysX limit now uses the paper's
maximum land speed, `0.088 m/s`. A `/cmd_vel` command must request that speed;
for example:

```bash
ros2 topic pub --rate 10 /cmd_vel geometry_msgs/msg/Twist \
  "{linear: {x: 0.088}, angular: {z: 0.0}}"
```

The reusable implementation is split into:

```text
docking/model.py             command parsing and face-pair geometry
isaac/docking.py             face discovery and FixedJoint lifecycle
isaac/multi_module_stage.py  module cloning and relationship retargeting
isaac/command_router.py      per-module controlled/passive actuator routing
isaac/primitive_executor.py  concurrent resource-aware goal execution
control/docking_teleop.py    native ROS 2 String subscriber
scenarios/two_module_docking.py
scenarios/multi_module_lift.py
```

`IsaacDockingManager` receives a `{module_id: USD_root}` registry, so the
docking behavior is not tied to this two-module demonstration. A face may have
only one active connection, while a module can connect different free faces
to different registered modules.

## Deterministic parallel self-assembly

The self-assembly scenario starts identical, fully dynamic SMORES-EP modules
separated on the floor. The validated three-module layout remains the default;
larger runs place one root candidate at the centroid and the remaining modules
on a ring. It deliberately does not load Isaac's internal
ROS 2 extension: all state, primitive goals and feedback cross the existing
atomic file bridge, keeping `rclpy` in the system ROS 2 process.

Start Isaac:

```bash
bash scripts/smores_ep/run_self_assembly.sh
```

For a smooth low-overhead GUI, use the performance profile:

```bash
bash scripts/smores_ep/run_self_assembly.sh --performance --log-interval 0
```

For a seven-module target:

```bash
bash scripts/smores_ep/run_self_assembly.sh \
  --module-count 7 \
  --spawn-radius 0.34 \
  --performance \
  --log-interval 0
```

Free-space motion toward each face-alignment staging pose uses an
assembly-only planar visibility graph. Every other module is represented by
an inflated `110 mm` centre-clearance footprint; when the direct segment is
blocked, the controller follows deterministic waypoints with a `15 mm`
margin and replans if a concurrent module invalidates the route. Packed
reconfiguration starts progressively densify the visibility graph from 16 to
32 and 64 angular samples. If a parallel peer temporarily occupies every
route, that alignment is deferred until the other actions settle and dock,
then retried serially from the updated live geometry. This check ends at
staging, so the intentional straight contact-and-dock phase and all
post-assembly morphology locomotion are unchanged. The defaults can be tuned
with `--staging-center-clearance` and `--staging-waypoint-margin`, or disabled
for an A/B comparison with `--disable-staging-collision-avoidance`.

This profile keeps the complete CAD, runs physics at `120 Hz`, presents the
viewport at a wall-clock paced `30 FPS`, and publishes the ROS state graph at
`5 Hz`. Physics, articulation drives, docking frames, rigid attachments and
primitive execution are unchanged. Contact-candidate graph edges are omitted;
latched docking edges remain available to the expert. Add `--simple-visuals`
only when collision-proxy rendering is explicitly desired.

In a second terminal start the external bridge:

```bash
cd ~/MSSR_thesis
source /opt/ros/humble/setup.bash
python3 ros2_bridge/mssr_file_bridge.py
```

In a third terminal start the deterministic expert:

```bash
cd ~/MSSR_thesis/mssr_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 run mssr_expert mssr_smores_self_assembly_node
```

The planner assigns physical modules to target slots with one
Kuhn-Munkres/Hungarian assignment. Its morphology-agnostic cost first avoids
leaving modules assigned to later waves inside earlier docking corridors,
then minimizes physical travel. The module closest to the swarm centroid is
selected as physical root and compatible root-to-leaf actions are planned in
waves. Every action in a wave remains independent and parallel. During
self-reconfiguration the same congestion cost is applied only to detached or
free movers, so it never sacrifices an already retained target connection.
Execution adds collective `REACH -> ALIGN -> APPROACH -> DOCK` barriers: all
participants finish collision-free positioning before pose adjustment, all
finish adjustment before the signed straight approach, and docking begins
only after every approach has settled. A BOTTOM-face mover therefore backs
along the docking centreline, while the opposite face convention uses the
corresponding signed direction. Pass a different target to the expert with
`-p target_graph_path:=/absolute/path/to/target.json`. The Isaac terminal
prints `TARGET TOPOLOGY REACHED` when the expected `N-1` rigid connections of
the target tree are present.

Retry and contact behavior are runtime policy, not target-graph metadata. The
defaults allow two fresh alignment retries, two dock recoveries through a new
alignment, and two local contact-quality reacquisitions. The same policy is
used by self-assembly and by every assembly stage of self-reconfiguration.
It can be overridden uniformly with ROS parameters such as
`align_retry_count`, `dock_recovery_count`,
`contact_quality_retry_count`, `top_bottom_contact_tolerance_m`, and
`max_concurrent_alignments_per_wave`. Its default is `0`, meaning no artificial
cap inside a planner wave; a positive value remains available only as a runtime
resource limit.

RC Car8 uses progressive planar waves without a helper: three modules form the
four-module centerline, four wheel modules dock at its two endpoints, and four
coordinated TILT goals fold the connected structure into its final posture.
The previous RC Car7 target remains available as a regression profile. A
global all-module pre-layout remains available as an
experiment through `global_layout_before_docking:=true`, but is disabled by
default. The `assisted_align_faces` primitive is retained only for target
graphs that explicitly require and enable a helping-module procedure.

The same scenario consumes `/mssr/actions` through `configs/actions.json`.
The external morphology behavior node uses this path to coordinate the wheels
of an assembled cluster. Commands have a `0.5 s` dead-man timeout and atomic
primitive goals retain priority over composed locomotion on resources they
currently own.

Terminal primitive results are retained in the status batch for the complete
session. This prevents a slower external bridge from missing a short-lived
`SUCCEEDED` or `FAILED` update while other parallel goals remain active. If a
straight connector approach reaches its target pose with residual lateral or
yaw error, the controller now retreats to staging and reacquires the face
instead of waiting motionless for the timeout.

Mobile `TOP` and `BOTTOM` faces use the same coupled pose-adjustment law for
both the three-module centerline and the four wheel connections. Inside the
local staging region, lateral error and yaw are corrected while translating
with a latched drive direction. A pivot is entered only when yaw still exceeds
`2 deg`; 2/4 mm lateral hysteresis prevents command chatter, while an aligned
yaw always returns to a correction arc if lateral error remains. A physically
stalled in-place turn—including the steering acquisition issued by the curve
controller—receives a bounded 0.5 s unstick arc without restarting global
staging. The final straight approach starts within `2.5 mm` lateral error and
`2 deg` planar yaw. Before requesting the physical
dock, alignment and the physical docking manager use one shared contact gate.
With the default shared policy, `TOP<->BOTTOM` requires at most `4 mm` normal
gap, `10 mm` 3-D lateral marker offset and `8 deg` normal error, while the
separate planar parking-quality limit remains `1.5 mm`. Consequently an
aligned contact can proceed directly to dock without a second, contradictory
filter.

Target JSON metadata is validated against the generated plan: declared module
count, semantic root and helping-module requirement must agree with the target
slots and docking faces. Snake7, Mobile Manipulator8 and RC Car8 use the same
progressive axial-face pipeline. The legacy Mobile Manipulator7 target remains
available only for regression. Operational morphology commands are rejected
until the transient-local expert task graph reports `done=true` and
`success=true`, avoiding resource conflicts with folding or reconfiguration.

### General morphology-to-morphology self-reconfiguration

After self-assembly has completed, stop its ROS expert with `Ctrl-C`; leave the
Isaac scenario and file bridge running and start the reconfiguration expert.
The source is recognized automatically among all installed targets, including
Snake7, RC Car8, Mobile Manipulator8 and Holonomic9. Select any different known
target by name and use a new execution
ID on every launch:

```bash
cd ~/MSSR_thesis/mssr_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 run mssr_expert mssr_smores_self_reconfiguration_node \
  --ros-args \
  -p source_graph_path:=auto \
  -p target_morphology:=snake7 \
  -p execution_id:=morphology-transition-run-1 \
  -p episode_id:=morphology_transition_episode_1 \
  -p dataset_path:=/home/lorenzo/MSSR_thesis/logs/datasets/smores_reconfiguration.jsonl
```

Reference target names include `snake7`, `rc_car8`, `mobile_manipulator8` and
`holonomic9`; `rc_car7` and `mobile_manipulator7` remain available as legacy
regressions.
All directed transitions are planned by the same algorithm: it maximizes the
root-connected common face topology, stows the source, releases only obsolete
edges, docks only missing edges, applies the target ready posture and verifies
the complete live graph. Depending on the pair it retains two to four of the
six original connections. Among assignments with the same retained topology,
the planner minimizes motion from the current live poses; this prevents
equivalent detached modules from being sent across one another.
Reconfiguration docks use the same parallel `REACH -> ALIGN -> APPROACH ->
DOCK` barriers, retry, contact and admission policy as assembly from separated
modules. Topologically independent actions in each planned stage remain
parallel; a temporarily blocked free-space route is replanned after moving
peers settle without serializing the whole morphology.

A new target can be supplied with `target_graph_path` instead of
`target_morphology`; an unknown source can likewise be declared with
`source_graph_path`. Both graphs must currently contain the same module count
and valid tree topology. The executable baseline intentionally rejects plans
that would require driving a still-connected moving subcluster.

Before planning, the node cancels any primitive goals left active by the
source expert, one at a time, and waits for their resources to be released.
If source auto-detection finds no unique catalog match, the node reports the
topology mismatch instead of silently applying the wrong plan.
