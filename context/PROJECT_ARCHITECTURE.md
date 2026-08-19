# MSSR architecture context

## Research objective

The project must demonstrate that the same imitation-learning and multi-agent
reinforcement-learning pipeline can operate on different modular robot
families, initially SMORES-EP and FreeBOT.

Robot-specific knowledge is confined to:

- the physical/visual asset and its parameterization;
- the module state adapter and observation model;
- the actuator and connector model;
- atomic commands and deterministic behavioral primitives;
- experts that know how that robot family can realize a task.

The shared learning layer receives a canonical dynamic attributed graph and a
canonical action/result envelope. It must not branch on CAD paths, Isaac prim
names, SMORES face names, sphere radii, or a particular locomotion model.

## Layering

1. **Robot description**
   Geometry, mass, joints, limits, connectors, sensors and functional payload
   requirements.
2. **Runtime backend**
   Isaac articulation access, collision/contact state, connector joint
   lifecycle and actuator commands.
3. **Atomic module API**
   Wheel/body commands, pan, tilt and connector activation.
4. **Behavioral primitives**
   Goal-oriented operations with lifecycle, feedback, timeout, cancellation
   and a terminal result.
5. **Dynamic graph adapter**
   Converts robot-specific runtime state into the canonical graph.
6. **Deterministic experts**
   Assign roles, select target morphology and compose behavioral primitives.
7. **Shared IL/MARL**
   Learns from the same graph/action representation for every robot family.

## SMORES-EP behavioral primitives

The externally accessible initial set is:

- `drive_to_pose(module_id, target_pose, tolerances)`;
- `align_faces(module_a, face_a, module_b, face_b, clocking)`;
- `dock(module_a, face_a, module_b, face_b, clocking)`;
- `undock(module_a, face_a, module_b, face_b)`;
- `set_pan(module_id, angle)` and `rotate_pan_by(module_id, delta)`;
- `set_tilt(module_id, angle)` and `rotate_tilt_by(module_id, delta)`.

Each goal has a unique ID and transitions through:

`ACCEPTED -> RUNNING -> SUCCEEDED | FAILED | CANCELED`.

Feedback includes current error, active phase and affected module IDs. A
terminal result includes a stable reason code and a human-readable message.
Concurrency is decided by actuator resources, not by module ID alone:

- `locomotion` owns LEFT and RIGHT wheel commands;
- `internal_motion` owns the two internal motors and executes either PAN or
  TILT at one time;
- connector activation owns the addressed face magnet.

A locomotion goal may therefore execute concurrently with one PAN or one TILT
goal on the same module. PAN and TILT goals are mutually exclusive. Connector
commands may coexist unless they invalidate the geometry or topology assumed
by an active motion goal. Goals on disjoint modules may also run concurrently.

LEFT and RIGHT wheel velocities are commanded simultaneously. A general
differential-drive command may translate and yaw along a curved path. Pure
spin has zero translational velocity by definition, so pure spin and straight
translation are not two independent motions that can be superimposed.

The ROS-facing process should be a real ROS 2 action server or an action-like
adapter outside Isaac. Isaac must not import the system `rclpy`: the existing
JSON file bridge keeps the ROS distribution isolated from Isaac Sim's Python
runtime. A topic/JSON transport is acceptable internally, while the public API
retains action semantics.

## Group functional requirements

`drive_connected_component` and `lift_chain` are not atomic capabilities of a
single module and are not exposed as hardware primitives. They are composed
behaviors and validation scenarios.

The enhanced SMORES-compatible module model has these explicit functional
requirements:

- a docked module can physically tow another module, with the expected loss of
  maneuverability caused by passive wheel/contact friction;
- a connected set can be driven cooperatively when all of its modules receive
  commands;
- one module is designed to lift a connected chain with a target payload of
  five to seven modules;
- connector, motor, transmission, traction and support requirements are
  calibrated so those behaviors are physically feasible for the target
  improved design.

Towing and chain lifting are encoded as target module capabilities and design
requirements, not as simulation-only capabilities. The current Isaac
implementation may still record its test setup (`IsaacGroundSupportAnchor`,
drive profile and unbreakable joint) for reproducibility; this metadata
describes how a benchmark was realized and does not downgrade the capability
to a simulator trick. The original SMORES-EP hardware profile and the enhanced
target design profile remain distinguishable engineering parameter sets.

## Canonical dynamic graph

### Module node

Every node contains:

- `module_id`, `robot_family`, `module_type`;
- world pose and twist, with frame and timestamp;
- actuator positions, velocities, limits and availability;
- connector states;
- sensor availability and observation confidence;
- `current_role`, `target_role` and role confidence;
- a structured `functional_role` profile;
- health, control availability and simulation-fixture flags.

`functional_role` is not just a name. It describes, for example:

- `support/base`: load-bearing anchor and support polygon responsibility;
- `joint`: effective DoF count, axes, limits and upstream/downstream faces;
- `elbow`: a one-DoF bending contribution;
- `wrist`: two- or three-DoF orientation contribution;
- `link`: rigid structural contribution;
- `locomotor`: faces/wheels responsible for cluster locomotion;
- `end_effector`: task interaction responsibility.

The vocabulary is extensible. Learning may predict `target_role`,
`functional_role` and confidence; deterministic experts provide supervised
labels and feasibility masks.

### Task-conditioned attributed multigraph

The canonical learning input is not the planner-specific kinematic tree.
Each expert transition uses a task-conditioned attributed multigraph with:

- physical-module nodes and their current state;
- logical `target_slot` nodes with target and functional roles;
- `current_connection`, `contact` and `target_connection` relations;
- one-to-one `assignment` relations from physical modules to target slots;
- expert execution state in global graph attributes.

Multiple relations between the same endpoint pair are preserved by a key
containing relation type, connector endpoints and optional relation ID.
Adjacency must be requested with an explicit relation filter whenever an
algorithm needs only latched, target or assignment edges.

Family-specific deterministic planners may project the target subgraph into a
strict internal representation such as `SmoresKinematicTree`. That projection
is not serialized as the primary observation and does not replace the
attributed graph in datasets.

The IL transition schema stores `graph_t`, the conditioned target graph, the
combined `task_graph_t`, the expert assignment and action, and optionally
`graph_t_plus_1`. This preserves assignment and topology evolution as
supervision rather than hiding them inside the expert.

### Connection edge

Every latched docking edge contains:

- the two module IDs and the connector/face at each endpoint;
- connection state and connector type;
- relative transform and discrete clocking/orientation;
- joint type and allowed relative DoFs;
- load-bearing/temporary flags and edge role;
- timestamp, health and optional estimated load.

SMORES-EP connections use explicit `LEFT`, `RIGHT`, `TOP`, `BOTTOM` endpoints.
FreeBOT may use continuous surface contact coordinates. Both map to the same
edge envelope without discarding their robot-specific connector attributes.

Transient contact candidates are represented separately from latched topology
edges. Graph topology is updated on dock, undock, reset, failure and module
addition/removal.

## Observation policy

The first deterministic and learning experiments use centralized, perfect
simulator state, equivalent to a simulated VICON system:

- pose and twist of every module;
- joint positions and velocities;
- per-connector free/contact/aligned/latched state;
- current topology and task/environment state.

This is an observation model, not an assertion that each physical module has
onboard vision. The supplied SMORES-EP references support joint position
sensing and external VICON/AprilTag localization; a standard module has no
integrated environmental camera, lidar or autonomous perception. Noise,
latency, dropout and partial observability are later curriculum dimensions.

## Reuse of the spherical prototype

Reuse:

- state registry and JSON/file bridge;
- multi-module command routing;
- graph/expert interfaces;
- role-assignment, curriculum and dataset logging infrastructure.

Replace or adapt:

- sphere radius and arbitrary surface attachment fields;
- holonomic `vx/vy` assumptions;
- `dock_to_surface` and `attach_as_pivot`;
- pivot rolling as a substitute for SMORES pan/tilt;
- graph edges that omit explicit connector endpoints and clocking.

There must be one canonical graph schema and one canonical action envelope,
with robot-family adapters. Do not introduce a third independent graph model.
