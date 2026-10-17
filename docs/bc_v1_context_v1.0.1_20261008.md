# BC-v1 context contract — v1.0.1

Date: 2026-10-08. Approved direction: policy selects individual operations from
current physical robot state, current stage/task and current environment geometry.
This amendment supersedes the earlier D1.1 configuration-goal input design and
clarifies section 6 of the BC-v1 architecture v1.0.0. Canonical v6 and its labels
remain immutable; no episode/label selection or counts change.

`(G_t, o_t, c_t) -> a_t`:

- `G_t`: current physical robot poses, resources, connectors and connections.
- `o_t`: current physical environment geometry and measurements. Known course
  collision boxes and physical navigation cones remain observable during assembly,
  reconfiguration and task execution, independently of task assignment. Preserve
  source frames and validity; relative geometry/distances use current robot poses
  in E1. No nominal Nav2 waypoint is an obstacle observation.
- `c_t`: current semantic task, current stage and permitted intrinsic task geometry.
  A task must be explicitly available online; otherwise it is UNKNOWN. Stage does
  not assign an environmental task. Unassigned mission instructions are excluded.

Desired configuration/morphology is not an input, even when known to the operator
or recorded causally. `TaskContext` has only `task_id`, `stage_id`, `static_geometry`
and `stage_geometry`. No `configuration_goal`, destination vocabulary, controller
selection, startup configuration, task-to-morphology imputation or target graph
enters the policy context. Current physical topology stays in `G_t`.

The policy's structural primitives determine the resulting configuration. This
correction adds no morphology prediction head, WAIT or dispatch/no-dispatch head.
It does not implement an autonomous stage scheduler or infer task identity from
nearby geometry, action sequences, desired morphology, future events or outcome.
For an explicitly assigned stairs objective during reconfiguration, context can
contain `task_id=stairs, stage_id=reconfiguration`; the environment provides the
physical stairs geometry, without an instruction to become Snake.

Regression gates: typed context rejects a destination field; changing desired
configuration anywhere in a current row/source/task metadata cannot change context
or observation; raw-reader metadata excludes destination metrics and commands;
physical obstacles/cones remain available with UNKNOWN task in all stages; nominal
routes and future mission changes do not change physical observations. Preserve
stdlib import isolation and input/source immutability.

Any derived sample/cache retaining the former configuration-goal input is
incompatible with this context revision and must be rebuilt using the changed
contract/config fingerprints. No cache, split, normalization, E1 transformation,
model, trainer, runtime, executor or recorder is implemented or modified here.
