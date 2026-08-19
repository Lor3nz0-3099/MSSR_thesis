# Legacy spherical code migration

## Decision

Do not delete the spherical Python implementation as obsolete. It is the
prototype FreeBOT backend and supplies the second robot family required by the
generalization claim. Isolate it behind the same contracts used by SMORES-EP,
then delete only superseded duplicates.

## Keep as shared infrastructure

- `ros2_bridge/mssr_file_bridge.py`
- `mssr_expert/graph/attributed_robot_graph.py`
- `mssr_expert/graph/graph_builder.py`
- expert base classes, registry and output envelope
- curriculum scheduling and dataset logging
- deterministic ordering and JSON utilities
- state registry and transport-independent publishing concepts
- multi-module command routing concepts

These components must become independent of sphere radius, arbitrary surface
contact and SMORES face names.

## Keep, but move behind the FreeBOT adapter

- `robots/spherical_robot.py`
- spherical control classes in `robots/control.py`
- `robots/surface_attachment.py`
- `robots/magnetic_attachment.py`
- sphere-oriented state readers and reset logic
- `worlds/scenario_config.py` fields tied to spherical modules
- `mssr_expert` primitives:
  - `roll_to`
  - `dock_to_surface`
  - `attach_as_pivot`
  - `rotate_around_attached`
  - `climb_on`
- stage 0/1/2 experts whose geometry and behaviors assume spheres
- the current `scripts/main.py` spherical scenario runner

These are not SMORES primitives. They should eventually live under explicit
namespaces such as `adapters/freebot`, `primitives/freebot` and
`experts/freebot`, while SMORES receives sibling namespaces.

## Refactor and then retire duplicates

- `graphs/robot_graph.py`: currently builds edges from
  `SurfaceAttachmentState`; migrate its remaining consumers to the canonical
  attributed graph and then remove it.
- `robots/module_state.py`: retain the common pose/twist shell, replace the
  sphere-first attachment model with generic actuator/connector extensions and
  family-specific payloads.
- `robots/actions.py` and `configs/action_schema_v1.json`: retain a common
  action envelope, move surface-pivot fields to the FreeBOT payload and
  discrete face/internal-motion fields to SMORES payloads.
- `robots/json_publishers.py` and
  `smores_ep/isaac/state_graph_publisher.py`: consolidate atomic JSON writing
  after the canonical state schema is stable.
- fixed scalar role indices in `graph_features.py`: replace with a canonical
  structured role feature schema and masks suitable for learned roles.

Do not remove any of these until both robot-family adapters pass the same
state, graph, action and dataset compatibility tests.

## Data retention

The spherical datasets and bridge histories are useful baselines. Tag or
migrate them with `robot_family=freebot` and a schema version rather than
deleting them. Large histories may be archived outside the active workspace
after checksums and dataset manifests are created.

## Target package shape

```text
mssr_core/
  schema/
  graph/
  actions/
  roles/
  dataset/
  curriculum/

robot_adapters/
  freebot/
    state/
    control/
    connectors/
    primitives/
    experts/
  smores_ep/
    state/
    control/
    connectors/
    primitives/
    experts/
```

The exact directory names may stay compatible with the existing ROS package,
but dependency direction must follow this layout: the shared core never
imports either robot adapter.

