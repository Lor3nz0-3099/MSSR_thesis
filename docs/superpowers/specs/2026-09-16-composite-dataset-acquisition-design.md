# Composite Dataset Acquisition Design

## Goal

Redesign composite-course data acquisition so that each deterministic expert
writes an authoritative raw dataset stream owned by that stage/expert, while
the final imitation-learning target remains:

    graph_t -> expert_action at module level -> graph_t_plus_1

For RC-Car/Nav2 locomotion, the supervised BC target is the effective module/
wheel command produced by the morphology behavior node after conversion from
Nav2 cmd_vel. Nav2 cmd_vel remains context/diagnostic information and is not a
second competing BC target.

The existing expert_v1_compact representation remains the derived compact
format and must not be replaced by a second incompatible compact schema.

## Current Problem

The composite runtime currently passes one shared dataset.jsonl path to
multiple independent producers.

Known producers include:

- persistent morphology behavior DatasetLogger;
- self-assembly DatasetLogger;
- self-reconfiguration DatasetLogger;
- run_rc_car_nav2_route.py direct JSONL append;
- run_button_expert_to_ik.py behavior/reconfiguration/manipulation writers.

These writers have independent timesteps, stage ids, sampling cadences and
record schemas, but append into the same file.

This causes:

1. duplicated supervision, especially during Nav2;
2. ambiguous provenance;
3. independently-reset timestep/stage_id counters inside one stream;
4. possible concurrent appends by unrelated writers;
5. very large raw files;
6. difficult per-skill sampling-frequency analysis.

The failed C05 run produced approximately 4.3 GB for one incomplete episode.
The separate normalize_dataset whole-file OOM has already been fixed by
streaming normalization; this design addresses the acquisition architecture
itself.

## Authoritative Raw Data

Raw expert demonstrations remain authoritative.

Acquisition must preserve full-fidelity expert transitions before any
compaction. No alias removal, RLE, constant extraction or graph reconstruction
is performed inside the simulation/runtime acquisition path.

Compaction remains an offline validated transformation.

A raw transition intended for imitation learning must retain, directly or
through its phase contract:

- graph_t;
- expert_action;
- graph_t_plus_1;
- action_valid;
- supervision information;
- task/stage provenance;
- terminal/success state.

## Per-Writer Ownership

No two independent processes may append learning transitions to the same JSONL
file.

Each stage or nested expert phase receives its own dataset path.

Example C05 layout:

    composite-c05-<run>/
      dataset_parts/
        00-gap-assembly.jsonl
        01-gap-behavior.jsonl
        02-rc-reconfiguration.jsonl
        03-rc-behavior.jsonl
        04-button/
          rc-alignment.jsonl
          rc-to-mm8-reconfiguration.jsonl
          manipulation.jsonl
          mm8-to-rc-reconfiguration.jsonl
      dataset_manifest.json

Exact filenames may be derived from composite stage id and task id, but writer
ownership must remain explicit and deterministic.

## RC-Car / Nav2 Supervision

Nav2 is the high-level trajectory generator/controller.

For BC, the action label is NOT the raw Nav2 cmd_vel.

The primary locomotion transition must be recorded by the morphology behavior
node after cmd_vel has been converted into effective commands for the active
RC-Car modules.

Therefore:

    state:
        graph_t

    policy context:
        task / route / goal / optional Nav2 context

    supervised action:
        expert_action.locomotion at module level

    successor:
        graph_t_plus_1

run_rc_car_nav2_route.py may continue recording route progress, goal,
completion metrics and cmd_vel for diagnostics, but it must not emit a second
BC transition stream into the same learning dataset.

Diagnostic Nav2 artifacts may be JSON/JSONL, but they are not counted as the
authoritative module-action behavior phase.

## Button Expert

run_button_expert_to_ik.py operates as a nested composite expert and contains
multiple semantically distinct phases.

When invoked with an external composite runtime it must not receive one shared
dataset path for every internal writer.

The composite must provide a stage-owned dataset directory or explicit
per-phase paths so that:

- RC alignment/locomotion supervision;
- RC -> MobileManipulator8 reconfiguration;
- manipulation/IK supervision;
- MobileManipulator8 -> RC reconfiguration;

remain separately attributable.

The manipulation writer retains the operational DoF / IK information required
by the existing button compact representation.

## Assembly and Reconfiguration

Assembly and reconfiguration experts keep their own DatasetLogger output.

Their stage dataset path is supplied by the composite orchestrator and is
unique for that invocation.

The expert-local timestep may start at zero because each raw file is an
independent phase stream.

Global episode ordering is represented by the composite dataset manifest, not
by forcing every subprocess to share one counter.

## Composite Dataset Manifest

Every composite run writes dataset_manifest.json.

The manifest records ordered raw streams rather than requiring immediate
physical concatenation.

Minimum information for each stream:

    stage_id
    task_id
    phase
    producer
    path
    action_space
    source_morphology
    target_morphology when applicable
    intended_for_behavior_cloning
    completion status

The ordered manifest is the authoritative description of episode chronology.

A failed episode keeps all completed/partial raw streams and marks the episode
unsuccessful. Acquisition data is never destroyed merely because a later stage
fails.

## Sampling Cadence

This architecture change does not initially change sampling frequency.

First remove duplicate writers and obtain per-stream byte/record-rate
measurements.

Only after measuring the unique authoritative streams will sampling periods be
changed.

Any later downsampling must be skill-specific and must preserve terminal and
transition-critical events.

## Compact Dataset Compatibility

The existing build_expert_v1.py compaction principles remain authoritative.

The compact representation must continue to:

- preserve graph_t, expert_action and graph_t_plus_1;
- remove attributed_graph only after proving equality with graph_t;
- remove attributed_task_graph only after proving equality with task_graph_t;
- remove graph-derived observation aliases only after validation;
- move phase-wide constants to phase manifests only after exact validation;
- retain dynamic operational DoFs, supervision, FSM/primitive context and
  task-dependent metadata;
- collapse only consecutive semantically identical transitions;
- retain _source_repeat_count and source index/timestep/stamp ranges.

RLE multiplicity must later be respected by the training DataLoader, either by
virtual expansion or equivalent sample weighting.

## Composite Import / Compaction

After acquisition has been corrected, build_expert_v1.py will be extended to
recognize successful composite episodes through dataset_manifest.json.

The importer must process each declared raw stream independently, preserving
phase boundaries.

A compact composite episode may therefore contain multiple ordered phase JSONL
files rather than one monolithic file.

The raw composite run remains the source of truth.

## Memory and I/O Requirements

Runtime acquisition must remain streaming.

No acquisition/finalization path may require loading a full JSONL file into
RAM.

Large JSONL processing must use line-by-line iteration and temporary-file plus
atomic-replace semantics when rewriting is necessary.

Independent writers must never concurrently append to the same learning JSONL.

## Testing Requirements

Implementation follows TDD.

Tests must prove at least:

- composite stages receive distinct dataset paths;
- Nav2 direct writer is not a duplicate BC producer when composite module-level
  behavior supervision is active;
- morphology behavior records module-level locomotion actions during Nav2;
- button nested phases use distinct raw paths;
- assembly/reconfiguration receive their own stage paths;
- dataset_manifest.json preserves execution order and producer provenance;
- failed episodes retain their raw stream manifest;
- no regression to the streaming normalize_dataset behavior;
- existing expert_v1 / expert_v1_compact contracts remain valid.

Full relevant suites must pass before runtime validation.

## Runtime Validation

The first runtime validation target is composite-c05.

During the run measure:

- records per raw stream;
- bytes per raw stream;
- average bytes per transition;
- effective sampling rate;
- whether any duplicate BC supervision remains.

The episode must no longer generate multiple learning records for the same RC
Nav2 control tick from independent writers.

Only after these measurements will dataset sampling cadence be tuned.

## Non-Goals

This change does not:

- introduce a centralized ROS dataset service;
- change BC/MARL model architecture;
- compact data online;
- delete existing raw expert_v1 data;
- change the existing compact dataset semantics;
- change Nav2 navigation behavior;
- change deterministic expert control logic.

## Repository Integration

Implementation must preserve unrelated local work.

No blanket git reset/restore/clean is permitted.

After the dataset architecture is validated, the complete remaining source,
tests, configs and documentation in the working tree will be reviewed and
pushed separately. Generated logs, datasets, runtime artifacts and temporary
backups are not part of that source push.
