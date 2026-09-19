# Teleoperation v1 — T4 Recording Amendment

Date: 2026-09-18
Status: approved design

This amendment supersedes the recording/dataset parts of the original
Teleoperation v1 design where they conflict with the decisions below.
The structure-only E-STOP amendment remains authoritative.

## T4 decisions

- OPTIONS/START starts recording; the next edge requests stop.
- Holding OPTIONS/START must not retrigger.
- Stop during a structural macro is deferred until its true terminal event.
- Recording completion and task success are separate concepts.

- Raw sessions live under `logs/teleop/recordings/<episode_id>/`.
- Human behavior is one continuous `human_behavior.jsonl` stream.
- Old demonstrations are never reused as data for a new session.

- Human records reuse `mssr.expert_transition.v3`.
- Human provenance is `human_expert`.
- Deterministic experts remain `deterministic_expert`.
- The primary BC target is the effective module action after safety/ownership.
- Controller input and high-level intent are auxiliary context.

- A transition is finalized only when its fresh `graph_t_plus_1` exists.
- Control remains 50 Hz; dataset sampling starts at configurable 10 Hz.

- Disk writes run in a background writer behind a bounded RAM queue.
- Queue saturation or writer failure marks the recording failed.
- No records may be silently dropped.
- Robot control must continue even if recording fails.

- Each episode has an atomic manifest containing provenance, commit,
  timestamps, recording status, task success, configured/measured rate,
  stream counts, morphologies, events, config, errors and structural refs.

- Assembly/reconfiguration keep their own deterministic-expert streams.
- T4 only references them; launching structural macros belongs to T5.

- Import and compaction happen offline after collection.
- Raw output must remain compatible with the existing expert/compact pipeline.
- T4 is not complete until software tests and the ROS synchronization/rate
  gate both pass.

## Structural data and graph-state coverage

Assembly and reconfiguration occurring while a teleoperation episode is open
must produce new deterministic-expert data for that same demonstration.

They remain separate authoritative streams, but the episode manifest links them
into the same temporal demonstration flow together with the continuous human
behavior stream. Existing old demonstrations must never be substituted for data
from the current episode.

Each recorded transition preserves the attributed robot graph. The graph state
represents individual modules and their relations, retaining available module
attributes such as position, orientation, relative position, linear/angular
velocity, role and attachment state, plus graph edges for contacts/connections
and relevant global/environment/task context.

Human transitions additionally retain controller input and high-level intent,
while the primary BC target remains the effective per-module action after
safety and actuator ownership.
