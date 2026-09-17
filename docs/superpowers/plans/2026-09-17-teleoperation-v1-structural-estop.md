# Structural E-STOP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use TDD and execute this plan task-by-task.

**Goal:** Replace Isaac timeline pause/resume with a structure-only emergency stop while keeping physics and camera running.

**Architecture:** The teleop coordinator latches E-STOP intent. The runtime channel transports a structure-stop state rather than a timeline operation. The Isaac production loop gives that state highest actuator priority, applies zero-wheel/current-posture hold, skips normal robot/macro commands while stopped, but continues normal simulation stepping.

**Tech Stack:** Python 3, ROS 2 Humble, Isaac Sim, pytest.

**Spec:** `docs/superpowers/specs/2026-09-17-teleoperation-v1-structural-estop-amendment.md`

## Global constraints

- Preserve the unrelated untracked `docs/2026-09-17-teleoperation-v1-design.md`.
- Never pause/play the Isaac timeline for teleoperation E-STOP.
- Authority remains `ESTOP > STRUCTURAL_MACRO > TELEOP`.
- E-STOP during a structural macro interrupts it; no automatic macro resume.
- Resume requires fresh input and neutral L2/R2.
- Camera and physics continue during E-STOP.
- Make surgical changes only; no broad reset/clean/restore.

## Task 1 — Remove timeline-specific E-STOP plumbing

Tests first:

- runtime request/status no longer contains `timeline_request`,
  `timeline_ack`, or `timeline_playing`;
- coordinator E-STOP no longer waits for a timeline pause/resume acknowledgment;
- obsolete native timeline checker tests are removed.

Production:

- remove timeline control from `teleop/runtime_channel.py`;
- remove timeline control from `smores_ep/isaac/teleop_runtime.py`;
- remove paused-loop special servicing;
- delete `scripts/teleop/check_isaac_runtime.py`;
- remove its timeout CLI test.

## Task 2 — Add latched structure-stop runtime state

Tests first:

- E-STOP request becomes active idempotently;
- repeated stop does not recapture/reset the posture incorrectly;
- explicit clear changes only the emergency-stop latch;
- camera requests remain independent.

Production runtime request/status carries structure-stop state and acknowledgment.

## Task 3 — Apply emergency hold at the actuator boundary

Tests first:

- all wheel commands become zero;
- PAN/TILT hold the measured posture reached at stop entry;
- normal robot/action/primitive work is skipped while stopped;
- simulation stepping continues while stopped;
- releasing E-STOP alone does not synthesize motion.

Production loop gives structure stop priority before normal command routing.

## Task 4 — Macro interruption and safe rearming

Tests first:

- E-STOP while macro-active marks the macro interrupted;
- no automatic macro continuation occurs after clear;
- detected topology is authoritative after interruption;
- held trigger cannot rearm locomotion;
- fresh post-resume neutral input can rearm.

Actual subprocess termination hooks are completed with the structural macro manager in T5; T2 must already guarantee that interrupted macro commands cannot reach actuators while E-STOP is active.

## Task 5 — T2 regression and runtime acceptance

Run targeted tests, then the complete regression.

Runtime acceptance must demonstrate:

1. robot moving or actuated;
2. structural E-STOP engaged;
3. robot actuators held/stopped;
4. Isaac simulation time continues;
5. physics/contact processing continues;
6. explicit clear plus fresh-neutral sequence;
7. robot remains stationary until a new post-neutral command.

GUI acceptance also verifies camera operation during structural E-STOP.
