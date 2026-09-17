# Teleoperation v1 — Structural E-STOP Amendment

Date: 2026-09-17

This amendment supersedes the Isaac-timeline E-STOP requirements in
`2026-09-17-teleoperation-v1-design.md`.

## E-STOP semantics

E-STOP is an emergency stop of the robot structure, not an Isaac Sim
timeline pause.

Isaac physics, simulation time and camera servicing continue normally.

Authority remains:

ESTOP > STRUCTURAL_MACRO > TELEOP

While E-STOP is active:

- wheel targets are zero;
- no new locomotion command reaches the modules;
- PAN/TILT are held at the actually reached posture;
- no new module-motion or structural primitive may progress;
- camera control remains available;
- physics and contacts continue to evolve;
- existing physical dock connections are preserved.

Engaging E-STOP during assembly/reconfiguration interrupts that structural
macro. The interrupted macro must not resume automatically after E-STOP is
cleared. The actually detected topology becomes authoritative before any
new morphology controller or structural transition is allowed.

Resume is explicit. After resume, locomotion remains disarmed until fresh
valid controller input has been received and L2/R2 have returned to their
neutral zone.

Controller-disconnect safe-hold semantics remain unchanged.

## Runtime transport

The Isaac runtime channel remains useful for camera requests and for the
latched structural-stop request/status. It must no longer call
`timeline.pause()` or `timeline.play()` and must not use timeline state as
the E-STOP acknowledgment.

## T2 acceptance

T2 validates:

- disconnect safe hold;
- camera runtime path;
- structural E-STOP priority and hold behavior;
- continued Isaac physics while structural E-STOP is active;
- explicit resume and fresh-neutral rearming.

The old native cube freeze/resume checker is obsolete and must be removed.
