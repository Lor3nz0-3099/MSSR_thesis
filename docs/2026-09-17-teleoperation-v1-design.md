# Teleoperation v1 Design

Date: 2026-09-17  
Branch: `snake8-global-path-ik-recovery`  
Status: approved design

## 1. Objective

Teleoperation becomes the primary source of human demonstrations for imitation learning. The system no longer depends on a fully successful end-to-end deterministic composite expert for data collection.

Human control is responsible for locomotion and manipulation. Existing deterministic self-assembly and self-reconfiguration remain automatic PC-side macros triggered from the controller.

The intended hierarchy is:

```text
Human intent
    -> morphology-specific whole-body controller
    -> module-level actions
    -> Isaac Sim
```

Structural transitions remain:

```text
Controller macro request
    -> self-assembly / self-reconfiguration planner and executor
    -> verified target topology
```

The dataset records the control source explicitly, including `human_teleop`, `automatic_assembly`, and `automatic_reconfiguration`.

## 2. System architecture

Use one external ROS 2 teleoperation node with separate morphology controllers. Do not implement a monolithic node and do not introduce a dynamic plugin system for the three active morphologies.

```text
DualSense
  -> ROS 2 game_controller_node / joy
  -> /joy
  -> mssr_smores_teleop_node
       -> global state machine
       -> recording manager
       -> structural macro manager
       -> camera controller
       -> active morphology controller
            -> RcCarTeleopController
            -> SnakeWholeBodyController
            -> MobileManipulatorTeleopController
       -> module-level action transport
  -> Isaac runtime
```

Active morphologies are restricted to `rc_car8`, `snake8`, and `mobile_manipulator8`.

## 3. DualSense input

Use existing ROS 2 joystick infrastructure rather than writing a custom DualSense driver. Preferred chain:

```text
DualSense -> SDL2/game_controller_node -> sensor_msgs/Joy -> DualSenseInput
```

`DualSenseInput` maps raw axes/buttons into semantic fields such as `left_x`, `left_y`, `right_x`, `right_y`, `l2`, `r2`, named face buttons, shoulder buttons, and START/options. It provides dead-zone handling, trigger normalization to `[0,1]`, rising-edge detection, last-message timestamp, and disconnect detection.

Exact PlayStation face-button mappings remain configurable and are not frozen until explicitly approved.

## 4. Global state machine

```text
STARTUP -> READY
READY -> STRUCTURAL_MACRO -> READY
READY -> ESTOP_PAUSED -> READY only after explicit resume
```

Controller connectivity and recording are orthogonal state variables.

Important state values:

- `requested_morphology`
- `detected_morphology`
- `active_controller`
- `controller_connected`
- `recording`
- `recording_stop_pending`
- `estop_active`

A requested morphology never activates its controller until the physical topology is verified as matching it.

## 5. Control authority

Only one actuator authority may command the robot at a time:

```text
ESTOP > STRUCTURAL_MACRO > TELEOP
```

During assembly/reconfiguration, human module commands are suspended. Camera control and recording may continue.

The teleop path reuses the existing module-action transport rather than introducing a second actuator backend.

## 6. Common locomotion semantics

Where the current mode is locomotion:

- `R2`: analog forward throttle.
- `L2`: analog reverse throttle.
- Releasing both immediately commands zero longitudinal speed.
- Releasing triggers does not reset posture or held targets.

Held-target semantics apply to RC height, Snake head target, and MobileManipulator end-effector target.

The left stick is reserved for the camera.

## 7. Camera

Left stick controls an orbit/follow camera centered on the active robot:

- left-stick X: horizontal orbit
- left-stick Y: vertical orbit

Camera commands do not enter robot action streams and are not BC targets. An Isaac-side runtime bridge applies the camera updates.

## 8. RC-Car8

Inputs:

- `R2/L2`: forward/reverse throttle
- right-stick X: steering
- right-stick Y: incremental chassis-height target with hold

Height control exposes a high-level `desired_chassis_height` and converts it to coordinated wheel-module tilt targets.

PAN and TILT are mutually exclusive on the same SMORES module because they share the internal motor pair. V1 prioritizes steering PAN on a module while retaining the requested height target; TILT convergence resumes when steering demand permits it.

## 9. Snake8 whole-body control

Snake8 uses head-led follow-the-leader whole-body control, not a fixed-duration gait macro.

The operator controls a body-relative head target and longitudinal progression. The connected chain adapts continuously so the body follows the head trajectory.

Inputs:

- `R2/L2`: longitudinal progression along the current head tangent
- right-stick X: incremental lateral head target `y`
- right-stick Y: incremental vertical head target `z`

The right stick is an integrator:

```text
head_target += input * configured_rate * dt
```

When released, the head target is held. If the right stick is untouched while R2 is held, the head continues advancing along its current tangent and the body follows.

## 10. Snake trajectory buffer and backbone

The controller stores a recent spatial trajectory of the head. The remaining modules sample target positions behind the head by arc length:

```text
head: s = 0
module 2: s = -d
module 3: s = -2d
...
tail: s = -7d
```

Spacing `d` is derived from robot geometry.

V1 uses a deterministic geometric backbone with arc-length parameterization, smoothing, curvature limits, height limits, and joint limits. It does not use a learned controller or a large nonlinear optimizer.

Head orientation is automatic from the local backbone tangent.

## 11. Snake PAN/TILT allocation

PAN/TILT exclusion is local to one module, not global to the robot. Different modules may execute PAN and TILT simultaneously.

If the same module needs both in one cycle, the allocator selects one mode from normalized tracking error and uses hysteresis to avoid rapid mode switching.

## 12. Snake wheel invariant and propulsion

For every Snake8 module:

```text
wheel_left_i == wheel_right_i
```

with the same direction and speed.

The robot does not curve through left/right wheel differential within a module. Curvature comes from body configuration and PAN.

Different modules may receive different paired-wheel speeds. V1 starts near-uniform and may apply deterministic corrections from tracking error, curvature, local slope, and chain position.

## 13. Snake recenter and manual override

A soft recenter gradually returns the body-relative head target toward neutral and propagates straightening through the backbone.

Manual module override is a fallback mode. One selected module temporarily leaves automatic allocation while the remaining modules continue whole-body control. Exiting override blends the module smoothly back to its automatic target.

Exact override button mappings remain deferred.

## 14. MobileManipulator8

Two explicit modes:

- `drive_ready`
- `manipulation_ready`

In `drive_ready`, the base is mobile and the arm is held in drive posture.

In `manipulation_ready`, base locomotion is disabled and the end-effector target is controlled in the base frame.

Drive mode:

- `R2/L2`: forward/reverse base motion

Manipulation mode:

- right-stick X: end-effector `Δy`
- right-stick Y: end-effector `Δz`
- `R2`: end-effector `+x`
- `L2`: end-effector `-x`

Targets are held when controls are released. Existing manipulation IK/CLIK must be reused rather than replaced. Manual module override is available as a fallback with smooth reintegration.

## 15. Morphology selection and structural macros

Three face buttons correspond permanently to the three active morphologies. Exact PlayStation symbols remain deferred.

A fourth face button acts as HOME/READY when pressed alone and as an assembly modifier when combined with a morphology button.

Conceptually:

```text
RC button alone    -> reconfigure to RC-Car8
Snake button alone -> reconfigure to Snake8
MM8 button alone   -> reconfigure to MobileManipulator8

Modifier + RC      -> initial RC-Car8 assembly
Modifier + Snake   -> initial Snake8 assembly
Modifier + MM8     -> initial MobileManipulator8 assembly
```

No long press is required. Assembly requests are ignored/rejected if the structure is already assembled.

## 16. Contextual HOME/READY

Pressed alone, the fourth face button is morphology-dependent:

- RC-Car8: return toward nominal chassis-height/posture target
- Snake8: soft backbone recenter
- MobileManipulator8: toggle `drive_ready <-> manipulation_ready`

## 17. Structural Macro Manager

Do not rewrite self-assembly or self-reconfiguration.

Reuse the existing one-shot ROS 2 executables and topology verification behavior. The macro manager launches the correct transition executable, passes target morphology and execution identifiers, waits for topology convergence, waits for the true terminal state including final posture cleanup, and reports success/failure to the teleop state machine.

A future ROS 2 Action refactor is out of scope for Teleoperation v1.

## 18. Topology verification

A button press sets `requested_morphology`; it does not directly activate a controller.

The target controller becomes active only after the physical graph matches the requested morphology. If reconfiguration fails, control follows the actually detected topology. If no supported morphology matches, no morphology controller may publish motion commands.

## 19. DualSense disconnect safe hold

If `/joy` becomes stale beyond a configurable timeout:

- wheel commands go to zero;
- no new PAN/TILT motion is generated;
- held targets are preserved;
- an already-running structural macro continues;
- recording continues and marks a controller-disconnect event.

Motion does not resume until fresh valid input arrives.

## 20. E-STOP

E-STOP is a real Isaac Sim pause. The external teleop node sends a pause request to an Isaac-side runtime bridge; the bridge pauses the Isaac timeline, freezing physics, wheels, PAN/TILT, assembly/reconfiguration execution, and simulated time.

Resume is explicit. After resume, motion remains disarmed until fresh input is received. In particular, locomotion triggers must return to the neutral zone before they can command motion again.

Exact E-STOP and resume buttons remain deferred.

## 21. Recording control

The controller START button is a recording toggle:

```text
START while OFF -> create/start a new episode
START while ON  -> request episode stop
```

If stop is requested during an active structural macro, `recording_stop_pending=true`; recording continues until the macro reaches its terminal state, then the episode is finalized.

## 22. Demonstration dataset semantics

Teleoperation data is treated as demonstrations, not expert-only data.

Each transition preserves, where available:

```text
graph_t
controller_input
high_level_intent
module_action
graph_t_plus_1
```

The primary BC target remains the effective module-level action actually sent to the robot. Controller input and high-level intent are auxiliary context.

Conceptual schema:

```text
mssr.demonstration_transition.v1

episode_id
stream_id
timestep
morphology
control_mode
control_source
graph_t
controller_input
high_level_intent
action.module_actions
graph_t_plus_1
events
```

## 23. Structural streams and manifest

Assembly/reconfiguration keep authoritative raw datasets and are linked into the same teleoperated episode through a manifest rather than copied into a monolithic JSONL.

Example episode streams:

```text
00 rc human drive
01 rc_to_snake automatic reconfiguration
02 snake human whole-body
03 snake_to_mm8 automatic reconfiguration
04 mm8 human drive
05 mm8 human manipulation
```

Runtime location:

```text
logs/teleop/<episode_id>/
```

`dataset_manifest.json` records at least git commit, episode id, start/end timestamps, controller identity, simulator/runtime metadata, streams, record counts, control sources, morphologies, and terminal reason/status.

Raw streams remain authoritative. IL normalization remains offline.

## 24. Control and logging rates

Control-loop rate and dataset logging rate are independent. Initial target ranges:

- control loop: around 50 Hz
- dataset transitions: around 20-30 Hz

These remain configurable and must be measured before finalizing constants.

## 25. Safety and target limits

All human-driven targets require position/workspace bounds, velocity/rate limits, and acceleration/target-slew limits where needed.

Joystick deflection changes desired targets but never bypasses safe target evolution.

Unreachable Cartesian targets are clamped. Temporary MM8 IK failure preserves the last valid command.

## 26. Configuration

Avoid hard-coded control constants. Add configuration conceptually as:

```text
config/smores_teleop.yaml
config/smores_dualsense.yaml
```

The first holds controller behavior, morphology parameters, recording, camera, safety, and timeouts. The second holds physical controller mappings. Face-button mappings are not frozen until separately approved.

## 27. Testing strategy

Most logic must be testable without a real controller or Isaac runtime.

Input tests cover trigger normalization, dead-zones, button edges, START recording toggle, and disconnect timeout.

RC tests cover analog forward/reverse, steering, height hold, nominal-height return, PAN/TILT arbitration, and zero locomotion on trigger release.

Snake tests cover straight trajectory, lateral and vertical head motion, arc-length spacing, local PAN/TILT exclusion, simultaneous PAN and TILT on different modules, `wheel_left_i == wheel_right_i`, follow-the-leader replay, smooth manual reintegration, and soft recenter.

MM8 tests cover drive-mode locomotion, manipulation-mode base lock, Cartesian target integration, target hold, IK failure behavior, and manual-override reintegration.

E-STOP tests cover one pause request per edge, actual Isaac pause, explicit resume, and trigger-neutral latch.

Recording tests cover episode start/stop, deferred stop during structural macros, terminal finalization, and manifest/raw count consistency.

## 28. Implementation milestones

1. **T0 - Controller input:** DualSense recognized, `/joy` inspected, normalized input model verified.
2. **T1 - Teleop shell:** new ROS 2 node, global state machine, no actuator commands yet.
3. **T2 - Runtime controls:** disconnect safe hold, camera, Isaac E-STOP pause/resume bridge.
4. **T3 - RC-Car8:** full RC teleoperation including steering and held height target.
5. **T4 - Recording:** teleoperation demonstration streams and episode manifest.
6. **T5 - Structural macros:** controller-triggered self-assembly and self-reconfiguration with topology verification.
7. **T6 - Snake8:** geometric head-led follow-the-leader whole-body controller.
8. **T7 - MobileManipulator8:** drive/manipulation modes and Cartesian end-effector teleoperation.
9. **T8 - Manual overrides and integration:** per-module fallback control, full recording validation, end-to-end demonstration workflow.

Do not start Snake8 before input, state-machine, RC, recording, and structural-macro infrastructure is stable.

## 29. Explicit non-goals

Teleoperation v1 does not introduce:

- a custom DualSense driver
- a new physics backend
- a new docking protocol
- a rewrite of self-assembly
- a rewrite of self-reconfiguration
- Nav2 as normal teleoperation control
- a learned Snake controller
- a global nonlinear Snake optimizer
- dynamic plugin architecture
- a new monolithic dataset JSONL

## 30. Deferred physical button decisions

These remain configurable and require explicit approval before the mapping is final:

- which of `□ △ ○ ×` selects RC-Car8
- which selects Snake8
- which selects MobileManipulator8
- which acts as HOME/READY/assembly modifier
- E-STOP button
- E-STOP resume button
- manual-override entry/exit button
- previous/next selected-module controls
- direct manual PAN/TILT controls

## 31. End state

The completed system lets the operator start recording with START, assemble an initial morphology from the controller, teleoperate RC-Car8, automatically reconfigure to Snake8, guide Snake8 by controlling its head while the body follows, reconfigure to MobileManipulator8, drive and manipulate Cartesianly, use local module override when needed, pause Isaac through E-STOP, and stop recording with deferred closure if a structural macro is active.

The resulting dataset keeps human intent, module-level actions, graph transitions, structural macro provenance, morphology, and control mode synchronized but semantically separated, providing a direct path from teleoperation demonstrations to BC/DAgger and later MARL.
