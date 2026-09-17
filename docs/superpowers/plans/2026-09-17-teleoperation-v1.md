# Teleoperation v1 Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task in this session. Use TDD and superpowers:verification-before-completion for every milestone.

**Goal:** Collect human MSSR demonstrations using DualSense input, the existing action backend, and automatic structural macros.

**Architecture:** An external ROS 2 shell coordinates independent input, state, recording, camera, macro and morphology controllers. Pure Python logic is tested without Isaac; runtime adapters reuse existing ROS/file transport. Controllers activate only after observed topology verification.

**Tech Stack:** Python 3, ROS 2 Humble, joy/game_controller_node, sensor_msgs/Joy, PyYAML, pytest, existing Isaac SMORES runtime.

**Spec:** `docs/superpowers/specs/2026-09-17-teleoperation-v1-design.md`

## Global Constraints

- Work in `/home/lorenzo/MSSR_thesis`, branch `snake8-global-path-ik-recovery`, as requested. Preserve the pre-existing untracked `docs/2026-09-17-teleoperation-v1-design.md`.
- Active morphologies: `rc_car8`, `snake8`, `mobile_manipulator8` only.
- Reuse `/mssr/actions`, current assembly/reconfiguration executables, topology matching, and existing MM8 IK/CLIK.
- Do not introduce a custom driver, Nav2 teleop, dynamic plugins, learned Snake control, global Snake optimizer, monolithic dataset, or FINAL_HANDOFF.
- Left stick is camera only. Released triggers stop propulsion immediately; posture targets hold.
- PAN/TILT exclusion is per module. Snake paired wheels are equal within each module; different modules may have different speeds.
- All deferred physical assignments remain unset/configurable. START is the approved recording toggle.
- Authority: ESTOP > STRUCTURAL_MACRO > TELEOP. Disconnect does not cancel structural macros or recording.
- E-STOP must pause the real Isaac timeline. Resume requires explicit request, fresh input, and neutral triggers before propulsion is armed.
- Recording stop during structural macros is deferred until the true terminal event, including posture cleanup.
- Before simulation/ROS runtime: inspect relevant process command lines, scoped SIGTERM/wait, SIGKILL surviving matches only, stop ROS daemon, verify cleanup. Never use broad name-based kills.
- Execute one milestone at a time. A hardware or GUI gate without observed evidence remains pending; passing offline tests alone does not complete that milestone.

## Verification and commit protocol

For each task below, execute these steps in order and record evidence in the task's progress entry:

1. Review `git status --short --untracked-files=all`, `git log -3 --oneline --decorate`, and the diff of files to modify.
2. Write the specified behavioral tests before production implementation. Run the targeted command and observe failure caused by the missing behavior.
3. Implement the interfaces and behavior specified below, keeping transport out of pure logic.
4. Rerun targeted tests, then the regression command:

```bash
source /opt/ros/humble/setup.bash
PYTHONPATH=mssr_ws/src/mssr_expert:scripts/smores_ep/src:$PYTHONPATH python3 -m pytest mssr_ws/src/mssr_expert/test scripts/smores_ep/tests -q
```

5. Inspect status, `git diff`, and `git diff --check`; stage only named task files and commit with the task's message after tests are green.
6. Run the specified integration gate. Record pending hardware/runtime evidence separately from automatic checks; do not check a milestone complete until both pass.

## T0 — Controller input

**Files:** Create `mssr_ws/src/mssr_expert/mssr_expert/teleop/{__init__,input}.py`, `config/smores_dualsense.yaml`, `config/smores_teleop.yaml`, `test/test_dualsense_input.py`; modify `package.xml` for sensor_msgs, joy and PyYAML. Create `scripts/teleop/check_dualsense.py`, its shell entry point and `runtime_cleanup.py`, and `test/test_dualsense_probe.py` tests; document the hardware procedure in `docs/teleoperation_v1_runbook.md`.

**Interfaces:** `DualSenseInput(config)` accepts raw axes/buttons with `update(axes, buttons, received_at)`; `snapshot(now)` supplies immutable normalized axes, triggers, pressed buttons, queued rising edges, semantic command edges, last valid receipt time and connectivity. Invalid packets never refresh connectivity. `load_input_config(path)` validates physical and semantic mappings.

**Input behavior:** Configurable stick deadzone with continuous rescaling; configurable trigger released/pressed endpoints normalized to [0,1]; receipt time uses a monotonic wall clock, independent of paused simulation time. Accumulate button edges until consumed once by the control loop. Startup/reconnect held buttons do not synthesize commands. Deferred semantic assignments are null, not invented defaults.

**Test example and first run:**

```python
def test_sdl_trigger_release_does_not_command_half_throttle():
    reader = DualSenseInput(config)
    reader.update([0, 0, 0, 0, 0, 0], [0] * 21, received_at=10.0)
    assert reader.snapshot(10.01).r2 == 0.0
```

```bash
PYTHONPATH=mssr_ws/src/mssr_expert python3 -m pytest mssr_ws/src/mssr_expert/test/test_dualsense_input.py -q
```

**Additional tests:** Full/partial trigger travel and alternate joy_node endpoint conventions; stick bounds/deadzone/sign; held START versus release/repress; edge retention across multiple messages; stale input and malformed/nonfinite/short packets; invalid configuration; reconnect suppresses held command buttons; deferred mapping stays disabled. Probe tests exercise real raw/normalized report aggregation and timed phase validation.

- [x] Red evidence recorded: missing input module; missing probe; subsequent behavioral red/green for kinematic-process selection, exclusive Joy topic, settled/fresh neutral, stale-gap reset, and device-only preflight.
- [x] Normalization/configuration and probe implemented. Ordered events retain multiplicity and modifier context.
- [x] Automatic checks observed: 77 targeted tests, 665 regression tests; shell syntax and Python compilation; independent review findings corrected and re-reviewed.
- [x] Targeted and regression tests green; dedicated commit `fce63c8d1168236286c3d261248e8690fc1189d3`, `feat: add configurable DualSense teleop input`. This is the verified software portion, not hardware acceptance.
- [x] Hardware gate observed and independently replayed: `logs/teleop/hardware_checks/20260917T124255.339508Z/report.json`, 5555 valid samples, no invalid packets, all checks true, driver removal/reopen confirmed, 13.41s disconnect gap. T0 complete; three distinct OPTIONS edges verified (each emitted once).

## T1 — Teleop shell and global state

**Files:** Create `teleop/{state,config,session}.py`, `nodes/smores_teleop_node.py`, `launch/smores_teleop.launch.py`, `test/test_teleop_state.py`, `test/test_teleop_session.py`, and `scripts/teleop/check_teleop_shell.py`; modify `setup.py` for `mssr_smores_teleop_node` entry point and extend scoped cleanup to this acceptance runner.

**Interfaces:** `TeleopState` holds phase, requested/detected morphology, active controller, connection, recording, stop pending and estop fields. `request_morphology(name)`, `observe_topology(name_or_none)`, `begin_macro()`, `finish_macro(success)`, `pause()`, `resume()` expose transitions. Input snapshots are consumed at configurable wall-clock control rate, initially 50 Hz.

```python
def test_requested_morphology_cannot_activate_without_observation():
    state = TeleopState()
    state.request_morphology("snake8")
    assert state.active_controller is None
```

**Tests:** STARTUP/READY transitions; supported-morphology restriction; failed macro follows observed topology; unsupported topology has no controller; estop and macro authority; connectivity/recording remain orthogonal. Shell publishes diagnostics only, no actuator publisher yet.

- [x] Red observed: 21 state tests and 21 session/config tests failed for missing implementations; installed ROS probe failed for missing shell. Cleanup runner classification red/green observed separately.
- [x] Minimal state, input coordination, validated config and installed ROS shell/launch implemented; no motion commands. Real topology verification, pause bridge and recording writer remain later milestones.
- [x] Automatic verification observed: 42 state/session tests and 51 probe tests pass; 727 full regressions pass. Independent review found no important blockers.
- [x] Installed build and ROS gate observed: fresh `build/teleop_t1` avoids an unrelated stale generated config link in the old build tree. `logs/teleop/shell_checks/ba5a717143d141df9b586505fb759df3/report.json` passes: 44 diagnostics, 50.0002 Hz, actual ROS stamps frozen at zero, valid START edges, Joy timeout, no robot publishers, final scoped cleanup zero survivors. First 5-second startup gate timed out with no diagnostics; bounded 20-second startup gate succeeds.
- [x] Commit `5ad1f7d`, `feat: add teleop shell and topology-gated state machine`.

## T2 — Runtime controls

**Files:** Create `teleop/safety.py`, `teleop/camera.py`, `scripts/smores_ep/src/smores_ep/isaac/teleop_runtime.py`, pure runtime tests; modify `ros2_bridge/mssr_file_bridge.py` and `scripts/smores_ep/src/smores_ep/scenarios/parallel_self_assembly.py` to service runtime requests even while physics is paused.

**Interfaces:** `SafetyGate` consumes connection, estop and trigger-neutral state; returns permitted authority and zero-wheel safe-hold intent. `CameraController.step(input, dt)` produces bounded orbit requests. Isaac adapter accepts idempotent pause/resume requests and publishes acknowledgments containing timeline state. Commands use a distinct camera/runtime channel, never robot action or BC streams.

```python
def test_resume_with_held_trigger_keeps_motion_disarmed():
    gate.pause()
    gate.resume(resumed_at=10.0)
    assert not gate.update(connected=True, l2=0.0, r2=1.0).motion_enabled
```

**Tests:** One pause request per edge; explicit resume; fresh-neutral latch; disconnect zero wheels and no new joints; target preservation; ongoing macro survives disconnect; camera bounds and independence. Resume processing must not depend on simulation steps or simulated time.

- [x] Red/green pure targeted safety, camera, adapter and runtime-channel tests; corrective cases included in user-run 779-pass regression.
- [ ] Regression tests; headless Isaac test observes timeline.is_playing false, frozen physics/time, then resumed physics.
- [ ] GUI camera orbit check pending until actually observed; ask for E-STOP/resume physical assignments only for physical validation.
- [ ] Commit `feat: add camera safe hold and Isaac timeline pause bridge`.

**T2 checkpoint:** 26 safety/camera tests, 11 injected-timeline adapter tests and 11 runtime-channel tests observed red/green (48 total). Safety neutral latch, bounded orbit intent, idempotent timeline adapter and ordered request/ack channel implemented. Cleanup recognition of the new native acceptance script observed red; the user-run full regression now confirms green: **776 passed** (user-supplied terminal summary; duration not supplied). Native acceptance script prepared at `scripts/teleop/check_isaac_runtime.py`; execution outside sandbox was rejected, so no native Isaac evidence exists. ROS shell/file-bridge/scenario wiring is not implemented yet. Independent review of T2 is in progress; T2 is not complete.

**User steering:** from this checkpoint the user executes terminal tests and supplies outputs. Do not launch tests or simulation autonomously. Inspect the supplied evidence before marking checks or making implementation commits. User supplied green regression evidence (776 passed). Commit the verified T2 foundation separately after review, then complete shell/file-bridge/scenario wiring with user-run test gates.

**Review gate:** independent review identified two important defects: pause must supersede pending resume delivery; neutral arming must be fenced against actual monotonic resume time, not only previously consumed Joy. Three new behavioral tests were added; their red evidence is pending user execution before production fixes. The earlier 776-pass summary predates these tests. Foundation commit is deferred until corrective red/green and full regression are observed. Next user command: `PYTHONPATH=mssr_ws/src/mssr_expert python3 -m pytest mssr_ws/src/mssr_expert/test/test_teleop_runtime_channel.py mssr_ws/src/mssr_expert/test/test_teleop_safety_camera.py -q --tb=short`.

**Corrective checkpoint:** user supplied the full baseline **776 passed in 18.13s** and the expected corrective red **3 failed, 37 passed in 0.15s**. Production fixes applied: pause preempts unacknowledged resume intent while retaining any already pending pause id; `SafetyGate.resume(resumed_at=...)` now requires explicit monotonic actual-resume time, and disarm preserves that fence through disconnect/invalid input. Tests use explicit synthetic resume times. Follow-up read-only review and user-run green full regression are pending; no tests were run by the agent. Next command is the complete regression above (779 tests expected). Commit and T2 acceptance remain deferred.

**Verified foundation checkpoint:** user supplied final **779 passed**; follow-up review found both corrections addressed with no remaining important issues in those fixes. Foundation committed as `1ff2a57465b5da5fed4862d7e8097ff0d56a82f3`, `feat: add teleop safety camera and runtime channel foundations`. This supersedes preceding pending foundation-commit entries. T2 itself remains incomplete: ROS shell/file-bridge/production-loop wiring, native timeline/physics and GUI camera gates are pending. Next single step is integration tests, with user-run red evidence before production wiring. Current detailed handoff is `2026-09-17-teleoperation-v1-handoff.md` beside this plan. Do not execute terminal tests/runtimes autonomously.

**Integration tests prepared:** after the user pushed `9f7e3db` and authorized a further bounded continuation, three test files add 15 behavior cases: coordinator pause/ack/fresh-neutral/camera/macro safety; ROS file-channel isolation and steady polling without DDS init; paused production-loop app servicing before robot/physics execution. Files: `test/test_teleop_coordinator.py`, `test/test_teleop_file_bridge.py`, `scripts/smores_ep/tests/test_teleop_paused_loop.py`. No production integration code changed and no tests were executed by the agent. User-run red evidence is pending; these tests are uncommitted.

**Integration implementation checkpoint:** user output confirmed **15 failed in 0.38s** for the expected missing boundaries. Added `teleop/coordinator.py` and wired the shell to dedicated runtime request/status topics; ESTOP remains latched until matching observed resume acknowledgment, then fresh-neutral fence uses acknowledgment receipt time. File bridge now routes runtime to separate configurable files and polls on steady time. Scenario services runtime before all robot/macro/physics work; paused branch only updates the app and polls again. Camera follows live module body-link centroid using the existing viewport API. Shell probe uses exclusive runtime topics as well. No agent tests/runtimes executed. Read-only integration review requested; user-run green regression pending (794 expected). No integration commit or T2 completion claimed.

**Integration review:** completed read-only, no important defects found in this bounded change. No reviewer tests/runtimes executed. Next gate remains user-run full regression.

**Integration green checkpoint:** user supplied complete **794 passed in 17.17s**. Integration committed as `39c2839c616bc47da707aea87b4ede2465147f43`, `feat: connect teleop runtime channel and paused Isaac servicing`. This supersedes preceding pending offline integration/commit entries. T2 remains incomplete: real native timeline/physics, updated installed ROS/file/production-loop runtime and GUI camera gates have not run. Next single step is user execution of `ROS_DOMAIN_ID=42 python3 scripts/teleop/check_isaac_runtime.py` after sourcing Humble in the checkout; inspect result/report, do not infer full production acceptance from this adapter-only native probe. Comprehensive camera/safety YAML configuration still needs checking before acceptance. Current handoff has exact continuation instructions; no agent test/runtime execution authorized.

## T3 — RC-Car8

**Files:** Create `teleop/rc_car.py`, `teleop/action_transport.py`, `test/test_rc_car_teleop.py`; modify shell and teleop YAML. Reuse assigned roles and posture geometry from `behaviors/morphology_library.py`, DOF bounds from `behaviors/morphology_dof_model.py`, and existing action payload contract.

**Interfaces:** `RcCarTeleopController.step(input, observation, dt)` returns effective module actions and high-level held height/steering intent. `home()` slews height toward nominal. Transport serializes the existing combined action envelope on `/mssr/actions`.

```python
def test_height_is_held_when_stick_is_released():
    controller.step(height_input, observation, 0.1)
    wanted = controller.desired_chassis_height
    controller.step(neutral_input, observation, 0.1)
    assert controller.desired_chassis_height == wanted
```

**Tests:** Analog forward/reverse; immediate zero on release; right-X steering; rate/bounds on height; nominal return; same-module steering PAN priority retains height and resumes TILT when free. No left-stick robot command.

- [ ] Red/green RC and transport tests; regression suite.
- [ ] Isaac RC smoke scenario verifies actual motion, steering/height and safe hold.
- [ ] Commit `feat: implement RC-Car8 held-height teleoperation`.

## T4 — Recording and manifest

**Files:** Create `teleop/recording.py`, `test/test_teleop_recording.py`; modify shell. Reuse graph serialization, `dataset/dataset_logger.py`, and manifest concepts from `scripts/smores_ep/src/smores_ep/dataset/composite_dataset.py` without merging raw streams.

**Interfaces:** `RecordingManager.toggle(macro_active)`, `record_transition(graph_t, input, intent, effective_actions, graph_next, events)`, `macro_terminal(status)` manage episode/stream lifecycle. Logging rate is independent of control rate, initially 25 Hz, measured during validation.

```python
def test_stop_during_macro_defers_manifest_finalization():
    recorder.toggle(macro_active=False)
    recorder.toggle(macro_active=True)
    assert recorder.recording_stop_pending
    assert recorder.recording
```

**Tests:** Start/stop edges; deferred stop and true terminal finalization; disconnect event without stopping; effective-action BC target; raw structural stream references; manifest counts equal raw line counts; provenance and failure terminal metadata; atomic manifest writes; distinct stream per source/morphology/mode.

- [ ] Red/green recorder tests in a temporary directory; regression suite.
- [ ] ROS replay checks actual action/graph synchronization and measured rates.
- [ ] Commit `feat: record teleoperation episodes with authoritative stream manifest`.

## T5 — Structural macros

**Files:** Create `teleop/macros.py`, `teleop/topology.py`, `test/test_teleop_macros.py`; modify shell. Reuse `nodes/smores_parallel_self_assembly_node.py`, `nodes/smores_self_reconfiguration_node.py`, their terminal state topics and existing attributed topology matching/executor completion behavior.

**Interfaces:** `StructuralMacroManager.request(target, assembly, episode_id)` launches existing one-shot executable with unique execution id, target/config, dataset path and scoped state topic; `observe_terminal(payload)` correlates execution and waits for final cleanup plus topology convergence. `detect_supported_topology(graph)` returns observed name and assignments or none; ambiguous matches do not activate motion.

```python
def test_topology_convergence_alone_does_not_finish_macro():
    manager.observe_topology("snake8")
    assert manager.active
```

**Tests:** Modifier chord versus HOME alone; assembly rejected if assembled; repeated requests while busy; matching topology before activation; late/wrong execution ids ignored; cleanup terminal needed; subprocess failure; failed transition falls back to real supported topology; disconnect continues; deferred recording closes only at terminal.

- [ ] Red/green macro and topology tests; regression suite.
- [ ] Isaac assembly and RC-to-Snake integration observed with distinct raw streams.
- [ ] Commit `feat: trigger existing structural macros from teleop requests`.

## T6 — Snake8 whole body

**Files:** Create `teleop/snake_backbone.py`, `teleop/snake.py`, `teleop/joint_allocator.py`, `test/test_snake_teleop.py`; modify YAML and shell. Derive module spacing from existing `smores_ep.config.geometry.SmoresGeometry`; reuse joint geometry/limits rather than gait macros.

**Interfaces:** `TrajectoryBuffer.append(point)` and `sample_behind(distance)` use spatial arc length, seeded with the observed chain. `SnakeWholeBodyController.step(input, observation, dt)` integrates body-relative head y/z and signed progression along current tangent, computes automatic orientation, samples seven following targets, and emits deterministic geometric corrections. `JointAllocator.allocate(module_id, pan_error, tilt_error)` applies normalized-error hysteresis locally.

```python
def test_paired_wheels_are_equal_for_every_module():
    commands = controller.step(curved_input, observation, 0.02)
    for command in commands.module_actions.values():
        assert command["wheel_left"] == command["wheel_right"]
```

**Tests:** Straight/lateral/vertical trajectories; signed forward/reverse; body-frame rotation; arc-length spacing; held targets; advancing untouched head follows tangent; bounded history, smoothing/curvature/height/joint constraints; per-module PAN/TILT exclusion with simultaneous different-module modes; per-module paired-wheel equality; deterministic follow-the-leader replay; soft recenter propagates through buffer.

- [ ] Start only after T0–T5 are stable and their verification gates are satisfied.
- [ ] Red/green geometry/allocation/controller tests; regression suite.
- [ ] Isaac curved and vertical trajectories verify connected chain behavior and effective wheel invariants.
- [ ] Commit `feat: add Snake8 head-led geometric follow-the-leader teleop`.

## T7 — MobileManipulator8

**Files:** Create `teleop/mobile_manipulator.py`, shared IK adapter and tests; extract reusable existing math from `scripts/smores_ep/run_button_expert_to_ik.py` only as needed, preserving `scripts/smores_ep/tests/test_button_ik_math.py` regression behavior; modify shell/YAML.

**Interfaces:** `MobileManipulatorTeleopController.home()` explicitly toggles drive/manipulation; `step(input, observation, dt)` integrates EE targets in base coordinates and invokes existing IK/CLIK through an adapter. Drive holds drive posture; manipulation emits zero base propulsion. Unreachable targets clamp and failed IK preserves last valid command.

```python
def test_manipulation_trigger_moves_target_without_driving_base():
    controller.home()
    output = controller.step(forward_input, observation, 0.02)
    assert all(pair == (0.0, 0.0) for pair in output.base_wheels)
    assert output.end_effector_target[0] > previous_x
```

**Tests:** Drive throttle; explicit modes; base lock; y/z/x Cartesian integration and hold; frame transform; workspace and slew bounds; IK failure and recovery; extracted math produces same existing IK results.

- [ ] Red/green MM8 and IK regression tests; full regression suite.
- [ ] Isaac drive/manipulation switching and actual Cartesian tracking observed.
- [ ] Commit `feat: reuse MobileManipulator IK for Cartesian teleoperation`.

## T8 — Module override and final integration

**Files:** Create `teleop/override.py`, `test/test_teleop_override.py`, `test/test_teleop_integration.py`; modify Snake/MM8 controllers and runbook; add end-to-end capture/manifest checker under `scripts/teleop/`.

**Interfaces:** `ManualOverride.enter(module_id)`, `step(manual_input, dt)`, `exit()` temporarily remove one module from automatic allocation and blend toward live automatic targets with bounded slew. Selection and PAN/TILT bindings remain configurable until approved.

```python
def test_reintegration_never_jumps_to_automatic_target():
    override.exit()
    command = override.step(neutral_input, 0.02)
    assert abs(command.pan - previous_pan) <= configured_pan_rate * 0.02
```

**Tests:** Exactly one overridden module; remaining chain still automatic; same-module PAN/TILT exclusion; Snake wheel invariant during override; bounded reintegration toward moving targets; disconnect and estop override safety; episode START/assembly/RC/reconfigure/Snake/reconfigure/MM8/override/E-STOP/STOP replay; manifest consistency and control-source attribution.

- [ ] Red/green override and integration tests; full regression suite.
- [ ] Hardware assignments explicitly approved and full Isaac demonstration observed.
- [ ] Check raw counts, runtime rates, terminal status and actual pause/resume; retain evidence.
- [ ] Commit `feat: integrate manual module overrides and teleop demonstration workflow`.

## Progress and handoff

- Design saved unchanged and committed as `3014511`; trailing Markdown hard-break spaces retained from approved source.
- Initial baseline without the ROS environment: five collection errors for missing ROS Python packages. Sourcing `/opt/ros/humble/setup.bash` resolved these without code changes; baseline: 588 passed.
- Latest automatic verification: 77 targeted tests passed in 0.32s; complete suite 665 passed in 18.29s. Actual tiny subprocess checks demonstrated SIGTERM, SIGKILL only for an ignoring selected process, and survival of an unrelated process. ROS daemon response is isolated in those process tests.
- Final read-only independent review: no remaining T0 software blockers. Review did not execute hardware/runtime.
- Sandbox preflight could not open the ROS daemon socket. Authorized retry outside the sandbox completed cleanup: selected=0, survivors=0, daemon not running. SDL enumeration exited 0 with no devices; hardware result false, `selected_dualsense_not_detected`.
- Preflight evidence: `logs/teleop/hardware_checks/20260917T115107.346827Z/report.json` (ignored runtime log). No actual Joy hardware samples were observed.
- T0 remains **in progress**, awaiting timed DualSense acceptance evidence. T1–T8 have not started. No milestone T0–T8 is complete.
- Deferred physical choices remain null. None is needed for the T0 command.
- Detailed session handoff: `docs/superpowers/plans/2026-09-17-teleoperation-v1-handoff.md`.
- Hardware follow-up: user enumeration recognized Sony `PS5 Controller`, GUID `030000004c050000e60c000011810000`, but Fast DDS initialization rejected the original default domain 239. The probe now defaults to 42, rejects invalid/noncanonical domains before runtime, and cleans up prior probe instances in this checkout. Observed red/green: 19 new regression cases; targeted total **96 passed**, full suite **684 passed in 18.83s**. Real `rmw_fastrtps_cpp` initialization and synthetic Joy pub/sub on domain 42 passed after scoped cleanup; this is middleware evidence, not hardware acceptance. CLI checks for 239 and 08 returned clear configuration errors before cleanup/runtime. Independent review found no remaining issues. T0 still awaits the six-phase hardware report.
- Acceptance supersedes the preceding historical pending entries: hardware report on commit `fa481b776eeb36b1f901b3f99a6ba5adcef0ada3` passed. Raw and normalized samples replayed with monotonic timeout polls, checked against reported fields, phases and counts; driver log confirms real unplug/replug. T0 complete, T1 next. Report SHA256: `78ba5b925ecb05a8eb9cedd03dd0c402f5a27a2623da1ae4c376a1b80fcf2ad0`; raw SHA256: `2fbb414a4c211011e2d6b540c50e44e4aa948c8d9f838b9b0a866032fac1ea0a`.
- At session end record HEAD SHA, finished/pending tasks, exact test results, open physical decisions, complete git status, and one next command with exact requested output.
