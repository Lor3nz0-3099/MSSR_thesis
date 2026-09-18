# Teleoperation v1 verification

Implementation follows [the approved design](superpowers/specs/2026-09-17-teleoperation-v1-design.md) and [milestone plan](superpowers/plans/2026-09-17-teleoperation-v1.md).

## Accepted milestones: T0, T1 and T3

Input normalization and its automatic checks are separate from the hardware acceptance gate. No later milestone is declared complete by these tests. No robot runtime is required for T0.

T0 acceptance was observed in `logs/teleop/hardware_checks/20260917T124255.339508Z/`: report checks all true; 5555 valid raw/normalized samples independently replayed, zero invalid packets, trigger/stick full travel, distinct START edges and a 13.41-second disconnect gap with driver removal/reopen confirmed. This validates controller input, not later morphology or Isaac behavior.

The shipped input configuration uses ROS `joy/game_controller_node` SDL order. Sticks apply a configurable deadzone once; the driver is launched with `deadzone:=0.0`. ROS SDL trigger values are 0 at rest and -1 fully pressed. Generic `joy_node` mappings may use different indices/endpoints and require an explicit configuration. See [the ROS joy README](https://github.com/ros-drivers/joystick_drivers/blob/3.3.0/joy/README.md) and [driver conversion](https://github.com/ros-drivers/joystick_drivers/blob/3.3.0/joy/src/game_controller.cpp).

Positive normalized X means stick right, positive Y means stick up. Morphology controllers must explicitly convert this to their coordinate frame. Left stick is reserved for camera. The user-approved mapping is Circle for HOME and Triangle alternating E-stop/resume; morphology/manual assignments remain deferred. OPTIONS/START is the recording toggle. Reconfiguration/assembly directions on the D-pad belong to future milestones.

Input code uses monotonic receipt time, independent of ROS/simulated time. Invalid packets do not refresh connectivity. Snapshot consumers must gate motion on `connected`. Ordered `command_events` preserve repeated toggles, and `button_events` preserve held modifier context; the set views are diagnostics. Startup/reconnect held buttons need release and repress to generate an edge.

## Single hardware command

Connect your DualSense by USB or Bluetooth, leave controls neutral, then run:

```bash
bash /home/lorenzo/MSSR_thesis/scripts/teleop/check_dualsense.sh
```

The command sources Humble, performs scoped cleanup, stops the current-domain ROS daemon, enumerates SDL devices and starts the existing game_controller_node. Its normal `/joy` output is remapped to an exclusive per-run topic shared with the probe, so another controller publisher cannot supply acceptance evidence. It saves raw Joy and normalized input alongside controller identity, commit and result. Current-user runtime command lines must also have this checkout as their working-directory context to be selected for cleanup; editors, tests, arbitrary shell strings and other projects are excluded. PID start time and full command are rechecked before signalling. SIGKILL is reserved for selected survivors of SIGTERM. Cleanup failure aborts the probe.

The probe uses ROS_DOMAIN_ID=42 unless already set. Only canonical ASCII decimal values in [0,232] are accepted; invalid values, leading zeros and alternate encodings are rejected before cleanup or middleware initialization. The original default 239 exceeded DDS UDP port limits and was corrected after the first real-controller attempt. See [ROS domain constraints](https://github.com/ros2/ros2_documentation/blob/humble/source/Concepts/Intermediate/About-Domain-ID.rst). A previous probe still running from this checkout is included in scoped cleanup; the current probe and its ancestors are excluded. Default device id is 0; if enumeration shows the DualSense at another id, rerun the same command with `--device-id ID`. Sony DualSense/Edge identity is checked from the SDL GUID; unidentified hardware fails rather than being silently accepted.

Follow the timed instructions printed on the terminal:

1. Neutral, 5 seconds: do not touch controls.
2. Travel, 25 seconds: move both sticks fully left/right/up/down; slowly press each trigger through partial and full travel and release.
3. START, 10 seconds: press/release OPTIONS twice.
4. Disconnect, 15 seconds: unplug USB or switch the Bluetooth controller off; leave it disconnected for the whole phase.
5. Reconnect, 20 seconds: reconnect with sticks/triggers neutral.
6. Final neutral, 5 seconds: release everything. Acceptance requires fresh neutral sticks/triggers and released buttons during the last half-second of the neutral phase.

Return the complete `T0_HARDWARE_RESULT=...` line and the contents of the `report.json` path printed as `REPORT=...`. If the result is false, also return `game_controller.log` from that directory when present. Logs remain under `logs/teleop/hardware_checks/<timestamp>/` and are Git-ignored. Do not infer a hardware pass from an offline replay or device enumeration alone.

## Automatic regression command

```bash
source /opt/ros/humble/setup.bash
PYTHONPATH=mssr_ws/src/mssr_expert:scripts/smores_ep/src:$PYTHONPATH python3 -m pytest mssr_ws/src/mssr_expert/test scripts/smores_ep/tests -q
```

## T1 installed shell acceptance

At T1 acceptance the shell published `/mssr/teleop/status` JSON at the configured wall-clock rate (default 50 Hz), with input/state/recording flags and no robot commands. The current T2 shell also publishes a separate camera/timeline runtime channel; its native behavior remains unverified. It does not write recording episodes or publish robot actions. Observed-topology activation is tested at the internal verified-result boundary; runtime matching belongs to T5. Physical assignments stay deferred.

Build using a fresh generated build directory to avoid the existing stale config symlink:

```bash
source /opt/ros/humble/setup.bash
cd /home/lorenzo/MSSR_thesis/mssr_ws
colcon build --build-base build/teleop_t1 --packages-select mssr_expert --symlink-install
```

Run the automatic installed ROS acceptance from the checkout:

```bash
source /opt/ros/humble/setup.bash
cd /home/lorenzo/MSSR_thesis
source mssr_ws/install/setup.bash
ROS_DOMAIN_ID=42 python3 scripts/teleop/check_teleop_shell.py
```

This runs scoped cleanup before and after a real launch, sends synthetic Joy on an exclusive topic, exercises START press/hold/release/repress and stale input, and verifies actual ROS clock stamps remain zero while diagnostics progress. `T1_SHELL_RESULT` must report `passed=true`; reports and captured statuses are ignored under `logs/teleop/shell_checks/`. It does not require physical button decisions.

Accepted T1 evidence: `logs/teleop/shell_checks/ba5a717143d141df9b586505fb759df3/report.json`, 44 diagnostics at 50.0002 Hz, all gates passing, final cleanup zero survivors. This predates T2 integration.

## T2 pending native gate

Foundation and production integration are committed, with user-run **794 passed in 17.17s** and read-only review. Native Isaac pause/frozen physics/resume, updated ROS runtime and GUI camera acceptance remain pending; T2 is not complete. The user executes terminal tests and runtime checks.

Next native adapter check, not yet executed:

```bash
cd /home/lorenzo/MSSR_thesis
source /opt/ros/humble/setup.bash
ROS_DOMAIN_ID=42 python3 scripts/teleop/check_isaac_runtime.py
```

Return the complete `T2_RUNTIME_RESULT=...` line and the contents of JSON printed as `REPORT=...`; on failure include relevant errors from that directory's `isaac.log`. This check uses a native rigid-body scene and the runtime adapter with scoped cleanup. It does not establish acceptance of the full production scenario or ROS transport. No physical E-STOP/resume assignment is needed for its explicit synthetic requests.

## T3 observed RC-Car8 acceptance

The preceding T2 timeline-gate notes describe an earlier checkpoint; current teleop uses a structure-only E-stop and keeps physics and camera running.

Real DualSense/native Isaac acceptance passed on commit `63276ea857dd9c97777f02a51df1e8db4718741f`: `logs/teleop/rc_car_checks/5b9efd91f2ce411c8a1bdf79e9654310/report.json`, `passed=true`, all 22 checks true, final cleanup zero survivors and daemon stopped. Software regression at that checkpoint passed 905 tests. The probe deliberately closes Isaac after observing the final zero command and archives native JSON evidence into the persistent report directory.

Run the same scoped probe to check the current camera distance:

```bash
ROS_DOMAIN_ID=42 bash /home/lorenzo/MSSR_thesis/scripts/teleop/check_rc_car.sh
```

R2/L2 control forward/reverse; right-X steers, right-Y raises/lowers the held chassis-height target. Circle returns gradually to nominal height. Left stick orbits the camera. The shipped `camera.radius_m` in `mssr_ws/src/mssr_expert/config/smores_teleop.yaml` is 1.188536394 m, matching this assembly scene's default startup distance (spawn radius 0.34 m). Custom scenes or spawn radii can use a different configured orbit distance; it is not measured dynamically from the viewport.

For `resume_held_fence`, keep R2 pressed continuously while releasing and pressing Triangle again. Release R2 only when `fresh_neutral` appears. The probe requires the held-trigger fence before accepting fresh-neutral rearm; releasing R2 before resume skips that evidence even if resume works. The camera-distance follow-up still requires the user's visual confirmation on a new real run.
