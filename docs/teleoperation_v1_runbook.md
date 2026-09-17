# Teleoperation v1 verification

Implementation follows [the approved design](superpowers/specs/2026-09-17-teleoperation-v1-design.md) and [milestone plan](superpowers/plans/2026-09-17-teleoperation-v1.md).

## Current milestone: T0, hardware verification pending

Input normalization and its automatic checks are separate from the hardware acceptance gate. No later milestone is declared complete by these tests. No robot runtime is required for T0.

The shipped input configuration uses ROS `joy/game_controller_node` SDL order. Sticks apply a configurable deadzone once; the driver is launched with `deadzone:=0.0`. ROS SDL trigger values are 0 at rest and -1 fully pressed. Generic `joy_node` mappings may use different indices/endpoints and require an explicit configuration. See [the ROS joy README](https://github.com/ros-drivers/joystick_drivers/blob/3.3.0/joy/README.md) and [driver conversion](https://github.com/ros-drivers/joystick_drivers/blob/3.3.0/joy/src/game_controller.cpp).

Positive normalized X means stick right, positive Y means stick up. Morphology controllers must explicitly convert this to their coordinate frame. Left stick is reserved for camera. Morphology/HOME, E-STOP/resume and manual-control assignments are null until explicitly approved. OPTIONS/START is the recording toggle.

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
