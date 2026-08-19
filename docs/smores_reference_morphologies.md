# SMORES-EP reference morphologies

This runbook covers the paper-inspired mobile manipulator, the nine-module
holonomic vehicle, and the helper-assisted RC car. The all-eight-module
obstacle route (`Snake8`, `Bridge8`, `MobileManipulator8`, `RC Car8`) has its
own command sequence in `docs/smores_obstacle_course_runbook.md`. Run every
command from the repository root unless a block says otherwise.

## Build and common runtime

Build the ROS 2 package after changing the expert or its JSON configurations:

```bash
source /opt/ros/humble/setup.bash
cd mssr_ws
colcon build --packages-select mssr_expert --symlink-install
cd ..
```

Start Isaac with the physical-module count required by the target. Use `8` for
the reference mobile manipulator, `9` for the holonomic vehicle, and `8` for the
reference RC car (seven target modules plus one dedicated helper):

```bash
bash scripts/smores_ep/run_self_assembly.sh --module-count 8 --performance
```

In a second terminal, start the file bridge:

```bash
source /opt/ros/humble/setup.bash
python3 ros2_bridge/mssr_file_bridge.py --publish-period 0.2
```

Use a new `execution_id` and `episode_id` for every attempt because primitive
status history is retained.

All target graphs use the same assembly execution policy. Targets describe
topology, role assignment and final posture only; alignment timeouts, contact
gates, retries, docking recovery and contact concurrency are ROS runtime
parameters shared by self-assembly and self-reconfiguration. Independent
modules remain parallel inside each planner wave, with collective `REACH ->
ALIGN -> APPROACH -> DOCK` barriers, two retries per motion phase and two dock
recoveries through a fresh alignment and approach. The default concurrency
value `0` imposes no artificial per-wave cap.

Restart Isaac after changing this package: the simulation imports the Python
control classes only at startup. Isolated modules remain backdrivable while
approaching a dock, but every module in an assembled component retains its
PAN/TILT posture during folding and operation. A fold pusher can energize its
LEFT/RIGHT wheels without releasing those internal joints. Isolated reserve
modules are not captured. Magnetic face constraints remain rigid, but no
articulation is anchored to the world.

## Eight-module PAN-driven RC car

The current RC car uses four serial chassis modules. The four wheel modules
remain attached to the two centerline endpoints. Its `-pi/4` TILT fold runs as
one coordinated four-wheel group: all goals are admitted before Isaac releases
their full targets in the same simulation frame. The previous `0.35 rad`
setpoint-error cap was removed because it reduced the position-drive torque
below the breakaway load of the longer eight-module chassis. Stow uses the
same four-wheel barrier and full target before reconfiguration:

```bash
source /opt/ros/humble/setup.bash
source mssr_ws/install/setup.bash
ros2 run mssr_expert mssr_smores_self_assembly_node --ros-args \
  -p target_graph_path:=$PWD/mssr_ws/src/mssr_expert/config/smores_rc_car8.json \
  -p execution_id:=rc8-paired-assembly-01 \
  -p episode_id:=rc8-paired-assembly-01
```

The behavior name is `rc_car8`; its four wheel modules are driven through PAN.
The legacy `rc_car7` graph and behavior remain available for regression.

## Eight-module mobile manipulator

This target uses the same shared TOP/BOTTOM contact policy as the other
morphologies. It no longer embeds a morphology-local tolerance or retry
override in its target JSON.

In a third terminal:

```bash
source /opt/ros/humble/setup.bash
source mssr_ws/install/setup.bash
ros2 run mssr_expert mssr_smores_self_assembly_node --ros-args \
  -p target_graph_path:=$PWD/mssr_ws/src/mssr_expert/config/smores_mobile_manipulator8.json \
  -p execution_id:=manip-assembly-01 \
  -p episode_id:=manip-assembly-01
```

Keep the successful assembly node running because it publishes the retained
module-to-role assignment. Start the behavior node in a fourth terminal:

```bash
source /opt/ros/humble/setup.bash
source mssr_ws/install/setup.bash
ros2 run mssr_expert mssr_smores_morphology_behavior_node
```

When changing morphology, stop and restart this behavior node after the new
assembly succeeds. A red `command requests ..., but the assembled target is
...` message means it still has the previous morphology assignment; the drive
command was rejected and never reached Isaac.

The following shell helper keeps the individual commands readable:

```bash
send_manip() {
  local command_id="$1"
  local behavior="$2"
  local parameters="$3"
  ros2 topic pub --once /mssr/morphology/command std_msgs/msg/String \
    "{data: '{\"schema_version\":\"mssr.morphology_command.v1\",\"command_id\":\"${command_id}\",\"morphology\":\"mobile_manipulator8\",\"behavior\":\"${behavior}\",\"parameters\":${parameters}}'}"
}
```

Arm postures:

```bash
send_manip arm-up-01 raise_arm '{}'
send_manip arm-forward-01 reach_forward '{}'
send_manip arm-down-01 lower_arm '{}'
```

Forward and backward translation do not change the arm posture. Four modules
form the longitudinal arm: `arm_ground_drive`, directly connected to the
central module, remains at `TILT=0` on the floor; `arm_lift`, `arm_link` and the
end effector form the raised part. Straight translation energizes only the
LEFT/RIGHT wheels of `front_support` and `arm_ground_drive`, with the same local
sign. It does not steer or drive the two lateral PAN joints and there is no
automatic lower/raise cycle:

```bash
send_manip forward-01 drive '{"linear_m_s":0.03,"yaw_rate_rad_s":0.0,"duration_s":5.0}'
send_manip backward-01 drive '{"linear_m_s":-0.03,"yaw_rate_rad_s":0.0,"duration_s":5.0}'
```

Pure rotations keep the raised arm and command opposite local wheel velocities
on the same grounded longitudinal pair. Translation and rotation must be
tested as separate commands:

```bash
send_manip turn-left-01 drive '{"linear_m_s":0.0,"yaw_rate_rad_s":0.3,"duration_s":4.0}'
send_manip turn-right-01 drive '{"linear_m_s":0.0,"yaw_rate_rad_s":-0.3,"duration_s":4.0}'
send_manip stop-01 stop '{}'
```

## Nine-module holonomic vehicle

Restart Isaac with `--module-count 9`, then start the same bridge and run:

```bash
source /opt/ros/humble/setup.bash
source mssr_ws/install/setup.bash
ros2 run mssr_expert mssr_smores_self_assembly_node --ros-args \
  -p target_graph_path:=$PWD/mssr_ws/src/mssr_expert/config/smores_holonomic9.json \
  -p execution_id:=holonomic-assembly-01 \
  -p episode_id:=holonomic-assembly-01
```

After success, start the behavior node and define:

```bash
send_holonomic() {
  local command_id="$1"
  local behavior="$2"
  local parameters="$3"
  ros2 topic pub --once /mssr/morphology/command std_msgs/msg/String \
    "{data: '{\"schema_version\":\"mssr.morphology_command.v1\",\"command_id\":\"${command_id}\",\"morphology\":\"holonomic9\",\"behavior\":\"${behavior}\",\"parameters\":${parameters}}'}"
}
```

Small, separate validation motions are preferable before combining axes:

```bash
send_holonomic x-forward-01 drive '{"linear_m_s":0.02,"lateral_m_s":0.0,"yaw_rate_rad_s":0.0,"duration_s":3.0}'
send_holonomic y-left-01 drive '{"linear_m_s":0.0,"lateral_m_s":0.02,"yaw_rate_rad_s":0.0,"duration_s":3.0}'
send_holonomic rotate-01 drive '{"linear_m_s":0.0,"lateral_m_s":0.0,"yaw_rate_rad_s":0.2,"duration_s":3.0}'
send_holonomic flatten-01 flatten '{}'
send_holonomic deploy-01 deploy '{}'
```

The four external modules keep the straight orientation acquired by the final
fold: locomotion never sends a PAN steering primitive. A longitudinal command
uses the north/south pair with opposite local signs, a lateral command uses the
west/east pair with opposite local signs, and a pure rotation applies the same
local differential-yaw request to all four external modules. The completed
fold remains structurally held throughout locomotion. `flatten` and `deploy`
remain explicit posture commands.

To reconfigure the assembled nine-module vehicle into the seven-module
reference RC car, stop the behavior node and run a fresh execution id:

```bash
ros2 run mssr_expert mssr_smores_self_reconfiguration_node --ros-args \
  -p target_morphology:=rc_car7_reference \
  -p source_graph_path:=auto \
  -p execution_id:=holo-to-rc-reference-01 \
  -p episode_id:=holo-to-rc-reference-01
```

For a count-reducing transition the planner now selects reserves before task
assignment, exclusively among source leaves, and rejects leaf pairs whose
remaining detach/assembly dependencies would form a cycle.

The `-1.35 rad` deployment angle is inferred from the conference video rather
than specified numerically by the paper. Validate ground contact in Isaac and
tune only this value if necessary.

During the final fold four coupled pusher/lifter pairs are actuated as one
barrier group. Each external leaf drives LEFT/RIGHT at `+0.025 m/s` while
retaining both PAN and TILT. At the same time its inner module drives TILT to
`-1.35 rad` and acts as the lifter, and the center module maintains neutral
PAN/TILT throughout the fold. The mapping is target-based (`v5->v1`,
`v6->v2`, `v7->v3`, `v8->v4`), so it follows the physical assignment after a
reconfiguration instead of assuming fixed `smores_XX` IDs. When all four TILT
targets are reached, pusher wheels stop and the runtime atomically captures a
structural HOLD for every PAN/TILT coordinate at its reached angle. Docked
LEFT/RIGHT faces are held, while the free LEFT/RIGHT wheels of each external
TOP-connected drive module remain available for holonomic locomotion.

## Reference RC car with helper

Restart Isaac with `--module-count 8`. The target assigns seven modules and
keeps the eighth as a dedicated helper; the target metadata enables the helper
sequence automatically:

```bash
source /opt/ros/humble/setup.bash
source mssr_ws/install/setup.bash
ros2 run mssr_expert mssr_smores_self_assembly_node --ros-args \
  -p target_graph_path:=$PWD/mssr_ws/src/mssr_expert/config/smores_rc_car7_reference.json \
  -p execution_id:=rc-reference-assembly-01 \
  -p episode_id:=rc-reference-assembly-01
```

The existing `rc_car7` PAN-driven morphology is deliberately unchanged. The
new `rc_car7_reference` profile instead drives the four outer modules through
their LEFT/RIGHT wheels.

## Reconfiguration between eight and nine modules

For `mobile_manipulator8 -> holonomic9`, Isaac must already contain one extra,
isolated reserve module. For `holonomic9 -> mobile_manipulator8`, the planner
detaches one source leaf and leaves it isolated as a reserve.

With Isaac running nine physical modules, choose the target explicitly:

```bash
source /opt/ros/humble/setup.bash
source mssr_ws/install/setup.bash
ros2 run mssr_expert mssr_smores_self_reconfiguration_node --ros-args \
  -p target_graph_path:=$PWD/mssr_ws/src/mssr_expert/config/smores_holonomic9.json \
  -p execution_id:=to-holonomic-01 \
  -p episode_id:=to-holonomic-01
```

Reverse the transition by replacing the target path and using fresh IDs:

```bash
ros2 run mssr_expert mssr_smores_self_reconfiguration_node --ros-args \
  -p target_graph_path:=$PWD/mssr_ws/src/mssr_expert/config/smores_mobile_manipulator8.json \
  -p execution_id:=to-manipulator-01 \
  -p episode_id:=to-manipulator-01
```

For an end-to-end regression with nine physical modules, use the validated
transition chain below. It avoids topology changes which require an
unimplemented intermediate parking maneuver:

```text
holonomic9 -> mobile_manipulator8 -> holonomic9
```

Stop the behavior node before every transition. Run exactly one assembly or
reconfiguration expert at a time; after its success, keep it alive while the
behavior node starts, or let the behavior node recover the unique assignment
from the live physical graph. Seven-module targets leave two of the nine
physical modules isolated as reserves. Before leaving `holonomic9`, the four
inner TILT joints actively unfold to neutral while the external modules and
the center remain structurally stabilized.
