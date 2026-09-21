"""Interactive T7 acceptance gate for MobileManipulator8 manual teleoperation.

This module also exposes ROS-independent helper functions used by the T7
checker contract tests.  The native interactive execution path is added only
after these physical/topological contracts are validated offline.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from uuid import uuid4

from check_rc_car import (
    CONFIG,
    ROOT,
    finalize_runtime,
    preflight,
    runtime_commands,
)


def mm8_runtime_commands(
    output,
    input_path,
    run_id,
    device_id,
    *,
    runtime_dir,
):
    """Build the standard teleop stack with direct MM8 self-assembly."""

    commands = runtime_commands(
        output,
        input_path,
        run_id,
        device_id,
        runtime_dir=runtime_dir,
    )

    commands["assembly"] = [
        "ros2",
        "run",
        "mssr_expert",
        "mssr_smores_self_assembly_node",
        "--ros-args",
        "-p",
        f"target_graph_path:={CONFIG / 'smores_mobile_manipulator8.json'}",
        "-p",
        f"dataset_path:={output / 'assembly.jsonl'}",
        "-p",
        f"execution_id:=t7-{run_id}",
    ]

    commands["teleop"] = [
        part.replace(
            "mssr_rc_teleop_",
            "mssr_mm8_teleop_",
        )
        for part in commands["teleop"]
    ]

    return commands


def physical_metrics(payload, target):
    """Resolve the live physical MM8 assignment and actuator state."""

    from mssr_expert.graph.serialization import (
        attributed_graph_from_dict,
    )
    from mssr_expert.teleop.mobile_manipulator import (
        MobileManipulatorObservation,
    )

    graph = attributed_graph_from_dict(payload)

    observation = MobileManipulatorObservation.from_graph(
        graph,
        target,
    )

    if observation is None:
        raise ValueError(
            "physical topology is not a valid MobileManipulator8"
        )

    nodes = graph.node_by_id()

    roles = {
        item.target_role: item.module_id
        for item in observation.assignments
    }

    positions = {}
    pan = {}
    tilt = {}

    try:
        for item in observation.assignments:
            attributes = nodes[item.module_id].attributes

            pose = attributes["pose"]

            positions[item.module_id] = tuple(
                float(value)
                for value in pose["position"]
            )

            actuators = attributes["actuators"]

            pan[item.module_id] = float(
                actuators["pan"]["position_rad"]
            )

            tilt[item.module_id] = float(
                actuators["tilt"]["position_rad"]
            )

    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "invalid physical MobileManipulator8 actuator state"
        ) from error

    required_roles = {
        "chassis_center",
        "right_drive",
        "front_support",
        "left_drive",
        "arm_ground_drive",
        "arm_lift",
        "arm_link",
        "end_effector",
    }

    if set(roles) != required_roles:
        raise ValueError(
            "physical topology is not a valid MobileManipulator8"
        )

    if not all(
        math.isfinite(value)
        for value in (
            *pan.values(),
            *tilt.values(),
            *(
                coordinate
                for position in positions.values()
                for coordinate in position
            ),
        )
    ):
        raise ValueError(
            "nonfinite physical MobileManipulator8 observation"
        )

    locomotor_ids = (
        roles["front_support"],
        roles["arm_lift"],
    )

    locomotor_center = tuple(
        sum(
            positions[module_id][axis]
            for module_id in locomotor_ids
        )
        / 2.0
        for axis in range(3)
    )

    return {
        "stamp": graph.stamp,
        "roles": roles,
        "positions": positions,
        "pan": pan,
        "tilt": tilt,
        "locomotor_ids": locomotor_ids,
        "locomotor_center": locomotor_center,
    }


def translation_pair_invariant(actions, physical):
    """Allow held posture targets, but locomotion only on the MM8 drive pair."""

    try:
        roles = physical["roles"]

        expected = {
            roles["front_support"],
            roles["arm_lift"],
        }

        if not expected.issubset(actions):
            return False

        for module_id, command in actions.items():
            vx = float(command.get("vx", 0.0))
            vy = float(command.get("vy", 0.0))
            yaw_rate = float(command.get("yaw_rate", 0.0))

            if not all(
                math.isfinite(value)
                for value in (vx, vy, yaw_rate)
            ):
                return False

            if module_id in expected:
                if (
                    abs(vy) >= 1.0e-9
                    or abs(yaw_rate) >= 1.0e-9
                ):
                    return False

            elif (
                abs(vx) >= 1.0e-9
                or abs(vy) >= 1.0e-9
                or abs(yaw_rate) >= 1.0e-9
            ):
                return False

        return True

    except (AttributeError, KeyError, TypeError, ValueError):
        return False

def physical_wheel_rates(payload, target):
    """Read native wheel feedback for the two MM8 locomotor modules."""

    physical = physical_metrics(
        payload,
        target,
    )

    from mssr_expert.graph.serialization import (
        attributed_graph_from_dict,
    )

    graph = attributed_graph_from_dict(
        payload
    )

    nodes = graph.node_by_id()
    rates = {}

    try:
        for module_id in physical["locomotor_ids"]:
            actuators = (
                nodes[module_id]
                .attributes["actuators"]
            )

            left = float(
                actuators["left_wheel"][
                    "velocity_rad_s"
                ]
            )
            right = float(
                actuators["right_wheel"][
                    "velocity_rad_s"
                ]
            )

            if not (
                math.isfinite(left)
                and math.isfinite(right)
            ):
                raise ValueError(
                    "nonfinite wheel feedback"
                )

            rates[module_id] = {
                "left": left,
                "right": right,
            }

    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "native graph has no valid MM8 wheel readback"
        ) from error

    return rates


def recorded_mm8_actions(path):
    """Summarize only effective MobileManipulator8 human actions."""

    empty = {
        "rows": 0,
        "wheel_rows": 0,
        "shape_rows": 0,
        "manual_pan_rows": 0,
        "manual_tilt_rows": 0,
        "manual_selected_modules": [],
    }

    if not path.is_file():
        return empty

    rows = 0
    wheel_rows = 0
    shape_rows = 0
    manual_pan_rows = 0
    manual_tilt_rows = 0
    selected_modules = set()

    for line in path.read_text(
        encoding="utf-8"
    ).splitlines():
        if not line:
            continue

        row = json.loads(line)

        if (
            row.get("task_type")
            != "mobile_manipulator8_teleop"
        ):
            continue

        rows += 1

        actions = (
            row.get(
                "expert_action",
                {},
            )
            .get(
                "locomotion",
                {},
            )
        )

        if any(
            abs(
                float(
                    command.get(
                        "vx",
                        0.0,
                    )
                )
            )
            > 1.0e-5
            for command in actions.values()
        ):
            wheel_rows += 1

        has_shape = any(
            (
                "pan_target_rad"
                in command
            )
            or (
                "tilt_target_rad"
                in command
            )
            for command in actions.values()
        )

        if has_shape:
            shape_rows += 1

        intent = (
            row.get(
                "observation",
                {},
            )
            .get(
                "intent",
                {},
            )
        )

        selected = intent.get(
            "selected_module_id"
        )

        if (
            intent.get("control_mode")
            == "manual"
            and selected in actions
        ):
            selected_modules.add(
                selected
            )

            if (
                "pan_target_rad"
                in actions[selected]
            ):
                manual_pan_rows += 1

            if (
                "tilt_target_rad"
                in actions[selected]
            ):
                manual_tilt_rows += 1

    return {
        "rows": rows,
        "wheel_rows": wheel_rows,
        "shape_rows": shape_rows,
        "manual_pan_rows": manual_pan_rows,
        "manual_tilt_rows": manual_tilt_rows,
        "manual_selected_modules": sorted(
            selected_modules
        ),
    }




def isaac_subprocess_environment(inherited):
    """Remove ROS/Gazebo overlays from the native Isaac subprocess only."""

    environment = dict(inherited)

    contaminated = {
        "PYTHONPATH",
        "LD_LIBRARY_PATH",
        "AMENT_PREFIX_PATH",
        "CMAKE_PREFIX_PATH",
        "COLCON_PREFIX_PATH",
        "ROS_DISTRO",
        "ROS_VERSION",
        "ROS_PYTHON_VERSION",
        "RMW_IMPLEMENTATION",
        "ROS_DOMAIN_ID",
        "GAZEBO_MODEL_PATH",
        "GAZEBO_PLUGIN_PATH",
    }

    for name in contaminated:
        environment.pop(name, None)

    return environment


def runtime_process_environment(name, inherited):
    """Return the child environment appropriate for one runtime process."""

    if name == "isaac":
        return isaac_subprocess_environment(
            inherited
        )

    return dict(inherited)

def mm8_mode_matches(
    status,
    mode,
    *,
    pending,
    behavior=None,
    selected_role=None,
):
    """Match the MM8 FSM state exposed by the ROS teleop status."""

    try:
        if status.get("active_controller") != "mobile_manipulator8":
            return False

        intent = status["mm8_intent"]

        if intent.get("mode") != mode:
            return False

        if intent.get("mode_transition_pending") is not pending:
            return False

        if (
            behavior is not None
            and intent.get("mode_transition_behavior") != behavior
        ):
            return False

        if (
            selected_role is not None
            and intent.get("selected_role") != selected_role
        ):
            return False

        return True

    except (AttributeError, KeyError, TypeError):
        return False

def _validate_cli(args):
    """Validate all arguments before cleanup or runtime launch."""

    mapping = preflight(
        args.input_config
    )

    if (
        mapping.commands.get(
            "record_toggle"
        )
        is None
    ):
        raise ValueError(
            "recording button must be configured"
        )

    if (
        isinstance(args.device_id, bool)
        or args.device_id < 0
    ):
        raise ValueError(
            "device ID must be nonnegative"
        )

    if (
        not math.isfinite(
            args.assembly_timeout
        )
        or args.assembly_timeout <= 0.0
    ):
        raise ValueError(
            "assembly timeout must be positive and finite"
        )

    return mapping


def run_native_acceptance(args, mapping):
    """Run the native Isaac + DualSense T7 MM8 acceptance sequence."""

    from check_dualsense import (
        configure_probe_environment,
        parse_devices,
    )
    from runtime_cleanup import scoped_cleanup
    from mssr_expert.graph.serialization import load_attributed_graph

    import rclpy
    from std_msgs.msg import String

    output = (
        ROOT
        / "logs/teleop/mobile_manipulator_checks"
        / uuid4().hex
    )

    configure_probe_environment(
        os.environ,
        output,
    )

    output.mkdir(parents=True)

    runtime_dir = Path(
        tempfile.mkdtemp(
            prefix=f"mssr-mm8-{output.name}-",
            dir="/dev/shm",
        )
    )

    commands = mm8_runtime_commands(
        output,
        args.input_config,
        output.name,
        args.device_id,
        runtime_dir=runtime_dir,
    )

    summary = {
        "passed": False,
        "source": "real_dualsense_and_native_isaac",
        "checks": {},
        "runtime_dir": str(runtime_dir),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
        ).strip(),
        "git_worktree_status": subprocess.check_output(
            [
                "git",
                "status",
                "--short",
                "--untracked-files=all",
            ],
            cwd=ROOT,
            text=True,
        ).splitlines(),
    }

    target = load_attributed_graph(
        CONFIG / "smores_mobile_manipulator8.json"
    )

    processes = []
    observer = None

    latest = {
        "status": None,
        "assembly": None,
    }

    records = (
        output / "observations.jsonl"
    ).open(
        "w",
        encoding="utf-8",
    )

    def launch(name, command):
        with (
            output / f"{name}.log"
        ).open(
            "w",
            encoding="utf-8",
        ) as log:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=runtime_process_environment(
                    name,
                    os.environ,
                ),
            )

        processes.append(
            (name, process)
        )

        return process

    def receive(kind, message):
        payload = json.loads(
            message.data
        )

        latest[kind] = payload

        records.write(
            json.dumps(
                {
                    "received_at": time.monotonic(),
                    "kind": kind,
                    "payload": payload,
                },
                allow_nan=False,
            )
            + "\n"
        )

        records.flush()

    def read(name):
        try:
            return json.loads(
                (
                    runtime_dir
                    / name
                ).read_text(
                    encoding="utf-8"
                )
            )

        except (OSError, ValueError):
            return {}

    def state():
        status = latest["status"]

        if (
            status is None
            or not (
                0.0
                <= (
                    time.monotonic()
                    - status["stamp_monotonic"]
                )
                <= 0.5
            )
        ):
            raise ValueError(
                "stale teleop status"
            )

        return status

    def metrics():
        return physical_metrics(
            read("robot_graph.json"),
            target,
        )

    def actions():
        return state().get(
            "mm8_effective_actions",
            {},
        )

    def zero():
        try:
            return all(
                abs(
                    float(
                        command.get(
                            "vx",
                            0.0,
                        )
                    )
                )
                < 1.0e-9
                and abs(
                    float(
                        command.get(
                            "vy",
                            0.0,
                        )
                    )
                )
                < 1.0e-9
                and abs(
                    float(
                        command.get(
                            "yaw_rate",
                            0.0,
                        )
                    )
                )
                < 1.0e-9
                for command in actions().values()
            )

        except (
            AttributeError,
            TypeError,
            ValueError,
        ):
            return False

    def wait(
        label,
        instruction,
        predicate,
        timeout=45,
    ):
        print(
            f"[{label}] {instruction}",
            flush=True,
        )

        deadline = (
            time.monotonic()
            + timeout
        )

        while (
            time.monotonic()
            < deadline
        ):
            for (
                process_name,
                process,
            ) in processes:
                if (
                    process.poll()
                    is not None
                ):
                    raise RuntimeError(
                        f"{process_name} exited "
                        f"{process.returncode}; "
                        f"inspect "
                        f"{output / (process_name + '.log')}"
                    )

            rclpy.spin_once(
                observer,
                timeout_sec=0.02,
            )

            if predicate():
                summary["checks"][
                    label
                ] = True
                return

        summary["failure_context"] = {
            "phase": label,
            "teleop_status": (
                latest["status"]
            ),
            "native_runtime_status": read(
                "smores_teleop_runtime_status.json"
            ),
            "robot_graph_stamp": read(
                "robot_graph.json"
            ).get("stamp"),
            "primitive_status": read(
                "primitive_status.json"
            ),
        }

        raise TimeoutError(
            f"{label} did not produce "
            "observed acceptance evidence"
        )

    drive_sequence = 0

    def drive(direction):
        nonlocal drive_sequence

        drive_sequence += 1

        evidence_key = {
            1: "forward_drive_evidence",
            2: "reverse_drive_evidence",
            3: "post_restore_drive_evidence",
        }.get(
            drive_sequence,
            f"drive_evidence_{drive_sequence}",
        )

        before = metrics()

        trigger = (
            "r2"
            if direction > 0
            else "l2"
        )

        peak_actual_wheel_rad_s = 0.0

        def observed():
            nonlocal peak_actual_wheel_rad_s

            status = state()
            current = metrics()
            commands_now = actions()

            actual_wheels = (
                physical_wheel_rates(
                    read("robot_graph.json"),
                    target,
                )
            )

            peak_actual_wheel_rad_s = max(
                peak_actual_wheel_rad_s,
                *(
                    abs(value)
                    for rates
                    in actual_wheels.values()
                    for value
                    in rates.values()
                ),
            )

            displacement = math.hypot(
                current[
                    "locomotor_center"
                ][0]
                - before[
                    "locomotor_center"
                ][0],
                current[
                    "locomotor_center"
                ][1]
                - before[
                    "locomotor_center"
                ][1],
            )

            summary[
                evidence_key
            ] = {
                "direction": direction,
                "trigger": trigger,
                "trigger_value": (
                    status[
                        "controller_input"
                    ][trigger]
                ),
                "displacement_m": (
                    displacement
                ),
                "commands": commands_now,
                "actual_wheel_rad_s": (
                    actual_wheels
                ),
                "peak_abs_actual_wheel_rad_s": (
                    peak_actual_wheel_rad_s
                ),
                "mode": status[
                    "mm8_intent"
                ].get("mode"),
            }

            try:
                commanded_direction = all(
                    direction
                    * float(
                        commands_now[
                            module_id
                        ].get(
                            "vx",
                            0.0,
                        )
                    )
                    > 0.005
                    for module_id
                    in current[
                        "locomotor_ids"
                    ]
                )

            except (
                KeyError,
                TypeError,
                ValueError,
            ):
                commanded_direction = False

            return (
                status[
                    "controller_input"
                ][trigger]
                > 0.15
                and status[
                    "safety"
                ][
                    "motion_enabled"
                ]
                and mm8_mode_matches(
                    status,
                    "drive_ready",
                    pending=False,
                )
                and translation_pair_invariant(
                    commands_now,
                    current,
                )
                and commanded_direction
                and peak_actual_wheel_rad_s
                > 0.1
                and displacement
                > 0.005
            )

        return observed

    try:
        summary["cleanup"] = (
            scoped_cleanup(ROOT)
        )

        enumeration = subprocess.run(
            [
                "/opt/ros/humble/lib/joy/"
                "joy_enumerate_devices"
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )

        summary[
            "enumeration"
        ] = enumeration.stdout

        device = next(
            (
                item
                for item
                in parse_devices(
                    enumeration.stdout
                )
                if (
                    item["device_id"]
                    == args.device_id
                )
            ),
            None,
        )

        if (
            enumeration.returncode
            or device is None
            or not device["gamepad"]
            or not device["dualsense"]
        ):
            raise RuntimeError(
                "selected Sony DualSense "
                "not detected"
            )

        summary["device"] = device

        rclpy.init(args=[])

        observer = rclpy.create_node(
            "mssr_mm8_acceptance"
        )

        observer.create_subscription(
            String,
            (
                "/mssr/teleop_probe/"
                f"run_{output.name}/status"
            ),
            lambda message: receive(
                "status",
                message,
            ),
            100,
        )

        observer.create_subscription(
            String,
            "/mssr/expert/self_assembly/state",
            lambda message: receive(
                "assembly",
                message,
            ),
            100,
        )

        launch(
            "isaac",
            commands["isaac"],
        )

        launch(
            "bridge",
            commands["bridge"],
        )

        wait(
            "native_ready",
            (
                "Attendi Isaac; lascia "
                "il DualSense a riposo."
            ),
            lambda: (
                0.0
                <= (
                    time.monotonic()
                    - read(
                        "smores_teleop_"
                        "runtime_status.json"
                    ).get(
                        "stamp_monotonic",
                        -1.0e12,
                    )
                )
                <= 0.5
            ),
            timeout=240,
        )

        assembly = launch(
            "assembly",
            commands["assembly"],
        )

        def assembled():
            status = latest["assembly"]

            if (
                status
                and status.get("done")
                and not status.get(
                    "success"
                )
            ):
                raise RuntimeError(
                    "MM8 assembly failed: "
                    + status.get(
                        "message",
                        "",
                    )
                )

            return bool(
                status
                and status.get("done")
                and status.get("success")
            )

        wait(
            "mm8_assembly",
            (
                "Attendi la self-assembly "
                "MobileManipulator8 diretta."
            ),
            assembled,
            timeout=args.assembly_timeout,
        )

        def mm8_topology_ready():
            try:
                return (
                    metrics()["stamp"]
                    > 0
                )

            except (
                KeyError,
                TypeError,
                ValueError,
            ):
                return False

        wait(
            "assembly_topology_ready",
            (
                "Attendo il graph fisico "
                "MM8 dopo il terminale "
                "assembly."
            ),
            mm8_topology_ready,
            timeout=30,
        )

        summary[
            "assembled_physical"
        ] = metrics()

        assembly.terminate()
        assembly.wait(timeout=5)

        processes[:] = [
            (name, process)
            for name, process
            in processes
            if process is not assembly
        ]

        wait(
            "mm8_topology",
            (
                "Verifica la topologia "
                "fisica MobileManipulator8 "
                "assemblata."
            ),
            mm8_topology_ready,
            timeout=30,
        )

        launch(
            "joy",
            commands["joy"],
        )

        launch(
            "teleop",
            commands["teleop"],
        )

        wait(
            "drive_ready_neutral",
            (
                "Rilascia trigger e stick; "
                "attendi Scorpion/drive_ready."
            ),
            lambda: (
                latest["status"]
                is not None
                and state()[
                    "safety"
                ][
                    "motion_enabled"
                ]
                and mm8_mode_matches(
                    state(),
                    "drive_ready",
                    pending=False,
                )
                and zero()
            ),
            timeout=45,
        )

        wait(
            "recording_start",
            (
                "Premi START una volta "
                "per iniziare la "
                "registrazione."
            ),
            lambda: (
                state()["recording"]
                and bool(
                    state().get(
                        "recording_episode_id"
                    )
                )
                and state().get(
                    "recording_backend_ready"
                )
            ),
            timeout=45,
        )

        episode_id = state()[
            "recording_episode_id"
        ]

        wait(
            "forward",
            (
                "Premi R2 circa 1/3: "
                "MM8 deve avanzare in "
                "Scorpion."
            ),
            drive(1),
            timeout=60,
        )

        wait(
            "forward_release",
            "Rilascia R2.",
            lambda: (
                state()[
                    "controller_input"
                ]["r2"]
                == 0
                and zero()
            ),
        )

        wait(
            "reverse",
            (
                "Premi L2 circa 1/3: "
                "MM8 deve arretrare."
            ),
            drive(-1),
            timeout=60,
        )

        wait(
            "reverse_release",
            "Rilascia L2.",
            lambda: (
                state()[
                    "controller_input"
                ]["l2"]
                == 0
                and zero()
            ),
        )

        home_button = (
            mapping.commands["home"]
        )

        wait(
            "to_manipulation_ready",
            (
                f"Premi {home_button} "
                "(Circle/HOME) una volta."
            ),
            lambda: (
                mm8_mode_matches(
                    state(),
                    "to_manipulation_ready",
                    pending=True,
                    behavior=(
                        "prepare_manipulation"
                    ),
                )
                and zero()
            ),
            timeout=45,
        )

        wait(
            "manipulation_ready",
            (
                "Attendi il completamento "
                "della postura "
                "manipulation_ready."
            ),
            lambda: (
                mm8_mode_matches(
                    state(),
                    "manipulation_ready",
                    pending=False,
                    selected_role=(
                        "end_effector"
                    ),
                )
                and zero()
            ),
            timeout=180,
        )

        before_pan = metrics()

        first_selected = (
            state()["mm8_intent"][
                "selected_module_id"
            ]
        )

        def manual_pan():
            status = state()
            current = metrics()
            selected = (
                status[
                    "mm8_intent"
                ].get(
                    "selected_module_id"
                )
            )

            delta = (
                abs(
                    current["pan"][
                        selected
                    ]
                    - before_pan["pan"][
                        selected
                    ]
                )
                if (
                    selected
                    in current["pan"]
                    and selected
                    in before_pan["pan"]
                )
                else 0.0
            )

            summary[
                "manual_pan_evidence"
            ] = {
                "selected_module_id": (
                    selected
                ),
                "selected_role": (
                    status[
                        "mm8_intent"
                    ].get(
                        "selected_role"
                    )
                ),
                "pan_delta_rad": delta,
                "commands": actions(),
            }

            return (
                selected
                == first_selected
                and mm8_mode_matches(
                    status,
                    "manipulation_ready",
                    pending=False,
                    selected_role=(
                        "end_effector"
                    ),
                )
                and status[
                    "controller_input"
                ][
                    "right_x"
                ]
                > 0.45
                and zero()
                and delta
                > 0.01
            )

        wait(
            "manual_pan",
            (
                "Stick destro a DESTRA: "
                "PAN del solo "
                "end_effector."
            ),
            manual_pan,
            timeout=90,
        )

        wait(
            "manual_pan_release",
            "Rilascia lo stick destro.",
            lambda: (
                abs(
                    state()[
                        "controller_input"
                    ][
                        "right_x"
                    ]
                )
                < 0.05
                and zero()
            ),
        )

        next_button = (
            mapping.commands.get(
                "next_module"
            )
            or "R1"
        )

        wait(
            "manual_next_module",
            (
                f"Premi {next_button} una "
                "volta: seleziona arm_link."
            ),
            lambda: (
                mm8_mode_matches(
                    state(),
                    "manipulation_ready",
                    pending=False,
                    selected_role=(
                        "arm_link"
                    ),
                )
                and not state()[
                    "mm8_intent"
                ].get(
                    "manual_transition_pending"
                )
            ),
            timeout=45,
        )

        second_selected = (
            state()["mm8_intent"][
                "selected_module_id"
            ]
        )

        before_tilt = metrics()

        def manual_tilt():
            status = state()
            current = metrics()
            selected = (
                status[
                    "mm8_intent"
                ].get(
                    "selected_module_id"
                )
            )

            delta = (
                abs(
                    current["tilt"][
                        selected
                    ]
                    - before_tilt[
                        "tilt"
                    ][
                        selected
                    ]
                )
                if (
                    selected
                    in current["tilt"]
                    and selected
                    in before_tilt["tilt"]
                )
                else 0.0
            )

            summary[
                "manual_tilt_evidence"
            ] = {
                "selected_module_id": (
                    selected
                ),
                "selected_role": (
                    status[
                        "mm8_intent"
                    ].get(
                        "selected_role"
                    )
                ),
                "tilt_delta_rad": delta,
                "commands": actions(),
            }

            return (
                selected
                == second_selected
                and mm8_mode_matches(
                    status,
                    "manipulation_ready",
                    pending=False,
                    selected_role=(
                        "arm_link"
                    ),
                )
                and status[
                    "controller_input"
                ][
                    "right_y"
                ]
                > 0.45
                and zero()
                and delta
                > 0.01
            )

        wait(
            "manual_tilt",
            (
                "Stick destro verso ALTO: "
                "TILT del solo arm_link."
            ),
            manual_tilt,
            timeout=90,
        )

        wait(
            "manual_tilt_release",
            "Rilascia lo stick destro.",
            lambda: (
                abs(
                    state()[
                        "controller_input"
                    ][
                        "right_y"
                    ]
                )
                < 0.05
                and zero()
            ),
        )

        wait(
            "to_drive_ready",
            (
                f"Premi {home_button} "
                "(Circle/HOME) una volta "
                "per ripristinare Scorpion."
            ),
            lambda: (
                mm8_mode_matches(
                    state(),
                    "to_drive_ready",
                    pending=True,
                    behavior="restore_drive",
                )
                and zero()
            ),
            timeout=45,
        )

        wait(
            "drive_ready_restored",
            (
                "Attendi il completamento "
                "di restore_drive."
            ),
            lambda: (
                mm8_mode_matches(
                    state(),
                    "drive_ready",
                    pending=False,
                )
                and zero()
            ),
            timeout=180,
        )

        wait(
            "post_restore_forward",
            (
                "Premi di nuovo R2 circa "
                "1/3: la locomozione deve "
                "essere riabilitata."
            ),
            drive(1),
            timeout=60,
        )

        wait(
            "pre_estop_drive",
            "Premi R2 circa 1/3.",
            drive(1),
            timeout=60,
        )

        wait(
            "estop",
            "Mantieni R2 premuto e premi TRIANGOLO (E-stop).",
            lambda: (
                state()["safety"]["authority"] == "ESTOP"
                and state()["runtime_structure_stopped"] is True
                and zero()
            ),
        )

        stopped = metrics()
        since = time.monotonic()

        wait(
            "stop_physics_camera",
            (
                "In E-stop muovi lo stick SINISTRO: "
                "camera e fisica devono continuare."
            ),
            lambda: (
                time.monotonic() - since > 1.0
                and metrics()["stamp"] > stopped["stamp"]
                and read(
                    "smores_teleop_runtime_status.json"
                ).get("camera_applied") is True
                and any(
                    abs(
                        state()["controller_input"][axis]
                    ) > 0.3
                    for axis in ("left_x", "left_y")
                )
                and zero()
            ),
        )

        wait(
            "resume_fence",
            (
                "Tieni R2 premuto; premi di nuovo TRIANGOLO "
                "per Resume."
            ),
            lambda: (
                state()["runtime_structure_stopped"] is False
                and state()["controller_input"]["r2"] > 0.1
                and not state()["safety"]["motion_enabled"]
                and zero()
            ),
        )

        wait(
            "fresh_neutral",
            (
                "Rilascia trigger e stick: serve un nuovo "
                "input neutro per riarmare."
            ),
            lambda: (
                state()["safety"]["motion_enabled"]
                and zero()
                and state()["controller_input"]["r2"] == 0
            ),
        )

        wait(
            "post_resume_forward",
            "Premi nuovamente R2 circa 1/3.",
            drive(1),
            timeout=60,
        )

        wait(
            "final_zero",
            "Rilascia tutti i controlli.",
            lambda: (
                state()[
                    "controller_input"
                ]["r2"]
                == 0
                and state()[
                    "controller_input"
                ]["l2"]
                == 0
                and zero()
            ),
        )

        wait(
            "recording_stop",
            (
                "Premi START una seconda "
                "volta per chiudere la "
                "registrazione."
            ),
            lambda: (
                not state()["recording"]
            ),
            timeout=45,
        )

        human = (
            ROOT
            / "logs/teleop/recordings"
            / episode_id
            / "human_behavior.jsonl"
        )

        manifest_path = (
            human.parent
            / "manifest.json"
        )

        def recording_written():
            if not manifest_path.is_file():
                return False

            manifest = json.loads(
                manifest_path.read_text(
                    encoding="utf-8"
                )
            )

            if (
                manifest.get("status")
                == "failed"
            ):
                raise RuntimeError(
                    "MM8 recording failed: "
                    + str(
                        manifest.get(
                            "error"
                        )
                    )
                )

            if (
                manifest.get("status")
                != "completed"
            ):
                return False

            counts = recorded_mm8_actions(
                human
            )

            summary[
                "recording_evidence"
            ] = {
                **counts,
                "path": str(human),
                "manifest": str(
                    manifest_path
                ),
                "eligible_for_import": (
                    manifest.get(
                        "eligible_for_import"
                    )
                ),
            }

            return (
                manifest.get(
                    "eligible_for_import"
                )
                is True
                and counts["rows"] > 0
                and counts[
                    "wheel_rows"
                ] > 0
                and counts[
                    "shape_rows"
                ] > 0
                and counts[
                    "manual_pan_rows"
                ] > 0
                and counts[
                    "manual_tilt_rows"
                ] > 0
                and len(
                    counts[
                        "manual_selected_modules"
                    ]
                )
                >= 2
            )

        wait(
            "recorded_effective_actions",
            (
                "Attendi la finalizzazione "
                "del dataset MM8."
            ),
            recording_written,
            timeout=30,
        )

        summary["passed"] = True

    except (
        Exception,
        KeyboardInterrupt,
    ) as error:
        summary["error"] = repr(error)

    finally:
        records.close()

        if observer is not None:
            observer.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()

        writers_stopped = False

        try:
            summary[
                "final_cleanup"
            ] = scoped_cleanup(ROOT)

            writers_stopped = True

        except Exception as error:
            summary["passed"] = False
            summary[
                "cleanup_error"
            ] = repr(error)

        try:
            summary.update(
                finalize_runtime(
                    runtime_dir,
                    output,
                    writers_stopped=(
                        writers_stopped
                    ),
                )
            )

        except Exception as error:
            summary["passed"] = False
            summary[
                "runtime_finalize_error"
            ] = repr(error)

            summary[
                "runtime_preserved"
            ] = str(runtime_dir)

        (
            output / "report.json"
        ).write_text(
            json.dumps(
                summary,
                indent=2,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )

    print(
        "T7_MM8_RESULT="
        + json.dumps(
            summary,
            allow_nan=False,
        ),
        flush=True,
    )

    print(
        f"REPORT={output / 'report.json'}",
        flush=True,
    )

    return (
        0
        if summary["passed"]
        else 1
    )

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
    )

    parser.add_argument(
        "--input-config",
        type=Path,
        default=(
            CONFIG
            / "smores_dualsense.yaml"
        ),
    )

    parser.add_argument(
        "--assembly-timeout",
        type=float,
        default=900.0,
    )

    parser.add_argument(
        "--device-id",
        type=int,
        default=0,
    )

    args = parser.parse_args()

    try:
        mapping = _validate_cli(args)

    except (OSError, ValueError) as error:
        print(
            f"T7 MM8 hardware preflight: {error}",
            file=sys.stderr,
        )
        return 2

    return run_native_acceptance(
        args,
        mapping,
    )


if __name__ == "__main__":
    raise SystemExit(main())
