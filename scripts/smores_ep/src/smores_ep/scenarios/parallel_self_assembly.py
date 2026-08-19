from __future__ import annotations

import math
import time
from typing import Any, Mapping

from smores_ep.config.simulation import SelfAssemblySimulationConfig
from smores_ep.control.teleop import SmoresCommand
from smores_ep.isaac.command_router import IsaacMultiModuleCommandRouter
from smores_ep.isaac.docking import IsaacDockingManager
from smores_ep.isaac.dynamic_stage import (
    ArticulationStateReader,
    DynamicDriveController,
    MechanismVisualController,
    configure_dynamic_stage,
)
from smores_ep.isaac.multi_module_stage import (
    clone_module,
    set_module_pose,
    use_collision_proxy_visuals,
)
from smores_ep.isaac.physics_asset import PHYSICS_ROOT
from smores_ep.isaac.primitive_executor import IsaacPrimitiveExecutor
from smores_ep.isaac.state_graph_publisher import SmoresStateGraphPublisher
from smores_ep.primitives.file_channel import (
    ActionFileChannel,
    PrimitiveFileChannel,
)
from smores_ep.primitives.pose_control import PoseControllerConfig


ASSEMBLY_ROOT_PREFIX = "/World/smores_ep_assembly"


def self_assembly_module_roots(
    module_ids: tuple[str, ...],
) -> dict[str, str]:
    """Map logical module IDs to one source and cloned USD roots."""

    if len(module_ids) < 2 or len(set(module_ids)) != len(module_ids):
        raise ValueError("Self-assembly needs at least two distinct module IDs")
    if any(not module_id.strip() for module_id in module_ids):
        raise ValueError("Self-assembly module IDs cannot be empty")
    return {
        module_id: (
            PHYSICS_ROOT
            if index == 0
            else f"{ASSEMBLY_ROOT_PREFIX}_{index + 1:02d}"
        )
        for index, module_id in enumerate(module_ids)
    }


def triangular_spawn_layout(
    config: SelfAssemblySimulationConfig,
) -> dict[str, tuple[float, float, float, float]]:
    """Return deterministic x, y, z and yaw poses around the swarm centroid."""

    left_id, center_id, right_id = config.module_ids
    return {
        left_id: (
            -config.spawn_half_width_m,
            config.outer_y_m,
            config.spawn_height_m,
            config.outer_yaw_deg,
        ),
        center_id: (
            0.0,
            config.center_y_m,
            config.spawn_height_m,
            0.0,
        ),
        right_id: (
            config.spawn_half_width_m,
            config.outer_y_m,
            config.spawn_height_m,
            -config.outer_yaw_deg,
        ),
    }


def radial_spawn_layout(
    config: SelfAssemblySimulationConfig,
) -> dict[str, tuple[float, float, float, float]]:
    """Place one root candidate at the centroid and all other modules on a ring."""

    center_id, *outer_ids = config.module_ids
    layout = {
        center_id: (0.0, 0.0, config.spawn_height_m, 0.0),
    }
    for index, module_id in enumerate(outer_ids):
        angle_rad = 2.0 * math.pi * index / len(outer_ids)
        layout[module_id] = (
            config.spawn_radius_m * math.cos(angle_rad),
            config.spawn_radius_m * math.sin(angle_rad),
            config.spawn_height_m,
            math.degrees(angle_rad) + 180.0,
        )
    return layout


def self_assembly_spawn_layout(
    config: SelfAssemblySimulationConfig,
) -> dict[str, tuple[float, float, float, float]]:
    """Keep the validated three-module layout and scale larger experiments."""

    if len(config.module_ids) == 3:
        return triangular_spawn_layout(config)
    return radial_spawn_layout(config)


def closest_module_to_centroid(
    layout: Mapping[str, tuple[float, float, float, float]],
) -> str:
    """Return the deterministic physical-root candidate for diagnostics."""

    if not layout:
        raise ValueError("Cannot calculate the centroid of an empty layout")
    centroid_x = sum(pose[0] for pose in layout.values()) / len(layout)
    centroid_y = sum(pose[1] for pose in layout.values()) / len(layout)
    return min(
        layout,
        key=lambda module_id: (
            math.hypot(
                layout[module_id][0] - centroid_x,
                layout[module_id][1] - centroid_y,
            ),
            module_id,
        ),
    )


def sparse_behavior_commands(
    commands: Mapping[str, SmoresCommand],
    module_ids: tuple[str, ...] | list[str] | set[str],
) -> dict[str, SmoresCommand]:
    """Validate a sparse command set without inventing wheel commands.

    A missing connected module keeps its shape through structural hold, while
    its free wheels remain towable. A selected locomotor instead receives an
    explicit wheel command and retains PAN/TILT through the action channel.
    """

    unknown = set(commands) - set(module_ids)
    if unknown:
        raise ValueError(
            "commands reference unknown modules: "
            + ", ".join(sorted(unknown))
        )
    return dict(commands)


def _world_position(
    stage: Any,
    prim_path: str,
) -> tuple[float, float, float]:
    from pxr import Usd, UsdGeom

    matrix = UsdGeom.Xformable(
        stage.GetPrimAtPath(prim_path)
    ).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    return tuple(float(value) for value in matrix.ExtractTranslation())


def _publish_primitive_statuses(
    primitive_channel: PrimitiveFileChannel,
    statuses: tuple[Any, ...],
    terminal_status_by_goal: dict[str, Any],
    now_s: float,
    physics_step: int,
    state_publish_interval: int,
    previous_serialized: str,
) -> str:
    """Publish concurrent executor feedback without flooding the file bridge."""

    if not statuses:
        return previous_serialized
    terminal_status = any(status.state.terminal for status in statuses)
    for status in statuses:
        if status.state.terminal:
            terminal_status_by_goal[status.goal_id] = status
    if not terminal_status and physics_step % state_publish_interval != 0:
        return previous_serialized
    merged_by_goal = dict(terminal_status_by_goal)
    merged_by_goal.update({status.goal_id: status for status in statuses})
    durable_statuses = tuple(
        merged_by_goal[goal_id]
        for goal_id in sorted(merged_by_goal)
    )
    serialized = "|".join(status.to_json() for status in durable_statuses)
    if serialized == previous_serialized:
        return previous_serialized
    primitive_channel.publish_many(durable_statuses, now_s)
    for status in statuses:
        if status.state.terminal:
            print(
                f"[primitive] {status.state.value.upper()} "
                f"{status.goal_id}: {status.code} {status.message}"
            )
    return serialized


def _advance_simulation(
    simulation_app: object,
    simulation_manager: Any,
    *,
    render: bool,
) -> None:
    """Advance one physics step and render only at the configured rate."""

    if render:
        simulation_app.update()  # type: ignore[attr-defined]
    else:
        simulation_manager.step(steps=1)


class _RealtimeRenderPacer:
    """Keep rendered simulation time close to wall time without catch-up bursts."""

    def __init__(self, render_hz: int) -> None:
        self._period_s = 1.0 / render_hz
        self._deadline_s = time.perf_counter()

    def wait(self) -> None:
        self._deadline_s += self._period_s
        now_s = time.perf_counter()
        delay_s = self._deadline_s - now_s
        if delay_s > 0.0:
            time.sleep(delay_s)
            return
        # If a frame was expensive, continue from the current wall time. Trying
        # to catch up would produce the visible burst/freeze cycle this mode is
        # intended to avoid.
        self._deadline_s = now_s


def run_parallel_self_assembly_scenario(
    config: SelfAssemblySimulationConfig,
    simulation_app: object,
) -> None:
    """Run fully dynamic modules controlled by the external ROS expert."""

    import isaacsim.core.experimental.utils.app as app_utils
    import isaacsim.core.experimental.utils.stage as stage_utils
    from isaacsim.core.rendering_manager import ViewportManager
    from isaacsim.core.simulation_manager import SimulationManager

    physics_path = config.physics_usd.resolve()
    if not physics_path.is_file():
        raise FileNotFoundError(f"Physics USD does not exist: {physics_path}")
    success, stage = stage_utils.open_stage(str(physics_path))
    if not success:
        raise RuntimeError(f"Could not open physics USD: {physics_path}")

    module_roots = self_assembly_module_roots(config.module_ids)
    for module_root in tuple(module_roots.values())[1:]:
        clone_module(stage, PHYSICS_ROOT, module_root)

    configure_dynamic_stage(
        stage,
        config.spawn_height_m,
        config.initial_pitch_deg,
        PHYSICS_ROOT,
    )
    obstacle_course = None
    if config.manual_obstacle_course:
        from smores_ep.isaac.obstacle_course import (
            install_manual_obstacle_course,
        )

        obstacle_course = install_manual_obstacle_course(stage)
    layout = self_assembly_spawn_layout(config)
    if obstacle_course is not None:
        layout = {
            module_id: (
                x_m - 2.0,
                y_m,
                z_m - 0.12,
                yaw_deg,
            )
            for module_id, (x_m, y_m, z_m, yaw_deg) in layout.items()
        }
    for module_id, module_root in module_roots.items():
        x_m, y_m, z_m, yaw_deg = layout[module_id]
        set_module_pose(
            stage,
            module_root,
            (x_m, y_m, z_m),
            pitch_deg=config.initial_pitch_deg,
            yaw_deg=yaw_deg,
        )

    if config.simple_visuals:
        use_collision_proxy_visuals(stage, module_roots)

    docking = IsaacDockingManager(stage, module_roots)
    mechanism_visuals = (
        {}
        if config.simple_visuals
        else {
            module_id: MechanismVisualController(
                stage,
                config.geometry.spur_to_pinion_ratio,
                module_root,
            )
            for module_id, module_root in module_roots.items()
        }
    )

    SimulationManager.set_physics_dt(1.0 / config.physics_hz)
    app_utils.play()
    simulation_app.update()

    states = {
        module_id: ArticulationStateReader(
            module_root,
            config.actuators,
            stage,
        )
        for module_id, module_root in module_roots.items()
    }
    drives = {
        module_id: DynamicDriveController(
            state,
            config.geometry,
            config.max_wheel_speed_rad_s,
        )
        for module_id, state in states.items()
    }
    command_router = IsaacMultiModuleCommandRouter(
        states,
        drives,
        docking,
    )
    primitive_channel = PrimitiveFileChannel(
        config.primitive_goal_file,
        config.primitive_cancel_file,
        config.primitive_status_file,
    )
    action_channel = ActionFileChannel(
        config.action_file,
        timeout_s=config.action_command_timeout_s,
    )
    primitive_executor = IsaacPrimitiveExecutor(
        stage,
        module_roots,
        states,
        docking,
        motion_module_ids=tuple(module_roots),
        geometry=config.geometry,
        staging_collision_avoidance=config.staging_collision_avoidance,
        staging_center_clearance_m=config.staging_center_clearance_m,
        staging_waypoint_margin_m=config.staging_waypoint_margin_m,
        pose_controller=PoseControllerConfig(
            max_linear_speed_m_s=(
                config.geometry.wheel_radius_m
                * config.max_wheel_speed_rad_s
            ),
        ),
    )
    state_graph_publisher = SmoresStateGraphPublisher(
        stage,
        module_roots,
        states,
        docking,
        output_dir=config.primitive_status_file.parent,
        geometry=config.geometry,
        actuator_profiles={
            module_id: config.actuators
            for module_id in module_roots
        },
        roles={
            module_id: {
                "current_role": "unassigned",
                "target_role": "",
                "role_confidence": 0.0,
                "role_source": "parallel_self_assembly_expert",
                "functional_role": {
                    "name": "unassigned",
                    "effective_dof_count": 4,
                    "responsibilities": [],
                },
            }
            for module_id in module_roots
        },
        include_contact_candidates=config.include_contact_candidates,
        course_observation=(
            obstacle_course.to_observation()
            if obstacle_course is not None
            else None
        ),
    )

    if not config.headless:
        camera_extent_m = max(
            0.52,
            config.spawn_radius_m * 1.8,
            2.4 if obstacle_course is not None else 0.0,
        )
        ViewportManager.set_camera_view(
            "/OmniverseKit_Persp",
            eye=[
                1.30 * camera_extent_m,
                -1.2 * camera_extent_m,
                max(0.46, 0.85 * camera_extent_m),
            ],
            target=(
                [1.25, 0.0, 0.08]
                if obstacle_course is not None
                else [0.0, 0.0, 0.03]
            ),
        )

    expected_root = closest_module_to_centroid(layout)
    print("SMORES-EP deterministic parallel self-assembly scenario")
    print(
        "modules: "
        + ", ".join(
            f"{module_id}={module_roots[module_id]}"
            for module_id in config.module_ids
        )
    )
    expected_connection_count = len(module_roots) - 1
    print(
        f"initial topology: {len(module_roots)} separated dynamic modules, "
        "0 connections"
    )
    print(
        f"expected physical root near centroid: {expected_root}; "
        "the target graph is selected by the external expert"
    )
    print(
        "ROS 2 is external to Isaac: start ros2_bridge/mssr_file_bridge.py "
        "and mssr_smores_self_assembly_node in separate terminals"
    )
    if obstacle_course is not None:
        print(
            "manual obstacle course: +X gap="
            f"{obstacle_course.gap_interval_x_m}, stairs="
            f"{obstacle_course.stair_top_heights_m}, button="
            f"{obstacle_course.button_center_xyz_m}, exit="
            f"{obstacle_course.exit_center_xyz_m}"
        )

    maximum_steps = config.steps if config.steps > 0 else None
    initial_step = SimulationManager.get_num_physics_steps()
    initial_time = SimulationManager.get_simulation_time()
    next_log_step = 0
    last_primitive_status = ""
    terminal_status_by_goal: dict[str, Any] = {}
    previous_connection_count = 0
    previous_behavior_commands: dict[str, SmoresCommand] = {}
    behavior_started_step: int | None = None
    last_behavior_diagnostic_step = 0
    # The deterministic expert does not need a graph traversal at physics
    # frequency. Keeping this configurable also prevents periodic file writes
    # from interrupting viewport presentation.
    state_publish_interval = max(
        1,
        config.physics_hz // config.state_publish_hz,
    )
    render_interval = max(1, config.physics_hz // config.render_hz)
    render_pacer = (
        _RealtimeRenderPacer(config.render_hz)
        if config.realtime_pacing and not config.headless
        else None
    )
    while simulation_app.is_running():
        physics_step = (
            SimulationManager.get_num_physics_steps()
            - initial_step
        )
        if maximum_steps is not None and physics_step >= maximum_steps:
            break

        now_s = SimulationManager.get_simulation_time()
        admission_statuses: list[Any] = []
        try:
            primitive_goal = primitive_channel.poll_goal()
            if primitive_goal is not None:
                accepted = primitive_executor.submit(
                    primitive_goal,
                    now_s,
                )
                admission_statuses.append(accepted)
                primitive_channel.publish(accepted)
                print(
                    f"[primitive] {accepted.state.value.upper()} "
                    f"{accepted.goal_id}: {accepted.message}"
                )
            cancel_goal_id = primitive_channel.poll_cancel()
            if cancel_goal_id is not None:
                canceled = primitive_executor.cancel(
                    cancel_goal_id,
                    now_s,
                )
                if canceled is not None:
                    admission_statuses.append(canceled)
                    primitive_channel.publish(canceled)
                    print(
                        f"[primitive] CANCELED {cancel_goal_id}: "
                        f"{canceled.message}"
                    )
        except (KeyError, TypeError, ValueError) as error:
            print(f"[primitive] REJECTED malformed payload: {error}")

        primitive_step = primitive_executor.step(now_s)
        last_primitive_status = _publish_primitive_statuses(
            primitive_channel,
            tuple(admission_statuses) + primitive_step.statuses,
            terminal_status_by_goal,
            now_s,
            physics_step,
            state_publish_interval,
            last_primitive_status,
        )
        # Keep the behavior baseline sparse. Missing connected modules retain
        # their shape but keep free wheels towable; selected train modules are
        # all explicitly actuated. Filling the mapping with synthetic zero
        # commands would turn non-locomotor wheels into active brakes.
        behavior_baseline: dict[str, SmoresCommand] = {}
        try:
            behavior_commands = action_channel.commands(time.monotonic())
            behavior_baseline = sparse_behavior_commands(
                behavior_commands,
                tuple(module_roots),
            )
        except (TypeError, ValueError) as error:
            print(f"[behavior] REJECTED malformed action payload: {error}")
        routed_commands = primitive_executor.compose_with_baseline(
            behavior_baseline,
            primitive_step.commands,
        )
        routed_rates = command_router.apply(routed_commands)
        if behavior_baseline != previous_behavior_commands:
            if behavior_baseline:
                behavior_started_step = physics_step
                last_behavior_diagnostic_step = physics_step
                target_text = ", ".join(
                    f"{module_id}:vx={command.linear_x_m_s:+.3f},"
                    f"yaw={command.angular_z_rad_s:+.3f},"
                    f"pan={command.pan_velocity_rad_s:+.3f}"
                    for module_id, command in sorted(
                        behavior_baseline.items()
                    )
                )
                wheel_text = ", ".join(
                    f"{module_id}=({rates[0]:+.2f},{rates[1]:+.2f})"
                    for module_id, rates in sorted(routed_rates.items())
                    if module_id in behavior_baseline
                )
                print(
                    f"[behavior] ACTIVE {target_text}; "
                    f"wheel_targets_rad_s={wheel_text}"
                )
            elif previous_behavior_commands:
                print(
                    "[behavior] STOPPED; locomotors returned to passive mode"
                )
                behavior_started_step = None
            previous_behavior_commands = dict(behavior_baseline)

        if (
            behavior_baseline
            and behavior_started_step is not None
            and physics_step - behavior_started_step >= config.physics_hz // 2
            and physics_step - last_behavior_diagnostic_step
            >= config.physics_hz
        ):
            last_behavior_diagnostic_step = physics_step
            actual_rates: dict[str, tuple[float, float]] = {}
            for module_id in behavior_baseline:
                joint_state = states[module_id].read()
                actual_rates[module_id] = (
                    joint_state.left_wheel_rad_s,
                    joint_state.right_wheel_rad_s,
                )
            actual_text = ", ".join(
                f"{module_id}=({rates[0]:+.2f},{rates[1]:+.2f})"
                for module_id, rates in sorted(actual_rates.items())
            )
            target_is_nonzero = any(
                abs(rate) >= 0.05
                for module_id, rates in routed_rates.items()
                if module_id in behavior_baseline
                for rate in rates
            )
            actual_is_stalled = all(
                abs(rate) < 0.05
                for rates in actual_rates.values()
                for rate in rates
            )
            label = " STALLED" if target_is_nonzero and actual_is_stalled else ""
            print(f"[behavior]{label} wheel_actual_rad_s={actual_text}")

        render_due = physics_step % render_interval == 0
        if render_due:
            # The decorative CAD gears follow the physical joints, but they
            # need USD transform updates only when a frame will be rendered.
            # Keeping them out of the other physics substeps materially
            # reduces GUI load for multi-module scenes.
            joint_states = {
                module_id: reader.read()
                for module_id, reader in states.items()
            }
            for module_id, visual in mechanism_visuals.items():
                visual.update(joint_states[module_id])
            if render_pacer is not None:
                render_pacer.wait()

        _advance_simulation(
            simulation_app,
            SimulationManager,
            render=render_due,
        )

        physics_step = (
            SimulationManager.get_num_physics_steps()
            - initial_step
        )
        if physics_step % state_publish_interval == 0:
            state_graph_publisher.publish(
                SimulationManager.get_simulation_time(),
                experiment_profile=(
                    "deterministic_parallel_self_assembly"
                ),
            )

        connection_count = len(docking.connections)
        if connection_count != previous_connection_count:
            previous_connection_count = connection_count
            print(
                "[assembly] rigid connections: "
                f"{connection_count}/{expected_connection_count}"
            )
            if connection_count == expected_connection_count:
                print(
                    "[assembly] TARGET TOPOLOGY REACHED: "
                    f"connected {len(module_roots)}-module tree"
                )

        if config.log_interval and physics_step >= next_log_step:
            positions = {
                module_id: _world_position(
                    stage,
                    f"{module_root}/body_link",
                )
                for module_id, module_root in module_roots.items()
            }
            moving = {
                module_id: tuple(
                    round(rate, 2)
                    for rate in routed_rates.get(
                        module_id,
                        (0.0, 0.0),
                    )
                )
                for module_id in module_roots
            }
            elapsed = (
                SimulationManager.get_simulation_time()
                - initial_time
            )
            position_text = " ".join(
                f"{module_id}=({position[0]:+.3f},"
                f"{position[1]:+.3f},{position[2]:+.3f})"
                for module_id, position in positions.items()
            )
            print(
                f"t={elapsed:7.3f}s {position_text} "
                f"wheel_cmd={moving} "
                f"connections={connection_count}"
            )
            next_log_step += config.log_interval

    print(f"final_connections={len(docking.connections)}")
    app_utils.stop()
