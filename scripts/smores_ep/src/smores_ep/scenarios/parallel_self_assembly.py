from __future__ import annotations

import json
import math
import time
from pathlib import Path
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
from smores_ep.isaac.teleop_runtime import TeleopRuntimeBridge
from smores_ep.primitives.file_channel import (
    ActionFileChannel,
    PrimitiveFileChannel,
)
from smores_ep.primitives.model import PrimitiveState, PrimitiveStatus
from smores_ep.primitives.pose_control import PoseControllerConfig


ASSEMBLY_ROOT_PREFIX = "/World/smores_ep_assembly"




def _write_rc_car_dynamic_obstacles(
    stage: Any,
    output_path: Path,
    now_s: float,
) -> None:
    """Export the CURRENT physical cone poses from the live USD Stage.

    The ROS/Nav2 route node runs in another process and cannot inspect the
    Isaac Stage directly.  This small atomic file is therefore the bridge
    between simulation truth and the live OccupancyGrid.

    The collision child is used as the geometric truth.  Moving the ConeXX
    parent in Isaac moves both visual and collider and is therefore reflected
    here automatically.
    """
    import json
    from pxr import UsdGeom

    root = stage.GetPrimAtPath(
        "/World/RCPlanarTestCourse/NavigationCones"
    )
    if not root or not root.IsValid():
        return

    cache = UsdGeom.XformCache()
    cones = []

    for cone in root.GetChildren():
        name = cone.GetName()
        if "cone" not in name.lower():
            continue

        # Use the collider as ground truth whenever present.
        probe = cone
        for child in cone.GetChildren():
            if "collision" in child.GetName().lower():
                probe = child
                break

        try:
            transform = cache.GetLocalToWorldTransform(probe)
            translation = transform.ExtractTranslation()
            x_m = float(translation[0])
            y_m = float(translation[1])
        except Exception:
            continue

        cones.append({
            "name": name,
            "x_m": x_m,
            "y_m": y_m,
        })

    if not cones:
        return

    cones.sort(key=lambda item: item["name"])

    payload = {
        "schema_version": "mssr.rc_car_dynamic_obstacles.v1",
        "simulation_time_s": float(now_s),
        "cones": cones,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True))
    temporary.replace(output_path)


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


def prepare_initial_manual_teleop_handoff(
    behavior_commands: Mapping[str, SmoresCommand],
    diagnostics: Any,
    drives: Mapping[str, DynamicDriveController],
    *,
    behavior_source_seen: bool,
) -> bool:
    """Capture posture before the first direct-assembly manual teleop HOLD.

    This remains deliberately narrower than a generic HOLD transition.

    If Snake8 or MobileManipulator8 manual teleoperation is the first
    operational behavior source in the current Isaac session, the robot came
    directly from self-assembly.  Structural assembly has the correct physical
    posture, while DynamicDriveController may still contain startup PAN/TILT
    targets.

    Capture the physically reached posture exactly once before that first
    manual teleop behavior packet is applied.

    If any operational behavior has already been seen, this is not the
    direct-assembly first-controller case and no recapture is performed.
    """

    if not behavior_commands:
        return behavior_source_seen

    if (
        not behavior_source_seen
        and diagnostics.phase in {
            "snake8_teleop",
            "mobile_manipulator8_teleop",
        }
    ):
        unknown = set(behavior_commands) - set(drives)
        if unknown:
            raise ValueError(
                "Manual teleop handoff references unknown drives: "
                + ", ".join(sorted(unknown))
            )

        for module_id in behavior_commands:
            drives[module_id].initialize_from_measured_posture()

        print(
            "[behavior] DIRECT-ASSEMBLY -> MANUAL TELEOP HANDOFF "
            f"({diagnostics.phase}): captured measured PAN/TILT for "
            + ", ".join(sorted(behavior_commands))
        )

    # From now on this is no longer a direct-assembly first-controller case.
    return True


def invalidate_module_command_sources(
    module_ids: tuple[str, ...], *, action_channel: ActionFileChannel,
    held_primitive_commands: dict[str, SmoresCommand],
) -> None:
    """Quarantine stale behavior/primitive commands at an ownership handoff."""
    action_channel.invalidate_modules(module_ids)
    for module_id in module_ids:
        held_primitive_commands.pop(module_id, None)


def apply_free_module_reset(
    module_ids: tuple[str, ...], *, action_channel: ActionFileChannel,
    held_primitive_commands: dict[str, SmoresCommand],
    command_router: IsaacMultiModuleCommandRouter,
) -> None:
    """Atomically drop stale command sources before resetting free drives."""
    invalidate_module_command_sources(
        module_ids,
        action_channel=action_channel,
        held_primitive_commands=held_primitive_commands,
    )
    command_router.reset_free_modules(module_ids)


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
    """Pace simulation time against wall time without catch-up bursts."""

    def __init__(self, render_hz: int, speed_factor: float = 1.0) -> None:
        self._period_s = 1.0 / (render_hz * speed_factor)
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


def _service_teleop_runtime(runtime) -> dict:
    """Poll teleop runtime without stopping simulation or physics progress."""
    return runtime.poll()


def _apply_structure_stop(
    active: bool,
    command_router,
    primitive_executor,
    action_channel,
    module_ids: tuple[str, ...],
    now_s: float,
    *,
    held_primitive_commands: dict[str, SmoresCommand] | None = None,
) -> tuple:
    """Apply structure E-STOP and terminate in-flight primitive authority.

    Entering E-STOP first closes the final actuator boundary, then quarantines
    any already-published behavior command and cancels every active primitive.
    Clearing E-STOP only releases the router latch; nothing resumes
    automatically.
    """
    command_router.set_emergency_stop(active)

    if not active:
        return ()

    if held_primitive_commands is None:
        # Compatibility for isolated/unit callers that do not own the
        # production held-command cache.
        action_channel.invalidate_modules(module_ids)
    else:
        invalidate_module_command_sources(
            module_ids,
            action_channel=action_channel,
            held_primitive_commands=held_primitive_commands,
        )

    statuses = []
    for goal in tuple(primitive_executor.active_goals):
        status = primitive_executor.cancel(goal.goal_id, now_s)
        if status is not None:
            statuses.append(status)

    return tuple(statuses)


def _admit_primitive_goal(
    primitive_executor,
    goal,
    now_s: float,
    *,
    structure_stopped: bool,
):
    """Reject new primitive authority while structure E-STOP is latched."""
    if not structure_stopped:
        return primitive_executor.submit(goal, now_s)

    return PrimitiveStatus(
        goal_id=goal.goal_id,
        primitive=goal.primitive,
        state=PrimitiveState.REJECTED,
        stamp_s=now_s,
        module_ids=goal.module_ids,
        phase="estop",
        progress=0.0,
        code="ESTOP_ACTIVE",
        message="primitive goal rejected while structure E-STOP is active",
    )


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
    if config.composite_mission_path is not None:
        from smores_ep.isaac.obstacle_course import (
            install_composite_obstacle_course,
        )

        mission = json.loads(
            config.composite_mission_path.read_text(encoding="utf-8")
        )
        seed_catalog = json.loads(
            config.composite_seed_catalog_path.read_text(encoding="utf-8")
        )
        if seed_catalog.get("schema_version") != "mssr.composite_seed_catalog.v1":
            raise ValueError("Unsupported composite seed catalog schema")
        raw_validated = seed_catalog.get("validated_seeds", {})
        if not isinstance(raw_validated, Mapping):
            raise ValueError("Composite seed catalog is malformed")
        obstacle_course = install_composite_obstacle_course(
            stage,
            mission,
            {
                str(task_type): frozenset(int(seed) for seed in seeds)
                for task_type, seeds in raw_validated.items()
            },
        )
    elif config.manual_obstacle_course:
        from smores_ep.isaac.obstacle_course import (
            install_manual_obstacle_course,
        )

        obstacle_course = install_manual_obstacle_course(stage)
    elif config.stair_test_course:
        from smores_ep.isaac.obstacle_course import (
            UniformStairSpec,
            install_snake8_stair_test_course,
        )

        obstacle_course = install_snake8_stair_test_course(
            stage,
            UniformStairSpec(
                rise_m=config.stair_rise_m,
                tread_depth_m=config.stair_depth_m,
                step_count=config.stair_count,
                first_riser_x_m=config.stair_first_riser_x_m,
                seed=config.stair_seed,
            ),
        )
    elif config.button_test_course:
        from smores_ep.isaac.obstacle_course import (
            install_mobile_manipulator_button_test_course,
            sample_button_target_spec,
        )

        button_spec = (
            sample_button_target_spec(config.button_seed)
            if config.button_seed is not None
            else None
        )

        obstacle_course = (
            install_mobile_manipulator_button_test_course(
                stage,
                button_spec,
            )
        )
    elif config.gap_test_course:
        from smores_ep.isaac.obstacle_course import (
            CoplanarGapSpec,
            install_snake8_gap_test_course,
        )

        obstacle_course = install_snake8_gap_test_course(
            stage,
            CoplanarGapSpec(
                width_m=config.gap_width_m,
                near_edge_x_m=config.gap_near_edge_x_m,
                seed=config.gap_seed,
            ),
        )
    elif config.rc_car_planar_test_course:
        from smores_ep.isaac.obstacle_course import (
            install_rc_car_planar_test_course,
            sample_rc_car_planar_spec,
        )
        obstacle_course = install_rc_car_planar_test_course(
            stage,
            sample_rc_car_planar_spec(
                0 if config.rc_car_seed is None else config.rc_car_seed
            ),
        )
    layout = self_assembly_spawn_layout(config)
    if config.composite_mission_path is not None:
        layout = {
            module_id: (x_m - 1.50, y_m, z_m, yaw_deg)
            for module_id, (x_m, y_m, z_m, yaw_deg) in layout.items()
        }
    elif config.manual_obstacle_course:
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
        if (
            config.composite_mission_path is not None
            and obstacle_course is not None
        ):
            # Start close to the assembly area instead of framing the whole
            # long composite course from several metres away.
            start_box = next(
                (
                    box
                    for box in obstacle_course.boxes
                    if box.semantic == "composite_start_platform"
                ),
                None,
            )

            if start_box is not None:
                start_x, start_y, _ = start_box.center_xyz_m
            else:
                start_x, start_y = -1.50, 0.0

            camera_target = [
                float(start_x),
                float(start_y),
                0.12,
            ]
            camera_eye = [
                float(start_x) + 1.90,
                float(start_y) - 2.00,
                1.55,
            ]

        else:
            course_extent_m = (
                2.4
                if config.manual_obstacle_course
                else 1.8
                if config.stair_test_course
                else 2.80
                if config.rc_car_planar_test_course
                else 1.25
                if config.button_test_course or config.gap_test_course
                else 0.0
            )
            camera_extent_m = max(
                0.52,
                config.spawn_radius_m * 1.8,
                course_extent_m,
            )
            camera_eye = [
                1.30 * camera_extent_m,
                -1.2 * camera_extent_m,
                max(0.46, 0.85 * camera_extent_m),
            ]
            camera_target = (
                [1.25, 0.0, 0.08]
                if config.manual_obstacle_course
                else [1.0, 0.0, 0.10]
                if config.stair_test_course
                else [0.45, 0.0, 0.08]
                if config.button_test_course or config.gap_test_course
                else [0.0, 0.0, 0.03]
            )

        ViewportManager.set_camera_view(
            "/OmniverseKit_Persp",
            eye=camera_eye,
            target=camera_target,
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
        course_observation = obstacle_course.to_observation()
        print(
            "test course: "
            f"{course_observation.get('course_profile', 'manual')} "
            f"scenario={course_observation.get('scenario', {})}"
        )
    maximum_steps = config.steps if config.steps > 0 else None
    initial_step = SimulationManager.get_num_physics_steps()
    initial_time = SimulationManager.get_simulation_time()
    next_log_step = 0
    last_primitive_status = ""
    terminal_status_by_goal: dict[str, Any] = {}

    # PhysX needs the validated 240 Hz integration rate, but the geometric
    # primitive controller does not. ALIGN_FACES in particular evaluates
    # multiple USD world transforms and collision-aware staging geometry.
    # Run that control work at 60 Hz and hold the latest actuator command
    # between controller ticks.
    primitive_control_hz = min(60, config.physics_hz)
    primitive_control_interval = max(
        1,
        config.physics_hz // primitive_control_hz,
    )
    held_primitive_commands: dict[str, SmoresCommand] = {}

    def invalidate_command_sources(module_ids: tuple[str, ...]) -> None:
        invalidate_module_command_sources(
            module_ids,
            action_channel=action_channel,
            held_primitive_commands=held_primitive_commands,
        )

    def reset_free_modules(module_ids: tuple[str, ...]) -> None:
        apply_free_module_reset(
            module_ids,
            action_channel=action_channel,
            held_primitive_commands=held_primitive_commands,
            command_router=command_router,
        )

    primitive_executor.invalidate_module_command_sources_callback = (
        invalidate_command_sources
    )
    primitive_executor.reset_free_modules_callback = reset_free_modules

    previous_connection_count = 0
    previous_behavior_commands: dict[str, SmoresCommand] = {}
    behavior_source_seen = False
    behavior_started_step: int | None = None
    last_behavior_diagnostic_step = 0
    previous_pan_traction_module_ids: frozenset[str] = frozenset()
    # The deterministic expert does not need a graph traversal at physics
    # frequency. Keeping this configurable also prevents periodic file writes
    # from interrupting viewport presentation.
    state_publish_interval = max(
        1,
        config.physics_hz // config.state_publish_hz,
    )
    render_interval = max(1, config.physics_hz // config.render_hz)
    dynamic_obstacle_interval = max(1, config.physics_hz // 5)
    dynamic_obstacle_path = Path(config.action_file).with_name(
        "rc_car_dynamic_obstacles.json"
    )
    render_pacer = (
        _RealtimeRenderPacer(
            config.render_hz,
            config.simulation_speed_factor,
        )
        if config.realtime_pacing
        else None
    )
    def camera_center():
        from pxr import UsdGeom
        cache = UsdGeom.XformCache()
        positions = [cache.GetLocalToWorldTransform(stage.GetPrimAtPath(f"{root}/body_link")).ExtractTranslation()
                     for root in module_roots.values()]
        return tuple(sum(float(position[index]) for position in positions) / len(positions) for index in range(3))

    def handle_structure_stop(active: bool) -> None:
        nonlocal last_primitive_status

        now_s = SimulationManager.get_simulation_time()
        statuses = _apply_structure_stop(
            active,
            command_router,
            primitive_executor,
            action_channel,
            tuple(module_roots),
            now_s,
            held_primitive_commands=held_primitive_commands,
        )

        if statuses:
            current_step = (
                SimulationManager.get_num_physics_steps()
                - initial_step
            )
            last_primitive_status = _publish_primitive_statuses(
                primitive_channel,
                statuses,
                terminal_status_by_goal,
                now_s,
                current_step,
                state_publish_interval,
                last_primitive_status,
            )

    runtime = TeleopRuntimeBridge(
        Path(config.action_file).with_name("smores_teleop_runtime_request.json"),
        config.primitive_status_file.with_name("smores_teleop_runtime_status.json"),
        structure_stop_callback=handle_structure_stop,
        camera_callback=(None if config.headless else
            lambda eye, target: ViewportManager.set_camera_view("/OmniverseKit_Persp", eye=list(eye), target=list(target))),
        center_callback=camera_center)
    while simulation_app.is_running():
        runtime_status = _service_teleop_runtime(runtime)
        physics_step = (
            SimulationManager.get_num_physics_steps()
            - initial_step
        )
        if maximum_steps is not None and physics_step >= maximum_steps:
            break

        now_s = SimulationManager.get_simulation_time()
        if (
            config.rc_car_planar_test_course
            and physics_step % dynamic_obstacle_interval == 0
        ):
            _write_rc_car_dynamic_obstacles(
                stage,
                dynamic_obstacle_path,
                now_s,
            )
        admission_statuses: list[Any] = []
        try:
            primitive_goal = primitive_channel.poll_goal()
            if primitive_goal is not None:
                accepted = _admit_primitive_goal(
                    primitive_executor,
                    primitive_goal,
                    now_s,
                    structure_stopped=bool(
                        runtime_status["structure_stopped"]
                    ),
                )
                admission_statuses.append(accepted)
                primitive_channel.publish(accepted)
                print(
                    f"[primitive] {accepted.state.value.upper()} "
                    f"{accepted.goal_id}: {accepted.message}"
                )
            for cancel_goal_id in primitive_channel.poll_cancels():
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

        primitive_statuses: tuple[Any, ...] = ()
        if physics_step % primitive_control_interval == 0:
            primitive_step = primitive_executor.step(now_s)
            held_primitive_commands = dict(primitive_step.commands)
            primitive_statuses = primitive_step.statuses

        last_primitive_status = _publish_primitive_statuses(
            primitive_channel,
            tuple(admission_statuses) + primitive_statuses,
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
        diagnostics = action_channel.diagnostics

        behavior_source_seen = prepare_initial_manual_teleop_handoff(
            behavior_baseline,
            diagnostics,
            drives,
            behavior_source_seen=behavior_source_seen,
        )

        pan_traction_module_ids = frozenset(
            diagnostics.pan_traction_module_ids
        )

        unknown_pan_traction = (
            set(pan_traction_module_ids) - set(states)
        )
        if unknown_pan_traction:
            raise ValueError(
                "PAN traction references unknown module(s): "
                + ", ".join(sorted(unknown_pan_traction))
            )

        if (
            pan_traction_module_ids
            != previous_pan_traction_module_ids
        ):
            for module_id, state in states.items():
                state.set_pan_contact_mode(
                    "wheel"
                    if module_id in pan_traction_module_ids
                    else "pan_face"
                )

            if pan_traction_module_ids:
                print(
                    "[contact] RC_CAR8 PAN TRACTION ON: "
                    + ", ".join(
                        sorted(pan_traction_module_ids)
                    )
                )
            elif previous_pan_traction_module_ids:
                print(
                    "[contact] RC_CAR8 PAN TRACTION OFF; "
                    "restored pan_face material"
                )

            previous_pan_traction_module_ids = (
                pan_traction_module_ids
            )

        routed_commands = primitive_executor.compose_with_baseline(
            behavior_baseline,
            held_primitive_commands,
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
            pan_moving = {
                module_id: round(
                    routed_commands.get(
                        module_id,
                        SmoresCommand(),
                    ).pan_velocity_rad_s,
                    3,
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
            pan02 = routed_commands.get(
                "smores_02",
                SmoresCommand(),
            )
            pan02_mode = getattr(
                pan02.internal_motion,
                "value",
                str(pan02.internal_motion),
            )
            pan02_target = (
                None
                if pan02.pan_target_rad is None
                else round(float(pan02.pan_target_rad), 4)
            )
            pan02_velocity = round(
                float(pan02.pan_velocity_rad_s),
                4,
            )

            print(
                f"t={elapsed:7.3f}s {position_text} "
                f"wheel_cmd={moving} "
                f"pan_cmd={pan_moving} "
                f"PAN02=(mode={pan02_mode},"
                f"target={pan02_target},"
                f"vel={pan02_velocity}) "
                f"connections={connection_count}"
            )
            next_log_step += config.log_interval

    print(f"final_connections={len(docking.connections)}")
    app_utils.stop()
