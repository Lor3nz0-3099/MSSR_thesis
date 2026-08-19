from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path

from smores_ep.config.geometry import SmoresGeometry
from smores_ep.config.physics import (
    SMORES_EP_MAX_WHEEL_SPEED_RAD_S,
    SmoresActuatorConfig,
)


@dataclass(frozen=True)
class KinematicSimulationConfig:
    visual_usd: Path
    headless: bool = False
    ros2_enabled: bool = True
    demo_enabled: bool = False
    steps: int = 0
    update_hz: int = 120
    log_interval: int = 120
    cmd_vel_topic: str = "/cmd_vel"
    pan_topic: str = "/smores_ep/pan_angle"
    pan_delta_topic: str = "/smores_ep/pan_delta"
    tilt_topic: str = "/smores_ep/tilt_angle"
    command_timeout_s: float = 0.5
    max_pan_speed_rad_s: float = 2.0
    max_tilt_speed_rad_s: float = 1.25
    geometry: SmoresGeometry = field(default_factory=SmoresGeometry)

    def __post_init__(self) -> None:
        if self.steps < 0:
            raise ValueError("Step count cannot be negative")
        if self.headless and self.steps == 0:
            raise ValueError("Headless runs require a finite positive --steps")
        if self.update_hz <= 0:
            raise ValueError("Update frequency must be positive")
        if self.log_interval < 0:
            raise ValueError("Log interval cannot be negative")
        numeric = (
            self.command_timeout_s,
            self.max_pan_speed_rad_s,
            self.max_tilt_speed_rad_s,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in numeric):
            raise ValueError("Timeout and actuator speeds must be positive")
        if not all(
            topic.startswith("/")
            for topic in (
                self.cmd_vel_topic,
                self.pan_topic,
                self.pan_delta_topic,
                self.tilt_topic,
            )
        ):
            raise ValueError("ROS 2 topics must be absolute")


@dataclass(frozen=True)
class DynamicSimulationConfig:
    physics_usd: Path
    headless: bool = False
    ros2_enabled: bool = True
    demo_enabled: bool = False
    steps: int = 0
    physics_hz: int = 240
    render_hz: int = 60
    log_interval: int = 240
    cmd_vel_topic: str = "/cmd_vel"
    pan_topic: str = "/smores_ep/pan_angle"
    pan_delta_topic: str = "/smores_ep/pan_delta"
    tilt_topic: str = "/smores_ep/tilt_angle"
    # Lowest transformed CAD wheel point is at about -31.41 mm in the body
    # frame. The small clearance avoids an initial penetration without dropping
    # the module enough to inject visible impact energy.
    spawn_height_m: float = 0.0316
    initial_pitch_deg: float = -0.25
    # 1.1 body lengths/s applied to the 80 mm SMORES-EP characteristic
    # length, then converted through the measured CAD wheel radius.
    max_wheel_speed_rad_s: float = SMORES_EP_MAX_WHEEL_SPEED_RAD_S
    geometry: SmoresGeometry = field(default_factory=SmoresGeometry)

    def __post_init__(self) -> None:
        if self.steps < 0:
            raise ValueError("Step count cannot be negative")
        if self.headless and self.steps == 0:
            raise ValueError("Headless runs require a finite positive --steps")
        if self.physics_hz <= 0 or self.render_hz <= 0:
            raise ValueError("Simulation frequencies must be positive")
        if self.physics_hz % self.render_hz != 0:
            raise ValueError(
                "Physics frequency must be divisible by render frequency"
            )
        if self.log_interval < 0:
            raise ValueError("Log interval cannot be negative")
        numeric = (
            self.spawn_height_m,
            self.initial_pitch_deg,
            self.max_wheel_speed_rad_s,
        )
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("Dynamic simulation values must be finite")
        if self.spawn_height_m <= 0.0 or self.max_wheel_speed_rad_s <= 0.0:
            raise ValueError(
                "Spawn height and maximum wheel speed must be positive"
            )
        if not all(
            topic.startswith("/")
            for topic in (
                self.cmd_vel_topic,
                self.pan_topic,
                self.pan_delta_topic,
                self.tilt_topic,
            )
        ):
            raise ValueError("ROS 2 topics must be absolute")


@dataclass(frozen=True)
class DockingSimulationConfig:
    """Two-module demonstration using the general docking manager."""

    physics_usd: Path
    headless: bool = False
    ros2_enabled: bool = True
    steps: int = 0
    physics_hz: int = 240
    render_hz: int = 60
    log_interval: int = 240
    cmd_vel_topic: str = "/cmd_vel"
    pan_topic: str = "/smores_ep/pan_angle"
    pan_delta_topic: str = "/smores_ep/pan_delta"
    tilt_topic: str = "/smores_ep/tilt_angle"
    docking_command_topic: str = "/smores_ep/docking_command"
    primitive_goal_file: Path = Path("configs/smores_primitive_goal.json")
    primitive_cancel_file: Path = Path("configs/smores_primitive_cancel.json")
    primitive_status_file: Path = Path(
        "logs/bridge/smores_primitive_status.json"
    )
    active_module_id: str = "active"
    passive_module_id: str = "passive"
    initial_active_face: str = "TOP"
    initial_passive_face: str = "LEFT"
    initial_face_gap_m: float = 0.012
    spawn_height_m: float = 0.0316
    initial_pitch_deg: float = 0.0
    passive_yaw_deg: float = 90.0
    max_wheel_speed_rad_s: float = SMORES_EP_MAX_WHEEL_SPEED_RAD_S
    anchor_active_on_attach: bool = True
    active_actuators: SmoresActuatorConfig = field(
        default_factory=SmoresActuatorConfig.payload_overdrive
    )
    geometry: SmoresGeometry = field(default_factory=SmoresGeometry)

    def __post_init__(self) -> None:
        if self.steps < 0:
            raise ValueError("Step count cannot be negative")
        if self.headless and self.steps == 0:
            raise ValueError("Headless runs require finite positive --steps")
        if self.physics_hz <= 0 or self.render_hz <= 0:
            raise ValueError("Simulation frequencies must be positive")
        if self.physics_hz % self.render_hz != 0:
            raise ValueError(
                "Physics frequency must be divisible by render frequency"
            )
        if self.log_interval < 0:
            raise ValueError("Log interval cannot be negative")
        numeric = (
            self.initial_face_gap_m,
            self.spawn_height_m,
            self.initial_pitch_deg,
            self.passive_yaw_deg,
            self.max_wheel_speed_rad_s,
        )
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("Docking simulation values must be finite")
        if (
            self.initial_face_gap_m < 0.0
            or self.spawn_height_m <= 0.0
            or self.max_wheel_speed_rad_s <= 0.0
        ):
            raise ValueError("Docking distances and speeds are invalid")
        if (
            not self.active_module_id
            or not self.passive_module_id
            or self.active_module_id == self.passive_module_id
        ):
            raise ValueError("Active and passive module IDs must be distinct")
        if any(
            not str(path)
            for path in (
                self.primitive_goal_file,
                self.primitive_cancel_file,
                self.primitive_status_file,
            )
        ):
            raise ValueError("Primitive channel paths cannot be empty")
        valid_faces = {"LEFT", "RIGHT", "TOP", "BOTTOM"}
        if (
            self.initial_active_face.upper() not in valid_faces
            or self.initial_passive_face.upper() not in valid_faces
        ):
            raise ValueError(
                "Initial docking faces must be LEFT, RIGHT, TOP or BOTTOM"
            )
        if not all(
            topic.startswith("/")
            for topic in (
                self.cmd_vel_topic,
                self.pan_topic,
                self.pan_delta_topic,
                self.tilt_topic,
                self.docking_command_topic,
            )
        ):
            raise ValueError("ROS 2 topics must be absolute")


@dataclass(frozen=True)
class MultiModuleLiftSimulationConfig:
    """One teleoperated module docking to a pre-connected module chain."""

    physics_usd: Path
    headless: bool = False
    ros2_enabled: bool = True
    steps: int = 0
    physics_hz: int = 240
    render_hz: int = 60
    log_interval: int = 240
    cmd_vel_topic: str = "/cmd_vel"
    pan_topic: str = "/smores_ep/pan_angle"
    pan_delta_topic: str = "/smores_ep/pan_delta"
    tilt_topic: str = "/smores_ep/tilt_angle"
    docking_command_topic: str = "/smores_ep/docking_command"
    primitive_goal_file: Path = Path("configs/smores_primitive_goal.json")
    primitive_cancel_file: Path = Path("configs/smores_primitive_cancel.json")
    primitive_status_file: Path = Path(
        "logs/bridge/smores_primitive_status.json"
    )
    active_module_id: str = "active"
    chain_module_prefix: str = "chain"
    chain_module_count: int = 5
    active_to_chain_gap_m: float = 0.0
    spawn_height_m: float = 0.0316
    initial_pitch_deg: float = 0.0
    max_wheel_speed_rad_s: float = SMORES_EP_MAX_WHEEL_SPEED_RAD_S
    anchor_active_on_attach: bool = True
    active_actuators: SmoresActuatorConfig = field(
        default_factory=lambda: SmoresActuatorConfig.payload_overdrive(6.0)
    )
    geometry: SmoresGeometry = field(default_factory=SmoresGeometry)

    def __post_init__(self) -> None:
        if self.steps < 0:
            raise ValueError("Step count cannot be negative")
        if self.headless and self.steps == 0:
            raise ValueError("Headless runs require finite positive --steps")
        if self.physics_hz <= 0 or self.render_hz <= 0:
            raise ValueError("Simulation frequencies must be positive")
        if self.physics_hz % self.render_hz != 0:
            raise ValueError(
                "Physics frequency must be divisible by render frequency"
            )
        if self.log_interval < 0:
            raise ValueError("Log interval cannot be negative")
        if not isinstance(self.chain_module_count, int):
            raise TypeError("Chain module count must be an integer")
        if self.chain_module_count < 1:
            raise ValueError(
                "The pre-connected chain needs at least one module"
            )
        numeric = (
            self.active_to_chain_gap_m,
            self.spawn_height_m,
            self.initial_pitch_deg,
            self.max_wheel_speed_rad_s,
        )
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("Multi-module simulation values must be finite")
        if (
            self.active_to_chain_gap_m < 0.0
            or self.spawn_height_m <= 0.0
            or self.max_wheel_speed_rad_s <= 0.0
        ):
            raise ValueError("Multi-module distances and speeds are invalid")
        if not self.active_module_id or not self.chain_module_prefix:
            raise ValueError("Module ID and chain prefix cannot be empty")
        if any(
            not str(path)
            for path in (
                self.primitive_goal_file,
                self.primitive_cancel_file,
                self.primitive_status_file,
            )
        ):
            raise ValueError("Primitive channel paths cannot be empty")
        if self.active_module_id.startswith(
            f"{self.chain_module_prefix}_"
        ):
            raise ValueError("Active module ID collides with the chain prefix")
        if not all(
            topic.startswith("/")
            for topic in (
                self.cmd_vel_topic,
                self.pan_topic,
                self.pan_delta_topic,
                self.tilt_topic,
                self.docking_command_topic,
            )
        ):
            raise ValueError("ROS 2 topics must be absolute")


@dataclass(frozen=True)
class SelfAssemblySimulationConfig:
    """Initially separated modules controlled through primitive files."""

    physics_usd: Path
    headless: bool = False
    steps: int = 0
    physics_hz: int = 240
    render_hz: int = 20
    state_publish_hz: int = 10
    log_interval: int = 240
    simple_visuals: bool = False
    realtime_pacing: bool = False
    include_contact_candidates: bool = True
    primitive_goal_file: Path = Path("configs/smores_primitive_goal.json")
    primitive_cancel_file: Path = Path("configs/smores_primitive_cancel.json")
    primitive_status_file: Path = Path(
        "logs/bridge/smores_primitive_status.json"
    )
    action_file: Path = Path("configs/actions.json")
    action_command_timeout_s: float = 0.5
    module_ids: tuple[str, ...] = (
        "smores_01",
        "smores_02",
        "smores_03",
    )
    spawn_height_m: float = 0.0316
    initial_pitch_deg: float = 0.0
    spawn_half_width_m: float = 0.18
    outer_y_m: float = -0.10
    center_y_m: float = 0.14
    outer_yaw_deg: float = 25.0
    spawn_radius_m: float = 0.34
    manual_obstacle_course: bool = False
    staging_collision_avoidance: bool = True
    staging_center_clearance_m: float = 0.110
    staging_waypoint_margin_m: float = 0.015
    max_wheel_speed_rad_s: float = SMORES_EP_MAX_WHEEL_SPEED_RAD_S
    actuators: SmoresActuatorConfig = field(
        default_factory=SmoresActuatorConfig.payload_overdrive
    )
    geometry: SmoresGeometry = field(default_factory=SmoresGeometry)

    def __post_init__(self) -> None:
        if self.steps < 0:
            raise ValueError("Step count cannot be negative")
        if self.headless and self.steps == 0:
            raise ValueError("Headless runs require finite positive --steps")
        if (
            self.physics_hz <= 0
            or self.render_hz <= 0
            or self.state_publish_hz <= 0
        ):
            raise ValueError("Simulation frequencies must be positive")
        if self.physics_hz % self.render_hz != 0:
            raise ValueError(
                "Physics frequency must be divisible by render frequency"
            )
        if self.physics_hz % self.state_publish_hz != 0:
            raise ValueError(
                "Physics frequency must be divisible by state publish "
                "frequency"
            )
        if self.log_interval < 0:
            raise ValueError("Log interval cannot be negative")
        if len(self.module_ids) < 2:
            raise ValueError("Self-assembly requires at least two module IDs")
        if any(not module_id.strip() for module_id in self.module_ids):
            raise ValueError("Self-assembly module IDs cannot be empty")
        if len(set(self.module_ids)) != len(self.module_ids):
            raise ValueError("Self-assembly module IDs must be distinct")
        numeric = (
            self.spawn_height_m,
            self.initial_pitch_deg,
            self.spawn_half_width_m,
            self.outer_y_m,
            self.center_y_m,
            self.outer_yaw_deg,
            self.spawn_radius_m,
            self.staging_center_clearance_m,
            self.staging_waypoint_margin_m,
            self.max_wheel_speed_rad_s,
            self.action_command_timeout_s,
        )
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("Self-assembly simulation values must be finite")
        if (
            self.spawn_height_m <= 0.0
            or self.spawn_half_width_m <= 0.0
            or self.spawn_radius_m <= 0.0
            or self.staging_center_clearance_m <= 0.0
            or self.staging_waypoint_margin_m <= 0.0
            or self.center_y_m <= self.outer_y_m
            or self.max_wheel_speed_rad_s <= 0.0
            or self.action_command_timeout_s <= 0.0
        ):
            raise ValueError("Self-assembly distances and speeds are invalid")
        if any(
            not str(path)
            for path in (
                self.primitive_goal_file,
                self.primitive_cancel_file,
                self.primitive_status_file,
                self.action_file,
            )
        ):
            raise ValueError("Primitive channel paths cannot be empty")
