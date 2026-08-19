"""Simulator-independent state model for modular robots.

The classes in this module are intentionally pure Python data containers.
Isaac Sim, ROS 2, graph planners, imitation learning, and reinforcement
learning code can all convert to and from this representation without
depending on each other.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any

from robots.joints import JointType


Vector3 = tuple[float, float, float]
Quaternion = tuple[float, float, float, float]


class ModuleRole(str, Enum):
    """High-level role assigned to a module by a planner or controller."""

    BODY = "body"
    ATTACHMENT = "attachment"
    SENSOR = "sensor"
    ACTUATOR = "actuator"
    UNKNOWN = "unknown"


class AttachmentStatus(str, Enum):
    """Discrete state of an electropermanent magnetic surface attachment."""

    ABSENT = "absent"
    AVAILABLE = "available"
    IN_CONTACT = "in_contact"
    MAGNETIZED = "magnetized"
    CONNECTED = "connected"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class ModulePose:
    """Pose of a module in a named reference frame."""

    frame_id: str
    position: Vector3 = (0.0, 0.0, 0.0)
    orientation_xyzw: Quaternion = (0.0, 0.0, 0.0, 1.0)


@dataclass(frozen=True)
class ModuleTwist:
    """Linear and angular velocity of a module."""

    linear: Vector3 = (0.0, 0.0, 0.0)
    angular: Vector3 = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class SurfaceAttachmentState:
    """State of one magnetic attachment on a continuous module surface.

    FreeBOT-like modules can attach at arbitrary points on the spherical
    surface. Therefore this state describes an active or potential surface
    contact, not a predefined mechanical port.
    """

    attachment_id: str
    status: AttachmentStatus = AttachmentStatus.ABSENT
    connected_module_id: str | None = None
    joint_type: JointType | None = None
    attachment_mode: str = "rolling_contact"
    is_magnet_enabled: bool = False
    local_contact_point: Vector3 | None = None
    local_contact_normal: Vector3 | None = None
    remote_contact_point: Vector3 | None = None
    relative_position: Vector3 | None = None
    relative_orientation_xyzw: Quaternion | None = None
    pivot_axis: Vector3 | None = None
    allows_rotation: bool = False
    is_load_bearing: bool = False
    is_temporary: bool = True
    edge_role: str | None = None


@dataclass(frozen=True)
class ModuleState:
    """Public state of one module, suitable as a graph node attribute source."""

    module_id: str
    prim_path: str
    body_frame_id: str
    pose: ModulePose
    twist: ModuleTwist = field(default_factory=ModuleTwist)
    radius: float | None = None
    mass: float | None = None
    role: ModuleRole = ModuleRole.UNKNOWN
    capabilities: tuple[str, ...] = ()
    surface_attachments: tuple[SurfaceAttachmentState, ...] = ()
    last_command: ModuleTwist = field(default_factory=ModuleTwist)
    battery_level: float | None = None
    custom_attributes: dict[str, Any] = field(default_factory=dict)

    def with_pose(self, pose: ModulePose) -> ModuleState:
        """Return a copy of this module state with an updated pose."""
        return replace(self, pose=pose)

    def with_twist(self, twist: ModuleTwist) -> ModuleState:
        """Return a copy of this module state with updated velocity."""
        return replace(self, twist=twist)

    def with_last_command(self, command: ModuleTwist) -> ModuleState:
        """Return a copy of this module state with the last applied command."""
        return replace(self, last_command=command)

    def with_surface_attachments(self, attachments: tuple[SurfaceAttachmentState, ...]) -> ModuleState:
        """Return a copy of this module state with updated surface attachments."""
        return replace(self, surface_attachments=attachments)

    def with_custom_attribute(self, name: str, value: Any) -> ModuleState:
        """Return a copy of this module state with one extra algorithm attribute."""
        return replace(self, custom_attributes={**self.custom_attributes, name: value})

    def graph_attributes(self) -> dict[str, Any]:
        """Return a flat attribute dictionary for graph-based algorithms."""
        return {
            "module_id": self.module_id,
            "prim_path": self.prim_path,
            "body_frame_id": self.body_frame_id,
            "position": self.pose.position,
            "orientation_xyzw": self.pose.orientation_xyzw,
            "linear_velocity": self.twist.linear,
            "angular_velocity": self.twist.angular,
            "command_linear": self.last_command.linear,
            "command_angular": self.last_command.angular,
            "radius": self.radius,
            "mass": self.mass,
            "role": self.role.value,
            "capabilities": self.capabilities,
            "surface_attachment_count": len(self.surface_attachments),
            "joint_types": tuple(
                attachment.joint_type.value
                for attachment in self.surface_attachments
                if attachment.joint_type is not None
            ),
            "attachment_modes": tuple(
                attachment.attachment_mode
                for attachment in self.surface_attachments
            ),
            "pivot_attachment_count": sum(
                attachment.attachment_mode == "surface_pivot"
                for attachment in self.surface_attachments
            ),
            "active_magnet_count": sum(
                attachment.is_magnet_enabled for attachment in self.surface_attachments
            ),
            "connected_attachment_count": sum(
                attachment.status == AttachmentStatus.CONNECTED for attachment in self.surface_attachments
            ),
            "battery_level": self.battery_level,
            **self.custom_attributes,
        }


def create_spherical_module_state(
    module_id: str,
    prim_path: str,
    body_frame_id: str,
    world_frame_id: str,
    radius: float,
    mass: float,
    position: Vector3,
) -> ModuleState:
    """Create the initial public state for the current single-sphere robot."""
    return ModuleState(
        module_id=module_id,
        prim_path=prim_path,
        body_frame_id=body_frame_id,
        pose=ModulePose(frame_id=world_frame_id, position=position),
        radius=radius,
        mass=mass,
        role=ModuleRole.BODY,
        capabilities=("planar_motion", "rolling_body", "continuous_surface_attachment"),
        surface_attachments=(),
    )
