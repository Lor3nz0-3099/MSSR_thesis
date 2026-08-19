"""JSON payloads for modular robot state, graph, and actions."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from robots.actions import MagneticAction, MagneticCommand, SimulationActions
from robots.control import PlanarVelocityCommand
from robots.joints import JointType

if TYPE_CHECKING:
    from graphs.robot_graph import RobotGraph
    from robots.state_registry import RobotStateSnapshot


def snapshot_to_dict(snapshot: RobotStateSnapshot) -> dict[str, Any]:
    """Convert a module state snapshot to a JSON-serializable dictionary."""
    return {
        "timestamp": snapshot.timestamp,
        "modules": [
            {
                "module_id": module.module_id,
                "prim_path": module.prim_path,
                "body_frame_id": module.body_frame_id,
                "pose": {
                    "frame_id": module.pose.frame_id,
                    "position": module.pose.position,
                    "orientation_xyzw": module.pose.orientation_xyzw,
                },
                "twist": {
                    "linear": module.twist.linear,
                    "angular": module.twist.angular,
                },
                "radius": module.radius,
                "mass": module.mass,
                "role": module.role.value,
                "capabilities": module.capabilities,
                "surface_attachments": [
                    {
                        "attachment_id": attachment.attachment_id,
                        "status": attachment.status.value,
                        "connected_module_id": attachment.connected_module_id,
                        "joint_type": (
                            attachment.joint_type.value
                            if attachment.joint_type is not None
                            else None
                        ),
                        "attachment_mode": attachment.attachment_mode,
                        "is_magnet_enabled": attachment.is_magnet_enabled,
                        "local_contact_point": attachment.local_contact_point,
                        "local_contact_normal": attachment.local_contact_normal,
                        "remote_contact_point": attachment.remote_contact_point,
                        "relative_position": attachment.relative_position,
                        "pivot_axis": attachment.pivot_axis,
                        "allows_rotation": attachment.allows_rotation,
                        "is_load_bearing": attachment.is_load_bearing,
                        "is_temporary": attachment.is_temporary,
                        "edge_role": attachment.edge_role,
                    }
                    for attachment in module.surface_attachments
                ],
                "last_command": {
                    "linear": module.last_command.linear,
                    "angular": module.last_command.angular,
                },
            }
            for module in snapshot.modules
        ],
    }


def graph_to_dict(graph: RobotGraph) -> dict[str, Any]:
    """Convert a robot graph to a JSON-serializable dictionary."""
    return {
        "timestamp": graph.timestamp,
        "nodes": [
            {
                "module_id": node.module_id,
                "attributes": node.attributes,
            }
            for node in graph.nodes
        ],
        "edges": [
            {
                "source": edge.source,
                "target": edge.target,
                "status": edge.status.value,
                "joint_type": edge.joint_type.value if edge.joint_type is not None else None,
                "is_magnet_enabled": edge.is_magnet_enabled,
                "attachment_mode": edge.attributes.get("attachment_mode"),
                "relative_position": edge.relative_position,
                "local_contact_point": edge.local_contact_point,
                "local_contact_normal": edge.local_contact_normal,
                "pivot_axis": edge.attributes.get("pivot_axis"),
                "allows_rotation": edge.attributes.get("allows_rotation", False),
                "is_load_bearing": edge.attributes.get("is_load_bearing", False),
                "is_temporary": edge.attributes.get("is_temporary", True),
                "attributes": edge.attributes,
            }
            for edge in graph.edges
        ],
    }


def actions_from_json(payload: str) -> SimulationActions:
    """Parse simulation actions from a JSON string."""
    data = json.loads(payload)
    action_data = _action_payload(data)
    locomotion = {
        module_id: PlanarVelocityCommand(
            vx=float(command.get("vx", 0.0)),
            vy=float(command.get("vy", 0.0)),
            yaw_rate=float(command.get("yaw_rate", 0.0)),
        )
        for module_id, command in action_data.get("locomotion", {}).items()
    }
    magnetic = tuple(
        MagneticAction(
            module_a_id=str(action["module_a_id"]),
            module_b_id=str(action["module_b_id"]),
            command=MagneticCommand(str(action.get("command", "hold"))),
            joint_type=JointType(str(action.get("joint_type", "spherical"))),
            joint_axis=_optional_string(action.get("joint_axis")),
            edge_role=_optional_string(action.get("edge_role") or action.get("role")),
            attachment_mode=str(action.get("attachment_mode", "rolling_contact")),
            contact_point_world=_optional_vector3(action.get("contact_point_world")),
            pivot_axis=_optional_vector3(action.get("pivot_axis")),
            allows_rotation=bool(action.get("allows_rotation", False)),
            is_load_bearing=bool(action.get("is_load_bearing", False)),
            is_temporary=bool(action.get("is_temporary", True)),
        )
        for action in action_data.get("magnetic", ())
    )
    return SimulationActions(
        locomotion=locomotion,
        magnetic=magnetic,
        reset_requested=_reset_requested(action_data),
    )


def actions_to_json(actions: SimulationActions) -> str:
    """Serialize simulation actions to a JSON string."""
    return json.dumps(
        {
            "schema_version": "mssr.actions.v1",
            "locomotion": {
                module_id: {
                    "vx": command.vx,
                    "vy": command.vy,
                    "yaw_rate": command.yaw_rate,
                }
                for module_id, command in actions.locomotion.items()
            },
            "magnetic": [
                {
                    "module_a_id": action.module_a_id,
                    "module_b_id": action.module_b_id,
                    "command": action.command.value,
                    "joint_type": action.joint_type.value,
                    "joint_axis": action.joint_axis,
                    "edge_role": action.edge_role,
                    "attachment_mode": action.attachment_mode,
                    "contact_point_world": action.contact_point_world,
                    "pivot_axis": action.pivot_axis,
                    "allows_rotation": action.allows_rotation,
                    "is_load_bearing": action.is_load_bearing,
                    "is_temporary": action.is_temporary,
                }
                for action in actions.magnetic
            ],
            "reset": actions.reset_requested,
        },
        sort_keys=True,
    )


def to_json(data: dict[str, Any]) -> str:
    """Serialize a JSON-compatible dictionary consistently."""
    return json.dumps(data, sort_keys=True)


def _action_payload(data: dict[str, Any]) -> dict[str, Any]:
    """Return the action object from either legacy or versioned JSON."""
    schema_version = data.get("schema_version")
    if schema_version in (None, "mssr.actions.v1", "mssr.actions.v2"):
        return data
    raise ValueError(f"Unsupported action schema_version: {schema_version}")


def _reset_requested(data: dict[str, Any]) -> bool:
    """Return whether the action payload requests a simulation reset."""
    reset = data.get("reset", False)
    if isinstance(reset, bool):
        return reset
    if isinstance(reset, dict):
        return bool(reset.get("requested", False)) or reset.get("command") == "reset"
    return False


def _optional_string(value: object) -> str | None:
    """Return an optional string from a JSON scalar."""
    if value is None:
        return None
    return str(value)


def _optional_vector3(value: object) -> tuple[float, float, float] | None:
    """Return an optional 3D vector from a JSON array."""
    if not isinstance(value, list | tuple) or len(value) < 3:
        return None
    return (float(value[0]), float(value[1]), float(value[2]))
