"""Human-readable debug formatting for robot state snapshots."""

from __future__ import annotations

from robots.module_state import Vector3
from robots.state_registry import RobotStateSnapshot


def format_state_snapshot(snapshot: RobotStateSnapshot) -> str:
    """Format a robot state snapshot for terminal debugging."""
    lines = [f"[state t={snapshot.timestamp:.3f}s] modules={len(snapshot.modules)}"]
    for module in snapshot.modules:
        lines.append(
            "  "
            f"{module.module_id}: "
            f"pos={_format_vector(module.pose.position)} "
            f"lin={_format_vector(module.twist.linear)} "
            f"ang={_format_vector(module.twist.angular)} "
            f"cmd={_format_vector(module.last_command.linear)}/{_format_vector(module.last_command.angular)} "
            f"attachments={len(module.surface_attachments)}"
        )
        for attachment in module.surface_attachments:
            lines.append(
                "    "
                f"{attachment.attachment_id}: "
                f"status={attachment.status.value} "
                f"other={attachment.connected_module_id} "
                f"joint={attachment.joint_type.value if attachment.joint_type is not None else 'none'} "
                f"magnet={attachment.is_magnet_enabled}"
            )
    return "\n".join(lines)


def _format_vector(vector: Vector3) -> str:
    """Format a 3D vector compactly."""
    return f"({vector[0]:+.3f}, {vector[1]:+.3f}, {vector[2]:+.3f})"
