"""Human-readable debug formatting for robot graphs."""

from __future__ import annotations

from graphs.robot_graph import RobotGraph


def format_robot_graph(graph: RobotGraph) -> str:
    """Format a robot graph for terminal debugging."""
    lines = [f"[graph t={graph.timestamp:.3f}s] nodes={len(graph.nodes)} edges={len(graph.edges)}"]
    for edge in graph.edges:
        joint_type = edge.joint_type.value if edge.joint_type is not None else "none"
        lines.append(
            "  "
            f"{edge.source} -- {edge.target}: "
            f"status={edge.status.value} "
            f"joint={joint_type} "
            f"magnet={edge.is_magnet_enabled}"
        )
    return "\n".join(lines)
