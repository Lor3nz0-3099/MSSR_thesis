"""Live-pose projection for coherent multi-module locomotion."""

from __future__ import annotations

import math
from typing import Any, Mapping

from mssr_expert.graph.attributed_robot_graph import AttributedRobotGraph


def coherent_planar_train_commands(
    graph: AttributedRobotGraph,
    commands: Mapping[str, Mapping[str, float]],
) -> dict[str, dict[str, float]]:
    """Make wheel locomotors push in one common world direction.

    ``vx`` is expressed in each module's local body frame.  Connector names
    and clocking alone do not determine whether two assembled bodies have
    equal or opposite local forward axes.  For pure translation this function
    uses the live body quaternions instead: the first usable locomotor defines
    the requested world direction, and every other usable locomotor receives
    the local sign whose projected velocity agrees with it.

    Commands involving yaw or PAN velocity are returned unchanged because
    their kinematics are not a one-dimensional train projection.
    """

    result = {
        str(module_id): dict(command)
        for module_id, command in commands.items()
    }
    if len(result) < 2 or any(
        abs(float(command.get("yaw_rate", 0.0))) > 1.0e-9
        or abs(float(command.get("pan_rate_rad_s", 0.0))) > 1.0e-9
        for command in result.values()
    ):
        return result

    nodes = graph.node_by_id()
    usable: list[tuple[str, tuple[float, float], float]] = []
    # Preserve the library's selector order.  It is the morphology-level
    # longitudinal order (rear-to-front for serial trains), whereas sorting
    # physical IDs would make the reference depend on Hungarian assignment.
    for module_id, command in result.items():
        speed = float(command.get("vx", 0.0))
        if abs(speed) <= 1.0e-9:
            continue
        node = nodes.get(module_id)
        if node is None:
            continue
        forward = _planar_body_forward(node.attributes)
        if forward is None:
            continue
        usable.append((module_id, forward, speed))
    if len(usable) < 2:
        return result

    _, reference_forward, reference_speed = usable[0]
    desired_world = (
        reference_forward[0] * reference_speed,
        reference_forward[1] * reference_speed,
    )
    for module_id, forward, speed in usable:
        agreement = (
            forward[0] * desired_world[0]
            + forward[1] * desired_world[1]
        )
        if abs(agreement) <= 1.0e-9:
            continue
        result[module_id]["vx"] = math.copysign(abs(speed), agreement)
    return result


def _planar_body_forward(
    attributes: Mapping[str, Any],
) -> tuple[float, float] | None:
    pose = attributes.get("pose", {})
    if not isinstance(pose, Mapping):
        return None
    raw = pose.get("orientation_xyzw", pose.get("orientation"))
    if not isinstance(raw, list | tuple) or len(raw) != 4:
        return None
    x, y, z, w = (float(value) for value in raw)
    if not all(math.isfinite(value) for value in (x, y, z, w)):
        return None
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 1.0e-12:
        return None
    x, y, z, w = (value / norm for value in (x, y, z, w))
    # First column of the quaternion rotation matrix: local +X in world.
    forward_x = 1.0 - 2.0 * (y * y + z * z)
    forward_y = 2.0 * (x * y + w * z)
    planar_norm = math.hypot(forward_x, forward_y)
    if planar_norm <= 1.0e-6:
        return None
    return forward_x / planar_norm, forward_y / planar_norm
