"""Action models for modular robot control."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum

from robots.control import PlanarVelocityCommand
from robots.joints import JointType
from robots.module_state import Vector3
from robots.state_registry import RobotStateSnapshot


class MagneticCommand(str, Enum):
    """Discrete magnetic attachment command."""

    HOLD = "hold"
    ATTACH = "attach"
    DETACH = "detach"


@dataclass(frozen=True)
class MagneticAction:
    """Command that changes magnetic attachment state between two modules."""

    module_a_id: str
    module_b_id: str
    command: MagneticCommand
    joint_type: JointType = JointType.SPHERICAL
    joint_axis: str | None = None
    edge_role: str | None = None
    attachment_mode: str = "rolling_contact"
    contact_point_world: Vector3 | None = None
    pivot_axis: Vector3 | None = None
    allows_rotation: bool = False
    is_load_bearing: bool = False
    is_temporary: bool = True


@dataclass(frozen=True)
class SimulationActions:
    """Locomotion and magnetic actions for one simulation step."""

    locomotion: Mapping[str, PlanarVelocityCommand] = field(default_factory=dict)
    magnetic: tuple[MagneticAction, ...] = ()
    reset_requested: bool = False


def create_attach_actions_from_contacts(
    snapshot: RobotStateSnapshot,
    joint_type: JointType = JointType.SPHERICAL,
) -> tuple[MagneticAction, ...]:
    """Create attach actions for every unordered module pair currently in contact."""
    actions: list[MagneticAction] = []
    seen_pairs: set[frozenset[str]] = set()

    for module in snapshot.modules:
        for attachment in module.surface_attachments:
            other_id = attachment.connected_module_id
            if other_id is None:
                continue

            pair = frozenset((module.module_id, other_id))
            if pair in seen_pairs:
                continue

            seen_pairs.add(pair)
            actions.append(
                MagneticAction(
                    module_a_id=module.module_id,
                    module_b_id=other_id,
                    command=MagneticCommand.ATTACH,
                    joint_type=attachment.joint_type or joint_type,
                    joint_axis=None,
                    edge_role="contact_attachment",
                    attachment_mode=attachment.attachment_mode,
                    pivot_axis=attachment.pivot_axis,
                    allows_rotation=attachment.allows_rotation,
                    is_load_bearing=attachment.is_load_bearing,
                    is_temporary=attachment.is_temporary,
                )
            )

    return tuple(actions)
