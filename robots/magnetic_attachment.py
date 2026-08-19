"""Physical magnetic attachment management for simulated modules."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping

from robots.actions import MagneticAction, MagneticCommand
from robots.joints import JointType
from robots.module_state import ModuleState, SurfaceAttachmentState, Vector3
from robots.state_registry import RobotStateSnapshot
from robots.surface_attachment import ModulePair


@dataclass(frozen=True)
class MagneticAttachment:
    """Runtime handle for one physical magnetic attachment constraint."""

    module_a_id: str
    module_b_id: str
    joint_path: str
    joint_type: JointType
    attachment_mode: str = "rolling_contact"
    pivot_axis: Vector3 | None = None
    allows_rotation: bool = False
    is_load_bearing: bool = False
    is_temporary: bool = True
    edge_role: str | None = None

    @property
    def pair(self) -> ModulePair:
        """Return the unordered module pair represented by this attachment."""
        return frozenset((self.module_a_id, self.module_b_id))


class MagneticAttachmentManager:
    """Create and remove spherical joints that approximate activated magnets.

    A spherical joint keeps the two contact points coincident while allowing
    relative rotation. This better matches magnetically attached spherical
    modules than a fixed joint, which would lock the full relative pose.
    """

    def __init__(self, root_path: str = "/World/MagneticAttachments") -> None:
        """Create a manager for physical attachment joints in the current stage."""
        import omni.usd
        from pxr import UsdGeom

        self._root_path = root_path
        self._attachments: dict[ModulePair, MagneticAttachment] = {}
        self._stage = omni.usd.get_context().get_stage()
        UsdGeom.Xform.Define(self._stage, self._root_path)
        self._fallback_contact_tolerance = 0.08
        self._fallback_penetration_tolerance = 0.08

    @property
    def connected_pairs(self) -> set[ModulePair]:
        """Return unordered module pairs currently constrained by physical joints."""
        return set(self._attachments)

    @property
    def connected_joint_types(self) -> dict[ModulePair, JointType]:
        """Return physical joint types indexed by unordered module pair."""
        return {pair: attachment.joint_type for pair, attachment in self._attachments.items()}

    @property
    def connected_attachment_metadata(self) -> dict[ModulePair, Mapping[str, object]]:
        """Return semantic attachment metadata indexed by unordered module pair."""
        return {
            pair: {
                "attachment_mode": attachment.attachment_mode,
                "pivot_axis": attachment.pivot_axis,
                "allows_rotation": attachment.allows_rotation,
                "is_load_bearing": attachment.is_load_bearing,
                "is_temporary": attachment.is_temporary,
                "edge_role": attachment.edge_role,
            }
            for pair, attachment in self._attachments.items()
        }

    def is_attached(self, module_a_id: str, module_b_id: str) -> bool:
        """Return whether two modules already have a physical attachment."""
        return _module_pair(module_a_id, module_b_id) in self._attachments

    def attach(
        self,
        module_a: ModuleState,
        module_b: ModuleState,
        local_pos_a: Vector3 | None = None,
        local_pos_b: Vector3 | None = None,
        joint_type: JointType = JointType.SPHERICAL,
        attachment_mode: str = "rolling_contact",
        pivot_axis: Vector3 | None = None,
        allows_rotation: bool = False,
        is_load_bearing: bool = False,
        is_temporary: bool = True,
        edge_role: str | None = None,
    ) -> MagneticAttachment:
        """Create a physical joint between two module rigid bodies if needed."""
        from pxr import Sdf, UsdPhysics

        pair = _module_pair(module_a.module_id, module_b.module_id)
        existing_attachment = self._attachments.get(pair)
        if existing_attachment is not None:
            return existing_attachment

        joint_path = self._joint_path(module_a.module_id, module_b.module_id)
        joint = _define_joint(self._stage, joint_path, joint_type, pivot_axis=pivot_axis)
        joint.CreateBody0Rel().SetTargets([Sdf.Path(module_a.prim_path)])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(module_b.prim_path)])
        joint.CreateLocalPos0Attr().Set(_gf_vec3(local_pos_a or (0.0, 0.0, 0.0)))
        joint.CreateLocalPos1Attr().Set(_gf_vec3(local_pos_b or (0.0, 0.0, 0.0)))

        attachment = MagneticAttachment(
            module_a_id=module_a.module_id,
            module_b_id=module_b.module_id,
            joint_path=joint_path,
            joint_type=joint_type,
            attachment_mode=attachment_mode,
            pivot_axis=pivot_axis,
            allows_rotation=allows_rotation,
            is_load_bearing=is_load_bearing,
            is_temporary=is_temporary,
            edge_role=edge_role,
        )
        self._attachments[pair] = attachment
        return attachment

    def detach(self, module_a_id: str, module_b_id: str) -> None:
        """Remove the physical joint between two modules if it exists."""
        pair = _module_pair(module_a_id, module_b_id)
        attachment = self._attachments.pop(pair, None)
        if attachment is not None:
            self._stage.RemovePrim(attachment.joint_path)

    def detach_all(self) -> None:
        """Remove every physical magnetic attachment managed here."""
        for attachment in tuple(self._attachments.values()):
            self._stage.RemovePrim(attachment.joint_path)
        self._attachments.clear()

    def apply_action(
        self,
        action: MagneticAction,
        snapshot: RobotStateSnapshot,
    ) -> MagneticAttachment | None:
        """Apply one explicit magnetic action using the current module snapshot."""
        if action.command == MagneticCommand.HOLD:
            return None

        if action.command == MagneticCommand.DETACH:
            self.detach(action.module_a_id, action.module_b_id)
            return None

        if action.command != MagneticCommand.ATTACH:
            raise ValueError(f"Unsupported magnetic command: {action.command}")

        modules_by_id = snapshot.module_by_id()
        module_a = modules_by_id.get(action.module_a_id)
        module_b = modules_by_id.get(action.module_b_id)
        if module_a is None or module_b is None:
            return None

        contact = _find_surface_attachment(module_a, action.module_b_id)
        if contact is None and not self._is_near_surface_contact(module_a, module_b):
            return None

        if contact is None:
            local_pos_a = _world_to_local_contact(action.contact_point_world, module_a)
            local_pos_b = _world_to_local_contact(action.contact_point_world, module_b)
        else:
            local_pos_a = contact.local_contact_point
            local_pos_b = contact.remote_contact_point

        return self.attach(
            module_a,
            module_b,
            local_pos_a=local_pos_a,
            local_pos_b=local_pos_b,
            joint_type=action.joint_type,
            attachment_mode=action.attachment_mode,
            pivot_axis=action.pivot_axis,
            allows_rotation=action.allows_rotation,
            is_load_bearing=action.is_load_bearing,
            is_temporary=action.is_temporary,
            edge_role=action.edge_role,
        )

    def _is_near_surface_contact(self, module_a: ModuleState, module_b: ModuleState) -> bool:
        if module_a.radius is None or module_b.radius is None:
            return False
        distance = _distance(module_a.pose.position, module_b.pose.position)
        surface_gap = distance - module_a.radius - module_b.radius
        return (
            surface_gap <= self._fallback_contact_tolerance
            and surface_gap >= -self._fallback_penetration_tolerance
        )

    def apply_actions(
        self,
        actions: tuple[MagneticAction, ...],
        snapshot: RobotStateSnapshot,
    ) -> tuple[MagneticAttachment, ...]:
        """Apply multiple explicit magnetic actions."""
        attachments: list[MagneticAttachment] = []
        for action in actions:
            attachment = self.apply_action(action, snapshot)
            if attachment is not None:
                attachments.append(attachment)
        return tuple(attachments)

    def attach_contacts(self, snapshot: RobotStateSnapshot) -> tuple[MagneticAttachment, ...]:
        """Attach all module pairs that are currently in contact in a snapshot."""
        modules_by_id = snapshot.module_by_id()
        created_or_existing: list[MagneticAttachment] = []
        for module in snapshot.modules:
            for surface_attachment in module.surface_attachments:
                other_id = surface_attachment.connected_module_id
                if other_id is None or module.module_id > other_id:
                    continue
                other_module = modules_by_id.get(other_id)
                if other_module is None:
                    continue
                created_or_existing.append(
                    self.attach(
                        module,
                        other_module,
                        local_pos_a=surface_attachment.local_contact_point,
                        local_pos_b=surface_attachment.remote_contact_point,
                        joint_type=surface_attachment.joint_type or JointType.SPHERICAL,
                        attachment_mode=surface_attachment.attachment_mode,
                        pivot_axis=surface_attachment.pivot_axis,
                        allows_rotation=surface_attachment.allows_rotation,
                        is_load_bearing=surface_attachment.is_load_bearing,
                        is_temporary=surface_attachment.is_temporary,
                        edge_role=surface_attachment.edge_role,
                    )
                )
        return tuple(created_or_existing)

    def _joint_path(self, module_a_id: str, module_b_id: str) -> str:
        """Return a deterministic USD prim path for an attachment joint."""
        first_id, second_id = sorted((module_a_id, module_b_id))
        return f"{self._root_path}/{first_id}__{second_id}_joint"


def _module_pair(module_a_id: str, module_b_id: str) -> ModulePair:
    """Return an unordered module pair key."""
    return frozenset((module_a_id, module_b_id))


def _find_surface_attachment(module: ModuleState, other_module_id: str) -> SurfaceAttachmentState | None:
    """Find this module's surface attachment state for another module."""
    for attachment in module.surface_attachments:
        if attachment.connected_module_id == other_module_id:
            return attachment
    return None


def _gf_vec3(vector: Vector3) -> object:
    """Convert a Python vector to a USD ``Gf.Vec3f``."""
    from pxr import Gf

    return Gf.Vec3f(float(vector[0]), float(vector[1]), float(vector[2]))


def _world_to_local_contact(contact_point_world: Vector3 | None, module: ModuleState) -> Vector3:
    """Approximate a world contact point as a local joint anchor for a sphere."""
    if contact_point_world is None:
        return (0.0, 0.0, 0.0)
    world_offset = (
        float(contact_point_world[0]) - float(module.pose.position[0]),
        float(contact_point_world[1]) - float(module.pose.position[1]),
        float(contact_point_world[2]) - float(module.pose.position[2]),
    )
    return _inverse_rotate_vector(world_offset, module.pose.orientation_xyzw)


def _distance(first: Vector3, second: Vector3) -> float:
    return (
        (float(first[0]) - float(second[0])) ** 2
        + (float(first[1]) - float(second[1])) ** 2
        + (float(first[2]) - float(second[2])) ** 2
    ) ** 0.5


def _inverse_rotate_vector(vector: Vector3, quaternion_xyzw: tuple[float, float, float, float]) -> Vector3:
    x, y, z, w = quaternion_xyzw
    return _quat_rotate(vector, (-x, -y, -z, w))


def _quat_rotate(vector: Vector3, quaternion_xyzw: tuple[float, float, float, float]) -> Vector3:
    x, y, z, w = quaternion_xyzw
    vx, vy, vz = vector
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    )


def _define_joint(
    stage: object,
    joint_path: str,
    joint_type: JointType,
    pivot_axis: Vector3 | None = None,
) -> object:
    """Define the requested USD Physics joint type."""
    from pxr import UsdPhysics

    if joint_type == JointType.RIGID:
        return UsdPhysics.FixedJoint.Define(stage, joint_path)

    if joint_type == JointType.SPHERICAL:
        joint = UsdPhysics.SphericalJoint.Define(stage, joint_path)
        joint.CreateAxisAttr().Set("X")
        joint.CreateConeAngle0LimitAttr().Set(-1.0)
        joint.CreateConeAngle1LimitAttr().Set(-1.0)
        return joint

    if joint_type == JointType.HINGE:
        joint = UsdPhysics.RevoluteJoint.Define(stage, joint_path)
        joint.CreateAxisAttr().Set(_joint_axis_from_vector(pivot_axis))
        return joint

    raise ValueError(f"Unsupported joint type: {joint_type}")


def _joint_axis_from_vector(vector: Vector3 | None) -> str:
    """Convert a pivot vector to the closest USD revolute axis token."""
    if vector is None:
        return "Z"
    abs_values = [abs(float(vector[0])), abs(float(vector[1])), abs(float(vector[2]))]
    axis_index = max(range(3), key=lambda index: abs_values[index])
    return ("X", "Y", "Z")[axis_index]
