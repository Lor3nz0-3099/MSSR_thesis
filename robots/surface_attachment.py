"""Continuous surface attachment utilities for FreeBOT-like spherical modules."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from itertools import combinations
from collections.abc import Mapping

from robots.joints import JointType
from robots.module_state import AttachmentStatus, ModuleState, SurfaceAttachmentState, Vector3


ModulePair = frozenset[str]


class SurfaceAttachmentMode(str, Enum):
    """Geometric model used to choose valid magnetic contact sites."""

    CONTINUOUS = "continuous"
    SIX_AXIS = "six_axis"


@dataclass(frozen=True)
class SurfaceAttachmentConfig:
    """Geometric thresholds for detecting sphere-surface attachment candidates."""

    contact_tolerance: float = 0.03
    penetration_tolerance: float = 0.08
    auto_magnetize_on_contact: bool = False
    mode: SurfaceAttachmentMode = SurfaceAttachmentMode.CONTINUOUS
    six_axis_alignment_threshold: float = 0.95


@dataclass(frozen=True)
class SurfaceContactCandidate:
    """Potential continuous-surface contact between two spherical modules."""

    module_a_id: str
    module_b_id: str
    center_distance: float
    surface_gap: float
    normal_a_to_b: Vector3
    world_contact_point_on_a: Vector3
    world_contact_point_on_b: Vector3
    local_contact_point_on_a: Vector3
    local_contact_point_on_b: Vector3
    relative_position_a_to_b: Vector3


def detect_surface_contact(
    module_a: ModuleState,
    module_b: ModuleState,
    config: SurfaceAttachmentConfig | None = None,
) -> SurfaceContactCandidate | None:
    """Detect whether two spherical modules have a possible surface contact."""
    attachment_config = config or SurfaceAttachmentConfig()
    if module_a.radius is None or module_b.radius is None:
        return None

    relative_position = _subtract(module_b.pose.position, module_a.pose.position)
    center_distance = _norm(relative_position)
    if center_distance == 0.0:
        return None

    surface_gap = center_distance - module_a.radius - module_b.radius
    if surface_gap > attachment_config.contact_tolerance:
        return None
    if surface_gap < -attachment_config.penetration_tolerance:
        return None

    normal_world_a_to_b = _scale(relative_position, 1.0 / center_distance)
    local_normal_a, local_normal_b = _contact_normals(
        normal_world_a_to_b,
        module_a.pose.orientation_xyzw,
        module_b.pose.orientation_xyzw,
        attachment_config,
    )
    if local_normal_a is None or local_normal_b is None:
        return None

    local_contact_point_on_a = _scale(local_normal_a, module_a.radius)
    local_contact_point_on_b = _scale(local_normal_b, module_b.radius)
    world_normal_a = _rotate_vector(local_normal_a, module_a.pose.orientation_xyzw)
    world_normal_b = _rotate_vector(local_normal_b, module_b.pose.orientation_xyzw)
    world_contact_point_on_a = _add(module_a.pose.position, _scale(world_normal_a, module_a.radius))
    world_contact_point_on_b = _add(module_b.pose.position, _scale(world_normal_b, module_b.radius))
    return SurfaceContactCandidate(
        module_a_id=module_a.module_id,
        module_b_id=module_b.module_id,
        center_distance=center_distance,
        surface_gap=surface_gap,
        normal_a_to_b=local_normal_a,
        world_contact_point_on_a=world_contact_point_on_a,
        world_contact_point_on_b=world_contact_point_on_b,
        local_contact_point_on_a=local_contact_point_on_a,
        local_contact_point_on_b=local_contact_point_on_b,
        relative_position_a_to_b=relative_position,
    )


def create_surface_attachment_pair(
    candidate: SurfaceContactCandidate,
    magnet_enabled: bool = False,
    connected: bool = False,
    joint_type: JointType | None = None,
    metadata: Mapping[str, object] | None = None,
) -> tuple[SurfaceAttachmentState, SurfaceAttachmentState]:
    """Create reciprocal surface attachment states for a contact candidate."""
    metadata = metadata or {}
    status = _attachment_status(magnet_enabled=magnet_enabled, connected=connected)
    effective_magnet_enabled = magnet_enabled or connected
    attachment_a = SurfaceAttachmentState(
        attachment_id=f"{candidate.module_a_id}__{candidate.module_b_id}",
        status=status,
        connected_module_id=candidate.module_b_id,
        joint_type=joint_type,
        attachment_mode=str(metadata.get("attachment_mode", "rolling_contact")),
        is_magnet_enabled=effective_magnet_enabled,
        local_contact_point=candidate.local_contact_point_on_a,
        local_contact_normal=candidate.normal_a_to_b,
        remote_contact_point=candidate.local_contact_point_on_b,
        relative_position=candidate.relative_position_a_to_b,
        pivot_axis=_optional_vector3(metadata.get("pivot_axis")),
        allows_rotation=bool(metadata.get("allows_rotation", False)),
        is_load_bearing=bool(metadata.get("is_load_bearing", False)),
        is_temporary=bool(metadata.get("is_temporary", True)),
        edge_role=_optional_string(metadata.get("edge_role")),
    )
    attachment_b = SurfaceAttachmentState(
        attachment_id=f"{candidate.module_b_id}__{candidate.module_a_id}",
        status=status,
        connected_module_id=candidate.module_a_id,
        joint_type=joint_type,
        attachment_mode=str(metadata.get("attachment_mode", "rolling_contact")),
        is_magnet_enabled=effective_magnet_enabled,
        local_contact_point=candidate.local_contact_point_on_b,
        local_contact_normal=_scale(candidate.normal_a_to_b, -1.0),
        remote_contact_point=candidate.local_contact_point_on_a,
        relative_position=_scale(candidate.relative_position_a_to_b, -1.0),
        pivot_axis=_optional_vector3(metadata.get("pivot_axis")),
        allows_rotation=bool(metadata.get("allows_rotation", False)),
        is_load_bearing=bool(metadata.get("is_load_bearing", False)),
        is_temporary=bool(metadata.get("is_temporary", True)),
        edge_role=_optional_string(metadata.get("edge_role")),
    )
    return attachment_a, attachment_b


def annotate_surface_contacts(
    modules: tuple[ModuleState, ...],
    config: SurfaceAttachmentConfig | None = None,
    connected_pairs: set[ModulePair] | None = None,
    connected_joint_types: dict[ModulePair, JointType] | None = None,
    connected_attachment_metadata: dict[ModulePair, Mapping[str, object]] | None = None,
) -> tuple[ModuleState, ...]:
    """Return module states annotated with current surface contact candidates."""
    attachment_config = config or SurfaceAttachmentConfig()
    connected_module_pairs = connected_pairs or set()
    connected_joint_type_by_pair = connected_joint_types or {}
    connected_metadata_by_pair = connected_attachment_metadata or {}
    attachments_by_module: dict[str, list[SurfaceAttachmentState]] = {
        module.module_id: [] for module in modules
    }

    for module_a, module_b in combinations(modules, 2):
        candidate = detect_surface_contact(module_a, module_b, attachment_config)
        if candidate is None:
            continue

        pair = _module_pair(module_a.module_id, module_b.module_id)
        attachment_a, attachment_b = create_surface_attachment_pair(
            candidate,
            magnet_enabled=attachment_config.auto_magnetize_on_contact,
            connected=pair in connected_module_pairs or pair in connected_joint_type_by_pair,
            joint_type=connected_joint_type_by_pair.get(pair),
            metadata=connected_metadata_by_pair.get(pair),
        )
        attachments_by_module[module_a.module_id].append(attachment_a)
        attachments_by_module[module_b.module_id].append(attachment_b)

    modules_by_id = {module.module_id: module for module in modules}
    for pair in connected_module_pairs | set(connected_joint_type_by_pair):
        first_id, second_id = sorted(pair)
        if first_id not in modules_by_id or second_id not in modules_by_id:
            continue
        if _has_attachment(attachments_by_module[first_id], second_id):
            continue
        candidate = _synthetic_contact_candidate(
            modules_by_id[first_id],
            modules_by_id[second_id],
        )
        if candidate is None:
            continue
        attachment_a, attachment_b = create_surface_attachment_pair(
            candidate,
            magnet_enabled=True,
            connected=True,
            joint_type=connected_joint_type_by_pair.get(pair),
            metadata=connected_metadata_by_pair.get(pair),
        )
        attachments_by_module[first_id].append(attachment_a)
        attachments_by_module[second_id].append(attachment_b)

    return tuple(
        module.with_surface_attachments(tuple(attachments_by_module[module.module_id]))
        for module in modules
    )


def _attachment_status(magnet_enabled: bool, connected: bool) -> AttachmentStatus:
    """Return the semantic attachment status for the current low-level state."""
    if connected:
        return AttachmentStatus.CONNECTED
    if magnet_enabled:
        return AttachmentStatus.MAGNETIZED
    return AttachmentStatus.IN_CONTACT


def _module_pair(module_a_id: str, module_b_id: str) -> ModulePair:
    """Return an unordered module pair key."""
    return frozenset((module_a_id, module_b_id))


def _has_attachment(attachments: list[SurfaceAttachmentState], other_id: str) -> bool:
    return any(attachment.connected_module_id == other_id for attachment in attachments)


def _synthetic_contact_candidate(
    module_a: ModuleState,
    module_b: ModuleState,
) -> SurfaceContactCandidate | None:
    """Build attachment geometry for an already-constrained pair without contact."""
    if module_a.radius is None or module_b.radius is None:
        return None
    relative_position = _subtract(module_b.pose.position, module_a.pose.position)
    center_distance = _norm(relative_position)
    if center_distance == 0.0:
        return None
    normal_world_a_to_b = _scale(relative_position, 1.0 / center_distance)
    local_normal_a = _inverse_rotate_vector(normal_world_a_to_b, module_a.pose.orientation_xyzw)
    local_normal_b = _inverse_rotate_vector(_scale(normal_world_a_to_b, -1.0), module_b.pose.orientation_xyzw)
    local_contact_point_on_a = _scale(local_normal_a, module_a.radius)
    local_contact_point_on_b = _scale(local_normal_b, module_b.radius)
    return SurfaceContactCandidate(
        module_a_id=module_a.module_id,
        module_b_id=module_b.module_id,
        center_distance=center_distance,
        surface_gap=center_distance - module_a.radius - module_b.radius,
        normal_a_to_b=local_normal_a,
        world_contact_point_on_a=_add(module_a.pose.position, _scale(normal_world_a_to_b, module_a.radius)),
        world_contact_point_on_b=_add(module_b.pose.position, _scale(normal_world_a_to_b, -module_b.radius)),
        local_contact_point_on_a=local_contact_point_on_a,
        local_contact_point_on_b=local_contact_point_on_b,
        relative_position_a_to_b=relative_position,
    )


def _contact_normals(
    normal_world_a_to_b: Vector3,
    orientation_a_xyzw: tuple[float, float, float, float],
    orientation_b_xyzw: tuple[float, float, float, float],
    config: SurfaceAttachmentConfig,
) -> tuple[Vector3 | None, Vector3 | None]:
    """Return local contact normals according to the configured site model."""
    local_a = _inverse_rotate_vector(normal_world_a_to_b, orientation_a_xyzw)
    local_b = _inverse_rotate_vector(_scale(normal_world_a_to_b, -1.0), orientation_b_xyzw)

    if config.mode == SurfaceAttachmentMode.CONTINUOUS:
        return local_a, local_b

    if config.mode == SurfaceAttachmentMode.SIX_AXIS:
        site_a, alignment_a = _nearest_axis(local_a)
        site_b, alignment_b = _nearest_axis(local_b)
        if min(alignment_a, alignment_b) < config.six_axis_alignment_threshold:
            return None, None
        return site_a, site_b

    raise ValueError(f"Unsupported surface attachment mode: {config.mode}")


def _inverse_rotate_vector(vector: Vector3, quaternion_xyzw: tuple[float, float, float, float]) -> Vector3:
    """Rotate a world vector into a body's local frame."""
    x, y, z, w = quaternion_xyzw
    return _quat_rotate(vector, (-x, -y, -z, w))


def _rotate_vector(vector: Vector3, quaternion_xyzw: tuple[float, float, float, float]) -> Vector3:
    """Rotate a local vector into the world frame."""
    return _quat_rotate(vector, quaternion_xyzw)


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


def _nearest_axis(vector: Vector3) -> tuple[Vector3, float]:
    """Return the closest axis-aligned site and its dot-product alignment."""
    axes: tuple[Vector3, ...] = (
        (1.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, -1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.0, 0.0, -1.0),
    )
    best_axis = max(axes, key=lambda axis: _dot(vector, axis))
    return best_axis, _dot(vector, best_axis)


def _dot(a: Vector3, b: Vector3) -> float:
    """Return the dot product between two vectors."""
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _add(a: Vector3, b: Vector3) -> Vector3:
    """Add two 3D vectors."""
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _subtract(a: Vector3, b: Vector3) -> Vector3:
    """Subtract two 3D vectors."""
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _scale(vector: Vector3, scalar: float) -> Vector3:
    """Scale a 3D vector."""
    return (vector[0] * scalar, vector[1] * scalar, vector[2] * scalar)


def _norm(vector: Vector3) -> float:
    """Return the Euclidean norm of a 3D vector."""
    return math.sqrt(vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2)


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_vector3(value: object) -> Vector3 | None:
    if not isinstance(value, list | tuple) or len(value) < 3:
        return None
    return (float(value[0]), float(value[1]), float(value[2]))
