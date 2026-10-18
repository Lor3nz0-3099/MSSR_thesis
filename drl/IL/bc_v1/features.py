"""E1 causal physical geometry derived ONLY from frozen D2 PolicyInputs.

This is an intermediate immutable view, not a learned feature schema. Source
names/scopes identify provenance; output_frame identifies actual coordinates.
No distance/bearing selection, R6 emission, deduplication or tensor ordering.

Matrix tuples are row-major. Plane tuples contain (nx, ny, nz, d), n.p = d;
multiple planes are concatenated. A line is (point3, direction3), not an
observed XYZ point: it preserves an XY observation's unknown height. Bounds
are four boundary planes, without silently turning rotated bounds into AABBs.
A planar_pose is (line6, heading3). An angle_constraint is (angle_radians,
source_basis9); yaw/pitch alone never invent the other attitude components.
Dimensions remain intrinsic lengths, never translated/rotated as vectors.

Connector quantities are module-body local; edge pose is endpoint-A body
relative. Both remain local. IDs are opaque references, never frame anchors.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Literal

import numpy as np

from .bc_v1_contract import (
    ConnectorRef, ConnectorState, ConnectionState, ModuleRef, PolicyInputs,
    ResourceRef, ResourceState, StateGraph,
)
from .policy_inputs import _validate_selected_inputs


Matrix3 = tuple[float, ...]
Vector3 = tuple[float, float, float]
Scope = Literal["g_current", "c_static", "o_dynamic"]
_IDENTITY = (1., 0., 0., 0., 1., 0., 0., 0., 1.)
_PHYSICAL_FRAMES = frozenset({"world", "map", "stage_local"})


def _finite(values, size=None):
    try:
        values = tuple(values)
    except TypeError as exc:
        raise ValueError("Expected finite physical components") from exc
    if not values or size is not None and len(values) != size:
        raise ValueError("Invalid physical component shape")
    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Real) or not math.isfinite(v) for v in values):
        raise ValueError("Physical components declared valid must be finite numbers")
    return tuple(float(v) for v in values)


def _rotation(values):
    r = np.array(_finite(values, 9)).reshape(3, 3)
    if not np.allclose(r.T @ r, np.eye(3), rtol=0., atol=1e-10) or not math.isclose(np.linalg.det(r), 1., abs_tol=1e-10):
        raise ValueError("Physical axes require a proper orthonormal rotation")
    return r


def _flat(array):
    return _finite(np.asarray(array).ravel())


@dataclass(frozen=True)
class SourceFrameTransform:
    """Certified online bridge: p_graph = rotation @ p_named + translation."""

    frame_id: str
    translation: Vector3
    rotation: Matrix3 = _IDENTITY

    def __post_init__(self):
        if type(self.frame_id) is not str or self.frame_id not in _PHYSICAL_FRAMES:
            raise ValueError("Source bridge requires a declared physical frame")
        object.__setattr__(self, "translation", _finite(self.translation, 3))
        object.__setattr__(self, "rotation", _flat(_rotation(self.rotation)))


@dataclass(frozen=True)
class PhysicalFrameConfig:
    """Explicit pose source and physical axes; never infer axes from entities.

    Default source/axes convention is world. For map/task-local axes the caller
    supplies the certified source-from-axes rotation. A named source without
    an explicit bridge is missing; world/map are never assumed identical.
    body_local task geometry has no owning module in D2 and cannot be inverted.
    """

    graph_frame: str = "world"
    axes_id: str = "world"
    rotation_source_from_axes: Matrix3 = _IDENTITY
    transforms: tuple[SourceFrameTransform, ...] = ()

    def __post_init__(self):
        if any(type(v) is not str or v not in _PHYSICAL_FRAMES for v in (self.graph_frame, self.axes_id)):
            raise ValueError("Configuration requires declared physical frame names")
        object.__setattr__(self, "rotation_source_from_axes", _flat(_rotation(self.rotation_source_from_axes)))
        if type(self.transforms) is not tuple or any(type(t) is not SourceFrameTransform for t in self.transforms):
            raise TypeError("Frame bridges must be an immutable typed tuple")
        names = [t.frame_id for t in self.transforms]
        if len(names) != len(set(names)) or self.graph_frame in names:
            raise ValueError("Ambiguous or redundant physical source bridge")


@dataclass(frozen=True)
class PhysicalFrame:
    origin: Vector3 | None
    rotation_source_from_axes: Matrix3
    graph_frame: str
    axes_id: str
    valid: bool


@dataclass(frozen=True)
class ModuleGeometry:
    ref: ModuleRef
    position: Vector3 | None
    orientation_matrix: Matrix3 | None
    linear_velocity: Vector3 | None
    angular_velocity: Vector3 | None
    pose_valid: bool
    source_scope: Scope = "g_current"
    output_frame: str = "robot_centroid"


@dataclass(frozen=True)
class DerivedQuantity:
    source_scope: Scope
    source_name: str
    source_frame: str | None
    output_frame: str
    kind: str
    values: tuple[float, ...] | None
    valid: bool
    entity_ref: ModuleRef | ConnectorRef | ResourceRef | tuple[ConnectorRef, ConnectorRef] | None = None
    coordinate_id: str | None = None  # Opaque resource-local routing reference.


@dataclass(frozen=True)
class FrameGeometryView:
    frame: PhysicalFrame
    modules: tuple[ModuleGeometry, ...]
    quantities: tuple[DerivedQuantity, ...]
    local_geometry: tuple[DerivedQuantity, ...]
    connectors: tuple[ConnectorState, ...]
    connections: tuple[ConnectionState, ...]
    resources: tuple[ResourceState, ...]


def physical_frame(graph: StateGraph, frame_config: PhysicalFrameConfig) -> PhysicalFrame:
    """Centroid of all observed valid current poses, independent of topology."""
    if type(graph) is not StateGraph or type(frame_config) is not PhysicalFrameConfig:
        raise TypeError("Frame construction requires typed graph/config")
    positions = []
    for module in graph.modules:
        if type(module.pose_valid) is not bool:
            raise ValueError("Pose validity must be explicit boolean")
        if module.pose_valid:
            positions.append(_finite(module.position, 3))
    # fsum is stable under permutation; dividing first also avoids large-sum
    # overflow for valid finite positions whose mean is representable.
    origin = tuple(math.fsum(p[i]/len(positions) for p in positions) for i in range(3)) if positions else None
    return PhysicalFrame(origin, frame_config.rotation_source_from_axes,
                         frame_config.graph_frame, frame_config.axes_id, origin is not None)


def transform_point(point, frame: PhysicalFrame, *, valid=True) -> Vector3 | None:
    if not valid or point is None:
        return None
    point = _finite(point, 3)
    if not frame.valid:
        return None
    return _flat(_rotation(frame.rotation_source_from_axes).T @ (np.array(point)-frame.origin))


def relative_position(point, frame: PhysicalFrame, *, valid=True) -> Vector3 | None:
    return transform_point(point, frame, valid=valid)


def transform_vector(vector, frame: PhysicalFrame, *, valid=True) -> Vector3 | None:
    if not valid or vector is None:
        return None
    # Rotation alone does not require a centroid; measured twist can survive
    # missing position. Frame axes remain physically declared in this case.
    return _flat(_rotation(frame.rotation_source_from_axes).T @ _finite(vector, 3))


def quaternion_to_matrix(quaternion) -> Matrix3:
    q = np.array(_finite(quaternion, 4))
    scale = np.max(np.abs(q))
    if scale == 0:
        raise ValueError("Quaternion declared valid cannot be zero")
    q /= scale
    q /= np.linalg.norm(q)
    x, y, z, w = q
    return _flat(((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)),
                  (2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)),
                  (2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y))))


def matrix_to_r6(matrix) -> tuple[float, ...]:
    """First two rotation columns, concatenated; no E2 column emission."""
    r = _rotation(matrix)
    return _flat(r[:, :2].T)


def quaternion_to_r6(quaternion) -> tuple[float, ...]:
    return matrix_to_r6(quaternion_to_matrix(quaternion))


def transform_orientation(quaternion, frame: PhysicalFrame, *, valid=True) -> Matrix3 | None:
    if not valid or quaternion is None:
        return None
    r = np.array(quaternion_to_matrix(quaternion)).reshape(3, 3)
    return _flat(_rotation(frame.rotation_source_from_axes).T @ r)


def transform_twist(linear_velocity, angular_velocity, frame: PhysicalFrame):
    return transform_vector(linear_velocity, frame), transform_vector(angular_velocity, frame)


def euclidean_distance(point, other=None) -> float | None:
    """Generic norm/distance in the supplied physical dimension; None stays None."""
    if point is None:
        return None
    p = _finite(point)
    if other is not None:
        q = _finite(other, len(p))
        p = _finite(tuple(a-b for a,b in zip(p,q)))
    distance = math.hypot(*p)
    if not math.isfinite(distance):
        raise ValueError("Computed physical distance is nonfinite")
    return distance


def _source_bridge(name, config):
    if name == config.graph_frame:
        return np.eye(3), np.zeros(3)
    for bridge in config.transforms:
        if bridge.frame_id == name:
            return _rotation(bridge.rotation), np.array(bridge.translation)
    return None


def _geometry_kind(base):
    # Physical quantity codecs, not task behavior or final learned features.
    # D2 has already checked names/shapes. Every approved field is retained.
    if base.endswith("size_xyz_m"):
        return "dimensions"
    if base.endswith("_xyz_m"):
        return "point"
    if base.endswith("center_xy_m"):
        return "line"
    if base.endswith("direction_world_xy") or base.endswith("direction_world_xyz"):
        return "vector"
    if base.endswith("xyyaw"):
        return "planar_pose"
    if base.endswith("bounds_xy_m") or base.endswith("height_m") or base.endswith("heights_m") or base.endswith("heights_world_m") or base.endswith("_x_m"):
        return "planes"
    if base.endswith("yaw_rad"):
        return "heading"
    if base.endswith("yaw_deg") or base.endswith("pitch_deg"):
        return "angle_constraint"
    raise ValueError(f"Approved spatial quantity has no physical codec: {base}")


def _task_quantity(feature, scope, frame, config):
    first, _, base = feature.name.partition(".")
    spatial = first in _PHYSICAL_FRAMES | {"unframed", "body_local"}
    if not spatial:
        return DerivedQuantity(scope, feature.name, None, "intrinsic", "intrinsic",
                               feature.values if feature.valid else None, feature.valid)
    kind = _geometry_kind(base)
    if kind == "dimensions":
        # Intrinsic lengths require neither centroid nor a source-axis bridge.
        # Preserve source validity, including invalid unframed measurements.
        return DerivedQuantity(scope, feature.name, first, "intrinsic", kind,
                               feature.values if feature.valid else None, feature.valid)
    bridge = _source_bridge(first, config)
    output_frame = "robot_centroid"
    # Body-local task positions have no owning module or certified origin.
    usable = feature.valid and bridge is not None
    needs_origin = kind in {"point", "line", "planes", "planar_pose"}
    usable = usable and (frame.valid or not needs_origin)
    if not usable:
        return DerivedQuantity(scope, feature.name, first, output_frame, kind, None, False)
    values = _finite(feature.values)
    source_rotation, source_translation = bridge
    axes = _rotation(frame.rotation_source_from_axes).T
    basis = axes @ source_rotation

    def point(p):
        return transform_point(source_rotation @ p + source_translation, frame)

    def line(x, y):
        # p(z)=(x,y,0)+z*(0,0,1). z=0 is a line parameter origin,
        # not an invented measured object height.
        return (*point(np.array((x,y,0.))), *_flat(basis[:,2]))

    def plane(normal, offset):
        normal = np.array(normal)
        n_graph = source_rotation @ normal
        d = offset + n_graph @ (source_translation-np.array(frame.origin))
        return (*_flat(axes @ n_graph), float(d))

    if kind == "point":
        derived = point(np.array(values))
    elif kind == "vector":
        vector = values if len(values) == 3 else (*values, 0.)
        derived = _flat(basis @ vector)
    elif kind == "line":
        derived = line(*values)
    elif kind == "planar_pose":
        x,y,yaw = values
        derived = (*line(x,y), *_flat(basis @ (math.cos(yaw),math.sin(yaw),0.)))
    elif kind == "heading":
        derived = _flat(basis @ (math.cos(values[0]),math.sin(values[0]),0.))
    elif kind == "planes":
        if base.endswith("bounds_xy_m"):
            xmin,xmax,ymin,ymax = values
            if xmin > xmax or ymin > ymax:
                raise ValueError("Invalid physical bounds")
            derived = (*plane((1.,0.,0.),xmin), *plane((1.,0.,0.),xmax),
                       *plane((0.,1.,0.),ymin), *plane((0.,1.,0.),ymax))
        else:
            normal = (1.,0.,0.) if base.endswith("_x_m") else (0.,0.,1.)
            derived = tuple(v for height in values for v in plane(normal,height))
    elif kind == "angle_constraint":
        derived = (math.radians(values[0]), *_flat(basis))
    else:
        raise ValueError(f"Unsupported physical codec: {kind}")
    return DerivedQuantity(scope, feature.name, first, output_frame, kind, _finite(derived), True)


def transform_inputs(inputs: PolicyInputs, frame_config: PhysicalFrameConfig) -> FrameGeometryView:
    """Transform G/c/o consistently, preserving every approved quantity.

    Revalidate the D2 boundary, including illicit low-level mutations of frozen
    input objects. No raw mapping/label/source metadata interface exists here.
    """
    _validate_selected_inputs(inputs)
    frame = physical_frame(inputs.graph, frame_config)
    modules = []
    for module in inputs.graph.modules:
        linear, angular = transform_twist(module.linear_velocity, module.angular_velocity, frame)
        modules.append(ModuleGeometry(module.ref,
            transform_point(module.position, frame, valid=module.pose_valid),
            transform_orientation(module.orientation_xyzw, frame, valid=module.pose_valid),
            linear, angular, module.pose_valid and frame.valid))
    quantities = tuple(_task_quantity(feature, scope, frame, frame_config)
        for scope, features in (("c_static", inputs.context.static_geometry),
                                ("o_dynamic", inputs.observation.features)) for feature in features)
    local = []
    for connector in inputs.graph.connectors:
        for feature in connector.features:
            local.append(DerivedQuantity("g_current", feature.name, "module_body", "module_body",
                "point" if feature.name == "position_body_m" else "vector",
                feature.values if feature.valid else None, feature.valid, connector.ref))
    for edge in inputs.graph.connections:
        for feature in edge.features:
            intrinsic = feature.name == "clocking_quarter_turns"
            local_frame = "connection_intrinsic" if intrinsic else "endpoint_a_body"
            if feature.valid and feature.name == "relative_orientation_xyzw":
                quaternion_to_matrix(feature.values)  # Validate, preserve XYZW.
            kind = "intrinsic" if intrinsic else "quaternion" if feature.name == "relative_orientation_xyzw" else "point"
            local.append(DerivedQuantity("g_current", feature.name, local_frame, local_frame,
                kind, feature.values if feature.valid else None, feature.valid, (edge.endpoint_a,edge.endpoint_b)))
    for resource in inputs.graph.resources:
        for coordinate in resource.coordinates:
            for name in ("position", "velocity"):
                value = getattr(coordinate, name)
                local.append(DerivedQuantity("g_current", name, "resource_local", "resource_local",
                    "coordinate", None if value is None else (value,), value is not None,
                    resource.ref, coordinate.coordinate_id))
    return FrameGeometryView(frame, tuple(modules), quantities, tuple(local),
                             inputs.graph.connectors, inputs.graph.connections, inputs.graph.resources)
