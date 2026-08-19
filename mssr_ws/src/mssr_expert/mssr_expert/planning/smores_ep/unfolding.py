"""Planar unfolding of a rooted SMORES-EP target configuration."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

from mssr_expert.planning.smores_ep.rooting import (
    RootedSmoresEdge,
    RootedSmoresTree,
)
from mssr_expert.planning.smores_ep.topology import VALID_FACES


FACE_ANGLE_RAD = {
    "TOP": 0.0,
    "LEFT": math.pi / 2.0,
    "BOTTOM": math.pi,
    "RIGHT": -math.pi / 2.0,
}


class PlanarUnfoldingError(ValueError):
    """Raised when a target topology cannot be unfolded on the plane."""


@dataclass(frozen=True)
class PlanarPose:
    """Planar position and orientation of one module."""

    x_m: float
    y_m: float
    yaw_rad: float

    def __post_init__(self) -> None:
        values = (self.x_m, self.y_m, self.yaw_rad)

        if not all(
            isinstance(value, int | float)
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            for value in values
        ):
            raise PlanarUnfoldingError(
                "A planar pose must contain finite numeric values."
            )


@dataclass(frozen=True)
class PlanarModuleGeometry:
    """Distances from the module center to its four docking planes."""

    top_offset_m: float = 0.043771
    bottom_offset_m: float = 0.033999
    left_offset_m: float = 0.043462
    right_offset_m: float = 0.043448

    def __post_init__(self) -> None:
        offsets = (
            self.top_offset_m,
            self.bottom_offset_m,
            self.left_offset_m,
            self.right_offset_m,
        )

        if not all(
            isinstance(value, int | float)
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and value > 0.0
            for value in offsets
        ):
            raise PlanarUnfoldingError(
                "Every face offset must be a positive finite number."
            )

    def offset_for_face(self, face: str) -> float:
        """Return the center-to-docking-plane distance for one face."""

        if face == "TOP":
            return self.top_offset_m

        if face == "BOTTOM":
            return self.bottom_offset_m

        if face == "LEFT":
            return self.left_offset_m

        if face == "RIGHT":
            return self.right_offset_m

        raise PlanarUnfoldingError(
            f"Unknown SMORES-EP face {face!r}."
        )


@dataclass(frozen=True)
class UnfoldedPlanarConfiguration:
    """Planar target poses generated from a rooted configuration."""

    root_id: str
    poses_by_vertex: Mapping[str, PlanarPose]


def unfold_tree_on_plane(
    tree: RootedSmoresTree,
    geometry: PlanarModuleGeometry | None = None,
    root_pose: PlanarPose | None = None,
    uniqueness_tolerance_m: float = 1e-6,
) -> UnfoldedPlanarConfiguration:
    """Compute every target module pose in breadth-first order."""

    if geometry is None:
        geometry = PlanarModuleGeometry()

    if root_pose is None:
        root_pose = PlanarPose(
            x_m=0.0,
            y_m=0.0,
            yaw_rad=0.0,
        )

    if (
        not math.isfinite(uniqueness_tolerance_m)
        or uniqueness_tolerance_m <= 0.0
    ):
        raise PlanarUnfoldingError(
            "uniqueness_tolerance_m must be positive and finite."
        )

    if tree.root_id not in tree.vertex_ids:
        raise PlanarUnfoldingError(
            f"Root {tree.root_id!r} is not present in the tree."
        )

    poses_by_vertex: dict[str, PlanarPose] = {
        tree.root_id: PlanarPose(
            x_m=root_pose.x_m,
            y_m=root_pose.y_m,
            yaw_rad=normalize_angle(root_pose.yaw_rad),
        )
    }

    pending_edges = list(tree.edges)

    while pending_edges:
        progress_was_made = False
        remaining_edges: list[RootedSmoresEdge] = []

        for edge in pending_edges:
            parent_pose = poses_by_vertex.get(edge.parent_vertex)

            if parent_pose is None:
                remaining_edges.append(edge)
                continue

            if edge.child_vertex in poses_by_vertex:
                raise PlanarUnfoldingError(
                    f"Vertex {edge.child_vertex!r} received more than one pose."
                )

            child_pose = _compute_child_pose(
                parent_pose=parent_pose,
                parent_face=edge.parent_face,
                child_face=edge.child_face,
                geometry=geometry,
            )

            _ensure_unique_center(
                child_vertex=edge.child_vertex,
                child_pose=child_pose,
                existing_poses=poses_by_vertex,
                tolerance_m=uniqueness_tolerance_m,
            )

            poses_by_vertex[edge.child_vertex] = child_pose
            progress_was_made = True

        if not progress_was_made:
            unresolved_children = sorted(
                edge.child_vertex for edge in remaining_edges
            )

            raise PlanarUnfoldingError(
                "The rooted edges are not ordered consistently. "
                f"Unresolved children: {unresolved_children}."
            )

        pending_edges = remaining_edges

    missing_vertices = set(tree.vertex_ids) - set(poses_by_vertex)

    if missing_vertices:
        raise PlanarUnfoldingError(
            "No planar pose was generated for vertices "
            f"{sorted(missing_vertices)}."
        )

    return UnfoldedPlanarConfiguration(
        root_id=tree.root_id,
        poses_by_vertex=dict(poses_by_vertex),
    )


def normalize_angle(angle_rad: float) -> float:
    """Normalize an angle to the interval [-pi, pi)."""

    if not math.isfinite(angle_rad):
        raise PlanarUnfoldingError(
            "Cannot normalize a non-finite angle."
        )

    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi


def _compute_child_pose(
    parent_pose: PlanarPose,
    parent_face: str,
    child_face: str,
    geometry: PlanarModuleGeometry,
) -> PlanarPose:
    """Place a child so that its selected face meets the parent face."""

    if parent_face not in VALID_FACES:
        raise PlanarUnfoldingError(
            f"Invalid parent face {parent_face!r}."
        )

    if child_face not in VALID_FACES:
        raise PlanarUnfoldingError(
            f"Invalid child face {child_face!r}."
        )

    parent_face_angle = FACE_ANGLE_RAD[parent_face]
    child_face_angle = FACE_ANGLE_RAD[child_face]

    child_yaw = normalize_angle(
        parent_pose.yaw_rad
        + parent_face_angle
        + math.pi
        - child_face_angle
    )

    parent_face_local = _face_center_local(
        face=parent_face,
        geometry=geometry,
    )
    child_face_local = _face_center_local(
        face=child_face,
        geometry=geometry,
    )

    parent_face_world = _rotate_vector(
        vector=parent_face_local,
        angle_rad=parent_pose.yaw_rad,
    )
    child_face_world = _rotate_vector(
        vector=child_face_local,
        angle_rad=child_yaw,
    )

    child_x = (
        parent_pose.x_m
        + parent_face_world[0]
        - child_face_world[0]
    )
    child_y = (
        parent_pose.y_m
        + parent_face_world[1]
        - child_face_world[1]
    )

    return PlanarPose(
        x_m=child_x,
        y_m=child_y,
        yaw_rad=child_yaw,
    )


def _face_center_local(
    face: str,
    geometry: PlanarModuleGeometry,
) -> tuple[float, float]:
    """Return the local XY position of a docking-face center."""

    angle = FACE_ANGLE_RAD[face]
    offset = geometry.offset_for_face(face)

    return (
        offset * math.cos(angle),
        offset * math.sin(angle),
    )


def _rotate_vector(
    vector: tuple[float, float],
    angle_rad: float,
) -> tuple[float, float]:
    """Rotate a two-dimensional vector."""

    cosine = math.cos(angle_rad)
    sine = math.sin(angle_rad)

    return (
        cosine * vector[0] - sine * vector[1],
        sine * vector[0] + cosine * vector[1],
    )


def _ensure_unique_center(
    child_vertex: str,
    child_pose: PlanarPose,
    existing_poses: Mapping[str, PlanarPose],
    tolerance_m: float,
) -> None:
    """Reject an unfolding where two module centers coincide."""

    for other_vertex, other_pose in existing_poses.items():
        distance = math.hypot(
            child_pose.x_m - other_pose.x_m,
            child_pose.y_m - other_pose.y_m,
        )

        if distance <= tolerance_m:
            raise PlanarUnfoldingError(
                f"Vertices {child_vertex!r} and {other_vertex!r} "
                "occupy the same planar location."
            )