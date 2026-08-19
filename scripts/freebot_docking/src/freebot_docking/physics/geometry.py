from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite

import numpy as np
import numpy.typing as npt

from freebot_docking.config.geometry import ShellGeometry
from freebot_docking.physics.state import (
    MagnetState,
    ShellState,
    Vector3,
    as_vector3,
)

from freebot_docking.config.magnet import MagnetConfig

from freebot_docking.config.simulation import (
    AxisymmetricGridConfig,
)

@dataclass(frozen=True)
class ShellPairGeometry:
    """Instantaneous relative geometry of two spherical shells."""

    normal_first_to_second_world: Vector3
    center_distance_m: float
    signed_gap_m: float
    point_on_first_world: Vector3
    point_on_second_world: Vector3

    def __post_init__(self) -> None:
        center_distance = float(self.center_distance_m)
        signed_gap = float(self.signed_gap_m)

        if not isfinite(center_distance) or center_distance <= 0.0:
            raise ValueError("Center distance must be finite and positive")

        if not isfinite(signed_gap):
            raise ValueError("Signed gap must be finite")

        object.__setattr__(
            self,
            "normal_first_to_second_world",
            as_vector3(self.normal_first_to_second_world),
        )
        object.__setattr__(
            self,
            "center_distance_m",
            center_distance,
        )
        object.__setattr__(
            self,
            "signed_gap_m",
            signed_gap,
        )
        object.__setattr__(
            self,
            "point_on_first_world",
            as_vector3(self.point_on_first_world),
        )
        object.__setattr__(
            self,
            "point_on_second_world",
            as_vector3(self.point_on_second_world),
        )


def compute_shell_pair_geometry(
    first_state: ShellState,
    first_geometry: ShellGeometry,
    second_state: ShellState,
    second_geometry: ShellGeometry,
) -> ShellPairGeometry:
    """Compute signed gap, normal and nearest nominal surface points."""

    center_delta = second_state.center_world - first_state.center_world
    center_distance = float(np.linalg.norm(center_delta))

    if center_distance <= 1.0e-12:
        raise ValueError(
            "Cannot compute shell-pair geometry for coincident centers"
        )

    normal = center_delta / center_distance

    point_on_first = (
        first_state.center_world
        + first_geometry.outer_radius_m * normal
    )
    point_on_second = (
        second_state.center_world
        - second_geometry.outer_radius_m * normal
    )

    signed_gap = (
        center_distance
        - first_geometry.outer_radius_m
        - second_geometry.outer_radius_m
    )

    return ShellPairGeometry(
        normal_first_to_second_world=normal,
        center_distance_m=center_distance,
        signed_gap_m=signed_gap,
        point_on_first_world=point_on_first,
        point_on_second_world=point_on_second,
    )

def compute_magnet_active_face_center(
    magnet_state: MagnetState,
    magnet_config: MagnetConfig,
) -> Vector3:
    """Return the center of the active circular pole face."""

    return (
        magnet_state.center_world
        + magnet_config.half_length_m * magnet_state.axis_world
    )


@dataclass(frozen=True)
class MagnetInnerShellGeometry:
    """Geometry between the active magnet face and its own inner shell."""

    face_center_world: Vector3
    point_on_inner_shell_world: Vector3
    inner_shell_normal_world: Vector3
    axial_gap_m: float
    radial_gap_m: float
    alignment_cosine: float

    def __post_init__(self) -> None:
        axial_gap = float(self.axial_gap_m)
        radial_gap = float(self.radial_gap_m)
        alignment = float(self.alignment_cosine)

        if not isfinite(axial_gap) or axial_gap < 0.0:
            raise ValueError("Axial magnet-shell gap must be finite and non-negative")

        if not isfinite(radial_gap) or radial_gap < 0.0:
            raise ValueError("Radial magnet-shell gap must be finite and non-negative")

        if not isfinite(alignment) or not 0.0 <= alignment <= 1.0:
            raise ValueError("Magnet-shell alignment must lie in [0, 1]")

        object.__setattr__(
            self,
            "face_center_world",
            as_vector3(self.face_center_world),
        )
        object.__setattr__(
            self,
            "point_on_inner_shell_world",
            as_vector3(self.point_on_inner_shell_world),
        )
        object.__setattr__(
            self,
            "inner_shell_normal_world",
            as_vector3(self.inner_shell_normal_world),
        )
        object.__setattr__(self, "axial_gap_m", axial_gap)
        object.__setattr__(self, "radial_gap_m", radial_gap)
        object.__setattr__(self, "alignment_cosine", alignment)


@dataclass(frozen=True)
class AxisymmetricGrid:
    """Cell faces and centers of an axisymmetric finite-volume grid."""

    config: AxisymmetricGridConfig

    radial_faces_m: npt.NDArray[np.float64] = field(
        init=False,
        repr=False,
    )
    radial_centers_m: npt.NDArray[np.float64] = field(
        init=False,
        repr=False,
    )
    axial_faces_m: npt.NDArray[np.float64] = field(
        init=False,
        repr=False,
    )
    axial_centers_m: npt.NDArray[np.float64] = field(
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        radial_faces = np.linspace(
            0.0,
            self.config.radial_max_m,
            self.config.radial_cells + 1,
            dtype=np.float64,
        )
        axial_faces = np.linspace(
            self.config.axial_min_m,
            self.config.axial_max_m,
            self.config.axial_cells + 1,
            dtype=np.float64,
        )

        radial_centers = 0.5 * (
            radial_faces[:-1] + radial_faces[1:]
        )
        axial_centers = 0.5 * (
            axial_faces[:-1] + axial_faces[1:]
        )

        for array in (
            radial_faces,
            radial_centers,
            axial_faces,
            axial_centers,
        ):
            array.setflags(write=False)

        object.__setattr__(
            self,
            "radial_faces_m",
            radial_faces,
        )
        object.__setattr__(
            self,
            "radial_centers_m",
            radial_centers,
        )
        object.__setattr__(
            self,
            "axial_faces_m",
            axial_faces,
        )
        object.__setattr__(
            self,
            "axial_centers_m",
            axial_centers,
        )

    @property
    def cell_shape(self) -> tuple[int, int]:
        """Array shape ordered as axial index, radial index."""
        return (
            self.config.axial_cells,
            self.config.radial_cells,
        )

    @property
    def cell_count(self) -> int:
        return (
            self.config.axial_cells
            * self.config.radial_cells
        )
    

def compute_magnet_inner_shell_geometry(
    shell_state: ShellState,
    shell_geometry: ShellGeometry,
    magnet_state: MagnetState,
    magnet_config: MagnetConfig,
) -> MagnetInnerShellGeometry:
    """Intersect the active magnet axis with the inner spherical surface."""

    face_center = compute_magnet_active_face_center(
        magnet_state,
        magnet_config,
    )

    face_from_shell_center = (
        face_center - shell_state.center_world
    )
    face_radius = float(np.linalg.norm(face_from_shell_center))
    inner_radius = shell_geometry.inner_radius_m

    if face_radius >= inner_radius:
        raise ValueError("Magnet active face must lie inside the shell")

    outward_projection = float(
        np.dot(
            face_from_shell_center,
            magnet_state.axis_world,
        )
    )

    if outward_projection <= 0.0:
        raise ValueError(
            "Magnet active axis must point toward the nearest inner shell"
        )

    discriminant = (
        outward_projection**2
        + inner_radius**2
        - face_radius**2
    )

    axial_gap = (
        -outward_projection
        + float(np.sqrt(max(discriminant, 0.0)))
    )

    point_on_inner_shell = (
        face_center
        + axial_gap * magnet_state.axis_world
    )

    inner_shell_normal = (
        point_on_inner_shell - shell_state.center_world
    ) / inner_radius

    alignment = float(
        np.clip(
            np.dot(
                magnet_state.axis_world,
                inner_shell_normal,
            ),
            0.0,
            1.0,
        )
    )

    return MagnetInnerShellGeometry(
        face_center_world=face_center,
        point_on_inner_shell_world=point_on_inner_shell,
        inner_shell_normal_world=inner_shell_normal,
        axial_gap_m=axial_gap,
        radial_gap_m=inner_radius - face_radius,
        alignment_cosine=alignment,
    )