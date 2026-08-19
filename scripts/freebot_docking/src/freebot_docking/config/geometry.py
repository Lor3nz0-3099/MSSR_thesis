from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class WheelRadialComplianceConfig:
    """Limited tread-normal wheel travel relative to the rigid carrier.

    The compliant degree of freedom is internal to the module: a D6 joint
    retains the driven wheel rotation while allowing a short translation in
    the wheel's radial plane.  Its force drive represents the elastic wheel
    support and always has an equal-and-opposite reaction on the carrier.
    """

    enabled: bool = True
    inward_travel_m: float = 0.0006
    outward_travel_m: float = 0.0021
    rest_position_m: float = 0.0017
    stiffness_n_per_m: float = 3_500.0
    damping_n_s_per_m: float = 12.0
    max_force_n: float = 15.0
    mount_mass_kg: float = 0.004

    def __post_init__(self) -> None:
        values = (
            self.inward_travel_m,
            self.outward_travel_m,
            self.rest_position_m,
            self.stiffness_n_per_m,
            self.damping_n_s_per_m,
            self.max_force_n,
            self.mount_mass_kg,
        )
        if not all(isfinite(float(value)) for value in values):
            raise ValueError("Wheel radial-compliance values must be finite")
        if self.inward_travel_m < 0.0 or self.outward_travel_m <= 0.0:
            raise ValueError(
                "Wheel radial travel must be non-negative and non-zero"
            )
        if not (
            -self.inward_travel_m
            <= self.rest_position_m
            <= self.outward_travel_m
        ):
            raise ValueError("Wheel radial rest position must lie inside its travel")
        if self.stiffness_n_per_m <= 0.0:
            raise ValueError("Wheel radial stiffness must be positive")
        if (
            self.damping_n_s_per_m < 0.0
            or self.max_force_n <= 0.0
            or self.mount_mass_kg <= 0.0
        ):
            raise ValueError("Wheel radial damping and force limit are invalid")


@dataclass(frozen=True)
class ShellGeometry:
    """Geometric properties of the spherical ferromagnetic shell"""

    outer_radius_m: float = 0.0633472
    inner_radius_m: float = 0.0613472
    center_from_body_origin_m: tuple[float, float, float] = (
        0.00155433,
        0.00087740,
        0.00466804
    )

    def __post_init__(self) -> None:
       outer_radius = float(self.outer_radius_m)
       inner_radius = float(self.inner_radius_m)
       center_offset = tuple(
             float(component)
             for component in self.center_from_body_origin_m
       )

       if not isfinite(outer_radius) or not isfinite(inner_radius):
             raise ValueError("Shell radii must be finite")
       
       if outer_radius <= 0.0 or inner_radius <= 0.0:
             raise ValueError("Shell radii must be positive")
       
       if inner_radius >= outer_radius:
             raise ValueError(
                   "Shell inner radius must be smaller than outer radius"
             )
       
       if len(center_offset) != 3:
             raise ValueError("Shell center offset must contain three components")
       
       if not all(isfinite(component) for component in center_offset):
             raise ValueError("Shell center offset components must be finite ")
       
       object.__setattr__(self, "outer_radius_m", outer_radius)
       object.__setattr__(self, "inner_radius_m", inner_radius)
       object.__setattr__(
            self,
            "center_from_body_origin_m",
            center_offset,
        )

    @property
    def thickness_m(self) -> float:
      return self.outer_radius_m - self.inner_radius_m


@dataclass(frozen=True)
class RunningGearGeometry:
    """CAD-fitted collision dimensions for wheels and caster balls."""

    tire_outer_radius_m: float = 0.016000287
    # The CAD radius describes the tire at its loaded running diameter.  The
    # collision envelope represents the unloaded rubber and is therefore
    # slightly larger.  Its overlap with the inner shell is resolved by the
    # compliant wheel material as tire compression, not as rigid penetration.
    tire_precompression_m: float = 0.0009
    tire_half_width_m: float = 0.003001301
    tire_center_axial_offset_m: float = 0.00105277
    tire_axis_tilt_deg: float = 0.16702
    caster_ball_radius_m: float = 0.004650
    # Effective elastic envelope of the caster/shell contact.  The ball itself
    # remains at its CAD radius; this small excess represents local shell,
    # mount and contact compliance so two caster contacts do not form a rigid
    # pivot that abruptly unloads both drive wheels.
    caster_precompression_m: float = 0.0001
    wheel_nominal_clearance_m: float = 0.0
    caster_nominal_clearance_m: float = 0.0
    wheel_contact_offset_m: float = 0.0005
    caster_contact_offset_m: float = 0.00025

    def __post_init__(self) -> None:
        values = (
            self.tire_outer_radius_m,
            self.tire_precompression_m,
            self.tire_half_width_m,
            self.tire_center_axial_offset_m,
            self.tire_axis_tilt_deg,
            self.caster_ball_radius_m,
            self.caster_precompression_m,
            self.wheel_nominal_clearance_m,
            self.caster_nominal_clearance_m,
            self.wheel_contact_offset_m,
            self.caster_contact_offset_m,
        )
        if not all(isfinite(float(value)) for value in values):
            raise ValueError("Running-gear dimensions must be finite")
        if self.tire_outer_radius_m <= 0.0 or self.tire_half_width_m <= 0.0:
            raise ValueError("Tire dimensions must be positive")
        if self.tire_precompression_m < 0.0:
            raise ValueError("Tire precompression must be non-negative")
        if self.tire_precompression_m >= self.tire_outer_radius_m:
            raise ValueError("Tire precompression must be smaller than its radius")
        if self.caster_ball_radius_m <= 0.0:
            raise ValueError("Caster radius must be positive")
        if self.caster_precompression_m < 0.0:
            raise ValueError("Caster precompression must be non-negative")
        if self.caster_precompression_m >= self.caster_ball_radius_m:
            raise ValueError(
                "Caster precompression must be smaller than its radius"
            )
        if self.wheel_nominal_clearance_m < 0.0:
            raise ValueError("Wheel nominal clearance must be non-negative")
        if self.caster_nominal_clearance_m < 0.0:
            raise ValueError("Caster nominal clearance must be non-negative")
        if self.wheel_contact_offset_m < 0.0 or self.caster_contact_offset_m < 0.0:
            raise ValueError("Contact offsets must be non-negative")
        if (
            self.caster_nominal_clearance_m > 0.0
            and self.caster_contact_offset_m
            >= self.caster_nominal_clearance_m
        ):
            raise ValueError(
                "Caster contact offset must remain below its nominal clearance"
            )

    @property
    def tire_collision_radius_m(self) -> float:
        """Unloaded tire envelope used by the compliant collision proxy."""

        return self.tire_outer_radius_m + self.tire_precompression_m

    @property
    def caster_collision_radius_m(self) -> float:
        """Effective caster envelope used by compliant contact."""

        return self.caster_ball_radius_m + self.caster_precompression_m
