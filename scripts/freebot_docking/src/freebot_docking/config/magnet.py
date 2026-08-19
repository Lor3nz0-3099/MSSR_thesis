from __future__ import annotations
from bisect import bisect_left
from dataclasses import dataclass
from math import isclose, isfinite, pi, sqrt



@dataclass(frozen=True)
class MagnetConfig:
    """Cylindrical equivalent model of the FreeBOT permanent magnet."""

    radius_m: float = 0.010
    length_m: float = 0.010
    remanence_t: float = 1.47
    active_axis_local: tuple[float, float, float] = (0.0, 0.0, -1.0)
    recoil_relative_permeability: float = 1.05

    def __post_init__(self) -> None:
        radius = float(self.radius_m)
        length = float(self.length_m)
        remanence = float(self.remanence_t)
        active_axis = tuple(float(component) for component in self.active_axis_local)
        recoil_relative_permeability = float(self.recoil_relative_permeability)

        if not isfinite(radius) or not isfinite(length):
            raise ValueError("Magnet dimensions must be finite")
        
        if radius <= 0.0 or length <= 0.0:
            raise ValueError("Magnet dimensions must be positive")
        
        if not isfinite(remanence) or remanence <= 0.0:
            raise ValueError("Magnet remanence must be finite and positive")
        
        if len(active_axis) != 3:
            raise ValueError("Magnet active axis must contain three components")

        if not all(isfinite(component) for component in active_axis):
            raise ValueError("Magnet active axis must be finite")
        
        axis_norm = sqrt(sum(component * component for component in active_axis))
        
        if (not isfinite(recoil_relative_permeability) or recoil_relative_permeability <= 0.0):
            raise ValueError("Magnet recoil relative permeability must be finite and positive")

        if not isclose(axis_norm, 1.0, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError("Magnet active axis must be a unit vector")

        object.__setattr__(self, "radius_m", radius)
        object.__setattr__(self, "length_m", length)
        object.__setattr__(self, "remanence_t", remanence)
        object.__setattr__(self, "active_axis_local", active_axis)
        object.__setattr__(self, "recoil_relative_permeability", recoil_relative_permeability)
    @property
    def diameter_m(self) -> float:
        return 2.0 * self.radius_m

    @property
    def pole_area_m2(self) -> float:
        return pi * self.radius_m**2

    @property
    def volume_m3(self) -> float:
        return self.pole_area_m2 * self.length_m

    @property
    def half_length_m(self) -> float:
        return 0.5 * self.length_m
    
@dataclass(frozen=True)
class TabulatedBHCurve:
    """Monotonic first-magnetization curve for a ferromagnetic material"""

    field_strength_a_per_m: tuple[float, ...]
    flux_density_t: tuple[float, ...]

    def __post_init__(self) -> None:
        field_strength = tuple(
            float(value)
            for value in self.field_strength_a_per_m
        )
        flux_density = tuple(
            float(value)
            for value in self.flux_density_t
        )

        if len(field_strength) != len(flux_density):
            raise ValueError("B-H arrays must have the same length")

        if len(field_strength) < 2:
            raise ValueError("B-H curve must contain at least two points")

        if not all(isfinite(value) for value in field_strength):
            raise ValueError("B-H field strengths must be finite")

        if not all(isfinite(value) for value in flux_density):
            raise ValueError("B-H flux densities must be finite")

        if field_strength[0] != 0.0 or flux_density[0] != 0.0:
            raise ValueError("B-H curve must start at the origin")
        
        if any(
            right <= left
            for left, right in zip(
                field_strength,
                field_strength[1:],
            )
        ):
            raise ValueError(
                "B-H field strengths must be strictly increasing"
            )

        if any(
            right <= left
            for left, right in zip(
                flux_density,
                flux_density[1:],
            )
        ):
            raise ValueError(
                "B-H flux densities must be strictly increasing"
            )

        object.__setattr__(
            self,
            "field_strength_a_per_m",
            field_strength,
        )
        object.__setattr__(
            self,
            "flux_density_t",
            flux_density,
        )
    
    def flux_density_for_field_strength_t(
            self,
            field_strength_a_per_m: float,
    ) -> float:
        
        """Interpolate the flux density for a given field strength."""
        field_strength = float(field_strength_a_per_m)

        if not isfinite(field_strength) or field_strength < 0.0:
            raise ValueError(
                "Field strength must be finite and non-negative"
            )

        if field_strength > self.field_strength_a_per_m[-1]:
            raise ValueError(
                "Field strength lies outside the tabulated B-H curve"
            )

        right_index = bisect_left(
           self.field_strength_a_per_m,
           field_strength,
       )
    
        if right_index == 0:
            return self.flux_density_t[0]
        
        if (
            self.field_strength_a_per_m[right_index]
            == field_strength
        ):
            return self.flux_density_t[right_index]

        left_index = right_index - 1

        left_h = self.field_strength_a_per_m[left_index]
        right_h = self.field_strength_a_per_m[right_index]
        left_b = self.flux_density_t[left_index]
        right_b = self.flux_density_t[right_index]

        fraction = (
            (field_strength - left_h)
            / (right_h - left_h)
        )

        return left_b + fraction * (right_b - left_b)
    

    def field_strength_for_flux_density_a_per_m(
            self,
            flux_density_t: float,
    ) -> float:
        """Interpolate the inverse material relation H(B) """
        flux_density = float(flux_density_t)

        if not isfinite(flux_density) or flux_density < 0.0:
            raise ValueError(
                "Flux density must be finite and non-negative"
            )

        if flux_density > self.flux_density_t[-1]:
            raise ValueError(
                "Flux density lies outside the tabulated B-H curve"
            )

        right_index = bisect_left(
           self.flux_density_t,
           flux_density,
       )
    
        if right_index == 0:
            return self.field_strength_a_per_m[0]
        
        if (
            self.flux_density_t[right_index]
            == flux_density
        ):
            return self.field_strength_a_per_m[right_index]

        left_index = right_index - 1

        left_b = self.flux_density_t[left_index]
        right_b = self.flux_density_t[right_index]
        left_h = self.field_strength_a_per_m[left_index]
        right_h = self.field_strength_a_per_m[right_index]

        fraction = (
            (flux_density - left_b)
            / (right_b - left_b)
        )

        return left_h + fraction * (right_h - left_h)
    
    def secant_reluctivity_m_per_h(
        self,
        flux_density_t: float,
    ) -> float:
        """
        Return secant reluctivity nu = H / B.

        At B = 0, use the slope of the first B-H segment.
        """
        flux_density = float(flux_density_t)

        field_strength = (
            self.field_strength_for_flux_density_a_per_m(
                flux_density
            )
        )

        if flux_density == 0.0:
            return (
                self.field_strength_a_per_m[1]
                / self.flux_density_t[1]
            )

        return field_strength / flux_density
    
    def secant_permeability_h_per_m(
        self,
        field_strength_a_per_m: float,
    ) -> float:
        """
        Return secant permeability mu = B / H.

        At H = 0, use the slope of the first B-H segment.
        """
        field_strength = float(field_strength_a_per_m)

        flux_density = (
            self.flux_density_for_field_strength_t(
                field_strength
            )
        )

        if field_strength == 0.0:
            return (
                self.flux_density_t[1]
                / self.field_strength_a_per_m[1]
            )

        return flux_density / field_strength


def pure_iron_bh_curve() -> TabulatedBHCurve:
    """
    Return the FEMM 4.2 first-magnetization curve for pure iron.

    Source: FEMM 4.2 material library, 21 April 2019 release.
    """
    return TabulatedBHCurve(
        field_strength_a_per_m=(
            0.0,
            13.8984,
            27.7967,
            42.3974,
            61.4157,
            82.3824,
            144.669,
            897.76,
            4581.74,
            17736.2,
            41339.3,
            68321.8,
            95685.5,
            123355.0,
            151083.0,
            178954.0,
            206825.0,
            234696.0,
            262568.0,
            290439.0,
            318310.0,
        ),
        flux_density_t=(
            0.0,
            0.227065,
            0.454130,
            0.681195,
            0.908260,
            1.135330,
            1.362390,
            1.589350,
            1.812360,
            2.010040,
            2.133160,
            2.199990,
            2.254790,
            2.299930,
            2.342510,
            2.378760,
            2.415010,
            2.451260,
            2.487500,
            2.523750,
            2.560000,
        ),
    )    
