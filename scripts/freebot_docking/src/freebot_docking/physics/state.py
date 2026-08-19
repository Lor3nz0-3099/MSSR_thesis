from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


Vector3 = npt.NDArray[np.float64]


def as_vector3(value: npt.ArrayLike) -> Vector3:
    """Return an independent finite three-dimensional vector."""
    vector = np.array(value, dtype=np.float64, copy=True)

    if vector.shape != (3,):
        raise ValueError(
            f"Expected a vector with shape (3,), received {vector.shape}"
        )

    if not np.all(np.isfinite(vector)):
        raise ValueError("Vector contains non-finite values")

    return vector

def rigid_point_velocity(
    com_world: npt.ArrayLike,
    linear_velocity_world: npt.ArrayLike,
    angular_velocity_world: npt.ArrayLike,
    point_world: npt.ArrayLike,
) -> Vector3:
    """Return the velocity of a point fixed to a rigid body."""
    com = as_vector3(com_world)
    linear_velocity = as_vector3(linear_velocity_world)
    angular_velocity = as_vector3(angular_velocity_world)
    point = as_vector3(point_world)

    return (
        linear_velocity
        + np.cross(
            angular_velocity,
            point - com,
        )
    )


@dataclass(frozen=True)
class ShellState:
    """Instantaneous dynamic state of a spherical shell."""

    center_world: Vector3
    com_world: Vector3
    linear_velocity_world: Vector3
    angular_velocity_world: Vector3

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "center_world",
            as_vector3(self.center_world),
        )
        object.__setattr__(
            self,
            "com_world",
            as_vector3(self.com_world),
        )
        object.__setattr__(
            self,
            "linear_velocity_world",
            as_vector3(self.linear_velocity_world),
        )
        object.__setattr__(
            self,
            "angular_velocity_world",
            as_vector3(self.angular_velocity_world),
        )

    def velocity_at(self, point_world: npt.ArrayLike) -> Vector3:
        return rigid_point_velocity(
            com_world=self.com_world,
            linear_velocity_world=self.linear_velocity_world,
            angular_velocity_world=self.angular_velocity_world,
            point_world=point_world,
        )
    
@dataclass(frozen=True)
class MagnetState:
    """Instantaneous state of a magnet fixed to its carrier body."""

    center_world: Vector3
    axis_world: Vector3
    carrier_com_world: Vector3
    carrier_linear_velocity_world: Vector3
    carrier_angular_velocity_world: Vector3

    def __post_init__(self) -> None:
        center = as_vector3(self.center_world)
        axis = as_vector3(self.axis_world)
        carrier_com = as_vector3(self.carrier_com_world)
        linear_velocity = as_vector3(
            self.carrier_linear_velocity_world
        )
        angular_velocity = as_vector3(
            self.carrier_angular_velocity_world
        )

        axis_norm = float(np.linalg.norm(axis))

        if not np.isclose(
            axis_norm,
            1.0,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError("Magnet world axis must be a unit vector")

        object.__setattr__(self, "center_world", center)
        object.__setattr__(self, "axis_world", axis)
        object.__setattr__(self, "carrier_com_world", carrier_com)
        object.__setattr__(
            self,
            "carrier_linear_velocity_world",
            linear_velocity,
        )
        object.__setattr__(
            self,
            "carrier_angular_velocity_world",
            angular_velocity,
        )

    def velocity_at(self, point_world: npt.ArrayLike) -> Vector3:
        return rigid_point_velocity(
            com_world=self.carrier_com_world,
            linear_velocity_world=self.carrier_linear_velocity_world,
            angular_velocity_world=self.carrier_angular_velocity_world,
            point_world=point_world,
        )