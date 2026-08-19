from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from freebot_docking.physics.state import Vector3, as_vector3

@dataclass(frozen=True)
class Wrench:
    """A wrench represents a force and a torque applied to a rigid body."""
    force: Vector3
    torque: Vector3
    reference_point: Vector3

    def __post_init__(self) -> None:
        object.__setattr__(self, "force", as_vector3(self.force))
        object.__setattr__(self, "torque", as_vector3(self.torque))
        object.__setattr__(
            self,
            "reference_point",
            as_vector3(self.reference_point),
        )

    @classmethod
    def zero(cls, reference_point: npt.ArrayLike) -> "Wrench":
        """Create a wrench with zero force and torque at the given reference point."""
        return cls(
            force=np.zeros(3),
            torque=np.zeros(3),
            reference_point=as_vector3(reference_point),
        )
    
    @classmethod
    def from_force_at_point(cls, force: npt.ArrayLike, application_point: npt.ArrayLike, reference_point: npt.ArrayLike,) -> "Wrench":
        """Create a wrench from a force applied at a specific point."""
        force_vector = as_vector3(force)
        point = as_vector3(application_point)
        reference = as_vector3(reference_point)

        torque = np.cross(point - reference, force_vector)
        return cls(force=force_vector, torque=torque, reference_point=reference)
    
    def expressed_at(self, new_reference_point: npt.ArrayLike) -> "Wrench":
        """Express the wrench at a new reference point."""
        new_reference = as_vector3(new_reference_point)

        shifted_torque = (self.torque + np.cross(self.reference_point - new_reference, self.force ))

        return Wrench(force=self.force.copy(), torque=shifted_torque, reference_point=new_reference)
    
    def __add__(self, other: "Wrench") -> "Wrench":
        """Add two wrenches expressed at the same reference point."""
        other_at_same_point = other.expressed_at(self.reference_point)

        return Wrench(
            force=self.force + other_at_same_point.force,
            torque=self.torque + other_at_same_point.torque,
            reference_point=self.reference_point.copy(),
        )
    
    def __neg__(self) -> "Wrench":
        """Negate the wrench."""
        return Wrench(
            force=-self.force,
            torque=-self.torque,
            reference_point=self.reference_point.copy(),
        )