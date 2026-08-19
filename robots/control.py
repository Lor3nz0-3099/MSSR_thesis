"""Control primitives shared by robot implementations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class PlanarVelocityCommand:
    """Planar velocity command for a robot moving on the ground plane."""

    vx: float = 0.0
    vy: float = 0.0
    yaw_rate: float = 0.0


class SphericalVelocityController:
    """Velocity controller for a spherical rigid body.

    The controller targets planar linear velocity and a matching rolling
    angular velocity. It only depends on the rigid body prim path, so the
    visual/collision asset can later be replaced by a modeled spherical robot.
    """

    def __init__(
        self,
        body_path: str,
        radius: float,
        max_linear_acceleration: float = 0.8,
        max_angular_acceleration: float = 1.6,
    ) -> None:
        """Initialize the controller for an existing rigid body prim."""
        from isaacsim.core.experimental.prims import RigidPrim

        if radius <= 0.0:
            raise ValueError("SphericalVelocityController requires a positive radius.")

        self._body = RigidPrim(paths=body_path)
        self._radius = radius
        self._max_linear_acceleration = max_linear_acceleration
        self._max_angular_acceleration = max_angular_acceleration

    def apply(self, command: PlanarVelocityCommand, dt: float) -> None:
        """Apply the desired planar velocity to the rigid body."""
        world_vx, world_vy = self._world_linear_velocity(command)
        angular_velocity = self._rolling_angular_velocity(world_vx, world_vy, command.yaw_rate)
        current_linear, current_angular = self._current_velocity()
        limited_linear = (
            _approach(float(current_linear[0]), world_vx, self._max_linear_acceleration * dt),
            _approach(float(current_linear[1]), world_vy, self._max_linear_acceleration * dt),
            float(current_linear[2]),
        )
        limited_angular = (
            _approach(float(current_angular[0]), angular_velocity[0], self._max_angular_acceleration * dt),
            _approach(float(current_angular[1]), angular_velocity[1], self._max_angular_acceleration * dt),
            _approach(float(current_angular[2]), angular_velocity[2], self._max_angular_acceleration * dt),
        )
        self._body.set_velocities(
            linear_velocities=list(limited_linear),
            angular_velocities=list(limited_angular),
        )

    def stop(self) -> None:
        """Stop planar and angular motion."""
        self._body.set_velocities(
            linear_velocities=[0.0, 0.0, 0.0],
            angular_velocities=[0.0, 0.0, 0.0],
        )

    def _world_linear_velocity(self, command: PlanarVelocityCommand) -> tuple[float, float]:
        """Return the commanded world-frame planar velocity."""
        return (command.vx, command.vy)

    def _rolling_angular_velocity(
        self,
        world_vx: float,
        world_vy: float,
        yaw_rate: float,
    ) -> tuple[float, float, float]:
        """Convert world-frame planar velocity to rolling angular velocity."""
        return (
            -world_vy / self._radius,
            world_vx / self._radius,
            yaw_rate,
        )

    def _current_velocity(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """Return current body velocities so gravity and impacts stay physical."""
        linear_velocities, angular_velocities = self._body.get_velocities(indices=[0])
        if hasattr(linear_velocities, "numpy"):
            linear_velocities = linear_velocities.numpy()
        if hasattr(angular_velocities, "numpy"):
            angular_velocities = angular_velocities.numpy()
        if hasattr(linear_velocities, "tolist"):
            linear_velocities = linear_velocities.tolist()
        if hasattr(angular_velocities, "tolist"):
            angular_velocities = angular_velocities.tolist()
        return (
            tuple(float(value) for value in linear_velocities[0]),
            tuple(float(value) for value in angular_velocities[0]),
        )


def _approach(current: float, target: float, max_delta: float) -> float:
    """Move current toward target by at most max_delta."""
    if max_delta <= 0.0:
        return target
    delta = target - current
    if abs(delta) <= max_delta:
        return target
    return current + math.copysign(max_delta, delta)


class MultiModuleVelocityController:
    """Apply planar velocity commands to multiple spherical modules."""

    def __init__(self, controllers: Mapping[str, SphericalVelocityController]) -> None:
        """Create a multi-module controller indexed by module id."""
        self._controllers = dict(controllers)

    @property
    def module_ids(self) -> tuple[str, ...]:
        """Return controlled module ids sorted for deterministic iteration."""
        return tuple(sorted(self._controllers))

    def apply(self, commands: Mapping[str, PlanarVelocityCommand], dt: float) -> None:
        """Apply commands only to addressed modules.

        Missing modules are left to the physics solver. This is important once
        modules are attached: passive modules should be dragged through joints
        instead of being forced to zero velocity every frame.
        """
        for module_id, controller in self._controllers.items():
            command = commands.get(module_id)
            if command is None:
                continue
            controller.apply(command, dt=dt)

    def stop(self) -> None:
        """Stop all controlled modules."""
        for controller in self._controllers.values():
            controller.stop()
