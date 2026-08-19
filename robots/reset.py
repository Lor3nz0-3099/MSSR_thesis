"""Reset utilities for repeatable simulation experiments."""

from __future__ import annotations

from collections.abc import Iterable

from robots.control import MultiModuleVelocityController
from robots.magnetic_attachment import MagneticAttachmentManager
from robots.spherical_robot import SphericalRobot


class SimulationResetManager:
    """Reset modules to their initial poses and remove magnetic attachments."""

    def __init__(
        self,
        robots: Iterable[SphericalRobot],
        controller: MultiModuleVelocityController,
        attachment_manager: MagneticAttachmentManager,
    ) -> None:
        """Store reset handles for the currently spawned modules."""
        from isaacsim.core.experimental.prims import RigidPrim

        self._robots = tuple(robots)
        self._controller = controller
        self._attachment_manager = attachment_manager
        self._bodies = {
            robot.module_id: RigidPrim(paths=robot.body_path)
            for robot in self._robots
        }

    def reset(self) -> None:
        """Restore initial module poses, clear velocities, and detach joints."""
        self._attachment_manager.detach_all()
        self._controller.stop()
        for robot in self._robots:
            body = self._bodies[robot.module_id]
            body.set_world_poses(
                positions=[list(robot.initial_state.pose.position)],
                orientations=[_xyzw_to_wxyz(robot.initial_state.pose.orientation_xyzw)],
            )
            body.set_velocities(
                linear_velocities=[0.0, 0.0, 0.0],
                angular_velocities=[0.0, 0.0, 0.0],
            )


def _xyzw_to_wxyz(orientation_xyzw: tuple[float, float, float, float]) -> list[float]:
    """Convert framework quaternion ordering to Isaac's WXYZ ordering."""
    x, y, z, w = orientation_xyzw
    return [w, x, y, z]
