"""Readers that convert simulator state into framework module state."""

from __future__ import annotations

from collections.abc import Iterable
from collections.abc import Mapping

from robots.module_state import ModulePose, ModuleState, ModuleTwist, Quaternion, Vector3
from robots.joints import JointType
from robots.state_registry import ModuleStateRegistry, RobotStateSnapshot
from robots.surface_attachment import ModulePair, SurfaceAttachmentConfig, annotate_surface_contacts


class IsaacRigidBodyStateReader:
    """Read one Isaac rigid body and produce an updated ``ModuleState``."""

    def __init__(self, initial_state: ModuleState) -> None:
        """Create a reader for the rigid body described by ``initial_state``."""
        from isaacsim.core.experimental.prims import RigidPrim

        self._state = initial_state
        self._body = RigidPrim(paths=initial_state.prim_path)

    @property
    def module_id(self) -> str:
        """Return the logical module id handled by this reader."""
        return self._state.module_id

    def read(self, last_command: ModuleTwist | None = None) -> ModuleState:
        """Read pose and velocity from Isaac and return the latest module state."""
        positions, orientations_wxyz = self._body.get_world_poses(indices=[0])
        linear_velocities, angular_velocities = self._body.get_velocities(indices=[0])

        pose = ModulePose(
            frame_id=self._state.pose.frame_id,
            position=_first_vector3(positions),
            orientation_xyzw=_first_quaternion_xyzw(orientations_wxyz),
        )
        twist = ModuleTwist(
            linear=_first_vector3(linear_velocities),
            angular=_first_vector3(angular_velocities),
        )

        self._state = self._state.with_pose(pose).with_twist(twist)
        if last_command is not None:
            self._state = self._state.with_last_command(last_command)
        return self._state


class ModuleStateTracker:
    """Update a module registry from simulator readers and surface contacts."""

    def __init__(
        self,
        readers: Iterable[IsaacRigidBodyStateReader],
        registry: ModuleStateRegistry | None = None,
        attachment_config: SurfaceAttachmentConfig | None = None,
    ) -> None:
        """Create a tracker for a collection of module state readers."""
        self._readers = tuple(readers)
        self._registry = registry or ModuleStateRegistry()
        self._attachment_config = attachment_config or SurfaceAttachmentConfig()

    @property
    def registry(self) -> ModuleStateRegistry:
        """Return the mutable registry maintained by this tracker."""
        return self._registry

    def update(
        self,
        timestamp: float,
        last_commands: dict[str, ModuleTwist] | None = None,
        connected_pairs: set[ModulePair] | None = None,
        connected_joint_types: dict[ModulePair, JointType] | None = None,
        connected_attachment_metadata: dict[ModulePair, Mapping[str, object]] | None = None,
    ) -> RobotStateSnapshot:
        """Read all modules, annotate contacts, update the registry, and snapshot."""
        commands = last_commands or {}
        states = tuple(
            reader.read(last_command=commands.get(reader.module_id))
            for reader in self._readers
        )
        states = annotate_surface_contacts(
            states,
            self._attachment_config,
            connected_pairs=connected_pairs,
            connected_joint_types=connected_joint_types,
            connected_attachment_metadata=connected_attachment_metadata,
        )
        self._registry.update_many(states)
        return self._registry.snapshot(timestamp)


def planar_command_to_twist(vx: float, vy: float, yaw_rate: float) -> ModuleTwist:
    """Represent a planar command as a generic module twist command."""
    return ModuleTwist(linear=(vx, vy, 0.0), angular=(0.0, 0.0, yaw_rate))


def _first_vector3(values: object) -> Vector3:
    """Convert the first row of an Isaac tensor-like value to a 3D tuple."""
    row = _first_row(values)
    return (float(row[0]), float(row[1]), float(row[2]))


def _first_quaternion_xyzw(values: object) -> Quaternion:
    """Convert Isaac's first quaternion from ``wxyz`` to framework ``xyzw``."""
    row = _first_row(values)
    return (float(row[1]), float(row[2]), float(row[3]), float(row[0]))


def _first_row(values: object) -> object:
    """Return the first row from Warp, NumPy, or sequence-like values."""
    if hasattr(values, "numpy"):
        values = values.numpy()
    if hasattr(values, "tolist"):
        values = values.tolist()
    return values[0]  # type: ignore[index]
