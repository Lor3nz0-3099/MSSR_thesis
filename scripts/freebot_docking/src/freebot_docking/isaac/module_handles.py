from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from freebot_docking.config.geometry import RunningGearGeometry, ShellGeometry
from freebot_docking.config.magnet import MagnetConfig
from freebot_docking.physics.contact_geometry import (
    ball_inside_sphere_clearance_m,
    cylinder_inside_sphere_clearance_m,
)
from freebot_docking.physics.state import MagnetState, ShellState, Vector3


def _first_vector(values: Any) -> Vector3:
    if hasattr(values, "numpy"):
        values = values.numpy()
    array = np.asarray(values, dtype=np.float64)
    return np.array(array[0] if array.ndim > 1 else array, copy=True)


def _first_quaternion_wxyz(values: Any) -> np.ndarray:
    if hasattr(values, "numpy"):
        values = values.numpy()
    array = np.asarray(values, dtype=np.float64)
    quaternion = np.array(array[0] if array.ndim > 1 else array, copy=True)
    if quaternion.shape != (4,):
        raise ValueError("Isaac quaternion must contain four components")
    return quaternion


def _first_scalar(values: Any) -> float:
    if hasattr(values, "numpy"):
        values = values.numpy()
    return float(np.asarray(values, dtype=np.float64).reshape(-1)[0])


def quaternion_rotate_wxyz(
    quaternion_wxyz: np.ndarray,
    vector: np.ndarray | tuple[float, float, float],
) -> Vector3:
    """Rotate a vector by an Isaac scalar-first unit quaternion."""

    quaternion = np.asarray(quaternion_wxyz, dtype=np.float64)
    value = np.asarray(vector, dtype=np.float64)
    scalar = quaternion[0]
    imaginary = quaternion[1:]
    return np.array(
        value
        + 2.0 * scalar * np.cross(imaginary, value)
        + 2.0 * np.cross(imaginary, np.cross(imaginary, value)),
        dtype=np.float64,
    )


@dataclass
class FreebotModuleHandles:
    """Isaac handles and state conversion for one complete FreeBOT module."""

    root_path: str
    shell_body: Any
    internal_body: Any
    magnet_frame: Any
    left_wheel_body: Any
    right_wheel_body: Any
    caster_1_body: Any
    caster_2_body: Any
    left_wheel_mount_body: Any | None = None
    right_wheel_mount_body: Any | None = None

    @classmethod
    def create(
        cls,
        root_path: str,
        rigid_prim_type: Any,
        xform_prim_type: Any,
        wheel_radial_compliance_enabled: bool = False,
    ) -> "FreebotModuleHandles":
        return cls(
            root_path=root_path,
            shell_body=rigid_prim_type(paths=f"{root_path}/shell_link"),
            internal_body=rigid_prim_type(paths=f"{root_path}/internal_link"),
            magnet_frame=xform_prim_type(
                paths=f"{root_path}/internal_link/magnet_frame"
            ),
            left_wheel_body=rigid_prim_type(
                paths=f"{root_path}/left_wheel_link"
            ),
            right_wheel_body=rigid_prim_type(
                paths=f"{root_path}/right_wheel_link"
            ),
            caster_1_body=rigid_prim_type(
                paths=f"{root_path}/caster_1_ball_link"
            ),
            caster_2_body=rigid_prim_type(
                paths=f"{root_path}/caster_2_ball_link"
            ),
            left_wheel_mount_body=(
                rigid_prim_type(
                    paths=f"{root_path}/left_wheel_radial_mount_link"
                )
                if wheel_radial_compliance_enabled
                else None
            ),
            right_wheel_mount_body=(
                rigid_prim_type(
                    paths=f"{root_path}/right_wheel_radial_mount_link"
                )
                if wheel_radial_compliance_enabled
                else None
            ),
        )

    @staticmethod
    def body_com_world(body: Any) -> Vector3:
        positions, orientations = body.get_world_poses()
        body_position = _first_vector(positions)
        body_orientation = _first_quaternion_wxyz(orientations)
        local_com_positions = body.get_coms()[0]
        local_com = _first_vector(local_com_positions)
        return body_position + quaternion_rotate_wxyz(
            body_orientation,
            local_com,
        )

    def shell_state(self, geometry: ShellGeometry) -> ShellState:
        positions, orientations = self.shell_body.get_world_poses()
        body_origin = _first_vector(positions)
        orientation = _first_quaternion_wxyz(orientations)
        center = body_origin + quaternion_rotate_wxyz(
            orientation,
            geometry.center_from_body_origin_m,
        )
        linear_velocities, angular_velocities = (
            self.shell_body.get_velocities()
        )
        return ShellState(
            center_world=center,
            com_world=self.body_com_world(self.shell_body),
            linear_velocity_world=_first_vector(linear_velocities),
            angular_velocity_world=_first_vector(angular_velocities),
        )

    def magnet_state(self, config: MagnetConfig) -> MagnetState:
        positions, orientations = self.magnet_frame.get_world_poses()
        center = _first_vector(positions)
        orientation = _first_quaternion_wxyz(orientations)
        axis = quaternion_rotate_wxyz(
            orientation,
            config.active_axis_local,
        )
        axis /= np.linalg.norm(axis)
        linear_velocities, angular_velocities = (
            self.internal_body.get_velocities()
        )
        return MagnetState(
            center_world=center,
            axis_world=axis,
            carrier_com_world=self.body_com_world(self.internal_body),
            carrier_linear_velocity_world=_first_vector(linear_velocities),
            carrier_angular_velocity_world=_first_vector(angular_velocities),
        )

    def wheel_joint_speed_deg_s(self, wheel_body: Any) -> float:
        """Return wheel speed relative to the carrier around local Y."""

        _, orientations = wheel_body.get_world_poses()
        orientation = _first_quaternion_wxyz(orientations)
        spin_axis = quaternion_rotate_wxyz(
            orientation,
            [0.0, 1.0, 0.0],
        )
        spin_axis /= np.linalg.norm(spin_axis)
        _, wheel_angular_velocities = wheel_body.get_velocities()
        _, carrier_angular_velocities = self.internal_body.get_velocities()
        relative_angular_velocity = (
            _first_vector(wheel_angular_velocities)
            - _first_vector(carrier_angular_velocities)
        )
        return float(np.degrees(np.dot(relative_angular_velocity, spin_axis)))

    def wheel_speeds_deg_s(self) -> tuple[float, float]:
        return (
            self.wheel_joint_speed_deg_s(self.left_wheel_body),
            self.wheel_joint_speed_deg_s(self.right_wheel_body),
        )

    def wheel_radial_projections_m(
        self,
        axes_local: tuple[Vector3, Vector3],
    ) -> tuple[float, float]:
        """Project both wheel-centre offsets onto carrier-fixed spring axes."""

        carrier_origin, _ = self._body_pose(self.internal_body)
        projections = []
        for wheel, axis_local in zip(
            (self.left_wheel_body, self.right_wheel_body),
            axes_local,
        ):
            wheel_origin, _ = self._body_pose(wheel)
            axis_world = self.carrier_direction_world(axis_local)
            projections.append(
                float(np.dot(wheel_origin - carrier_origin, axis_world))
            )
        return (projections[0], projections[1])

    def carrier_direction_world(self, local_direction: Vector3) -> Vector3:
        """Rotate a carrier-fixed direction into the world frame."""

        direction = np.asarray(local_direction, dtype=np.float64)
        if direction.shape != (3,):
            raise ValueError("Carrier direction must contain three components")
        norm = float(np.linalg.norm(direction))
        if norm <= 0.0:
            raise ValueError("Carrier direction cannot be zero")
        _, orientation = self._body_pose(self.internal_body)
        return quaternion_rotate_wxyz(orientation, direction / norm)

    def rigid_bodies(self) -> tuple[Any, ...]:
        """Return every massive body of the complete module."""

        bodies = (
            self.shell_body,
            self.internal_body,
            self.left_wheel_body,
            self.right_wheel_body,
            self.caster_1_body,
            self.caster_2_body,
        )
        mounts = tuple(
            body
            for body in (
                self.left_wheel_mount_body,
                self.right_wheel_mount_body,
            )
            if body is not None
        )
        return bodies + mounts

    @staticmethod
    def body_mass_kg(body: Any) -> float:
        return _first_scalar(body.get_masses())

    def total_mass_kg(self) -> float:
        return sum(self.body_mass_kg(body) for body in self.rigid_bodies())

    def module_com_world(self) -> Vector3:
        masses = np.asarray(
            [self.body_mass_kg(body) for body in self.rigid_bodies()],
            dtype=np.float64,
        )
        centers = np.asarray(
            [self.body_com_world(body) for body in self.rigid_bodies()],
            dtype=np.float64,
        )
        return np.sum(masses[:, None] * centers, axis=0) / np.sum(masses)

    def gravity_wrench_about(
        self,
        reference_world: Vector3,
        gravity_m_s2: float = 9.81,
    ) -> tuple[Vector3, Vector3]:
        """Return total gravity force and moment for the complete module."""

        reference = np.asarray(reference_world, dtype=np.float64)
        force = np.zeros(3, dtype=np.float64)
        moment = np.zeros(3, dtype=np.float64)
        for body in self.rigid_bodies():
            body_force = np.array(
                [0.0, 0.0, -self.body_mass_kg(body) * gravity_m_s2],
                dtype=np.float64,
            )
            body_com = self.body_com_world(body)
            force += body_force
            moment += np.cross(body_com - reference, body_force)
        return force, moment

    @staticmethod
    def _body_pose(body: Any) -> tuple[Vector3, np.ndarray]:
        positions, orientations = body.get_world_poses()
        return (
            _first_vector(positions),
            _first_quaternion_wxyz(orientations),
        )

    def wheel_inner_shell_clearance_m(
        self,
        wheel_body: Any,
        side: str,
        shell_center_world: Vector3,
        shell_geometry: ShellGeometry,
        running_gear: RunningGearGeometry,
    ) -> float:
        """Return CAD-proxy tire clearance from the analytic inner shell."""

        if side not in {"left", "right"}:
            raise ValueError("Wheel side must be 'left' or 'right'")
        body_center, orientation = self._body_pose(wheel_body)
        outward_sign = 1.0 if side == "left" else -1.0
        nominal_axis = quaternion_rotate_wxyz(
            orientation,
            [0.0, 1.0, 0.0],
        )
        wheel_center = (
            body_center
            + outward_sign
            * running_gear.tire_center_axial_offset_m
            * nominal_axis
        )
        tilt = np.radians(running_gear.tire_axis_tilt_deg)
        fitted_axis = quaternion_rotate_wxyz(
            orientation,
            [0.0, np.cos(tilt), np.sin(tilt)],
        )
        return cylinder_inside_sphere_clearance_m(
            wheel_center - shell_center_world,
            fitted_axis,
            running_gear.tire_collision_radius_m,
            running_gear.tire_half_width_m,
            shell_geometry.inner_radius_m,
        )

    @staticmethod
    def caster_inner_shell_clearance_m(
        caster_body: Any,
        shell_center_world: Vector3,
        shell_geometry: ShellGeometry,
        running_gear: RunningGearGeometry,
    ) -> float:
        """Return CAD-proxy caster clearance from the analytic inner shell."""

        positions, _ = caster_body.get_world_poses()
        caster_center = _first_vector(positions)
        return ball_inside_sphere_clearance_m(
            caster_center - shell_center_world,
            running_gear.caster_collision_radius_m,
            shell_geometry.inner_radius_m,
        )

    def inner_shell_clearances_m(
        self,
        shell_center_world: Vector3,
        shell_geometry: ShellGeometry,
        running_gear: RunningGearGeometry,
    ) -> tuple[float, float, float, float]:
        """Return left/right wheel then caster 1/2 analytic clearances."""

        return (
            self.wheel_inner_shell_clearance_m(
                self.left_wheel_body,
                "left",
                shell_center_world,
                shell_geometry,
                running_gear,
            ),
            self.wheel_inner_shell_clearance_m(
                self.right_wheel_body,
                "right",
                shell_center_world,
                shell_geometry,
                running_gear,
            ),
            self.caster_inner_shell_clearance_m(
                self.caster_1_body,
                shell_center_world,
                shell_geometry,
                running_gear,
            ),
            self.caster_inner_shell_clearance_m(
                self.caster_2_body,
                shell_center_world,
                shell_geometry,
                running_gear,
            ),
        )
