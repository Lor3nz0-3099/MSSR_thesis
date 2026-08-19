from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class WheelRates:
    left_rad_s: float
    right_rad_s: float


@dataclass(frozen=True)
class PlanarPose:
    x_m: float = 0.0
    y_m: float = 0.0
    yaw_rad: float = 0.0


def twist_to_wheel_rates(
    linear_x_m_s: float,
    angular_z_rad_s: float,
    wheel_radius_m: float,
    track_width_m: float,
) -> WheelRates:
    """Map a ROS Twist to left/right wheel rates.

    Positive rates use the mathematical rolling convention. The Isaac visual
    applies the required sign for its common +Y axle orientation.
    """

    values = (
        linear_x_m_s,
        angular_z_rad_s,
        wheel_radius_m,
        track_width_m,
    )
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("Differential-drive inputs must be finite")
    if wheel_radius_m <= 0.0 or track_width_m <= 0.0:
        raise ValueError("Wheel radius and track width must be positive")
    half_track = 0.5 * track_width_m
    return WheelRates(
        left_rad_s=(
            linear_x_m_s - angular_z_rad_s * half_track
        )
        / wheel_radius_m,
        right_rad_s=(
            linear_x_m_s + angular_z_rad_s * half_track
        )
        / wheel_radius_m,
    )


def integrate_planar_pose(
    pose: PlanarPose,
    linear_x_m_s: float,
    angular_z_rad_s: float,
    dt_s: float,
) -> PlanarPose:
    """Integrate a body-frame planar twist using midpoint orientation."""

    values = (
        pose.x_m,
        pose.y_m,
        pose.yaw_rad,
        linear_x_m_s,
        angular_z_rad_s,
        dt_s,
    )
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("Pose integration inputs must be finite")
    if dt_s < 0.0:
        raise ValueError("Integration time cannot be negative")
    yaw_mid = pose.yaw_rad + 0.5 * angular_z_rad_s * dt_s
    yaw = math.remainder(
        pose.yaw_rad + angular_z_rad_s * dt_s,
        2.0 * math.pi,
    )
    return PlanarPose(
        x_m=pose.x_m + linear_x_m_s * math.cos(yaw_mid) * dt_s,
        y_m=pose.y_m + linear_x_m_s * math.sin(yaw_mid) * dt_s,
        yaw_rad=yaw,
    )

