from __future__ import annotations

from dataclasses import dataclass
import math

from smores_ep.control.differential_drive import PlanarPose


def wrap_angle(angle_rad: float) -> float:
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


@dataclass(frozen=True)
class PoseControllerConfig:
    max_linear_speed_m_s: float = 0.088
    max_angular_speed_rad_s: float = 1.5
    position_tolerance_m: float = 0.008
    yaw_tolerance_rad: float = math.radians(5.0)
    rotate_first_threshold_rad: float = math.radians(35.0)
    linear_gain_s: float = 1.8
    angular_gain_s: float = 3.0

    def __post_init__(self) -> None:
        values = (
            self.max_linear_speed_m_s,
            self.max_angular_speed_rad_s,
            self.position_tolerance_m,
            self.yaw_tolerance_rad,
            self.rotate_first_threshold_rad,
            self.linear_gain_s,
            self.angular_gain_s,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in values):
            raise ValueError("Pose-controller parameters must be positive")


@dataclass(frozen=True)
class PoseControlStep:
    linear_x_m_s: float
    angular_z_rad_s: float
    position_error_m: float
    yaw_error_rad: float
    phase: str
    done: bool


@dataclass(frozen=True)
class AxialPoseAdjustmentReference:
    """Regularized reference for Eq. (4)-(5) of Liu et al. (ICRA 2020).

    For a mobile TOP/BOTTOM face, the paper controls the lateral coordinate
    ``y'`` and relative yaw ``theta'`` in the goal-face frame:

    ``y'_dot = sin(theta') * v`` and ``theta'_dot = omega``.

    Directly evaluating ``v = -k_y*y'/sin(theta')`` is singular at the goal.
    We retain the same closed-loop condition by fixing a physically useful
    signed rolling speed and solving for the bounded heading whose sine gives
    ``y'_dot = -k_y*y'``. The heading servo then realizes the second row of
    the paper controller without an unbounded wheel command.
    """

    linear_x_m_s: float
    desired_relative_yaw_rad: float
    lateral_velocity_m_s: float
    saturated: bool


def axial_pose_adjustment_reference(
    lateral_error_m: float,
    drive_direction: float,
    *,
    translation_speed_m_s: float = 0.025,
    lateral_gain_s: float = 2.0,
    maximum_relative_yaw_rad: float = math.radians(25.0),
) -> AxialPoseAdjustmentReference:
    """Return a nonsingular implementation of the paper's pose controller."""

    values = (
        lateral_error_m,
        drive_direction,
        translation_speed_m_s,
        lateral_gain_s,
        maximum_relative_yaw_rad,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Pose-adjustment inputs must be finite")
    if drive_direction not in {-1.0, 1.0}:
        raise ValueError("drive_direction must be -1.0 or +1.0")
    if translation_speed_m_s <= 0.0 or lateral_gain_s <= 0.0:
        raise ValueError("Pose-adjustment speed and gain must be positive")
    if not 0.0 < maximum_relative_yaw_rad < 0.5 * math.pi:
        raise ValueError("maximum_relative_yaw_rad must be in (0, pi/2)")

    requested_sine = (
        -lateral_gain_s
        * lateral_error_m
        / (drive_direction * translation_speed_m_s)
    )
    sine_limit = math.sin(maximum_relative_yaw_rad)
    bounded_sine = max(-sine_limit, min(sine_limit, requested_sine))
    desired_yaw = math.asin(bounded_sine)
    linear = drive_direction * translation_speed_m_s
    return AxialPoseAdjustmentReference(
        linear_x_m_s=linear,
        desired_relative_yaw_rad=desired_yaw,
        lateral_velocity_m_s=linear * math.sin(desired_yaw),
        saturated=not math.isclose(
            bounded_sine,
            requested_sine,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ),
    )


def drive_to_pose_step(
    current: PlanarPose,
    target: PlanarPose,
    config: PoseControllerConfig | None = None,
) -> PoseControlStep:
    """Generate a non-holonomic differential-drive command."""

    limits = config or PoseControllerConfig()
    dx = target.x_m - current.x_m
    dy = target.y_m - current.y_m
    distance = math.hypot(dx, dy)
    final_yaw_error = wrap_angle(target.yaw_rad - current.yaw_rad)

    if distance <= limits.position_tolerance_m:
        if abs(final_yaw_error) <= limits.yaw_tolerance_rad:
            return PoseControlStep(
                0.0,
                0.0,
                distance,
                final_yaw_error,
                "complete",
                True,
            )
        angular = max(
            -limits.max_angular_speed_rad_s,
            min(
                limits.max_angular_speed_rad_s,
                limits.angular_gain_s * final_yaw_error,
            ),
        )
        return PoseControlStep(
            0.0,
            angular,
            distance,
            final_yaw_error,
            "final_yaw",
            False,
        )

    bearing = math.atan2(dy, dx)
    heading_error = wrap_angle(bearing - current.yaw_rad)
    direction = 1.0
    if abs(heading_error) > 0.5 * math.pi:
        direction = -1.0
        heading_error = wrap_angle(heading_error - math.copysign(math.pi, heading_error))

    angular = max(
        -limits.max_angular_speed_rad_s,
        min(
            limits.max_angular_speed_rad_s,
            limits.angular_gain_s * heading_error,
        ),
    )
    if abs(heading_error) >= limits.rotate_first_threshold_rad:
        linear = 0.0
        phase = "orient_to_path"
    else:
        alignment_scale = max(0.15, math.cos(heading_error))
        linear = direction * min(
            limits.max_linear_speed_m_s,
            limits.linear_gain_s * distance,
        ) * alignment_scale
        phase = "translate"

    return PoseControlStep(
        linear,
        angular,
        distance,
        final_yaw_error,
        phase,
        False,
    )
