from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class WheelDriveConfig:
    """Measured motor limits and explicit reduced-order drive controls.

    The no-load speed and stall torque come from Table I of the FreeBOT
    paper. Heading gains are simulation/control parameters and are deliberately
    kept separate from the motor data.
    """

    linear_scale_deg_s: float = 720.0
    yaw_scale_deg_s: float = 360.0
    damping: float = 500.0
    no_load_speed_deg_s: float = 360.0
    stall_torque_nm: float = 7.0 * 0.0980665
    # Effective output-side rotor/gear inertia.  A value of 3e-3 kg m^2 is
    # consistent with a small DC rotor reflected through a high-ratio gearbox
    # and prevents the nearly massless CAD tire from changing speed by
    # thousands of degrees per second in one solver interval.
    armature_kg_m2: float = 0.003
    zero_command_brake_torque_nm: float = 0.12
    zero_command_threshold_deg_s: float = 1.0e-6
    climb_heading_enabled: bool = False
    climb_heading_gain_s_inv: float = 2.0
    climb_heading_max_turn_deg_s: float = 90.0
    climb_heading_capture_gap_m: float = 0.005

    def __post_init__(self) -> None:
        values = (
            self.linear_scale_deg_s,
            self.yaw_scale_deg_s,
            self.damping,
            self.no_load_speed_deg_s,
            self.stall_torque_nm,
            self.armature_kg_m2,
            self.zero_command_brake_torque_nm,
            self.zero_command_threshold_deg_s,
            self.climb_heading_gain_s_inv,
            self.climb_heading_max_turn_deg_s,
            self.climb_heading_capture_gap_m,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("Wheel-drive parameters must be finite")
        if (
            self.linear_scale_deg_s <= 0.0
            or self.yaw_scale_deg_s <= 0.0
            or self.no_load_speed_deg_s <= 0.0
        ):
            raise ValueError("Wheel command scales must be positive")
        if (
            self.damping < 0.0
            or self.stall_torque_nm < 0.0
            or self.armature_kg_m2 < 0.0
            or self.zero_command_brake_torque_nm < 0.0
            or self.zero_command_threshold_deg_s < 0.0
            or self.climb_heading_gain_s_inv < 0.0
            or self.climb_heading_max_turn_deg_s < 0.0
            or self.climb_heading_capture_gap_m < 0.0
        ):
            raise ValueError("Drive damping and torque must be non-negative")
        if self.zero_command_brake_torque_nm > self.stall_torque_nm:
            raise ValueError("Brake torque cannot exceed motor stall torque")


@dataclass(frozen=True)
class WheelVelocityTargets:
    left_deg_s: float
    right_deg_s: float


@dataclass(frozen=True)
class WheelTorqueLimits:
    left_nm: float
    right_nm: float


def _clamp_speed(speed_deg_s: float, limit_deg_s: float) -> float:
    return max(-limit_deg_s, min(limit_deg_s, float(speed_deg_s)))


def twist_to_wheel_targets(
    linear_x: float,
    angular_z: float,
    config: WheelDriveConfig | None = None,
) -> WheelVelocityTargets:
    """Convert a planar ROS Twist into differential wheel targets."""

    parameters = WheelDriveConfig() if config is None else config
    linear = float(linear_x)
    yaw = float(angular_z)
    if not math.isfinite(linear) or not math.isfinite(yaw):
        raise ValueError("Twist command must be finite")

    forward = parameters.linear_scale_deg_s * linear
    turn = parameters.yaw_scale_deg_s * yaw
    return WheelVelocityTargets(
        left_deg_s=_clamp_speed(
            forward - turn,
            parameters.no_load_speed_deg_s,
        ),
        right_deg_s=_clamp_speed(
            forward + turn,
            parameters.no_load_speed_deg_s,
        ),
    )


def apply_climb_heading_correction(
    targets: WheelVelocityTargets,
    heading_error_rad: float,
    config: WheelDriveConfig | None = None,
) -> WheelVelocityTargets:
    """Steer through wheel-speed difference, without applying a fake force."""

    parameters = WheelDriveConfig() if config is None else config
    error = float(heading_error_rad)
    if not math.isfinite(error):
        raise ValueError("Heading error must be finite")
    turn = parameters.yaw_scale_deg_s * parameters.climb_heading_gain_s_inv * error
    turn = max(
        -parameters.climb_heading_max_turn_deg_s,
        min(parameters.climb_heading_max_turn_deg_s, turn),
    )
    return WheelVelocityTargets(
        left_deg_s=_clamp_speed(
            targets.left_deg_s - turn,
            parameters.no_load_speed_deg_s,
        ),
        right_deg_s=_clamp_speed(
            targets.right_deg_s + turn,
            parameters.no_load_speed_deg_s,
        ),
    )


def signed_heading_error_rad(
    current_axis_world: Sequence[float],
    desired_axis_world: Sequence[float],
    rotation_axis_world: Sequence[float],
) -> float:
    """Return the shortest signed axle error about the carrier yaw axis.

    A wheel axle is an unoriented line, so the desired direction is flipped
    when necessary to avoid requesting an unnecessary 180-degree turn.
    """

    def vector3(
        values: Sequence[float],
        label: str,
    ) -> tuple[float, float, float]:
        vector = tuple(float(value) for value in values)
        if len(vector) != 3 or not all(math.isfinite(value) for value in vector):
            raise ValueError(f"{label} must contain three finite components")
        return vector

    def dot(first: tuple[float, ...], second: tuple[float, ...]) -> float:
        return sum(a * b for a, b in zip(first, second))

    def normalized(
        vector: tuple[float, float, float],
        label: str,
    ) -> tuple[float, float, float]:
        norm = math.sqrt(dot(vector, vector))
        if norm <= 1.0e-12:
            raise ValueError(f"{label} cannot be zero")
        return tuple(value / norm for value in vector)

    rotation = normalized(
        vector3(rotation_axis_world, "Rotation axis"),
        "Rotation axis",
    )
    current_raw = vector3(current_axis_world, "Current axle")
    desired_raw = vector3(desired_axis_world, "Desired axle")

    def projected(vector: tuple[float, float, float]) -> tuple[float, float, float]:
        axial = dot(vector, rotation)
        return tuple(value - axial * axis for value, axis in zip(vector, rotation))

    current = normalized(projected(current_raw), "Projected current axle")
    desired = normalized(projected(desired_raw), "Projected desired axle")
    if dot(current, desired) < 0.0:
        desired = tuple(-value for value in desired)
    cross = (
        current[1] * desired[2] - current[2] * desired[1],
        current[2] * desired[0] - current[0] * desired[2],
        current[0] * desired[1] - current[1] * desired[0],
    )
    return math.atan2(dot(rotation, cross), dot(current, desired))


def dc_motor_torque_limit_nm(
    target_speed_deg_s: float,
    actual_speed_deg_s: float,
    config: WheelDriveConfig | None = None,
) -> float:
    """Return the DC-motor torque envelope for the requested voltage.

    A target speed represents the no-load speed produced by the requested
    motor voltage.  The linear torque-speed law is therefore proportional to
    the target/actual speed error and is capped at the measured stall torque.
    """

    parameters = WheelDriveConfig() if config is None else config
    target = _clamp_speed(
        target_speed_deg_s,
        parameters.no_load_speed_deg_s,
    )
    actual = float(actual_speed_deg_s)
    if not math.isfinite(actual):
        raise ValueError("Actual wheel speed must be finite")
    normalized_error = abs(target - actual) / parameters.no_load_speed_deg_s
    dynamic_limit = parameters.stall_torque_nm * min(1.0, normalized_error)
    if abs(target) <= parameters.zero_command_threshold_deg_s:
        # An enabled zero-speed drive with a finite force cap represents the
        # active motor/gearbox brake. Unlike pure back-EMF damping, it can
        # balance a static gravity torque up to the calibrated brake limit.
        return max(dynamic_limit, parameters.zero_command_brake_torque_nm)
    return dynamic_limit


def motor_torque_limits(
    targets: WheelVelocityTargets,
    actual_left_deg_s: float,
    actual_right_deg_s: float,
    config: WheelDriveConfig | None = None,
) -> WheelTorqueLimits:
    parameters = WheelDriveConfig() if config is None else config
    return WheelTorqueLimits(
        left_nm=dc_motor_torque_limit_nm(
            targets.left_deg_s,
            actual_left_deg_s,
            parameters,
        ),
        right_nm=dc_motor_torque_limit_nm(
            targets.right_deg_s,
            actual_right_deg_s,
            parameters,
        ),
    )


def apply_wheel_targets(
    stage: object,
    module_root: str,
    targets: WheelVelocityTargets,
    config: WheelDriveConfig | None = None,
    torque_limits: WheelTorqueLimits | None = None,
    drive_instance: str = "angular",
) -> None:
    """Set the authored angular drives of one FreeBOT articulation."""

    parameters = WheelDriveConfig() if config is None else config
    limits = (
        WheelTorqueLimits(
            parameters.stall_torque_nm,
            parameters.stall_torque_nm,
        )
        if torque_limits is None
        else torque_limits
    )
    if drive_instance not in {"angular", "rotY"}:
        raise ValueError("Wheel drive instance must be 'angular' or 'rotY'")
    drive_prefix = f"drive:{drive_instance}:physics"
    for side, velocity, torque_limit in (
        ("left", targets.left_deg_s, limits.left_nm),
        ("right", targets.right_deg_s, limits.right_nm),
    ):
        path = f"{module_root}/joints/{side}_wheel_joint"
        joint = stage.GetPrimAtPath(path)
        if not joint:
            raise RuntimeError(f"Missing wheel joint: {path}")
        from pxr import PhysxSchema

        PhysxSchema.PhysxJointAPI.Apply(joint).CreateArmatureAttr().Set(
            float(parameters.armature_kg_m2)
        )
        if drive_instance == "angular":
            joint.GetAttribute("physics:axis").Set("Y")
        joint.GetAttribute(f"{drive_prefix}:targetVelocity").Set(
            float(velocity)
        )
        joint.GetAttribute(f"{drive_prefix}:damping").Set(
            float(parameters.damping)
        )
        joint.GetAttribute(f"{drive_prefix}:maxForce").Set(
            float(torque_limit)
        )
