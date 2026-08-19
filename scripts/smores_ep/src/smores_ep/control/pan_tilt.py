from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class DifferentialMotorAngles:
    motor_a_rad: float
    motor_b_rad: float


class ContinuousAngleTracker:
    """Unwrap a revolute-joint coordinate into a continuous angle."""

    def __init__(self) -> None:
        self._previous_raw_rad: float | None = None
        self._continuous_rad = 0.0

    def update(self, raw_rad: float) -> float:
        if not math.isfinite(raw_rad):
            raise ValueError("Joint angle must be finite")
        if self._previous_raw_rad is None:
            self._previous_raw_rad = raw_rad
            self._continuous_rad = raw_rad
            return self._continuous_rad

        delta = raw_rad - self._previous_raw_rad
        delta = (delta + math.pi) % (2.0 * math.pi) - math.pi
        self._continuous_rad += delta
        self._previous_raw_rad = raw_rad
        return self._continuous_rad


def continuous_position_servo_velocity(
    current_rad: float,
    target_rad: float,
    proportional_gain_s: float,
    maximum_speed_rad_s: float,
    tolerance_rad: float = 1.0e-3,
) -> float:
    """Convert an unbounded position error into a bounded velocity target."""

    values = (
        current_rad,
        target_rad,
        proportional_gain_s,
        maximum_speed_rad_s,
        tolerance_rad,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Pan servo values must be finite")
    if (
        proportional_gain_s <= 0.0
        or maximum_speed_rad_s <= 0.0
        or tolerance_rad < 0.0
    ):
        raise ValueError("Pan servo gains and limits are invalid")
    error = target_rad - current_rad
    if abs(error) <= tolerance_rad:
        return 0.0
    velocity = proportional_gain_s * error
    return max(-maximum_speed_rad_s, min(maximum_speed_rad_s, velocity))


def normalize_revolute_target(target_rad: float) -> float:
    """Map a continuous logical PAN target into PhysX's revolute range."""
    if not math.isfinite(target_rad):
        raise ValueError("Revolute target must be finite")
    return math.atan2(math.sin(target_rad), math.cos(target_rad))


def clamp_tilt(
    target_rad: float,
    minimum_rad: float = -math.pi / 2.0,
    maximum_rad: float = math.pi / 2.0,
) -> float:
    if not all(
        math.isfinite(value)
        for value in (target_rad, minimum_rad, maximum_rad)
    ):
        raise ValueError("Tilt values must be finite")
    if minimum_rad >= maximum_rad:
        raise ValueError("Tilt limits are inverted")
    return max(minimum_rad, min(maximum_rad, target_rad))


def move_toward(
    current_rad: float,
    target_rad: float,
    max_speed_rad_s: float,
    dt_s: float,
) -> float:
    values = (current_rad, target_rad, max_speed_rad_s, dt_s)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Motion-profile values must be finite")
    if max_speed_rad_s <= 0.0 or dt_s < 0.0:
        raise ValueError("Speed must be positive and time non-negative")
    error = target_rad - current_rad
    maximum_step = max_speed_rad_s * dt_s
    return current_rad + max(-maximum_step, min(maximum_step, error))


def pan_tilt_to_motor_angles(
    pan_rad: float,
    tilt_rad: float,
    output_to_motor_ratio: float = 1.0,
) -> DifferentialMotorAngles:
    """Mix output pan/tilt coordinates into the two internal actuators.

    In the SMORES differential, opposite actuator rotations create pan and
    concordant rotations create tilt. The ratio is kept explicit because the
    current kinematic scenario commands output coordinates, while a later
    dynamic motor model will include the measured gear train.
    """

    if not all(
        math.isfinite(value)
        for value in (pan_rad, tilt_rad, output_to_motor_ratio)
    ):
        raise ValueError("Pan/tilt mixer inputs must be finite")
    if output_to_motor_ratio <= 0.0:
        raise ValueError("Output-to-motor ratio must be positive")
    return DifferentialMotorAngles(
        motor_a_rad=output_to_motor_ratio * (tilt_rad + pan_rad),
        motor_b_rad=output_to_motor_ratio * (tilt_rad - pan_rad),
    )


def motor_angles_to_pan_tilt(
    motor_a_rad: float,
    motor_b_rad: float,
    output_to_motor_ratio: float = 1.0,
) -> tuple[float, float]:
    if not all(
        math.isfinite(value)
        for value in (
            motor_a_rad,
            motor_b_rad,
            output_to_motor_ratio,
        )
    ):
        raise ValueError("Pan/tilt mixer inputs must be finite")
    if output_to_motor_ratio <= 0.0:
        raise ValueError("Output-to-motor ratio must be positive")
    scale = 0.5 / output_to_motor_ratio
    pan = scale * (motor_a_rad - motor_b_rad)
    tilt = scale * (motor_a_rad + motor_b_rad)
    return pan, tilt
