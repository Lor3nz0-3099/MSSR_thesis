from __future__ import annotations

import math

import pytest

from smores_ep.control.pan_tilt import (
    ContinuousAngleTracker,
    clamp_tilt,
    continuous_position_servo_velocity,
    motor_angles_to_pan_tilt,
    move_toward,
    normalize_revolute_target,
    pan_tilt_to_motor_angles,
)


def test_opposite_motor_motion_produces_pan() -> None:
    motors = pan_tilt_to_motor_angles(0.7, 0.0)
    assert motors.motor_a_rad == pytest.approx(0.7)
    assert motors.motor_b_rad == pytest.approx(-0.7)


def test_concordant_motor_motion_produces_tilt() -> None:
    motors = pan_tilt_to_motor_angles(0.0, 0.4)
    assert motors.motor_a_rad == pytest.approx(0.4)
    assert motors.motor_b_rad == pytest.approx(0.4)


def test_motor_mixer_round_trip() -> None:
    motors = pan_tilt_to_motor_angles(1.2, -0.3, 5.0)
    pan, tilt = motor_angles_to_pan_tilt(
        motors.motor_a_rad,
        motors.motor_b_rad,
        5.0,
    )
    assert pan == pytest.approx(1.2)
    assert tilt == pytest.approx(-0.3)


def test_tilt_is_limited_and_profile_is_rate_limited() -> None:
    assert clamp_tilt(math.pi) == pytest.approx(math.pi / 2.0)
    assert move_toward(0.0, 1.0, 2.0, 0.1) == pytest.approx(0.2)


def test_continuous_angle_tracker_survives_wraparound() -> None:
    tracker = ContinuousAngleTracker()
    assert tracker.update(6.20) == pytest.approx(6.20)
    assert tracker.update(0.05) == pytest.approx(
        6.20 + (0.05 - 6.20 + math.pi) % (2.0 * math.pi) - math.pi
    )


def test_continuous_pan_servo_accepts_targets_beyond_two_pi() -> None:
    velocity = continuous_position_servo_velocity(
        current_rad=2.0 * math.pi,
        target_rad=4.0 * math.pi,
        proportional_gain_s=4.0,
        maximum_speed_rad_s=2.4,
    )
    assert velocity == pytest.approx(2.4)
    assert continuous_position_servo_velocity(
        current_rad=4.0 * math.pi,
        target_rad=4.0 * math.pi,
        proportional_gain_s=4.0,
        maximum_speed_rad_s=2.4,
    ) == 0.0


def test_revolute_target_is_normalized_for_physx() -> None:
    assert normalize_revolute_target(4.0 * math.pi) == pytest.approx(0.0)
    assert normalize_revolute_target(2.5 * math.pi) == pytest.approx(
        math.pi / 2.0
    )
