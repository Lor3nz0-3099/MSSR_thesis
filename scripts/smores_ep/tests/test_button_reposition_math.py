from smores_ep.control import button_ik_math


def test_mm8_reposition_is_translation_only_along_fixed_drive_axis():
    # Direct assembly aligns MM8 forward with -press_direction.
    # Seed 6101 press_direction = (-1, 0), therefore forward = (+1, 0).
    common = dict(
        current_x_m=2.557,
        current_y_m=-1.102,
        target_x_m=1.918,
        target_y_m=-1.062,
        drive_axis_x=1.0,
        drive_axis_y=0.0,
    )

    a = button_ik_math.plan_mm8_reposition_command(
        current_yaw_rad=0.0,
        **common,
    )

    b = button_ik_math.plan_mm8_reposition_command(
        current_yaw_rad=2.49,
        **common,
    )

    # Target is behind the MM8 along its fixed longitudinal axis:
    # it must simply reverse.
    assert a["phase"] == "translate"
    assert a["linear_x_m_s"] < 0.0
    assert a["angular_z_rad_s"] == 0.0

    # Odom/root yaw must not affect the decision.
    assert b == a


def test_mm8_reposition_stops_at_longitudinal_target():
    command = button_ik_math.plan_mm8_reposition_command(
        current_x_m=1.918,
        current_y_m=-1.102,
        current_yaw_rad=2.49,
        target_x_m=1.918,
        target_y_m=-1.062,
        drive_axis_x=1.0,
        drive_axis_y=0.0,
    )

    assert command["phase"] == "done"
    assert command["linear_x_m_s"] == 0.0
    assert command["angular_z_rad_s"] == 0.0
