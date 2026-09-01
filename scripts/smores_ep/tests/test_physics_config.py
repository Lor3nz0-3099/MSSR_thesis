from __future__ import annotations

import pytest

from smores_ep.config.geometry import SmoresGeometry
from smores_ep.config.physics import (
    SMORES_DOF_NO_LOAD_SPEED_RAD_S,
    SMORES_EP_MAX_LAND_SPEED_M_S,
    SMORES_EP_MAX_WHEEL_SPEED_RAD_S,
    SmoresActuatorConfig,
    SmoresContactConfig,
    SmoresMassConfig,
)
from smores_ep.config.simulation import DynamicSimulationConfig
from smores_ep.isaac.physics_asset import (
    CHASSIS_PROXY_CENTER_M,
    CHASSIS_PROXY_SIZE_M,
    PAN_FACE_COLLISION_RADIUS_M,
)


def test_link_mass_budget_matches_paper_total() -> None:
    assert SmoresMassConfig().total_kg == pytest.approx(
        SmoresGeometry().module_mass_kg
    )


def test_estimated_center_of_mass_stays_over_wheel_axis() -> None:
    geometry = SmoresGeometry()
    center_x = SmoresMassConfig().estimated_total_com_x_m(geometry)
    assert center_x == pytest.approx(-0.001, abs=0.0003)


def test_passive_skid_has_less_friction_than_wheels_and_chassis() -> None:
    contacts = SmoresContactConfig()
    assert contacts.skid_static_friction < contacts.body_static_friction
    assert contacts.skid_dynamic_friction < contacts.body_dynamic_friction
    assert contacts.skid_dynamic_friction < contacts.wheel_dynamic_friction


def test_chassis_proxy_stays_well_inside_wheel_envelope_at_every_pitch() -> None:
    geometry = SmoresGeometry()
    half_x = 0.5 * CHASSIS_PROXY_SIZE_M[0]
    half_z = 0.5 * CHASSIS_PROXY_SIZE_M[2]
    radial_extent = (
        (abs(CHASSIS_PROXY_CENTER_M[0]) + half_x) ** 2
        + (abs(CHASSIS_PROXY_CENTER_M[2]) + half_z) ** 2
    ) ** 0.5

    # An eight-millimetre minimum gap remains larger than the collision skin
    # observed at the seed-3000 stair edge.
    assert geometry.wheel_radius_m - radial_extent >= 0.008


def test_pan_face_collision_proxy_is_inset_from_visual_rim() -> None:
    geometry = SmoresGeometry()

    # Keep the planner/visual envelope exact while preventing the perfectly
    # sharp PhysX rim from becoming a stair-edge brake.
    assert PAN_FACE_COLLISION_RADIUS_M < geometry.pan_face_radius_m

    # The proxy should also sit inside the driven-wheel rolling envelope so
    # small rim contacts do not unload the wheels first.
    assert PAN_FACE_COLLISION_RADIUS_M < geometry.wheel_radius_m

    # Do not hide large portions of the real connector geometry.
    assert geometry.pan_face_radius_m - PAN_FACE_COLLISION_RADIUS_M <= 0.002


def test_payload_overdrive_scales_effort_and_stiffens_holding_drives() -> None:
    nominal = SmoresActuatorConfig()
    payload = SmoresActuatorConfig.payload_overdrive(
        3.0,
        wheel_max_speed_rad_s=5.0,
        tilt_effort_scale=3.0,
    )
    assert payload.wheel_max_effort_nm == pytest.approx(
        3.0 * nominal.wheel_max_effort_nm
    )
    assert payload.tilt_max_effort_nm == pytest.approx(
        3.0 * nominal.tilt_max_effort_nm
    )
    assert payload.pan_max_effort_nm == pytest.approx(
        3.0 * nominal.pan_max_effort_nm
    )
    assert payload.wheel_damping_nm_s_per_rad == pytest.approx(
        nominal.wheel_damping_nm_s_per_rad
    )
    assert payload.tilt_stiffness_nm_per_rad > (
        nominal.tilt_stiffness_nm_per_rad
    )
    assert payload.wheel_max_speed_rad_s == pytest.approx(5.0)
    assert payload.internal_max_speed_rad_s == pytest.approx(
        SMORES_DOF_NO_LOAD_SPEED_RAD_S
    )


def test_payload_overdrive_can_exaggerate_only_the_tilt_payload() -> None:
    payload = SmoresActuatorConfig.payload_overdrive(
        4.0,
        tilt_effort_scale=8.0,
    )

    assert payload.wheel_max_effort_nm == pytest.approx(4.8)
    assert payload.pan_max_effort_nm == pytest.approx(5.6)
    assert payload.tilt_max_effort_nm == pytest.approx(18.4)
    assert payload.tilt_stiffness_nm_per_rad == pytest.approx(512.0)
    assert payload.hold_stiffness_nm_per_rad == pytest.approx(512.0)


def test_default_payload_can_lift_the_other_seven_snake_modules() -> None:
    geometry = SmoresGeometry()
    payload = SmoresActuatorConfig.payload_overdrive()
    gravity_m_s2 = 9.81
    # Conservative horizontal cantilever: seven complete modules represented
    # at consecutive TOP-to-BOTTOM spacings from the supporting hinge.
    seven_module_torque_nm = sum(
        geometry.module_mass_kg
        * gravity_m_s2
        * geometry.top_to_bottom_spacing_m
        * link_index
        for link_index in range(1, 8)
    )

    assert payload.wheel_max_effort_nm == pytest.approx(4.8)
    assert payload.tilt_max_effort_nm == pytest.approx(18.4)
    assert payload.tilt_max_effort_nm >= 1.5 * seven_module_torque_nm
    assert 0.025 * payload.tilt_stiffness_nm_per_rad >= (
        1.25 * seven_module_torque_nm
    )


def test_reference_land_speed_is_converted_through_cad_wheel_radius() -> None:
    geometry = SmoresGeometry()
    assert SMORES_EP_MAX_LAND_SPEED_M_S == pytest.approx(0.088)
    assert SMORES_EP_MAX_WHEEL_SPEED_RAD_S * geometry.wheel_radius_m == (
        pytest.approx(SMORES_EP_MAX_LAND_SPEED_M_S)
    )


def test_dynamic_frequency_requires_integer_render_decimation(
    tmp_path,
) -> None:
    with pytest.raises(ValueError):
        DynamicSimulationConfig(
            physics_usd=tmp_path / "physics.usd",
            headless=True,
            steps=1,
            physics_hz=240,
            render_hz=59,
        )


def test_dynamic_spawn_starts_just_above_the_lowest_cad_wheel(
    tmp_path,
) -> None:
    geometry = SmoresGeometry()
    config = DynamicSimulationConfig(
        physics_usd=tmp_path / "physics.usd",
        geometry=geometry,
    )
    lowest_wheel_z = min(
        geometry.left_wheel_center_body_m[2] - geometry.wheel_radius_m,
        geometry.right_wheel_center_body_m[2] - geometry.wheel_radius_m,
    )
    clearance = config.spawn_height_m + lowest_wheel_z
    # The physical wheel axle is now symmetric in Z.  With the
    # 31.6-mm spawn height and 31.06-mm wheel radius this leaves
    # about 0.54 mm of initial wheel clearance.
    assert clearance == pytest.approx(0.0001005, abs=5.0e-5)
