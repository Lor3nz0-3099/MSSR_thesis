import pytest

from freebot_docking.config.mass import ModuleMassConfig


def test_mass_budget_matches_freebot_table_i() -> None:
    masses = ModuleMassConfig()

    total = (
        masses.shell_kg
        + masses.internal_link_kg
        + 2.0 * masses.wheel_kg
        + 2.0 * masses.caster_ball_kg
    )

    assert total == pytest.approx(0.3079, abs=1.0e-12)
    assert masses.wheel_kg == pytest.approx(0.00311844754375)
    assert masses.caster_ball_kg == pytest.approx(0.0033061086717)
    assert masses.shell_kg == pytest.approx(0.090)
    assert masses.internal_link_kg == pytest.approx(0.2050508875691)


def test_internal_mechanism_mass_excludes_shell() -> None:
    masses = ModuleMassConfig()

    assert masses.internal_mechanism_total_kg == pytest.approx(0.2179)


def test_internal_body_has_finite_reduced_inertia() -> None:
    inertia = ModuleMassConfig().internal_box_diagonal_inertia_kg_m2

    assert all(value > 0.0 for value in inertia)
    assert inertia[2] > inertia[0]


def test_diagnostic_mass_scaling_preserves_body_mass_ratios() -> None:
    nominal = ModuleMassConfig()
    scaled = nominal.scaled(0.25)

    assert scaled.module_total_kg == pytest.approx(0.25 * nominal.module_total_kg)
    assert scaled.shell_kg == pytest.approx(0.25 * nominal.shell_kg)
    assert scaled.internal_link_kg == pytest.approx(0.25 * nominal.internal_link_kg)
    assert scaled.wheel_kg == pytest.approx(0.25 * nominal.wheel_kg)
    assert scaled.caster_ball_kg == pytest.approx(0.25 * nominal.caster_ball_kg)
    assert scaled.internal_box_diagonal_inertia_kg_m2 == pytest.approx(
        tuple(0.25 * value for value in nominal.internal_box_diagonal_inertia_kg_m2)
    )


@pytest.mark.parametrize("factor", [0.0, -1.0, float("inf")])
def test_diagnostic_mass_scaling_rejects_invalid_factor(factor: float) -> None:
    with pytest.raises(ValueError, match="Mass scale"):
        ModuleMassConfig().scaled(factor)
