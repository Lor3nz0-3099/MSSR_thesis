from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from smores_ep.config.simulation import (
    SelfAssemblySimulationConfig,
)
from smores_ep.self_assembly_cli import (
    build_argument_parser,
)


def _config(**overrides):
    values = dict(
        physics_usd=Path("unused.usd"),
        module_ids=tuple(
            f"smores_{index:02d}"
            for index in range(1, 9)
        ),
    )

    values.update(overrides)

    return SelfAssemblySimulationConfig(
        **values
    )


def test_phase1_stairs_runtime_config_defaults_fail_safe():
    config = _config()

    assert config.phase1_stairs_course is False
    assert config.phase1_stairs_split == "train"
    assert config.phase1_stairs_seed is None


def test_phase1_stairs_runtime_requires_seed_when_enabled():
    with pytest.raises(
        ValueError,
        match="seed",
    ):
        _config(
            phase1_stairs_course=True,
        )


def test_phase1_stairs_runtime_rejects_unknown_split():
    with pytest.raises(
        ValueError,
        match="split",
    ):
        _config(
            phase1_stairs_course=True,
            phase1_stairs_split="banana",
            phase1_stairs_seed=12000,
        )


def test_phase1_stairs_runtime_is_mutually_exclusive():
    with pytest.raises(
        ValueError,
        match="exclusive",
    ):
        _config(
            phase1_stairs_course=True,
            phase1_stairs_seed=12000,
            stair_test_course=True,
        )


def test_phase1_stairs_cli_contract():
    parser = build_argument_parser()

    args = parser.parse_args(
        [
            "--phase1-stairs-course",
            "--phase1-stairs-split",
            "validation",
            "--phase1-stairs-seed",
            "13017",
        ]
    )

    assert args.phase1_stairs_course is True
    assert args.phase1_stairs_split == "validation"
    assert args.phase1_stairs_seed == 13017


def test_phase1_stairs_cli_course_is_exclusive():
    parser = build_argument_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--phase1-stairs-course",
                "--stair-test-course",
                "--phase1-stairs-seed",
                "12000",
            ]
        )


def test_ascent_spawn_stays_at_nominal_height():
    from smores_ep.isaac.phase1_stairs_curriculum import (
        phase1_stairs_curriculum_course,
        place_phase1_stairs_spawn_layout,
    )

    course = phase1_stairs_curriculum_course(
        12000,
        split="train",
    )

    source = {
        "smores_01": (
            0.0,
            0.0,
            0.0316,
            0.0,
        )
    }

    placed = place_phase1_stairs_spawn_layout(
        source,
        course,
    )

    assert course.direction == "ascent"
    assert course.start_height_m == pytest.approx(0.0)

    assert placed["smores_01"][2] == pytest.approx(
        0.0316
    )


def test_descent_spawn_is_elevated_to_start_platform():
    from smores_ep.isaac.phase1_stairs_curriculum import (
        phase1_stairs_curriculum_course,
        place_phase1_stairs_spawn_layout,
    )

    # Hard ordinal 650 -> mode 2 -> descent/uniform.
    course = phase1_stairs_curriculum_course(
        12650,
        split="train",
    )

    source = {
        "smores_01": (
            0.10,
            -0.20,
            0.0316,
            15.0,
        ),
        "smores_02": (
            -0.10,
            +0.20,
            0.0316,
            -15.0,
        ),
    }

    placed = place_phase1_stairs_spawn_layout(
        source,
        course,
    )

    assert course.direction == "descent"
    assert course.start_height_m > 0.0

    assert placed["smores_01"] == pytest.approx(
        (
            0.10,
            -0.20,
            0.0316 + course.start_height_m,
            15.0,
        )
    )

    assert placed["smores_02"] == pytest.approx(
        (
            -0.10,
            +0.20,
            0.0316 + course.start_height_m,
            -15.0,
        )
    )


def test_phase1_stairs_installer_has_explicit_split_and_seed():
    from smores_ep.isaac.phase1_stairs_curriculum import (
        install_phase1_stairs_curriculum_course,
    )

    signature = inspect.signature(
        install_phase1_stairs_curriculum_course
    )

    assert tuple(signature.parameters) == (
        "stage",
        "seed",
        "split",
    )


def test_scenario_contains_phase1_stairs_materialization_and_spawn_handoff():
    scenario = Path(
        "scripts/smores_ep/src/smores_ep/"
        "scenarios/parallel_self_assembly.py"
    ).read_text()

    assert (
        "install_phase1_stairs_curriculum_course"
        in scenario
    )

    assert (
        "place_phase1_stairs_spawn_layout"
        in scenario
    )

    assert "config.phase1_stairs_course" in scenario
