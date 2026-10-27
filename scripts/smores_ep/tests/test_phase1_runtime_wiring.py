from __future__ import annotations

import importlib
import inspect

import pytest


def _installer():
    module = importlib.import_module(
        "smores_ep.isaac.phase1_training_course"
    )

    result = getattr(
        module,
        "install_phase1_training_course",
        None,
    )

    assert callable(
        result
    ), "TASK4C1_RED_PHASE1_INSTALLER_MISSING"

    return result


def test_task4c1_installer_materializes_exact_generated_course():
    installer = _installer()

    stage = object()

    observed = {
        "boxes": None,
        "cones": None,
    }

    def box_installer(
        received_stage,
        root_path,
        boxes,
    ):
        assert received_stage is stage

        observed["boxes"] = (
            root_path,
            tuple(boxes),
        )

    def cone_installer(
        received_stage,
        root_path,
        cones,
    ):
        assert received_stage is stage

        observed["cones"] = (
            root_path,
            tuple(cones),
        )

    course = installer(
        stage,
        seed=5100,
        box_installer=box_installer,
        cone_installer=cone_installer,
    )

    assert course.seed == 5100

    assert observed["boxes"] == (
        "/World/Phase1TrainingCourse/Supports",
        course.support_boxes,
    )

    assert observed["cones"] == (
        "/World/Phase1TrainingCourse/NavigationCones",
        course.cones,
    )

    assert len(course.support_boxes) > 2
    assert len(course.cones) == 6

    assert all(
        cone.static is True
        for cone in course.cones
    )

    public = course.to_observation()

    assert public["course_profile"] == "marl_phase1_rc_car8"

    # Generator-only reward path must remain private.
    assert "centerline_xy_m" not in repr(public)


def test_task4c1_self_assembly_config_exposes_seeded_phase1_course(
    tmp_path,
):
    from smores_ep.config.simulation import (
        SelfAssemblySimulationConfig,
    )

    fields = (
        SelfAssemblySimulationConfig
        .__dataclass_fields__
    )

    assert (
        "phase1_training_course"
        in fields
    ), "TASK4C1_RED_CONFIG_FLAG_MISSING"

    assert (
        "phase1_seed"
        in fields
    ), "TASK4C1_RED_CONFIG_SEED_MISSING"

    config = SelfAssemblySimulationConfig(
        physics_usd=tmp_path / "physics.usd",
        phase1_training_course=True,
        phase1_seed=5102,
    )

    assert config.phase1_training_course is True
    assert config.phase1_seed == 5102

    with pytest.raises(
        ValueError,
        match="exclusive",
    ):
        SelfAssemblySimulationConfig(
            physics_usd=tmp_path / "physics.usd",
            phase1_training_course=True,
            rc_car_planar_test_course=True,
        )


def test_task4c1_cli_exposes_phase1_course_and_seed():
    from smores_ep.self_assembly_cli import (
        build_argument_parser,
    )

    parser = build_argument_parser()

    option_strings = {
        option
        for action in parser._actions
        for option in action.option_strings
    }

    assert (
        "--phase1-training-course"
        in option_strings
    ), "TASK4C1_RED_CLI_FLAG_MISSING"

    assert (
        "--phase1-seed"
        in option_strings
    ), "TASK4C1_RED_CLI_SEED_MISSING"

    args = parser.parse_args(
        [
            "--phase1-training-course",
            "--phase1-seed",
            "5102",
        ]
    )

    assert args.phase1_training_course is True
    assert args.phase1_seed == 5102

    # It must live in the same mutually-exclusive course group.
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--phase1-training-course",
                "--rc-car-planar-test-course",
            ]
        )


def test_task4c1_parallel_scenario_wires_phase1_installer():
    scenario = importlib.import_module(
        "smores_ep.scenarios.parallel_self_assembly"
    )

    source = inspect.getsource(
        scenario.run_parallel_self_assembly_scenario
    )

    assert (
        "config.phase1_training_course"
        in source
    ), "TASK4C1_RED_SCENARIO_FLAG_MISSING"

    assert (
        "install_phase1_training_course"
        in source
    ), "TASK4C1_RED_SCENARIO_INSTALL_MISSING"

    assert (
        "config.phase1_seed"
        in source
    )
