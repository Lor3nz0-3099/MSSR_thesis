from __future__ import annotations

import importlib
import inspect

import pytest


def test_task4e2_marl_installer_materializes_exact_curriculum_course():
    module = importlib.import_module(
        "smores_ep.isaac.phase1_marl_curriculum"
    )

    installer = getattr(
        module,
        "install_phase1_marl_curriculum_course",
        None,
    )

    assert callable(
        installer
    ), "TASK4E2_RED_INSTALLER_MISSING"

    stage = object()
    observed = {}

    def boxes(
        received_stage,
        root_path,
        specifications,
    ):
        assert received_stage is stage
        observed["boxes"] = (
            root_path,
            tuple(specifications),
        )

    def cones(
        received_stage,
        root_path,
        specifications,
    ):
        assert received_stage is stage
        observed["cones"] = (
            root_path,
            tuple(specifications),
        )

    course = installer(
        stage,
        seed=7654,
        split="train",
        box_installer=boxes,
        cone_installer=cones,
    )

    assert (
        "hairpin"
        in course.track_profile
    )

    assert observed["boxes"] == (
        "/World/Phase1MarlCurriculum/Supports",
        course.support_boxes,
    )

    assert observed["cones"] == (
        "/World/Phase1MarlCurriculum/NavigationCones",
        course.cones,
    )


def test_task4e2_simulation_config_exposes_marl_curriculum():
    from smores_ep.config.simulation import (
        SelfAssemblySimulationConfig,
    )

    fields = (
        SelfAssemblySimulationConfig
        .__dataclass_fields__
    )

    assert (
        "phase1_marl_course"
        in fields
    ), "TASK4E2_RED_CONFIG_MISSING"

    assert (
        "phase1_marl_split"
        in fields
    )

    assert (
        "phase1_marl_seed"
        in fields
    )


def test_task4e2_cli_exposes_marl_course_split_and_seed():
    from smores_ep.self_assembly_cli import (
        build_argument_parser,
    )

    parser = build_argument_parser()

    options = {
        option
        for action in parser._actions
        for option in action.option_strings
    }

    assert (
        "--phase1-marl-course"
        in options
    ), "TASK4E2_RED_CLI_MISSING"

    assert (
        "--phase1-marl-split"
        in options
    )

    assert (
        "--phase1-marl-seed"
        in options
    )

    args = parser.parse_args(
        [
            "--phase1-marl-course",
            "--phase1-marl-split",
            "train",
            "--phase1-marl-seed",
            "7653",
        ]
    )

    assert args.phase1_marl_course
    assert args.phase1_marl_split == "train"
    assert args.phase1_marl_seed == 7653

    # Reference/debug Phase-1 and MARL curriculum are mutually exclusive.
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--phase1-training-course",
                "--phase1-marl-course",
            ]
        )


def test_task4e2_parallel_scenario_uses_new_curriculum_installer():
    module = importlib.import_module(
        "smores_ep.scenarios.parallel_self_assembly"
    )

    source = inspect.getsource(
        module.run_parallel_self_assembly_scenario
    )

    assert (
        "config.phase1_marl_course"
        in source
    ), "TASK4E2_RED_SCENARIO_MISSING"

    assert (
        "install_phase1_marl_curriculum_course"
        in source
    )

    assert (
        "config.phase1_marl_split"
        in source
    )

    assert (
        "config.phase1_marl_seed"
        in source
    )
