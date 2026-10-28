from __future__ import annotations

from pathlib import Path

import pytest

from smores_ep.config.simulation import (
    SelfAssemblySimulationConfig,
)
from smores_ep.scenarios.parallel_self_assembly import (
    place_button_spawn_layout,
    self_assembly_spawn_layout,
)


GEN = Path(
    "scripts/smores_ep/src/"
    "smores_ep/isaac/"
    "phase1_manipulation_curriculum.py"
)

CLI = Path(
    "scripts/smores_ep/src/"
    "smores_ep/self_assembly_cli.py"
)

SCENARIO = Path(
    "scripts/smores_ep/src/"
    "smores_ep/scenarios/"
    "parallel_self_assembly.py"
)


def _config(
    tmp_path,
    **kwargs,
):
    values = {
        "physics_usd":
            tmp_path / "physics.usd",
    }

    values.update(
        kwargs
    )

    return SelfAssemblySimulationConfig(
        **values
    )


def test_config_defaults_keep_manipulation_curriculum_disabled(
    tmp_path,
):
    config = _config(
        tmp_path
    )

    assert (
        config.phase1_manipulation_course
        is False
    )

    assert (
        config.phase1_manipulation_split
        == "train"
    )

    assert (
        config.phase1_manipulation_seed
        is None
    )


def test_config_requires_seed_when_manipulation_course_enabled(
    tmp_path,
):
    with pytest.raises(
        ValueError,
        match="manipulation",
    ):
        _config(
            tmp_path,
            phase1_manipulation_course=True,
        )


def test_config_accepts_valid_manipulation_course(
    tmp_path,
):
    config = _config(
        tmp_path,
        phase1_manipulation_course=True,
        phase1_manipulation_split="train",
        phase1_manipulation_seed=20650,
    )

    assert (
        config.phase1_manipulation_course
        is True
    )

    assert (
        config.phase1_manipulation_seed
        == 20650
    )


def test_config_rejects_bad_split_and_course_conflict(
    tmp_path,
):
    with pytest.raises(
        ValueError
    ):
        _config(
            tmp_path,
            phase1_manipulation_course=True,
            phase1_manipulation_split="banana",
            phase1_manipulation_seed=20650,
        )

    with pytest.raises(
        ValueError,
        match="exclusive|course",
    ):
        _config(
            tmp_path,
            phase1_manipulation_course=True,
            phase1_manipulation_split="train",
            phase1_manipulation_seed=20650,
            button_test_course=True,
        )


def test_cli_exposes_manipulation_curriculum_contract():
    from smores_ep.self_assembly_cli import (
        build_argument_parser,
    )

    parser = (
        build_argument_parser()
    )

    args = parser.parse_args(
        [
            "--phase1-manipulation-course",
            "--phase1-manipulation-split",
            "train",
            "--phase1-manipulation-seed",
            "20650",
        ]
    )

    assert (
        args.phase1_manipulation_course
        is True
    )

    assert (
        args.phase1_manipulation_split
        == "train"
    )

    assert (
        args.phase1_manipulation_seed
        == 20650
    )


def test_cli_course_switch_remains_mutually_exclusive():
    from smores_ep.self_assembly_cli import (
        build_argument_parser,
    )

    parser = (
        build_argument_parser()
    )

    with pytest.raises(
        SystemExit
    ):
        parser.parse_args(
            [
                "--phase1-manipulation-course",
                "--button-test-course",
            ]
        )


def test_installer_materializes_curriculum_boxes():
    from pxr import Usd, UsdGeom, UsdShade
    from smores_ep.isaac.phase1_manipulation_curriculum import (
        install_phase1_manipulation_curriculum_course,
    )

    stage = Usd.Stage.CreateInMemory()
    UsdShade.Material.Define(stage, "/World/materials/dynamic_ground")
    course = install_phase1_manipulation_curriculum_course(
        stage, seed=20650, split="train",
    )
    for box in course.support_boxes:
        prim = stage.GetPrimAtPath(f"/World/Phase1ManipulationCurriculum/{box.name}")
        assert prim.IsA(UsdGeom.Cube)
        assert prim.GetAttribute("mssr:obstacleSemantic").Get() == box.semantic
        transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        assert tuple(transform.ExtractTranslation()) == pytest.approx(box.center_xyz_m)


def test_existing_button_spawn_transform_accepts_new_course(
    tmp_path,
):
    from smores_ep.isaac.phase1_manipulation_curriculum import (
        phase1_manipulation_curriculum_course,
    )

    module_ids = tuple(
        f"smores_{index:02d}"
        for index in range(
            1,
            9,
        )
    )

    config = _config(
        tmp_path,
        module_ids=module_ids,
    )

    base = (
        self_assembly_spawn_layout(
            config
        )
    )

    course = (
        phase1_manipulation_curriculum_course(
            20650,
            split="train",
        )
    )

    placed = (
        place_button_spawn_layout(
            base,
            course,
        )
    )

    assert set(placed) == set(base)

    cx = sum(
        pose[0]
        for pose
        in placed.values()
    ) / len(placed)

    cy = sum(
        pose[1]
        for pose
        in placed.values()
    ) / len(placed)

    assert cx == pytest.approx(
        course
        .assembly_spawn_center_xy_m[0]
    )

    assert cy == pytest.approx(
        course
        .assembly_spawn_center_xy_m[1]
    )

    radius = max(
        (
            (pose[0] - cx) ** 2
            + (pose[1] - cy) ** 2
        )
        ** 0.5
        for pose
        in placed.values()
    )

    assert radius == pytest.approx(
        course.assembly_spawn_radius_m
    )


def test_scenario_selects_and_places_manipulation_course():
    source = (
        SCENARIO
        .read_text()
    )

    assert (
        "config.phase1_manipulation_course"
        in source
    )

    assert (
        "install_phase1_manipulation_curriculum_course"
        in source
    )

    assert (
        "config.phase1_manipulation_seed"
        in source
    )

    assert (
        "config.phase1_manipulation_split"
        in source
    )

    # Reuse the validated button-relative transform.
    assert (
        "place_button_spawn_layout"
        in source
    )


def test_cli_passes_manipulation_fields_into_config():
    source = (
        CLI
        .read_text()
    )

    assert (
        "phase1_manipulation_course="
        "args.phase1_manipulation_course"
        in source
    )

    assert (
        "phase1_manipulation_split="
        "args.phase1_manipulation_split"
        in source
    )

    assert (
        "phase1_manipulation_seed="
        "args.phase1_manipulation_seed"
        in source
    )
