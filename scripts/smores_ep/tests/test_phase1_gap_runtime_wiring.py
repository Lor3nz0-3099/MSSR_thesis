from __future__ import annotations

from pathlib import Path

import pytest


def test_gap_runtime_installer_materializes_exact_support_banks(
    monkeypatch,
):
    from smores_ep.isaac import obstacle_course
    from smores_ep.isaac.phase1_gap_curriculum import (
        install_phase1_gap_curriculum_course,
        phase1_gap_curriculum_course,
    )

    calls = []

    def fake_install(
        stage,
        root_path,
        boxes,
    ):
        calls.append(
            (
                stage,
                root_path,
                boxes,
            )
        )

    monkeypatch.setattr(
        obstacle_course,
        "_install_course_boxes",
        fake_install,
    )

    stage = object()

    installed = (
        install_phase1_gap_curriculum_course(
            stage,
            seed=16673,
            split="train",
        )
    )

    expected = (
        phase1_gap_curriculum_course(
            16673,
            split="train",
        )
    )

    assert installed == expected

    assert calls == [
        (
            stage,
            "/World/Phase1GapCurriculum",
            expected.support_boxes,
        )
    ]


def test_self_assembly_config_accepts_gap_curriculum():
    from smores_ep.config.simulation import (
        SelfAssemblySimulationConfig,
    )

    config = SelfAssemblySimulationConfig(
        physics_usd=Path(
            "physics.usd"
        ),
        phase1_gap_course=True,
        phase1_gap_split="train",
        phase1_gap_seed=16000,
    )

    assert config.phase1_gap_course is True
    assert config.phase1_gap_split == "train"
    assert config.phase1_gap_seed == 16000


def test_gap_config_requires_seed():
    from smores_ep.config.simulation import (
        SelfAssemblySimulationConfig,
    )

    with pytest.raises(
        ValueError,
        match="gap.*seed|seed.*gap",
    ):
        SelfAssemblySimulationConfig(
            physics_usd=Path(
                "physics.usd"
            ),
            phase1_gap_course=True,
        )


def test_gap_config_rejects_invalid_split():
    from smores_ep.config.simulation import (
        SelfAssemblySimulationConfig,
    )

    with pytest.raises(
        ValueError,
        match="split",
    ):
        SelfAssemblySimulationConfig(
            physics_usd=Path(
                "physics.usd"
            ),
            phase1_gap_course=True,
            phase1_gap_split="banana",
            phase1_gap_seed=16000,
        )


def test_gap_and_stairs_curricula_are_mutually_exclusive():
    from smores_ep.config.simulation import (
        SelfAssemblySimulationConfig,
    )

    with pytest.raises(
        ValueError,
    ):
        SelfAssemblySimulationConfig(
            physics_usd=Path(
                "physics.usd"
            ),
            phase1_stairs_course=True,
            phase1_stairs_split="train",
            phase1_stairs_seed=12000,
            phase1_gap_course=True,
            phase1_gap_split="train",
            phase1_gap_seed=16000,
        )


def test_gap_course_does_not_change_flat_spawn_layout():
    from smores_ep.config.simulation import (
        SelfAssemblySimulationConfig,
    )
    from smores_ep.scenarios.parallel_self_assembly import (
        self_assembly_spawn_layout,
    )

    baseline = SelfAssemblySimulationConfig(
        physics_usd=Path(
            "physics.usd"
        ),
    )

    gap = SelfAssemblySimulationConfig(
        physics_usd=Path(
            "physics.usd"
        ),
        phase1_gap_course=True,
        phase1_gap_split="train",
        phase1_gap_seed=16000,
    )

    assert (
        self_assembly_spawn_layout(
            gap
        )
        ==
        self_assembly_spawn_layout(
            baseline
        )
    )


def test_cli_contains_gap_curriculum_contract():
    source = (
        Path(
            "scripts/smores_ep/src/"
            "smores_ep/self_assembly_cli.py"
        )
        .read_text()
    )

    assert "--phase1-gap-course" in source
    assert "--phase1-gap-split" in source
    assert "--phase1-gap-seed" in source

    assert (
        "phase1_gap_course="
        "args.phase1_gap_course"
        in source
    )

    assert (
        "phase1_gap_split="
        "args.phase1_gap_split"
        in source
    )

    assert (
        "phase1_gap_seed="
        "args.phase1_gap_seed"
        in source
    )


def test_scenario_materializes_gap_course_without_spawn_height_shift():
    source = (
        Path(
            "scripts/smores_ep/src/"
            "smores_ep/scenarios/"
            "parallel_self_assembly.py"
        )
        .read_text()
    )

    assert (
        "install_phase1_gap_curriculum_course"
        in source
    )

    assert "config.phase1_gap_course" in source
    assert "config.phase1_gap_seed" in source
    assert "config.phase1_gap_split" in source

    # Unlike stairs, gap banks are coplanar at z=0.
    assert (
        "place_phase1_gap_spawn_layout"
        not in source
    )
