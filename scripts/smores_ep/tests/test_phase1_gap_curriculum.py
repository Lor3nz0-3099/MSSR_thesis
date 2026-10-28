from __future__ import annotations

import json

import pytest


def test_gap_split_sizes_and_seed_disjointness():
    from smores_ep.isaac.phase1_gap_curriculum import (
        gap_curriculum_seeds,
    )

    train = gap_curriculum_seeds("train")
    validation = gap_curriculum_seeds(
        "validation"
    )
    test = gap_curriculum_seeds("test")
    challenge = gap_curriculum_seeds(
        "challenge"
    )

    assert len(train) == 1000
    assert len(validation) == 120
    assert len(test) == 200
    assert len(challenge) == 30

    assert train[0] == 16000
    assert train[-1] == 16999

    assert validation[0] == 17000
    assert validation[-1] == 17119

    assert test[0] == 18000
    assert test[-1] == 18199

    assert challenge[0] == 18500
    assert challenge[-1] == 18529

    sets = (
        set(train),
        set(validation),
        set(test),
        set(challenge),
    )

    for index, first in enumerate(sets):
        for second in sets[index + 1 :]:
            assert first.isdisjoint(
                second
            )


def test_train_difficulty_allocation_is_250_400_350():
    from smores_ep.isaac.phase1_gap_curriculum import (
        gap_curriculum_identity,
        gap_curriculum_seeds,
    )

    counts = {
        "easy": 0,
        "medium": 0,
        "hard": 0,
    }

    for seed in gap_curriculum_seeds(
        "train"
    ):
        identity = gap_curriculum_identity(
            seed,
            split="train",
        )

        counts[
            identity.difficulty
        ] += 1

    assert counts == {
        "easy": 250,
        "medium": 400,
        "hard": 350,
    }


def test_validation_and_test_have_all_difficulties():
    from smores_ep.isaac.phase1_gap_curriculum import (
        gap_curriculum_identity,
        gap_curriculum_seeds,
    )

    validation = [
        gap_curriculum_identity(
            seed,
            split="validation",
        ).difficulty
        for seed in gap_curriculum_seeds(
            "validation"
        )
    ]

    test = [
        gap_curriculum_identity(
            seed,
            split="test",
        ).difficulty
        for seed in gap_curriculum_seeds(
            "test"
        )
    ]

    assert validation.count("easy") == 40
    assert validation.count("medium") == 40
    assert validation.count("hard") == 40

    assert test.count("easy") == 60
    assert test.count("medium") == 70
    assert test.count("hard") == 70


def test_easy_gap_contract():
    from smores_ep.isaac.phase1_gap_curriculum import (
        gap_curriculum_seeds,
        phase1_gap_curriculum_course,
    )

    courses = (
        phase1_gap_curriculum_course(
            seed,
            split="train",
        )
        for seed in gap_curriculum_seeds(
            "train"
        )[:250]
    )

    for course in courses:
        assert course.difficulty == "easy"

        assert (
            0.120
            <= course.gap_width_m
            <= 0.145
        )

        assert (
            0.620
            <= course.near_edge_x_m
            <= 0.750
        )

        assert (
            1.150
            <= course.landing_length_m
            <= 1.350
        )

        assert (
            course.landing_center_y_m
            == pytest.approx(0.0)
        )


def test_medium_gap_contract():
    from smores_ep.isaac.phase1_gap_curriculum import (
        gap_curriculum_seeds,
        phase1_gap_curriculum_course,
    )

    seeds = gap_curriculum_seeds(
        "train"
    )[250:650]

    offsets = []

    for seed in seeds:
        course = (
            phase1_gap_curriculum_course(
                seed,
                split="train",
            )
        )

        assert course.difficulty == "medium"

        assert (
            0.140
            <= course.gap_width_m
            <= 0.160
        )

        assert (
            0.560
            <= course.near_edge_x_m
            <= 0.700
        )

        assert (
            1.050
            <= course.landing_length_m
            <= 1.300
        )

        assert (
            abs(
                course.landing_center_y_m
            )
            <= 0.025
        )

        offsets.append(
            course.landing_center_y_m
        )

    assert any(
        value < 0.0
        for value in offsets
    )

    assert any(
        value > 0.0
        for value in offsets
    )


def test_hard_gap_contract_and_physical_cap():
    from smores_ep.isaac.phase1_gap_curriculum import (
        gap_curriculum_seeds,
        phase1_gap_curriculum_course,
    )

    seeds = gap_curriculum_seeds(
        "train"
    )[650:]

    offsets = []

    for seed in seeds:
        course = (
            phase1_gap_curriculum_course(
                seed,
                split="train",
            )
        )

        assert course.difficulty == "hard"

        assert (
            0.155
            <= course.gap_width_m
            <= 0.178
        )

        assert course.gap_width_m <= 0.178

        assert (
            0.520
            <= course.near_edge_x_m
            <= 0.640
        )

        assert (
            0.950
            <= course.landing_length_m
            <= 1.200
        )

        assert (
            abs(
                course.landing_center_y_m
            )
            <= 0.060
        )

        offsets.append(
            course.landing_center_y_m
        )

    assert any(
        value < 0.0
        for value in offsets
    )

    assert any(
        value > 0.0
        for value in offsets
    )


def test_all_train_stages_preserve_assembly_clearance():
    from smores_ep.isaac.phase1_gap_curriculum import (
        gap_curriculum_seeds,
        phase1_gap_curriculum_course,
    )

    # Existing loose-module radial spawn is about 0.34 m.
    assembly_positive_x_extent_m = 0.340

    minimum_margin = float("inf")

    for seed in gap_curriculum_seeds(
        "train"
    ):
        course = (
            phase1_gap_curriculum_course(
                seed,
                split="train",
            )
        )

        margin = (
            course.near_edge_x_m
            - assembly_positive_x_extent_m
        )

        minimum_margin = min(
            minimum_margin,
            margin,
        )

        assert margin >= 0.180

    assert minimum_margin >= 0.180


def test_support_boxes_leave_exact_gap_interval_open():
    from smores_ep.isaac.phase1_gap_curriculum import (
        gap_curriculum_seeds,
        phase1_gap_curriculum_course,
    )

    for seed in gap_curriculum_seeds(
        "train"
    ):
        course = (
            phase1_gap_curriculum_course(
                seed,
                split="train",
            )
        )

        assert len(
            course.support_boxes
        ) == 2

        approach = course.approach_bank
        landing = course.landing_bank

        approach_end_x = (
            approach.center_xyz_m[0]
            + 0.5
            * approach.size_xyz_m[0]
        )

        landing_start_x = (
            landing.center_xyz_m[0]
            - 0.5
            * landing.size_xyz_m[0]
        )

        assert approach_end_x == pytest.approx(
            course.near_edge_x_m
        )

        assert landing_start_x == pytest.approx(
            course.far_edge_x_m
        )

        assert (
            landing_start_x
            - approach_end_x
        ) == pytest.approx(
            course.gap_width_m
        )

        assert approach.pitch_deg == pytest.approx(
            0.0
        )

        assert landing.pitch_deg == pytest.approx(
            0.0
        )


def test_landing_offsets_never_remove_safe_bank_overlap():
    from smores_ep.isaac.phase1_gap_curriculum import (
        gap_curriculum_seeds,
        phase1_gap_curriculum_course,
    )

    for split in (
        "train",
        "validation",
        "test",
        "challenge",
    ):
        for seed in gap_curriculum_seeds(
            split
        ):
            course = (
                phase1_gap_curriculum_course(
                    seed,
                    split=split,
                )
            )

            overlap_y_m = (
                course.bank_width_m
                - abs(
                    course.landing_center_y_m
                )
            )

            assert overlap_y_m >= 1.120


def test_challenge_uses_max_validated_envelope_not_beyond_it():
    from smores_ep.isaac.phase1_gap_curriculum import (
        gap_curriculum_seeds,
        phase1_gap_curriculum_course,
    )

    courses = [
        phase1_gap_curriculum_course(
            seed,
            split="challenge",
        )
        for seed in gap_curriculum_seeds(
            "challenge"
        )
    ]

    assert all(
        course.difficulty == "challenge"
        for course in courses
    )

    assert all(
        0.165
        <= course.gap_width_m
        <= 0.178
        for course in courses
    )

    assert all(
        0.500
        <= course.near_edge_x_m
        <= 0.600
        for course in courses
    )

    assert all(
        0.850
        <= course.landing_length_m
        <= 1.050
        for course in courses
    )

    assert all(
        abs(
            course.landing_center_y_m
        )
        <= 0.080
        for course in courses
    )

    assert any(
        course.landing_center_y_m < 0.0
        for course in courses
    )

    assert any(
        course.landing_center_y_m > 0.0
        for course in courses
    )


def test_train_distribution_has_structural_variety():
    from smores_ep.isaac.phase1_gap_curriculum import (
        gap_curriculum_seeds,
        phase1_gap_curriculum_course,
    )

    structures = set()

    for seed in gap_curriculum_seeds(
        "train"
    ):
        course = (
            phase1_gap_curriculum_course(
                seed,
                split="train",
            )
        )

        structures.add(
            (
                course.gap_width_m,
                course.near_edge_x_m,
                course.landing_length_m,
                course.landing_center_y_m,
            )
        )

    assert len(structures) >= 950


def test_course_generation_is_deterministic():
    from smores_ep.isaac.phase1_gap_curriculum import (
        phase1_gap_curriculum_course,
    )

    first = phase1_gap_curriculum_course(
        16673,
        split="train",
    )

    second = phase1_gap_curriculum_course(
        16673,
        split="train",
    )

    assert first == second


def test_public_observation_contains_geometry_but_no_reward_or_success_oracle():
    from smores_ep.isaac.phase1_gap_curriculum import (
        phase1_gap_curriculum_course,
    )

    course = phase1_gap_curriculum_course(
        16673,
        split="train",
    )

    observation = course.to_observation()

    assert observation["frame_id"] == "world"

    assert (
        observation["course_profile"]
        == "phase1_gap_curriculum_v1"
    )

    assert observation["gap"]["width_m"] == pytest.approx(
        course.gap_width_m
    )

    assert observation["gap"]["near_edge_x_m"] == pytest.approx(
        course.near_edge_x_m
    )

    assert observation["gap"]["far_edge_x_m"] == pytest.approx(
        course.far_edge_x_m
    )

    serialized = json.dumps(
        observation
    ).lower()

    assert "reward" not in serialized
    assert "success" not in serialized
    assert "progress" not in serialized


def test_wrong_split_or_seed_fails_closed():
    from smores_ep.isaac.phase1_gap_curriculum import (
        gap_curriculum_identity,
        gap_curriculum_seeds,
        phase1_gap_curriculum_course,
    )

    with pytest.raises(
        ValueError,
        match="split",
    ):
        gap_curriculum_seeds(
            "banana"
        )

    with pytest.raises(
        ValueError,
        match="seed",
    ):
        gap_curriculum_identity(
            16000,
            split="validation",
        )

    with pytest.raises(
        ValueError,
        match="seed",
    ):
        phase1_gap_curriculum_course(
            999,
            split="train",
        )
