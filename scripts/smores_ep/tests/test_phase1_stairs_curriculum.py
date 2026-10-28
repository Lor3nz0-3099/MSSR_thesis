from __future__ import annotations

from collections import Counter
import math

import pytest


def _api():
    from smores_ep.isaac import phase1_stairs_curriculum as api

    return api


def test_stair_curriculum_split_sizes_and_seed_disjointness():
    api = _api()

    train = set(api.stair_curriculum_seeds("train"))
    validation = set(api.stair_curriculum_seeds("validation"))
    test = set(api.stair_curriculum_seeds("test"))
    challenge = set(api.stair_curriculum_seeds("challenge"))

    assert len(train) == 1000
    assert len(validation) == 120
    assert len(test) == 200
    assert len(challenge) == 30

    assert train.isdisjoint(validation)
    assert train.isdisjoint(test)
    assert train.isdisjoint(challenge)
    assert validation.isdisjoint(test)
    assert validation.isdisjoint(challenge)
    assert test.isdisjoint(challenge)


def test_stair_train_difficulty_distribution():
    api = _api()

    counts = Counter(
        api.stair_curriculum_identity(
            seed,
            split="train",
        ).difficulty
        for seed in api.stair_curriculum_seeds("train")
    )

    assert counts == {
        "easy": 250,
        "medium": 400,
        "hard": 350,
    }


def test_easy_stairs_are_uniform_ascending_and_conservative():
    api = _api()

    checked = 0

    for seed in api.stair_curriculum_seeds("train"):
        identity = api.stair_curriculum_identity(
            seed,
            split="train",
        )

        if identity.difficulty != "easy":
            continue

        course = api.phase1_stairs_curriculum_course(
            seed,
            split="train",
        )

        assert course.direction == "ascent"
        assert course.profile == "uniform"

        assert 2 <= course.step_count <= 4

        assert all(
            0.050 <= rise <= 0.062
            for rise in course.step_rises_m
        )

        assert all(
            0.270 <= depth <= 0.340
            for depth in course.tread_depths_m
        )

        assert len(set(course.step_rises_m)) == 1
        assert len(set(course.tread_depths_m)) == 1

        assert course.start_height_m == pytest.approx(0.0)
        assert course.goal_height_m > 0.0

        checked += 1

    assert checked == 250


def test_medium_stairs_expand_uniform_ascent_envelope():
    api = _api()

    checked = 0

    for seed in api.stair_curriculum_seeds("train"):
        identity = api.stair_curriculum_identity(
            seed,
            split="train",
        )

        if identity.difficulty != "medium":
            continue

        course = api.phase1_stairs_curriculum_course(
            seed,
            split="train",
        )

        assert course.direction == "ascent"
        assert course.profile == "uniform"

        assert 3 <= course.step_count <= 5

        assert all(
            0.052 <= rise <= 0.068
            for rise in course.step_rises_m
        )

        assert all(
            0.230 <= depth <= 0.310
            for depth in course.tread_depths_m
        )

        assert course.goal_height_m > course.start_height_m

        checked += 1

    assert checked == 400


def test_hard_distribution_contains_all_four_structural_modes():
    api = _api()

    modes = Counter()

    for seed in api.stair_curriculum_seeds("train"):
        identity = api.stair_curriculum_identity(
            seed,
            split="train",
        )

        if identity.difficulty != "hard":
            continue

        course = api.phase1_stairs_curriculum_course(
            seed,
            split="train",
        )

        modes[
            (
                course.direction,
                course.profile,
            )
        ] += 1

    assert modes[("ascent", "uniform")] >= 80
    assert modes[("ascent", "nonuniform")] >= 80
    assert modes[("descent", "uniform")] >= 80
    assert modes[("descent", "nonuniform")] >= 80


def test_hard_nonuniform_stairs_really_vary_inside_episode():
    api = _api()

    checked = 0

    for seed in api.stair_curriculum_seeds("train"):
        identity = api.stair_curriculum_identity(
            seed,
            split="train",
        )

        if (
            identity.difficulty != "hard"
            or identity.profile != "nonuniform"
        ):
            continue

        course = api.phase1_stairs_curriculum_course(
            seed,
            split="train",
        )

        assert (
            len(set(course.step_rises_m)) > 1
            or len(set(course.tread_depths_m)) > 1
        )

        checked += 1

    assert checked >= 160


def test_all_train_stairs_are_physically_bounded_and_monotone():
    api = _api()

    for seed in api.stair_curriculum_seeds("train"):
        course = api.phase1_stairs_curriculum_course(
            seed,
            split="train",
        )

        assert 2 <= course.step_count <= 6
        assert len(course.step_rises_m) == course.step_count
        assert len(course.tread_depths_m) == course.step_count
        assert len(course.step_top_heights_m) == course.step_count

        assert all(
            0.045 <= rise <= 0.075
            for rise in course.step_rises_m
        )

        assert all(
            0.180 <= depth <= 0.340
            for depth in course.tread_depths_m
        )

        assert course.start_height_m >= 0.0
        assert course.goal_height_m >= 0.0

        if course.direction == "ascent":
            heights = (
                course.start_height_m,
                *course.step_top_heights_m,
            )

            assert all(
                upper > lower
                for lower, upper in zip(
                    heights[:-1],
                    heights[1:],
                )
            )

            assert course.goal_height_m == pytest.approx(
                course.step_top_heights_m[-1]
            )

        else:
            heights = (
                course.start_height_m,
                *course.step_top_heights_m,
            )

            assert all(
                upper < lower
                for lower, upper in zip(
                    heights[:-1],
                    heights[1:],
                )
            )

            assert course.goal_height_m == pytest.approx(0.0)
            assert course.step_top_heights_m[-1] == pytest.approx(0.0)


def test_support_boxes_are_contiguous_and_match_stair_heights():
    api = _api()

    for seed in (
        12000,
        12249,
        12250,
        12649,
        12650,
        12651,
        12652,
        12653,
        12999,
    ):
        course = api.phase1_stairs_curriculum_course(
            seed,
            split="train",
        )

        assert len(course.step_boxes) == course.step_count

        previous_end = course.start_platform_end_x_m

        for box, expected_height in zip(
            course.step_boxes,
            course.step_top_heights_m,
        ):
            start_x = (
                box.center_xyz_m[0]
                - 0.5 * box.size_xyz_m[0]
            )

            end_x = (
                box.center_xyz_m[0]
                + 0.5 * box.size_xyz_m[0]
            )

            top_z = (
                box.center_xyz_m[2]
                + 0.5 * box.size_xyz_m[2]
            )

            assert start_x == pytest.approx(
                previous_end,
                abs=1.0e-9,
            )

            assert top_z == pytest.approx(
                expected_height,
                abs=1.0e-9,
            )

            previous_end = end_x

        assert previous_end == pytest.approx(
            course.goal_platform_start_x_m,
            abs=1.0e-9,
        )


def test_challenge_is_nonuniform_and_contains_ascent_and_descent():
    api = _api()

    directions = set()

    for seed in api.stair_curriculum_seeds("challenge"):
        identity = api.stair_curriculum_identity(
            seed,
            split="challenge",
        )

        course = api.phase1_stairs_curriculum_course(
            seed,
            split="challenge",
        )

        assert identity.difficulty == "challenge"
        assert course.profile == "nonuniform"

        directions.add(course.direction)

        assert 4 <= course.step_count <= 6

    assert directions == {
        "ascent",
        "descent",
    }


def test_generation_is_exactly_reproducible():
    api = _api()

    for split, seed in (
        ("train", 12000),
        ("train", 12651),
        ("validation", 13017),
        ("test", 14091),
        ("challenge", 14507),
    ):
        first = api.phase1_stairs_curriculum_course(
            seed,
            split=split,
        )

        second = api.phase1_stairs_curriculum_course(
            seed,
            split=split,
        )

        assert first == second


def test_public_observation_contains_geometry_but_no_hidden_reward_state():
    api = _api()

    course = api.phase1_stairs_curriculum_course(
        12653,
        split="train",
    )

    observation = course.to_observation()

    assert observation["frame_id"] == "world"
    assert observation["course_profile"] == "phase1_stairs_curriculum_v1"

    stairs = observation["stairs"]

    assert stairs["direction"] == course.direction
    assert stairs["profile"] == course.profile
    assert stairs["step_rises_m"] == list(course.step_rises_m)
    assert stairs["tread_depths_m"] == list(course.tread_depths_m)
    assert stairs["top_heights_m"] == list(course.step_top_heights_m)

    assert "reward" not in observation
    assert "success" not in observation
    assert "progress" not in observation


def test_unknown_split_and_wrong_seed_fail_closed():
    api = _api()

    with pytest.raises(ValueError):
        api.stair_curriculum_seeds("banana")

    with pytest.raises(ValueError):
        api.stair_curriculum_identity(
            12000,
            split="validation",
        )

    with pytest.raises(ValueError):
        api.phase1_stairs_curriculum_course(
            999,
            split="train",
        )
