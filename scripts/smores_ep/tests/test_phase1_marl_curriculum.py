from __future__ import annotations

import importlib
import importlib.util
import math

import pytest


def _api():
    spec = importlib.util.find_spec(
        "smores_ep.isaac.phase1_marl_curriculum"
    )

    assert (
        spec is not None
    ), "TASK4E1_RED_MARL_CURRICULUM_MISSING"

    return importlib.import_module(
        "smores_ep.isaac.phase1_marl_curriculum"
    )


def _path_length(points):
    return sum(
        math.hypot(
            right[0] - left[0],
            right[1] - left[1],
        )
        for left, right
        in zip(points, points[1:])
    )


def _final_heading(points):
    left = points[-2]
    right = points[-1]

    return math.atan2(
        right[1] - left[1],
        right[0] - left[0],
    )


def _wrapped_abs(angle):
    return abs(
        math.atan2(
            math.sin(angle),
            math.cos(angle),
        )
    )


def test_task4e1_split_ranges_are_fixed_disjoint_and_expected_sizes():
    api = _api()

    train = api.curriculum_seeds("train")
    val = api.curriculum_seeds("validation")
    test = api.curriculum_seeds("test")
    challenge = api.curriculum_seeds("challenge")

    assert len(train) == 1000
    assert len(val) == 120
    assert len(test) == 200
    assert len(challenge) == 30

    assert train[0] == 7000
    assert train[-1] == 7999

    assert val[0] == 8000
    assert val[-1] == 8119

    assert test[0] == 9000
    assert test[-1] == 9199

    assert challenge[0] == 9500
    assert challenge[-1] == 9529

    sets = [
        set(train),
        set(val),
        set(test),
        set(challenge),
    ]

    for index, left in enumerate(sets):
        for right in sets[index + 1:]:
            assert left.isdisjoint(right)

    # Expert/reference family is explicitly outside MARL TRAIN.
    assert 5100 not in set(train)
    assert 5101 not in set(train)
    assert 5102 not in set(train)


def test_task4e1_generation_is_deterministic_and_split_enforced():
    api = _api()

    first = api.phase1_marl_curriculum_course(
        7000,
        split="train",
    )

    second = api.phase1_marl_curriculum_course(
        7000,
        split="train",
    )

    assert first == second

    assert first.track_profile.startswith(
        "marl_unseen_v1/"
    )

    with pytest.raises(
        ValueError,
        match="does not belong",
    ):
        api.phase1_marl_curriculum_course(
            7000,
            split="validation",
        )

    with pytest.raises(
        ValueError,
        match="does not belong",
    ):
        api.phase1_marl_curriculum_course(
            5100,
            split="train",
        )


def test_task4e1_full_train_distribution_is_physically_valid_and_diverse():
    api = _api()

    families = set()
    difficulties = set()
    lengths = []
    targets = set()
    cone_counts = []

    for seed in api.curriculum_seeds(
        "train"
    ):
        identity = api.curriculum_identity(
            seed,
            split="train",
        )

        course = api.phase1_marl_curriculum_course(
            seed,
            split="train",
        )

        families.add(identity.family)
        difficulties.add(identity.difficulty)

        assert (
            identity.family
            in course.track_profile
        )

        assert (
            identity.difficulty
            in course.track_profile
        )

        assert course.forward_only is True
        assert len(course.centerline_xy_m) >= 10

        assert (
            course.road_width_m
            > course.required_gate_opening_m
        )

        assert all(
            cone.static is True
            for cone in course.cones
        )

        # No physical side-wall primitives.
        assert not any(
            "wall"
            in box.semantic.lower()
            for box in course.support_boxes
        )

        assert not any(
            "side"
            in box.semantic.lower()
            and "road"
            in box.semantic.lower()
            for box in course.support_boxes
        )

        lengths.append(
            _path_length(
                course.centerline_xy_m
            )
        )

        targets.add(
            (
                round(
                    course.target_platform.center_xy_m[0],
                    2,
                ),
                round(
                    course.target_platform.center_xy_m[1],
                    2,
                ),
            )
        )

        cone_counts.append(
            len(course.cones)
        )

    assert families == {
        "straight",
        "gentle_curve",
        "s_curve",
        "chicane",
        "u_turn",
        "hairpin",
        "compound",
    }

    assert difficulties == {
        "easy",
        "medium",
        "hard",
    }

    # Real variation in route length, not merely obstacle jitter.
    assert min(lengths) < 3.5
    assert max(lengths) > 7.0
    assert max(lengths) - min(lengths) > 4.0

    # Seeded target geometry genuinely varies.
    assert len(targets) > 700

    # 4E3 hardening:
    # obstacle density now scales deliberately with difficulty.
    assert min(cone_counts) == 8
    assert max(cone_counts) == 16


def test_task4e1_u_turn_and_hairpin_reverse_heading_without_reverse_motion():
    api = _api()

    found = {}

    for seed in api.curriculum_seeds(
        "train"
    ):
        identity = api.curriculum_identity(
            seed,
            split="train",
        )

        if (
            identity.family
            in {"u_turn", "hairpin"}
            and identity.family
            not in found
        ):
            found[
                identity.family
            ] = (
                api.phase1_marl_curriculum_course(
                    seed,
                    split="train",
                )
            )

        if len(found) == 2:
            break

    assert set(found) == {
        "u_turn",
        "hairpin",
    }

    for family, course in found.items():
        heading = _wrapped_abs(
            _final_heading(
                course.centerline_xy_m
            )
        )

        assert heading >= math.radians(
            130.0
        ), family

        # Vehicle always progresses along the generated centerline:
        # the route itself turns around; no reverse command is required.
        assert course.forward_only is True


def test_task4e1_challenge_split_contains_only_structurally_hard_families():
    api = _api()

    families = {
        api.curriculum_identity(
            seed,
            split="challenge",
        ).family
        for seed in api.curriculum_seeds(
            "challenge"
        )
    }

    difficulties = {
        api.curriculum_identity(
            seed,
            split="challenge",
        ).difficulty
        for seed in api.curriculum_seeds(
            "challenge"
        )
    }

    assert difficulties == {
        "hard",
    }

    assert families == {
        "u_turn",
        "hairpin",
        "compound",
    }


def test_task4e1_public_observation_never_exposes_centerline_or_generator_path():
    api = _api()

    representative = (
        7000,
        7350,
        7700,
        8000,
        9000,
        9500,
    )

    for seed in representative:
        split = api.split_for_seed(
            seed
        )

        course = (
            api.phase1_marl_curriculum_course(
                seed,
                split=split,
            )
        )

        public = course.to_observation()

        assert "centerline_xy_m" not in repr(
            public
        )

        assert "waypoints" not in repr(
            public
        )
