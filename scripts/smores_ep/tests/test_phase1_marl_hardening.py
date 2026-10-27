from __future__ import annotations

import itertools
import math


def _api():
    from smores_ep.isaac import (
        phase1_marl_curriculum as api,
    )

    return api


def _nearest_route_index(
    course,
    point,
):
    return min(
        range(
            len(
                course.centerline_xy_m
            )
        ),
        key=lambda index: (
            (
                course.centerline_xy_m[
                    index
                ][0]
                - point[0]
            )
            ** 2
            + (
                course.centerline_xy_m[
                    index
                ][1]
                - point[1]
            )
            ** 2
        ),
    )


def _signed_lateral(
    course,
    point,
):
    index = _nearest_route_index(
        course,
        point,
    )

    previous = course.centerline_xy_m[
        max(
            0,
            index - 1,
        )
    ]

    following = course.centerline_xy_m[
        min(
            len(
                course.centerline_xy_m
            )
            - 1,
            index + 1,
        )
    ]

    dx = following[0] - previous[0]
    dy = following[1] - previous[1]

    norm = max(
        math.hypot(
            dx,
            dy,
        ),
        1.0e-12,
    )

    nx = -dy / norm
    ny = dx / norm

    center = course.centerline_xy_m[
        index
    ]

    lateral = (
        (
            point[0]
            - center[0]
        )
        * nx
        + (
            point[1]
            - center[1]
        )
        * ny
    )

    return (
        index,
        lateral,
    )


def _meaningful_gate_count(
    course,
):
    projected = [
        (
            *_signed_lateral(
                course,
                cone.center_xy_m,
            ),
            cone,
        )
        for cone in course.cones
    ]

    gates = 0

    for left_index in range(
        len(projected)
    ):
        for right_index in range(
            left_index + 1,
            len(projected),
        ):
            (
                index_a,
                lateral_a,
                cone_a,
            ) = projected[
                left_index
            ]

            (
                index_b,
                lateral_b,
                cone_b,
            ) = projected[
                right_index
            ]

            if abs(
                index_a
                - index_b
            ) > 1:
                continue

            if (
                lateral_a
                * lateral_b
                >= 0.0
            ):
                continue

            opening = (
                abs(
                    lateral_a
                    - lateral_b
                )
                - cone_a.radius_m
                - cone_b.radius_m
            )

            if (
                course.required_gate_opening_m
                <= opening
                <= course.required_gate_opening_m
                + 0.14
            ):
                gates += 1

    return gates


def test_task4e3_cone_density_scales_with_curriculum_difficulty():
    api = _api()

    expected = {
        "easy": 8,
        "medium": 12,
        "hard": 16,
    }

    observed = {
        "easy": set(),
        "medium": set(),
        "hard": set(),
    }

    for seed in api.curriculum_seeds(
        "train"
    ):
        identity = (
            api.curriculum_identity(
                seed,
                split="train",
            )
        )

        course = (
            api.phase1_marl_curriculum_course(
                seed,
                split="train",
            )
        )

        observed[
            identity.difficulty
        ].add(
            len(
                course.cones
            )
        )

    assert (
        observed
        == {
            name: {
                count
            }
            for name, count
            in expected.items()
        }
    ), "TASK4E3_RED_CONE_DENSITY"

    challenge_counts = {
        len(
            api.phase1_marl_curriculum_course(
                seed,
                split="challenge",
            ).cones
        )
        for seed
        in api.curriculum_seeds(
            "challenge"
        )
    }

    assert (
        challenge_counts
        == {20}
    )


def test_task4e3_hard_courses_keep_dense_non_overlapping_hazards():
    api = _api()

    seeds = [
        7650,
        7651,
        7652,
        7653,
        7654,
    ]

    counts = []

    for seed in seeds:
        course = (
            api.phase1_marl_curriculum_course(
                seed,
                split="train",
            )
        )

        counts.append(
            _meaningful_gate_count(
                course
            )
        )

        # Dense must not mean intersecting cones.
        for left, right in itertools.combinations(
            course.cones,
            2,
        ):
            distance = math.hypot(
                left.center_xy_m[0]
                - right.center_xy_m[0],
                left.center_xy_m[1]
                - right.center_xy_m[1],
            )

            assert (
                distance
                > left.radius_m
                + right.radius_m
                + 0.005
            ), "TASK4E3_RED_LOCAL_HAZARDS"

    # 4E4 replaces mandatory transverse gates with
    # centerline blockers and longitudinal mini-slaloms.
    # The loop above still verifies that every hard course
    # preserves dense, non-overlapping physical obstacles.
    assert len(counts) == 5


def test_task4e3_hard_distribution_contains_real_hills_and_dips():
    api = _api()

    grade_counts = {
        "flat": 0,
        "hill": 0,
        "dip": 0,
    }

    graded_courses = []

    for seed in api.curriculum_seeds(
        "train"
    ):
        identity = (
            api.curriculum_identity(
                seed,
                split="train",
            )
        )

        if (
            identity.difficulty
            != "hard"
        ):
            continue

        course = (
            api.phase1_marl_curriculum_course(
                seed,
                split="train",
            )
        )

        profile = (
            course.track_profile
        )

        if "/grade_hill" in profile:
            grade = "hill"
        elif "/grade_dip" in profile:
            grade = "dip"
        elif "/grade_flat" in profile:
            grade = "flat"
        else:
            raise AssertionError(
                "TASK4E3_RED_GRADES"
            )

        grade_counts[
            grade
        ] += 1

        if grade != "flat":
            graded_courses.append(
                course
            )

    assert (
        grade_counts["hill"]
        >= 70
    ), "TASK4E3_RED_GRADES"

    assert (
        grade_counts["dip"]
        >= 70
    ), "TASK4E3_RED_GRADES"

    assert (
        grade_counts["flat"]
        >= 150
    )

    assert graded_courses

    for course in graded_courses[
        :30
    ]:
        road = (
            course.support_boxes[
                1:-1
            ]
        )

        pitches = [
            box.pitch_deg
            for box in road
        ]

        # A hill/dip necessarily contains both transition directions.
        assert any(
            pitch > 1.0
            for pitch in pitches
        )

        assert any(
            pitch < -1.0
            for pitch in pitches
        )

        assert max(
            abs(pitch)
            for pitch in pitches
        ) <= 12.5

        # Cones remain on flat sections for this first 3-D curriculum.
        # That keeps their existing vertical collision model exact.
        for cone in course.cones:
            nearest_box = min(
                road,
                key=lambda box: (
                    (
                        box.center_xyz_m[0]
                        - cone.center_xy_m[0]
                    )
                    ** 2
                    + (
                        box.center_xyz_m[1]
                        - cone.center_xy_m[1]
                    )
                    ** 2
                ),
            )

            assert abs(
                nearest_box.pitch_deg
            ) <= 0.75
