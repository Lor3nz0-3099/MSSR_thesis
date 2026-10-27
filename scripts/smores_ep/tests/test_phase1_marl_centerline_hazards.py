from __future__ import annotations

import itertools
import math


def _api():
    from smores_ep.isaac import (
        phase1_marl_curriculum as api,
    )

    return api


def _projection(
    course,
    cone,
):
    points = (
        course.centerline_xy_m
    )

    index = min(
        range(len(points)),
        key=lambda i: (
            (
                points[i][0]
                - cone.center_xy_m[0]
            ) ** 2
            + (
                points[i][1]
                - cone.center_xy_m[1]
            ) ** 2
        ),
    )

    previous = points[
        max(0, index - 1)
    ]

    following = points[
        min(
            len(points) - 1,
            index + 1,
        )
    ]

    dx = (
        following[0]
        - previous[0]
    )

    dy = (
        following[1]
        - previous[1]
    )

    norm = max(
        math.hypot(dx, dy),
        1.0e-12,
    )

    nx = -dy / norm
    ny = dx / norm

    center = points[index]

    lateral = (
        (
            cone.center_xy_m[0]
            - center[0]
        )
        * nx
        + (
            cone.center_xy_m[1]
            - center[1]
        )
        * ny
    )

    return (
        index,
        lateral,
    )


def test_task4e4_every_hazard_zone_has_a_real_centerline_blocker():
    api = _api()

    expected = {
        "easy": 2,
        "medium": 3,
        "hard": 4,
    }

    representatives = {
        "easy": (
            7000,
            7001,
            7002,
        ),
        "medium": (
            7250,
            7254,
            7300,
        ),
        "hard": (
            7650,
            7651,
            7653,
            7654,
        ),
    }

    for difficulty, seeds in representatives.items():
        for seed in seeds:
            course = (
                api.phase1_marl_curriculum_course(
                    seed,
                    split="train",
                )
            )

            blockers = [
                (
                    index,
                    lateral,
                )
                for cone in course.cones
                for index, lateral
                in [
                    _projection(
                        course,
                        cone,
                    )
                ]
                if abs(lateral) <= 0.045
            ]

            assert (
                len(blockers)
                >= expected[
                    difficulty
                ]
            ), (
                "TASK4E4_RED_CENTERLINE_BLOCKERS "
                f"seed={seed} "
                f"blockers={blockers}"
            )


def test_task4e4_center_blocker_has_no_same_section_side_pair():
    api = _api()

    for seed in (
        7001,
        7254,
        7650,
        7651,
        7653,
        7654,
    ):
        course = (
            api.phase1_marl_curriculum_course(
                seed,
                split="train",
            )
        )

        projected = [
            (
                *_projection(
                    course,
                    cone,
                ),
                cone,
            )
            for cone in course.cones
        ]

        blockers = [
            (
                index,
                lateral,
            )
            for (
                index,
                lateral,
                _cone,
            ) in projected
            if abs(lateral) <= 0.045
        ]

        assert blockers, (
            "TASK4E4_RED_FREE_BYPASS "
            f"seed={seed}"
        )

        for (
            blocker_index,
            _,
        ) in blockers:
            same_section = [
                lateral
                for (
                    index,
                    lateral,
                    _cone,
                ) in projected
                if (
                    index
                    == blocker_index
                    and abs(
                        lateral
                    ) > 0.045
                )
            ]

            assert not same_section, (
                "TASK4E4_RED_FREE_BYPASS "
                f"seed={seed} "
                f"index={blocker_index} "
                f"side_cones={same_section}"
            )


def test_task4e4_preserves_density_and_non_overlap():
    api = _api()

    expected = {
        "easy": 8,
        "medium": 12,
        "hard": 16,
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

        assert (
            len(course.cones)
            == expected[
                identity.difficulty
            ]
        )

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
            )
