from __future__ import annotations

import json
import math

import pytest

from smores_ep.isaac.obstacle_course import (
    BUTTON_ASSEMBLY_SPAWN_RADIUS_M,
    BUTTON_ASSEMBLY_SPAWN_STANDOFF_M,
    BUTTON_PLATFORM_CENTER_XY_M,
    BUTTON_PLATFORM_SIZE_XY_M,
    BUTTON_SUPPORT_OFFSET_M,
)


def _nearest_cardinal_offset_deg(
    yaw_rad: float,
) -> float:
    degrees = math.degrees(yaw_rad) % 360.0

    return min(
        abs(
            (
                degrees
                - cardinal
                + 180.0
            )
            % 360.0
            - 180.0
        )
        for cardinal
        in (
            0.0,
            90.0,
            180.0,
            270.0,
        )
    )


def test_manipulation_split_sizes_and_seed_disjointness():
    from smores_ep.isaac.phase1_manipulation_curriculum import (
        manipulation_curriculum_seeds,
    )

    expected = {
        "train": 1000,
        "validation": 120,
        "test": 200,
        "challenge": 30,
    }

    all_sets = {}

    for split, size in expected.items():
        seeds = tuple(
            manipulation_curriculum_seeds(
                split
            )
        )

        assert len(seeds) == size
        assert len(set(seeds)) == size

        all_sets[split] = set(seeds)

    names = tuple(all_sets)

    for i, lhs in enumerate(names):
        for rhs in names[i + 1:]:
            assert (
                all_sets[lhs]
                .isdisjoint(
                    all_sets[rhs]
                )
            )


def test_train_difficulty_allocation():
    from smores_ep.isaac.phase1_manipulation_curriculum import (
        manipulation_curriculum_identity,
    )

    counts = {
        "easy": 0,
        "medium": 0,
        "hard": 0,
    }

    for seed in range(
        20000,
        21000,
    ):
        identity = (
            manipulation_curriculum_identity(
                seed,
                split="train",
            )
        )

        counts[
            identity.difficulty
        ] += 1

    assert counts == {
        "easy": 250,
        "medium": 400,
        "hard": 350,
    }


def test_validation_and_test_allocations():
    from smores_ep.isaac.phase1_manipulation_curriculum import (
        manipulation_curriculum_identity,
        manipulation_curriculum_seeds,
    )

    validation = [
        manipulation_curriculum_identity(
            seed,
            split="validation",
        ).difficulty
        for seed
        in manipulation_curriculum_seeds(
            "validation"
        )
    ]

    assert validation.count("easy") == 40
    assert validation.count("medium") == 40
    assert validation.count("hard") == 40

    test = [
        manipulation_curriculum_identity(
            seed,
            split="test",
        ).difficulty
        for seed
        in manipulation_curriculum_seeds(
            "test"
        )
    ]

    assert test.count("easy") == 60
    assert test.count("medium") == 70
    assert test.count("hard") == 70


@pytest.mark.parametrize(
    ("seed", "split", "difficulty", "low", "high"),
    (
        (
            20000,
            "train",
            "easy",
            8.0,
            18.0,
        ),
        (
            20250,
            "train",
            "medium",
            18.0,
            32.0,
        ),
        (
            20650,
            "train",
            "hard",
            32.0,
            44.0,
        ),
        (
            22500,
            "challenge",
            "challenge",
            40.0,
            45.0,
        ),
    ),
)
def test_fixture_normals_are_structurally_non_cardinal(
    seed,
    split,
    difficulty,
    low,
    high,
):
    from smores_ep.isaac.phase1_manipulation_curriculum import (
        phase1_manipulation_curriculum_course,
    )

    course = (
        phase1_manipulation_curriculum_course(
            seed,
            split=split,
        )
    )

    assert course.difficulty == difficulty

    nx, ny = (
        course.press_direction_world_xy
    )

    assert math.hypot(
        nx,
        ny,
    ) == pytest.approx(1.0)

    offset = (
        _nearest_cardinal_offset_deg(
            course.press_yaw_rad
        )
    )

    assert low <= offset <= high
    assert offset > 0.0


def test_button_height_envelopes():
    from smores_ep.isaac.phase1_manipulation_curriculum import (
        phase1_manipulation_curriculum_course,
        manipulation_curriculum_seeds,
    )

    ranges = {
        "easy": (
            0.115,
            0.155,
        ),
        "medium": (
            0.105,
            0.168,
        ),
        "hard": (
            0.100,
            0.17623346377092517,
        ),
    }

    for seed in manipulation_curriculum_seeds(
        "train"
    ):
        course = (
            phase1_manipulation_curriculum_course(
                seed,
                split="train",
            )
        )

        low, high = ranges[
            course.difficulty
        ]

        z = (
            course.button_center_xyz_m[2]
        )

        assert low <= z <= high

    for seed in manipulation_curriculum_seeds(
        "challenge"
    ):
        course = (
            phase1_manipulation_curriculum_course(
                seed,
                split="challenge",
            )
        )

        z = (
            course.button_center_xyz_m[2]
        )

        assert (
            0.100 <= z <= 0.115
            or
            0.161 <= z
            <= 0.17623346377092517
        )


def test_assembly_spawn_preserves_standoff_and_platform_clearance():
    from smores_ep.isaac.phase1_manipulation_curriculum import (
        phase1_manipulation_curriculum_course,
        manipulation_curriculum_seeds,
    )

    platform_cx, platform_cy = (
        BUTTON_PLATFORM_CENTER_XY_M
    )

    platform_sx, platform_sy = (
        BUTTON_PLATFORM_SIZE_XY_M
    )

    xmin = (
        platform_cx
        - 0.5 * platform_sx
    )

    xmax = (
        platform_cx
        + 0.5 * platform_sx
    )

    ymin = (
        platform_cy
        - 0.5 * platform_sy
    )

    ymax = (
        platform_cy
        + 0.5 * platform_sy
    )

    clearance = 0.08

    for split in (
        "train",
        "validation",
        "test",
        "challenge",
    ):
        for seed in manipulation_curriculum_seeds(
            split
        ):
            course = (
                phase1_manipulation_curriculum_course(
                    seed,
                    split=split,
                )
            )

            bx, by, _ = (
                course.button_center_xyz_m
            )

            sx, sy = (
                course.assembly_spawn_center_xy_m
            )

            nx, ny = (
                course.press_direction_world_xy
            )

            signed = (
                (bx - sx) * nx
                + (by - sy) * ny
            )

            assert signed == pytest.approx(
                BUTTON_ASSEMBLY_SPAWN_STANDOFF_M
            )

            margin = (
                BUTTON_ASSEMBLY_SPAWN_RADIUS_M
                + clearance
            )

            assert (
                xmin + margin
                <= sx
                <= xmax - margin
            )

            assert (
                ymin + margin
                <= sy
                <= ymax - margin
            )


def test_manipulation_ready_pose_is_exactly_0325m_from_button():
    from smores_ep.isaac.phase1_manipulation_curriculum import (
        phase1_manipulation_curriculum_course,
    )

    course = (
        phase1_manipulation_curriculum_course(
            20650,
            split="train",
        )
    )

    bx, by, _ = (
        course.button_center_xyz_m
    )

    rx, ry = (
        course.manipulation_ready_base_xy_m
    )

    nx, ny = (
        course.press_direction_world_xy
    )

    standoff = (
        (bx - rx) * nx
        + (by - ry) * ny
    )

    assert standoff == pytest.approx(
        0.325
    )

    assert (
        course.manipulation_ready_standoff_m
        == pytest.approx(
            0.325
        )
    )

    assert (
        course.base_drive_axis_xy[0]
        == pytest.approx(-nx)
    )

    assert (
        course.base_drive_axis_xy[1]
        == pytest.approx(-ny)
    )


def test_wall_and_plunger_follow_arbitrary_press_yaw():
    from smores_ep.isaac.phase1_manipulation_curriculum import (
        phase1_manipulation_curriculum_course,
    )

    course = (
        phase1_manipulation_curriculum_course(
            20651,
            split="train",
        )
    )

    wall = course.button_wall
    plunger = course.button_plunger

    assert (
        wall.yaw_deg
        == pytest.approx(
            math.degrees(
                course.press_yaw_rad
            )
        )
    )

    assert (
        plunger.yaw_deg
        == pytest.approx(
            wall.yaw_deg
        )
    )

    bx, by, _ = (
        course.button_center_xyz_m
    )

    wx, wy, _ = (
        wall.center_xyz_m
    )

    nx, ny = (
        course.press_direction_world_xy
    )

    assert (
        (wx - bx) * nx
        + (wy - by) * ny
        == pytest.approx(
            BUTTON_SUPPORT_OFFSET_M
        )
    )


def test_challenge_covers_all_quadrants_and_both_height_extremes():
    from smores_ep.isaac.phase1_manipulation_curriculum import (
        manipulation_curriculum_seeds,
        phase1_manipulation_curriculum_course,
    )

    quadrants = set()
    low = 0
    high = 0

    for seed in manipulation_curriculum_seeds(
        "challenge"
    ):
        course = (
            phase1_manipulation_curriculum_course(
                seed,
                split="challenge",
            )
        )

        nx, ny = (
            course.press_direction_world_xy
        )

        quadrants.add(
            (
                1 if nx >= 0.0 else -1,
                1 if ny >= 0.0 else -1,
            )
        )

        z = (
            course.button_center_xyz_m[2]
        )

        if z <= 0.115:
            low += 1

        if z >= 0.161:
            high += 1

    assert len(quadrants) == 4
    assert low > 0
    assert high > 0


def test_train_has_at_least_950_unique_structural_tasks():
    from smores_ep.isaac.phase1_manipulation_curriculum import (
        manipulation_curriculum_seeds,
        phase1_manipulation_curriculum_course,
    )

    structures = set()

    for seed in manipulation_curriculum_seeds(
        "train"
    ):
        course = (
            phase1_manipulation_curriculum_course(
                seed,
                split="train",
            )
        )

        x, y, z = (
            course.button_center_xyz_m
        )

        structures.add(
            (
                round(x, 4),
                round(y, 4),
                round(z, 4),
                round(
                    course.press_yaw_rad,
                    4,
                ),
            )
        )

    assert len(structures) >= 950


def test_generation_is_deterministic():
    from smores_ep.isaac.phase1_manipulation_curriculum import (
        phase1_manipulation_curriculum_course,
    )

    a = (
        phase1_manipulation_curriculum_course(
            20777,
            split="train",
        )
    )

    b = (
        phase1_manipulation_curriculum_course(
            20777,
            split="train",
        )
    )

    assert a == b


def test_observation_has_geometry_but_no_reward_oracle():
    from smores_ep.isaac.phase1_manipulation_curriculum import (
        phase1_manipulation_curriculum_course,
    )

    course = (
        phase1_manipulation_curriculum_course(
            22500,
            split="challenge",
        )
    )

    observation = (
        course.to_observation()
    )

    assert (
        observation["course_profile"]
        == "phase1_mobile_manipulator8_manipulation"
    )

    assert (
        observation["button"]
        ["center_xyz_m"]
        == pytest.approx(
            course.button_center_xyz_m
        )
    )

    serialized = (
        json.dumps(
            observation
        )
        .lower()
    )

    assert "reward" not in serialized
    assert "success" not in serialized
    assert "progress" not in serialized


def test_invalid_split_and_seed_fail_closed():
    from smores_ep.isaac.phase1_manipulation_curriculum import (
        manipulation_curriculum_identity,
        manipulation_curriculum_seeds,
        phase1_manipulation_curriculum_course,
    )

    with pytest.raises(ValueError):
        tuple(
            manipulation_curriculum_seeds(
                "banana"
            )
        )

    with pytest.raises(ValueError):
        manipulation_curriculum_identity(
            20000,
            split="validation",
        )

    with pytest.raises(ValueError):
        phase1_manipulation_curriculum_course(
            19999,
            split="train",
        )
