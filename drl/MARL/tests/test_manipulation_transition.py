from __future__ import annotations

import dataclasses
import inspect
import json

import pytest

from drl.MARL.phase1_reward import (
    ActionOutcomeSignals,
)


def _course():
    from smores_ep.isaac.phase1_manipulation_curriculum import (
        phase1_manipulation_curriculum_course,
    )

    return (
        phase1_manipulation_curriculum_course(
            20650,
            split="train",
        )
    )


def _modules(
    course,
    standoff_m,
):
    bx, by, _ = (
        course.button_center_xyz_m
    )

    nx, ny = (
        course.press_direction_world_xy
    )

    cx = (
        bx
        - nx * standoff_m
    )

    cy = (
        by
        - ny * standoff_m
    )

    offsets = (
        (-0.03, -0.03),
        (-0.03,  0.00),
        (-0.03, +0.03),
        ( 0.00, -0.03),
        ( 0.00, +0.03),
        (+0.03, -0.03),
        (+0.03,  0.00),
        (+0.03, +0.03),
    )

    return {
        f"smores_{index:02d}":
            (
                cx + dx,
                cy + dy,
                0.05,
            )
        for index, (dx, dy)
        in enumerate(
            offsets,
            start=1,
        )
    }


def _ee_center(
    course,
    *,
    normal_offset_m=-0.03,
):
    from smores_ep.isaac.obstacle_course import (
        BUTTON_PLUNGER_DEPTH_M,
    )

    bx, by, bz = (
        course.button_center_xyz_m
    )

    nx, ny = (
        course.press_direction_world_xy
    )

    return (
        bx
        - (
            0.5
            * BUTTON_PLUNGER_DEPTH_M
        )
        * nx
        + normal_offset_m * nx,

        by
        - (
            0.5
            * BUTTON_PLUNGER_DEPTH_M
        )
        * ny
        + normal_offset_m * ny,

        bz,
    )


def _observation(
    course,
    *,
    standoff_m,
    depression_m=0.0,
    connection_count=7,
    normal_offset_m=-0.03,
):
    from drl.MARL.manipulation_reward import (
        ManipulationObservation,
    )

    nx, ny = (
        course.press_direction_world_xy
    )

    return ManipulationObservation(
        module_xyz_m=_modules(
            course,
            standoff_m,
        ),

        end_effector_bottom_center_xyz_m=(
            _ee_center(
                course,
                normal_offset_m=(
                    normal_offset_m
                ),
            )
        ),

        end_effector_bottom_normal_world=(
            nx,
            ny,
            0.0,
        ),

        button_depression_m=(
            depression_m
        ),

        connection_count=(
            connection_count
        ),
    )


def test_callable_contract_matches_stairs_and_gap():
    from drl.MARL.gap_transition import (
        GapTransitionEvaluator,
    )

    from drl.MARL.stairs_transition import (
        StairsTransitionEvaluator,
    )

    from drl.MARL.manipulation_transition import (
        ManipulationTransitionEvaluator,
    )

    def parameter_names(
        cls,
    ):
        return tuple(
            inspect.signature(
                cls.__call__
            ).parameters
        )

    expected = (
        "self",
        "env_index",
        "previous_state",
        "next_state",
        "actuation_result",
        "physics_result",
        "episode_step",
    )

    assert (
        parameter_names(
            ManipulationTransitionEvaluator
        )
        == expected
    )

    assert (
        parameter_names(
            StairsTransitionEvaluator
        )
        == expected
    )

    assert (
        parameter_names(
            GapTransitionEvaluator
        )
        == expected
    )


def test_approach_transition_evaluates_raw_terms():
    from drl.MARL.manipulation_transition import (
        ManipulationTransitionEvaluator,
    )

    course = _course()

    evaluator = (
        ManipulationTransitionEvaluator(
            course=course,
        )
    )

    result = evaluator(
        env_index=0,
        previous_state=_observation(
            course,
            standoff_m=1.20,
        ),
        next_state=_observation(
            course,
            standoff_m=1.00,
        ),
        actuation_result=None,
        physics_result=None,
        episode_step=0,
    )

    assert (
        result.raw_reward
        .team
        .approach_progress
        > 0.0
    )

    assert result.success is False
    assert result.terminated is False
    assert result.failure_reason is None
    assert result.episode_step == 0
    assert result.env_index == 0


def test_success_terminates_episode():
    from drl.MARL.manipulation_reward import (
        BUTTON_SUCCESS_DEPRESSION_M,
    )

    from drl.MARL.manipulation_transition import (
        ManipulationTransitionEvaluator,
    )

    course = _course()

    evaluator = (
        ManipulationTransitionEvaluator(
            course=course,
        )
    )

    result = evaluator(
        env_index=2,
        previous_state=_observation(
            course,
            standoff_m=0.325,
            depression_m=0.0020,
        ),
        next_state=_observation(
            course,
            standoff_m=0.325,
            depression_m=(
                BUTTON_SUCCESS_DEPRESSION_M
            ),
        ),
        actuation_result=None,
        physics_result=None,
        episode_step=17,
    )

    assert result.success is True
    assert result.terminated is True
    assert result.failure_reason is None

    assert (
        result.raw_reward
        .team
        .success
        == 1.0
    )


def test_connection_loss_is_terminal_failure():
    from drl.MARL.manipulation_transition import (
        ManipulationTransitionEvaluator,
    )

    course = _course()

    evaluator = (
        ManipulationTransitionEvaluator(
            course=course,
        )
    )

    result = evaluator(
        env_index=0,
        previous_state=_observation(
            course,
            standoff_m=0.325,
            connection_count=7,
        ),
        next_state=_observation(
            course,
            standoff_m=0.325,
            connection_count=6,
        ),
        actuation_result=None,
        physics_result=None,
        episode_step=8,
    )

    assert result.success is False
    assert result.terminated is True

    assert (
        result.failure_reason
        == "connection_loss"
    )

    assert (
        result.raw_reward
        .team
        .connection_loss
        == pytest.approx(
            1.0 / 7.0
        )
    )


def test_builder_can_adapt_runtime_states():
    from drl.MARL.manipulation_transition import (
        ManipulationTransitionEvaluator,
    )

    course = _course()

    previous_observation = (
        _observation(
            course,
            standoff_m=0.90,
        )
    )

    next_observation = (
        _observation(
            course,
            standoff_m=0.70,
        )
    )

    previous_raw = object()
    next_raw = object()

    seen = []

    def builder(
        state,
    ):
        seen.append(
            state
        )

        if state is previous_raw:
            return (
                previous_observation
            )

        if state is next_raw:
            return (
                next_observation
            )

        raise AssertionError(
            "unexpected raw state"
        )

    evaluator = (
        ManipulationTransitionEvaluator(
            course=course,
            observation_builder=builder,
        )
    )

    result = evaluator(
        env_index=0,
        previous_state=previous_raw,
        next_state=next_raw,
        actuation_result=None,
        physics_result=None,
        episode_step=3,
    )

    assert seen == [
        previous_raw,
        next_raw,
    ]

    assert (
        result.raw_reward
        .team
        .approach_progress
        > 0.0
    )


def test_signal_builder_receives_exact_transition_context():
    from drl.MARL.manipulation_transition import (
        ManipulationTransitionEvaluator,
    )

    course = _course()

    actuation = object()
    physics = object()

    calls = []

    def signal_builder(
        *,
        env_index,
        previous_state,
        next_state,
        actuation_result,
        physics_result,
        episode_step,
    ):
        calls.append(
            (
                env_index,
                previous_state,
                next_state,
                actuation_result,
                physics_result,
                episode_step,
            )
        )

        return ActionOutcomeSignals(
            invalid_owner_ids=frozenset(
                {
                    "smores_01",
                }
            )
        )

    previous = _observation(
        course,
        standoff_m=0.325,
    )

    next_state = _observation(
        course,
        standoff_m=0.325,
    )

    evaluator = (
        ManipulationTransitionEvaluator(
            course=course,
            signal_builder=(
                signal_builder
            ),
        )
    )

    result = evaluator(
        env_index=4,
        previous_state=previous,
        next_state=next_state,
        actuation_result=actuation,
        physics_result=physics,
        episode_step=11,
    )

    assert calls == [
        (
            4,
            previous,
            next_state,
            actuation,
            physics,
            11,
        )
    ]

    assert (
        result.raw_reward
        .owners[
            "smores_01"
        ]
        .invalid
        == 1.0
    )


def test_bad_observation_builder_fails_closed():
    from drl.MARL.manipulation_transition import (
        ManipulationTransitionEvaluator,
    )

    course = _course()

    evaluator = (
        ManipulationTransitionEvaluator(
            course=course,
            observation_builder=(
                lambda state:
                    {"not": "an observation"}
            ),
        )
    )

    with pytest.raises(
        TypeError
    ):
        evaluator(
            env_index=0,
            previous_state=object(),
            next_state=object(),
            actuation_result=None,
            physics_result=None,
            episode_step=0,
        )


def test_bad_signal_builder_fails_closed():
    from drl.MARL.manipulation_transition import (
        ManipulationTransitionEvaluator,
    )

    course = _course()

    evaluator = (
        ManipulationTransitionEvaluator(
            course=course,
            signal_builder=(
                lambda **kwargs:
                    {"bad": "signals"}
            ),
        )
    )

    with pytest.raises(
        TypeError
    ):
        evaluator(
            env_index=0,
            previous_state=_observation(
                course,
                standoff_m=0.325,
            ),
            next_state=_observation(
                course,
                standoff_m=0.325,
            ),
            actuation_result=None,
            physics_result=None,
            episode_step=0,
        )


def test_raw_jsonl_logger_contract(
    tmp_path,
):
    from drl.MARL.manipulation_reward import (
        BUTTON_SUCCESS_DEPRESSION_M,
    )

    from drl.MARL.manipulation_transition import (
        MANIPULATION_RAW_TERMS_SCHEMA,
        ManipulationRawTermsJsonlLogger,
        ManipulationTransitionEvaluator,
    )

    course = _course()

    path = (
        tmp_path
        / "manipulation_raw.jsonl"
    )

    logger = (
        ManipulationRawTermsJsonlLogger(
            path
        )
    )

    evaluator = (
        ManipulationTransitionEvaluator(
            course=course,
            logger=logger,
        )
    )

    result = evaluator(
        env_index=3,
        previous_state=_observation(
            course,
            standoff_m=0.325,
            depression_m=0.002,
        ),
        next_state=_observation(
            course,
            standoff_m=0.325,
            depression_m=(
                BUTTON_SUCCESS_DEPRESSION_M
            ),
        ),
        actuation_result=None,
        physics_result=None,
        episode_step=21,
    )

    lines = (
        path.read_text()
        .splitlines()
    )

    assert len(lines) == 1

    record = json.loads(
        lines[0]
    )

    assert (
        record[
            "schema_version"
        ]
        == MANIPULATION_RAW_TERMS_SCHEMA
    )

    assert (
        record[
            "env_index"
        ]
        == 3
    )

    assert (
        record[
            "episode_step"
        ]
        == 21
    )

    assert (
        record[
            "course"
        ][
            "seed"
        ]
        == course.seed
    )

    assert (
        record[
            "course"
        ][
            "split"
        ]
        == course.split
    )

    assert (
        record[
            "course"
        ][
            "difficulty"
        ]
        == course.difficulty
    )

    assert (
        record[
            "terminal"
        ]
        is True
    )

    assert (
        record[
            "success"
        ]
        is True
    )

    assert (
        record[
            "failure_reason"
        ]
        is None
    )

    assert (
        record[
            "team_raw_terms"
        ][
            "success"
        ]
        == 1.0
    )

    assert (
        len(
            record[
                "owner_raw_terms"
            ]
        )
        == 8
    )

    serialized = (
        json.dumps(
            record
        )
        .lower()
    )

    assert "weight" not in serialized

    assert result.success is True


def test_logger_is_append_only(
    tmp_path,
):
    from drl.MARL.manipulation_transition import (
        ManipulationRawTermsJsonlLogger,
        ManipulationTransitionEvaluator,
    )

    course = _course()

    path = (
        tmp_path
        / "raw.jsonl"
    )

    evaluator = (
        ManipulationTransitionEvaluator(
            course=course,
            logger=(
                ManipulationRawTermsJsonlLogger(
                    path
                )
            ),
        )
    )

    for step in (
        0,
        1,
        2,
    ):
        evaluator(
            env_index=0,
            previous_state=_observation(
                course,
                standoff_m=(
                    1.20
                    - 0.10 * step
                ),
            ),
            next_state=_observation(
                course,
                standoff_m=(
                    1.10
                    - 0.10 * step
                ),
            ),
            actuation_result=None,
            physics_result=None,
            episode_step=step,
        )

    records = [
        json.loads(line)
        for line
        in path
        .read_text()
        .splitlines()
    ]

    assert [
        item[
            "episode_step"
        ]
        for item
        in records
    ] == [
        0,
        1,
        2,
    ]


def test_transition_result_contains_no_scalar_weighted_reward():
    from drl.MARL.manipulation_transition import (
        ManipulationTransitionEvaluation,
    )

    names = {
        field.name.lower()
        for field
        in dataclasses.fields(
            ManipulationTransitionEvaluation
        )
    }

    assert "reward" not in names
    assert "scalar_reward" not in names
    assert "weighted_reward" not in names

    assert (
        "raw_reward"
        in names
    )
