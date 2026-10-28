from __future__ import annotations

import json
from pathlib import Path

import pytest

from drl.MARL.phase1_reward import (
    ActionOutcomeSignals,
)
from smores_ep.config.geometry import (
    SmoresGeometry,
)
from smores_ep.isaac.phase1_stairs_curriculum import (
    phase1_stairs_curriculum_course,
)


OWNERS = tuple(
    f"smores_{index:02d}"
    for index in range(1, 9)
)


def _state(
    *,
    x: float,
    y: float,
    z: float,
):
    return {
        owner: (x, y, z)
        for owner in OWNERS
    }


def _neutral_signals(
    env_index,
    actuation_result,
    previous_state,
    next_state,
):
    del (
        env_index,
        actuation_result,
        previous_state,
        next_state,
    )

    return ActionOutcomeSignals()


def test_stairs_evaluator_matches_existing_transition_callable_contract():
    from drl.MARL.stairs_transition import (
        StairsTransitionEvaluator,
    )

    course = phase1_stairs_curriculum_course(
        12000,
        split="train",
    )

    evaluator = StairsTransitionEvaluator(
        courses=(course,),
        max_episode_steps=100,
        signals_provider=_neutral_signals,
    )

    previous = _state(
        x=course.first_transition_x_m - 0.08,
        y=0.0,
        z=SmoresGeometry().wheel_radius_m,
    )

    next_state = _state(
        x=course.first_transition_x_m - 0.04,
        y=0.0,
        z=SmoresGeometry().wheel_radius_m,
    )

    result = evaluator(
        env_index=0,
        previous_state=previous,
        next_state=next_state,
        actuation_result=object(),
        physics_result=object(),
        episode_step=1,
    )

    assert result.raw_reward.team.profile_progress > 0.0
    assert result.signals == ActionOutcomeSignals()
    assert result.terminated is False
    assert result.truncated is False


def test_success_terminates_and_wins_over_time_limit():
    from drl.MARL.stairs_transition import (
        StairsTransitionEvaluator,
    )

    course = phase1_stairs_curriculum_course(
        12000,
        split="train",
    )

    evaluator = StairsTransitionEvaluator(
        courses=(course,),
        max_episode_steps=5,
        signals_provider=_neutral_signals,
    )

    r = SmoresGeometry().wheel_radius_m

    previous = _state(
        x=course.goal_platform_start_x_m - 0.02,
        y=0.0,
        z=course.goal_height_m + r,
    )

    next_state = _state(
        x=course.goal_platform_start_x_m + 0.20,
        y=0.0,
        z=course.goal_height_m + r,
    )

    result = evaluator(
        env_index=0,
        previous_state=previous,
        next_state=next_state,
        actuation_result=None,
        physics_result=None,
        episode_step=5,
    )

    assert result.raw_reward.team.success == pytest.approx(1.0)
    assert result.terminated is True
    assert result.truncated is False


def test_time_limit_truncates_only_when_not_successful():
    from drl.MARL.stairs_transition import (
        StairsTransitionEvaluator,
    )

    course = phase1_stairs_curriculum_course(
        12651,
        split="train",
    )

    evaluator = StairsTransitionEvaluator(
        courses=(course,),
        max_episode_steps=7,
        signals_provider=_neutral_signals,
    )

    r = SmoresGeometry().wheel_radius_m

    state = _state(
        x=course.first_transition_x_m - 0.10,
        y=0.0,
        z=course.start_height_m + r,
    )

    result = evaluator(
        env_index=0,
        previous_state=state,
        next_state=state,
        actuation_result=None,
        physics_result=None,
        episode_step=7,
    )

    assert result.raw_reward.team.success == pytest.approx(0.0)
    assert result.terminated is False
    assert result.truncated is True


def test_signals_provider_is_propagated_into_stairs_reward():
    from drl.MARL.stairs_transition import (
        StairsTransitionEvaluator,
    )

    course = phase1_stairs_curriculum_course(
        12000,
        split="train",
    )

    expected = ActionOutcomeSignals(
        collision_owner_ids=frozenset(
            {"smores_01"}
        ),
        conflict_owner_ids=frozenset(
            {"smores_02"}
        ),
        invalid_owner_ids=frozenset(
            {"smores_03"}
        ),
        redundant_action_count_by_owner={
            "smores_04": 2,
        },
        control_effort_by_owner={
            "smores_05": 0.5,
        },
        stagnating=True,
    )

    def provider(
        env_index,
        actuation_result,
        previous_state,
        next_state,
    ):
        del (
            env_index,
            actuation_result,
            previous_state,
            next_state,
        )
        return expected

    evaluator = StairsTransitionEvaluator(
        courses=(course,),
        max_episode_steps=100,
        signals_provider=provider,
    )

    r = SmoresGeometry().wheel_radius_m

    state = _state(
        x=course.first_transition_x_m - 0.10,
        y=0.0,
        z=r,
    )

    result = evaluator(
        env_index=0,
        previous_state=state,
        next_state=state,
        actuation_result=None,
        physics_result=None,
        episode_step=1,
    )

    assert result.signals == expected

    assert (
        result
        .raw_reward
        .owners["smores_01"]
        .collision
        == 1.0
    )

    assert (
        result
        .raw_reward
        .owners["smores_02"]
        .conflict
        == 1.0
    )

    assert (
        result
        .raw_reward
        .owners["smores_03"]
        .invalid
        == 1.0
    )

    assert (
        result
        .raw_reward
        .owners["smores_04"]
        .redundant_action
        == 2.0
    )

    assert (
        result
        .raw_reward
        .owners["smores_05"]
        .control_effort
        == pytest.approx(0.5)
    )

    assert result.raw_reward.team.stagnation == 1.0


def test_raw_logger_writes_one_json_record_without_weights(
    tmp_path: Path,
):
    from drl.MARL.stairs_transition import (
        StairsRawTermJSONLLogger,
        StairsTransitionEvaluator,
    )

    course = phase1_stairs_curriculum_course(
        12653,
        split="train",
    )

    log_path = (
        tmp_path
        / "stairs_raw_terms.jsonl"
    )

    logger = StairsRawTermJSONLLogger(
        log_path
    )

    evaluator = StairsTransitionEvaluator(
        courses=(course,),
        max_episode_steps=100,
        signals_provider=_neutral_signals,
        raw_term_logger=logger,
    )

    r = SmoresGeometry().wheel_radius_m

    previous = _state(
        x=course.first_transition_x_m - 0.05,
        y=0.0,
        z=course.start_height_m + r,
    )

    next_state = _state(
        x=course.first_transition_x_m + 0.02,
        y=0.0,
        z=(
            course.step_top_heights_m[0]
            + r
        ),
    )

    evaluator(
        env_index=0,
        previous_state=previous,
        next_state=next_state,
        actuation_result=None,
        physics_result=None,
        episode_step=3,
    )

    lines = log_path.read_text().splitlines()

    assert len(lines) == 1

    payload = json.loads(
        lines[0]
    )

    assert payload["schema_version"] == (
        "mssr.stairs_raw_terms.v1"
    )

    assert payload["task"] == "stairs"
    assert payload["env_index"] == 0
    assert payload["episode_step"] == 3

    assert payload["course"]["seed"] == 12653
    assert payload["course"]["split"] == "train"
    assert payload["course"]["direction"] == "ascent"
    assert payload["course"]["profile"] == "nonuniform"

    assert "profile_progress" in payload["team"]
    assert "vertical_progress" in payload["team"]
    assert "level_progress" in payload["team"]

    assert sorted(
        payload["owners"]
    ) == list(OWNERS)

    serialized = json.dumps(
        payload
    ).lower()

    assert "weight" not in serialized


def test_logger_contains_normalized_runtime_signals(
    tmp_path: Path,
):
    from drl.MARL.stairs_transition import (
        StairsRawTermJSONLLogger,
        StairsTransitionEvaluator,
    )

    course = phase1_stairs_curriculum_course(
        12000,
        split="train",
    )

    logger = StairsRawTermJSONLLogger(
        tmp_path / "raw.jsonl"
    )

    signals = ActionOutcomeSignals(
        collision_owner_ids=frozenset(
            {"smores_08"}
        ),
        redundant_action_count_by_owner={
            "smores_02": 3,
        },
        control_effort_by_owner={
            "smores_04": 1.25,
        },
        stagnating=True,
    )

    evaluator = StairsTransitionEvaluator(
        courses=(course,),
        max_episode_steps=100,
        signals_provider=lambda *_: signals,
        raw_term_logger=logger,
    )

    r = SmoresGeometry().wheel_radius_m

    state = _state(
        x=0.0,
        y=0.0,
        z=r,
    )

    evaluator(
        env_index=0,
        previous_state=state,
        next_state=state,
        actuation_result=None,
        physics_result=None,
        episode_step=4,
    )

    payload = json.loads(
        (
            tmp_path
            / "raw.jsonl"
        )
        .read_text()
        .splitlines()[0]
    )

    assert payload["signals"]["collision_owner_ids"] == [
        "smores_08"
    ]

    assert payload["signals"]["redundant_action_count_by_owner"] == {
        "smores_02": 3
    }

    assert payload["signals"]["control_effort_by_owner"] == {
        "smores_04": 1.25
    }

    assert payload["signals"]["stagnating"] is True


def test_invalid_env_index_fails_closed():
    from drl.MARL.stairs_transition import (
        StairsTransitionEvaluator,
    )

    course = phase1_stairs_curriculum_course(
        12000,
        split="train",
    )

    evaluator = StairsTransitionEvaluator(
        courses=(course,),
        max_episode_steps=100,
        signals_provider=_neutral_signals,
    )

    r = SmoresGeometry().wheel_radius_m

    state = _state(
        x=0.0,
        y=0.0,
        z=r,
    )

    with pytest.raises(
        IndexError,
        match="env_index",
    ):
        evaluator(
            env_index=1,
            previous_state=state,
            next_state=state,
            actuation_result=None,
            physics_result=None,
            episode_step=1,
        )


def test_constructor_contract_is_fail_closed():
    from drl.MARL.stairs_transition import (
        StairsTransitionEvaluator,
    )

    course = phase1_stairs_curriculum_course(
        12000,
        split="train",
    )

    with pytest.raises(ValueError):
        StairsTransitionEvaluator(
            courses=(),
            max_episode_steps=100,
        )

    with pytest.raises(ValueError):
        StairsTransitionEvaluator(
            courses=(course,),
            max_episode_steps=0,
        )

    with pytest.raises(TypeError):
        StairsTransitionEvaluator(
            courses=(course,),
            max_episode_steps=100,
            signals_provider=123,
        )

    with pytest.raises(TypeError):
        StairsTransitionEvaluator(
            courses=(course,),
            max_episode_steps=100,
            raw_term_logger=123,
        )
