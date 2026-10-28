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
from smores_ep.isaac.phase1_gap_curriculum import (
    phase1_gap_curriculum_course,
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


def test_gap_evaluator_matches_direct_marl_callable_contract():
    from drl.MARL.gap_transition import (
        GapTransitionEvaluator,
    )

    course = phase1_gap_curriculum_course(
        18527,
        split="challenge",
    )

    evaluator = GapTransitionEvaluator(
        courses=(course,),
        max_episode_steps=100,
        signals_provider=_neutral_signals,
    )

    r = SmoresGeometry().wheel_radius_m

    previous = _state(
        x=course.near_edge_x_m - 0.10,
        y=0.0,
        z=r,
    )

    next_state = _state(
        x=course.near_edge_x_m - 0.05,
        y=0.0,
        z=r,
    )

    result = evaluator(
        env_index=0,
        previous_state=previous,
        next_state=next_state,
        actuation_result=object(),
        physics_result=object(),
        episode_step=1,
    )

    assert (
        result.raw_reward.team.approach_progress
        == pytest.approx(0.05)
    )

    assert (
        result.signals
        == ActionOutcomeSignals()
    )

    assert result.terminated is False
    assert result.truncated is False


def test_gap_success_terminates_and_wins_over_time_limit():
    from drl.MARL.gap_transition import (
        GapTransitionEvaluator,
    )

    course = phase1_gap_curriculum_course(
        18527,
        split="challenge",
    )

    evaluator = GapTransitionEvaluator(
        courses=(course,),
        max_episode_steps=5,
        signals_provider=_neutral_signals,
    )

    r = SmoresGeometry().wheel_radius_m

    previous = _state(
        x=course.far_edge_x_m - 0.02,
        y=course.landing_center_y_m,
        z=r,
    )

    next_state = _state(
        x=course.far_edge_x_m + 0.20,
        y=course.landing_center_y_m,
        z=r,
    )

    result = evaluator(
        env_index=0,
        previous_state=previous,
        next_state=next_state,
        actuation_result=None,
        physics_result=None,
        episode_step=5,
    )

    assert (
        result.raw_reward.team.success
        == pytest.approx(1.0)
    )

    assert result.terminated is True
    assert result.truncated is False


def test_gap_time_limit_truncates_only_without_success():
    from drl.MARL.gap_transition import (
        GapTransitionEvaluator,
    )

    course = phase1_gap_curriculum_course(
        18527,
        split="challenge",
    )

    evaluator = GapTransitionEvaluator(
        courses=(course,),
        max_episode_steps=7,
        signals_provider=_neutral_signals,
    )

    r = SmoresGeometry().wheel_radius_m

    state = _state(
        x=course.near_edge_x_m - 0.10,
        y=0.0,
        z=r,
    )

    result = evaluator(
        env_index=0,
        previous_state=state,
        next_state=state,
        actuation_result=None,
        physics_result=None,
        episode_step=7,
    )

    assert (
        result.raw_reward.team.success
        == pytest.approx(0.0)
    )

    assert result.terminated is False
    assert result.truncated is True


def test_gap_signals_are_propagated_to_raw_reward():
    from drl.MARL.gap_transition import (
        GapTransitionEvaluator,
    )

    course = phase1_gap_curriculum_course(
        18527,
        split="challenge",
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

    evaluator = GapTransitionEvaluator(
        courses=(course,),
        max_episode_steps=100,
        signals_provider=provider,
    )

    r = SmoresGeometry().wheel_radius_m

    state = _state(
        x=course.near_edge_x_m - 0.10,
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
        == pytest.approx(1.0)
    )

    assert (
        result
        .raw_reward
        .owners["smores_02"]
        .conflict
        == pytest.approx(1.0)
    )

    assert (
        result
        .raw_reward
        .owners["smores_03"]
        .invalid
        == pytest.approx(1.0)
    )

    assert (
        result
        .raw_reward
        .owners["smores_04"]
        .redundant_action
        == pytest.approx(2.0)
    )

    assert (
        result
        .raw_reward
        .owners["smores_05"]
        .control_effort
        == pytest.approx(0.5)
    )

    assert (
        result.raw_reward.team.stagnation
        == pytest.approx(1.0)
    )


def test_gap_raw_logger_writes_geometry_and_raw_terms(
    tmp_path: Path,
):
    from drl.MARL.gap_transition import (
        GapRawTermJSONLLogger,
        GapTransitionEvaluator,
    )

    course = phase1_gap_curriculum_course(
        18527,
        split="challenge",
    )

    log_path = (
        tmp_path
        / "gap_raw_terms.jsonl"
    )

    evaluator = GapTransitionEvaluator(
        courses=(course,),
        max_episode_steps=100,
        signals_provider=_neutral_signals,
        raw_term_logger=(
            GapRawTermJSONLLogger(
                log_path
            )
        ),
    )

    r = SmoresGeometry().wheel_radius_m

    previous = _state(
        x=(
            course.near_edge_x_m
            + 0.25
            * course.gap_width_m
        ),
        y=0.0,
        z=r,
    )

    next_state = _state(
        x=(
            course.near_edge_x_m
            + 0.75
            * course.gap_width_m
        ),
        y=0.0,
        z=r,
    )

    evaluator(
        env_index=0,
        previous_state=previous,
        next_state=next_state,
        actuation_result=None,
        physics_result=None,
        episode_step=3,
    )

    lines = (
        log_path
        .read_text()
        .splitlines()
    )

    assert len(lines) == 1

    payload = json.loads(
        lines[0]
    )

    assert payload["schema_version"] == (
        "mssr.gap_raw_terms.v1"
    )

    assert payload["task"] == "gap"
    assert payload["env_index"] == 0
    assert payload["episode_step"] == 3

    assert (
        payload["course"]["seed"]
        == 18527
    )

    assert (
        payload["course"]["split"]
        == "challenge"
    )

    assert (
        payload["course"]["gap_width_m"]
        == pytest.approx(
            course.gap_width_m
        )
    )

    assert (
        payload["course"]["landing_center_y_m"]
        == pytest.approx(
            course.landing_center_y_m
        )
    )

    assert (
        payload["team"]["crossing_progress"]
        == pytest.approx(0.5)
    )

    assert "fall" in payload["team"]

    assert sorted(
        payload["owners"]
    ) == list(OWNERS)

    serialized = json.dumps(
        payload
    ).lower()

    assert "weight" not in serialized
    assert "lateral_alignment" not in serialized


def test_gap_logger_normalizes_runtime_signals(
    tmp_path: Path,
):
    from drl.MARL.gap_transition import (
        GapRawTermJSONLLogger,
        GapTransitionEvaluator,
    )

    course = phase1_gap_curriculum_course(
        16000,
        split="train",
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

    log_path = (
        tmp_path
        / "raw.jsonl"
    )

    evaluator = GapTransitionEvaluator(
        courses=(course,),
        max_episode_steps=100,
        signals_provider=lambda *_: signals,
        raw_term_logger=(
            GapRawTermJSONLLogger(
                log_path
            )
        ),
    )

    r = SmoresGeometry().wheel_radius_m

    state = _state(
        x=course.near_edge_x_m - 0.10,
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
        log_path
        .read_text()
        .splitlines()[0]
    )

    assert (
        payload["signals"]["collision_owner_ids"]
        == ["smores_08"]
    )

    assert (
        payload["signals"]
        ["redundant_action_count_by_owner"]
        == {"smores_02": 3}
    )

    assert (
        payload["signals"]
        ["control_effort_by_owner"]
        == {"smores_04": 1.25}
    )

    assert (
        payload["signals"]["stagnating"]
        is True
    )


def test_gap_invalid_env_index_fails_closed():
    from drl.MARL.gap_transition import (
        GapTransitionEvaluator,
    )

    course = phase1_gap_curriculum_course(
        16000,
        split="train",
    )

    evaluator = GapTransitionEvaluator(
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


def test_gap_constructor_contract_is_fail_closed():
    from drl.MARL.gap_transition import (
        GapTransitionEvaluator,
    )

    course = phase1_gap_curriculum_course(
        16000,
        split="train",
    )

    with pytest.raises(ValueError):
        GapTransitionEvaluator(
            courses=(),
            max_episode_steps=100,
        )

    with pytest.raises(ValueError):
        GapTransitionEvaluator(
            courses=(course,),
            max_episode_steps=0,
        )

    with pytest.raises(TypeError):
        GapTransitionEvaluator(
            courses=(course,),
            max_episode_steps=100,
            signals_provider=123,
        )

    with pytest.raises(TypeError):
        GapTransitionEvaluator(
            courses=(course,),
            max_episode_steps=100,
            raw_term_logger=123,
        )
