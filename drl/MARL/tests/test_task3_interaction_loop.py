from __future__ import annotations

import importlib.util
from types import SimpleNamespace

import pytest
import torch


def _activate_marl(handoff) -> None:
    assert handoff.begin_expert(
        "rc_car8"
    )

    assert handoff.observe_expert_terminal(
        success=True
    )

    assert handoff.observe_topology(
        "rc_car8"
    )

    assert handoff.marl_active is True


def _callbacks():
    return {
        "expert_launcher": (
            lambda env_index, target:
            True
        ),
        "expert_status_reader": (
            lambda env_index:
            None
        ),
        "topology_observer": (
            lambda env_index:
            None
        ),
    }


class _NoopBridge:
    def step(
        self,
        env_index,
    ):
        return {
            "env_index": env_index,
        }


def test_task3_transition_is_hard_gated_by_marl_authority():
    from drl.MARL.direct_marl_env import (
        DirectMARLEnv,
    )
    from drl.MARL.runtime_handoff import (
        DirectMARLHandoff,
    )

    probe = DirectMARLEnv(
        handoffs=(
            DirectMARLHandoff(),
        ),
        actuation_bridge=_NoopBridge(),
        target_morphology="rc_car8",
        **_callbacks(),
    )

    assert hasattr(
        probe,
        "step_marl_transitions",
    ), "TASK3_RED_DIRECT_TRANSITION_MISSING"

    events = []

    class Bridge:
        def step(
            self,
            env_index,
        ):
            events.append(
                (
                    "actuate",
                    env_index,
                )
            )

            return {
                "env_index": env_index,
            }

    env = DirectMARLEnv(
        handoffs=(
            DirectMARLHandoff(),
        ),
        actuation_bridge=Bridge(),
        target_morphology="rc_car8",
        state_reader=(
            lambda env_index:
            events.append(
                (
                    "read",
                    env_index,
                )
            )
            or {
                "value": 0,
            }
        ),
        physics_step=(
            lambda:
            events.append(
                (
                    "physics",
                )
            )
        ),
        transition_evaluator=(
            lambda **kwargs:
            events.append(
                (
                    "evaluate",
                    kwargs[
                        "env_index"
                    ],
                )
            )
        ),
        **_callbacks(),
    )

    # Nessuna autorità MARL:
    # nessuna lettura fisica, actuation, physics o reward.
    assert (
        env.step_marl_transitions()
        == (
            None,
        )
    )

    assert events == []


def test_task3_vector_transition_orders_state_action_physics_next_state_once():
    from drl.MARL.direct_marl_env import (
        DirectMARLEnv,
    )
    from drl.MARL.runtime_handoff import (
        DirectMARLHandoff,
    )

    assert hasattr(
        DirectMARLEnv,
        "step_marl_transitions",
    ), "TASK3_RED_DIRECT_TRANSITION_MISSING"

    handoffs = (
        DirectMARLHandoff(),
        DirectMARLHandoff(),
    )

    for handoff in handoffs:
        _activate_marl(
            handoff
        )

    physical = {
        0: {
            "x": 0,
        },
        1: {
            "x": 10,
        },
    }

    events = []

    def state_reader(
        env_index,
    ):
        events.append(
            (
                "read",
                env_index,
                physical[
                    env_index
                ][
                    "x"
                ],
            )
        )

        # Restituisce volutamente lo STESSO oggetto mutabile:
        # DirectMARLEnv deve congelare s_t prima della fisica.
        return physical[
            env_index
        ]

    class Bridge:
        def step(
            self,
            env_index,
        ):
            events.append(
                (
                    "actuate",
                    env_index,
                )
            )

            return {
                "env_index":
                    env_index,
            }

    def physics_step():
        events.append(
            (
                "physics",
            )
        )

        physical[
            0
        ][
            "x"
        ] += 1

        physical[
            1
        ][
            "x"
        ] += 1

        return "one-physics-tick"

    def evaluator(
        *,
        env_index,
        previous_state,
        next_state,
        actuation_result,
        physics_result,
        episode_step,
    ):
        events.append(
            (
                "evaluate",
                env_index,
                previous_state[
                    "x"
                ],
                next_state[
                    "x"
                ],
                episode_step,
            )
        )

        return {
            "env_index":
                env_index,
            "previous":
                previous_state[
                    "x"
                ],
            "next":
                next_state[
                    "x"
                ],
            "physics_result":
                physics_result,
            "actuation_env":
                actuation_result[
                    "env_index"
                ],
        }

    env = DirectMARLEnv(
        handoffs=handoffs,
        actuation_bridge=Bridge(),
        target_morphology="rc_car8",
        state_reader=state_reader,
        physics_step=physics_step,
        transition_evaluator=evaluator,
        **_callbacks(),
    )

    transitions = (
        env.step_marl_transitions()
    )

    assert events == [
        (
            "read",
            0,
            0,
        ),
        (
            "read",
            1,
            10,
        ),
        (
            "actuate",
            0,
        ),
        (
            "actuate",
            1,
        ),
        (
            "physics",
        ),
        (
            "read",
            0,
            1,
        ),
        (
            "read",
            1,
            11,
        ),
        (
            "evaluate",
            0,
            0,
            1,
            1,
        ),
        (
            "evaluate",
            1,
            10,
            11,
            1,
        ),
    ]

    first = transitions[
        0
    ]
    second = transitions[
        1
    ]

    assert first is not None
    assert second is not None

    # s_t deve essere rimasto realmente pre-fisica.
    assert first.previous_state == {
        "x": 0,
    }

    assert first.next_state == {
        "x": 1,
    }

    assert second.previous_state == {
        "x": 10,
    }

    assert second.next_state == {
        "x": 11,
    }

    assert (
        first.physics_result
        == "one-physics-tick"
    )

    assert (
        second.physics_result
        == "one-physics-tick"
    )

    assert env.episode_steps == (
        1,
        1,
    )


def test_task3_phase1_evaluator_uses_real_pre_post_reward_and_done_semantics():
    spec = importlib.util.find_spec(
        "drl.MARL.phase1_transition"
    )

    assert (
        spec is not None
    ), "TASK3_RED_PHASE1_MODULE_MISSING"

    from drl.MARL.phase1_transition import (
        Phase1TransitionEvaluator,
        phase1_signals_from_native_step,
    )

    class TargetPlatform:
        def contains_xy(
            self,
            point,
        ):
            return (
                float(
                    point[
                        0
                    ]
                )
                >= 2.0
            )

    class Course:
        centerline_xy_m = (
            (
                0.0,
                0.0,
            ),
            (
                2.0,
                0.0,
            ),
        )

        road_width_m = 1.0

        target_platform = (
            TargetPlatform()
        )

        def success(
            self,
            module_xy_m,
        ):
            return (
                len(
                    module_xy_m
                )
                == 8
                and all(
                    self.target_platform
                    .contains_xy(
                        point
                    )
                    for point
                    in module_xy_m.values()
                )
            )

    previous = {
        f"module_{index}": (
            0.0,
            0.0,
        )
        for index
        in range(
            8
        )
    }

    success_state = {
        f"module_{index}": (
            2.0,
            0.0,
        )
        for index
        in range(
            8
        )
    }

    busy_step = SimpleNamespace(
        proposal_outcome=(
            SimpleNamespace(
                accepted=False,
                code="RESOURCE_BUSY",
                owner_ids=(
                    "module_0",
                ),
            )
        )
    )

    evaluator = (
        Phase1TransitionEvaluator(
            courses=(
                Course(),
            ),
            max_episode_steps=2,
        )
    )

    evaluation = evaluator(
        env_index=0,
        previous_state=previous,
        next_state=success_state,
        actuation_result=busy_step,
        physics_result=None,
        episode_step=2,
    )

    assert (
        evaluation
        .signals
        .conflict_owner_ids
        == frozenset(
            {
                "module_0",
            }
        )
    )

    assert (
        evaluation
        .raw_reward
        .owners[
            "module_0"
        ]
        .conflict
        == 1.0
    )

    assert (
        evaluation
        .raw_reward
        .team
        .success
        == 1.0
    )

    # Se successo e horizon coincidono, è terminale vero,
    # non time-limit truncation.
    assert (
        evaluation.terminated
        is True
    )

    assert (
        evaluation.truncated
        is False
    )

    unfinished = {
        f"module_{index}": (
            1.0,
            0.0,
        )
        for index
        in range(
            8
        )
    }

    horizon = evaluator(
        env_index=0,
        previous_state=previous,
        next_state=unfinished,
        actuation_result=(
            SimpleNamespace(
                proposal_outcome=None
            )
        ),
        physics_result=None,
        episode_step=2,
    )

    assert (
        horizon.terminated
        is False
    )

    assert (
        horizon.truncated
        is True
    )

    invalid = (
        phase1_signals_from_native_step(
            SimpleNamespace(
                proposal_outcome=(
                    SimpleNamespace(
                        accepted=False,
                        code="INVALID_GOAL",
                        owner_ids=(
                            "module_1",
                        ),
                    )
                )
            )
        )
    )

    assert (
        invalid.invalid_owner_ids
        == frozenset(
            {
                "module_1",
            }
        )
    )

    assert (
        invalid.conflict_owner_ids
        == frozenset()
    )


def test_task3_expected_resource_busy_rejection_becomes_rollout_outcome():
    from drl.MARL.smores_runtime_binding import (
        SMORESMARLDecisionBridge,
        SMORESNativeDispatcher,
    )

    class FakeGoal:
        def __init__(
            self,
            payload,
        ):
            self.goal_id = (
                payload[
                    "goal_id"
                ]
            )

            self.primitive = (
                payload[
                    "primitive"
                ]
            )

            self.module_ids = tuple(
                payload[
                    "module_ids"
                ]
            )

            self.parameters = dict(
                payload[
                    "parameters"
                ]
            )

            self.timeout_s = float(
                payload[
                    "timeout_s"
                ]
            )

    class BusyExecutor:
        def submit(
            self,
            goal,
            now_s,
        ):
            return SimpleNamespace(
                state="rejected",
                code="RESOURCE_BUSY",
                message="busy",
            )

        def cancel(
            self,
            goal_id,
            now_s,
        ):
            raise AssertionError(
                "No accepted goal should need rollback"
            )

    executor = BusyExecutor()

    dispatcher = (
        SMORESNativeDispatcher(
            executor_provider=(
                lambda env_index:
                executor
            ),
            now_provider=(
                lambda env_index:
                1.0
            ),
            geometry=(
                SimpleNamespace(
                    wheel_radius_m=0.04,
                    track_width_m=0.16,
                )
            ),
            command_factory=(
                lambda **kwargs:
                SimpleNamespace(
                    **kwargs
                )
            ),
            pan_velocity_mode=(
                "pan_velocity"
            ),
            primitive_goal_factory=(
                lambda payload:
                FakeGoal(
                    payload
                )
            ),
            accepted_state=(
                "accepted"
            ),
        )
    )

    decoded = (
        SimpleNamespace(
            command={
                "command":
                    "dock",
                "module_ids": [
                    "module_0",
                    "module_1",
                ],
                "parameters": {
                    "face_a":
                        "TOP",
                    "face_b":
                        "BOTTOM",
                },
            }
        ),
    )

    class Binding:
        def decode_marl_action(
            self,
            **kwargs,
        ):
            return decoded

    placeholder_calls = []

    bridge = (
        SMORESMARLDecisionBridge(
            binding=Binding(),
            normalizers=object(),
            dispatcher=dispatcher,
            placeholder_action=(
                lambda env_index, context:
                placeholder_calls.append(
                    env_index
                )
                or {
                    "module_0":
                        torch.zeros(
                            1
                        ),
                }
            ),
            decoder_batch_builder=(
                lambda env_index, context:
                {
                    "inputs": {},
                    "routing": {},
                }
            ),
        )
    )

    decision = SimpleNamespace(
        action=object(),
        action_set_prediction_template={},
    )

    context = SimpleNamespace(
        actor_graph={},
        critic_graph={},
        routing_snapshot={},
        hard_mask=None,
    )

    # Nel codice pre-Task3:
    # RESOURCE_BUSY risale come RuntimeError.
    #
    # Dopo Task3 deve diventare un outcome RL,
    # senza applicare fisicamente la proposta respinta.
    result = bridge(
        decision,
        0,
        context,
    )

    assert placeholder_calls == [
        0,
    ]

    torch.testing.assert_close(
        result[
            "module_0"
        ],
        torch.zeros(
            1
        ),
    )

    assert (
        dispatcher.continuous_commands(
            0
        )
        == {}
    )

    outcome = (
        dispatcher.proposal_outcome(
            0
        )
    )

    assert outcome is not None

    assert (
        outcome.accepted
        is False
    )

    assert (
        outcome.code
        == "RESOURCE_BUSY"
    )

    assert (
        outcome.owner_ids
        == (
            "module_0",
            "module_1",
        )
    )
