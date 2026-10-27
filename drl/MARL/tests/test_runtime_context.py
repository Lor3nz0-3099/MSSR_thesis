import torch

from drl.MARL.runtime_handoff import DirectMARLHandoff


class GraphSequence:
    def __init__(self, batches):
        self._batches = list(batches)
        self.calls = 0

    def __call__(self):
        if self.calls >= len(self._batches):
            raise AssertionError(
                "graph source called more often than expected"
            )

        batch = self._batches[self.calls]
        self.calls += 1
        return batch


def make_marl_handoff():
    state = DirectMARLHandoff()

    assert state.begin_expert("rc_car8")
    assert state.observe_expert_terminal(True)
    assert state.observe_topology("rc_car8")
    assert state.marl_active

    return state


def actor_visibility(graph):
    # Deliberately mutate the supplied graph. The provider must have
    # forked it from the critic snapshot first.
    graph.pop("critic_only", None)
    return graph


def routing_builder(actor_graph):
    assert "critic_only" not in actor_graph

    return {
        "state_id": actor_graph["state_id"],
        "route": "actor-visible",
    }


def hard_mask_builder(actor_graph, routing):
    assert "critic_only" not in actor_graph
    assert routing["state_id"] == actor_graph["state_id"]

    return torch.tensor(
        [False, True],
        dtype=torch.bool,
    )


def neutral_action(env_index, context):
    return {
        "module_0": torch.tensor(
            [0.0, 0.0],
        ),
        "module_1": torch.tensor(
            [0.0, 0.0],
        ),
    }


def test_provider_acquires_full_graph_once_and_forks_actor_critic_views():
    from drl.MARL.runtime_context import (
        DirectMARLRuntimeContextProvider,
    )

    source_graph = {
        "state_id": 1,
        "visible": 10,
        "critic_only": {
            "floor_height": 0.25,
        },
    }

    source = GraphSequence(
        [
            [source_graph],
        ]
    )

    provider = DirectMARLRuntimeContextProvider(
        graph_source=source,
        actor_visibility=actor_visibility,
        routing_builder=routing_builder,
        hard_mask_builder=hard_mask_builder,
        handoffs=[
            DirectMARLHandoff(),
        ],
        action_encoder=lambda *_: {
            "module_0": torch.ones(2),
            "module_1": torch.ones(2),
        },
        neutral_action=neutral_action,
    )

    first = provider.current_action_contexts()
    second = provider.current_action_contexts()

    assert source.calls == 1
    assert len(first) == 1
    assert len(second) == 1

    context = first[0]

    assert context.actor_graph == {
        "state_id": 1,
        "visible": 10,
    }

    assert context.critic_graph == {
        "state_id": 1,
        "visible": 10,
        "critic_only": {
            "floor_height": 0.25,
        },
    }

    assert context.actor_graph is not context.critic_graph

    # The visibility function was intentionally destructive.
    # It must not mutate either the critic snapshot or source object.
    assert "critic_only" in context.critic_graph
    assert "critic_only" in source_graph

    assert context.routing_snapshot == {
        "state_id": 1,
        "route": "actor-visible",
    }

    torch.testing.assert_close(
        context.hard_mask,
        torch.tensor(
            [False, True],
            dtype=torch.bool,
        ),
    )


def test_critic_only_perturbation_changes_critic_view_not_actor_view():
    from drl.MARL.runtime_context import (
        DirectMARLRuntimeContextProvider,
    )

    def build(secret):
        provider = DirectMARLRuntimeContextProvider(
            graph_source=GraphSequence(
                [[{
                    "state_id": 7,
                    "visible": 42,
                    "critic_only": {
                        "secret": secret,
                    },
                }]]
            ),
            actor_visibility=actor_visibility,
            routing_builder=routing_builder,
            hard_mask_builder=hard_mask_builder,
            handoffs=[
                DirectMARLHandoff(),
            ],
            action_encoder=lambda *_: {
                "module_0": torch.ones(2),
                "module_1": torch.ones(2),
            },
            neutral_action=neutral_action,
        )

        return provider.current_action_contexts()[0]

    left = build(1.0)
    right = build(999.0)

    assert left.actor_graph == right.actor_graph
    assert left.critic_graph != right.critic_graph

    assert (
        left.critic_graph[
            "critic_only"
        ]["secret"]
        == 1.0
    )

    assert (
        right.critic_graph[
            "critic_only"
        ]["secret"]
        == 999.0
    )


def test_routing_and_hard_mask_are_built_only_from_actor_visible_graph():
    from drl.MARL.runtime_context import (
        DirectMARLRuntimeContextProvider,
    )

    observed = {
        "routing": None,
        "mask": None,
    }

    def routing(graph):
        assert "critic_only" not in graph
        observed["routing"] = dict(graph)

        return {
            "candidate_ids": [
                "candidate-0",
            ],
        }

    def mask(graph, routing_snapshot):
        assert "critic_only" not in graph
        assert routing_snapshot == {
            "candidate_ids": [
                "candidate-0",
            ],
        }

        observed["mask"] = dict(graph)

        return torch.tensor(
            [False],
            dtype=torch.bool,
        )

    provider = DirectMARLRuntimeContextProvider(
        graph_source=GraphSequence(
            [[{
                "state_id": 3,
                "visible": "yes",
                "critic_only": "never-route-this",
            }]]
        ),
        actor_visibility=actor_visibility,
        routing_builder=routing,
        hard_mask_builder=mask,
        handoffs=[
            DirectMARLHandoff(),
        ],
        action_encoder=lambda *_: {
            "module_0": torch.ones(2),
        },
        neutral_action=lambda *_: {
            "module_0": torch.zeros(2),
        },
    )

    context = provider.current_action_contexts()[0]

    assert observed["routing"] == context.actor_graph
    assert observed["mask"] == context.actor_graph
    assert "critic_only" not in observed["routing"]
    assert "critic_only" not in observed["mask"]


def test_encode_env_actions_never_calls_marl_encoder_without_marl_authority():
    from drl.MARL.runtime_context import (
        DirectMARLRuntimeContextProvider,
    )

    handoff = DirectMARLHandoff()

    encoded = []

    def marl_encoder(
        decision,
        env_index,
        context,
    ):
        encoded.append(
            (
                decision,
                env_index,
                context,
            )
        )

        return {
            "module_0": torch.tensor(
                [5.0, 6.0],
            ),
            "module_1": torch.tensor(
                [7.0, 8.0],
            ),
        }

    provider = DirectMARLRuntimeContextProvider(
        graph_source=GraphSequence(
            [[{
                "state_id": 1,
                "visible": 1,
                "critic_only": 2,
            }]]
        ),
        actor_visibility=actor_visibility,
        routing_builder=routing_builder,
        hard_mask_builder=hard_mask_builder,
        handoffs=[
            handoff,
        ],
        action_encoder=marl_encoder,
        neutral_action=neutral_action,
    )

    provider.current_action_contexts()

    actions = provider.encode_env_actions(
        [
            object(),
        ]
    )

    assert encoded == []

    torch.testing.assert_close(
        actions["module_0"],
        torch.tensor(
            [[0.0, 0.0]]
        ),
    )

    torch.testing.assert_close(
        actions["module_1"],
        torch.tensor(
            [[0.0, 0.0]]
        ),
    )


def test_encode_env_actions_uses_marl_encoder_after_verified_handoff():
    from drl.MARL.runtime_context import (
        DirectMARLRuntimeContextProvider,
    )

    handoff = make_marl_handoff()

    decision = object()
    calls = []

    def marl_encoder(
        received_decision,
        env_index,
        context,
    ):
        calls.append(
            (
                received_decision,
                env_index,
                context.actor_graph["state_id"],
            )
        )

        return {
            "module_0": torch.tensor(
                [5.0, 6.0],
            ),
            "module_1": torch.tensor(
                [7.0, 8.0],
            ),
        }

    provider = DirectMARLRuntimeContextProvider(
        graph_source=GraphSequence(
            [[{
                "state_id": 11,
                "visible": 1,
                "critic_only": 2,
            }]]
        ),
        actor_visibility=actor_visibility,
        routing_builder=routing_builder,
        hard_mask_builder=hard_mask_builder,
        handoffs=[
            handoff,
        ],
        action_encoder=marl_encoder,
        neutral_action=neutral_action,
    )

    provider.current_action_contexts()

    actions = provider.encode_env_actions(
        [
            decision,
        ]
    )

    assert len(calls) == 1
    assert calls[0][0] is decision
    assert calls[0][1:] == (
        0,
        11,
    )

    torch.testing.assert_close(
        actions["module_0"],
        torch.tensor(
            [[5.0, 6.0]]
        ),
    )

    torch.testing.assert_close(
        actions["module_1"],
        torch.tensor(
            [[7.0, 8.0]]
        ),
    )


def test_post_step_critic_capture_is_reused_by_next_actor_context():
    from drl.MARL.runtime_context import (
        DirectMARLRuntimeContextProvider,
    )

    source = GraphSequence(
        [
            [{
                "state_id": 1,
                "visible": "before",
                "critic_only": "critic-before",
            }],
            [{
                "state_id": 2,
                "visible": "after",
                "critic_only": "critic-after",
            }],
        ]
    )

    provider = DirectMARLRuntimeContextProvider(
        graph_source=source,
        actor_visibility=actor_visibility,
        routing_builder=routing_builder,
        hard_mask_builder=hard_mask_builder,
        handoffs=[
            make_marl_handoff(),
        ],
        action_encoder=lambda *_: {
            "module_0": torch.tensor(
                [1.0, 2.0]
            ),
        },
        neutral_action=lambda *_: {
            "module_0": torch.zeros(2),
        },
    )

    current = provider.current_action_contexts()

    assert source.calls == 1
    assert current[0].actor_graph["state_id"] == 1

    provider.encode_env_actions(
        [
            object(),
        ]
    )

    next_critic = provider.current_critic_graphs()

    assert source.calls == 2
    assert next_critic[0]["state_id"] == 2
    assert (
        next_critic[0]["critic_only"]
        == "critic-after"
    )

    next_actor = provider.current_action_contexts()

    # The next actor input must be derived from exactly the same captured
    # post-step physical state. No third graph-source read.
    assert source.calls == 2
    assert next_actor[0].actor_graph["state_id"] == 2
    assert (
        next_actor[0].actor_graph["visible"]
        == "after"
    )
    assert "critic_only" not in next_actor[0].actor_graph


def test_repeated_current_critic_graphs_does_not_recapture_same_state():
    from drl.MARL.runtime_context import (
        DirectMARLRuntimeContextProvider,
    )

    source = GraphSequence(
        [
            [{
                "state_id": 1,
                "visible": 1,
                "critic_only": 1,
            }],
            [{
                "state_id": 2,
                "visible": 2,
                "critic_only": 2,
            }],
        ]
    )

    provider = DirectMARLRuntimeContextProvider(
        graph_source=source,
        actor_visibility=actor_visibility,
        routing_builder=routing_builder,
        hard_mask_builder=hard_mask_builder,
        handoffs=[
            make_marl_handoff(),
        ],
        action_encoder=lambda *_: {
            "module_0": torch.ones(2),
        },
        neutral_action=lambda *_: {
            "module_0": torch.zeros(2),
        },
    )

    provider.current_action_contexts()
    provider.encode_env_actions(
        [
            object(),
        ]
    )

    first = provider.current_critic_graphs()
    second = provider.current_critic_graphs()

    assert source.calls == 2
    assert first == second
    assert first[0]["state_id"] == 2


def test_reset_cache_forces_fresh_graph_capture():
    from drl.MARL.runtime_context import (
        DirectMARLRuntimeContextProvider,
    )

    source = GraphSequence(
        [
            [{
                "state_id": 1,
                "visible": 1,
                "critic_only": 1,
            }],
            [{
                "state_id": 100,
                "visible": 100,
                "critic_only": 100,
            }],
        ]
    )

    provider = DirectMARLRuntimeContextProvider(
        graph_source=source,
        actor_visibility=actor_visibility,
        routing_builder=routing_builder,
        hard_mask_builder=hard_mask_builder,
        handoffs=[
            DirectMARLHandoff(),
        ],
        action_encoder=lambda *_: {
            "module_0": torch.ones(2),
        },
        neutral_action=lambda *_: {
            "module_0": torch.zeros(2),
        },
    )

    assert (
        provider.current_action_contexts()[
            0
        ].actor_graph["state_id"]
        == 1
    )

    assert source.calls == 1

    provider.reset()

    assert (
        provider.current_action_contexts()[
            0
        ].actor_graph["state_id"]
        == 100
    )

    assert source.calls == 2
