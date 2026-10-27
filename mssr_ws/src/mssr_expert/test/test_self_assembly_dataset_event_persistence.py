from types import SimpleNamespace

from mssr_expert.nodes import smores_parallel_self_assembly_node as assembly_node


def _decision(goal):
    return SimpleNamespace(
        done=False,
        success=False,
        message="",
        expert_output=SimpleNamespace(
            primitive_goal=goal,
        ),
    )


def test_same_graph_stamp_preserves_each_new_primitive_goal():
    """A new structural dispatch must not be lost just because G_t is unchanged."""

    graph = SimpleNamespace(stamp=123)
    decisions = iter(
        (
            _decision("goal-a"),
            _decision("goal-b"),
            _decision(None),
        )
    )

    node = SimpleNamespace(
        _latest_observation={"available": True},
        _latest_graph_payload={},
        _graph_builder=SimpleNamespace(
            build=lambda **kwargs: graph,
        ),
        _pending_transition=None,
        _terminal_reached=False,
        _planning_result=object(),
        _executor=SimpleNamespace(
            step=lambda status: next(decisions),
        ),
        _latest_status_payload={},
        _timestep=0,
    )

    node._create_plan = lambda current_graph: True
    node._current_task_graph = lambda current_graph, decision: current_graph
    node._expert_output = lambda decision: decision.expert_output
    node._publish_decision = lambda decision, task_graph: None

    flushed = []

    def flush_pending_transition(*, next_graph):
        pending = node._pending_transition
        assert pending is not None
        flushed.append(pending.expert_output.primitive_goal)
        node._pending_transition = None

    node._flush_pending_transition = flush_pending_transition

    step = assembly_node.SmoresParallelSelfAssemblyNode._step

    # First dispatch on graph stamp 123.
    step(node)
    assert flushed == []
    assert node._pending_transition.expert_output.primitive_goal == "goal-a"

    # A second dispatch occurs while the physical graph stamp is still 123.
    # The first event must be finalized and the new event must become pending.
    step(node)
    assert flushed == ["goal-a"]
    assert node._pending_transition.expert_output.primitive_goal == "goal-b"

    # Merely polling the executor again with no new primitive must NOT create
    # another transition or duplicate the pending structural event.
    step(node)
    assert flushed == ["goal-a"]
    assert node._pending_transition.expert_output.primitive_goal == "goal-b"
