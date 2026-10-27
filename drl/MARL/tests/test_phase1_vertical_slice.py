from __future__ import annotations

import importlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def _runner_api():
    spec = importlib.util.find_spec(
        "drl.MARL.phase1_vertical_slice"
    )

    assert (
        spec is not None
    ), "TASK4B_RED_VERTICAL_SLICE_MISSING"

    module = importlib.import_module(
        "drl.MARL.phase1_vertical_slice"
    )

    return (
        module.Phase1VerticalSliceRunner,
        module.Phase1VerticalSliceStep,
    )


def _write_state_graph(
    runtime_dir: Path,
    *,
    stamp: float,
    x_m: float,
) -> None:
    runtime_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    modules = [
        {
            "module_id": f"smores_{index:02d}",
            "position": [
                float(x_m),
                float(index) * 0.01,
                0.04,
            ],
        }
        for index in range(1, 9)
    ]

    state = {
        "schema_version": "mssr.module_states.v2",
        "stamp": float(stamp),
        "timestamp": float(stamp),
        "modules": modules,
    }

    graph = {
        "schema_version": "mssr.robot_graph.v2",
        "stamp": float(stamp),
        "nodes": [],
        "edges": [],
        "global_attributes": {
            "module_count": 8,
        },
    }

    payload = {
        "schema_version": "mssr.state_graph.v2",
        "stamp": float(stamp),
        "state": state,
        "graph": graph,
    }

    path = runtime_dir / "state_graph.json"
    temporary = path.with_suffix(".json.tmp")

    temporary.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    temporary.replace(path)


def test_task4b_control_step_keeps_exact_raw_frame_and_chains_post_to_next_pre(
    tmp_path,
):
    from drl.MARL.isaac_control_step import (
        FileBackedIsaacControlStep,
    )

    runtime = tmp_path / "env0"

    _write_state_graph(
        runtime,
        stamp=1.0,
        x_m=0.0,
    )

    clock = [0.0]
    sleeps = [0]

    def monotonic():
        return clock[0]

    def sleep(duration):
        sleeps[0] += 1
        clock[0] += duration

        if sleeps[0] == 1:
            _write_state_graph(
                runtime,
                stamp=1.1,
                x_m=1.0,
            )

    control = FileBackedIsaacControlStep(
        runtime_dirs=(runtime,),
        expected_module_count=8,
        timeout_s=1.0,
        poll_interval_s=0.01,
        monotonic=monotonic,
        sleep=sleep,
    )

    pre = control.state_reader(0)

    assert pre["smores_01"][0] == pytest.approx(0.0)

    raw_pre = control.cached_raw_snapshot(0)

    assert raw_pre["stamp"] == pytest.approx(1.0)
    assert raw_pre["state"]["stamp"] == pytest.approx(1.0)

    result = control.physics_step()

    assert result.previous_stamp_by_env[0] == pytest.approx(1.0)
    assert result.next_stamp_by_env[0] == pytest.approx(1.1)

    post = control.state_reader(0)

    assert post["smores_01"][0] == pytest.approx(1.0)

    # Even after the DirectMARLEnv post-state read, the exact accepted
    # physical frame must remain available to the post-step critic.
    raw_post = control.cached_raw_snapshot(0)

    assert raw_post["stamp"] == pytest.approx(1.1)
    assert raw_post["state"]["modules"][0]["position"][0] == pytest.approx(
        1.0
    )

    # Isaac may already publish another frame before the next policy call.
    _write_state_graph(
        runtime,
        stamp=1.2,
        x_m=2.0,
    )

    # Markov chaining:
    # previous s_(t+1) must become EXACTLY the next s_t.
    next_pre = control.state_reader(0)

    assert next_pre["smores_01"][0] == pytest.approx(1.0)
    assert control.baseline_stamp(0) == pytest.approx(1.1)

    raw_next_pre = control.cached_raw_snapshot(0)

    assert raw_next_pre["stamp"] == pytest.approx(1.1)

    control.reset()

    # Only an explicit reset may discard the carry frame and start from
    # the newest physical publication.
    fresh = control.state_reader(0)

    assert fresh["smores_01"][0] == pytest.approx(2.0)
    assert control.baseline_stamp(0) == pytest.approx(1.2)


class _Handoff:
    def __init__(self, active: bool):
        self.marl_active = active


class _Sync:
    def __init__(self, events, num_envs):
        self.events = events
        self.num_envs = num_envs

    def state_reader(self, env_index):
        self.events.append(
            f"arm:{env_index}"
        )
        return {
            f"module_{env_index}": (
                float(env_index),
                0.0,
            )
        }


class _ContextProvider:
    def __init__(
        self,
        events,
        contexts,
    ):
        self.events = events
        self._contexts = tuple(contexts)
        self.num_envs = len(self._contexts)
        self.received_decisions = None

    def current_action_contexts(self):
        self.events.append("contexts")
        return list(self._contexts)

    def encode_env_actions(self, decisions):
        self.events.append("encode")
        self.received_decisions = tuple(
            decisions
        )

        return {
            "module_0": "placeholder"
        }

    def current_critic_graphs(self):
        self.events.append("next_critic")

        return [
            f"next-critic-{index}"
            for index in range(
                self.num_envs
            )
        ]


class _DecisionBuilder:
    def __init__(self, events):
        self.events = events
        self.calls = []

    def build_online_rollout_decision(
        self,
        *,
        actor_graph,
        critic_graph,
        routing_snapshot,
        hard_mask,
    ):
        self.events.append("decision")

        self.calls.append(
            (
                actor_graph,
                critic_graph,
                routing_snapshot,
                hard_mask,
            )
        )

        return (
            "decision:"
            + str(actor_graph)
        )


class _Env:
    def __init__(
        self,
        events,
        active,
        transitions,
    ):
        self.events = events
        self.handoffs = tuple(
            _Handoff(value)
            for value in active
        )
        self.num_envs = len(
            self.handoffs
        )
        self._transitions = tuple(
            transitions
        )
        self.calls = 0

    def step_marl_transitions(self):
        self.events.append("transition")
        self.calls += 1
        return self._transitions


def _context(index):
    return SimpleNamespace(
        actor_graph=f"actor-{index}",
        critic_graph=f"critic-{index}",
        routing_snapshot=f"routing-{index}",
        hard_mask=f"mask-{index}",
    )


def test_task4b_runner_does_nothing_before_marl_authority():
    (
        Phase1VerticalSliceRunner,
        _,
    ) = _runner_api()

    events = []

    env = _Env(
        events,
        active=(False,),
        transitions=(None,),
    )

    sync = _Sync(
        events,
        num_envs=1,
    )

    contexts = _ContextProvider(
        events,
        [_context(0)],
    )

    builder = _DecisionBuilder(
        events
    )

    runner = Phase1VerticalSliceRunner(
        env=env,
        runtime_context=contexts,
        decision_builder=builder,
        frame_synchronizer=sync,
    )

    result = runner.step()

    assert result is None
    assert events == []
    assert builder.calls == []
    assert env.calls == 0


def test_task4b_runner_orders_frame_policy_action_transition_next_critic():
    (
        Phase1VerticalSliceRunner,
        Phase1VerticalSliceStep,
    ) = _runner_api()

    events = []

    env = _Env(
        events,
        active=(True,),
        transitions=("transition-0",),
    )

    sync = _Sync(
        events,
        num_envs=1,
    )

    contexts = _ContextProvider(
        events,
        [_context(0)],
    )

    builder = _DecisionBuilder(
        events
    )

    runner = Phase1VerticalSliceRunner(
        env=env,
        runtime_context=contexts,
        decision_builder=builder,
        frame_synchronizer=sync,
    )

    result = runner.step()

    assert isinstance(
        result,
        Phase1VerticalSliceStep,
    )

    assert events == [
        "arm:0",
        "contexts",
        "decision",
        "encode",
        "transition",
        "next_critic",
    ]

    assert result.active_env_indices == (0,)
    assert result.decisions == (
        "decision:actor-0",
    )

    assert result.transitions == (
        "transition-0",
    )

    assert result.next_critic_graphs == (
        "next-critic-0",
    )

    assert contexts.received_decisions == (
        "decision:actor-0",
    )


def test_task4b_runner_never_builds_policy_decision_for_inactive_vector_env():
    (
        Phase1VerticalSliceRunner,
        _,
    ) = _runner_api()

    events = []

    env = _Env(
        events,
        active=(
            True,
            False,
        ),
        transitions=(
            "transition-0",
            None,
        ),
    )

    sync = _Sync(
        events,
        num_envs=2,
    )

    contexts = _ContextProvider(
        events,
        [
            _context(0),
            _context(1),
        ],
    )

    builder = _DecisionBuilder(
        events
    )

    runner = Phase1VerticalSliceRunner(
        env=env,
        runtime_context=contexts,
        decision_builder=builder,
        frame_synchronizer=sync,
    )

    result = runner.step()

    assert result is not None

    assert result.active_env_indices == (
        0,
    )

    assert builder.calls == [
        (
            "actor-0",
            "critic-0",
            "routing-0",
            "mask-0",
        )
    ]

    assert contexts.received_decisions == (
        "decision:actor-0",
        None,
    )

    assert result.transitions == (
        "transition-0",
        None,
    )

    # Only the environment that actually owns MARL authority is armed.
    assert events.count("arm:0") == 1
    assert "arm:1" not in events
