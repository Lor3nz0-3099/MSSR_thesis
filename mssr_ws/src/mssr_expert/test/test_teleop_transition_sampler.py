from mssr_expert.graph.attributed_robot_graph import (
    AttributedRobotGraph,
    GraphNode,
)
from mssr_expert.teleop.recording import TransitionSampler


def graph(stamp, x):
    return AttributedRobotGraph(
        stamp=stamp,
        nodes=(
            GraphNode(
                "smores_01",
                {
                    "position": [x, 0.0, 0.05],
                    "orientation": [0.0, 0.0, 0.0, 1.0],
                    "linear_velocity": [0.0, 0.0, 0.0],
                    "angular_velocity": [0.0, 0.0, 0.0],
                },
            ),
        ),
    )


def test_transition_waits_for_a_newer_graph_and_uses_effective_action():
    sampler = TransitionSampler(dataset_rate_hz=10.0)

    assert sampler.step(
        now=10.00,
        graph=graph(1.0, 0.0),
        controller_input={"r2": 1.0},
        intent={"forward": 1.0},
        effective_actions={
            "smores_01": {"pan_rate_rad_s": 0.25},
        },
        morphology="rc_car8",
    ) is None

    # Stesso snapshot: non può inventare s_(t+1).
    assert sampler.step(
        now=10.02,
        graph=graph(1.0, 0.0),
        controller_input={"r2": 1.0},
        intent={"forward": 1.0},
        effective_actions={
            "smores_01": {"pan_rate_rad_s": 0.25},
        },
        morphology="rc_car8",
    ) is None

    record = sampler.step(
        now=10.04,
        graph=graph(1.1, 0.01),
        controller_input={"r2": 1.0},
        intent={"forward": 1.0},
        effective_actions={
            "smores_01": {"pan_rate_rad_s": 0.25},
        },
        morphology="rc_car8",
    )

    assert record["schema_version"] == "mssr.expert_transition.v3"
    assert record["graph_t"]["stamp"] == 1.0
    assert record["graph_t_plus_1"]["stamp"] == 1.1

    assert (
        record["expert_action"]["locomotion"]["smores_01"]
        ["pan_rate_rad_s"]
        == 0.25
    )

    assert record["observation"]["controller_input"]["r2"] == 1.0
    assert record["observation"]["intent"]["forward"] == 1.0

    assert record["supervision"]["label_source"] == "human_expert"


def test_sampling_is_10_hz_not_control_loop_rate():
    sampler = TransitionSampler(dataset_rate_hz=10.0)

    produced = []

    # Simula il control loop a 50 Hz per circa mezzo secondo.
    for index in range(26):
        now = 20.0 + index * 0.02
        current_graph = graph(100.0 + index, index * 0.001)

        record = sampler.step(
            now=now,
            graph=current_graph,
            controller_input={"r2": 0.5},
            intent={"forward": 0.5},
            effective_actions={
                "smores_01": {"pan_rate_rad_s": 0.1},
            },
            morphology="rc_car8",
        )

        if record is not None:
            produced.append(record)

    # 50 Hz di controllo non deve diventare 50 Hz di dataset.
    assert 4 <= len(produced) <= 6


def test_close_does_not_fabricate_successor_for_pending_transition():
    sampler = TransitionSampler(dataset_rate_hz=10.0)

    sampler.step(
        now=30.0,
        graph=graph(5.0, 0.0),
        controller_input={"r2": 0.0},
        intent={"forward": 0.0},
        effective_actions={
            "smores_01": {"pan_rate_rad_s": 0.0},
        },
        morphology="rc_car8",
    )

    assert sampler.pending
    assert sampler.close() == 1
    assert not sampler.pending
