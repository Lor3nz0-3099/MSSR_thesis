from mssr_expert.dataset.dataset_logger import DatasetLogger
from mssr_expert.experts.expert_output import ExpertOutput
from mssr_expert.graph.attributed_robot_graph import (
    AttributedRobotGraph,
    GraphNode,
)


def test_human_record_can_be_built_without_disk_io(tmp_path):
    graph_t = AttributedRobotGraph(
        stamp=1.0,
        nodes=(
            GraphNode(
                "smores_01",
                {
                    "position": [0.0, 0.0, 0.05],
                    "orientation": [0.0, 0.0, 0.0, 1.0],
                },
            ),
        ),
    )
    graph_next = AttributedRobotGraph(
        stamp=1.1,
        nodes=graph_t.nodes,
    )

    path = tmp_path / "human_behavior.jsonl"

    logger = DatasetLogger(
        path,
        label_source="human_expert",
        executed_action_source="human_expert",
    )

    record = logger.build_record(
        episode_id="teleop-001",
        timestep=0,
        observation={"controller_input": {"r2": 0.5}},
        graph=graph_t,
        expert_output=ExpertOutput(
            locomotion={
                "smores_01": {"pan_rate_rad_s": 0.25},
            },
            fsm_state="TELEOP",
        ),
        stage_name="human_behavior",
        stage_id=0,
        task_type="rc_car8_teleop",
        difficulty=0.0,
        next_graph=graph_next,
        next_observation={"controller_input": {"r2": 0.5}},
    )

    assert record["schema_version"] == "mssr.expert_transition.v3"
    assert record["supervision"]["label_source"] == "human_expert"
    assert record["supervision"]["executed_action_source"] == "human_expert"
    assert record["graph_t"]["stamp"] == 1.0
    assert record["graph_t_plus_1"]["stamp"] == 1.1

    # build_record prepara il dato in RAM: non deve scrivere sul disco.
    assert not path.exists()
