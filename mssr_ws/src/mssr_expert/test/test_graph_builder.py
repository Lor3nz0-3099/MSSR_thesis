from __future__ import annotations

from mssr_expert.graph.graph_builder import GraphBuilder


def test_smores_roles_connectors_and_docking_endpoints_are_preserved() -> None:
    observation = {
        "stamp": 2.0,
        "modules": [
            {
                "module_id": "m0",
                "robot_family": "smores_ep",
                "module_type": "smores_ep_v1",
                "design_profile": "enhanced_smores_ep_compatible",
                "design_requirements": {
                    "lift_chain_module_count_target": 7,
                },
                "pose": {
                    "position": [0.0, 0.0, 0.03],
                    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
                "current_role": "elbow",
                "target_role": "elbow",
                "role_confidence": 0.9,
                "role_source": "learned",
                "functional_role": {
                    "name": "joint",
                    "effective_dof_count": 1,
                    "axes": ["tilt"],
                },
                "actuators": {"tilt": {"position_rad": 0.3}},
                "connectors": [
                    {
                        "connector_id": "TOP",
                        "status": "connected",
                    }
                ],
                "simulation_fixtures": {
                    "ground_support_anchor": False,
                },
            },
            {
                "module_id": "m1",
                "robot_family": "smores_ep",
                "pose": {
                    "position": [0.08, 0.0, 0.03],
                    "orientation": [0.0, 0.0, 0.0, 1.0],
                },
                "current_role": "wrist",
            },
        ],
        "attachments": [
            {
                "module_a_id": "m0",
                "module_b_id": "m1",
                "face_a": "TOP",
                "face_b": "BOTTOM",
                "connector_type": "smores_ep_face",
                "status": "connected",
                "joint_type": "rigid",
                "clocking_quarter_turns": 1,
                "allowed_relative_dofs": [],
            }
        ],
    }
    graph = GraphBuilder().build(observation)
    nodes = graph.node_by_id()
    assert nodes["m0"].attributes["robot_family"] == "smores_ep"
    assert (
        nodes["m0"].attributes["design_requirements"][
            "lift_chain_module_count_target"
        ]
        == 7
    )
    assert nodes["m0"].attributes["current_role"] == "elbow"
    assert (
        nodes["m0"].attributes["functional_role"]["effective_dof_count"]
        == 1
    )
    assert nodes["m0"].attributes["connectors"][0]["connector_id"] == "TOP"

    edge = graph.edges[0]
    assert edge.attributes["face_a"] == "TOP"
    assert edge.attributes["face_b"] == "BOTTOM"
    assert edge.attributes["clocking_quarter_turns"] == 1
    assert edge.attributes["is_attached"]
