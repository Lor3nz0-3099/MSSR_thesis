import copy
import unittest

from preprocessing import CanonicalTimestepBuilder


def graph(stamp, *, connected=False):
    def node(mid, x):
        return {
            "module_id": mid,
            "node_id": mid,
            "attributes": {
                "module_id": mid,
                "position": [x, 0.0, 0.0],
                "orientation": [0.0, 0.0, 0.0, 1.0],
                "pose": {"position": [x, 0.0, 0.0]},
                "linear_velocity": [0.0, 0.0, 0.0],
                "angular_velocity": [0.0, 0.0, 0.0],
                "actuators": {
                    "pan": {"position_rad": 0.0, "velocity_rad_s": 0.0},
                    "tilt": {"position_rad": 0.0, "velocity_rad_s": 0.0},
                },
                "connectors": [
                    {"connector_id": "TOP", "status": "connected" if connected else "available"},
                    {"connector_id": "BOTTOM", "status": "available"},
                ],
                "current_role": "unassigned",
            },
        }
    edges = []
    if connected:
        edges = [{
            "module_a_id": "smores_01", "module_b_id": "smores_02",
            "attributes": {"face_a": "TOP", "face_b": "TOP", "is_attached": True},
        }]
    return {
        "stamp": stamp,
        "nodes": [node("smores_01", 0.0), node("smores_02", 0.1)],
        "edges": edges,
        "global_attributes": {"course": {"gap": {
            "near_edge_x_m": 1.0, "far_edge_x_m": 1.2, "width_m": 0.2,
        }, "course_profile": "gap_test"}},
    }


def base_record(stamp=1.0, *, connected=False):
    return {
        "stamp": stamp,
        "timestep": int(stamp * 10),
        "task_type": "self_assembly",
        "_loader_phase": "assembly",
        "_loader_source_path": "assembly.jsonl",
        "graph_t": graph(stamp, connected=connected),
        "graph_t_plus_1": graph(stamp + 0.2, connected=connected),
        "action_valid": True,
        "supervision": {"valid_for_behavior_cloning": True},
        "expert_action": {},
        "debug": {"command_id": "cmd"},
        "fsm_state": "RUNNING",
    }


def primitive_record(stamp, goal_id, primitive, module_ids, parameters, *, connected=False):
    row = base_record(stamp, connected=connected)
    row["expert_action"] = {"primitive_goal": {
        "goal_id": goal_id,
        "primitive": primitive,
        "module_ids": list(module_ids),
        "parameters": dict(parameters),
    }}
    return row


class CanonicalV5Tests(unittest.TestCase):
    def test_parallel_internal_commands_on_different_modules_form_one_joint_action(self):
        builder = CanonicalTimestepBuilder()
        a = primitive_record(1.0, "a", "set_tilt", ["smores_01"], {"angle_rad": 0.2})
        b = primitive_record(1.0, "b", "set_tilt", ["smores_02"], {"angle_rad": -0.3})
        result = builder.build_bucket([a, b])
        modules = result["action_t"]["modules"]
        self.assertEqual(modules["smores_01"]["internal"]["mode"], "tilt_target")
        self.assertEqual(modules["smores_02"]["internal"]["mode"], "tilt_target")
        self.assertTrue(result["supervision"]["module_resource_masks"]["smores_01"]["internal"])
        self.assertTrue(result["supervision"]["module_resource_masks"]["smores_02"]["internal"])

    def test_pan_and_tilt_same_module_same_physical_state_are_not_bc_targets(self):
        builder = CanonicalTimestepBuilder()
        a = primitive_record(1.0, "pan", "set_pan", ["smores_01"], {"angle_rad": 0.2})
        b = primitive_record(1.0, "tilt", "set_tilt", ["smores_01"], {"angle_rad": -0.3})
        result = builder.build_bucket([a, b])
        self.assertFalse(result["supervision"]["module_resource_masks"]["smores_01"]["internal"])
        self.assertEqual(len(result["supervision"]["conflicts"]), 1)
        self.assertEqual(result["supervision"]["conflicts"][0]["reason"],
                         "multiple_discrete_decisions_without_intermediate_state")

    def test_last_continuous_internal_revision_wins_on_cached_state(self):
        builder = CanonicalTimestepBuilder()
        a = base_record(1.0)
        b = base_record(1.0)
        a["expert_action"] = {"locomotion": {"smores_01": {"pan_rate_rad_s": 0.2}}}
        b["expert_action"] = {"locomotion": {"smores_01": {"pan_rate_rad_s": 0.8}}}
        result = builder.build_bucket([a, b])
        internal = result["action_t"]["modules"]["smores_01"]["internal"]
        self.assertEqual(internal["mode"], "pan_velocity")
        self.assertAlmostEqual(internal["value"], 0.8)
        self.assertTrue(result["supervision"]["module_resource_masks"]["smores_01"]["internal"])

    def test_set_tilt_makes_internal_busy_until_recorded_barrier(self):
        builder = CanonicalTimestepBuilder()
        dispatch = primitive_record(1.0, "tilt", "set_tilt", ["smores_01"], {"angle_rad": 0.7})
        first = builder.build_bucket([dispatch])
        self.assertTrue(first["graph_t"]["nodes"][0]["control_state"]["internal"]["available"])

        waiting = base_record(1.2)
        waiting["supervision"] = {"valid_for_behavior_cloning": False}
        second = builder.build_bucket([waiting])
        by_id = {node["module_id"]: node for node in second["graph_t"]["nodes"]}
        self.assertFalse(by_id["smores_01"]["control_state"]["internal"]["available"])
        self.assertFalse(second["action_t"]["modules"]["smores_01"]["internal"]["commanded"])
        self.assertFalse(second["supervision"]["module_resource_masks"]["smores_01"]["internal"])

        barrier = base_record(1.4)
        barrier["supervision"] = {"valid_for_behavior_cloning": False}
        barrier["fsm_state"] = "PROGRAM_BARRIER"
        barrier["debug"]["message"] = "posture reached"
        third = builder.build_bucket([barrier])
        by_id = {node["module_id"]: node for node in third["graph_t"]["nodes"]}
        self.assertTrue(by_id["smores_01"]["control_state"]["internal"]["available"])


    def test_successful_source_completion_clears_structural_primitive_busy_state(self):
        builder = CanonicalTimestepBuilder()
        dock = primitive_record(1.0, "dock-final", "dock",
                                ["smores_01", "smores_02"],
                                {"face_a": "TOP", "face_b": "BOTTOM"})
        first = builder.build_bucket([dock])
        self.assertTrue(first["graph_t"]["nodes"][0]["control_state"]["wheels"]["available"])

        done = base_record(1.2)
        done["expert_action"] = {}
        done["supervision"] = {"valid_for_behavior_cloning": False}
        done["done"] = True
        done["success"] = True
        second = builder.build_bucket([done])
        by_id = {node["module_id"]: node for node in second["graph_t"]["nodes"]}
        self.assertTrue(by_id["smores_01"]["control_state"]["wheels"]["available"])
        self.assertTrue(by_id["smores_02"]["control_state"]["wheels"]["available"])
        self.assertEqual(by_id["smores_01"]["control_state"]["active_primitives"], [])
        self.assertEqual(by_id["smores_02"]["control_state"]["active_primitives"], [])


    def test_expert_phase_change_clears_previous_phase_structural_busy_without_done_flag(self):
        builder = CanonicalTimestepBuilder()
        dock = primitive_record(1.0, "dock-final", "dock",
                                ["smores_01", "smores_02"],
                                {"face_a": "TOP", "face_b": "BOTTOM"})
        first = builder.build_bucket([dock])
        self.assertTrue(first["graph_t"]["nodes"][0]["control_state"]["wheels"]["available"])

        behavior = base_record(1.2)
        behavior["_loader_phase"] = "behavior"
        behavior["_loader_source_path"] = "behavior.jsonl"
        behavior["expert_action"] = {}
        behavior["supervision"] = {"valid_for_behavior_cloning": False}
        second = builder.build_bucket([behavior])
        by_id = {node["module_id"]: node for node in second["graph_t"]["nodes"]}
        self.assertTrue(by_id["smores_01"]["control_state"]["wheels"]["available"])
        self.assertTrue(by_id["smores_02"]["control_state"]["wheels"]["available"])
        self.assertEqual(by_id["smores_01"]["control_state"]["active_primitives"], [])
        self.assertEqual(by_id["smores_02"]["control_state"]["active_primitives"], [])

    def test_connected_idle_module_reports_inferred_structural_hold_but_is_commandable(self):
        builder = CanonicalTimestepBuilder()
        row = base_record(1.0, connected=True)
        row["supervision"] = {"valid_for_behavior_cloning": False}
        result = builder.build_bucket([row])
        for node in result["graph_t"]["nodes"]:
            internal = node["control_state"]["internal"]
            self.assertEqual(internal["mode"], "structural_hold")
            self.assertEqual(internal["mode_source"], "executor_default_inferred")
            self.assertTrue(internal["available"])

    def test_align_faces_is_one_policy_action_while_later_phase_is_executor_state(self):
        builder = CanonicalTimestepBuilder()
        params = {"face_a": "BOTTOM", "face_b": "TOP", "execution_phase": "reach",
                  "clocking_quarter_turns": 0}
        reach = primitive_record(1.0, "wave-a0-reach", "align_faces",
                                 ["smores_01", "smores_02"], params)
        reach["debug"]["active_goal_ids"] = ["wave-a0-reach"]
        first = builder.build_bucket([reach])
        self.assertEqual(first["action_t"]["modules"]["smores_01"]["structural"]["primitive"],
                         "align_faces")

        align = primitive_record(1.2, "wave-a0-align", "align_faces",
                                 ["smores_01", "smores_02"],
                                 {**params, "execution_phase": "align"})
        align["debug"]["active_goal_ids"] = ["wave-a0-align"]
        second = builder.build_bucket([align])
        self.assertFalse(second["action_t"]["modules"]["smores_01"]["structural"]["commanded"])
        by_id = {node["module_id"]: node for node in second["graph_t"]["nodes"]}
        self.assertTrue(by_id["smores_01"]["control_state"]["active_primitives"])


    def test_post_undock_executor_history_overrides_one_frame_stale_physical_edge(self):
        builder = CanonicalTimestepBuilder()

        undock = primitive_record(1.0, "undock-old", "undock",
                                  ["smores_01", "smores_02"],
                                  {"face_a": "TOP", "face_b": "TOP"},
                                  connected=True)
        undock["debug"]["active_goal_ids"] = ["undock-old"]
        first = builder.build_bucket([undock])
        first_nodes = {node["module_id"]: node for node in first["graph_t"]["nodes"]}
        self.assertTrue(first_nodes["smores_01"]["control_state"]["faces"]["TOP"]["undock_available"])
        self.assertFalse(first_nodes["smores_01"]["control_state"]["faces"]["TOP"]["align_faces_available"])

        # The next physical snapshot still carries the old edge, but the
        # executor has completed UNDOCK and already accepts a new alignment.
        align = primitive_record(1.2, "align-new-reach", "align_faces",
                                 ["smores_01", "smores_02"],
                                 {"face_a": "TOP", "face_b": "TOP",
                                  "execution_phase": "reach",
                                  "clocking_quarter_turns": 0},
                                 connected=True)
        align["debug"]["active_goal_ids"] = ["align-new-reach"]
        second = builder.build_bucket([align])
        second_nodes = {node["module_id"]: node for node in second["graph_t"]["nodes"]}
        face = second_nodes["smores_01"]["control_state"]["faces"]["TOP"]
        self.assertTrue(face["physical_connected"])
        self.assertFalse(face["connected_for_executor"])
        self.assertEqual(face["availability_source"], "executor_structural_history")
        self.assertTrue(face["align_faces_available"])
        self.assertTrue(second["supervision"]["module_resource_masks"]["smores_01"]["structural"])

    def test_legacy_button_sequence_is_preserved_but_not_low_level_bc(self):
        builder = CanonicalTimestepBuilder()
        row = base_record(1.0)
        row["expert_action"] = {
            "joint_configuration": {"target_rad": {"arm.pan": 0.2, "arm.tilt": 0.4}},
            "primitive_sequence": [
                {"goal_id": "p", "primitive": "set_pan", "module_ids": ["smores_01"],
                 "parameters": {"angle_rad": 0.2}},
                {"goal_id": "t", "primitive": "set_tilt", "module_ids": ["smores_01"],
                 "parameters": {"angle_rad": 0.4}},
            ],
        }
        result = builder.build_bucket([row])
        self.assertFalse(result["supervision"]["valid_for_behavior_cloning"])
        self.assertEqual(len(result["provenance"]["unresolved_recorded_sequences"]), 1)
        self.assertFalse(result["action_t"]["modules"]["smores_01"]["internal"]["commanded"])

    def test_legacy_button_episode_is_quarantined_until_low_level_rerecording(self):
        from canonical_dataset import iter_canonical_episode
        from dataset_loader import EpisodeSpec
        from unittest.mock import patch
        import tempfile
        from pathlib import Path

        row = base_record(1.0)
        row["expert_action"] = {
            "joint_configuration": {"target_rad": {"arm.pan": 0.2}},
            "primitive_sequence": [
                {"goal_id": "p", "primitive": "set_pan", "module_ids": ["smores_01"],
                 "parameters": {"angle_rad": 0.2}},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            spec = EpisodeSpec("expert:button:seed-000001", "expert", Path(directory),
                               task="button", seed=1)
            with patch("canonical_dataset.iter_episode_records", return_value=iter([row])):
                rows = list(iter_canonical_episode(spec, repo_root=directory))
        self.assertEqual(len(rows), 1)
        self.assertFalse(rows[0]["supervision"]["valid_for_behavior_cloning"])
        self.assertEqual(rows[0]["supervision"]["episode_exclusion_reason"],
                         "legacy_button_requires_low_level_rerecording")


if __name__ == "__main__":
    unittest.main()
