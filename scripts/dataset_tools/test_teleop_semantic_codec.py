import copy

from teleop_semantic_codec import TeleopCodec


def record(stamp, depression):
    graph = {
        "stamp": stamp,
        "nodes": [{"id": f"m{i}"} for i in range(8)],
        "edges": [],
        "global_attributes": {"course": {"mission": {"tasks": [
            {"type": "button", "parameters": {"button": {"depression_m": depression}}}
        ]}, "geometry_sha256": "fixed"}, "module_count": 8, "contact_candidate_count": 0},
    }
    next_graph = copy.deepcopy(graph)
    next_graph["stamp"] += 0.1
    row = {
        "schema_version": "test", "episode_id": "teleop-test", "stage_name": "human_behavior",
        "task_type": "rc_car8_teleop", "difficulty": 0.0,
        "graph_t": graph, "graph_t_plus_1": next_graph,
        "task_graph_t": copy.deepcopy(graph),
        "attributed_graph": copy.deepcopy(graph),
        "attributed_task_graph": copy.deepcopy(graph),
        "observation": {"schema_version": "observation.v1", "controller_input": {"x": stamp}},
        "observation_t_plus_1": {"schema_version": "observation.v1", "morphology": "rc_car8"},
        "expert_action": {"locomotion": {"speed": 0.0}},
        "expert_annotation": {"active_primitive": "human_teleop", "debug": {}, "fsm_state": "TELEOP", "primitive_params": {}, "task_metrics": {}},
        "active_primitive": "human_teleop", "debug": {}, "fsm_state": "TELEOP", "primitive_params": {}, "task_metrics": {},
        "stage_id": int(stamp), "timestep": int(stamp), "stamp": stamp,
        "action_valid": True, "supervision": {"valid_for_behavior_cloning": True},
    }
    return row


def test_course_geometry_once_dynamic_button_retained_and_all_rows_restored():
    rows = [record(1.0, 0.0), record(2.0, 0.003), record(3.0, 0.003)]
    codec = TeleopCodec.from_first_record(rows[0])
    encoded = [codec.encode(row) for row in rows]
    assert [codec.decode(row) for row in encoded] == rows
    assert len(encoded) == len(rows)
    assert "course" not in encoded[1]["graph_t"]["global_attributes"] if "global_attributes" in encoded[1]["graph_t"] else True
    assert "attributed_graph" not in encoded[1]
    assert codec.metadata()["global_attributes_baseline"]["course"]["geometry_sha256"] == "fixed"
    assert any("depression_m" in str(path) for path in codec.metadata()["patch_paths"])


def test_dynamic_alias_and_constant_fallback_preserves_unusual_row():
    first = record(1.0, 0.0)
    changed = record(2.0, 0.0)
    changed["task_graph_t"]["stamp"] = -1.0
    changed["stage_name"] = "different"
    changed["observation"]["schema_version"] = "observation.v2"
    codec = TeleopCodec.from_first_record(first)
    compact = codec.encode(changed)
    assert codec.decode(compact) == changed
    assert compact["task_graph_t"]["stamp"] == -1.0
    assert compact["stage_name"] == "different"


def test_archive_reader_restores_every_row(tmp_path):
    import hashlib
    import json

    from compact_teleop_semantic import compact_stream
    from teleop_semantic_codec import iter_decoded_archive

    rows = [record(1.0, 0.0), record(2.0, 0.003), record(3.0, 0.003)]
    source = tmp_path / "human.jsonl"
    raw = "".join(json.dumps(row) + "\n" for row in rows).encode()
    source.write_bytes(raw)
    archive = tmp_path / "human.jsonl.zst"
    meta = compact_stream(source, archive, {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "samples": len(rows),
    })
    assert meta["source_records"] == meta["compact_records"] == len(rows)
    assert list(iter_decoded_archive(archive, meta)) == rows


def test_macro_repeats_collapse_but_decode_restores_every_timestamp(tmp_path):
    import hashlib
    import json

    from compact_teleop_semantic import compact_stream
    from teleop_semantic_codec import iter_decoded_archive

    rows = [record(float(stamp), 0.0) for stamp in (1, 2, 3)]
    for row in rows:
        row["stage_name"] = "self_reconfiguration"
        row["observation"]["controller_input"] = {"x": 0.0}
    rows.append(record(4.0, 0.003))
    rows[-1]["stage_name"] = "self_reconfiguration"
    rows[-1]["observation"]["controller_input"] = {"x": 0.0}
    source = tmp_path / "macro.jsonl"
    raw = b"".join((json.dumps(row) + "\n").encode() for row in rows)
    source.write_bytes(raw)
    archive = tmp_path / "macro.jsonl.zst"
    meta = compact_stream(source, archive, {
        "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
        "samples": len(rows), "producer": "deterministic_expert",
    })
    assert meta["source_records"] == 4
    assert meta["compact_records"] == 2
    assert meta["collapsed_records"] == 2
    assert list(iter_decoded_archive(archive, meta)) == rows


def test_human_pause_keeps_each_original_row(tmp_path):
    import hashlib
    import json

    from compact_teleop_semantic import compact_stream
    from teleop_semantic_codec import iter_decoded_archive

    rows = [record(float(stamp), 0.0) for stamp in (1, 2, 3)]
    for row in rows:
        row["observation"]["controller_input"] = {"x": 0.0}
    source = tmp_path / "human_pause.jsonl"
    raw = b"".join((json.dumps(row) + "\n").encode() for row in rows)
    source.write_bytes(raw)
    archive = tmp_path / "human_pause.jsonl.zst"
    meta = compact_stream(source, archive, {
        "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
        "samples": len(rows), "producer": "human_expert",
    })
    assert meta["source_records"] == meta["compact_records"] == 3
    assert meta["collapsed_records"] == 0
    assert list(iter_decoded_archive(archive, meta)) == rows
