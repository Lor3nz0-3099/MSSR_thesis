"""Generic contract examples and explicitly SMORES-specific adapter fixtures."""

from pathlib import Path

import pytest

from drl.IL.bc_v1.bc_v1_contract import (
    ActionSpec,
    ConnectorRef,
    ConnectorSpec,
    ConnectorState,
    CoordinateSpec,
    CoordinateState,
    ModuleRef,
    ModuleState,
    ModuleTypeSpec,
    NumericFeature,
    ParameterSpec,
    ResourceModeSpec,
    ResourceRef,
    ResourceSpec,
    ResourceState,
    RobotSpec,
    StateGraph,
)


def require_canonical_v6_fixture(repo_root):
    """Skip only an absent external prerequisite; consumers validate its content."""
    manifest = Path(repo_root) / "datasets/expert_v1_canonical_v6/manifest.json"
    # Existing invalid paths (including dangling links) must fail at the reader.
    if not manifest.exists() and not manifest.is_symlink():
        pytest.skip("Canonical v6 external fixture is not installed")
    return manifest


def robot_spec(connector_count=1, *, surface=False):
    connectors = tuple(
        ConnectorSpec(connector_id=f"port-{i}", kind="fixed_port")
        for i in range(connector_count)
    )
    if surface:
        connectors += (
            ConnectorSpec(
                connector_id="shell", kind="surface",
                geometry=(NumericFeature("radius_m", (0.05,)),),
            ),
        )
    resource = ResourceSpec(
        resource_id="joint",
        coordinates=(CoordinateSpec("rotation", "rad", periodic=True),),
        modes=(ResourceModeSpec("rate", payload_size=1, units=("rad/s",)),),
        capabilities=("rotate",),
    )
    return RobotSpec(
        schema_version="mssr.bc_v1.robot_spec.v1",
        robot_id="synthetic",
        module_types=(ModuleTypeSpec("unit", connectors, (resource,)),),
        actions=(
            ActionSpec("connect", kind="connect", scope="connector_pair"),
            ActionSpec(
                "neutral_recovery", kind="skill", scope="modules",
                parameters=(
                    ParameterSpec("duration", kind="scalar", units="s"),
                    ParameterSpec("passive_modules", kind="module_refs"),
                    ParameterSpec("clear_on_finish", kind="boolean"),
                ),
            ),
        ),
    )


def state_graph(module_count, connector_count, *, surface=False):
    modules, connectors, resources = [], [], []
    for i in range(module_count):
        module = ModuleRef(f"unit-{i}")
        modules.append(ModuleState(
            ref=module, module_type_id="unit",
            position=(float(i), 0.0, 0.0),
            orientation_xyzw=(0.0, 0.0, 0.0, 1.0), pose_valid=True,
        ))
        for j in range(connector_count):
            connectors.append(ConnectorState(
                ref=ConnectorRef(module, f"port-{j}"), kind="fixed_port",
            ))
        if surface:
            connectors.append(ConnectorState(
                ref=ConnectorRef(module, "shell"), kind="surface",
            ))
        resources.append(ResourceState(
            ref=ResourceRef(module, "joint"),
            coordinates=(CoordinateState("rotation", position=0.1, velocity=0.0),),
        ))
    return StateGraph(tuple(modules), tuple(connectors), tuple(resources))


def generic_surface_spec():
    """Type catalog with a surface, seven ports and a three-component mode."""
    from dataclasses import replace

    shell = robot_spec(0, surface=True).module_types[0]
    drive = ResourceSpec(
        "drive",
        coordinates=(CoordinateSpec("travel", "m", limits=(-2.0, 2.0)),),
        modes=(ResourceModeSpec(
            "twist", 3, ("m/s", "m/s", "rad/s"), codec="continuous",
        ),),
        capabilities=("translate",),
    )
    probe = ModuleTypeSpec(
        "probe",
        connectors=robot_spec(7).module_types[0].connectors,
        resources=(drive,),
    )
    actions = (
        ActionSpec(
            "contact", "prepare_connection", "connector_pair",
            parameters=(
                ParameterSpec("peer", "module_ref", module_type_id="unit"),
                ParameterSpec("surface", "connector_ref", module_type_id="unit",
                              connector_id="shell"),
                ParameterSpec("uv", "vector", units="1", size=2,
                              limits=(-1.0, 1.0)),
                ParameterSpec("quality", "category", categories=("loose", "tight")),
            ),
            required_capabilities=("rotate",),
        ),
        ActionSpec(
            "drive", "continuous", "resource",
            parameters=(ParameterSpec(
                "speed", "scalar", units="m/s", module_type_id="probe",
                resource_id="drive",
            ),),
        ),
        robot_spec(0).actions[1],
    )
    return replace(robot_spec(0, surface=True), module_types=(shell, probe),
                   actions=actions)


def robot_spec_document(spec=None):
    """Fresh external JSON representation, suitable for negative mutations."""
    from dataclasses import asdict
    import json

    return json.loads(json.dumps(asdict(robot_spec() if spec is None else spec)))


def schema_accepts(document, schema):
    """Test-only interpreter for the schema's closed JSON Schema keyword subset.

    Exercises the actual artifact without adding jsonschema to runtime/tests.
    Unknown validation keywords fail loudly rather than being silently ignored.
    Cross-reference/physical checks remain the Python validator's responsibility.
    """
    import math
    import re

    annotations = {"$schema", "$id", "$comment", "$defs", "title", "description", "default"}
    keywords = {"$ref", "type", "const", "enum", "not", "anyOf", "allOf", "if", "then", "else",
                "properties", "required", "additionalProperties", "items", "minItems", "maxItems",
                "uniqueItems", "minLength", "pattern", "minimum"}

    def check(value, rule):
        assert not set(rule) - annotations - keywords, "Unimplemented schema keyword"
        if "$ref" in rule:
            return check(value, schema["$defs"][rule["$ref"].split("/")[-1]])
        types = {
            "null": value is None, "boolean": type(value) is bool,
            "object": isinstance(value, dict), "array": isinstance(value, list),
            "string": isinstance(value, str),
            "number": type(value) in (int, float) and math.isfinite(value),
            "integer": type(value) is int or type(value) is float and math.isfinite(value) and value.is_integer(),
        }
        if "type" in rule:
            allowed = rule["type"] if isinstance(rule["type"], list) else [rule["type"]]
            if not any(types[t] for t in allowed): return False
        if "const" in rule and value != rule["const"]: return False
        if "enum" in rule and value not in rule["enum"]: return False
        if "not" in rule and check(value, rule["not"]): return False
        if "anyOf" in rule and not any(check(value, r) for r in rule["anyOf"]): return False
        if "allOf" in rule and not all(check(value, r) for r in rule["allOf"]): return False
        if "if" in rule:
            branch = "then" if check(value, rule["if"]) else "else"
            if branch in rule and not check(value, rule[branch]): return False
        if isinstance(value, dict):
            if any(k not in value for k in rule.get("required", [])): return False
            properties = rule.get("properties", {})
            if rule.get("additionalProperties") is False and set(value) - set(properties): return False
            if not all(check(v, properties[k]) for k, v in value.items() if k in properties): return False
        if isinstance(value, list):
            if len(value) < rule.get("minItems", 0) or len(value) > rule.get("maxItems", float("inf")): return False
            if "items" in rule and not all(check(v, rule["items"]) for v in value): return False
            if rule.get("uniqueItems") and any(v in value[:i] for i, v in enumerate(value)): return False
        if isinstance(value, str):
            if len(value) < rule.get("minLength", 0): return False
            if "pattern" in rule and not re.search(rule["pattern"], value): return False
        if type(value) in (int, float) and "minimum" in rule and value < rule["minimum"]: return False
        return True

    return check(document, schema)


def smores_graph():
    """Canonical current-state shape; deliberately two modules, not eight."""
    return {
        "nodes": [
            {"module_id": mid, "position": [3., 4., 0.],
             "orientation": [0., 0., 2 ** -.5, 2 ** -.5],
             "linear_velocity": [0.1, 0., 0.], "angular_velocity": [0., 0., 0.2],
             "actuators": {"pan": {"position_rad": 0.4, "velocity_rad_s": 0.1},
                           "tilt": {"position_rad": -0.3}},
             "faces": {"TOP": {"state": "free", "position": [3., 4.05, 0.02],
                                "normal": [0., 1., 0.], "tangent": [-1., 0., 0.]},
                       "BOTTOM": {"state": "free"}},
             "role": "FORBIDDEN_TEMPLATE_ROLE", "control_state": {"busy": True}}
            for mid in ("moving", "peer")
        ], "edges": [], "morphology": "FORBIDDEN_MORPHOLOGY",
        "root_module_id": "moving", "target_morphology": "FORBIDDEN_TARGET",
    }


def smores_label(primitive=None, *, peer=True, internal=None, wheels=False):
    """Only explicitly commanded resources receive a loss mask."""
    modules = {mid: {"participates_in": []} for mid in ("moving", "peer")}
    masks = {mid: {"wheels": False, "internal": False, "structural": False}
             for mid in modules}
    active = modules["moving"]
    if wheels:
        active["wheels"] = {"commanded": True, "left_velocity_rad_s": 0.8,
                            "right_velocity_rad_s": -0.2}
        masks["moving"]["wheels"] = True
    if internal:
        active["internal"] = {"commanded": True, "mode": internal, "value": 0.3,
                              "unit": "rad/s" if internal.endswith("velocity") else "rad"}
        masks["moving"]["internal"] = True
    if primitive:
        params = ({"duration_s": 1.5, "passive_module_ids": ["moving", "peer"]}
                  if primitive == "gravity_settle" else
                  {"face_a": "TOP", "face_b": "BOTTOM", "clocking_quarter_turns": 1})
        active["structural"] = {"commanded": True, "primitive": primitive,
                                "parameters": params}
        masks["moving"]["structural"] = True
        if peer and primitive != "gravity_settle":
            modules["peer"]["participates_in"] = [{"primitive": primitive,
                "initiator_module_id": "moving", "own_face": "BOTTOM",
                "initiator_face": "TOP"}]
    return {"modules": modules}, {"module_resource_masks": masks}


def smores_source_goal(primitive="align_faces", peer="peer", goal_id="decision-1"):
    return {"expert_action": {"primitive_goal": {
        "goal_id": goal_id, "primitive": primitive, "module_ids": ["moving", peer],
        "parameters": {"face_a": "TOP", "face_b": "BOTTOM", "clocking_quarter_turns": 1,
                       "execution_phase": "REACH"}}},
        "graph_t": {"target_graph": "FORBIDDEN"}, "graph_t_plus_1": "FORBIDDEN"}


def task_row(task=None, stage=None):
    """Current semantic fields, independent of robot hardware and labels."""
    observation = {}
    if task is not None:
        observation["task_type"] = task
    if stage is not None:
        observation["stage_name"] = stage
    return {"observation_t": observation, "graph_t": {"stamp": 1.0}}


def gap_metadata(width=0.2, frame="world"):
    course = {"gap": {"width_m": width, "near_edge_x_m": 1.0,
                       "far_edge_x_m": 1.0 + width if type(width) in (int, float) else 1.2}}
    if frame is not None:
        course["frame_id"] = frame
    return {"course": course}


def mission_task_fixture(index=100):
    """Declared mission + independently specified historical handoff ranges."""
    tasks = [
        {"task_id": "nav-1", "type": "flat_navigation", "parameters": {
            "cone_centers_xy_m": [[1., 2.]], "cone_radius_m": .03}},
        {"task_id": "gap-2", "type": "gap", "parameters": {
            "gap": {"width_m": .2, "near_edge_world_xyz_m": [3., 0., 0.]}}},
        {"task_id": "stairs-3", "type": "stairs", "parameters": {
            "stairs": {"rise_m": .1, "step_count": 3,
                       "first_riser_center_world_xyz_m": [5., 0., 0.]}}},
        {"task_id": "nav-return", "type": "flat_navigation", "parameters": {
            "cone_centers_xy_m": [[7., 2.]], "cone_radius_m": .03}},
        {"task_id": "button-4", "type": "button", "parameters": {
            "button": {"center_xyz_m": [9., 0., .2], "plunger_stroke_m": .004,
                       "current_center_xyz_m": [8.998, 0., .2], "depression_m": .002}}},
        {"task_id": "goal", "type": "goal", "parameters": {"center_xyz_m": [11., 0., 0.]}},
    ]
    course = {"frame_id": "world", "mission": {"tasks": tasks}, "collision_boxes": [
        {"name": "gap_2_Bank", "center_xyz_m": [3., 0., -.1], "size_xyz_m": [1., 1., .2]},
        {"name": "stairs_3_Step", "center_xyz_m": [5., 0., .1], "size_xyz_m": [1., 1., .2]},
        {"name": "button_4_Wall", "center_xyz_m": [9., 0., .2], "size_xyz_m": [.1, 1., 1.]},
    ], "navigation_cones": [
        {"task_id": "nav-1", "center_xyz_m": [1., 2., .1], "radius_m": .03, "height_m": .2}]}
    row = {**task_row(stage="reconfiguration"), "provenance": {
        "logical_episode_id": "mission-fixture", "canonical_transition_index": index}}
    protocol = {"episode_id": "mission-fixture", "mission_order": [
        {"task_id": t["task_id"], "type": t["type"]} for t in tasks], "intervals": [
        {"start_index": 100, "end_index": 199, "predecessor_task_id": "nav-1", "successor_task_id": "gap-2"},
        {"start_index": 300, "end_index": 399, "predecessor_task_id": "gap-2", "successor_task_id": "stairs-3"},
        {"start_index": 500, "end_index": 599, "predecessor_task_id": "stairs-3", "successor_task_id": "nav-return"},
        {"start_index": 700, "end_index": 799, "predecessor_task_id": "nav-return", "successor_task_id": "button-4"},
        {"start_index": 900, "end_index": 999, "predecessor_task_id": "button-4", "successor_task_id": "goal"},
    ]}
    return row, {"course": course}, protocol
