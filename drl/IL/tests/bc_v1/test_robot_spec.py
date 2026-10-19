"""A2 behavior: validate catalogs before any adapter or tensorization."""

import copy
import json
import re
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from drl.IL.bc_v1 import bc_v1_contract as c
from drl.IL.tests.bc_v1 import fixtures


def api():
    # Missing A2 behavior is an assertion failure, never a collection/import error.
    for name in ("load_robot_spec", "validate_robot_spec", "robot_spec_fingerprint"):
        assert callable(getattr(c, name, None)), f"Missing A2 interface: {name}"
    return c


def load_document(tmp_path, document):
    path = tmp_path / "robot.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return api().load_robot_spec(path)


@pytest.mark.parametrize("field,value", [
    ("module_type_id", "missing-type"),
    ("resource_id", "missing-resource"),
    ("connector_id", "missing-connector"),
])
def test_spec_rejects_dangling_resource_and_connector_refs(tmp_path, field, value):
    api()
    spec = fixtures.generic_surface_spec()
    data = fixtures.robot_spec_document(spec)
    argument = data["actions"][1 if field == "resource_id" else 0]["parameters"][
        0 if field == "resource_id" else 1
    ]
    argument[field] = value
    with pytest.raises(ValueError, match=field):
        load_document(tmp_path, data)


@pytest.mark.parametrize("fault", ["duplicate", "nan", "inf", "-inf", "codec"])
def test_spec_rejects_duplicate_ids_nonfinite_limits_and_unknown_codec(fault):
    validator = api().validate_robot_spec
    spec = fixtures.robot_spec()
    module = spec.module_types[0]
    resource = module.resources[0]
    if fault == "duplicate":
        spec = replace(spec, module_types=(module, module))
    elif fault == "codec":
        mode = replace(resource.modes[0], codec="load_python_plugin")
        resource = replace(resource, modes=(mode,))
        spec = replace(spec, module_types=(replace(module, resources=(resource,)),))
    else:
        coord = replace(resource.coordinates[0], limits=(0.0, float(fault)))
        resource = replace(resource, coordinates=(coord,))
        spec = replace(spec, module_types=(replace(module, resources=(resource,)),))
    with pytest.raises(ValueError):
        validator(spec)


def test_generic_surface_and_variable_payload_spec_roundtrip(tmp_path):
    api()
    original = fixtures.generic_surface_spec()
    loaded = load_document(tmp_path, fixtures.robot_spec_document(original))
    assert len(loaded.module_types) == 2
    modules = {m.type_id: m for m in loaded.module_types}
    assert modules["unit"].connectors[0].kind == "surface"
    assert len(modules["probe"].connectors) == 7
    mode = modules["probe"].resources[0].modes[0]
    assert mode.payload_size == 3 and mode.units == ("m/s", "m/s", "rad/s")
    assert api().robot_spec_fingerprint(original) == api().robot_spec_fingerprint(loaded)
    recovery = next(a for a in loaded.actions if a.action_id == "neutral_recovery")
    assert recovery.scope == "modules" and recovery.enabled


def test_spec_fingerprint_is_order_canonical(tmp_path):
    api()
    data = fixtures.robot_spec_document(fixtures.generic_surface_spec())
    reordered = copy.deepcopy(data)
    reordered["module_types"].reverse()
    reordered["actions"].reverse()
    for module in reordered["module_types"]:
        module["connectors"].reverse()
        module["resources"].reverse()
    for action in reordered["actions"]:
        action["parameters"].reverse()
    first = load_document(tmp_path, data)
    second = load_document(tmp_path, dict(reversed(list(reordered.items()))))
    assert first == second  # Irrelevant catalog order normalized at load time.
    assert re.fullmatch("[0-9a-f]{64}", api().robot_spec_fingerprint(first))
    assert api().robot_spec_fingerprint(first) == api().robot_spec_fingerprint(second)


@pytest.mark.parametrize("module_count,connector_count", [(1, 0), (3, 1), (5, 6)])
def test_fixed_ports_and_type_cardinality_are_variable(tmp_path, module_count, connector_count):
    api()
    spec = fixtures.robot_spec(connector_count)
    spec = replace(spec, module_types=tuple(
        replace(spec.module_types[0], type_id=f"type-{i}") for i in range(module_count)
    ))
    loaded = load_document(tmp_path, fixtures.robot_spec_document(spec))
    assert len(loaded.module_types) == module_count
    assert all(len(m.connectors) == connector_count for m in loaded.module_types)


def test_enabled_defaults_and_explicit_override_are_semantic(tmp_path):
    api()
    data = fixtures.robot_spec_document()
    for action in data["actions"]:
        action.pop("enabled")
    default = load_document(tmp_path, data)
    assert all(a.enabled for a in default.actions)
    data["actions"][1]["enabled"] = False
    override = load_document(tmp_path, data)
    assert next(a for a in override.actions if a.action_id == "neutral_recovery").enabled is False
    assert api().robot_spec_fingerprint(default) != api().robot_spec_fingerprint(override)
    data["label_support"] = {"neutral_recovery": 0}
    with pytest.raises(ValueError, match="label_support"):
        load_document(tmp_path, data)


@pytest.mark.parametrize("scope", ["module_types", "actions", "connectors", "resources",
                                   "coordinates", "modes", "parameters", "capabilities"])
def test_duplicate_ids_are_rejected_in_their_scopes(tmp_path, scope):
    api()
    data = fixtures.robot_spec_document()
    resource = data["module_types"][0]["resources"][0]
    target = {
        "module_types": data["module_types"], "actions": data["actions"],
        "connectors": data["module_types"][0]["connectors"],
        "resources": data["module_types"][0]["resources"],
        "coordinates": resource["coordinates"], "modes": resource["modes"],
        "parameters": data["actions"][1]["parameters"],
        "capabilities": resource["capabilities"],
    }[scope]
    target.append(copy.deepcopy(target[0]))
    with pytest.raises(ValueError, match="duplicate"):
        load_document(tmp_path, data)


@pytest.mark.parametrize("case", [
    "units_length", "empty_units", "inverted_limits", "bool_limits",
    "zero_payload", "float_payload", "zero_vector", "category_empty",
    "category_duplicate", "category_units", "scalar_categories", "ref_units",
    "ref_set_codec", "unknown_argument_codec", "missing_owner", "bad_capability",
    "resource_unit_mismatch",
])
def test_invalid_units_limits_payloads_and_argument_definitions(tmp_path, case):
    api()
    data = fixtures.robot_spec_document(fixtures.generic_surface_spec())
    mode = data["module_types"][1]["resources"][0]["modes"][0]
    params = data["actions"][0]["parameters"]
    if case == "units_length": mode["units"] = ["m/s"]
    elif case == "empty_units": mode["units"][0] = ""
    elif case == "inverted_limits": params[2]["limits"] = [2, -2]
    elif case == "bool_limits": params[2]["limits"] = [False, 1]
    elif case == "zero_payload": mode["payload_size"] = 0
    elif case == "float_payload": mode["payload_size"] = 3.5
    elif case == "zero_vector": params[2]["size"] = 0
    elif case == "category_empty": params[3]["categories"] = []
    elif case == "category_duplicate": params[3]["categories"] = ["same", "same"]
    elif case == "category_units": params[3]["units"] = "rad"
    elif case == "scalar_categories": params[2]["categories"] = ["unexpected"]
    elif case == "ref_units": params[0]["units"] = "m"
    elif case == "ref_set_codec": data["actions"][2]["parameters"][1]["codec"] = "module_ref"
    elif case == "unknown_argument_codec": params[0]["codec"] = "python"
    elif case == "missing_owner": params[1]["module_type_id"] = None
    elif case == "bad_capability": data["actions"][0]["required_capabilities"] = ["missing"]
    elif case == "resource_unit_mismatch": data["actions"][1]["parameters"][0]["units"] = "s"
    with pytest.raises(ValueError):
        load_document(tmp_path, data)


@pytest.mark.parametrize("fault", ["bad_kind", "pose_length", "zero_quaternion",
                                   "nonunit_quaternion", "nonfinite_geometry", "empty_surface"])
def test_malformed_connector_variants_fail(tmp_path, fault):
    api()
    data = fixtures.robot_spec_document(fixtures.robot_spec(0, surface=True))
    connector = data["module_types"][0]["connectors"][0]
    if fault == "bad_kind": connector["kind"] = "face_enum"
    elif fault == "pose_length": connector["local_position"] = [0, 0]
    elif fault == "zero_quaternion": connector["local_orientation_xyzw"] = [0, 0, 0, 0]
    elif fault == "nonunit_quaternion": connector["local_orientation_xyzw"] = [0, 0, 0, 2]
    elif fault == "nonfinite_geometry": connector["geometry"][0]["values"] = [float("inf")]
    elif fault == "empty_surface": connector["geometry"] = []
    with pytest.raises(ValueError):
        load_document(tmp_path, data)


@pytest.mark.parametrize("field,value", [
    ("schema_version", "mssr.bc_v1.robot_spec.v2"), ("robot_id", ""),
    ("module_types", {}), ("observed_label_support", 24),
])
def test_external_schema_mismatch_and_unknown_fields_fail(tmp_path, field, value):
    api()
    data = fixtures.robot_spec_document()
    data[field] = value
    with pytest.raises(ValueError, match=field):
        load_document(tmp_path, data)


@pytest.mark.parametrize("text", ["{", "[]", "null", "42",
                                  '{"robot_id":"a","robot_id":"b"}'])
def test_loader_rejects_malformed_nonobject_and_duplicate_key_json(tmp_path, text):
    api()
    path = tmp_path / "robot.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        api().load_robot_spec(path)


def test_loaded_contract_is_deeply_immutable(tmp_path):
    api()
    data = fixtures.robot_spec_document()
    loaded = load_document(tmp_path, data)
    data["module_types"][0]["connectors"][0]["local_position"][0] = 999
    assert loaded.module_types[0].connectors[0].local_position == (0.0, 0.0, 0.0)
    with pytest.raises(FrozenInstanceError):
        loaded.robot_id = "other"
    assert isinstance(loaded.module_types, tuple)
    assert isinstance(loaded.module_types[0].resources[0].modes[0].units, tuple)
    with pytest.raises(ValueError, match="tuple"):
        api().validate_robot_spec(replace(loaded, module_types=list(loaded.module_types)))


@pytest.mark.parametrize("change", ["limit", "units", "geometry", "codec", "payload",
                                    "capability", "catalog", "category_order"])
def test_fingerprint_tracks_semantic_changes(change):
    api()
    spec = fixtures.generic_surface_spec()
    module, probe = spec.module_types
    actions = spec.actions
    if change == "geometry":
        connector = replace(module.connectors[0], geometry=(c.NumericFeature("radius_m", (0.08,)),))
        module = replace(module, connectors=(connector,))
    elif change == "capability": module = replace(module, capabilities=("new-capability",))
    elif change in {"units", "codec", "payload", "limit"}:
        resource = module.resources[0]
        mode = resource.modes[0]
        if change == "units": mode = replace(mode, units=("deg/s",))
        if change == "codec": mode = replace(mode, codec="continuous")
        if change == "payload": mode = replace(mode, payload_size=2, units=("rad/s", "rad/s"))
        resource = replace(resource, modes=(mode,))
        if change == "limit":
            resource = replace(resource, coordinates=(replace(resource.coordinates[0], limits=(-1, 1)),))
        module = replace(module, resources=(resource,))
    elif change == "catalog": actions = actions + (c.ActionSpec("extra", "skill", "module"),)
    elif change == "category_order":
        args = tuple(replace(p, categories=tuple(reversed(p.categories)))
                     if p.kind == "category" else p for p in actions[0].parameters)
        actions = (replace(actions[0], parameters=args),) + actions[1:]
    changed = replace(spec, module_types=(module, probe), actions=actions)
    assert api().robot_spec_fingerprint(changed) != api().robot_spec_fingerprint(spec)


def test_json_schema_documents_the_same_contract():
    api()
    path = Path(__file__).resolve().parents[4] / "configs/bc_v1/robot_spec.schema.json"
    schema = json.loads(path.read_text(encoding="utf-8"))
    assert schema["properties"]["schema_version"]["const"] == "mssr.bc_v1.robot_spec.v1"
    assert schema["additionalProperties"] is False
    assert schema["$defs"]["ActionSpec"]["properties"]["enabled"]["default"] is True
    assert set(schema["$defs"]["ParameterSpec"]["properties"]["codec"]["enum"]) == {
        None, "continuous", "categorical", "module_ref", "connector_ref", "reference_set",
    }
    assert fixtures.schema_accepts(fixtures.robot_spec_document(), schema)
    assert fixtures.schema_accepts(fixtures.robot_spec_document(fixtures.generic_surface_spec()), schema)


@pytest.mark.parametrize("fault", ["unknown_field", "unknown_codec", "scalar_dimension",
                                   "category_units", "bad_ref_set", "empty_surface",
                                   "missing_owner", "fractional_payload", "bool_payload"])
def test_json_schema_and_loader_agree_on_local_constraints(tmp_path, fault):
    path = Path(__file__).resolve().parents[4] / "configs/bc_v1/robot_spec.schema.json"
    schema = json.loads(path.read_text(encoding="utf-8"))
    data = fixtures.robot_spec_document(fixtures.generic_surface_spec())
    parameter = data["actions"][0]["parameters"][0]
    if fault == "unknown_field": data["extra"] = 1
    elif fault == "unknown_codec": parameter["codec"] = "plugin"
    elif fault == "scalar_dimension": data["actions"][1]["parameters"][0]["size"] = 2
    elif fault == "category_units": data["actions"][0]["parameters"][3]["units"] = "m"
    elif fault == "bad_ref_set": data["actions"][2]["parameters"][1]["codec"] = "module_ref"
    elif fault == "empty_surface": data["module_types"][0]["connectors"][0]["geometry"] = []
    elif fault == "missing_owner": data["actions"][0]["parameters"][1]["module_type_id"] = None
    elif fault == "fractional_payload": data["module_types"][1]["resources"][0]["modes"][0]["payload_size"] = 3.5
    elif fault == "bool_payload": data["module_types"][1]["resources"][0]["modes"][0]["payload_size"] = True
    assert not fixtures.schema_accepts(data, schema)
    with pytest.raises(ValueError):
        load_document(tmp_path, data)


def test_categorical_mode_and_all_argument_codecs_roundtrip(tmp_path):
    api()
    data = fixtures.robot_spec_document(fixtures.generic_surface_spec())
    resource = data["module_types"][0]["resources"][0]
    resource["modes"].append({
        "mode_id": "latch", "payload_size": 1, "units": ["1"],
        "kind": "discrete", "codec": "categorical", "categories": ["off", "on"],
    })
    args = data["actions"][0]["parameters"]
    for argument, codec in zip(args, ["module_ref", "connector_ref", "continuous", "categorical"]):
        argument["codec"] = codec
    args.append({"name": "contacts", "kind": "connector_refs", "codec": "reference_set",
                 "module_type_id": "unit", "connector_id": "shell"})
    data["module_types"][1]["resources"][0]["modes"][0]["payload_size"] = 3.0
    loaded = load_document(tmp_path, data)
    assert any(m.codec == "categorical" for t in loaded.module_types for r in t.resources for m in r.modes)
    assert type(next(t for t in loaded.module_types if t.type_id == "probe").resources[0].modes[0].payload_size) is int


@pytest.mark.parametrize("field,value", [("codec", "continuous"), ("kind", "continuous"),
                                         ("categories", []), ("units", ["rad"])])
def test_categorical_resource_mode_rejects_inconsistent_definition(tmp_path, field, value):
    api()
    data = fixtures.robot_spec_document()
    mode = {"mode_id": "latch", "payload_size": 1, "units": ["1"],
            "kind": "discrete", "codec": "categorical", "categories": ["off", "on"]}
    mode[field] = value
    data["module_types"][0]["resources"][0]["modes"] = [mode]
    with pytest.raises(ValueError):
        load_document(tmp_path, data)


def test_fingerprint_preserves_large_finite_integer_differences():
    # A float cast would silently hash these distinct geometry values identically.
    spec = fixtures.robot_spec()
    module = spec.module_types[0]
    a = replace(spec, module_types=(replace(module, geometry=(
        c.NumericFeature("domain_scale", (9007199254740992,)),
    )),))
    b = replace(spec, module_types=(replace(module, geometry=(
        c.NumericFeature("domain_scale", (9007199254740993,)),
    )),))
    assert api().robot_spec_fingerprint(a) != api().robot_spec_fingerprint(b)


def test_fingerprint_handles_validated_json_escaped_unicode(tmp_path):
    # Every accepted spec must be fingerprintable, including JSON escapes.
    path = tmp_path / "escaped.json"
    path.write_text(
        '{"schema_version":"mssr.bc_v1.robot_spec.v1",'
        '"robot_id":"\\ud800","module_types":[{"type_id":"unit"}]}',
        encoding="utf-8",
    )
    spec = api().load_robot_spec(path)
    assert api().validate_robot_spec(spec) is None
    assert re.fullmatch("[0-9a-f]{64}", api().robot_spec_fingerprint(spec))
