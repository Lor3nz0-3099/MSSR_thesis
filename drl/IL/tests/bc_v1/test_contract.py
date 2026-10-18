"""A1 boundary tests: isolation, entity cardinality and label separation."""

import importlib
import subprocess
import sys
from dataclasses import FrozenInstanceError, fields
from pathlib import Path

import pytest


def contract():
    # Keep an absent implementation an explicit RED assertion, not a collection error.
    try:
        return importlib.import_module("drl.IL.bc_v1.bc_v1_contract")
    except ModuleNotFoundError as error:
        pytest.fail(f"The approved A1 contract is missing: {error}")


def fixtures():
    return importlib.import_module("drl.IL.tests.bc_v1.fixtures")


def test_contract_imports_without_hardware_or_tensor_packages():
    # Catches a dependency accidentally imported by either public entry point.
    script = """
import importlib.abc
import sys
blocked = {'isaacsim', 'isaaclab', 'rclpy', 'mssr_expert', 'smores_ep',
           'freebot_docking', 'torch', 'numpy'}
class BlockHardware(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in blocked:
            raise ImportError('Forbidden contract dependency: ' + fullname)
sys.meta_path.insert(0, BlockHardware())
import drl.IL.bc_v1
import drl.IL.bc_v1.bc_v1_contract
assert not blocked.intersection(sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[4], capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


def test_policy_sample_separates_inputs_targets_and_audit():
    # Catches accidental label/provenance fields in the model input boundary.
    c = contract()
    inputs = c.PolicyInputs(
        graph=fixtures().state_graph(2, 1), observation=c.ObservationBundle(),
        context=c.TaskContext(task_id="gap"),
    )
    sample = c.PolicySample(inputs, c.SupervisionBundle(), c.SampleAudit("ep-1", 7))
    assert {f.name for f in fields(sample)} == {"inputs", "supervision", "audit"}
    assert {f.name for f in fields(inputs)} == {"graph", "observation", "context"}
    with pytest.raises(FrozenInstanceError):
        sample.inputs = None
    with pytest.raises(FrozenInstanceError):
        inputs.context = None
    with pytest.raises(FrozenInstanceError):
        inputs.graph.modules[0].position = (9.0, 0.0, 0.0)


@pytest.mark.parametrize("module_count", [2, 5, 8, 12])
@pytest.mark.parametrize("connector_count", [0, 1, 4, 6])
def test_state_supports_variable_entity_counts(module_count, connector_count):
    # Catches truncation/padding or fixed hardware inventory in the contract.
    contract()
    graph = fixtures().state_graph(module_count, connector_count)
    assert len(graph.modules) == module_count
    assert len(graph.connectors) == module_count * connector_count
    assert len(graph.resources) == module_count
    assert graph.resources[-1].ref.module == graph.modules[-1].ref
    if connector_count:
        assert graph.connectors[-1].ref.module == graph.modules[-1].ref


def test_surface_connector_and_connection_preserve_generic_endpoints():
    # Catches reducing connector endpoints to a fixed categorical face index.
    c = contract()
    graph = fixtures().state_graph(2, 0, surface=True)
    assert tuple(x.kind for x in graph.connectors) == ("surface", "surface")
    connection = c.ConnectionState(
        graph.connectors[0].ref, graph.connectors[1].ref,
        relation_id="rolling_contact", allowed_relative_dofs=("roll", "tangent"),
    )
    assert connection.endpoint_a.module == c.ModuleRef("unit-0")
    assert connection.endpoint_b.connector_id == "shell"
    assert connection.allowed_relative_dofs == ("roll", "tangent")
    spec = fixtures().robot_spec(0, surface=True)
    assert len(spec.module_types[0].connectors) == 1
    assert spec.module_types[0].connectors[0].kind == "surface"


def test_configured_neutral_recovery_is_enabled_and_needs_no_connector():
    # Catches disabling a rare configured skill or requiring connector arguments.
    c = contract()
    action = fixtures().robot_spec(0).actions[1]
    assert action.enabled
    assert action.scope == "modules"
    assert tuple(p.kind for p in action.parameters) == (
        "scalar", "module_refs", "boolean",
    )
    initiator, peer = c.ModuleRef("unit-0"), c.ModuleRef("unit-1")
    target = c.StructuralTarget(
        initiator=initiator, operation_id=action.action_id,
        participants=(initiator, peer),
        parameters=(
            c.ParameterValue("duration", 1.5),
            c.ParameterValue("passive_modules", (initiator, peer)),
            c.ParameterValue("clear_on_finish", True),
        ),
    )
    assert target.connectors == ()
    assert target.supervised
    assert target.parameters[1].value == (initiator, peer)


def test_joint_targets_preserve_independent_initiators_and_concurrent_resource_loss():
    # Catches losing one parallel action or replacing a resource label with it.
    c = contract()
    a, b, anchor = (c.ModuleRef(x) for x in ("a", "b", "anchor"))
    targets = (
        c.StructuralTarget(a, "connect", participants=(a, anchor),
                           connectors=(c.ConnectorRef(a, "shell"),
                                       c.ConnectorRef(anchor, "port-a"))),
        c.StructuralTarget(b, "connect", participants=(b, anchor),
                           connectors=(c.ConnectorRef(b, "shell"),
                                       c.ConnectorRef(anchor, "port-b"))),
    )
    rate = c.ResourceTarget(c.ResourceRef(a, "drive"), "rate", (0.2, -0.1))
    internal = c.ResourceTarget(c.ResourceRef(a, "joint"), "position", (0.3,))
    labels = c.SupervisionBundle(
        resource_targets=(rate, internal), structural_targets=targets,
    )
    assert tuple(t.initiator for t in labels.structural_targets) == (a, b)
    assert tuple(t.resource.resource_id for t in labels.resource_targets) == (
        "drive", "joint",
    )
    assert all(t.supervised for t in labels.resource_targets)
    assert len(labels.structural_targets) == 2
    assert len(c.SupervisionBundle().structural_targets) == 0


def test_missing_task_and_stage_stay_unknown_without_target_or_provenance():
    # Catches filling missing context from action labels or future task stages.
    c = contract()
    context = c.TaskContext()
    assert context.task_id is None
    assert not hasattr(context, "stage_id")
    assert context.static_geometry == ()
    inputs = c.PolicyInputs(c.StateGraph(), c.ObservationBundle(), context)
    evidence = c.LabelEvidence("source.jsonl", 12, "fingerprint", "goal-7")
    sample = c.PolicySample(
        inputs, c.SupervisionBundle(label_evidence=(evidence,)),
        c.SampleAudit("episode", 2, physical_stamp=1.2, provenance="teleop"),
    )
    assert sample.inputs.context.task_id is None
    assert sample.supervision.label_evidence[0].goal_id == "goal-7"
    assert not hasattr(inputs, "label_evidence")
    assert not hasattr(inputs, "audit")


def test_robot_spec_keeps_resource_modes_and_payload_dimensions_configured():
    # Catches assuming every resource has one coordinate or a scalar payload.
    c = contract()
    drive = c.ResourceSpec(
        "drive", coordinates=(c.CoordinateSpec("roll", "rad"),),
        modes=(c.ResourceModeSpec("twist", payload_size=3,
                                  units=("m/s", "m/s", "rad/s")),),
        capabilities=("omnidirectional",),
    )
    spec = c.RobotSpec(
        "mssr.bc_v1.robot_spec.v1", "synthetic-ball",
        module_types=(c.ModuleTypeSpec("ball", resources=(drive,)),),
    )
    assert spec.module_types[0].resources[0].modes[0].payload_size == 3
    assert spec.module_types[0].connectors == ()



def test_task_context_cannot_carry_configuration_destination():
    from dataclasses import asdict
    c = contract()
    assert set(asdict(c.TaskContext())) == {
        "task_id", "static_geometry"}
    with pytest.raises(TypeError, match="configuration_goal"):
        c.TaskContext(configuration_goal="snake8")


def test_policy_context_rejects_stage_and_future_graph_arguments():
    # A stage field would expose the structural decision as a model input.
    c = contract()
    for name in ("stage_id", "stage", "stage_geometry", "configuration_goal"):
        with pytest.raises(TypeError):
            c.TaskContext(**{name: "reconfiguration"})
    inputs = c.PolicyInputs(c.StateGraph(), c.ObservationBundle(), c.TaskContext(task_id="gap"))
    assert set(inputs.context.__dataclass_fields__) == {"task_id", "static_geometry"}
    with pytest.raises(TypeError):
        c.PolicyInputs(inputs.graph, inputs.observation, inputs.context, graph_t_plus_1=c.StateGraph())
