"""CPU-only target topology and folding checks for guided C1 evaluation.

Targets belong to the evaluator. Physical validity, temporal stability and
the functional probe remain the episode evaluator's separate requirements.
"""

from dataclasses import dataclass
import math

from mssr_expert.graph.attributed_robot_graph import (
    AttributedRobotGraph,
    GraphEdge,
    GraphNode,
)
from mssr_expert.graph.serialization import attributed_graph_from_dict
from mssr_expert.planning.smores_ep.attributed_adapter import target_graph_to_kinematic_tree
from mssr_expert.planning.smores_ep.self_reconfiguration_planner import (
    SmoresSelfReconfigurationPlanner,
    SelfReconfigurationPlanningError,
)
from mssr_expert.planning.smores_ep.unfolding import normalize_angle

from drl.MARL.c1_structure import C1Frame


@dataclass(frozen=True)
class GuidedAssemblyStatus:
    target_graph_complete: bool
    fold_complete: bool
    assignment: dict[str, str]  # Logical target vertex -> physical owner.
    reason: str


class _TopologyAssignments(SmoresSelfReconfigurationPlanner):
    """Retain the full matches already visited by configuration_assignment.

    Each occupied connector has one peer, so a fixed target/physical root
    determines at most one full face-preserving tree mapping. The expert
    matcher visits every physical root before selecting its canonical map.
    Recording those results avoids a second matching algorithm and permits
    folding to select another globally consistent map when needed.
    """

    def __init__(self):
        super().__init__()
        self.assignments = []

    def _common_subtree_match(self, current_edges, target_tree, target_root,
                              physical_root, pair_cost):
        match = super()._common_subtree_match(
            current_edges, target_tree, target_root, physical_root, pair_cost,
        )
        if (match.edge_count == len(target_tree.edges)
                and len(match.target_to_module) == len(target_tree.vertex_ids)):
            self.assignments.append(dict(match.target_to_module))
        return match


class GuidedAssemblyGate:
    """Require the exact target graph and its measured PAN/TILT posture.

    Cache matching by port-labelled topology; recompute folding from current
    measured joints on every call. Only joints explicitly targeted by the
    existing post-assembly attributes are required here.
    """

    def __init__(self, target_graph, owner_ids):
        if not isinstance(target_graph, AttributedRobotGraph):
            target_graph = attributed_graph_from_dict(target_graph.get("target", target_graph))
        self._target = target_graph
        tree = target_graph_to_kinematic_tree(target_graph)
        self._owners = tuple(owner_ids)
        if (len(self._owners) != len(tree.vertex_ids)
                or len(set(self._owners)) != len(self._owners)):
            raise ValueError("Guided target and unique physical owners must have equal counts")
        self._edge_count = len(tree.edges)
        self._matcher = _TopologyAssignments()
        identity = {vertex: vertex for vertex in tree.vertex_ids}
        # These expert helpers validate target vertices, finite angles and
        # the physical TILT range exactly as the existing fold planner does.
        self._targets = {
            "tilt": self._matcher._tilt_targets_from_graph(
                target_graph, "post_assembly_tilt_rad_by_vertex", identity,
            ),
            "pan": self._matcher._pan_targets_from_graph(
                target_graph, "post_assembly_pan_rad_by_vertex", identity,
            ),
        }
        # Defaults match the existing SET_TILT/SET_PAN primitive tolerances.
        self._tolerances = {}
        for joint, default in (("tilt", math.radians(3)), ("pan", math.radians(2))):
            value = float(target_graph.global_attributes.get(
                f"post_assembly_{joint}_tolerance_rad", default,
            ))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"Invalid post-assembly {joint} tolerance")
            self._tolerances[joint] = value
        self._last_signature = None
        self._assignments = ()

    def assess(self, frame: C1Frame) -> GuidedAssemblyStatus:
        owners = tuple(module.module_id for module in frame.modules)
        if len(owners) != len(self._owners) or set(owners) != set(self._owners):
            return GuidedAssemblyStatus(False, False, {}, "missing_or_duplicate_module")
        if (any(not connection.joint_valid for connection in frame.connections)
                or len({connection.joint_id for connection in frame.connections}) != len(frame.connections)):
            return GuidedAssemblyStatus(False, False, {}, "invalid_docking_constraint")
        if len(frame.connections) != self._edge_count:
            return GuidedAssemblyStatus(False, False, {}, "target_edge_count_mismatch")
        signature = tuple(sorted(tuple(sorted(connection.endpoints)) for connection in frame.connections))
        if signature != self._last_signature:
            self._assignments = self._match(frame)
            self._last_signature = signature
        if not self._assignments:
            return GuidedAssemblyStatus(False, False, {}, "target_topology_mismatch")
        modules = {module.module_id: module for module in frame.modules}
        reasons = []
        for assignment in self._assignments:
            reason = self._fold_reason(modules, assignment)
            if reason == "complete":
                return GuidedAssemblyStatus(True, True, dict(assignment), reason)
            reasons.append(reason)
        reason = "missing_fold_measurement" if "missing_fold_measurement" in reasons else "fold_target_not_reached"
        return GuidedAssemblyStatus(True, False, dict(self._assignments[0]), reason)

    def _match(self, frame):
        graph = AttributedRobotGraph(
            nodes=tuple(GraphNode(module.module_id) for module in frame.modules),
            edges=tuple(GraphEdge(connection.module_a, connection.module_b, {
                "relation_type": "current_connection", "is_attached": True,
                "face_a": connection.connector_a, "face_b": connection.connector_b,
            }) for connection in frame.connections),
        )
        self._matcher.assignments = []
        try:
            canonical = self._matcher.configuration_assignment(graph, self._target)
        except (SelfReconfigurationPlanningError, ValueError):
            return ()
        if canonical is None:
            return ()
        return tuple(sorted(self._matcher.assignments, key=lambda mapping: tuple(sorted(mapping.items()))))

    def _fold_reason(self, modules, assignment):
        for joint, targets in self._targets.items():
            for vertex, target in targets.items():
                module = modules[assignment[vertex]]
                measured = dict(module.joints)
                value = measured.get(f"{joint}.position_rad")
                if (len(measured) != len(module.joints) or value is None
                        or not math.isfinite(float(value))):
                    return "missing_fold_measurement"
                error = float(value) - target
                if joint == "pan":
                    error = normalize_angle(error)
                if abs(error) > self._tolerances[joint]:
                    return "fold_target_not_reached"
        return "complete"
