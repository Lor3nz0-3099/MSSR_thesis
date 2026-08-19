"""Deterministic topology-preserving self-reconfiguration for SMORES-EP."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
import math
from typing import Mapping

from mssr_expert.graph.attributed_robot_graph import (
    AttributedRobotGraph,
    GraphEdge,
)
from mssr_expert.execution.parallel_assembly_executor import (
    physical_fold_push_pairs,
    physical_posture_groups,
)
from mssr_expert.planning.smores_ep.assembly_sequence import (
    AssemblyAction,
    AssemblyWave,
    ParallelAssemblyPlan,
    generate_parallel_assembly_plan,
)
from mssr_expert.planning.smores_ep.assignment import (
    AssignmentResult,
    congestion_aware_pair_costs,
    future_blocker_counts_by_pair,
    solve_rectangular_assignment,
    solve_linear_assignment,
)
from mssr_expert.planning.smores_ep.attributed_adapter import (
    target_graph_to_kinematic_tree,
)
from mssr_expert.planning.smores_ep.parallel_self_assembly_planner import (
    ParallelSelfAssemblyPlanner,
)
from mssr_expert.planning.smores_ep.rooting import root_kinematic_tree
from mssr_expert.planning.smores_ep.topology import (
    SmoresKinematicTree,
    SmoresTopologyEdge,
)
from mssr_expert.planning.smores_ep.unfolding import (
    PlanarPose,
    UnfoldedPlanarConfiguration,
    unfold_tree_on_plane,
)


class SelfReconfigurationPlanningError(ValueError):
    """Raised when the connected robot cannot be reconfigured safely."""


@dataclass(frozen=True)
class ReconfigurationDetachAction:
    """One existing connection that must be released."""

    module_a_id: str
    face_a: str
    module_b_id: str
    face_b: str


@dataclass(frozen=True)
class _CommonSubtreeMatch:
    """One rooted face-preserving common-subtree embedding."""

    target_to_module: Mapping[str, str]
    retained_edges: tuple[SmoresTopologyEdge, ...]
    edge_count: int
    motion_cost: float


@dataclass(frozen=True)
class SelfReconfigurationStage:
    """Release and relocate one independent wave of source leaves."""

    mobile_module_ids: tuple[str, ...]
    source_depth: int
    detach_actions: tuple[ReconfigurationDetachAction, ...]
    assembly_plan: ParallelAssemblyPlan


@dataclass(frozen=True)
class SelfReconfigurationPlan:
    """Topology delta between one connected configuration and a target."""

    source_graph: AttributedRobotGraph | None
    source_morphology: str
    target_morphology: str
    target_graph: AttributedRobotGraph
    target_tree: SmoresKinematicTree
    assignment: AssignmentResult
    retained_target_edges: tuple[SmoresTopologyEdge, ...]
    retained_module_ids: tuple[str, ...]
    prepare_tilt_by_module: Mapping[str, float]
    prepare_tilt_groups_by_module: tuple[tuple[str, ...], ...]
    prepare_stabilize_module_ids: tuple[str, ...]
    final_tilt_by_module: Mapping[str, float]
    final_pan_by_module: Mapping[str, float]
    coordinate_final_tilts: bool
    final_tilt_groups_by_module: tuple[tuple[str, ...], ...]
    final_push_by_lifter_module: Mapping[str, tuple[str, float]]
    reserve_module_ids: tuple[str, ...]
    reserve_detach_actions: tuple[ReconfigurationDetachAction, ...]
    detach_actions: tuple[ReconfigurationDetachAction, ...]
    assembly_plan: ParallelAssemblyPlan
    stages: tuple[SelfReconfigurationStage, ...]

    @property
    def retained_connection_count(self) -> int:
        return len(self.retained_target_edges)

    @property
    def new_connection_count(self) -> int:
        return self.assembly_plan.action_count


class SmoresSelfReconfigurationPlanner:
    """Keep the largest root-connected common subconfiguration.

    Moving source branches are peeled from the leaves toward the retained
    component.  Independent leaves with separated motion corridors form one
    parallel wave; that complete wave is docked before the next source edges
    are removed.  Compact morphologies therefore retain safe free-space
    corridors without forcing independent symmetric motions to be serial.
    """

    def __init__(
        self,
        parallel_path_clearance_m: float = 0.12,
        max_parallel_actions: int = 0,
        assignment_staging_distance_m: float = 0.070,
        assignment_corridor_clearance_m: float = 0.110,
    ) -> None:
        if (
            not math.isfinite(parallel_path_clearance_m)
            or parallel_path_clearance_m <= 0.0
        ):
            raise SelfReconfigurationPlanningError(
                "parallel_path_clearance_m must be positive and finite."
            )
        if (
            not isinstance(max_parallel_actions, int)
            or isinstance(max_parallel_actions, bool)
            or max_parallel_actions < 0
        ):
            raise SelfReconfigurationPlanningError(
                "max_parallel_actions must be a non-negative integer."
            )
        self.parallel_path_clearance_m = parallel_path_clearance_m
        self.max_parallel_actions = max_parallel_actions
        if (
            not math.isfinite(assignment_staging_distance_m)
            or assignment_staging_distance_m <= 0.0
        ):
            raise SelfReconfigurationPlanningError(
                "assignment_staging_distance_m must be positive and finite."
            )
        if (
            not math.isfinite(assignment_corridor_clearance_m)
            or assignment_corridor_clearance_m <= 0.0
        ):
            raise SelfReconfigurationPlanningError(
                "assignment_corridor_clearance_m must be positive and "
                "finite."
            )
        self.assignment_staging_distance_m = assignment_staging_distance_m
        self.assignment_corridor_clearance_m = (
            assignment_corridor_clearance_m
        )

    def plan(
        self,
        current_graph: AttributedRobotGraph,
        target_graph: AttributedRobotGraph,
        source_graph: AttributedRobotGraph | None = None,
        source_assignment: AssignmentResult | None = None,
    ) -> SelfReconfigurationPlan:
        physical_module_ids = self._physical_module_ids(current_graph)
        current_edges = self._attached_edges(current_graph)
        self._validate_current_tree(physical_module_ids, current_edges)

        target_tree = target_graph_to_kinematic_tree(target_graph)
        if len(physical_module_ids) < len(target_tree.vertex_ids):
            raise SelfReconfigurationPlanningError(
                "The target needs more modules than are present in the "
                "current state. Start the episode with the missing modules "
                "as disconnected reserves."
            )

        root_id = ParallelSelfAssemblyPlanner._declared_target_root(
            target_graph
        )
        rooted_target = root_kinematic_tree(target_tree, root_id=root_id)
        physical_poses = ParallelSelfAssemblyPlanner(
            require_disconnected_modules=False,
        )._extract_physical_poses(current_graph)
        unfolded_target = unfold_tree_on_plane(rooted_target)
        degree_by_module = {module_id: 0 for module_id in physical_module_ids}
        for edge in current_edges:
            degree_by_module[edge.module_a_id] += 1
            degree_by_module[edge.module_b_id] += 1
        reserve_count = len(physical_module_ids) - len(target_tree.vertex_ids)
        assignment, retained_edges, reserve_module_ids = (
            self._leaf_safe_assignment(
                physical_module_ids,
                current_edges,
                target_tree,
                rooted_target.root_id,
                physical_poses,
                unfolded_target,
                degree_by_module,
                reserve_count,
            )
        )
        assigned_modules = set(assignment.target_to_module.values())
        reserve_edges = tuple(
            edge
            for edge in current_edges
            if edge.module_a_id in reserve_module_ids
            or edge.module_b_id in reserve_module_ids
        )
        reserve_detach_actions = tuple(
            self._detach_action(edge)
            for edge in sorted(reserve_edges, key=self._physical_edge_key)
        )
        active_module_ids = tuple(sorted(assigned_modules))
        active_current_edges = tuple(
            edge
            for edge in current_edges
            if edge.module_a_id in assigned_modules
            and edge.module_b_id in assigned_modules
        )

        retained_keys = {
            self._mapped_target_key(edge, assignment.target_to_module)
            for edge in retained_edges
        }
        retained_module_ids = tuple(
            sorted(
                {
                    assignment.target_to_module[edge.vertex_a]
                    for edge in retained_edges
                }
                | {
                    assignment.target_to_module[edge.vertex_b]
                    for edge in retained_edges
                }
                | {assignment.target_to_module[rooted_target.root_id]}
            )
        )

        detach_edges = tuple(
            edge
            for edge in active_current_edges
            if self._physical_edge_key(edge) not in retained_keys
        )
        self._validate_isolated_movers(
            active_module_ids,
            active_current_edges,
            retained_keys,
            set(retained_module_ids),
        )

        detach_actions = tuple(
            self._detach_action(edge)
            for edge in sorted(
                detach_edges,
                key=lambda item: self._physical_edge_key(item),
            )
        )

        full_assembly = generate_parallel_assembly_plan(
            rooted_target,
            assignment,
        )
        retained_target_keys = {
            self._target_edge_key(edge) for edge in retained_edges
        }
        reduced_actions = tuple(
            action
            for wave in full_assembly.waves
            for action in wave.actions
            if self._action_target_key(action) not in retained_target_keys
        )
        target_xy_by_module = self._target_xy_by_module(
            assignment.target_to_module,
            rooted_target.root_id,
            physical_poses,
            unfolded_target,
        )
        action_waves, source_depth, detach_by_module = (
            self._progressive_action_waves(
                active_module_ids,
                active_current_edges,
                set(retained_module_ids),
                detach_actions,
                reduced_actions,
                physical_poses,
                target_xy_by_module,
            )
        )
        waves = tuple(
            AssemblyWave(
                wave_index=index,
                depth=max(action.depth for action in action_wave),
                phase="PROGRESSIVE_RECONFIGURATION",
                actions=action_wave,
            )
            for index, action_wave in enumerate(action_waves)
        )
        assembly_plan = ParallelAssemblyPlan(
            root_target_vertex=full_assembly.root_target_vertex,
            root_module_id=full_assembly.root_module_id,
            waves=waves,
        )
        stages = tuple(
            SelfReconfigurationStage(
                mobile_module_ids=tuple(
                    action.mobile_module_id for action in action_wave
                ),
                source_depth=max(
                    source_depth[action.mobile_module_id]
                    for action in action_wave
                ),
                detach_actions=tuple(
                    detach_by_module[action.mobile_module_id]
                    for action in action_wave
                    if action.mobile_module_id in detach_by_module
                ),
                assembly_plan=ParallelAssemblyPlan(
                    root_target_vertex=full_assembly.root_target_vertex,
                    root_module_id=full_assembly.root_module_id,
                    waves=(
                        AssemblyWave(
                            wave_index=0,
                            depth=max(
                                action.depth for action in action_wave
                            ),
                            phase="PROGRESSIVE_RECONFIGURATION",
                            actions=action_wave,
                        ),
                    ),
                ),
            )
            for action_wave in action_waves
        )

        if source_graph is not None:
            if source_assignment is None:
                source_assignment = self.configuration_assignment(
                    current_graph,
                    source_graph,
                )
            if source_assignment is None:
                raise SelfReconfigurationPlanningError(
                    "The current graph does not match the declared source "
                    "morphology."
                )
            prepare_tilt_by_module = self._tilt_targets_from_graph(
                source_graph,
                "pre_reconfiguration_tilt_rad_by_vertex",
                source_assignment.target_to_module,
            )
        else:
            detached_modules = (
                set(physical_module_ids) - set(retained_module_ids)
            )
            prepare_tilt_by_module = {
                module_id: 0.0 for module_id in sorted(detached_modules)
            }

        prepare_stabilize_module_ids = self._module_ids_from_vertices(
            source_graph,
            "pre_reconfiguration_stabilize_vertices",
            source_assignment.target_to_module if source_assignment else {},
        )
        prepare_tilt_groups_by_module = physical_posture_groups(
            (
                source_graph.global_attributes.get(
                    "pre_reconfiguration_tilt_groups_by_vertex"
                )
                if source_graph is not None
                else None
            ),
            source_assignment.target_to_module if source_assignment else {},
            set(prepare_tilt_by_module),
            field_name="pre_reconfiguration_tilt_groups_by_vertex",
        )

        final_tilt_by_module = self._tilt_targets_from_graph(
            target_graph,
            "post_assembly_tilt_rad_by_vertex",
            assignment.target_to_module,
        )
        final_pan_by_module = self._pan_targets_from_graph(
            target_graph,
            "post_assembly_pan_rad_by_vertex",
            assignment.target_to_module,
        )
        final_tilt_groups_by_module = physical_posture_groups(
            target_graph.global_attributes.get(
                "post_assembly_tilt_groups_by_vertex"
            ),
            assignment.target_to_module,
            set(final_tilt_by_module),
        )
        final_push_by_lifter_module = physical_fold_push_pairs(
            target_graph.global_attributes.get(
                "post_assembly_push_pairs_by_vertex"
            ),
            assignment.target_to_module,
            set(final_tilt_by_module),
        )

        return SelfReconfigurationPlan(
            source_graph=source_graph,
            source_morphology=self._morphology_name(
                source_graph,
                "unknown",
            ),
            target_morphology=self._morphology_name(
                target_graph,
                "target",
            ),
            target_graph=target_graph,
            target_tree=target_tree,
            assignment=assignment,
            retained_target_edges=retained_edges,
            retained_module_ids=retained_module_ids,
            prepare_tilt_by_module=prepare_tilt_by_module,
            prepare_tilt_groups_by_module=(
                prepare_tilt_groups_by_module
            ),
            prepare_stabilize_module_ids=prepare_stabilize_module_ids,
            final_tilt_by_module=final_tilt_by_module,
            final_pan_by_module=final_pan_by_module,
            coordinate_final_tilts=bool(
                target_graph.global_attributes.get(
                    "coordinate_post_assembly_tilts",
                    False,
                )
            ),
            final_tilt_groups_by_module=final_tilt_groups_by_module,
            final_push_by_lifter_module=final_push_by_lifter_module,
            reserve_module_ids=reserve_module_ids,
            reserve_detach_actions=reserve_detach_actions,
            detach_actions=reserve_detach_actions + detach_actions,
            assembly_plan=assembly_plan,
            stages=stages,
        )

    def _leaf_safe_assignment(
        self,
        physical_module_ids: tuple[str, ...],
        current_edges: tuple[GraphEdge, ...],
        target_tree: SmoresKinematicTree,
        target_root: str,
        physical_poses: Mapping[str, PlanarPose],
        unfolded_target: UnfoldedPlanarConfiguration,
        degree_by_module: Mapping[str, int],
        reserve_count: int,
    ) -> tuple[
        AssignmentResult,
        tuple[SmoresTopologyEdge, ...],
        tuple[str, ...],
    ]:
        """Choose parked modules before assignment, restricted to leaves.

        Solving the rectangular assignment first can accidentally omit an
        internal source module even when a perfectly valid pair of leaves is
        available.  Enumerating the small set of leaf reserve combinations
        makes the physical constraint part of the optimization rather than a
        warning after the fact.
        """

        if reserve_count <= 0:
            assignment, retained = self._maximum_common_assignment(
                physical_module_ids,
                current_edges,
                target_tree,
                target_root,
                physical_poses,
                unfolded_target,
            )
            return assignment, retained, ()
        leaf_ids = tuple(
            sorted(
                module_id
                for module_id in physical_module_ids
                if degree_by_module[module_id] <= 1
            )
        )
        if len(leaf_ids) < reserve_count:
            raise SelfReconfigurationPlanningError(
                "Count-reducing reconfiguration needs "
                f"{reserve_count} source leaves, but only {len(leaf_ids)} "
                "are available."
            )
        best: tuple[
            tuple[int, int, int, float, tuple[str, ...], tuple[str, ...]],
            AssignmentResult,
            tuple[SmoresTopologyEdge, ...],
            tuple[str, ...],
        ] | None = None
        all_ids = set(physical_module_ids)
        for raw_reserves in combinations(leaf_ids, reserve_count):
            reserves = tuple(sorted(raw_reserves))
            active = tuple(sorted(all_ids - set(reserves)))
            active_set = set(active)
            active_edges = tuple(
                edge
                for edge in current_edges
                if edge.module_a_id in active_set
                and edge.module_b_id in active_set
            )
            active_poses = {
                module_id: physical_poses[module_id]
                for module_id in active
            }
            assignment, retained = self._maximum_common_assignment(
                active,
                active_edges,
                target_tree,
                target_root,
                active_poses,
                unfolded_target,
            )
            if not self._has_progressive_plan(
                active,
                active_edges,
                target_tree,
                target_root,
                assignment,
                retained,
                active_poses,
                unfolded_target,
            ):
                continue
            root_degree = sum(
                edge.vertex_a == target_root or edge.vertex_b == target_root
                for edge in retained
            )
            assignment_tuple = tuple(
                assignment.target_to_module[target_id]
                for target_id in sorted(target_tree.vertex_ids)
            )
            score = (
                -len(retained),
                -root_degree,
                assignment.total_future_blockers,
                assignment.total_cost,
                assignment_tuple,
                reserves,
            )
            candidate = (score, assignment, retained, reserves)
            if best is None or score < best[0]:
                best = candidate
        if best is None:
            raise SelfReconfigurationPlanningError(
                "No leaf-safe count-reducing assignment has an acyclic "
                "progressive detach-and-assembly sequence."
            )
        return best[1], best[2], best[3]

    def _has_progressive_plan(
        self,
        active_module_ids: tuple[str, ...],
        active_current_edges: tuple[GraphEdge, ...],
        target_tree: SmoresKinematicTree,
        target_root: str,
        assignment: AssignmentResult,
        retained_edges: tuple[SmoresTopologyEdge, ...],
        physical_poses: Mapping[str, PlanarPose],
        unfolded_target: UnfoldedPlanarConfiguration,
    ) -> bool:
        """Reject reserve choices whose remaining tree cannot be peeled."""

        retained_keys = {
            self._mapped_target_key(edge, assignment.target_to_module)
            for edge in retained_edges
        }
        retained_module_ids = {
            assignment.target_to_module[target_root]
        }
        for edge in retained_edges:
            retained_module_ids.add(
                assignment.target_to_module[edge.vertex_a]
            )
            retained_module_ids.add(
                assignment.target_to_module[edge.vertex_b]
            )
        detach_edges = tuple(
            edge
            for edge in active_current_edges
            if self._physical_edge_key(edge) not in retained_keys
        )
        try:
            self._validate_isolated_movers(
                active_module_ids,
                active_current_edges,
                retained_keys,
                retained_module_ids,
            )
            detach_actions = tuple(
                self._detach_action(edge)
                for edge in sorted(
                    detach_edges,
                    key=self._physical_edge_key,
                )
            )
            full_assembly = generate_parallel_assembly_plan(
                root_kinematic_tree(target_tree, root_id=target_root),
                assignment,
            )
            retained_target_keys = {
                self._target_edge_key(edge) for edge in retained_edges
            }
            reduced_actions = tuple(
                action
                for wave in full_assembly.waves
                for action in wave.actions
                if self._action_target_key(action)
                not in retained_target_keys
            )
            target_xy_by_module = self._target_xy_by_module(
                assignment.target_to_module,
                target_root,
                physical_poses,
                unfolded_target,
            )
            self._progressive_action_waves(
                active_module_ids,
                active_current_edges,
                retained_module_ids,
                detach_actions,
                reduced_actions,
                physical_poses,
                target_xy_by_module,
            )
        except SelfReconfigurationPlanningError:
            return False
        return True

    @staticmethod
    def _module_ids_from_vertices(
        graph: AttributedRobotGraph | None,
        attribute_name: str,
        target_to_module: Mapping[str, str],
    ) -> tuple[str, ...]:
        """Resolve optional source-posture module lists by target vertex."""

        if graph is None:
            return ()
        raw_vertices = graph.global_attributes.get(attribute_name, ())
        if not isinstance(raw_vertices, list | tuple):
            raise SelfReconfigurationPlanningError(
                f"{attribute_name} must be an array."
            )
        result: list[str] = []
        for raw_vertex in raw_vertices:
            vertex = str(raw_vertex)
            module_id = target_to_module.get(vertex)
            if module_id is None:
                raise SelfReconfigurationPlanningError(
                    f"{attribute_name} references unknown target vertex "
                    f"{vertex!r}."
                )
            result.append(module_id)
        return tuple(result)

    @staticmethod
    def _morphology_name(
        graph: AttributedRobotGraph | None,
        fallback: str,
    ) -> str:
        if graph is None:
            return fallback
        return str(
            graph.global_attributes.get("morphology_name", fallback)
        )

    @staticmethod
    def _tilt_targets_from_graph(
        graph: AttributedRobotGraph,
        attribute_name: str,
        target_to_module: Mapping[str, str],
    ) -> dict[str, float]:
        raw_targets = graph.global_attributes.get(attribute_name, {})
        if not isinstance(raw_targets, Mapping):
            raise SelfReconfigurationPlanningError(
                f"{attribute_name} must be an object."
            )
        result: dict[str, float] = {}
        for raw_vertex, raw_angle in raw_targets.items():
            vertex = str(raw_vertex)
            module_id = target_to_module.get(vertex)
            if module_id is None:
                raise SelfReconfigurationPlanningError(
                    f"{attribute_name} references unknown target vertex "
                    f"{vertex!r}."
                )
            angle = float(raw_angle)
            if not -1.5707963267948966 <= angle <= 1.5707963267948966:
                raise SelfReconfigurationPlanningError(
                    f"Invalid tilt target {angle} for {vertex}."
                )
            result[module_id] = angle
        return result

    @staticmethod
    def _pan_targets_from_graph(
        graph: AttributedRobotGraph,
        attribute_name: str,
        target_to_module: Mapping[str, str],
    ) -> dict[str, float]:
        raw_targets = graph.global_attributes.get(attribute_name, {})
        if not isinstance(raw_targets, Mapping):
            raise SelfReconfigurationPlanningError(
                f"{attribute_name} must be an object."
            )
        result: dict[str, float] = {}
        for raw_vertex, raw_angle in raw_targets.items():
            vertex = str(raw_vertex)
            module_id = target_to_module.get(vertex)
            if module_id is None:
                raise SelfReconfigurationPlanningError(
                    f"{attribute_name} references unknown target vertex "
                    f"{vertex!r}."
                )
            angle = float(raw_angle)
            if not math.isfinite(angle):
                raise SelfReconfigurationPlanningError(
                    f"Invalid PAN target {angle} for {vertex}."
                )
            result[module_id] = angle
        return result

    def configuration_assignment(
        self,
        current_graph: AttributedRobotGraph,
        morphology_graph: AttributedRobotGraph,
    ) -> AssignmentResult | None:
        """Return an exact face-topology match for a source morphology."""

        module_ids = self._physical_module_ids(current_graph)
        current_edges = self._attached_edges(current_graph)
        self._validate_current_tree(module_ids, current_edges)
        morphology_tree = target_graph_to_kinematic_tree(morphology_graph)
        if len(module_ids) < len(morphology_tree.vertex_ids):
            return None
        target_ids = tuple(sorted(morphology_tree.vertex_ids))
        target_root = (
            ParallelSelfAssemblyPlanner._declared_target_root(
                morphology_graph
            )
            or target_ids[0]
        )
        zero_cost = {
            (target_id, module_id): 0.0
            for target_id in target_ids
            for module_id in module_ids
        }
        best_mapping: dict[str, str] | None = None
        best_tuple: tuple[str, ...] | None = None
        for physical_root in sorted(module_ids):
            match = self._common_subtree_match(
                current_edges,
                morphology_tree,
                target_root,
                physical_root,
                zero_cost,
            )
            if (
                match.edge_count != len(target_ids) - 1
                or len(match.target_to_module) != len(target_ids)
            ):
                continue
            mapping = dict(match.target_to_module)
            mapped_module_ids = set(mapping.values())
            if any(
                edge.module_a_id not in mapped_module_ids
                or edge.module_b_id not in mapped_module_ids
                for edge in current_edges
            ):
                # A smaller catalog morphology is not an exact source match
                # when the surplus modules are still part of its connected
                # component.  Isolated surplus modules remain valid reserves.
                continue
            assignment_tuple = tuple(mapping[item] for item in target_ids)
            if best_tuple is None or assignment_tuple < best_tuple:
                best_tuple = assignment_tuple
                best_mapping = mapping
        if best_mapping is None:
            return None
        return AssignmentResult(
            target_to_module=best_mapping,
            cost_by_target={target_id: 0.0 for target_id in target_ids},
            total_cost=0.0,
        )

    def target_reached(
        self,
        current_graph: AttributedRobotGraph,
        plan: SelfReconfigurationPlan,
    ) -> bool:
        """Check the complete face-attributed target under the fixed map."""

        current_keys = {
            self._physical_edge_key(edge)
            for edge in self._attached_edges(current_graph)
        }
        expected_keys = {
            self._mapped_target_key(
                edge,
                plan.assignment.target_to_module,
            )
            for edge in plan.target_tree.edges
        }
        return current_keys == expected_keys

    def _maximum_common_assignment(
        self,
        physical_module_ids: tuple[str, ...],
        current_edges: tuple[GraphEdge, ...],
        target_tree: SmoresKinematicTree,
        target_root: str,
        physical_poses: Mapping[str, PlanarPose],
        unfolded_target: UnfoldedPlanarConfiguration,
    ) -> tuple[AssignmentResult, tuple[SmoresTopologyEdge, ...]]:
        target_ids = tuple(sorted(target_tree.vertex_ids))
        rooted_target = root_kinematic_tree(
            target_tree,
            root_id=target_root,
        )
        best_score: tuple[int, int] | None = None
        best_motion_cost = math.inf
        best_assignment_tuple: tuple[str, ...] | None = None
        best_physical_root: str | None = None
        best_common_mapping: dict[str, str] = {}
        best_retained: tuple[SmoresTopologyEdge, ...] = ()

        # First select the maximum retained component exactly as before:
        # retained edges, root degree and physical travel remain authoritative.
        for physical_root in sorted(physical_module_ids):
            motion_cost_by_pair = self._motion_cost_by_pair(
                target_root,
                physical_root,
                physical_poses,
                unfolded_target,
            )
            common = self._common_subtree_match(
                current_edges,
                target_tree,
                target_root,
                physical_root,
                motion_cost_by_pair,
            )
            mapping = dict(common.target_to_module)
            remaining_targets = sorted(set(target_ids) - set(mapping))
            remaining_modules = sorted(
                set(physical_module_ids) - set(mapping.values())
            )
            moving_cost_matrix = tuple(
                tuple(
                    motion_cost_by_pair[(target_id, module_id)]
                    for module_id in remaining_modules
                )
                for target_id in remaining_targets
            )
            selected_columns = solve_rectangular_assignment(
                moving_cost_matrix
            )
            for row_index, column_index in enumerate(selected_columns):
                mapping[remaining_targets[row_index]] = remaining_modules[
                    column_index
                ]

            retained = common.retained_edges
            root_degree = sum(
                edge.vertex_a == target_root or edge.vertex_b == target_root
                for edge in retained
            )
            score = (len(retained), root_degree)
            motion_cost = sum(
                motion_cost_by_pair[(target_id, mapping[target_id])]
                for target_id in target_ids
            )
            assignment_tuple = tuple(mapping[item] for item in target_ids)
            if (
                best_score is None
                or score > best_score
                or (
                    score == best_score
                    and (
                        motion_cost < best_motion_cost - 1.0e-12
                        or (
                            abs(motion_cost - best_motion_cost) <= 1.0e-12
                            and (
                                best_assignment_tuple is None
                                or assignment_tuple < best_assignment_tuple
                            )
                        )
                    )
                )
            ):
                best_score = score
                best_motion_cost = motion_cost
                best_assignment_tuple = assignment_tuple
                best_physical_root = physical_root
                best_common_mapping = dict(common.target_to_module)
                best_retained = retained

        if best_physical_root is None:
            raise SelfReconfigurationPlanningError(
                "No complete module-to-target assignment exists."
            )

        # With the retained component fixed, assign only detached/free movers
        # through the congestion-aware Hungarian matrix. Existing connections
        # can therefore never be traded merely to improve a staging corridor.
        motion_cost_by_pair = self._motion_cost_by_pair(
            target_root,
            best_physical_root,
            physical_poses,
            unfolded_target,
        )
        future_blockers_by_pair = future_blocker_counts_by_pair(
            physical_poses=physical_poses,
            physical_root_id=best_physical_root,
            target=unfolded_target,
            target_parent_by_vertex=rooted_target.parent_by_vertex,
            target_depth_by_vertex=rooted_target.depth_by_vertex,
            staging_distance_m=self.assignment_staging_distance_m,
            staging_corridor_clearance_m=(
                self.assignment_corridor_clearance_m
            ),
        )
        selection_cost_by_pair = congestion_aware_pair_costs(
            motion_cost_by_pair=motion_cost_by_pair,
            future_blockers_by_pair=future_blockers_by_pair,
            target_ids=target_ids,
            module_ids=physical_module_ids,
        )
        mapping = dict(best_common_mapping)
        moving_targets = sorted(set(target_ids) - set(mapping))
        moving_modules = sorted(
            set(physical_module_ids) - set(mapping.values())
        )
        selected_columns = solve_rectangular_assignment(
            tuple(
                tuple(
                    selection_cost_by_pair[(target_id, module_id)]
                    for module_id in moving_modules
                )
                for target_id in moving_targets
            )
        )
        for row_index, column_index in enumerate(selected_columns):
            mapping[moving_targets[row_index]] = moving_modules[column_index]

        cost_by_target = {
            target_id: motion_cost_by_pair[
                (target_id, mapping[target_id])
            ]
            for target_id in target_ids
        }
        blockers_by_target = {
            target_id: (
                future_blockers_by_pair.get(
                    (target_id, mapping[target_id]),
                    0,
                )
                if target_id in moving_targets
                else 0
            )
            for target_id in target_ids
        }
        return (
            AssignmentResult(
                target_to_module=mapping,
                cost_by_target=cost_by_target,
                total_cost=sum(cost_by_target.values()),
                future_blockers_by_target=blockers_by_target,
                total_future_blockers=sum(blockers_by_target.values()),
            ),
            tuple(sorted(best_retained, key=self._target_edge_key)),
        )

    def _common_subtree_match(
        self,
        current_edges: tuple[GraphEdge, ...],
        target_tree: SmoresKinematicTree,
        target_root: str,
        physical_root: str,
        pair_cost: Mapping[tuple[str, str], float],
    ) -> _CommonSubtreeMatch:
        """Match rooted face-labelled trees without enumerating ``n!`` maps."""

        target_adjacency: dict[
            str,
            list[tuple[str, str, str, SmoresTopologyEdge]],
        ] = {target_id: [] for target_id in target_tree.vertex_ids}
        for edge in target_tree.edges:
            target_adjacency[edge.vertex_a].append(
                (edge.vertex_b, edge.face_a, edge.face_b, edge)
            )
            target_adjacency[edge.vertex_b].append(
                (edge.vertex_a, edge.face_b, edge.face_a, edge)
            )

        physical_ids = sorted(
            {module_id for _, module_id in pair_cost}
        )
        physical_adjacency: dict[
            str,
            list[tuple[str, str, str]],
        ] = {module_id: [] for module_id in physical_ids}
        for edge in current_edges:
            face_a, face_b = self._edge_faces(edge)
            physical_adjacency[edge.module_a_id].append(
                (edge.module_b_id, face_a, face_b)
            )
            physical_adjacency[edge.module_b_id].append(
                (edge.module_a_id, face_b, face_a)
            )

        module_count = len(target_tree.vertex_ids)
        maximum_pair_cost = max(pair_cost.values(), default=0.0)
        motion_tier = module_count * max(1.0, maximum_pair_cost) + 1.0
        edge_tier = (module_count + 1) * motion_tier
        invalid_cost = (module_count + 1) * edge_tier

        @lru_cache(maxsize=None)
        def match(
            target_id: str,
            target_parent: str | None,
            module_id: str,
            module_parent: str | None,
        ) -> _CommonSubtreeMatch:
            target_children = sorted(
                (
                    item
                    for item in target_adjacency[target_id]
                    if item[0] != target_parent
                ),
                key=lambda item: (item[0], item[1], item[2]),
            )
            physical_children = sorted(
                (
                    item
                    for item in physical_adjacency[module_id]
                    if item[0] != module_parent
                ),
                key=lambda item: (item[0], item[1], item[2]),
            )
            compatible: dict[tuple[int, int], _CommonSubtreeMatch] = {}
            for target_index, target_child in enumerate(target_children):
                for module_index, physical_child in enumerate(
                    physical_children
                ):
                    if (
                        target_child[1] != physical_child[1]
                        or target_child[2] != physical_child[2]
                    ):
                        continue
                    compatible[(target_index, module_index)] = match(
                        target_child[0],
                        target_id,
                        physical_child[0],
                        module_id,
                    )

            target_count = len(target_children)
            physical_count = len(physical_children)
            matrix_size = target_count + physical_count
            selected_columns: tuple[int, ...] = ()
            if matrix_size:
                matrix = [
                    [0.0 for _ in range(matrix_size)]
                    for _ in range(matrix_size)
                ]
                for target_index in range(target_count):
                    for module_index in range(physical_count):
                        child = compatible.get(
                            (target_index, module_index)
                        )
                        if child is None:
                            matrix[target_index][module_index] = invalid_cost
                            continue
                        retained_gain = 1 + child.edge_count
                        root_degree_gain = (
                            motion_tier if target_id == target_root else 0.0
                        )
                        matrix[target_index][module_index] = (
                            -retained_gain * edge_tier
                            - root_degree_gain
                            + child.motion_cost
                        )
                selected_columns = solve_linear_assignment(matrix)

            mapping: dict[str, str] = {target_id: module_id}
            retained_edges: list[SmoresTopologyEdge] = []
            motion_cost = pair_cost[(target_id, module_id)]
            for target_index in range(target_count):
                module_index = selected_columns[target_index]
                child = compatible.get((target_index, module_index))
                if child is None:
                    continue
                mapping.update(child.target_to_module)
                retained_edges.append(target_children[target_index][3])
                retained_edges.extend(child.retained_edges)
                motion_cost += child.motion_cost
            return _CommonSubtreeMatch(
                target_to_module=mapping,
                retained_edges=tuple(retained_edges),
                edge_count=len(retained_edges),
                motion_cost=motion_cost,
            )

        if physical_root not in physical_adjacency:
            raise SelfReconfigurationPlanningError(
                f"Unknown physical root module {physical_root!r}."
            )
        return match(target_root, None, physical_root, None)

    @staticmethod
    def _motion_cost_by_pair(
        target_root: str,
        physical_root: str,
        physical_poses: Mapping[str, PlanarPose],
        unfolded_target: UnfoldedPlanarConfiguration,
    ) -> dict[tuple[str, str], float]:
        """Return every module-to-slot travel cost for one fixed root."""

        target_root_pose = unfolded_target.poses_by_vertex[target_root]
        physical_root_pose = physical_poses[physical_root]
        rotation = physical_root_pose.yaw_rad - target_root_pose.yaw_rad
        cosine = math.cos(rotation)
        sine = math.sin(rotation)
        result: dict[tuple[str, str], float] = {}
        for target_id, target_pose in unfolded_target.poses_by_vertex.items():
            local_x = target_pose.x_m - target_root_pose.x_m
            local_y = target_pose.y_m - target_root_pose.y_m
            desired_x = (
                physical_root_pose.x_m + cosine * local_x - sine * local_y
            )
            desired_y = (
                physical_root_pose.y_m + sine * local_x + cosine * local_y
            )
            for module_id, module_pose in physical_poses.items():
                result[(target_id, module_id)] = math.hypot(
                    module_pose.x_m - desired_x,
                    module_pose.y_m - desired_y,
                )
        return result

    @staticmethod
    def _target_xy_by_module(
        mapping: Mapping[str, str],
        target_root: str,
        physical_poses: Mapping[str, PlanarPose],
        unfolded_target: UnfoldedPlanarConfiguration,
    ) -> dict[str, tuple[float, float]]:
        """Anchor the unfolded target at the assigned live physical root."""

        target_root_pose = unfolded_target.poses_by_vertex[target_root]
        physical_root_pose = physical_poses[mapping[target_root]]
        rotation = physical_root_pose.yaw_rad - target_root_pose.yaw_rad
        cosine = math.cos(rotation)
        sine = math.sin(rotation)
        result: dict[str, tuple[float, float]] = {}
        for target_id, target_pose in unfolded_target.poses_by_vertex.items():
            local_x = target_pose.x_m - target_root_pose.x_m
            local_y = target_pose.y_m - target_root_pose.y_m
            result[mapping[target_id]] = (
                physical_root_pose.x_m
                + cosine * local_x
                - sine * local_y,
                physical_root_pose.y_m
                + sine * local_x
                + cosine * local_y,
            )
        return result

    def _progressive_action_waves(
        self,
        module_ids: tuple[str, ...],
        current_edges: tuple[GraphEdge, ...],
        retained_modules: set[str],
        detach_actions: tuple[ReconfigurationDetachAction, ...],
        actions: tuple[AssemblyAction, ...],
        physical_poses: Mapping[str, PlanarPose],
        target_xy_by_module: Mapping[str, tuple[float, float]],
    ) -> tuple[
        tuple[tuple[AssemblyAction, ...], ...],
        dict[str, int],
        dict[str, ReconfigurationDetachAction],
    ]:
        """Build safe outside-in waves from source and target dependencies."""

        adjacency: dict[str, list[tuple[str, GraphEdge]]] = {
            module_id: [] for module_id in module_ids
        }
        for edge in current_edges:
            adjacency[edge.module_a_id].append((edge.module_b_id, edge))
            adjacency[edge.module_b_id].append((edge.module_a_id, edge))

        source_depth = {module_id: 0 for module_id in retained_modules}
        parent_edge_by_module: dict[str, GraphEdge] = {}
        queue: deque[str] = deque(sorted(retained_modules))
        while queue:
            module_id = queue.popleft()
            for neighbour, edge in adjacency[module_id]:
                if neighbour in source_depth:
                    continue
                source_depth[neighbour] = source_depth[module_id] + 1
                parent_edge_by_module[neighbour] = edge
                queue.append(neighbour)
        isolated_modules = {
            module_id
            for module_id in module_ids
            if module_id not in source_depth and not adjacency[module_id]
        }
        isolated_depth = max(source_depth.values(), default=0) + 1
        source_depth.update(
            {module_id: isolated_depth for module_id in isolated_modules}
        )
        if set(source_depth) != set(module_ids):
            raise SelfReconfigurationPlanningError(
                "Cannot root the source topology at the retained component."
            )

        detach_by_key = {
            tuple(
                sorted(
                    (
                        (action.module_a_id, action.face_a),
                        (action.module_b_id, action.face_b),
                    )
                )
            ): action
            for action in detach_actions
        }
        detach_by_module: dict[str, ReconfigurationDetachAction] = {}
        for module_id in sorted(set(module_ids) - retained_modules):
            parent_edge = parent_edge_by_module.get(module_id)
            if parent_edge is None:
                continue
            key = self._physical_edge_key(parent_edge)
            detach = detach_by_key.get(key)
            if detach is None:
                raise SelfReconfigurationPlanningError(
                    f"Moving module {module_id} has no source detach action."
                )
            detach_by_module[module_id] = detach

        action_by_module = {
            action.mobile_module_id: action for action in actions
        }
        moving_modules = set(module_ids) - retained_modules
        if set(action_by_module) != moving_modules:
            raise SelfReconfigurationPlanningError(
                "Progressive reconfiguration requires exactly one target "
                "docking action per moving module."
            )

        successors = {module_id: set() for module_id in moving_modules}
        indegree = {module_id: 0 for module_id in moving_modules}

        def require_before(first: str, second: str) -> None:
            if first == second or second in successors[first]:
                return
            successors[first].add(second)
            indegree[second] += 1

        # A source child must leave before its parent can be detached and
        # driven.  This is the outside-in peel requested by the geometry.
        for module_id, parent_edge in parent_edge_by_module.items():
            if module_id not in moving_modules:
                continue
            parent = (
                parent_edge.module_b_id
                if parent_edge.module_a_id == module_id
                else parent_edge.module_a_id
            )
            if parent in moving_modules:
                require_before(module_id, parent)

        # Conversely, a target child can dock only after its target parent.
        for action in actions:
            if action.parent_module_id in moving_modules:
                require_before(
                    action.parent_module_id,
                    action.mobile_module_id,
                )

        original_index = {
            action.mobile_module_id: index
            for index, action in enumerate(actions)
        }
        ready = [
            module_id
            for module_id in moving_modules
            if indegree[module_id] == 0
        ]
        action_waves: list[tuple[AssemblyAction, ...]] = []
        while ready:
            ready.sort(
                key=lambda module_id: (
                    -source_depth[module_id],
                    original_index[module_id],
                    module_id,
                )
            )
            wave_modules: list[str] = []
            wave_depth = source_depth[ready[0]]
            for module_id in tuple(ready):
                if source_depth[module_id] != wave_depth:
                    continue
                if (
                    self.max_parallel_actions > 0
                    and len(wave_modules) >= self.max_parallel_actions
                ):
                    break
                if all(
                    self._parallel_actions_independent(
                        action_by_module[module_id],
                        action_by_module[peer],
                        detach_by_module.get(module_id),
                        detach_by_module.get(peer),
                        physical_poses,
                        target_xy_by_module,
                    )
                    for peer in wave_modules
                ):
                    wave_modules.append(module_id)
            if not wave_modules:
                wave_modules.append(ready[0])
            action_waves.append(
                tuple(action_by_module[module_id] for module_id in wave_modules)
            )
            for module_id in wave_modules:
                ready.remove(module_id)
            for module_id in wave_modules:
                for successor in sorted(successors[module_id]):
                    indegree[successor] -= 1
                    if indegree[successor] == 0:
                        ready.append(successor)
        if sum(len(wave) for wave in action_waves) != len(moving_modules):
            raise SelfReconfigurationPlanningError(
                "Source peel and target assembly dependencies form a cycle; "
                "this transition requires an intermediate parking action."
            )

        return (
            tuple(action_waves),
            source_depth,
            detach_by_module,
        )

    def _parallel_actions_independent(
        self,
        first: AssemblyAction,
        second: AssemblyAction,
        first_detach: ReconfigurationDetachAction | None,
        second_detach: ReconfigurationDetachAction | None,
        physical_poses: Mapping[str, PlanarPose],
        target_xy_by_module: Mapping[str, tuple[float, float]],
    ) -> bool:
        """Conservatively reject waves with shared or nearby corridors."""

        if first.requires_helper or second.requires_helper:
            return False
        if first.depth != second.depth:
            return False
        first_source_edge = (
            {first_detach.module_a_id, first_detach.module_b_id}
            if first_detach is not None
            else set()
        )
        second_source_edge = (
            {second_detach.module_a_id, second_detach.module_b_id}
            if second_detach is not None
            else set()
        )
        shared_source_modules = first_source_edge & second_source_edge
        if shared_source_modules:
            if len(shared_source_modules) != 1:
                return False
            shared_source = next(iter(shared_source_modules))
            if shared_source in {
                first.mobile_module_id,
                second.mobile_module_id,
            }:
                return False
            if self._detach_face(first_detach, shared_source) == (
                self._detach_face(second_detach, shared_source)
            ):
                return False
        shared_target_parent = (
            first.parent_module_id == second.parent_module_id
        )
        if shared_target_parent and first.parent_face == second.parent_face:
            return False

        first_start = physical_poses[first.mobile_module_id]
        second_start = physical_poses[second.mobile_module_id]
        first_start_xy = (first_start.x_m, first_start.y_m)
        second_start_xy = (second_start.x_m, second_start.y_m)
        first_target = target_xy_by_module[first.mobile_module_id]
        second_target = target_xy_by_module[second.mobile_module_id]
        first_parent = target_xy_by_module[first.parent_module_id]
        second_parent = target_xy_by_module[second.parent_module_id]
        clearance = self.parallel_path_clearance_m
        return (
            math.dist(first_start_xy, second_start_xy) >= clearance
            and math.dist(first_target, second_target) >= clearance
            and (
                shared_target_parent
                or math.dist(first_parent, second_parent) >= clearance
            )
            and not self._segments_intersect(
                first_start_xy,
                first_target,
                second_start_xy,
                second_target,
            )
            and self._synchronous_segment_distance(
                first_start_xy,
                first_target,
                second_start_xy,
                second_target,
            )
            >= clearance
        )

    @staticmethod
    def _detach_face(
        action: ReconfigurationDetachAction,
        module_id: str,
    ) -> str:
        if action.module_a_id == module_id:
            return action.face_a
        if action.module_b_id == module_id:
            return action.face_b
        raise SelfReconfigurationPlanningError(
            f"Detach action does not contain module {module_id!r}."
        )

    @staticmethod
    def _synchronous_segment_distance(
        first_start: tuple[float, float],
        first_end: tuple[float, float],
        second_start: tuple[float, float],
        second_end: tuple[float, float],
    ) -> float:
        """Minimum separation when both paths use equal normalized progress."""

        relative_start = (
            first_start[0] - second_start[0],
            first_start[1] - second_start[1],
        )
        relative_velocity = (
            (first_end[0] - first_start[0])
            - (second_end[0] - second_start[0]),
            (first_end[1] - first_start[1])
            - (second_end[1] - second_start[1]),
        )
        speed_squared = (
            relative_velocity[0] * relative_velocity[0]
            + relative_velocity[1] * relative_velocity[1]
        )
        if speed_squared <= 1.0e-18:
            return math.hypot(*relative_start)
        progress = -(
            relative_start[0] * relative_velocity[0]
            + relative_start[1] * relative_velocity[1]
        ) / speed_squared
        progress = min(1.0, max(0.0, progress))
        return math.hypot(
            relative_start[0] + progress * relative_velocity[0],
            relative_start[1] + progress * relative_velocity[1],
        )

    @classmethod
    def _segment_distance(
        cls,
        first_start: tuple[float, float],
        first_end: tuple[float, float],
        second_start: tuple[float, float],
        second_end: tuple[float, float],
    ) -> float:
        if cls._segments_intersect(
            first_start,
            first_end,
            second_start,
            second_end,
        ):
            return 0.0
        return min(
            cls._point_segment_distance(first_start, second_start, second_end),
            cls._point_segment_distance(first_end, second_start, second_end),
            cls._point_segment_distance(second_start, first_start, first_end),
            cls._point_segment_distance(second_end, first_start, first_end),
        )

    @staticmethod
    def _point_segment_distance(
        point: tuple[float, float],
        start: tuple[float, float],
        end: tuple[float, float],
    ) -> float:
        delta_x = end[0] - start[0]
        delta_y = end[1] - start[1]
        length_squared = delta_x * delta_x + delta_y * delta_y
        if length_squared <= 1.0e-18:
            return math.dist(point, start)
        projection = (
            (point[0] - start[0]) * delta_x
            + (point[1] - start[1]) * delta_y
        ) / length_squared
        projection = min(1.0, max(0.0, projection))
        closest = (
            start[0] + projection * delta_x,
            start[1] + projection * delta_y,
        )
        return math.dist(point, closest)

    @staticmethod
    def _segments_intersect(
        first_start: tuple[float, float],
        first_end: tuple[float, float],
        second_start: tuple[float, float],
        second_end: tuple[float, float],
    ) -> bool:
        def cross(
            first: tuple[float, float],
            second: tuple[float, float],
            third: tuple[float, float],
        ) -> float:
            return (
                (second[0] - first[0]) * (third[1] - first[1])
                - (second[1] - first[1]) * (third[0] - first[0])
            )

        first_a = cross(first_start, first_end, second_start)
        first_b = cross(first_start, first_end, second_end)
        second_a = cross(second_start, second_end, first_start)
        second_b = cross(second_start, second_end, first_end)
        epsilon = 1.0e-12

        def on_segment(
            point: tuple[float, float],
            start: tuple[float, float],
            end: tuple[float, float],
        ) -> bool:
            return (
                min(start[0], end[0]) - epsilon
                <= point[0]
                <= max(start[0], end[0]) + epsilon
                and min(start[1], end[1]) - epsilon
                <= point[1]
                <= max(start[1], end[1]) + epsilon
            )

        if first_a * first_b < -epsilon and second_a * second_b < -epsilon:
            return True
        return (
            abs(first_a) <= epsilon
            and on_segment(second_start, first_start, first_end)
        ) or (
            abs(first_b) <= epsilon
            and on_segment(second_end, first_start, first_end)
        ) or (
            abs(second_a) <= epsilon
            and on_segment(first_start, second_start, second_end)
        ) or (
            abs(second_b) <= epsilon
            and on_segment(first_end, second_start, second_end)
        )

    def _validate_isolated_movers(
        self,
        module_ids: tuple[str, ...],
        current_edges: tuple[GraphEdge, ...],
        retained_keys: set[tuple[tuple[str, str], tuple[str, str]]],
        retained_modules: set[str],
    ) -> None:
        adjacency = {module_id: set() for module_id in module_ids}
        for edge in current_edges:
            if self._physical_edge_key(edge) not in retained_keys:
                continue
            adjacency[edge.module_a_id].add(edge.module_b_id)
            adjacency[edge.module_b_id].add(edge.module_a_id)
        non_isolated = sorted(
            module_id
            for module_id in module_ids
            if module_id not in retained_modules and adjacency[module_id]
        )
        if non_isolated:
            raise SelfReconfigurationPlanningError(
                "This baseline can move only modules isolated after "
                "undocking; "
                f"remaining moving clusters contain {non_isolated}."
            )

    @staticmethod
    def _physical_module_ids(graph: AttributedRobotGraph) -> tuple[str, ...]:
        module_ids = tuple(
            sorted(
                node.module_id
                for node in graph.nodes
                if node.node_type == "physical_module"
            )
        )
        if not module_ids:
            raise SelfReconfigurationPlanningError(
                "The current graph contains no physical modules."
            )
        return module_ids

    @staticmethod
    def _attached_edges(graph: AttributedRobotGraph) -> tuple[GraphEdge, ...]:
        return tuple(
            edge
            for edge in graph.edges
            if bool(edge.attributes.get("is_attached"))
            or edge.relation_type == "current_connection"
        )

    def _validate_current_tree(
        self,
        module_ids: tuple[str, ...],
        edges: tuple[GraphEdge, ...],
    ) -> None:
        if len(edges) > len(module_ids) - 1:
            raise SelfReconfigurationPlanningError(
                "The current attachment graph contains too many edges to be "
                "a forest."
            )
        known = set(module_ids)
        adjacency = {module_id: set() for module_id in module_ids}
        used_faces: set[tuple[str, str]] = set()
        for edge in edges:
            if edge.module_a_id not in known or edge.module_b_id not in known:
                raise SelfReconfigurationPlanningError(
                    "An attachment references an unknown physical module."
                )
            face_a, face_b = self._edge_faces(edge)
            for endpoint in (
                (edge.module_a_id, face_a),
                (edge.module_b_id, face_b),
            ):
                if endpoint in used_faces:
                    raise SelfReconfigurationPlanningError(
                        f"Connector {endpoint[0]}.{endpoint[1]} is used twice."
                    )
                used_faces.add(endpoint)
            adjacency[edge.module_a_id].add(edge.module_b_id)
            adjacency[edge.module_b_id].add(edge.module_a_id)
        visited: set[str] = set()
        for component_root in module_ids:
            if component_root in visited:
                continue
            component_nodes: set[str] = set()
            component_edge_twice = 0
            queue: deque[str] = deque([component_root])
            while queue:
                module_id = queue.popleft()
                if module_id in component_nodes:
                    continue
                component_nodes.add(module_id)
                component_edge_twice += len(adjacency[module_id])
                queue.extend(adjacency[module_id] - component_nodes)
            if component_edge_twice // 2 != max(0, len(component_nodes) - 1):
                raise SelfReconfigurationPlanningError(
                    "Every current connected component must be a tree."
                )
            visited.update(component_nodes)

    def _detach_action(self, edge: GraphEdge) -> ReconfigurationDetachAction:
        face_a, face_b = self._edge_faces(edge)
        return ReconfigurationDetachAction(
            module_a_id=edge.module_a_id,
            face_a=face_a,
            module_b_id=edge.module_b_id,
            face_b=face_b,
        )

    @staticmethod
    def _edge_faces(edge: GraphEdge) -> tuple[str, str]:
        face_a = str(
            edge.attributes.get("connector_a_id")
            or edge.attributes.get("face_a")
            or ""
        ).upper()
        face_b = str(
            edge.attributes.get("connector_b_id")
            or edge.attributes.get("face_b")
            or ""
        ).upper()
        valid = {"LEFT", "RIGHT", "TOP", "BOTTOM"}
        if face_a not in valid or face_b not in valid:
            raise SelfReconfigurationPlanningError(
                f"Attachment has invalid faces {face_a!r}, {face_b!r}."
            )
        return face_a, face_b

    def _physical_edge_key(
        self,
        edge: GraphEdge,
    ) -> tuple[tuple[str, str], tuple[str, str]]:
        face_a, face_b = self._edge_faces(edge)
        return tuple(
            sorted(
                (
                    (edge.module_a_id, face_a),
                    (edge.module_b_id, face_b),
                )
            )
        )  # type: ignore[return-value]

    @staticmethod
    def _mapped_target_key(
        edge: SmoresTopologyEdge,
        mapping: Mapping[str, str],
    ) -> tuple[tuple[str, str], tuple[str, str]]:
        return tuple(
            sorted(
                (
                    (mapping[edge.vertex_a], edge.face_a),
                    (mapping[edge.vertex_b], edge.face_b),
                )
            )
        )  # type: ignore[return-value]

    @staticmethod
    def _target_edge_key(
        edge: SmoresTopologyEdge,
    ) -> tuple[tuple[str, str], tuple[str, str]]:
        return tuple(
            sorted(
                (
                    (edge.vertex_a, edge.face_a),
                    (edge.vertex_b, edge.face_b),
                )
            )
        )  # type: ignore[return-value]

    @staticmethod
    def _action_target_key(action) -> tuple[tuple[str, str], tuple[str, str]]:
        return tuple(
            sorted(
                (
                    (action.mobile_target_vertex, action.mobile_face),
                    (action.parent_target_vertex, action.parent_face),
                )
            )
        )  # type: ignore[return-value]
