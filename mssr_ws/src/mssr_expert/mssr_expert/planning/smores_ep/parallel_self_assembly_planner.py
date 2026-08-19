"""Complete deterministic planner for parallel SMORES-EP self-assembly."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from mssr_expert.graph.attributed_robot_graph import (
    AttributedRobotGraph,
    GraphNode,
)
from mssr_expert.graph.task_graph import TaskGraphBuilder
from mssr_expert.planning.smores_ep.assembly_sequence import (
    ParallelAssemblyPlan,
    generate_parallel_assembly_plan,
)
from mssr_expert.planning.smores_ep.assignment import (
    AssignmentResult,
    assign_modules_to_targets,
)
from mssr_expert.planning.smores_ep.attributed_adapter import (
    target_roles_from_graph,
    target_graph_to_kinematic_tree,
)
from mssr_expert.planning.smores_ep.rooting import (
    ModulePosition,
    RootedSmoresTree,
    choose_physical_root,
    root_kinematic_tree,
)
from mssr_expert.planning.smores_ep.topology import (
    SmoresKinematicTree,
)
from mssr_expert.planning.smores_ep.unfolding import (
    FACE_ANGLE_RAD,
    PlanarModuleGeometry,
    PlanarPose,
    UnfoldedPlanarConfiguration,
    normalize_angle,
    unfold_tree_on_plane,
)


class ParallelSelfAssemblyPlannerError(ValueError):
    """Raised when a self-assembly plan cannot be generated."""


@dataclass(frozen=True)
class ParallelSelfAssemblyPlanningResult:
    """Complete output of the deterministic self-assembly planner."""

    current_graph: AttributedRobotGraph
    target_graph: AttributedRobotGraph
    task_graph: AttributedRobotGraph

    target_tree: SmoresKinematicTree
    rooted_target_tree: RootedSmoresTree
    unfolded_target: UnfoldedPlanarConfiguration

    physical_root_id: str
    assignment: AssignmentResult
    assembly_plan: ParallelAssemblyPlan
    layout_pose_by_module: Mapping[str, PlanarPose]
    reserve_module_ids: tuple[str, ...]


class ParallelSelfAssemblyPlanner:
    """Build the complete Algorithm 1 plan from attributed graphs."""

    def __init__(
        self,
        geometry: PlanarModuleGeometry | None = None,
        orientation_weight_m_per_rad: float = 0.0,
        require_disconnected_modules: bool = True,
        layout_clearance_m: float = 0.070,
        assignment_staging_distance_m: float = 0.070,
        assignment_corridor_clearance_m: float = 0.110,
    ) -> None:
        self.geometry = geometry or PlanarModuleGeometry()
        self.orientation_weight_m_per_rad = (
            orientation_weight_m_per_rad
        )
        self.require_disconnected_modules = (
            require_disconnected_modules
        )
        if not math.isfinite(layout_clearance_m) or layout_clearance_m < 0.0:
            raise ParallelSelfAssemblyPlannerError(
                "layout_clearance_m must be finite and non-negative."
            )
        self.layout_clearance_m = layout_clearance_m
        if (
            not math.isfinite(assignment_staging_distance_m)
            or assignment_staging_distance_m <= 0.0
        ):
            raise ParallelSelfAssemblyPlannerError(
                "assignment_staging_distance_m must be positive and finite."
            )
        if (
            not math.isfinite(assignment_corridor_clearance_m)
            or assignment_corridor_clearance_m <= 0.0
        ):
            raise ParallelSelfAssemblyPlannerError(
                "assignment_corridor_clearance_m must be positive and "
                "finite."
            )
        self.assignment_staging_distance_m = assignment_staging_distance_m
        self.assignment_corridor_clearance_m = (
            assignment_corridor_clearance_m
        )
        self._task_graph_builder = TaskGraphBuilder()

    def plan(
        self,
        current_graph: AttributedRobotGraph,
        target_graph: AttributedRobotGraph,
    ) -> ParallelSelfAssemblyPlanningResult:
        """Generate a complete deterministic parallel assembly plan."""

        if self.require_disconnected_modules:
            self._validate_modules_are_disconnected(current_graph)

        physical_poses = self._extract_physical_poses(current_graph)

        target_tree = target_graph_to_kinematic_tree(
            target_graph
        )

        self._validate_target_module_count(target_graph, target_tree)

        if len(physical_poses) < len(target_tree.vertex_ids):
            raise ParallelSelfAssemblyPlannerError(
                "The number of physical modules is smaller than the number "
                "of target vertices; add reserve modules before assembly."
            )

        rooted_target_tree = root_kinematic_tree(
            target_tree,
            root_id=self._declared_target_root(target_graph),
        )

        conditioned_target_graph = self._mark_target_root(
            target_graph,
            rooted_target_tree.root_id,
        )

        unfolded_target = unfold_tree_on_plane(
            rooted_target_tree,
            geometry=self.geometry,
        )

        module_positions = tuple(
            ModulePosition(
                module_id=module_id,
                x_m=pose.x_m,
                y_m=pose.y_m,
                z_m=self._module_height(
                    current_graph,
                    module_id,
                ),
            )
            for module_id, pose in sorted(physical_poses.items())
        )

        physical_root_id = choose_physical_root(
            module_positions
        )

        assignment = assign_modules_to_targets(
            physical_poses=physical_poses,
            physical_root_id=physical_root_id,
            target=unfolded_target,
            orientation_weight_m_per_rad=(
                self.orientation_weight_m_per_rad
            ),
            target_parent_by_vertex=(
                rooted_target_tree.parent_by_vertex
            ),
            target_depth_by_vertex=(
                rooted_target_tree.depth_by_vertex
            ),
            staging_distance_m=self.assignment_staging_distance_m,
            staging_corridor_clearance_m=(
                self.assignment_corridor_clearance_m
            ),
        )
        assigned_module_ids = set(assignment.target_to_module.values())
        reserve_module_ids = tuple(
            sorted(set(physical_poses) - assigned_module_ids)
        )
        required_reserves = target_graph.global_attributes.get(
            "dedicated_helper_module_count", 0
        )
        if (
            not isinstance(required_reserves, int)
            or isinstance(required_reserves, bool)
            or required_reserves < 0
        ):
            raise ParallelSelfAssemblyPlannerError(
                "dedicated_helper_module_count must be a non-negative integer."
            )
        if len(reserve_module_ids) < required_reserves:
            raise ParallelSelfAssemblyPlannerError(
                f"Target needs {required_reserves} dedicated helper/reserve "
                f"module(s), but only {len(reserve_module_ids)} are available."
            )

        assembly_plan = generate_parallel_assembly_plan(
            tree=rooted_target_tree,
            assignment=assignment,
        )
        self._validate_helper_metadata(target_graph, assembly_plan)

        layout_pose_by_module = self._layout_poses(
            physical_poses=physical_poses,
            physical_root_id=physical_root_id,
            target=unfolded_target,
            assignment=assignment,
            rooted_tree=rooted_target_tree,
        )

        task_graph = self._task_graph_builder.build(
            current_graph=current_graph,
            target_graph=conditioned_target_graph,
            assignment=assignment.target_to_module,
            execution_state={
                "expert": "parallel_self_assembly",
                "phase": "PLANNED",
                "target_root_vertex": rooted_target_tree.root_id,
                "physical_root_module": physical_root_id,
                "wave_index": 0,
                "wave_count": len(assembly_plan.waves),
                "action_count": assembly_plan.action_count,
                "reserve_module_ids": list(reserve_module_ids),
                "assignment_motion_cost_m": assignment.total_cost,
                "assignment_future_blockers": (
                    assignment.total_future_blockers
                ),
            },
        )

        return ParallelSelfAssemblyPlanningResult(
            current_graph=current_graph,
            target_graph=conditioned_target_graph,
            task_graph=task_graph,
            target_tree=target_tree,
            rooted_target_tree=rooted_target_tree,
            unfolded_target=unfolded_target,
            physical_root_id=physical_root_id,
            assignment=assignment,
            assembly_plan=assembly_plan,
            layout_pose_by_module=layout_pose_by_module,
            reserve_module_ids=reserve_module_ids,
        )

    @staticmethod
    def _declared_target_root(
        target_graph: AttributedRobotGraph,
    ) -> str | None:
        """Resolve the optional semantic root declared by the target JSON."""

        roles = target_roles_from_graph(target_graph)
        node_roots = sorted(
            vertex_id
            for vertex_id, attributes in roles.items()
            if bool(attributes.get("is_target_root", False))
        )
        if len(node_roots) > 1:
            raise ParallelSelfAssemblyPlannerError(
                "Target graph declares more than one is_target_root: "
                f"{node_roots}."
            )

        global_root_value = target_graph.global_attributes.get(
            "target_root_vertex"
        )
        global_root = (
            str(global_root_value)
            if global_root_value is not None
            else None
        )
        node_root = node_roots[0] if node_roots else None
        if node_root is not None and global_root is not None:
            if node_root != global_root:
                raise ParallelSelfAssemblyPlannerError(
                    "Target root declarations disagree: "
                    f"node={node_root!r}, global={global_root!r}."
                )
        return node_root or global_root

    @staticmethod
    def _validate_target_module_count(
        target_graph: AttributedRobotGraph,
        target_tree: SmoresKinematicTree,
    ) -> None:
        declared = target_graph.global_attributes.get("module_count")
        if declared is None:
            return
        if not isinstance(declared, int) or isinstance(declared, bool):
            raise ParallelSelfAssemblyPlannerError(
                "Target module_count must be an integer."
            )
        if declared != len(target_tree.vertex_ids):
            raise ParallelSelfAssemblyPlannerError(
                "Target module_count does not match its target slots: "
                f"declared={declared}, actual={len(target_tree.vertex_ids)}."
            )

    @staticmethod
    def _validate_helper_metadata(
        target_graph: AttributedRobotGraph,
        assembly_plan: ParallelAssemblyPlan,
    ) -> None:
        declared = target_graph.global_attributes.get(
            "requires_helping_module"
        )
        if declared is None:
            return
        if not isinstance(declared, bool):
            raise ParallelSelfAssemblyPlannerError(
                "requires_helping_module must be boolean."
            )
        if declared != assembly_plan.requires_helper:
            raise ParallelSelfAssemblyPlannerError(
                "Target helper metadata disagrees with its docking faces: "
                f"declared={declared}, required={assembly_plan.requires_helper}."
            )

    def _layout_poses(
        self,
        physical_poses: Mapping[str, PlanarPose],
        physical_root_id: str,
        target: UnfoldedPlanarConfiguration,
        assignment: AssignmentResult,
        rooted_tree: RootedSmoresTree,
    ) -> dict[str, PlanarPose]:
        """Place the unfolded target around the physical root with clearance."""

        physical_root = physical_poses[physical_root_id]
        target_root = target.poses_by_vertex[target.root_id]
        staged_by_target: dict[str, PlanarPose] = {
            target.root_id: target_root,
        }
        for edge in rooted_tree.edges:
            exact_parent = target.poses_by_vertex[edge.parent_vertex]
            exact_child = target.poses_by_vertex[edge.child_vertex]
            staged_parent = staged_by_target[edge.parent_vertex]
            direction = exact_parent.yaw_rad + FACE_ANGLE_RAD[edge.parent_face]
            staged_by_target[edge.child_vertex] = PlanarPose(
                x_m=(
                    staged_parent.x_m
                    + exact_child.x_m
                    - exact_parent.x_m
                    + self.layout_clearance_m * math.cos(direction)
                ),
                y_m=(
                    staged_parent.y_m
                    + exact_child.y_m
                    - exact_parent.y_m
                    + self.layout_clearance_m * math.sin(direction)
                ),
                yaw_rad=exact_child.yaw_rad,
            )
        result: dict[str, PlanarPose] = {}
        for target_id, target_pose in staged_by_target.items():
            local_x = target_pose.x_m - target_root.x_m
            local_y = target_pose.y_m - target_root.y_m
            cosine = math.cos(physical_root.yaw_rad)
            sine = math.sin(physical_root.yaw_rad)
            module_id = assignment.target_to_module[target_id]
            result[module_id] = PlanarPose(
                x_m=(
                    physical_root.x_m
                    + cosine * local_x
                    - sine * local_y
                ),
                y_m=(
                    physical_root.y_m
                    + sine * local_x
                    + cosine * local_y
                ),
                yaw_rad=normalize_angle(
                    physical_root.yaw_rad
                    + target_pose.yaw_rad
                    - target_root.yaw_rad
                ),
            )
        return result

    def _extract_physical_poses(
        self,
        graph: AttributedRobotGraph,
    ) -> dict[str, PlanarPose]:
        """Extract planar poses of controllable SMORES-EP modules."""

        physical_nodes = [
            node
            for node in graph.nodes
            if node.node_type == "physical_module"
        ]

        if not physical_nodes:
            raise ParallelSelfAssemblyPlannerError(
                "The current graph contains no physical modules."
            )

        poses: dict[str, PlanarPose] = {}

        for node in physical_nodes:
            robot_family = str(
                node.attributes.get(
                    "robot_family",
                    "",
                )
            ).lower()

            if robot_family not in {
                "smores_ep",
                "smores-ep",
            }:
                raise ParallelSelfAssemblyPlannerError(
                    f"Module {node.module_id!r} belongs to unsupported "
                    f"robot family {robot_family!r}."
                )

            if node.attributes.get("control_available") is False:
                raise ParallelSelfAssemblyPlannerError(
                    f"Module {node.module_id!r} is not controllable."
                )

            position = self._position_from_node(node)
            yaw_rad = self._yaw_from_node(node)

            poses[node.module_id] = PlanarPose(
                x_m=position[0],
                y_m=position[1],
                yaw_rad=yaw_rad,
            )

        return poses

    def _position_from_node(
        self,
        node: GraphNode,
    ) -> tuple[float, float, float]:
        """Read a three-dimensional position from a graph node."""

        position = node.attributes.get("position")

        if position is None:
            pose = node.attributes.get("pose", {})

            if isinstance(pose, Mapping):
                position = pose.get("position")

        if (
            not isinstance(position, list | tuple)
            or len(position) < 3
        ):
            raise ParallelSelfAssemblyPlannerError(
                f"Module {node.module_id!r} has no valid position."
            )

        result = (
            float(position[0]),
            float(position[1]),
            float(position[2]),
        )

        if not all(math.isfinite(value) for value in result):
            raise ParallelSelfAssemblyPlannerError(
                f"Module {node.module_id!r} has a non-finite position."
            )

        return result

    def _yaw_from_node(
        self,
        node: GraphNode,
    ) -> float:
        """Read planar yaw from a direct angle or quaternion."""

        direct_yaw = node.attributes.get("yaw_rad")

        if direct_yaw is not None:
            yaw = float(direct_yaw)

            if not math.isfinite(yaw):
                raise ParallelSelfAssemblyPlannerError(
                    f"Module {node.module_id!r} has invalid yaw."
                )

            return yaw

        orientation: Any = node.attributes.get("orientation")

        if orientation is None:
            pose = node.attributes.get("pose", {})

            if isinstance(pose, Mapping):
                orientation = (
                    pose.get("orientation_xyzw")
                    or pose.get("orientation")
                )

        if isinstance(orientation, Mapping):
            orientation = (
                orientation.get("x", 0.0),
                orientation.get("y", 0.0),
                orientation.get("z", 0.0),
                orientation.get("w", 1.0),
            )

        if (
            not isinstance(orientation, list | tuple)
            or len(orientation) < 4
        ):
            raise ParallelSelfAssemblyPlannerError(
                f"Module {node.module_id!r} has no valid orientation."
            )

        quaternion = tuple(
            float(orientation[index])
            for index in range(4)
        )

        if not all(math.isfinite(value) for value in quaternion):
            raise ParallelSelfAssemblyPlannerError(
                f"Module {node.module_id!r} has invalid orientation."
            )

        x, y, z, w = quaternion
        norm = math.sqrt(
            x * x + y * y + z * z + w * w
        )

        if norm <= 1e-12:
            raise ParallelSelfAssemblyPlannerError(
                f"Module {node.module_id!r} has a zero quaternion."
            )

        x /= norm
        y /= norm
        z /= norm
        w /= norm

        sine_yaw = 2.0 * (w * z + x * y)
        cosine_yaw = 1.0 - 2.0 * (y * y + z * z)

        return math.atan2(
            sine_yaw,
            cosine_yaw,
        )

    def _module_height(
        self,
        graph: AttributedRobotGraph,
        module_id: str,
    ) -> float:
        """Return the Z coordinate of one physical module."""

        node = graph.node_by_id().get(module_id)

        if node is None:
            raise ParallelSelfAssemblyPlannerError(
                f"Unknown module {module_id!r}."
            )

        return self._position_from_node(node)[2]

    def _validate_modules_are_disconnected(
        self,
        graph: AttributedRobotGraph,
    ) -> None:
        """Reject already assembled modules in the self-assembly planner."""

        attached_edges = tuple(
            edge
            for edge in graph.edges
            if edge.relation_type == "current_connection"
            and bool(edge.attributes.get("is_attached", True))
        )

        if attached_edges:
            raise ParallelSelfAssemblyPlannerError(
                "Parallel self-assembly requires initially separated "
                "modules. Use the reconfiguration expert for connected "
                "configurations."
            )

    def _mark_target_root(
        self,
        target_graph: AttributedRobotGraph,
        root_vertex_id: str,
    ) -> AttributedRobotGraph:
        """Return a target graph containing the selected root attribute."""

        updated_nodes: list[GraphNode] = []

        for node in target_graph.nodes:
            target_vertex_id = str(
                node.attributes.get(
                    "target_vertex_id",
                    self._strip_target_prefix(node.node_id),
                )
            )

            updated_nodes.append(
                GraphNode(
                    node.module_id,
                    {
                        **dict(node.attributes),
                        "target_vertex_id": target_vertex_id,
                        "is_target_root": (
                            target_vertex_id == root_vertex_id
                        ),
                    },
                )
            )

        return AttributedRobotGraph(
            stamp=target_graph.stamp,
            nodes=tuple(updated_nodes),
            edges=target_graph.edges,
            global_attributes={
                **dict(target_graph.global_attributes),
                "schema_version": "mssr.target_graph.v1",
                "graph_kind": "target_morphology",
                "target_root_vertex": root_vertex_id,
            },
        )

    @staticmethod
    def _strip_target_prefix(node_id: str) -> str:
        if node_id.startswith("target:"):
            return node_id[len("target:"):]

        return node_id
