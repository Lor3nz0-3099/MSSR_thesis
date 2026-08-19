"""Planning algorithms for the SMORES-EP modular robot."""

from mssr_expert.planning.smores_ep.attributed_adapter import (
    target_graph_to_kinematic_tree,
    target_roles_from_graph,
)
from mssr_expert.planning.smores_ep.assembly_sequence import (
    AssemblyAction,
    AssemblySequenceError,
    AssemblyWave,
    ParallelAssemblyPlan,
    generate_parallel_assembly_plan,
)
from mssr_expert.planning.smores_ep.assignment import (
    AssignmentError,
    AssignmentResult,
    assign_modules_to_targets,
    solve_linear_assignment,
)
from mssr_expert.planning.smores_ep.rooting import (
    ModulePosition,
    PhysicalRootSelectionError,
    RootedSmoresEdge,
    RootedSmoresTree,
    choose_physical_root,
    root_kinematic_tree,
    swarm_centroid,
    vertices_by_depth,
)
from mssr_expert.planning.smores_ep.topology import (
    SmoresKinematicTree,
    SmoresTopologyEdge,
    TopologyValidationError,
    choose_target_root,
    graph_centers,
    validate_kinematic_tree,
)
from mssr_expert.planning.smores_ep.unfolding import (
    PlanarModuleGeometry,
    PlanarPose,
    PlanarUnfoldingError,
    UnfoldedPlanarConfiguration,
    normalize_angle,
    unfold_tree_on_plane,
)

from mssr_expert.planning.smores_ep.parallel_self_assembly_planner import (
    ParallelSelfAssemblyPlanner,
    ParallelSelfAssemblyPlannerError,
    ParallelSelfAssemblyPlanningResult,
)

__all__ = [
    "AssemblyAction",
    "AssemblySequenceError",
    "AssemblyWave",
    "AssignmentError",
    "AssignmentResult",
    "ModulePosition",
    "ParallelAssemblyPlan",
    "PhysicalRootSelectionError",
    "PlanarModuleGeometry",
    "PlanarPose",
    "PlanarUnfoldingError",
    "RootedSmoresEdge",
    "RootedSmoresTree",
    "SmoresKinematicTree",
    "SmoresTopologyEdge",
    "TopologyValidationError",
    "UnfoldedPlanarConfiguration",
    "assign_modules_to_targets",
    "choose_physical_root",
    "choose_target_root",
    "generate_parallel_assembly_plan",
    "graph_centers",
    "normalize_angle",
    "root_kinematic_tree",
    "solve_linear_assignment",
    "swarm_centroid",
    "target_graph_to_kinematic_tree",
    "target_roles_from_graph",
    "unfold_tree_on_plane",
    "validate_kinematic_tree",
    "vertices_by_depth",
    "ParallelSelfAssemblyPlanner",
    "ParallelSelfAssemblyPlannerError",
    "ParallelSelfAssemblyPlanningResult",
]
