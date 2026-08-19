"""Parallel assembly-wave generation for SMORES-EP."""

from __future__ import annotations

from dataclasses import dataclass

from mssr_expert.planning.smores_ep.assignment import (
    AssignmentResult,
)
from mssr_expert.planning.smores_ep.rooting import (
    RootedSmoresTree,
)


class AssemblySequenceError(ValueError):
    """Raised when a parallel assembly sequence cannot be generated."""


@dataclass(frozen=True)
class AssemblyAction:
    """One child-to-parent docking operation."""

    mobile_module_id: str
    mobile_face: str
    parent_module_id: str
    parent_face: str

    mobile_target_vertex: str
    parent_target_vertex: str

    depth: int
    clocking_quarter_turns: int

    requires_helper: bool


@dataclass(frozen=True)
class AssemblyWave:
    """A group of assembly actions that can run in parallel."""

    wave_index: int
    depth: int
    phase: str
    actions: tuple[AssemblyAction, ...]

    @property
    def can_execute_in_parallel(self) -> bool:
        """Return whether the actions belong to one parallel group."""

        return len(self.actions) > 1


@dataclass(frozen=True)
class ParallelAssemblyPlan:
    """Complete root-to-leaves parallel assembly plan."""

    root_target_vertex: str
    root_module_id: str
    waves: tuple[AssemblyWave, ...]

    @property
    def all_actions(self) -> tuple[AssemblyAction, ...]:
        """Return all actions in their execution order."""

        return tuple(
            action
            for wave in self.waves
            for action in wave.actions
        )

    @property
    def action_count(self) -> int:
        """Return the total number of required docking actions."""

        return len(self.all_actions)

    @property
    def requires_helper(self) -> bool:
        """Return whether any action needs the assisted docking procedure."""

        return any(action.requires_helper for action in self.all_actions)


def generate_parallel_assembly_plan(
    tree: RootedSmoresTree,
    assignment: AssignmentResult,
) -> ParallelAssemblyPlan:
    """Generate Algorithm 1 assembly waves from root to leaves."""

    _validate_assignment(tree, assignment)

    root_module_id = assignment.target_to_module[tree.root_id]

    actions_by_depth: dict[int, list[AssemblyAction]] = {}

    for edge in tree.edges:
        depth = tree.depth_by_vertex.get(edge.child_vertex)

        if depth is None:
            raise AssemblySequenceError(
                f"Missing depth for vertex {edge.child_vertex!r}."
            )

        mobile_module_id = assignment.target_to_module[
            edge.child_vertex
        ]
        parent_module_id = assignment.target_to_module[
            edge.parent_vertex
        ]

        action = AssemblyAction(
            mobile_module_id=mobile_module_id,
            mobile_face=edge.child_face,
            parent_module_id=parent_module_id,
            parent_face=edge.parent_face,
            mobile_target_vertex=edge.child_vertex,
            parent_target_vertex=edge.parent_vertex,
            depth=depth,
            clocking_quarter_turns=edge.clocking_quarter_turns,
            requires_helper=edge.child_face in {"LEFT", "RIGHT"},
        )

        actions_by_depth.setdefault(depth, []).append(action)

    waves: list[AssemblyWave] = []

    for depth in sorted(actions_by_depth):
        actions = sorted(
            actions_by_depth[depth],
            key=_action_sort_key,
        )

        if depth == 1:
            lateral_root_actions = tuple(
                action
                for action in actions
                if action.parent_target_vertex == tree.root_id
                and action.parent_face in {"LEFT", "RIGHT"}
            )

            axial_root_actions = tuple(
                action
                for action in actions
                if action.parent_target_vertex == tree.root_id
                and action.parent_face in {"TOP", "BOTTOM"}
            )

            other_actions = tuple(
                action
                for action in actions
                if action not in lateral_root_actions
                and action not in axial_root_actions
            )

            if lateral_root_actions:
                waves.append(
                    AssemblyWave(
                        wave_index=len(waves),
                        depth=depth,
                        phase="ROOT_LEFT_RIGHT",
                        actions=lateral_root_actions,
                    )
                )

            if axial_root_actions:
                waves.append(
                    AssemblyWave(
                        wave_index=len(waves),
                        depth=depth,
                        phase="ROOT_TOP_BOTTOM",
                        actions=axial_root_actions,
                    )
                )

            if other_actions:
                waves.append(
                    AssemblyWave(
                        wave_index=len(waves),
                        depth=depth,
                        phase="DEPTH_PARALLEL",
                        actions=other_actions,
                    )
                )

            continue

        waves.append(
            AssemblyWave(
                wave_index=len(waves),
                depth=depth,
                phase="DEPTH_PARALLEL",
                actions=tuple(actions),
            )
        )

    plan = ParallelAssemblyPlan(
        root_target_vertex=tree.root_id,
        root_module_id=root_module_id,
        waves=tuple(waves),
    )

    expected_action_count = max(0, len(tree.vertex_ids) - 1)

    if plan.action_count != expected_action_count:
        raise AssemblySequenceError(
            "The assembly plan does not contain exactly one action "
            "for every non-root target vertex."
        )

    return plan


def _validate_assignment(
    tree: RootedSmoresTree,
    assignment: AssignmentResult,
) -> None:
    """Validate the target-to-module mapping used by the plan."""

    expected_vertices = set(tree.vertex_ids)
    assigned_vertices = set(assignment.target_to_module)

    if assigned_vertices != expected_vertices:
        missing = sorted(expected_vertices - assigned_vertices)
        unexpected = sorted(assigned_vertices - expected_vertices)

        raise AssemblySequenceError(
            "Assignment and target topology do not match. "
            f"Missing={missing}, unexpected={unexpected}."
        )

    assigned_modules = tuple(
        assignment.target_to_module[vertex_id]
        for vertex_id in tree.vertex_ids
    )

    if len(assigned_modules) != len(set(assigned_modules)):
        raise AssemblySequenceError(
            "A physical module was assigned to multiple target vertices."
        )

    if tree.root_id not in assignment.target_to_module:
        raise AssemblySequenceError(
            "The target root has no assigned physical module."
        )


def _action_sort_key(
    action: AssemblyAction,
) -> tuple[str, int, str, str]:
    """Return a deterministic order inside an assembly wave."""

    face_priority = {
        "LEFT": 0,
        "RIGHT": 1,
        "TOP": 2,
        "BOTTOM": 3,
    }

    return (
        action.parent_module_id,
        face_priority[action.parent_face],
        action.mobile_module_id,
        action.mobile_face,
    )
