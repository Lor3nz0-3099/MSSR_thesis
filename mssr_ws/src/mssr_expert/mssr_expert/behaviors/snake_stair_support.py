"""Compliance policy for wheeled Snake8 stair waves.

The gait planner owns geometry; this module owns the much smaller question of
which non-moving TILT joints are structural and which remain backdrivable while
the local climbing cell travels toward the tail.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

from mssr_expert.behaviors.morphology_library import (
    AssignedModule,
    BehaviorProgramStep,
)


def apply_trailing_compliance(
    program: Sequence[BehaviorProgramStep],
    assignments: Sequence[AssignedModule],
    *,
    trailing_support_modules: int = 1,
) -> tuple[BehaviorProgramStep, ...]:
    """Release the remote tail while retaining one support behind the wave.

    The original executor implicitly made every module outside a coordinated
    TILT group a structural hold.  That turns the rear of a long Snake8 into a
    rigid beam and inhibits load transfer.  Here the local wave still has one
    structural neighbour behind it, while modules farther toward the tail are
    explicitly passive/backdrivable.  Modules ahead of the wave stay rigid so
    the head/upper-deck portion does not flop during the first experiments.

    ``trailing_support_modules=1`` means that for a moving cell whose first
    active vertex is i, vertex i-1 is structural and 0..i-2 are passive.
    """

    if trailing_support_modules < 0:
        raise ValueError("trailing_support_modules must be non-negative")

    ordered = tuple(sorted(assignments, key=_vertex_index))
    index_by_module = {
        assignment.module_id: index
        for index, assignment in enumerate(ordered)
    }
    result: list[BehaviorProgramStep] = []

    for step in program:
        if not step.posture_targets:
            result.append(step)
            continue

        moving_indices = sorted(
            index_by_module[target.module_id]
            for target in step.posture_targets
        )
        first_moving = moving_indices[0]
        passive_stop = max(0, first_moving - trailing_support_modules)
        passive = tuple(
            assignment.module_id for assignment in ordered[:passive_stop]
        )

        # A posture that already addresses every module is the flat approach
        # lock.  Passing an explicit empty tuple also clears any earlier
        # latched passive policy at the end of a run.
        if len(moving_indices) == len(ordered):
            passive = ()

        result.append(
            replace(
                step,
                posture_targets=tuple(
                    replace(target, passive_module_ids=passive)
                    for target in step.posture_targets
                ),
            )
        )

    return tuple(result)


def _vertex_index(assignment: AssignedModule) -> int:
    prefix = "v"
    if not assignment.target_vertex_id.startswith(prefix):
        raise ValueError(
            "Snake stair compliance requires vN target vertex identifiers"
        )
    return int(assignment.target_vertex_id[len(prefix) :])
