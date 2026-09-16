"""Expand semantic mission tasks into reusable expert execution stages."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from mssr_expert.planning.smores_ep.obstacle_course_policy import (
    MissionTask,
    ObstacleCoursePolicy,
)


@dataclass(frozen=True)
class MissionStage:
    """One assembly, transition, navigation, or behavior boundary."""

    stage_id: int
    task_id: str
    task_type: str
    kind: str
    target_morphology: str
    source_morphology: str | None = None
    behavior: str | None = None
    parameters: Mapping[str, Any] = field(default_factory=dict)


class CompositeMissionPlanner:
    """Create a deterministic stage program without low-level control logic."""

    def __init__(self, policy: ObstacleCoursePolicy) -> None:
        self._policy = policy

    def build(
        self,
        tasks: Sequence[MissionTask | Mapping[str, Any]],
        *,
        initial_morphology: str | None = None,
    ) -> tuple[MissionStage, ...]:
        decisions = self._policy.plan(tasks, initial_morphology)
        stages: list[MissionStage] = []
        current = initial_morphology

        def append(
            task_id: str,
            task_type: str,
            kind: str,
            target: str,
            *,
            behavior: str | None = None,
            parameters: Mapping[str, Any] | None = None,
        ) -> None:
            nonlocal current
            stage_parameters = dict(parameters or {})
            stage_parameters.setdefault("task_id", task_id)
            stage_parameters.setdefault("task_type", task_type)
            stages.append(
                MissionStage(
                    stage_id=len(stages),
                    task_id=task_id,
                    task_type=task_type,
                    kind=kind,
                    source_morphology=current,
                    target_morphology=target,
                    behavior=behavior,
                    parameters=stage_parameters,
                )
            )
            if kind in {"assembly", "reconfiguration"}:
                current = target

        def ensure_morphology(
            task_id: str,
            task_type: str,
            target: str,
        ) -> None:
            if current == target:
                return
            append(
                task_id,
                task_type,
                "assembly" if current is None else "reconfiguration",
                target,
            )

        for decision in decisions:
            if decision.execution_kind == "button_expert":
                # The validated button expert establishes heading in RC-Car8,
                # transitions to MM8, presses/clears/retreats, then returns to
                # RC-Car8.  Never invent a direct Snake8 -> MM8 shortcut.
                rc_car = self._policy.choose_morphology("flat_navigation")
                ensure_morphology(decision.task, decision.task_type, rc_car)
                append(
                    decision.task,
                    decision.task_type,
                    "button_rc_alignment",
                    rc_car,
                    parameters=decision.parameters,
                )
                ensure_morphology(
                    decision.task,
                    decision.task_type,
                    decision.morphology,
                )
                append(
                    decision.task,
                    decision.task_type,
                    "button_expert",
                    decision.morphology,
                    parameters=decision.parameters,
                )
                ensure_morphology(decision.task, decision.task_type, rc_car)
                continue

            # The preceding RC-Car8 Nav2 stage already brings the
            # vehicle onto the gap reconfiguration pad.  Nav2 is allowed
            # to finish coarsely; a local reverse-arc expert establishes
            # the precise heading before RC-Car8 -> Snake8.
            if decision.task_type == "gap":
                rc_car = self._policy.choose_morphology(
                    "flat_navigation"
                )

                raw_goal = decision.parameters.get(
                    "reconfiguration_pose_xyyaw"
                )

                if (
                    current == rc_car
                    and isinstance(raw_goal, (list, tuple))
                    and len(raw_goal) == 3
                ):
                    append(
                        decision.task,
                        decision.task_type,
                        "gap_rc_alignment",
                        rc_car,
                        parameters={
                            "target_yaw_rad": float(raw_goal[2]),
                        },
                    )

            ensure_morphology(
                decision.task,
                decision.task_type,
                decision.morphology,
            )
            append(
                decision.task,
                decision.task_type,
                decision.execution_kind,
                decision.morphology,
                behavior=decision.behavior,
                parameters=decision.parameters,
            )

        if not stages or stages[-1].task_type != "goal":
            raise ValueError("Expanded composite mission has no terminal goal")
        return tuple(stages)
