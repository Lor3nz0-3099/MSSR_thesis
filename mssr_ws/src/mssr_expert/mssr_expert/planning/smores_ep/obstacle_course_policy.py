"""Task-level morphology policy for the SMORES-EP obstacle course.

The capability mapping follows the SMORES-EP morphology literature stored in
``references/SMORES-EP.pdf``, ``references/design and characterization of the
EP-Face Connector.pdf`` and ``references/chao_smores_reconfiguration_2019.pdf``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class CourseStep:
    """One morphology transition or operational behavior in the course."""

    task: str
    morphology: str
    behavior: str | None = None
    parameters: Mapping[str, float] | None = None
    navigation: str | None = None
    requires_button: bool = False
    requires_goal: bool = False


class ObstacleCoursePolicy:
    """Select the lowest-complexity morphology that satisfies each task."""

    _CAPABILITIES = {
        "snake8": frozenset({"assembly", "stairs", "train"}),
        "bridge8": frozenset({"gap", "train"}),
        "mobile_manipulator8": frozenset({"button", "train"}),
        "rc_car8": frozenset({"exit", "train"}),
    }
    _PREFERENCE = ("bridge8", "snake8", "mobile_manipulator8", "rc_car8")

    def choose_morphology(self, required_capability: str) -> str:
        """Return the preferred available morphology for one task capability."""
        for morphology in self._PREFERENCE:
            if required_capability in self._CAPABILITIES[morphology]:
                return morphology
        raise ValueError(
            f"No obstacle-course morphology supports {required_capability!r}."
        )

    def steps(self) -> tuple[CourseStep, ...]:
        """Return the complete assembly, reconfiguration and task program."""
        snake = self.choose_morphology("stairs")
        manipulator = self.choose_morphology("button")
        rc_car = self.choose_morphology("exit")
        return (
            CourseStep("assembly", rc_car),
            CourseStep(
                "ramp_climb",
                rc_car,
                navigation="ramp_exit",
            ),
            CourseStep(
                "rc_car_pre_gap",
                rc_car,
                navigation="safe_before_snake_reconfiguration",
            ),
            CourseStep("snake_reconfiguration", snake),
            CourseStep("snake_gap_approach", snake, navigation="front_before_gap"),
            CourseStep(
                "snake_gap_crossing",
                snake,
                "cross_gap_as_bridge",
                {
                    "linear_m_s": 0.030,
                    "span_duration_s": 4.0,
                    "tail_clear_duration_s": 6.0,
                },
            ),
            CourseStep("gap_clearance", snake, navigation="rear_past_gap"),
            CourseStep("stairs_approach", snake, navigation="front_before_stair_1"),
            CourseStep("stairs_lift_01", snake, "lift_head"),
            CourseStep("stairs_hook_01", snake, "hook_step"),
            CourseStep(
                "stairs_pull_01",
                snake,
                "pull_over_step",
                {
                    "linear_m_s": 0.022,
                    "riser_approach_duration_s": 8.0,
                    "front_pull_duration_s": 4.0,
                    "transfer_pull_duration_s": 4.0,
                    "tread_advance_duration_s": 5.0,
                },
            ),
            CourseStep("stairs_clear_01", snake, navigation="front_before_stair_2"),
            CourseStep("stairs_lift_02", snake, "lift_head"),
            CourseStep("stairs_hook_02", snake, "hook_step"),
            CourseStep(
                "stairs_pull_02",
                snake,
                "pull_over_step",
                {
                    "linear_m_s": 0.022,
                    "riser_approach_duration_s": 8.0,
                    "front_pull_duration_s": 4.0,
                    "transfer_pull_duration_s": 4.0,
                    "tread_advance_duration_s": 5.0,
                },
            ),
            CourseStep("stairs_clear_02", snake, navigation="front_before_stair_3"),
            CourseStep("stairs_lift_03", snake, "lift_head"),
            CourseStep("stairs_hook_03", snake, "hook_step"),
            CourseStep(
                "stairs_pull_03",
                snake,
                "pull_over_step",
                {
                    "linear_m_s": 0.022,
                    "riser_approach_duration_s": 8.0,
                    "front_pull_duration_s": 4.0,
                    "transfer_pull_duration_s": 4.0,
                    "tread_advance_duration_s": 5.0,
                },
            ),
            CourseStep("upper_deck_clearance", snake, navigation="front_on_upper_deck"),
            CourseStep("button_reconfiguration", manipulator),
            CourseStep("button_approach", manipulator, navigation="button_standoff"),
            CourseStep(
                "button",
                manipulator,
                "press_button",
                requires_button=True,
            ),
            CourseStep("button_release", manipulator, "release_button"),
            CourseStep("button_retreat", manipulator, navigation="button_retreat"),
            CourseStep("exit_reconfiguration", rc_car),
            CourseStep("exit", rc_car, navigation="cross_exit", requires_goal=True),
            CourseStep("exit_stop", rc_car, "stop"),
        )
