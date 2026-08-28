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
        "snake8": frozenset({"assembly", "gap", "stairs", "train"}),
        "bridge8": frozenset({"gap", "train"}),
        "mobile_manipulator8": frozenset({"button", "train"}),
        "rc_car8": frozenset({"exit", "train"}),
    }
    _PREFERENCE = ("snake8", "bridge8", "mobile_manipulator8", "rc_car8")

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
                "gap_crossing",
                {
                    "linear_m_s": 0.040,
                    "approach_linear_m_s": 0.050,
                    "gap_goal_tolerance_m": 0.004,
                },
            ),
            CourseStep("gap_clearance", snake, navigation="rear_past_gap"),
            CourseStep("stairs_approach", snake, navigation="front_before_stair_1"),
            CourseStep(
                "stairs_crawl",
                snake,
                "crawl_stairs_arch_wave",
                {
                    "linear_m_s": 0.040,
                    "riser_approach_linear_m_s": 0.060,
                    "riser_approach_tolerance_m": 0.010,
                    "crawl_goal_tolerance_m": 0.004,
                    "profile_substeps": 6,
                    "transition_clearance_m": 0.0065,
    "arch_clearance_m": 0.010,
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
