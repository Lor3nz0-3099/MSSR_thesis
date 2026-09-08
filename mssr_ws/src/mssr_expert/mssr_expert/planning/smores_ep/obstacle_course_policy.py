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
        "snake8": frozenset({"assembly", "ramp", "gap", "stairs", "train"}),
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

    def step_index(self, task: str) -> int:
        """Return the unique index of one named course task."""

        name = str(task).strip()

        if not name:
            raise ValueError("Course task name must not be empty.")

        matches = [
            index
            for index, step in enumerate(self.steps())
            if step.task == name
        ]

        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one course task {name!r}; "
                f"found {len(matches)}."
            )

        return matches[0]

    def steps(self) -> tuple[CourseStep, ...]:
        """Return the complete morphology-aware task program."""
        snake = self.choose_morphology("ramp")
        manipulator = self.choose_morphology("button")
        rc_car = self.choose_morphology("exit")

        return (
            # Non-planar terrain starts directly in Snake8.  RC-Car8 is not
            # asked to negotiate the ramp after the physical validation
            # showed a morphology/breakover limitation rather than a control
            # or friction limitation.
            CourseStep("assembly", snake),
            CourseStep(
                "ramp_climb",
                snake,
                navigation="ramp_exit",
            ),
            CourseStep(
                "snake_gap_approach",
                snake,
                navigation="front_before_gap",
            ),
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
            CourseStep(
                "gap_clearance",
                snake,
                navigation="rear_past_gap",
            ),
            CourseStep(
                "stairs_approach",
                snake,
                navigation="front_before_stair_1",
            ),
            CourseStep(
                "stairs_crawl",
                snake,
                "crawl_stairs_spatial_concertina",
                {
                    "linear_m_s": 0.040,
                    "crawl_goal_tolerance_m": 0.012,
                    "path_corner_safety_m": 0.020,
                    "trajectory_step_m": 0.005,
                },
            ),
            CourseStep(
                "upper_deck_clearance",
                snake,
                navigation="front_on_upper_deck",
            ),
            # ------------------------------------------------------
            # CAMERA-GUIDED BUTTON SUB-EXPERT
            #
            # The button pose is not assumed fixed: x/y/z are consumed
            # from the live course/perception observation at runtime.
            # ------------------------------------------------------

            # First recover the fully steerable planar morphology.
            CourseStep(
                "button_rc_car_reconfiguration",
                rc_car,
            ),

            # RC-Car8 performs all x/y/yaw alignment.  Its final pose is
            # chosen for the *future* manipulator, so the MM8 arm/rear
            # side already faces the button after reconfiguration.
            CourseStep(
                "button_rc_car_pre_alignment",
                rc_car,
                navigation="button_pre_reconfiguration",
            ),

            # Deterministic RC-Car8 -> MobileManipulator8 transition.
            CourseStep(
                "button_reconfiguration",
                manipulator,
            ),

            # MM8 cannot steer in the validated posture.  It therefore
            # performs only signed longitudinal motion (normally reverse)
            # to reach the manipulation standoff.
            CourseStep(
                "button_mm8_approach",
                manipulator,
                navigation="button_mm8_pre_manipulation",
            ),

            # Put the base on the ground:
            # chassis_center, front_support, arm_ground_drive -> 0 rad.
            CourseStep(
                "button_manipulation_ready",
                manipulator,
                "prepare_manipulation",
            ),

            # Deliberate integration barrier.
            #
            # This stage will later contain:
            #   camera target -> IK -> pre-press -> press -> retract.
            # Until that controller exists it remains non-terminal.
            CourseStep(
                "button_press",
                manipulator,
                navigation="button_arm_press_pending",
                requires_button=True,
            ),

            # These stages are already defined so the complete FSM is
            # explicit.  They become reachable once BUTTON_ARM_PRESS is
            # implemented and physically validated.
            CourseStep(
                "button_restore_drive",
                manipulator,
                "restore_drive",
            ),
            CourseStep(
                "button_mm8_retreat",
                manipulator,
                navigation="button_mm8_retreat",
            ),

            # Return to the fully steerable morphology after the task.
            CourseStep(
                "button_return_rc_car",
                rc_car,
            ),

            CourseStep(
                "exit",
                rc_car,
                navigation="cross_exit",
                requires_goal=True,
            ),
            CourseStep(
                "exit_stop",
                rc_car,
                "stop",
            ),
        )
