"""Library-driven high-level policy for composite SMORES-EP missions.

The policy owns only task semantics. Morphology capabilities remain in the
installed target graphs and executable behaviors remain in the morphology
behavior library, avoiding a second morphology registry in Python.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class TaskRequirement:
    """Capability and existing expert required by one semantic task type."""

    capability: str
    behavior: str | None = None
    execution_kind: str = "behavior"


@dataclass(frozen=True)
class MissionTask:
    """One ordered task supplied by the simulator course description."""

    task_id: str
    task_type: str
    parameters: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any], index: int) -> "MissionTask":
        task_type = str(payload.get("type", payload.get("task_type", ""))).strip()
        if not task_type:
            raise ValueError(f"Mission task {index} has no semantic type")
        task_id = str(payload.get("task_id", f"{task_type}-{index:02d}")).strip()
        if not task_id:
            raise ValueError(f"Mission task {index} has an empty task_id")
        parameters = payload.get("parameters", {})
        if not isinstance(parameters, Mapping):
            raise ValueError(f"Mission task {task_id!r} parameters must be an object")
        normalized_parameters = dict(parameters)
        if "seed" in payload:
            normalized_parameters.setdefault("seed", int(payload["seed"]))
        return cls(task_id, task_type, normalized_parameters)


@dataclass(frozen=True)
class CourseStep:
    """Resolved high-level decision for a single mission task."""

    task: str
    task_type: str
    capability: str
    morphology: str
    behavior: str | None = None
    execution_kind: str = "behavior"
    parameters: Mapping[str, Any] = field(default_factory=dict)
    navigation: str | None = None
    requires_button: bool = False
    requires_goal: bool = False


@dataclass(frozen=True)
class ValidatedSeedCatalog:
    """Allowlist backed by the canonical expert dataset manifest."""

    seeds_by_task_type: Mapping[str, frozenset[int]]

    @classmethod
    def load(cls, path: Path) -> "ValidatedSeedCatalog":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != "mssr.composite_seed_catalog.v1":
            raise ValueError("Unsupported composite seed catalog schema")
        raw = payload.get("validated_seeds")
        if not isinstance(raw, Mapping):
            raise ValueError("Composite seed catalog has no validated_seeds")
        result: dict[str, frozenset[int]] = {}
        for task_type, values in raw.items():
            if not isinstance(values, list) or not values:
                raise ValueError(f"Seed list for {task_type!r} is empty")
            seeds = frozenset(int(value) for value in values)
            if len(seeds) != len(values) or min(seeds) < 0:
                raise ValueError(f"Seed list for {task_type!r} is invalid")
            result[str(task_type)] = seeds
        return cls(result)

    def require(self, task_type: str, seed: int) -> None:
        allowed = self.seeds_by_task_type.get(task_type, frozenset())
        if int(seed) not in allowed:
            raise ValueError(
                f"Seed {seed} is not validated for task type {task_type!r}"
            )


class ObstacleCoursePolicy:
    """Select morphologies directly from target-graph capabilities."""

    TASK_REQUIREMENTS: Mapping[str, TaskRequirement] = {
        "flat_navigation": TaskRequirement(
            "flat_navigation", execution_kind="nav2"
        ),
        "gap": TaskRequirement("cross_gap", "gap_crossing"),
        "stairs": TaskRequirement(
            "climb_stairs", "crawl_stairs_spatial_concertina"
        ),
        "button": TaskRequirement(
            "press_button", execution_kind="button_expert"
        ),
        "goal": TaskRequirement(
            "flat_navigation", execution_kind="nav2_goal"
        ),
    }

    def __init__(self, morphology_capabilities: Mapping[str, Iterable[str]]) -> None:
        normalized: dict[str, frozenset[str]] = {}
        for morphology, capabilities in morphology_capabilities.items():
            name = str(morphology).strip()
            values = frozenset(str(item).strip() for item in capabilities)
            if not name or not values or "" in values:
                raise ValueError("Morphology capability catalog is malformed")
            normalized[name] = values
        if not normalized:
            raise ValueError("Morphology capability catalog is empty")
        self._capabilities = normalized

    @classmethod
    def from_morphology_catalog(
        cls,
        catalog: Mapping[str, Any],
    ) -> "ObstacleCoursePolicy":
        capabilities: dict[str, tuple[str, ...]] = {}
        for name, graph in catalog.items():
            attributes = getattr(graph, "global_attributes", None)
            if not isinstance(attributes, Mapping):
                raise ValueError(f"Morphology {name!r} has no global attributes")
            raw = attributes.get("capabilities", ())
            if not isinstance(raw, (list, tuple, set, frozenset)):
                raise ValueError(f"Morphology {name!r} capabilities must be a sequence")
            capabilities[str(name)] = tuple(str(item) for item in raw)
        return cls(capabilities)

    @property
    def supported_task_types(self) -> tuple[str, ...]:
        return tuple(self.TASK_REQUIREMENTS)

    def choose_morphology(
        self,
        required_capability: str,
        current_morphology: str | None = None,
    ) -> str:
        capability = str(required_capability).strip()
        if not capability:
            raise ValueError("Required capability must not be empty")
        if (
            current_morphology in self._capabilities
            and capability in self._capabilities[current_morphology]
        ):
            return str(current_morphology)
        candidates = sorted(
            morphology
            for morphology, capabilities in self._capabilities.items()
            if capability in capabilities
        )
        if not candidates:
            raise ValueError(f"No target morphology provides {capability!r}")
        return candidates[0]

    def resolve(
        self,
        task: MissionTask,
        current_morphology: str | None = None,
    ) -> CourseStep:
        try:
            requirement = self.TASK_REQUIREMENTS[task.task_type]
        except KeyError as error:
            raise ValueError(f"Unsupported mission task type {task.task_type!r}") from error
        morphology = self.choose_morphology(
            requirement.capability,
            current_morphology=current_morphology,
        )
        return CourseStep(
            task=task.task_id,
            task_type=task.task_type,
            capability=requirement.capability,
            morphology=morphology,
            behavior=requirement.behavior,
            execution_kind=requirement.execution_kind,
            parameters={
                **(
                    {"crawl_goal_tolerance_m": 0.016}
                    if task.task_type == "stairs"
                    else {}
                ),
                **dict(task.parameters),
            },
        )

    def plan(
        self,
        tasks: Sequence[MissionTask | Mapping[str, Any]],
        current_morphology: str | None = None,
    ) -> tuple[CourseStep, ...]:
        if not tasks:
            raise ValueError("Composite mission must contain at least one task")
        seen_ids: set[str] = set()
        steps: list[CourseStep] = []
        selected = current_morphology
        for index, raw in enumerate(tasks):
            task = raw if isinstance(raw, MissionTask) else MissionTask.from_mapping(raw, index)
            if task.task_id in seen_ids:
                raise ValueError(f"Duplicate mission task_id {task.task_id!r}")
            seen_ids.add(task.task_id)
            step = self.resolve(task, current_morphology=selected)
            steps.append(step)
            selected = step.morphology
        if steps[-1].task_type != "goal":
            raise ValueError("Composite mission must terminate with a goal task")
        return tuple(steps)

    def steps(self) -> tuple[CourseStep, ...]:
        """Return the legacy fixed-course expansion without the ramp.

        New composite runs use :meth:`plan` with simulator-provided tasks.
        This method keeps the existing ROS entry point operational while it
        is migrated to the task-driven executor.
        """

        snake = self.choose_morphology("cross_gap")
        manipulator = self.choose_morphology("press_button")
        rc_car = self.choose_morphology("flat_navigation")
        stair_parameters = {
            "linear_m_s": 0.040,
            "crawl_goal_tolerance_m": 0.016,
            "path_corner_safety_m": 0.020,
            "trajectory_step_m": 0.005,
        }
        return (
            CourseStep("assembly", "assembly", "flat_navigation", rc_car),
            CourseStep(
                "gap_approach", "gap", "cross_gap", snake,
                navigation="front_before_gap",
            ),
            CourseStep(
                "gap_crossing", "gap", "cross_gap", snake,
                "gap_crossing", parameters={
                    "linear_m_s": 0.040,
                    "approach_linear_m_s": 0.050,
                    "gap_goal_tolerance_m": 0.004,
                },
            ),
            CourseStep(
                "gap_clearance", "gap", "cross_gap", snake,
                navigation="rear_past_gap",
            ),
            CourseStep(
                "stairs_approach", "stairs", "climb_stairs", snake,
                navigation="front_before_stair_1",
            ),
            CourseStep(
                "stairs_crawl", "stairs", "climb_stairs", snake,
                "crawl_stairs_spatial_concertina",
                parameters=stair_parameters,
            ),
            CourseStep(
                "upper_deck_clearance", "stairs", "climb_stairs", snake,
                navigation="front_on_upper_deck",
            ),
            CourseStep(
                "button_rc_car_reconfiguration", "button", "flat_navigation", rc_car,
            ),
            CourseStep(
                "button_rc_car_pre_alignment", "button", "flat_navigation", rc_car,
                navigation="button_pre_reconfiguration",
            ),
            CourseStep(
                "button_reconfiguration", "button", "press_button", manipulator,
            ),
            CourseStep(
                "button_mm8_approach", "button", "press_button", manipulator,
                navigation="button_mm8_pre_manipulation",
            ),
            CourseStep(
                "button_manipulation_ready", "button", "press_button", manipulator,
                "prepare_manipulation",
            ),
            CourseStep(
                "button_press", "button", "press_button", manipulator,
                navigation="button_arm_press_pending", requires_button=True,
            ),
            CourseStep(
                "button_restore_drive", "button", "press_button", manipulator,
                "restore_drive",
            ),
            CourseStep(
                "button_mm8_retreat", "button", "press_button", manipulator,
                navigation="button_mm8_retreat",
            ),
            CourseStep(
                "button_return_rc_car", "button", "flat_navigation", rc_car,
            ),
            CourseStep(
                "exit", "goal", "flat_navigation", rc_car,
                navigation="cross_exit", requires_goal=True,
            ),
            CourseStep(
                "exit_stop", "goal", "flat_navigation", rc_car, "stop",
            ),
        )

    def step_index(self, task: str) -> int:
        matches = [
            index for index, step in enumerate(self.steps())
            if step.task == str(task).strip()
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one course task {task!r}; found {len(matches)}"
            )
        return matches[0]

    @staticmethod
    def requires_reconfiguration(
        current_morphology: str | None,
        selected_morphology: str,
    ) -> bool:
        return current_morphology != selected_morphology
