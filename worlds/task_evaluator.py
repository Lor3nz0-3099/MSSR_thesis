"""Task evaluation metrics for MSSR scenarios."""

from __future__ import annotations

import math
from dataclasses import dataclass

from graphs.robot_graph import RobotGraph
from robots.module_state import AttachmentStatus, ModuleState, Vector3
from robots.state_registry import RobotStateSnapshot
from worlds.scenario_config import ScenarioConfig


@dataclass(frozen=True)
class TaskEvaluation:
    """Task-level metrics computed from one simulation state."""

    scenario_name: str
    task_type: str
    step: int
    timestamp: float
    module_count: int
    connected_component_count: int
    largest_component_size: int
    assembled_ratio: float
    centroid_position: Vector3
    max_module_height: float
    fallen_module_count: int
    distance_to_goal: float | None
    is_goal_reached: bool
    is_assembled: bool
    is_success: bool
    is_timeout: bool
    is_done: bool

    def to_dict(self) -> dict[str, object]:
        """Convert the evaluation to a JSON-serializable dictionary."""
        return {
            "scenario_name": self.scenario_name,
            "task_type": self.task_type,
            "step": self.step,
            "timestamp": self.timestamp,
            "module_count": self.module_count,
            "connected_component_count": self.connected_component_count,
            "largest_component_size": self.largest_component_size,
            "assembled_ratio": self.assembled_ratio,
            "centroid_position": self.centroid_position,
            "max_module_height": self.max_module_height,
            "fallen_module_count": self.fallen_module_count,
            "distance_to_goal": self.distance_to_goal,
            "is_goal_reached": self.is_goal_reached,
            "is_assembled": self.is_assembled,
            "is_success": self.is_success,
            "is_timeout": self.is_timeout,
            "is_done": self.is_done,
        }


@dataclass(frozen=True)
class TaskEvaluatorConfig:
    """Thresholds used to decide task-level events."""

    assembled_ratio_threshold: float = 0.8


class TaskEvaluator:
    """Evaluate progress and termination conditions for one scenario.

    The evaluator deliberately computes reusable measurements instead of a
    single reward. Later, IL logging, deterministic experts, MARL rewards, and
    ablation metrics can all reuse the same task facts.
    """

    def __init__(
        self,
        scenario: ScenarioConfig,
        config: TaskEvaluatorConfig | None = None,
    ) -> None:
        """Store the scenario and evaluation thresholds."""
        self._scenario = scenario
        self._config = config or TaskEvaluatorConfig()

    def evaluate(
        self,
        snapshot: RobotStateSnapshot,
        graph: RobotGraph,
        step: int,
    ) -> TaskEvaluation:
        """Compute task metrics for one simulation step."""
        module_count = len(snapshot.modules)
        component_sizes = _connected_component_sizes(graph)
        largest_component_size = max(component_sizes, default=0)
        connected_component_count = len(component_sizes)
        assembled_ratio = largest_component_size / module_count if module_count else 0.0
        centroid_position = _centroid(snapshot.modules)
        max_module_height = max((module.pose.position[2] for module in snapshot.modules), default=0.0)
        fallen_module_count = _fallen_module_count(snapshot.modules)
        distance_to_goal = self._distance_to_goal(centroid_position)
        is_goal_reached = self._is_goal_reached(distance_to_goal)
        is_assembled = assembled_ratio >= self._config.assembled_ratio_threshold
        is_timeout = step >= self._scenario.episode_timeout_steps
        is_success = is_goal_reached and is_assembled
        is_done = is_success or is_timeout

        return TaskEvaluation(
            scenario_name=self._scenario.name,
            task_type=self._scenario.curriculum.task_type,
            step=step,
            timestamp=snapshot.timestamp,
            module_count=module_count,
            connected_component_count=connected_component_count,
            largest_component_size=largest_component_size,
            assembled_ratio=assembled_ratio,
            centroid_position=centroid_position,
            max_module_height=max_module_height,
            fallen_module_count=fallen_module_count,
            distance_to_goal=distance_to_goal,
            is_goal_reached=is_goal_reached,
            is_assembled=is_assembled,
            is_success=is_success,
            is_timeout=is_timeout,
            is_done=is_done,
        )

    def _distance_to_goal(self, centroid_position: Vector3) -> float | None:
        """Return distance from module centroid to the scenario goal."""
        if self._scenario.goal is None:
            return None
        return _distance(centroid_position, self._scenario.goal.position)

    def _is_goal_reached(self, distance_to_goal: float | None) -> bool:
        """Return whether the centroid is inside the scenario goal region."""
        if self._scenario.goal is None or distance_to_goal is None:
            return False
        return distance_to_goal <= self._scenario.goal.tolerance


def _connected_component_sizes(graph: RobotGraph) -> tuple[int, ...]:
    """Return connected-component sizes using only physically connected edges."""
    adjacency: dict[str, set[str]] = {node.module_id: set() for node in graph.nodes}
    for edge in graph.edges:
        if edge.status != AttachmentStatus.CONNECTED:
            continue
        adjacency.setdefault(edge.source, set()).add(edge.target)
        adjacency.setdefault(edge.target, set()).add(edge.source)

    visited: set[str] = set()
    sizes: list[int] = []
    for module_id in sorted(adjacency):
        if module_id in visited:
            continue
        stack = [module_id]
        size = 0
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            size += 1
            stack.extend(sorted(adjacency[current] - visited))
        sizes.append(size)
    return tuple(sorted(sizes, reverse=True))


def _centroid(modules: tuple[ModuleState, ...]) -> Vector3:
    """Return the centroid of all module positions."""
    if not modules:
        return (0.0, 0.0, 0.0)
    scale = 1.0 / len(modules)
    return (
        sum(module.pose.position[0] for module in modules) * scale,
        sum(module.pose.position[1] for module in modules) * scale,
        sum(module.pose.position[2] for module in modules) * scale,
    )


def _fallen_module_count(modules: tuple[ModuleState, ...]) -> int:
    """Return the number of modules that are implausibly below the ground contact height."""
    return sum(
        module.pose.position[2] < 0.5 * (module.radius or 0.0)
        for module in modules
    )


def _distance(first: Vector3, second: Vector3) -> float:
    """Return Euclidean distance between two 3D points."""
    return math.sqrt(
        (first[0] - second[0]) ** 2
        + (first[1] - second[1]) ** 2
        + (first[2] - second[2]) ** 2
    )
