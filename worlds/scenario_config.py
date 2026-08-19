"""Scenario configuration models for repeatable MSSR simulations."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from robots.spherical_robot import SphericalRobotConfig, create_indexed_spherical_robot_config

Vector3 = tuple[float, float, float]


@dataclass(frozen=True)
class CurriculumConfig:
    """Metadata used to order scenarios during curriculum learning."""

    task_type: str = "unspecified"
    stage: int = 0
    difficulty: float = 0.0


@dataclass(frozen=True)
class GoalConfig:
    """Task target used by planners, experts, and learning code."""

    name: str
    position: Vector3
    tolerance: float = 0.25


@dataclass(frozen=True)
class BoxObstacleConfig:
    """Static box obstacle spawned into the scenario."""

    name: str
    prim_path: str
    position: Vector3
    size: Vector3
    color: Vector3 = (0.55, 0.55, 0.55)


@dataclass(frozen=True)
class RandomSpawnConfig:
    """Generate module initial poses from a repeatable random area."""

    module_count: int
    x_range: tuple[float, float]
    y_range: tuple[float, float]
    z: float = 1.0
    min_distance: float = 1.45
    radius: float = 0.6
    mass: float = 5.0
    seed: int = 0
    world_frame_id: str = "odom"


@dataclass(frozen=True)
class ModuleScenarioConfig:
    """Spawn configuration for one spherical module in a scenario."""

    index: int
    module_id: str | None = None
    position: Vector3 = (0.0, 0.0, 1.0)
    radius: float = 0.6
    mass: float = 5.0
    color: Vector3 | None = None
    world_frame_id: str = "odom"


@dataclass(frozen=True)
class ScenarioConfig:
    """Configuration for a complete simulation scenario."""

    name: str
    modules: tuple[ModuleScenarioConfig, ...]
    attachment_mode: str = "continuous"
    reset_every_steps: int = 0
    curriculum: CurriculumConfig = CurriculumConfig()
    goal: GoalConfig | None = None
    box_obstacles: tuple[BoxObstacleConfig, ...] = ()
    episode_timeout_steps: int = 3600


def load_scenario_config(path: str | Path) -> ScenarioConfig:
    """Load a scenario configuration from a JSON file."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    schema_version = data.get("schema_version", "mssr.scenario.v1")
    if schema_version not in ("mssr.scenario.v1", "mssr.scenario.v2"):
        raise ValueError(f"Unsupported scenario schema_version: {schema_version}")

    modules = _modules_from_scenario_data(data)
    if not modules:
        raise ValueError("Scenario must define modules or random_spawn.")

    return ScenarioConfig(
        name=str(data.get("name", Path(path).stem)),
        modules=modules,
        attachment_mode=str(data.get("attachment_mode", "continuous")),
        reset_every_steps=int(data.get("reset_every_steps", 0)),
        curriculum=_curriculum_from_dict(data.get("curriculum", {})),
        goal=_optional_goal_from_dict(data.get("goal")),
        box_obstacles=tuple(
            _box_obstacle_from_dict(obstacle)
            for obstacle in data.get("box_obstacles", ())
        ),
        episode_timeout_steps=int(data.get("episode_timeout_steps", 3600)),
    )


def create_spherical_robot_configs(
    scenario: ScenarioConfig,
) -> tuple[SphericalRobotConfig, ...]:
    """Convert scenario module configs to concrete spherical robot configs."""
    return tuple(_robot_config_from_module(module) for module in scenario.modules)


def create_linear_scenario(
    module_count: int,
    module_spacing: float,
    attachment_mode: str,
    reset_every_steps: int = 0,
) -> ScenarioConfig:
    """Create the default line of modules used when no scenario file is given."""
    if module_count <= 0:
        raise ValueError("module_count must be greater than zero.")
    if module_spacing <= 0.0:
        raise ValueError("module_spacing must be greater than zero.")

    return ScenarioConfig(
        name="linear_modules",
        modules=tuple(
            ModuleScenarioConfig(
                index=index,
                position=(index * module_spacing, 0.0, 1.0),
            )
            for index in range(module_count)
        ),
        attachment_mode=attachment_mode,
        reset_every_steps=reset_every_steps,
        curriculum=CurriculumConfig(task_type="debug_line", stage=0, difficulty=0.0),
    )


def _modules_from_scenario_data(data: dict[str, Any]) -> tuple[ModuleScenarioConfig, ...]:
    """Parse explicit modules or generate repeatable random module poses."""
    explicit_modules = tuple(_module_from_dict(module) for module in data.get("modules", ()))
    random_spawn_data = data.get("random_spawn")

    if explicit_modules and random_spawn_data is not None:
        raise ValueError("Use either modules or random_spawn, not both.")

    if explicit_modules:
        return explicit_modules
    if random_spawn_data is None:
        return ()

    return _generate_random_modules(_random_spawn_from_dict(random_spawn_data))


def _random_spawn_from_dict(data: dict[str, Any]) -> RandomSpawnConfig:
    """Parse repeatable random spawn settings."""
    return RandomSpawnConfig(
        module_count=int(data["module_count"]),
        x_range=_range2(data["x_range"]),
        y_range=_range2(data["y_range"]),
        z=float(data.get("z", 1.0)),
        min_distance=float(data.get("min_distance", 1.45)),
        radius=float(data.get("radius", 0.6)),
        mass=float(data.get("mass", 5.0)),
        seed=int(data.get("seed", 0)),
        world_frame_id=str(data.get("world_frame_id", "odom")),
    )


def _generate_random_modules(config: RandomSpawnConfig) -> tuple[ModuleScenarioConfig, ...]:
    """Generate non-overlapping module spawn poses from a random area."""
    if config.module_count <= 0:
        raise ValueError("random_spawn.module_count must be greater than zero.")
    if config.min_distance <= 0.0:
        raise ValueError("random_spawn.min_distance must be greater than zero.")

    rng = random.Random(config.seed)
    positions: list[Vector3] = []
    max_attempts = max(200, config.module_count * 200)

    while len(positions) < config.module_count and max_attempts > 0:
        max_attempts -= 1
        candidate = (
            rng.uniform(config.x_range[0], config.x_range[1]),
            rng.uniform(config.y_range[0], config.y_range[1]),
            config.z,
        )
        if all(_planar_distance(candidate, position) >= config.min_distance for position in positions):
            positions.append(candidate)

    if len(positions) != config.module_count:
        raise ValueError("Could not place all random modules with the requested min_distance.")

    return tuple(
        ModuleScenarioConfig(
            index=index,
            module_id=f"sphere_{index}",
            position=position,
            radius=config.radius,
            mass=config.mass,
            world_frame_id=config.world_frame_id,
        )
        for index, position in enumerate(positions)
    )


def _curriculum_from_dict(data: dict[str, Any]) -> CurriculumConfig:
    """Parse curriculum metadata from a scenario dictionary."""
    return CurriculumConfig(
        task_type=str(data.get("task_type", "unspecified")),
        stage=int(data.get("stage", 0)),
        difficulty=float(data.get("difficulty", 0.0)),
    )


def _optional_goal_from_dict(data: dict[str, Any] | None) -> GoalConfig | None:
    """Parse an optional task goal."""
    if data is None:
        return None
    return GoalConfig(
        name=str(data.get("name", "goal")),
        position=_vector3(data["position"]),
        tolerance=float(data.get("tolerance", 0.25)),
    )


def _box_obstacle_from_dict(data: dict[str, Any]) -> BoxObstacleConfig:
    """Parse one static box obstacle entry."""
    name = str(data["name"])
    return BoxObstacleConfig(
        name=name,
        prim_path=str(data.get("prim_path", f"/World/Obstacles/{name}")),
        position=_vector3(data["position"]),
        size=_vector3(data["size"]),
        color=_vector3(data.get("color", (0.55, 0.55, 0.55))),
    )


def _module_from_dict(data: dict[str, Any]) -> ModuleScenarioConfig:
    """Parse one module scenario entry."""
    return ModuleScenarioConfig(
        index=int(data["index"]),
        module_id=str(data["module_id"]) if data.get("module_id") is not None else None,
        position=_vector3(data.get("position", (0.0, 0.0, 1.0))),
        radius=float(data.get("radius", 0.6)),
        mass=float(data.get("mass", 5.0)),
        color=_optional_vector3(data.get("color")),
        world_frame_id=str(data.get("world_frame_id", "odom")),
    )


def _robot_config_from_module(module: ModuleScenarioConfig) -> SphericalRobotConfig:
    """Create one concrete robot config from a scenario module entry."""
    config = create_indexed_spherical_robot_config(
        index=module.index,
        position=module.position,
        radius=module.radius,
        mass=module.mass,
        world_frame_id=module.world_frame_id,
    )
    return SphericalRobotConfig(
        module_id=module.module_id or config.module_id,
        prim_path=config.prim_path,
        body_frame_id=config.body_frame_id,
        world_frame_id=config.world_frame_id,
        radius=config.radius,
        mass=config.mass,
        position=config.position,
        color=module.color or config.color,
    )


def _optional_vector3(values: object) -> Vector3 | None:
    """Convert an optional JSON sequence to a 3D vector."""
    if values is None:
        return None
    return _vector3(values)


def _vector3(values: object) -> Vector3:
    """Convert a JSON sequence to a 3D vector."""
    sequence = list(values)  # type: ignore[arg-type]
    if len(sequence) != 3:
        raise ValueError("Expected a 3D vector.")
    return (float(sequence[0]), float(sequence[1]), float(sequence[2]))


def _range2(values: object) -> tuple[float, float]:
    """Convert a JSON sequence to a two-value inclusive range."""
    sequence = list(values)  # type: ignore[arg-type]
    if len(sequence) != 2:
        raise ValueError("Expected a two-value range.")
    start = float(sequence[0])
    end = float(sequence[1])
    if end <= start:
        raise ValueError("Range end must be greater than range start.")
    return (start, end)


def _planar_distance(first: Vector3, second: Vector3) -> float:
    """Return XY distance between two 3D positions."""
    return ((first[0] - second[0]) ** 2 + (first[1] - second[1]) ** 2) ** 0.5
