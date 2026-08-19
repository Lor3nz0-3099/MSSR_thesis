"""Minimal Isaac Sim 6.0 world used as the framework entry point."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BasicWorldConfig:
    """Configuration for the initial MSSR simulation world."""

    stage_units_in_meters: float = 1.0
    physics_scene_path: str = "/World/PhysicsScene"
    gravity_magnitude: float = 19.62
    physics_steps_per_second: int = 60
    minimum_simulation_frame_rate: int = 60
    ground_path: str = "/World/GroundPlane"
    light_path: str = "/World/DistantLight"
    dome_light_path: str = "/World/DomeLight"
    light_intensity: float = 1200.0
    dome_light_intensity: float = 500.0


class BasicWorldBuilder:
    """Builds a small reusable Isaac Sim world.

    This class owns only stage population. It does not launch Isaac Sim and it
    does not run the simulation loop, so it can later be reused by tests,
    planners, graph loaders, or reinforcement-learning environments.
    """

    def __init__(self, config: BasicWorldConfig | None = None) -> None:
        """Initialize the builder with immutable world settings."""
        self._config = config or BasicWorldConfig()

    def build(self) -> None:
        """Create a clean stage with physics, a ground plane, and lights.

        Isaac/Omniverse imports stay inside this method because standalone
        Isaac APIs must be imported after ``SimulationApp`` has been created.
        """
        import isaacsim.core.experimental.utils.stage as stage_utils
        from isaacsim.core.experimental.objects import DistantLight, DomeLight, GroundPlane

        stage_utils.create_new_stage()
        stage_utils.set_stage_units(meters_per_unit=self._config.stage_units_in_meters)

        self._create_physics_scene()

        GroundPlane(self._config.ground_path, positions=[0.0, 0.0, 0.0])

        distant_light = DistantLight(self._config.light_path)
        distant_light.set_intensities(self._config.light_intensity)

        dome_light = DomeLight(self._config.dome_light_path)
        dome_light.set_intensities(self._config.dome_light_intensity)

    def _create_physics_scene(self) -> None:
        """Create the USD physics scene used by PhysX simulation."""
        import carb
        import omni.usd
        from pxr import Gf, PhysxSchema, UsdPhysics

        stage = omni.usd.get_context().get_stage()
        physics_scene = UsdPhysics.Scene.Define(stage, self._config.physics_scene_path)
        physics_scene.CreateGravityDirectionAttr(Gf.Vec3f(0.0, 0.0, -1.0))
        physics_scene.CreateGravityMagnitudeAttr(self._config.gravity_magnitude)
        physx_scene = PhysxSchema.PhysxSceneAPI.Apply(physics_scene.GetPrim())
        physx_scene.GetTimeStepsPerSecondAttr().Set(self._config.physics_steps_per_second)
        carb.settings.get_settings().set(
            "/persistent/simulation/minFrameRate",
            self._config.minimum_simulation_frame_rate,
        )
