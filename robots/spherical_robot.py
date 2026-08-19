"""Spherical robot primitive for the first MSSR simulation experiments.

Legacy status: this is the FreeBOT prototype module representation, kept as
the seed of the future FreeBOT adapter (see
``context/LEGACY_SPHERE_MIGRATION.md``). SMORES-EP uses the CAD-derived
articulation in ``scripts/smores_ep`` instead of this sphere primitive.
"""

from __future__ import annotations

from dataclasses import dataclass

from robots.module_state import ModuleState, create_spherical_module_state


Vector3 = tuple[float, float, float]


@dataclass(frozen=True)
class SphericalRobotConfig:
    """Configuration for a single spherical robot instance."""

    module_id: str = "sphere_0"
    prim_path: str = "/World/Robots/SphericalRobot_0"
    body_frame_id: str = "base_link"
    world_frame_id: str = "odom"
    radius: float = 0.6
    mass: float = 5.0
    position: Vector3 = (0.0, 0.0, 1.0)
    color: Vector3 = (0.1, 0.35, 1.0)


@dataclass(frozen=True)
class SphericalRobot:
    """Runtime handle for the spherical robot prims."""

    module_id: str
    body_path: str
    body_frame_id: str
    world_frame_id: str
    radius: float
    mass: float
    initial_state: ModuleState


class SphericalRobotBuilder:
    """Builds a spherical robot with visual geometry, collision, and rigid body physics."""

    def __init__(self, config: SphericalRobotConfig | None = None) -> None:
        """Initialize the builder with immutable robot settings."""
        self._config = config or SphericalRobotConfig()

    def build(self) -> SphericalRobot:
        """Create the robot sphere and attach physics APIs.

        Isaac/Omniverse imports stay inside this method because standalone
        Isaac APIs must be imported after ``SimulationApp`` has been created.
        """
        import omni.usd
        from pxr import Gf, UsdGeom, UsdPhysics

        stage = omni.usd.get_context().get_stage()
        UsdGeom.Xform.Define(stage, "/World/Robots")

        sphere = UsdGeom.Sphere.Define(stage, self._config.prim_path)
        sphere.CreateRadiusAttr(self._config.radius)
        sphere.GetDisplayColorAttr().Set([Gf.Vec3f(*self._config.color)])
        self._create_visual_marker(stage, self._config.prim_path, self._config.radius)

        xformable = UsdGeom.Xformable(sphere.GetPrim())
        xformable.AddTranslateOp().Set(Gf.Vec3d(*self._config.position))

        UsdPhysics.CollisionAPI.Apply(sphere.GetPrim())
        UsdPhysics.RigidBodyAPI.Apply(sphere.GetPrim())

        mass_api = UsdPhysics.MassAPI.Apply(sphere.GetPrim())
        mass_api.CreateMassAttr(self._config.mass)

        initial_state = create_spherical_module_state(
            module_id=self._config.module_id,
            prim_path=self._config.prim_path,
            body_frame_id=self._config.body_frame_id,
            world_frame_id=self._config.world_frame_id,
            radius=self._config.radius,
            mass=self._config.mass,
            position=self._config.position,
        )
        return SphericalRobot(
            module_id=self._config.module_id,
            body_path=self._config.prim_path,
            body_frame_id=self._config.body_frame_id,
            world_frame_id=self._config.world_frame_id,
            radius=self._config.radius,
            mass=self._config.mass,
            initial_state=initial_state,
        )

    def _create_visual_marker(self, stage: object, prim_path: str, radius: float) -> None:
        """Create a small non-colliding marker so sphere rotation is visible."""
        from pxr import Gf, UsdGeom

        marker_path = f"{prim_path}/RollingMarker"
        marker = UsdGeom.Sphere.Define(stage, marker_path)
        marker.CreateRadiusAttr(radius * 0.12)
        marker.GetDisplayColorAttr().Set([Gf.Vec3f(1.0, 1.0, 1.0)])

        xformable = UsdGeom.Xformable(marker.GetPrim())
        xformable.AddTranslateOp().Set(Gf.Vec3d(radius * 0.78, 0.0, radius * 0.45))


def create_indexed_spherical_robot_config(
    index: int,
    position: Vector3,
    radius: float = 0.6,
    mass: float = 5.0,
    world_frame_id: str = "odom",
) -> SphericalRobotConfig:
    """Create a deterministic config for one spherical module in a swarm."""
    if index < 0:
        raise ValueError("Module index must be non-negative.")

    return SphericalRobotConfig(
        module_id=f"sphere_{index}",
        prim_path=f"/World/Robots/SphericalRobot_{index}",
        body_frame_id="base_link" if index == 0 else f"base_link_{index}",
        world_frame_id=world_frame_id,
        radius=radius,
        mass=mass,
        position=position,
        color=_indexed_color(index),
    )


def _indexed_color(index: int) -> Vector3:
    """Return a readable display color for a module index."""
    colors: tuple[Vector3, ...] = (
        (0.1, 0.35, 1.0),
        (0.95, 0.35, 0.15),
        (0.2, 0.75, 0.35),
        (0.85, 0.25, 0.7),
    )
    return colors[index % len(colors)]
