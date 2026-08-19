"""Build static scenario obstacles in Isaac Sim."""

from __future__ import annotations

from worlds.scenario_config import BoxObstacleConfig


class ScenarioObstacleBuilder:
    """Create static collision obstacles from scenario configuration."""

    def __init__(self, obstacles: tuple[BoxObstacleConfig, ...]) -> None:
        """Store obstacle definitions to spawn."""
        self._obstacles = obstacles

    def build(self) -> None:
        """Create all configured static obstacles in the current USD stage."""
        import omni.usd
        from pxr import Gf, UsdGeom, UsdPhysics

        if not self._obstacles:
            return

        stage = omni.usd.get_context().get_stage()
        UsdGeom.Xform.Define(stage, "/World/Obstacles")

        for obstacle in self._obstacles:
            cube = UsdGeom.Cube.Define(stage, obstacle.prim_path)
            cube.CreateSizeAttr(1.0)
            cube.GetDisplayColorAttr().Set([Gf.Vec3f(*obstacle.color)])

            xformable = UsdGeom.Xformable(cube.GetPrim())
            xformable.AddTranslateOp().Set(Gf.Vec3d(*obstacle.position))
            xformable.AddScaleOp().Set(Gf.Vec3f(*obstacle.size))

            UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
