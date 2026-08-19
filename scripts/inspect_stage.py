"""Inspect the current Isaac Sim stage in a standalone session."""

from __future__ import annotations

import sys
from pathlib import Path

from isaacsim import SimulationApp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def print_stage_prims() -> None:
    """Print all prim paths currently present in the USD stage."""
    from pxr import Usd

    import omni.usd

    stage: Usd.Stage = omni.usd.get_context().get_stage()
    for prim in stage.Traverse():
        print(prim.GetPath())


def main() -> None:
    """Create the default MSSR world and print its USD prim paths."""
    simulation_app = SimulationApp({"headless": True})

    from worlds.basic_world import BasicWorldBuilder
    from robots.spherical_robot import SphericalRobotBuilder

    try:
        BasicWorldBuilder().build()
        SphericalRobotBuilder().build()
        simulation_app.update()
        print_stage_prims()
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
