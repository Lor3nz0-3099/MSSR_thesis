from __future__ import annotations

import argparse
from pathlib import Path

from smores_ep.isaac.physics_asset import build_physics_asset


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def main() -> None:
    root = repository_root()
    parser = argparse.ArgumentParser(
        description="Generate the SMORES-EP physics articulation USD"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            root
            / "assets/smores-ep/usd_physics/smores_ep_physics_v1.usd"
        ),
    )
    parser.add_argument(
        "--visual-reference",
        type=Path,
        default=Path("../usd_visual/smores_ep_usd_visual_v1.usd"),
        help="Reference authored into the output layer, normally relative",
    )
    args = parser.parse_args()
    output = build_physics_asset(args.output, args.visual_reference)
    print(f"Generated: {output}")


if __name__ == "__main__":
    main()

