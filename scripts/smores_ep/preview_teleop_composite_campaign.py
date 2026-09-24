#!/usr/bin/env python3
"""Validate and export physical missions plus top views without starting Isaac."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_composite_course import (
    DEFAULT_SEED_CATALOG, EXPERT_SRC, ValidatedSeedCatalog,
    composite_obstacle_course, select_episode,
)
from smores_ep.isaac.course_geometry_audit import box_polygon, validate_teleop_course

DEFAULT_CAMPAIGN = EXPERT_SRC / "config/smores_teleop_composite_campaign13.json"


def draw_course(ax, course, name):
    from matplotlib.patches import Polygon, Circle
    colors = {"stair_test_riser": "#d99844", "flat_navigation_road": "#8596a3",
              "gap_test_near_bank": "#58aaa0", "gap_test_far_bank": "#58aaa0",
              "button_test_platform": "#b6a0d0", "button_support": "#523a66",
              "button": "#d63b45", "teleop_turn_pad": "#c8d8e5"}
    for box in course.boxes:
        ax.add_patch(Polygon(box_polygon(box), closed=True,
                             facecolor=colors.get(box.semantic, "#dae1e5"),
                             edgecolor="#55636d", linewidth=.35))
    for cone in course.navigation_cones:
        ax.add_patch(Circle(cone.center_xyz_m[:2], cone.radius_m, color="#ef6c24"))
    for i, task in enumerate(course.tasks[:-1], 1):
        p = task["parameters"]
        x, y, yaw = p["entry_pose_xyyaw"]
        ax.text(x, y, str(i), ha="center", va="center", fontsize=8,
                bbox={"facecolor": "white", "alpha": .85, "boxstyle": "circle,pad=.15", "edgecolor": "none"})
        if task["type"] == "flat_navigation":
            route = p["waypoints_xyyaw"]
            ax.plot([r[0] for r in route], [r[1] for r in route], "--", color="#245a7a", lw=.65)
        elif task["type"] == "gap":
            a, b = (p["gap"][k] for k in ("near_edge_center_world_xyz_m", "far_edge_center_world_xyz_m"))
            ax.annotate("", xy=b[:2], xytext=a[:2], arrowprops={"arrowstyle": "->", "color": "#bd3442", "lw": 1.1})
    for button in course.buttons:
        x, y = button.center_xyz_m[:2]; dx, dy = button.press_direction_world_xy
        ax.annotate("", xy=(x, y), xytext=(x-.5*dx, y-.5*dy),
                    arrowprops={"arrowstyle": "->", "color": "#a52332", "lw": 1})
    ax.plot(-1.65, 0, "o", color="#23794e", markersize=4)
    ax.plot(*course.goal_center_xyz_m[:2], "*", color="#be2837", markersize=9)
    abbreviations = {"flat_navigation": "RC", "gap": "G", "stairs": "S", "button": "B"}
    sequence = " > ".join(f"{abbreviations[t['type']]} {t['seed']}" for t in course.tasks[:-1])
    ax.set_title(f"{name} | z finale {course.final_floor_height_m:.3f} m\n{sequence}", fontsize=9)
    ax.autoscale_view(); ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X [m]", fontsize=7); ax.set_ylabel("Y [m]", fontsize=7)
    ax.tick_params(labelsize=6); ax.grid(alpha=.15)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, default=DEFAULT_CAMPAIGN)
    parser.add_argument("--seed-catalog", type=Path, default=DEFAULT_SEED_CATALOG)
    parser.add_argument("--output-dir", type=Path, default=Path("logs/teleop/course_previews/t01-t13-v1"))
    args = parser.parse_args(argv)
    campaign = json.loads(args.campaign.read_text())
    catalog = ValidatedSeedCatalog.load(args.seed_catalog)
    built = []
    for episode in campaign["episodes"]:
        mission = select_episode(campaign, episode["episode_id"])
        if mission.get("layout_profile") != "teleop_connected_v1":
            raise ValueError("Preview exporter requires teleop_connected_v1")
        course = composite_obstacle_course(mission, catalog.seeds_by_task_type)
        audit = validate_teleop_course(course)
        if not audit["valid"]:
            raise ValueError(f"{episode['episode_id']}: {audit['errors']}")
        built.append((mission, course, audit))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = (len(built) + 2) // 3
    overview, axes = plt.subplots(rows, 3, figsize=(18, 5*rows), squeeze=False)
    report = {"campaign": str(args.campaign), "episodes": [],
              "physical_rollouts_verified": False,
              "legend": "S stairs; G gap; RC navigation; B button. Green start, red star goal. Geometry only."}
    for index, (mission, course, audit) in enumerate(built):
        name = mission["episode_id"]
        observation = course.to_observation()
        digest = observation["geometry_sha256"]
        for suffix, data in (("mission", mission), ("course", observation)):
            (args.output_dir / f"{name}.{suffix}.json").write_text(json.dumps(data, indent=2)+"\n")
        fig, ax = plt.subplots(figsize=(12, 8))
        draw_course(ax, course, name)
        fig.tight_layout(); fig.savefig(args.output_dir / f"{name}.png", dpi=150); plt.close(fig)
        draw_course(axes.flat[index], course, name)
        report["episodes"].append({"episode_id": name, "geometry_sha256": digest, **audit})
    for ax in list(axes.flat)[len(built):]:
        ax.axis("off")
    overview.suptitle("T01–T13 | Geometria per teleoperazione — verifica dinamica ancora necessaria", fontsize=15)
    overview.tight_layout(rect=(0, 0, 1, .98))
    overview.savefig(args.output_dir / "overview.png", dpi=130)
    overview.savefig(args.output_dir / "overview.pdf")
    plt.close(overview)
    (args.output_dir / "geometry_audit.json").write_text(json.dumps(report, indent=2)+"\n")
    print(f"Validated and exported {len(built)} courses to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
