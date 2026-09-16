#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

# Workstation compatibility:
# Ubuntu Matplotlib 3.5.1 was compiled against Ubuntu NumPy 1.x.
# run.sh invokes /usr/bin/python3 -S and this path exposes the distro stack.
DIST_PACKAGES = "/usr/lib/python3/dist-packages"
if DIST_PACKAGES not in sys.path:
    sys.path.insert(0, DIST_PACKAGES)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SNAKE_ORDER = [
    "smores_05", "smores_06", "smores_04", "smores_01",
    "smores_07", "smores_03", "smores_02", "smores_08",
]

SHORT = {
    "snake_tail": "T",
    "snake_rear": "R",
    "snake_hip": "H",
    "snake_center_rear": "CR",
    "snake_center_front": "CF",
    "snake_shoulder": "S",
    "snake_neck": "N",
    "snake_head": "HEAD",
    "chassis_left": "CL",
    "chassis_center_left": "CCL",
    "chassis_center_right": "CCR",
    "chassis_right": "CR",
    "wheel_left_front": "WLF",
    "wheel_left_rear": "WLR",
    "wheel_right_front": "WRF",
    "wheel_right_rear": "WRR",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("datasets/expert_v1_compact"))
    p.add_argument("--out", type=Path, default=Path("figures/expert_dataset_v1"))
    p.add_argument("--stairs-seed", type=int, default=3101)
    p.add_argument("--gap-seed", type=int, default=4107)
    p.add_argument("--rc-car-seed", type=int, default=5101)
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


def read_json(path: Path):
    return json.loads(path.read_text())


def read_rows(path: Path):
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def positions(graph):
    out = {}
    for node in graph.get("nodes", []):
        if not isinstance(node, dict):
            continue
        mid = node.get("module_id") or node.get("id") or node.get("name")
        attrs = node.get("attributes", {})
        p = attrs.get("position")
        if mid is not None and isinstance(p, list) and len(p) >= 3:
            out[str(mid)] = tuple(float(v) for v in p[:3])
    if len(out) != 8:
        raise RuntimeError(f"Expected 8 module positions, got {len(out)}")
    return out


def topology(graph):
    out = []
    for edge in graph.get("edges", []):
        a = edge.get("module_a_id")
        b = edge.get("module_b_id")
        if a is None or b is None:
            raise RuntimeError(f"Unexpected edge schema: {sorted(edge.keys())}")
        out.append(tuple(sorted((str(a), str(b)))))
    return tuple(sorted(out))


def centroid(graph):
    ps = positions(graph)
    return tuple(sum(p[i] for p in ps.values()) / len(ps) for i in range(3))


def t_start(row):
    return float(row["_source_stamp_start"])


def t_end(row):
    return float(row["_source_stamp_end"])


def load_episode(root: Path, task: str, seed: int):
    d = root / "episodes" / task / f"seed-{seed:06d}"
    if not d.is_dir():
        raise RuntimeError(f"Episode not found: {task}/{seed}")
    em = read_json(d / "manifest.json")
    am = read_json(d / "assembly_manifest.json")
    bm = read_json(d / "behavior_manifest.json")
    ar = read_rows(root / am["compact"]["path"])
    br = read_rows(root / bm["compact"]["path"])
    return em, am, bm, ar, br


def save(fig, out: Path, relative: str, dpi: int, created: list[str]):
    base = out / relative
    base.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        path = base.with_suffix("." + ext)
        kwargs = {"bbox_inches": "tight"}
        if ext == "png":
            kwargs["dpi"] = dpi
        fig.savefig(path, **kwargs)
        created.append(str(path.relative_to(out)))
    plt.close(fig)


def select_rows(rows, task: str, n: int = 8):
    centers = [centroid(row["graph_t"]) for row in rows]
    if task in ("stairs", "gap"):
        metric = [c[0] for c in centers]
    else:
        metric = [0.0]
        for a, b in zip(centers, centers[1:]):
            metric.append(metric[-1] + math.hypot(b[0] - a[0], b[1] - a[1]))

    lo, hi = metric[0], metric[-1]
    if abs(hi - lo) < 1e-12:
        idx = [round(i * (len(rows) - 1) / (n - 1)) for i in range(n)]
    else:
        targets = [lo + (hi - lo) * i / (n - 1) for i in range(n)]
        idx = [
            min(range(len(metric)), key=lambda j: abs(metric[j] - target))
            for target in targets
        ]
    return list(dict.fromkeys([0] + idx + [len(rows) - 1]))


def draw_stairs(ax, course, xmin, xmax):
    s = course["scenario"]
    x0 = float(s["first_riser_x_m"])
    rise = float(s["rise_m"])
    tread = float(s["tread_depth_m"])
    count = int(s["step_count"])
    ax.plot([xmin, x0], [0.0, 0.0], linewidth=2.0)
    for k in range(1, count + 1):
        xl = x0 + (k - 1) * tread
        xr = x0 + k * tread
        z0 = (k - 1) * rise
        z1 = k * rise
        ax.plot([xl, xl], [z0, z1], linewidth=2.0)
        ax.plot([xl, xr], [z1, z1], linewidth=2.0)
    ax.plot(
        [x0 + count * tread, xmax],
        [count * rise, count * rise],
        linewidth=2.0,
    )


def draw_gap(ax, course, xmin, xmax):
    g = course["gap"]
    near = float(g["near_edge_x_m"])
    far = float(g["far_edge_x_m"])
    ax.plot([xmin, near], [0.0, 0.0], linewidth=2.0)
    ax.plot([far, xmax], [0.0, 0.0], linewidth=2.0)
    ax.axvline(near, linestyle="--", linewidth=1.0, alpha=0.6)
    ax.axvline(far, linestyle="--", linewidth=1.0, alpha=0.6)
    ax.text(
        (near + far) / 2.0,
        -0.008,
        f"{(far - near) * 100:.1f} cm gap",
        ha="center",
        va="top",
        fontsize=8,
    )


def assembly_states(episode_manifest, assembly, behavior):
    states = []
    previous = None
    for index, row in enumerate(assembly):
        graph = row["graph_t"]
        current = topology(graph)
        if current != previous:
            states.append(
                {
                    "graph": graph,
                    "time": t_start(row),
                    "label": f"{len(current)} edges",
                    "source_row": index,
                }
            )
            previous = current

    last_index = len(assembly) - 1
    last_graph = assembly[-1]["graph_t"]
    if states[-1]["source_row"] != last_index:
        states.append(
            {
                "graph": last_graph,
                "time": t_start(assembly[-1]),
                "label": f"{len(topology(last_graph))} edges (final assembly pose)",
                "source_row": last_index,
            }
        )

    terminal = assembly[-1]["graph_t_plus_1"]
    if topology(terminal) != topology(states[-1]["graph"]):
        states.append(
            {
                "graph": terminal,
                "time": t_end(assembly[-1]),
                "label": f"{len(topology(terminal))} edges (assembly terminal)",
                "source_row": None,
            }
        )

    boundary = episode_manifest["assembly_behavior_boundary"]
    if boundary["kind"] == "single_expected_late_edge":
        graph = behavior[0]["graph_t"]
        if topology(graph) != topology(states[-1]["graph"]):
            states.append(
                {
                    "graph": graph,
                    "time": t_start(behavior[0]),
                    "label": "7 edges (behavior start)",
                    "source_row": None,
                }
            )
    return states


def plot_assembly(task, seed, em, bm, assembly, behavior, out, dpi, created):
    roles = bm["top_level_constants"]["module_roles"]
    states = assembly_states(em, assembly, behavior)

    count = len(states)
    cols = min(4, count)
    nrows = math.ceil(count / cols)
    fig, axes = plt.subplots(
        nrows, cols, figsize=(4.2 * cols, 3.9 * nrows), squeeze=False
    )
    flat = [ax for row in axes for ax in row]

    allp = [
        p
        for state in states
        for p in positions(state["graph"]).values()
    ]
    xmin, xmax = min(p[0] for p in allp) - 0.08, max(p[0] for p in allp) + 0.08
    ymin, ymax = min(p[1] for p in allp) - 0.08, max(p[1] for p in allp) + 0.08

    for ax, state in zip(flat, states):
        graph = state["graph"]
        ps = positions(graph)
        for a, b in topology(graph):
            ax.plot([ps[a][0], ps[b][0]], [ps[a][1], ps[b][1]], linewidth=1.5)

        for mid, p in ps.items():
            role = roles.get(mid, mid)
            ax.scatter([p[0]], [p[1]], s=75, zorder=3)
            ax.text(
                p[0], p[1], SHORT.get(role, mid.replace("smores_", "")),
                ha="center", va="center", fontsize=7,
            )

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.grid(alpha=0.2)
        ax.set_title(f"t={state['time']:.1f}s — {state['label']}", fontsize=9)

    for ax in flat[len(states):]:
        ax.axis("off")

    fig.suptitle(f"{task.upper()} seed {seed} — self-assembly evolution", fontsize=14)
    fig.tight_layout()
    save(fig, out, f"{task}_{seed}/{task}_{seed}_assembly_xy", dpi, created)


def plot_snake_shape(task, seed, bm, behavior, out, dpi, created):
    course = bm["observation_constants"]["course"]
    chosen = select_rows(behavior, task, n=8)
    allp = [
        p
        for row in behavior
        for p in positions(row["graph_t"]).values()
    ]
    xmin = min(p[0] for p in allp) - 0.08
    xmax = max(p[0] for p in allp) + 0.08
    zmax = max(p[2] for p in allp) + 0.05

    fig, ax = plt.subplots(figsize=(12, 5.7))
    if task == "stairs":
        draw_stairs(ax, course, xmin, xmax)
    else:
        draw_gap(ax, course, xmin, xmax)

    for index in chosen:
        row = behavior[index]
        graph = row["graph_t"]
        ps = positions(graph)
        for a, b in topology(graph):
            ax.plot(
                [ps[a][0], ps[b][0]],
                [ps[a][2], ps[b][2]],
                linewidth=1.6,
                alpha=0.72,
            )
        ax.scatter(
            [p[0] for p in ps.values()],
            [p[2] for p in ps.values()],
            s=22,
            alpha=0.8,
        )
        head = ps["smores_08"]
        ax.text(
            head[0],
            head[2] + 0.008,
            f"{t_start(row):.0f}s",
            ha="center",
            fontsize=7,
        )

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-0.025, zmax)
    ax.set_xlabel("Longitudinal position x [m]")
    ax.set_ylabel("Module center height z [m]")
    ax.grid(alpha=0.2)
    title = (
        "Snake8 stair-climbing shape evolution"
        if task == "stairs"
        else "Snake8 gap-crossing shape evolution"
    )
    ax.set_title(f"{title} — seed {seed}\nHead labels indicate simulation time")
    fig.tight_layout()
    save(fig, out, f"{task}_{seed}/{task}_{seed}_behavior_shape_xz", dpi, created)


def plot_snake_time(task, seed, bm, behavior, out, dpi, created):
    roles = bm["top_level_constants"]["module_roles"]
    fig, ax = plt.subplots(figsize=(12, 6))

    for mid in SNAKE_ORDER:
        times, values = [], []
        for row in behavior:
            value = positions(row["graph_t"])[mid][2 if task == "stairs" else 0]
            times.append(t_start(row))
            values.append(value)
            if t_end(row) > t_start(row):
                times.append(t_end(row))
                values.append(value)
        ax.plot(times, values, linewidth=1.3, label=roles[mid])

    if task == "stairs":
        for h in bm["observation_constants"]["course"]["stairs"]["top_heights_m"]:
            ax.axhline(float(h), linestyle=":", linewidth=0.8, alpha=0.4)
        ylabel = "Module center height z [m]"
        title = f"Snake8 module height propagation on stairs — seed {seed}"
        filename = f"stairs_{seed}/stairs_{seed}_behavior_height_vs_time"
    else:
        gap = bm["observation_constants"]["course"]["gap"]
        ax.axhline(float(gap["near_edge_x_m"]), linestyle="--", linewidth=1.2, label="Near edge")
        ax.axhline(float(gap["far_edge_x_m"]), linestyle="--", linewidth=1.2, label="Far edge")
        ylabel = "Longitudinal position x [m]"
        title = f"Snake8 module progression across the gap — seed {seed}"
        filename = f"gap_{seed}/gap_{seed}_behavior_crossing_vs_time"

    ax.set_xlabel("Simulation time [s]")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    save(fig, out, filename, dpi, created)


def plot_rc(
    seed,
    bm,
    behavior,
    environment_geometry,
    out,
    dpi,
    created,
):
    layout = environment_geometry["layout"]
    roles = bm["top_level_constants"]["module_roles"]

    centerline = [
        (float(p[0]), float(p[1]))
        for p in layout["centerline_xy_m"]
    ]

    road_width = float(layout["corridor_width_m"])
    half_width = 0.5 * road_width

    # --------------------------------------------------------
    # Build left/right physical road boundaries from the
    # sampled centerline using the local normal direction.
    # --------------------------------------------------------
    left = []
    right = []

    for i, (x, y) in enumerate(centerline):
        if i == 0:
            ax_, ay_ = centerline[0]
            bx_, by_ = centerline[1]
        elif i == len(centerline) - 1:
            ax_, ay_ = centerline[-2]
            bx_, by_ = centerline[-1]
        else:
            ax_, ay_ = centerline[i - 1]
            bx_, by_ = centerline[i + 1]

        dx = bx_ - ax_
        dy = by_ - ay_
        norm = max(math.hypot(dx, dy), 1.0e-12)

        nx = -dy / norm
        ny = dx / norm

        left.append(
            (x + half_width * nx,
             y + half_width * ny)
        )
        right.append(
            (x - half_width * nx,
             y - half_width * ny)
        )

    road_polygon = left + list(reversed(right))

    centers = [
        centroid(row["graph_t"])
        for row in behavior
    ]

    # ========================================================
    # MAIN RC-CAR ENVIRONMENT FIGURE
    # ========================================================

    fig, ax = plt.subplots(figsize=(12, 7.5))

    # Physical road surface.
    ax.fill(
        [p[0] for p in road_polygon],
        [p[1] for p in road_polygon],
        alpha=0.12,
        label=f"Physical road corridor ({road_width:.2f} m)",
        zorder=0,
    )

    ax.plot(
        [p[0] for p in left],
        [p[1] for p in left],
        linewidth=1.2,
        alpha=0.7,
    )

    ax.plot(
        [p[0] for p in right],
        [p[1] for p in right],
        linewidth=1.2,
        alpha=0.7,
    )

    ax.plot(
        [p[0] for p in centerline],
        [p[1] for p in centerline],
        linestyle="--",
        linewidth=1.3,
        label="Physical road centerline",
        zorder=1,
    )

    # Assembly/start pad.
    pad = layout.get("start_pad_bounds_xy_m")

    if pad and len(pad) == 4:
        x0, x1, y0, y1 = map(float, pad)

        px = [x0, x1, x1, x0, x0]
        py = [y0, y0, y1, y1, y0]

        ax.plot(
            px,
            py,
            linestyle=":",
            linewidth=1.2,
            label="Assembly / start pad",
            zorder=1,
        )

    # Physical cones.
    cones = [
        (float(p[0]), float(p[1]))
        for p in layout["cone_centers_xy_m"]
    ]

    radius = float(layout["cone_radius_m"])

    if cones:
        ax.scatter(
            [p[0] for p in cones],
            [p[1] for p in cones],
            marker="^",
            s=80,
            label="Traffic cones",
            zorder=5,
        )

        for index, (x, y) in enumerate(cones):
            circle = plt.Circle(
                (x, y),
                radius,
                fill=False,
                linewidth=1.2,
                alpha=0.8,
            )

            ax.add_patch(circle)

            ax.text(
                x,
                y + radius + 0.025,
                f"C{index + 1}",
                fontsize=7,
                ha="center",
            )

    # Finish line perpendicular to road heading.
    gx = float(layout["finish_x_m"])
    gy = float(layout["finish_y_m"])
    yaw = float(layout["finish_yaw_rad"])

    nx = -math.sin(yaw)
    ny = math.cos(yaw)

    finish_a = (
        gx + half_width * nx,
        gy + half_width * ny,
    )

    finish_b = (
        gx - half_width * nx,
        gy - half_width * ny,
    )

    ax.plot(
        [finish_a[0], finish_b[0]],
        [finish_a[1], finish_b[1]],
        linewidth=3,
        label="Finish line",
        zorder=4,
    )

    # Actual vehicle trajectory from graph observations.
    ax.plot(
        [c[0] for c in centers],
        [c[1] for c in centers],
        linewidth=2.2,
        label="Measured RC-Car8 centroid",
        zorder=4,
    )

    # Selected physical morphology snapshots.
    selected = select_rows(
        behavior,
        "rc_car",
        n=6,
    )

    for index in selected:
        row = behavior[index]
        graph = row["graph_t"]
        ps = positions(graph)

        for a, b in topology(graph):
            ax.plot(
                [ps[a][0], ps[b][0]],
                [ps[a][1], ps[b][1]],
                linewidth=1.2,
                alpha=0.55,
                zorder=3,
            )

        ax.scatter(
            [p[0] for p in ps.values()],
            [p[1] for p in ps.values()],
            s=18,
            alpha=0.75,
            zorder=4,
        )

        c = centroid(graph)

        ax.text(
            c[0],
            c[1] + 0.055,
            f"{t_start(row):.0f}s",
            fontsize=7,
            ha="center",
            zorder=6,
        )

    # Limits based on relevant physical geometry rather than
    # the much larger supporting platform.
    xs = (
        [p[0] for p in left]
        + [p[0] for p in right]
        + [p[0] for p in cones]
        + [c[0] for c in centers]
    )

    ys = (
        [p[1] for p in left]
        + [p[1] for p in right]
        + [p[1] for p in cones]
        + [c[1] for c in centers]
    )

    margin = 0.18

    ax.set_xlim(
        min(xs) - margin,
        max(xs) + margin,
    )

    ax.set_ylim(
        min(ys) - margin,
        max(ys) + margin,
    )

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)

    ax.set_title(
        "RC-Car8 navigation through physical stage "
        f"— seed {seed}\n"
        f"{layout['track_profile']}, "
        f"{layout['cone_count']} cones"
    )

    ax.legend(
        fontsize=8,
        loc="best",
    )

    fig.tight_layout()

    save(
        fig,
        out,
        f"rc_car_{seed}/rc_car_{seed}_behavior_xy",
        dpi,
        created,
    )

    # ========================================================
    # SECONDARY DIAGNOSTIC:
    # individual module trajectories.
    # ========================================================

    fig, ax = plt.subplots(figsize=(11, 7))

    for mid, role in sorted(
        roles.items(),
        key=lambda item: item[1],
    ):
        xs = []
        ys = []

        for row in behavior:
            p = positions(row["graph_t"])[mid]
            xs.append(p[0])
            ys.append(p[1])

        ax.plot(
            xs,
            ys,
            linewidth=1.25,
            label=role,
        )

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)

    ax.legend(
        fontsize=8,
        ncol=2,
    )

    ax.set_title(
        "RC-Car8 individual module trajectories "
        f"— seed {seed}\n"
        "Diagnostic of morphology coherence during motion"
    )

    fig.tight_layout()

    save(
        fig,
        out,
        f"rc_car_{seed}/rc_car_{seed}_behavior_xy_by_module",
        dpi,
        created,
    )

def plot_summary(selected_data, out, dpi, created):
    tasks = ["stairs", "gap", "rc_car"]
    labels = ["Stairs 3101", "Gap 4107", "RC-Car 5101"]
    assembly_duration, behavior_duration = [], []
    logical, compact, categories = [], [], []

    for task, label in zip(tasks, labels):
        assembly = selected_data[task]["assembly"]
        behavior = selected_data[task]["behavior"]

        assembly_duration.append(t_end(assembly[-1]) - t_start(assembly[0]))
        behavior_duration.append(t_end(behavior[-1]) - t_start(behavior[0]))

        categories.extend([label + "\nAssembly", label + "\nBehavior"])
        logical.extend([
            sum(int(r["_source_repeat_count"]) for r in assembly),
            sum(int(r["_source_repeat_count"]) for r in behavior),
        ])
        compact.extend([len(assembly), len(behavior)])

    width = 0.36
    x = list(range(3))
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar([v - width/2 for v in x], assembly_duration, width, label="Assembly")
    ax.bar([v + width/2 for v in x], behavior_duration, width, label="Behavior")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Duration [s]")
    ax.set_title("Selected expert episode durations")
    ax.grid(axis="y", alpha=0.2)
    ax.legend()
    fig.tight_layout()
    save(fig, out, "summary/selected_episodes_duration", dpi, created)

    x = list(range(6))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar([v - width/2 for v in x], logical, width, label="Logical/source records")
    ax.bar([v + width/2 for v in x], compact, width, label="Stored compact records")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=8)
    ax.set_ylabel("Records")
    ax.set_title("Logical samples versus compact RLE storage")
    ax.grid(axis="y", alpha=0.2)
    ax.legend()
    fig.tight_layout()
    save(fig, out, "summary/selected_episodes_rle", dpi, created)


def main():
    args = parse_args()
    root = args.root.resolve()
    out = args.out.resolve()
    selected = {
        "stairs": args.stairs_seed,
        "gap": args.gap_seed,
        "rc_car": args.rc_car_seed,
    }

    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"Missing compact dataset manifest: {manifest_path}")

    out.mkdir(parents=True, exist_ok=True)
    created = []
    selected_data = {}

    print("============================================================")
    print("EXPERT DATASET FIGURE GENERATOR")
    print("============================================================")
    print("Dataset:", root)
    print("Output :", out)
    print("Seeds  :", selected)
    print()

    for task in ("stairs", "gap", "rc_car"):
        seed = selected[task]
        em, am, bm, assembly, behavior = load_episode(root, task, seed)
        selected_data[task] = {"assembly": assembly, "behavior": behavior}

        print(
            f"{task:7s} seed={seed}: "
            f"{len(assembly)} assembly + {len(behavior)} behavior compact rows"
        )

        plot_assembly(
            task, seed, em, bm, assembly, behavior,
            out, args.dpi, created,
        )

        if task in ("stairs", "gap"):
            plot_snake_shape(
                task, seed, bm, behavior,
                out, args.dpi, created,
            )
            plot_snake_time(
                task, seed, bm, behavior,
                out, args.dpi, created,
            )
        else:
            env_path = (
                root
                / "episodes"
                / task
                / f"seed-{seed:06d}"
                / "environment_geometry.json"
            )

            if not env_path.is_file():
                raise RuntimeError(
                    f"Missing RC-Car environment metadata: {env_path}"
                )

            environment_geometry = read_json(env_path)

            plot_rc(
                seed,
                bm,
                behavior,
                environment_geometry,
                out,
                args.dpi,
                created,
            )

    plot_summary(selected_data, out, args.dpi, created)

    figure_manifest = {
        "schema_version": "mssr.expert_figures.v1",
        "source_dataset": str(root),
        "source_manifest_sha256": hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest(),
        "representative_episodes": selected,
        "render": {
            "dpi_png": args.dpi,
            "formats": ["png", "pdf", "svg"],
            "matplotlib_version": matplotlib.__version__,
        },
        "figure_count": len(created) // 3,
        "files": sorted(created),
    }

    (out / "figure_manifest.json").write_text(
        json.dumps(figure_manifest, indent=2, sort_keys=True) + "\n"
    )

    print()
    print("============================================================")
    print("GENERATED")
    print("============================================================")
    print("Figures:", figure_manifest["figure_count"])
    print("Files  :", len(created), "+ figure_manifest.json")
    for path in sorted(out.rglob("*.png")):
        print(" ", path.relative_to(out))
    print()
    print("Dataset modified: NO")


if __name__ == "__main__":
    main()
