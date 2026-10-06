#!/usr/bin/env python3
"""Validate and curate the eight approved button teleoperation recordings."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RECORDINGS_ROOT = ROOT / "logs/teleop/recordings"
DEFAULT_DEST_ROOT = ROOT / "datasets/expert_v1/teleop"

CHECK_SCHEMA_VERSION = "mssr.button_teleop_curation_check.v1"
CURATED_SCHEMA_VERSION = "mssr.curated_teleop_episode.v1"
COLLECTION_SCHEMA_VERSION = "mssr.teleop_dataset_collection.v1"

BUTTON_THRESHOLD_M = 0.0035

EPISODES = {
    "button-01": ("teleop-c78e0455357147369ad16bc1bf39a0c2", 6217),
    "button-02": ("teleop-1f8b406b56f74377834cd863ebccf696", 6307),
    "button-03": ("teleop-170338d489a64eb397a10cf9910392a1", 6533),
    "button-04": ("teleop-8b55584534d24c86888b350e09c74e7b", 6284),
    "button-05": ("teleop-b1456ab292aa4a319c9d4bcc17e7e57f", 6341),
    "button-06": ("teleop-beaa63ebe5f347c6abc58baff9d1394a", 6443),
    "button-07": ("teleop-dd6bf8b9926d4a2aab6af7572646d026", 6265),
    "button-08": ("teleop-edbefc66cecf40c180864b452aff8dab", 6316),
}


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def iter_jsonl(path: Path):
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL in {path}:{line_number}"
                ) from exc

            if not isinstance(row, dict):
                raise ValueError(
                    f"Expected object in {path}:{line_number}"
                )

            yield row


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)

    return digest.hexdigest()


def button_depressions(value: Any):
    if isinstance(value, dict):
        for key, child in value.items():
            if (
                key == "depression_m"
                and isinstance(child, (int, float))
            ):
                yield float(child)
            else:
                yield from button_depressions(child)

    elif isinstance(value, list):
        for child in value:
            yield from button_depressions(child)


def validate_human_stream(path: Path) -> dict[str, Any]:
    samples = 0
    action_valid_samples = 0
    valid_bc_samples = 0

    max_button_depression_m: float | None = None
    button_pressed_at_sim_s: float | None = None

    first_stamp: float | None = None
    last_stamp: float | None = None
    geometry_sha256: str | None = None

    for row in iter_jsonl(path):
        samples += 1

        stamp = float(row["stamp"])

        if first_stamp is None:
            first_stamp = stamp

        if last_stamp is not None and stamp < last_stamp:
            raise ValueError(
                f"Nonmonotonic human stream: {path}"
            )

        last_stamp = stamp

        if row.get("action_valid") is True:
            action_valid_samples += 1

        supervision = row.get("supervision")

        if (
            isinstance(supervision, dict)
            and supervision.get(
                "valid_for_behavior_cloning"
            ) is True
        ):
            valid_bc_samples += 1

        graph_t = row.get("graph_t")
        graph_t_plus_1 = row.get("graph_t_plus_1")

        for graph_name, graph in (
            ("graph_t", graph_t),
            ("graph_t_plus_1", graph_t_plus_1),
        ):
            if not isinstance(graph, dict):
                raise ValueError(
                    f"{path}: missing {graph_name}"
                )

            nodes = graph.get("nodes")

            if not isinstance(nodes, list) or len(nodes) != 8:
                raise ValueError(
                    f"{path}: {graph_name} "
                    "does not contain 8 modules"
                )

        if geometry_sha256 is None:
            try:
                geometry_sha256 = (
                    graph_t["global_attributes"]
                    ["course"]["geometry_sha256"]
                )
            except (KeyError, TypeError):
                geometry_sha256 = None

        for depression in button_depressions(row):
            if (
                max_button_depression_m is None
                or depression > max_button_depression_m
            ):
                max_button_depression_m = depression

            if (
                button_pressed_at_sim_s is None
                and depression >= BUTTON_THRESHOLD_M
            ):
                button_pressed_at_sim_s = stamp

    if samples == 0:
        raise ValueError(
            f"Empty human stream: {path}"
        )

    if max_button_depression_m is None:
        raise ValueError(
            f"No button depression evidence: {path}"
        )

    return {
        "human_samples": samples,
        "human_action_valid_samples": action_valid_samples,
        "human_valid_for_behavior_cloning_samples": valid_bc_samples,
        "max_button_depression_m": max_button_depression_m,
        "button_pressed_at_sim_s": button_pressed_at_sim_s,
        "first_stamp": first_stamp,
        "last_stamp": last_stamp,
        "geometry_sha256": geometry_sha256,
    }


def last_jsonl_row(path: Path) -> dict[str, Any]:
    last = None

    for row in iter_jsonl(path):
        last = row

    if last is None:
        raise ValueError(
            f"Empty structural stream: {path}"
        )

    return last


def validate_structural_streams(
    recording: Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    structural_dir = recording / "structural"

    if not structural_dir.is_dir():
        raise ValueError(
            f"Missing structural directory: {structural_dir}"
        )

    streams = sorted(
        structural_dir.glob("*.jsonl")
    )

    if not streams:
        raise ValueError(
            f"No structural streams: {recording}"
        )

    manifest_streams = manifest.get(
        "structural_streams"
    )

    if not isinstance(manifest_streams, list):
        raise ValueError(
            f"Invalid structural_streams inventory: "
            f"{recording}"
        )

    manifest_names = {
        Path(item["path"]).name
        for item in manifest_streams
        if (
            isinstance(item, dict)
            and isinstance(item.get("path"), str)
        )
    }

    actual_names = {
        path.name
        for path in streams
    }

    if manifest_names != actual_names:
        raise ValueError(
            f"Structural inventory mismatch for "
            f"{recording.name}: "
            f"manifest={sorted(manifest_names)}, "
            f"actual={sorted(actual_names)}"
        )

    all_terminal_success = True

    for stream in streams:
        terminal = last_jsonl_row(stream)

        ok = (
            terminal.get("done") is True
            and terminal.get("is_last") is True
            and terminal.get("is_terminal") is True
            and terminal.get("success") is True
        )

        if not ok:
            all_terminal_success = False

    return {
        "structural_streams": len(streams),
        "all_structural_terminal_success": (
            all_terminal_success
        ),
    }


def validate_episode(
    label: str,
    episode_id: str,
    seed: int,
) -> dict[str, Any]:
    recording = (
        RECORDINGS_ROOT / episode_id
    )

    if not recording.is_dir():
        raise ValueError(
            f"Missing recording: {recording}"
        )

    manifest_path = (
        recording / "manifest.json"
    )
    human_path = (
        recording / "human_behavior.jsonl"
    )

    if not manifest_path.is_file():
        raise ValueError(
            f"Missing manifest: {manifest_path}"
        )

    if not human_path.is_file():
        raise ValueError(
            f"Missing human stream: {human_path}"
        )

    manifest = load_json(
        manifest_path
    )

    if manifest.get("episode_id") != episode_id:
        raise ValueError(
            f"{label}: manifest episode_id mismatch "
            f"{manifest.get('episode_id')!r}"
        )

    human = validate_human_stream(
        human_path
    )

    structural = validate_structural_streams(
        recording,
        manifest,
    )

    result = {
        "label": label,
        "episode_id": episode_id,
        "seed": seed,
        "source_status": manifest.get("status"),
        "eligible_for_import": manifest.get(
            "eligible_for_import"
        ),
        "unwritten_records": manifest.get(
            "unwritten_records"
        ),
        **human,
        **structural,
    }

    result["source_valid"] = (
        result["source_status"] == "completed"
        and result["eligible_for_import"] is True
        and result["unwritten_records"] == 0
        and (
            result["human_action_valid_samples"]
            == result["human_samples"]
        )
        and (
            result[
                "human_valid_for_behavior_cloning_samples"
            ]
            == result["human_samples"]
        )
        and (
            result["max_button_depression_m"]
            >= BUTTON_THRESHOLD_M
        )
        and result["structural_streams"] >= 1
        and (
            result[
                "all_structural_terminal_success"
            ]
            is True
        )
    )

    return result


def run_check() -> dict[str, Any]:
    episodes = [
        validate_episode(
            label,
            episode_id,
            seed,
        )
        for label, (
            episode_id,
            seed,
        ) in EPISODES.items()
    ]

    return {
        "mode": "check",
        "schema_version": CHECK_SCHEMA_VERSION,
        "button_threshold_m": BUTTON_THRESHOLD_M,
        "episodes": episodes,
        "all_sources_valid": all(
            episode["source_valid"]
            for episode in episodes
        ),
    }


def stream_stats(
    path: Path,
    *,
    human: bool,
) -> dict[str, Any]:
    stats = Counter()

    first_stamp = None
    last_stamp = None

    for row in iter_jsonl(path):
        stamp = float(row["stamp"])

        if first_stamp is None:
            first_stamp = stamp

        if last_stamp is not None and stamp < last_stamp:
            raise ValueError(
                f"Nonmonotonic stream: {path}"
            )

        last_stamp = stamp

        stats["samples"] += 1

        supervision = row.get(
            "supervision"
        )

        if (
            isinstance(supervision, dict)
            and supervision.get(
                "valid_for_behavior_cloning"
            ) is True
        ):
            stats[
                "valid_for_behavior_cloning_samples"
            ] += 1

        if row.get("action_valid") is True:
            stats["action_valid_samples"] += 1

    if not stats["samples"]:
        raise ValueError(
            f"Empty stream: {path}"
        )

    return {
        "samples": stats["samples"],
        "valid_for_behavior_cloning_samples": (
            stats[
                "valid_for_behavior_cloning_samples"
            ]
        ),
        "action_valid_samples": (
            stats["action_valid_samples"]
        ),
        "first_stamp": first_stamp,
        "last_stamp": last_stamp,
        "producer": (
            "human_expert"
            if human
            else "deterministic_expert"
        ),
    }


def materialize_file(
    source: Path,
    target: Path,
) -> str:
    """Hardlink when possible, otherwise make a byte copy."""

    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        shutil.copyfile(source, target)
        return "copy"


def materialize_stream(
    source: Path,
    target: Path,
    *,
    human: bool,
    dest_root: Path,
    episode_id: str,
) -> dict[str, Any]:
    stats = stream_stats(
        source,
        human=human,
    )

    source_hash = sha256(source)
    source_bytes = source.stat().st_size

    storage = materialize_file(
        source,
        target,
    )

    if (
        target.stat().st_size != source_bytes
        or sha256(target) != source_hash
    ):
        raise ValueError(
            f"Materialized stream differs: {source}"
        )

    relative = target.relative_to(
        dest_root
        / "episodes"
        / f".{episode_id}.staging"
    )

    return {
        "sha256": source_hash,
        "bytes": source_bytes,
        **stats,
        "storage": storage,
        "path": (
            Path("episodes")
            / episode_id
            / relative
        ).as_posix(),
        "source_path": (
            source.relative_to(ROOT).as_posix()
        ),
    }


def update_totals(
    collection: dict[str, Any],
    entries: list[dict[str, Any]],
) -> None:
    totals = collection.setdefault(
        "totals",
        {},
    )

    human_samples = sum(
        item["samples"]
        for item in entries
        if item["producer"] == "human_expert"
    )

    structural_samples = sum(
        item["samples"]
        for item in entries
        if item["producer"]
        == "deterministic_expert"
    )

    totals["episodes"] = len(
        collection["episodes"]
    )

    totals["human_samples"] = (
        int(totals.get("human_samples", 0))
        + human_samples
    )

    totals["structural_samples"] = (
        int(
            totals.get(
                "structural_samples",
                0,
            )
        )
        + structural_samples
    )

    totals["samples"] = (
        int(totals.get("samples", 0))
        + sum(
            item["samples"]
            for item in entries
        )
    )

    totals[
        "valid_for_behavior_cloning_samples"
    ] = (
        int(
            totals.get(
                "valid_for_behavior_cloning_samples",
                0,
            )
        )
        + sum(
            item[
                "valid_for_behavior_cloning_samples"
            ]
            for item in entries
        )
    )

    totals["bytes"] = (
        int(totals.get("bytes", 0))
        + sum(
            item["bytes"]
            for item in entries
        )
    )


def run_import(
    dest_root: Path,
) -> dict[str, Any]:
    check = run_check()

    if not check["all_sources_valid"]:
        raise ValueError(
            "Button RAW source validation failed"
        )

    collection_path = (
        dest_root / "manifest.json"
    )
    episodes_root = (
        dest_root / "episodes"
    )

    if not collection_path.is_file():
        raise ValueError(
            f"Missing teleop collection manifest: "
            f"{collection_path}"
        )

    if not episodes_root.is_dir():
        raise ValueError(
            f"Missing teleop episodes directory: "
            f"{episodes_root}"
        )

    original_collection_bytes = (
        collection_path.read_bytes()
    )

    collection = load_json(
        collection_path
    )

    if (
        collection.get("schema_version")
        != COLLECTION_SCHEMA_VERSION
    ):
        raise ValueError(
            "Unexpected teleop collection schema"
        )

    existing_ids = {
        item.get("episode_id")
        for item in collection.get(
            "episodes",
            [],
        )
        if isinstance(item, dict)
    }

    for episode in check["episodes"]:
        episode_id = episode["episode_id"]

        final = (
            episodes_root / episode_id
        )

        staging = (
            episodes_root
            / f".{episode_id}.staging"
        )

        if (
            episode_id in existing_ids
            or final.exists()
            or staging.exists()
        ):
            raise ValueError(
                f"Episode already imported or staged: "
                f"{episode_id}"
            )

    staged: list[
        tuple[
            Path,
            Path,
            dict[str, Any],
        ]
    ] = []

    renamed_finals: list[Path] = []

    try:
        for episode in check["episodes"]:
            label = episode["label"]
            episode_id = episode[
                "episode_id"
            ]
            seed = episode["seed"]

            source = (
                RECORDINGS_ROOT
                / episode_id
            )

            final = (
                episodes_root
                / episode_id
            )

            staging = (
                episodes_root
                / f".{episode_id}.staging"
            )

            staging.mkdir()
            (
                staging / "structural"
            ).mkdir()

            human_entry = materialize_stream(
                source
                / "human_behavior.jsonl",
                staging
                / "human_behavior.jsonl",
                human=True,
                dest_root=dest_root,
                episode_id=episode_id,
            )

            entries = [
                human_entry
            ]

            for structural_source in sorted(
                (
                    source / "structural"
                ).glob("*.jsonl")
            ):
                entry = materialize_stream(
                    structural_source,
                    staging
                    / "structural"
                    / structural_source.name,
                    human=False,
                    dest_root=dest_root,
                    episode_id=episode_id,
                )

                entries.append(entry)

            shutil.copyfile(
                source / "manifest.json",
                staging
                / "source_manifest.json",
            )

            curated = {
                "schema_version": (
                    CURATED_SCHEMA_VERSION
                ),
                "episode_id": episode_id,
                "course": "BUTTON",
                "task_success": True,
                "success_basis": (
                    "physical_button_depression_"
                    "and_structural_terminal_success"
                ),
                "success_scope": (
                    "isolated_button_press"
                ),
                "full_mission_success": True,
                "geometry_sha256": (
                    episode[
                        "geometry_sha256"
                    ]
                ),
                "source_manifest": (
                    source
                    / "manifest.json"
                ).relative_to(
                    ROOT
                ).as_posix(),
                "source_manifest_copy": (
                    "source_manifest.json"
                ),
                "evidence": {
                    "button_label": label,
                    "button_seed": seed,
                    "button_threshold_m": (
                        BUTTON_THRESHOLD_M
                    ),
                    "max_button_depression_m": (
                        episode[
                            "max_button_depression_m"
                        ]
                    ),
                    "button_physically_pressed_at_sim_s": (
                        episode[
                            "button_pressed_at_sim_s"
                        ]
                    ),
                    "human_samples": (
                        episode[
                            "human_samples"
                        ]
                    ),
                    "all_structural_terminal_success": (
                        episode[
                            "all_structural_terminal_success"
                        ]
                    ),
                    "structural_streams": (
                        episode[
                            "structural_streams"
                        ]
                    ),
                },
                "datasets": entries,
            }

            write_json(
                staging / "manifest.json",
                curated,
            )

            collection_entry = {
                "episode_id": episode_id,
                "course": "BUTTON",
                "task_success": True,
                "success_basis": curated[
                    "success_basis"
                ],
                "success_scope": curated[
                    "success_scope"
                ],
                "full_mission_success": True,
                "geometry_sha256": (
                    curated[
                        "geometry_sha256"
                    ]
                ),
                "manifest": (
                    f"episodes/{episode_id}/"
                    "manifest.json"
                ),
                "datasets": entries,
            }

            staged.append(
                (
                    staging,
                    final,
                    collection_entry,
                )
            )

        # Nothing becomes visible in the collection until every
        # episode has been fully staged and validated.
        for (
            staging,
            final,
            _entry,
        ) in staged:
            staging.rename(final)
            renamed_finals.append(final)

        for (
            _staging,
            _final,
            entry,
        ) in staged:
            collection["episodes"].append(
                entry
            )
            update_totals(
                collection,
                entry["datasets"],
            )

        policy = str(
            collection.get(
                "selection_policy",
                "",
            )
        ).strip()

        addition = (
            "Eight validated isolated button "
            "teleoperation demonstrations "
            "(button-01 through button-08) "
            "imported with complete human streams "
            "and successful structural assembly streams."
        )

        if addition not in policy:
            collection[
                "selection_policy"
            ] = (
                f"{policy} {addition}".strip()
            )

        temporary = (
            collection_path.with_suffix(
                ".json.tmp"
            )
        )

        write_json(
            temporary,
            collection,
        )

        temporary.replace(
            collection_path
        )

    except Exception:
        for staging, _final, _entry in staged:
            if staging.exists():
                shutil.rmtree(staging)

        for final in reversed(
            renamed_finals
        ):
            if final.exists():
                shutil.rmtree(final)

        collection_path.write_bytes(
            original_collection_bytes
        )

        temporary = (
            collection_path.with_suffix(
                ".json.tmp"
            )
        )

        if temporary.exists():
            temporary.unlink()

        raise

    return {
        "mode": "import",
        "schema_version": (
            CURATED_SCHEMA_VERSION
        ),
        "imported_episodes": len(
            check["episodes"]
        ),
        "episodes": check["episodes"],
        "all_sources_valid": check[
            "all_sources_valid"
        ],
        "destination": str(
            dest_root
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    mode = (
        parser.add_mutually_exclusive_group(
            required=True
        )
    )

    mode.add_argument(
        "--check",
        action="store_true",
        help=(
            "Validate the eight approved RAW "
            "recordings without writing."
        ),
    )

    mode.add_argument(
        "--import",
        dest="do_import",
        action="store_true",
        help=(
            "Import the eight approved recordings "
            "into a teleop dataset collection."
        ),
    )

    parser.add_argument(
        "--dest-root",
        type=Path,
        default=DEFAULT_DEST_ROOT,
        help=(
            "Teleop collection root. "
            "Defaults to datasets/expert_v1/teleop."
        ),
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.check:
        result = run_check()

    else:
        result = run_import(
            args.dest_root.resolve()
        )

    print(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        )
    )

    return (
        0
        if result["all_sources_valid"]
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
