"""Tests for campaign selection and composite runner inputs."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "run_composite_course.py"
SPEC = importlib.util.spec_from_file_location("run_composite_course", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_select_episode_materializes_mission_schema() -> None:
    campaign = {
        "schema_version": "mssr.composite_campaign.v1",
        "episodes": [
            {
                "episode_id": "test-1",
                "tasks": [{"task_id": "gap", "type": "gap", "seed": 4100}],
            }
        ],
    }
    mission = MODULE.select_episode(campaign, "test-1")
    assert mission["schema_version"] == "mssr.composite_mission.v1"
    assert mission["tasks"][0]["seed"] == 4100


def test_select_episode_rejects_unknown_id() -> None:
    with pytest.raises(ValueError, match="was not found exactly once"):
        MODULE.select_episode(
            {"schema_version": "mssr.composite_campaign.v1", "episodes": []},
            "missing",
        )


def test_reconfiguration_executor_selection_is_explicit_and_legacy_by_default():
    assert MODULE.argument_parser().parse_args([]).reconfiguration_executor == "legacy"
    assert MODULE.argument_parser().parse_args([
        "--reconfiguration-executor", "v2"
    ]).reconfiguration_executor == "v2"
    assert MODULE.transition_executable(False, "legacy") == "mssr_smores_self_reconfiguration_node"
    assert MODULE.transition_executable(False, "v2") == "mssr_smores_deterministic_reconfiguration_node"
    assert MODULE.transition_executable(True, "v2") == "mssr_smores_self_assembly_node"


def test_normalize_dataset_creates_one_episode_terminal(tmp_path: Path) -> None:
    path = tmp_path / "dataset.jsonl"
    records = [
        {
            "episode_id": "local-a",
            "timestep": 0,
            "done": True,
            "success": True,
            "graph_t": {"stamp": 1},
            "observation_t": {"phase": "assembly"},
        },
        {
            "episode_id": "local-b",
            "timestep": 0,
            "done": True,
            "success": True,
            "graph_t": {"stamp": 2},
            "observation_t": {"phase": "stairs"},
        },
    ]
    path.write_text("".join(__import__("json").dumps(item) + "\n" for item in records))

    MODULE.normalize_dataset(path, "composite-test", True)

    normalized = [
        __import__("json").loads(line)
        for line in path.read_text().splitlines()
    ]
    assert [item["timestep"] for item in normalized] == [0, 1]
    assert [item["done"] for item in normalized] == [False, True]
    assert normalized[0]["graph_t_plus_1"] == normalized[1]["graph_t"]
    assert normalized[-1]["episode_success"] is True


def test_execute_navigation_uses_single_final_goal(
    tmp_path,
    monkeypatch,
) -> None:
    import importlib.util
    import math
    import sys
    from pathlib import Path
    from types import SimpleNamespace

    script = Path(__file__).parents[1] / "run_composite_course.py"
    spec = importlib.util.spec_from_file_location(
        "_run_composite_course_single_goal_test",
        script,
    )
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    class FakeProcess:
        pass

    monkeypatch.setattr(
        module.subprocess,
        "Popen",
        lambda *args, **kwargs: FakeProcess(),
    )
    monkeypatch.setattr(
        module,
        "wait_nav2",
        lambda environment: None,
    )
    monkeypatch.setattr(
        module,
        "stop_process",
        lambda process: None,
    )

    calls = []

    def fake_run_logged(
        command,
        *,
        environment,
        log_path,
        timeout_s,
    ):
        calls.append(
            {
                "command": list(command),
                "log_path": Path(log_path),
                "timeout_s": timeout_s,
            }
        )

    monkeypatch.setattr(module, "run_logged", fake_run_logged)

    stage = SimpleNamespace(
        stage_id=3,
        kind="nav2",
        task_id="rc-1",
        parameters={
            "seed": 5100,
            "waypoints_xyyaw": [
                [0.0, 0.0, 0.0],
                [2.471, 0.620, 1.571],
            ],
            "navigation_goal_xyyaw": [
                2.471,
                0.620,
                1.571,
            ],

            # Deliberately include stale metadata: the runner must no
            # longer interpret it as a second navigation command.
            "navigation_via_xyyaw": [
                2.10,
                0.18,
                1.571,
            ],

            "navigation_accept_position_m": 0.10,
            "navigation_accept_yaw_rad": math.pi,
        },
    )

    module.execute_navigation(
        stage,
        runtime_dir=tmp_path,
        episode_id="composite-c05",
        environment={},
    )

    assert len(calls) == 1

    command = calls[0]["command"]

    assert "--dataset-path" not in command

    def value(flag):
        index = command.index(flag)
        return command[index + 1]

    assert float(value("--goal-x")) == pytest.approx(2.471)
    assert float(value("--goal-y")) == pytest.approx(0.620)

    assert float(
        value("--accept-position-m")
    ) == pytest.approx(0.10)

    assert float(
        value("--accept-yaw-rad")
    ) == pytest.approx(math.pi)

    assert calls[0]["log_path"].name == "stage-03-route.log"
    assert value("--result-json").endswith("stage-03-nav2.json")


def test_composite_main_uses_manifest_instead_of_monolithic_dataset() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    start = source.index("def main() -> int:")
    end = source.index(
        'if __name__ == "__main__":',
        start,
    )
    main_source = source[start:end]

    assert "dataset.jsonl" not in main_source
    assert "normalize_dataset(" not in main_source

    assert '"behavior_dataset_path:="' in main_source

    assert (
        "dataset_manifest.finalize_episode(success=True)"
        in main_source
    )
    assert (
        "dataset_manifest.finalize_episode(success=False)"
        in main_source
    )


def test_main_uses_manifest_instead_of_monolithic_dataset() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "run_composite_course.py"
    )
    text = script.read_text(encoding="utf-8")
    main = text.split("def main() -> int:", 1)[1]

    assert (
        'f"behavior_dataset_path:='
        "{runtime_dir / 'dataset.jsonl'}"
        '"'
        not in main
    )
    assert (
        'dataset_path = runtime_dir / "dataset.jsonl"'
        not in main
    )

    assert (
        "dataset_manifest.finalize_episode(success=True)"
        in main
    )
    assert (
        "dataset_manifest.finalize_episode(success=False)"
        in main
    )

    # Keep normalize_dataset() as a legacy utility, but the new
    # composite main path must never invoke it.
    assert "normalize_dataset(" not in main
