import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = ROOT / "scripts" / "smores_ep" / "run_composite_course.py"


def _load_module():
    name = "_composite_dataset_stage_routing_test"
    spec = importlib.util.spec_from_file_location(name, MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class _Manifest:
    def __init__(self, root: Path):
        self.root = root
        self.registered = []
        self.finalized = []

    def register(self, **kwargs):
        self.registered.append(kwargs)
        path = (
            self.root
            / "dataset_parts"
            / f"{kwargs['stage_id']:02d}-{kwargs['phase']}.jsonl"
        )
        return SimpleNamespace(path=path)

    def finalize_stage(self, stage_id: int, success: bool):
        self.finalized.append((stage_id, success))


def _stage(stage_id, kind, source, target):
    return SimpleNamespace(
        stage_id=stage_id,
        task_id=f"task-{stage_id}",
        task_type="test",
        kind=kind,
        source_morphology=source,
        target_morphology=target,
        behavior=None,
        parameters={},
    )


def test_transition_stages_receive_distinct_manifest_paths(
    tmp_path,
    monkeypatch,
):
    module = _load_module()

    received_paths = []

    def fake_start_transition(stage, **kwargs):
        received_paths.append(Path(kwargs["dataset_path"]))

    monkeypatch.setattr(module, "start_transition", fake_start_transition)

    manifest = _Manifest(tmp_path)

    stages = (
        _stage(0, "assembly", None, "snake8"),
        _stage(1, "reconfiguration", "snake8", "rc_car8"),
    )

    result = module.execute_stages(
        stages,
        runtime=object(),
        runtime_dir=tmp_path,
        dataset_manifest=manifest,
        episode_id="composite-test",
        environment={},
        headless=True,
    )

    assert result is True
    assert len(received_paths) == 2
    assert received_paths[0] != received_paths[1]

    assert [item["phase"] for item in manifest.registered] == [
        "assembly",
        "reconfiguration",
    ]
    assert manifest.finalized == [(0, True), (1, True)]


def test_execute_stages_no_longer_accepts_shared_dataset_path():
    module = _load_module()

    import inspect

    parameters = inspect.signature(module.execute_stages).parameters

    assert "dataset_manifest" in parameters
    assert "dataset_path" not in parameters

def test_nav2_configures_module_behavior_stream_before_navigation(
    tmp_path,
    monkeypatch,
):
    module = _load_module()
    manifest = _Manifest(tmp_path)
    events = []

    def fake_set_behavior_dataset_context(**kwargs):
        events.append(
            (
                "context",
                Path(kwargs["dataset_path"]),
                kwargs["stage_name"],
            )
        )

    def fake_execute_navigation(stage, **kwargs):
        events.append(("navigation", stage.stage_id))

    monkeypatch.setattr(
        module,
        "set_behavior_dataset_context",
        fake_set_behavior_dataset_context,
    )
    monkeypatch.setattr(
        module,
        "execute_navigation",
        fake_execute_navigation,
    )

    stage = _stage(3, "nav2", "rc_car8", "rc_car8")

    result = module.execute_stages(
        (stage,),
        runtime=object(),
        runtime_dir=tmp_path,
        dataset_manifest=manifest,
        episode_id="composite-c05",
        environment={},
        headless=True,
    )

    assert result is True
    assert len(manifest.registered) == 1

    registered = manifest.registered[0]
    assert registered["phase"] == "behavior"
    assert registered["producer"] == "morphology_behavior_node"
    assert registered["action_space"] == "module_locomotion"

    assert events[0][0] == "context"
    assert events[0][1] == (
        tmp_path / "dataset_parts" / "03-behavior.jsonl"
    )
    assert events[1] == ("navigation", 3)
    assert manifest.finalized == [(3, True)]

def test_set_behavior_dataset_context_sets_path_and_stage_name(
    tmp_path,
    monkeypatch,
):
    module = _load_module()
    calls = []

    def fake_run(command, **kwargs):
        calls.append((list(command), kwargs))
        return SimpleNamespace(
            returncode=0,
            stdout="Set parameter successful\n",
            stderr="",
        )

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    dataset_path = tmp_path / "dataset_parts" / "03-behavior.jsonl"

    module.set_behavior_dataset_context(
        environment={"ROS_DOMAIN_ID": "0"},
        dataset_path=dataset_path,
        stage_name="composite-03-rc-1-nav2",
    )

    assert [item[0] for item in calls] == [
        [
            "ros2", "param", "set",
            "/smores_morphology_behavior_node",
            "behavior_dataset_path",
            str(dataset_path),
        ],
        [
            "ros2", "param", "set",
            "/smores_morphology_behavior_node",
            "behavior_dataset_stage_name",
            "composite-03-rc-1-nav2",
        ],
    ]

def test_behavior_stage_uses_its_own_module_action_stream_before_execution(
    tmp_path,
    monkeypatch,
):
    module = _load_module()
    manifest = _Manifest(tmp_path)
    events = []

    def fake_set_behavior_dataset_context(**kwargs):
        events.append(
            (
                "context",
                Path(kwargs["dataset_path"]),
                kwargs["stage_name"],
            )
        )

    def fake_run_logged(command, **kwargs):
        events.append(("behavior", list(command)))

    monkeypatch.setattr(
        module,
        "set_behavior_dataset_context",
        fake_set_behavior_dataset_context,
    )
    monkeypatch.setattr(
        module,
        "run_logged",
        fake_run_logged,
    )

    stage = _stage(2, "behavior", "snake8", "snake8")
    stage.behavior = "gap_crossing"

    result = module.execute_stages(
        (stage,),
        runtime=object(),
        runtime_dir=tmp_path,
        dataset_manifest=manifest,
        episode_id="composite-test",
        environment={},
        headless=True,
    )

    assert result is True
    assert len(manifest.registered) == 1

    registered = manifest.registered[0]
    assert registered["phase"] == "behavior"
    assert registered["producer"] == "morphology_behavior_node"
    assert registered["action_space"] == "module_locomotion"

    assert events[0][0] == "context"
    assert events[0][1] == (
        tmp_path / "dataset_parts" / "02-behavior.jsonl"
    )
    assert events[1][0] == "behavior"

    assert manifest.finalized == [(2, True)]


def test_button_stage_registers_five_streams_and_passes_layout(
    tmp_path,
    monkeypatch,
):
    import json

    module = _load_module()
    manifest = _Manifest(tmp_path)
    commands = []

    def fake_run_logged(command, **kwargs):
        commands.append(list(command))

    monkeypatch.setattr(module, "run_logged", fake_run_logged)

    stage = _stage(
        4,
        "button_rc_alignment",
        "rc_car8",
        "rc_car8",
    )
    stage.parameters = {"seed": 6101}

    result = module.execute_stages(
        (stage,),
        runtime=object(),
        runtime_dir=tmp_path,
        dataset_manifest=manifest,
        episode_id="composite-c05",
        environment={},
        headless=True,
    )

    assert result is True

    assert [
        item["phase"]
        for item in manifest.registered
    ] == [
        "rc_behavior",
        "rc_to_mm8_reconfiguration",
        "mm8_behavior",
        "manipulation",
        "mm8_to_rc_reconfiguration",
    ]

    assert [
        item["producer"]
        for item in manifest.registered
    ] == [
        "morphology_behavior_node",
        "self_reconfiguration",
        "morphology_behavior_node",
        "button_expert",
        "self_reconfiguration",
    ]

    assert manifest.finalized == [(4, True)]
    assert len(commands) == 1

    command = commands[0]

    assert "--dataset-layout-json" in command
    assert "--dataset-path" not in command

    layout_path = Path(
        command[
            command.index("--dataset-layout-json") + 1
        ]
    )

    layout = json.loads(
        layout_path.read_text(encoding="utf-8")
    )

    assert set(layout) == {
        "rc_behavior",
        "rc_to_mm8_reconfiguration",
        "mm8_behavior",
        "manipulation",
        "mm8_to_rc_reconfiguration",
    }

    for item in manifest.registered:
        expected = (
            tmp_path
            / "dataset_parts"
            / f"04-{item['phase']}.jsonl"
        )
        assert Path(layout[item["phase"]]) == expected


def test_button_stage_finalizes_partial_when_expert_fails(
    tmp_path,
    monkeypatch,
):
    import pytest

    module = _load_module()
    manifest = _Manifest(tmp_path)

    def fake_run_logged(command, **kwargs):
        raise RuntimeError("button failed")

    monkeypatch.setattr(module, "run_logged", fake_run_logged)

    stage = _stage(
        4,
        "button_rc_alignment",
        "rc_car8",
        "rc_car8",
    )
    stage.parameters = {"seed": 6101}

    with pytest.raises(
        RuntimeError,
        match="button failed",
    ):
        module.execute_stages(
            (stage,),
            runtime=object(),
            runtime_dir=tmp_path,
            dataset_manifest=manifest,
            episode_id="composite-c05",
            environment={},
            headless=True,
        )

    assert len(manifest.registered) == 5
    assert manifest.finalized == [(4, False)]
