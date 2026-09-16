import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
MODULE = (
    ROOT
    / "scripts"
    / "smores_ep"
    / "src"
    / "smores_ep"
    / "dataset"
    / "composite_dataset.py"
)


def _load_module():
    assert MODULE.is_file(), "composite_dataset.py is not implemented yet"
    name = "_composite_dataset_manifest_test"
    spec = importlib.util.spec_from_file_location(name, MODULE)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_manifest_owns_distinct_ordered_streams_and_counts_streaming(
    tmp_path,
    monkeypatch,
):
    module = _load_module()

    manifest = module.CompositeDatasetManifest(
        runtime_dir=tmp_path,
        episode_id="composite-c05",
    )

    first = manifest.register(
        stage_id=0,
        task_id="gap-1",
        phase="assembly",
        producer="self_assembly",
        action_space="module_primitives",
        source_morphology=None,
        target_morphology="snake8",
        intended_for_behavior_cloning=True,
    )
    second = manifest.register(
        stage_id=1,
        task_id="gap-1",
        phase="behavior",
        producer="morphology_behavior_node",
        action_space="module_locomotion",
        source_morphology="snake8",
        target_morphology="snake8",
        intended_for_behavior_cloning=True,
    )

    assert first.path != second.path
    assert first.path.parent == tmp_path / "dataset_parts"
    assert second.path.parent == tmp_path / "dataset_parts"

    first.path.write_text('{"x":1}\n{"x":2}\n', encoding="utf-8")
    second.path.write_text('{"x":3}\n', encoding="utf-8")

    protected = {first.path.resolve(), second.path.resolve()}
    original_read_text = Path.read_text

    def guarded_read_text(self, *args, **kwargs):
        if self.resolve() in protected:
            raise AssertionError("dataset stream must not be read whole")
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read_text)

    manifest.finalize_stage(0, success=True)
    manifest.finalize_stage(1, success=True)
    manifest.finalize_episode(success=True)

    manifest_path = tmp_path / "dataset_manifest.json"

    with manifest_path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)

    assert payload["schema_version"] == "mssr.composite_dataset_manifest.v1"
    assert payload["episode_id"] == "composite-c05"
    assert payload["status"] == "succeeded"

    streams = payload["streams"]

    assert [item["stage_id"] for item in streams] == [0, 1]
    assert [item["records"] for item in streams] == [2, 1]
    assert all(item["bytes"] > 0 for item in streams)
    assert all(item["status"] == "completed" for item in streams)

    assert not list(tmp_path.glob("*.tmp"))


def test_finalize_episode_remeasures_and_classifies_unfinished_streams(
    tmp_path,
):
    module = _load_module()

    manifest = module.CompositeDatasetManifest(
        runtime_dir=tmp_path,
        episode_id="composite-failure",
    )

    completed = manifest.register(
        stage_id=0,
        task_id="gap-1",
        phase="assembly",
        producer="self_assembly",
        action_space="module_primitives",
        source_morphology=None,
        target_morphology="snake8",
        intended_for_behavior_cloning=True,
    )

    partial = manifest.register(
        stage_id=1,
        task_id="gap-1",
        phase="behavior",
        producer="morphology_behavior_node",
        action_space="module_locomotion",
        source_morphology="snake8",
        target_morphology="snake8",
        intended_for_behavior_cloning=True,
    )

    manifest.register(
        stage_id=2,
        task_id="gap-1",
        phase="reconfiguration",
        producer="self_reconfiguration",
        action_space="module_primitives",
        source_morphology="snake8",
        target_morphology="rc_car8",
        intended_for_behavior_cloning=True,
    )

    completed.path.write_text(
        '{"step":0}\n',
        encoding="utf-8",
    )
    manifest.finalize_stage(0, success=True)

    # Data can still arrive after stage finalization, for example when
    # a behavior pending transition is flushed at the next boundary.
    with completed.path.open("a", encoding="utf-8") as stream:
        stream.write('{"step":1}\n')

    partial.path.write_text(
        '{"step":0}\n',
        encoding="utf-8",
    )

    manifest.finalize_episode(success=False)

    with (
        tmp_path / "dataset_manifest.json"
    ).open("r", encoding="utf-8") as stream:
        payload = json.load(stream)

    assert payload["status"] == "failed"

    streams = payload["streams"]

    assert [item["records"] for item in streams] == [2, 1, 0]
    assert [item["status"] for item in streams] == [
        "completed",
        "partial",
        "not_started",
    ]


def test_finalize_failed_episode_measures_registered_streams(
    tmp_path,
    monkeypatch,
):
    module = _load_module()

    manifest = module.CompositeDatasetManifest(
        runtime_dir=tmp_path,
        episode_id="composite-failed",
    )

    completed = manifest.register(
        stage_id=0,
        task_id="gap-1",
        phase="behavior",
        producer="morphology_behavior_node",
        action_space="module_locomotion",
        source_morphology="snake8",
        target_morphology="snake8",
        intended_for_behavior_cloning=True,
    )
    partial = manifest.register(
        stage_id=1,
        task_id="stairs-1",
        phase="behavior",
        producer="morphology_behavior_node",
        action_space="module_locomotion",
        source_morphology="snake8",
        target_morphology="snake8",
        intended_for_behavior_cloning=True,
    )
    not_started = manifest.register(
        stage_id=2,
        task_id="button-1",
        phase="manipulation",
        producer="button_expert",
        action_space="manipulation_5dof",
        source_morphology="mobile_manipulator8",
        target_morphology="mobile_manipulator8",
        intended_for_behavior_cloning=True,
    )

    completed.path.write_text(
        '{"x":1}\n',
        encoding="utf-8",
    )
    manifest.finalize_stage(0, success=True)

    partial.path.write_text(
        '{"x":2}\n{"x":3}\n',
        encoding="utf-8",
    )

    protected = {
        completed.path.resolve(),
        partial.path.resolve(),
        not_started.path.resolve(),
    }
    original_read_text = Path.read_text

    def guarded_read_text(self, *args, **kwargs):
        if self.resolve() in protected:
            raise AssertionError(
                "episode finalization must measure streams incrementally"
            )
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(
        Path,
        "read_text",
        guarded_read_text,
    )

    manifest.finalize_episode(success=False)

    with (
        tmp_path / "dataset_manifest.json"
    ).open("r", encoding="utf-8") as stream:
        payload = json.load(stream)

    assert payload["status"] == "failed"

    by_phase = {
        item["phase"]: item
        for item in payload["streams"]
    }

    assert by_phase["behavior"]["status"] in {
        "completed",
        "partial",
    }

    streams = payload["streams"]

    assert streams[0]["status"] == "completed"
    assert streams[0]["records"] == 1

    assert streams[1]["status"] == "partial"
    assert streams[1]["records"] == 2
    assert streams[1]["bytes"] > 0

    assert streams[2]["status"] == "not_started"
    assert streams[2]["records"] == 0
    assert streams[2]["bytes"] == 0
