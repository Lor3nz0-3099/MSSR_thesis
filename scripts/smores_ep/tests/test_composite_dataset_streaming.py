import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RUNNER = ROOT / "scripts" / "smores_ep" / "run_composite_course.py"


def _load_runner():
    name = "_composite_runner_dataset_streaming_test"
    spec = importlib.util.spec_from_file_location(name, RUNNER)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_normalize_dataset_does_not_read_whole_jsonl(
    tmp_path,
    monkeypatch,
) -> None:
    runner = _load_runner()

    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "graph_t": {"step": 0},
                        "done": False,
                        "success": False,
                    }
                ),
                json.dumps(
                    {
                        "graph_t": {"step": 1},
                        "done": True,
                        "success": True,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    original_read_text = Path.read_text
    dataset_resolved = dataset.resolve()

    def guarded_read_text(self, *args, **kwargs):
        if self.resolve() == dataset_resolved:
            raise AssertionError(
                "whole-file read forbidden for dataset.jsonl"
            )
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read_text)

    runner.normalize_dataset(
        dataset,
        episode_id="stream-test",
        success=True,
    )

    with dataset.open("r", encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream if line.strip()]

    assert len(records) == 2
    assert [record["timestep"] for record in records] == [0, 1]
    assert all(
        record["episode_id"] == "stream-test"
        for record in records
    )
    assert records[0]["skill_done"] is False
    assert records[1]["skill_done"] is True
    assert records[0]["skill_success"] is False
    assert records[1]["skill_success"] is True
