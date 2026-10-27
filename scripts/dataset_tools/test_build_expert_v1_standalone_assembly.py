import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/dataset_tools/build_expert_v1.py"

spec = importlib.util.spec_from_file_location(
    "build_expert_v1_standalone_assembly_test",
    SCRIPT,
)
builder = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = builder
spec.loader.exec_module(builder)


def test_successful_standalone_assembly_result_is_single_phase_candidate(
    tmp_path: Path,
) -> None:
    run = tmp_path / "mm8-assembly-demo"
    run.mkdir()

    # Il dataset viene completato prima del completion artifact.
    (run / "assembly_dataset.jsonl").write_text(
        '{"schema_version":"mssr.expert_transition.v3"}\n',
        encoding="utf-8",
    )

    result = {
        "schema_version": "mssr.self_assembly_episode.v1",
        "episode_id": "mm8-assembly-demo-test",
        "seed": 6217,
        "success": True,
        "target_morphology": "mobile_manipulator8",
    }
    result_path = run / "result.json"
    result_path.write_text(
        json.dumps(result, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    candidate = builder.candidate_from_result(result_path)

    assert candidate is not None
    assert candidate.task == "assembly"
    assert candidate.seed == 6217
    assert candidate.phases == ("assembly",)
    assert candidate.successful is True
    assert candidate.stable is True
    assert candidate.run_dir == run
    assert candidate.completion_path == result_path


def test_standalone_assembly_task_requires_only_assembly_phase() -> None:
    assert builder.PHASES_BY_TASK["assembly"] == ("assembly",)
    assert builder.REQUIRED_PHASES_BY_TASK["assembly"] == ("assembly",)
