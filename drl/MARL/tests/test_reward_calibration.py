"""Calibration contracts: missing signals never become measured zero samples."""
from dataclasses import dataclass, replace
import json
from types import SimpleNamespace

import pytest

from drl.MARL.direct_marl_env import DirectMARLTransition
from drl.MARL import reward_calibration as calibration


def manifest(**changes):
    values = dict(run_id="run", episode_id="ep", domain="navigation", morphology="rc_car8",
                  seed=7000, split="train", difficulty="easy", family=None,
                  collection_policy="bc_v4_stochastic", checkpoint_path="bc.pt",
                  checkpoint_sha256="d" * 64, base_commit="a" * 40, working_tree_dirty=True,
                  physics_hz=240, decision_hz=10, source="physical")
    values.update(changes)
    return calibration.EpisodeManifest(**values)


@dataclass(frozen=True)
class Terms:
    progress: float
    success: float = 0
    collision: float = 0
    time: float = 1


@dataclass(frozen=True)
class Frame:
    team: Terms
    owners: dict


def transition(step=1, progress=1., success=0., terminated=False, truncated=False):
    raw = Frame(Terms(progress, success), {f"m{i}": Terms(progress, success) for i in range(8)})
    return DirectMARLTransition(0, {}, SimpleNamespace(),
        SimpleNamespace(previous_stamp_by_env={0: (step-1)*.1}, next_stamp_by_env={0: step*.1}),
        {}, SimpleNamespace(raw_reward=raw, terminated=terminated, truncated=truncated), step)


def availability(progress=1.):
    return {"progress": {"status": "measured_present" if progress else "measured_absent", "units": "m", "source": "captured_pose"},
            "success": {"status": "measured_absent", "units": "event", "source": "evaluator"},
            "time": {"status": "measured_present", "units": "decision", "source": "evaluator"},
            "collision": {"status": "not_instrumented", "units": "event", "source": "none"}}


def rows(tmp_path):
    recorder = calibration.TransitionRecorder(tmp_path / "raw.jsonl", manifest())
    first = recorder.observe(transition(), availability=availability(), application_provenance={"command_id": "a"})
    second = recorder.observe(transition(2, progress=-1., terminated=True), availability=availability())
    return [first, second]


def test_recorder_uses_captured_completed_transition_and_jsonl(tmp_path):
    recorded = rows(tmp_path)
    assert recorded[0]["team_raw_terms"]["progress"] == 1.
    assert recorded[0]["term_availability"]["collision"]["status"] == "not_instrumented"
    assert recorded[0]["delta_seconds"] == pytest.approx(.1)
    assert recorded[0]["application_provenance"] == {"command_id": "a"}
    disk = [json.loads(line) for line in (tmp_path / "raw.jsonl").read_text().splitlines()]
    assert disk == recorded


@pytest.mark.parametrize("changes", [
    {"split": "validation"}, {"seed": 8000}, {"seed": 12000}, {"seed": True},
    {"seed": 7000.0}, {"difficulty": "hard"}, {"morphology": "snake8"}, {"physics_hz": 120},
    {"domain": "bogus"}, {"checkpoint_sha256": "missing"}, {"source": "unknown"},
])
def test_manifest_fails_closed_on_training_provenance(changes):
    with pytest.raises((TypeError, ValueError)):
        manifest(**changes)


def test_recorder_gates_terminal_and_sequence(tmp_path):
    recorder = calibration.TransitionRecorder(tmp_path / "raw.jsonl", manifest())
    recorder.observe(transition(terminated=True), availability=availability())
    with pytest.raises(ValueError, match="terminal"):
        recorder.observe(transition(2), availability=availability())
    recorder = calibration.TransitionRecorder(tmp_path / "other.jsonl", manifest(episode_id="other"))
    with pytest.raises(ValueError, match="step"):
        recorder.observe(transition(2), availability=availability())


def test_signed_statistics_missing_and_episode_returns_do_not_multiply_owners(tmp_path):
    report = calibration.CalibrationDataset(rows(tmp_path)).report(gamma=.5)
    stats = report["groups"]["navigation/easy/bc_v4_stochastic"]["team_terms"]
    assert stats["progress"]["count"] == 2
    assert stats["progress"]["mean"] == 0
    assert stats["progress"]["std"] == 1
    assert stats["progress"]["positive_fraction"] == .5
    assert stats["progress"]["negative_fraction"] == .5
    assert stats["progress"]["p95"] == pytest.approx(.9)
    assert stats["collision"]["count"] == 0
    assert stats["collision"]["missing_fraction"] == 1
    assert stats["collision"]["mean"] is None
    episode = report["episodes"][0]
    assert episode["length_steps"] == 2
    assert episode["length_seconds"] == pytest.approx(.2)
    assert episode["raw_sum"]["progress"] == 0
    assert episode["discounted_return"]["progress"] == .5
    assert episode["raw_sum"]["collision"] is None
    assert len(report["episodes"]) == 1
    assert report["groups"]["navigation/easy/bc_v4_stochastic"]["correlations"]["pearson"]["time"]["progress"] is None


def test_correlations_rank_ties_and_pairwise_missing():
    result = calibration.term_correlations([{"x": 1., "y": 3.}, {"x": 2., "y": 2.}, {"x": 2., "y": 2.}, {"x": 3., "y": 1.}])
    assert result["pearson"]["x"]["y"] == pytest.approx(-1)
    assert result["spearman"]["x"]["y"] == pytest.approx(-1)


def test_unmeasured_nonfinite_is_null_and_missing_metadata_fails_safe(tmp_path):
    recorder = calibration.TransitionRecorder(tmp_path / "raw.jsonl", manifest())
    row = recorder.observe(transition(), availability={})
    assert row["term_availability"]["progress"]["status"] == "not_instrumented"
    assert calibration.CalibrationDataset([row]).report()["episodes"][0]["raw_sum"]["progress"] is None


def test_synthetic_rows_cannot_be_fit_and_pilot_is_24_training_layouts(tmp_path):
    recorder = calibration.TransitionRecorder(tmp_path / "raw.jsonl", manifest(source="synthetic"))
    row = recorder.observe(transition(), availability=availability())
    with pytest.raises(ValueError, match="physical"):
        calibration.CalibrationDataset([row]).require_physical_train()
    pilot = calibration.pilot_layouts()
    assert len(pilot) == 24
    assert len({(item["domain"], item["seed"]) for item in pilot}) == 24
    assert {item["difficulty"] for item in pilot} == {"easy", "medium", "hard"}
    assert all(item["split"] == "train" for item in pilot)


def test_observer_nonfinite_measurements_are_invalid_not_nan_json(tmp_path):
    recorder = calibration.TransitionRecorder(tmp_path / "raw.jsonl", manifest())
    row = recorder.observe(transition(progress=float("nan")), availability=availability())
    assert row["team_raw_terms"]["progress"] is None
    assert row["term_availability"]["progress"]["status"] == "invalid"
    assert "NaN" not in (tmp_path / "raw.jsonl").read_text()


def test_dataset_rejects_episode_manifest_drift_and_noncontiguous_steps(tmp_path):
    recorded = rows(tmp_path)
    recorded[1]["manifest"]["collection_policy"] = "different_policy"
    with pytest.raises(ValueError, match="manifest"):
        calibration.CalibrationDataset(recorded)
    recorded[1]["manifest"]["collection_policy"] = "bc_v4_stochastic"
    recorded[1]["episode_step"] = 3
    with pytest.raises(ValueError, match="contiguous"):
        calibration.CalibrationDataset(recorded)


def test_dataset_rejects_post_terminal_rows_and_invalid_physical_duration(tmp_path):
    recorded = rows(tmp_path)
    recorded[0]["terminated"] = True
    with pytest.raises(ValueError, match="terminal"):
        calibration.CalibrationDataset(recorded)
    recorded[0]["terminated"] = False
    recorded[1]["delta_seconds"] = 0
    with pytest.raises(ValueError, match="duration"):
        calibration.CalibrationDataset(recorded)


def test_episode_keeps_owner_returns_without_counting_owners_as_episodes(tmp_path):
    report = calibration.CalibrationDataset(rows(tmp_path)).report(gamma=.5)
    assert report["episodes"][0]["owner_raw_sum"]["m0"]["progress"] == 0.
    assert report["episodes"][0]["owner_discounted_return"]["m7"]["progress"] == .5


def test_report_groups_actual_decision_policy_without_changing_episode_manifest(tmp_path):
    recorded = rows(tmp_path)
    for row in recorded:
        row["manifest"]["collection_policy"] = "bc_v4_mixture"
    recorded[0]["application_provenance"]["collection_policy"] = "bc_v4_stochastic"
    report = calibration.CalibrationDataset(recorded).report()
    assert set(report["groups"]) == {"navigation/easy/bc_v4_stochastic", "navigation/easy/bc_v4_mixture"}
    assert report["groups"]["navigation/easy/bc_v4_stochastic"]["team_terms"]["progress"]["count"] == 1
    assert len(report["episodes"]) == 1
    assert report["episodes"][0]["manifest"]["collection_policy"] == "bc_v4_mixture"
