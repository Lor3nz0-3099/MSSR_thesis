import json
from pathlib import Path

from mssr_expert.teleop.recording import RecordingManager


def read_manifest(manager):
    return json.loads(manager.manifest_path.read_text(encoding="utf-8"))


def test_start_creates_new_episode_and_running_manifest(tmp_path):
    manager = RecordingManager(
        root=tmp_path,
        git_commit="abc123",
        dataset_rate_hz=10.0,
        queue_capacity=16,
    )

    manager.start(
        episode_id="demo-001",
        started_at=100.0,
        config={"control_rate_hz": 50.0},
    )

    assert manager.recording
    assert manager.episode_dir == tmp_path / "demo-001"
    assert manager.human_path == tmp_path / "demo-001" / "human_behavior.jsonl"

    manifest = read_manifest(manager)

    assert manifest["schema_version"] == "mssr.teleop_recording_manifest.v1"
    assert manifest["episode_id"] == "demo-001"
    assert manifest["git_commit"] == "abc123"
    assert manifest["status"] == "running"
    assert manifest["dataset_rate_hz_configured"] == 10.0
    assert manifest["config"]["control_rate_hz"] == 50.0
    assert manifest["human_stream"]["producer"] == "human_expert"

    # Questo test verifica lo stato RUNNING, ma deve comunque terminare
    # esplicitamente il worker creato durante il test.
    manager.stop(
        ended_at=101.0,
        task_success=None,
    )


def test_records_are_flushed_and_final_counts_are_measured(tmp_path):
    manager = RecordingManager(
        root=tmp_path,
        git_commit="abc123",
        dataset_rate_hz=10.0,
        queue_capacity=16,
    )

    manager.start(
        episode_id="demo-002",
        started_at=100.0,
    )

    manager.enqueue_record({"timestep": 0, "value": "first"})
    manager.enqueue_record({"timestep": 1, "value": "second"})

    manager.stop(
        ended_at=101.0,
        task_success=False,
    )

    assert not manager.recording

    lines = [
        json.loads(line)
        for line in manager.human_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert [row["timestep"] for row in lines] == [0, 1]

    manifest = read_manifest(manager)

    assert manifest["status"] == "completed"

    # Fine della registrazione != successo del task.
    assert manifest["task_success"] is False

    assert manifest["human_stream"]["records"] == 2
    assert manifest["human_stream"]["bytes"] == manager.human_path.stat().st_size


def test_structural_stream_is_referenced_not_copied_into_human_stream(tmp_path):
    structural = tmp_path / "reconfiguration_dataset.jsonl"
    structural.write_text('{"source":"deterministic"}\n', encoding="utf-8")

    manager = RecordingManager(
        root=tmp_path / "recordings",
        git_commit="abc123",
        dataset_rate_hz=10.0,
        queue_capacity=16,
    )

    manager.start(
        episode_id="demo-003",
        started_at=100.0,
    )

    manager.register_structural_stream(
        stream_id="rc-to-snake-001",
        phase="reconfiguration",
        path=structural,
        producer="deterministic_expert",
    )

    manager.stop(
        ended_at=101.0,
        task_success=True,
    )

    manifest = read_manifest(manager)

    assert manifest["structural_streams"] == [
        {
            "stream_id": "rc-to-snake-001",
            "phase": "reconfiguration",
            "producer": "deterministic_expert",
            "path": str(structural),
        }
    ]

    # Il dataset expert resta separato: non deve essere copiato nello stream umano.
    assert "deterministic" not in manager.human_path.read_text(encoding="utf-8")


def test_queue_saturation_latches_failure(tmp_path, monkeypatch):
    import queue
    import pytest

    manager = RecordingManager(
        root=tmp_path,
        git_commit="abc123",
        dataset_rate_hz=10.0,
        queue_capacity=1,
    )
    manager.start(episode_id="demo-full", started_at=100.0)

    def saturated(_record):
        raise queue.Full

    monkeypatch.setattr(manager._queue, "put_nowait", saturated)

    with pytest.raises(RuntimeError, match="writer_queue_saturated"):
        manager.enqueue_record({"timestep": 0})

    assert manager.failed
    assert manager.error == "writer_queue_saturated"

    # Dopo il primo errore il backend non deve accettare altri record.
    with pytest.raises(RuntimeError, match="writer_queue_saturated"):
        manager.enqueue_record({"timestep": 1})

    manager.stop(ended_at=101.0, task_success=False)

    manifest = read_manifest(manager)
    assert manifest["status"] == "failed"
    assert manifest["eligible_for_import"] is False
    assert manifest["error"] == "writer_queue_saturated"


def test_writer_failure_is_not_silent_and_rejects_later_records(
    tmp_path,
    monkeypatch,
):
    import pytest
    import mssr_expert.teleop.recording as recording

    manager = RecordingManager(
        root=tmp_path,
        git_commit="abc123",
        dataset_rate_hz=10.0,
        queue_capacity=8,
    )
    manager.start(episode_id="demo-disk-error", started_at=100.0)

    def broken_json(_record):
        raise OSError("simulated disk failure")

    monkeypatch.setattr(recording, "dumps_json", broken_json)

    manager.enqueue_record({"timestep": 0})

    # Aspetta deterministicamente che il worker abbia processato il record.
    manager._queue.join()

    assert manager.failed
    assert "writer_error:OSError" in manager.error

    with pytest.raises(RuntimeError, match="writer_error"):
        manager.enqueue_record({"timestep": 1})

    manager.stop(ended_at=101.0, task_success=False)

    manifest = read_manifest(manager)
    assert manifest["status"] == "failed"
    assert manifest["eligible_for_import"] is False
    assert manifest["unwritten_records"] >= 1
    assert "simulated disk failure" in manifest["error"]


def test_request_stop_finalizes_in_background(tmp_path):
    manager = RecordingManager(
        root=tmp_path,
        git_commit="abc123",
        dataset_rate_hz=10.0,
        queue_capacity=8,
    )
    manager.start(episode_id="demo-async", started_at=10.0)
    manager.enqueue_record({"timestep": 0})

    manager.request_stop(
        ended_at=11.0,
        task_success=True,
    )

    assert manager.finalizing
    manager.wait_finalized(timeout=2.0)

    assert not manager.recording
    assert not manager.finalizing

    manifest = read_manifest(manager)
    assert manifest["status"] == "completed"
    assert manifest["human_stream"]["records"] == 1
