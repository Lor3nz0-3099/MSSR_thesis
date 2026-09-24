from pathlib import Path
from types import SimpleNamespace

import pytest

from mssr_expert.nodes import smores_morphology_behavior_node as behavior_node


class _Parameter:
    def __init__(self, value):
        self.value = value


class _FakeDatasetLogger:
    created_paths = []

    def __init__(self, path: Path):
        self.path = Path(path)
        self.created_paths.append(self.path)


def _fake_node(
    *,
    current_path: Path | None,
    requested_path: Path | None,
    pending=None,
    timestep: int = 7,
    tick: int = 11,
):
    logger = (
        _FakeDatasetLogger(current_path)
        if current_path is not None
        else None
    )

    requested = "" if requested_path is None else str(requested_path)

    node = SimpleNamespace(
        _behavior_dataset_logger=logger,
        _behavior_dataset_pending=pending,
        _behavior_dataset_timestep=timestep,
        _behavior_dataset_tick=tick,
        _command_dataset_path="",
        _command_dataset_episode_id="",
        _command_dataset_stage_name="",
    )
    node.get_parameter = lambda name: _Parameter(
        requested if name == "behavior_dataset_path" else ""
    )
    return node


def test_refresh_behavior_dataset_logger_switches_path_and_resets_counters(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        behavior_node,
        "DatasetLogger",
        _FakeDatasetLogger,
    )
    _FakeDatasetLogger.created_paths.clear()

    first = tmp_path / "stage-a.jsonl"
    second = tmp_path / "stage-b.jsonl"

    node = _fake_node(
        current_path=first,
        requested_path=second,
        pending=None,
        timestep=7,
        tick=11,
    )

    logger = (
        behavior_node.SmoresMorphologyBehaviorNode
        ._refresh_behavior_dataset_logger(node)
    )

    assert logger is node._behavior_dataset_logger
    assert logger.path == second
    assert node._behavior_dataset_timestep == 0
    assert node._behavior_dataset_tick == 0
    assert _FakeDatasetLogger.created_paths[-1] == second


def test_refresh_behavior_dataset_logger_keeps_same_logger_for_same_path(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        behavior_node,
        "DatasetLogger",
        _FakeDatasetLogger,
    )

    path = tmp_path / "stage-a.jsonl"

    node = _fake_node(
        current_path=path,
        requested_path=path,
        pending=None,
        timestep=7,
        tick=11,
    )
    original = node._behavior_dataset_logger

    logger = (
        behavior_node.SmoresMorphologyBehaviorNode
        ._refresh_behavior_dataset_logger(node)
    )

    assert logger is original
    assert node._behavior_dataset_timestep == 7
    assert node._behavior_dataset_tick == 11


def test_refresh_behavior_dataset_logger_rejects_switch_with_pending_transition(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        behavior_node,
        "DatasetLogger",
        _FakeDatasetLogger,
    )

    first = tmp_path / "stage-a.jsonl"
    second = tmp_path / "stage-b.jsonl"

    node = _fake_node(
        current_path=first,
        requested_path=second,
        pending=object(),
        timestep=7,
        tick=11,
    )
    original = node._behavior_dataset_logger

    with pytest.raises(RuntimeError, match="pending"):
        (
            behavior_node.SmoresMorphologyBehaviorNode
            ._refresh_behavior_dataset_logger(node)
        )

    assert node._behavior_dataset_logger is original
    assert node._behavior_dataset_timestep == 7
    assert node._behavior_dataset_tick == 11


def test_behavior_dataset_parameter_callback_flushes_pending_before_path_switch() -> None:
    from pathlib import Path as _Path

    source = (
        _Path(__file__).resolve().parents[1]
        / "mssr_expert/nodes/smores_morphology_behavior_node.py"
    ).read_text(encoding="utf-8")

    assert "def _flush_behavior_dataset_pending(" in source
    assert "def _on_behavior_dataset_parameters(" in source

    flush = source.split(
        "def _flush_behavior_dataset_pending(",
        1,
    )[1].split(
        "def _on_behavior_dataset_parameters(",
        1,
    )[0]

    callback = source.split(
        "def _on_behavior_dataset_parameters(",
        1,
    )[1].split(
        "def _refresh_behavior_dataset_logger(",
        1,
    )[0]

    assert "next_graph=current_graph" in flush
    assert "next_observation=observation" in flush
    assert "self._behavior_dataset_pending = None" in flush
    assert "self._flush_behavior_dataset_pending()" in callback
