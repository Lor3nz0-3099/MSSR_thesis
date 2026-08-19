"""File-based JSON publishers for state and graph payloads."""

from __future__ import annotations

from pathlib import Path

from graphs.robot_graph import RobotGraph
from robots.json_codec import graph_to_dict, snapshot_to_dict, to_json
from robots.state_registry import RobotStateSnapshot


class JsonFileStateGraphPublisher:
    """Publish module state and robot graph payloads as JSON files.

    This publisher is intentionally transport-agnostic. A ROS 2 workspace can
    read the same JSON schema and expose it on topics without importing ROS 2
    Python libraries inside Isaac Sim.
    """

    def __init__(self, output_dir: str | Path, write_history: bool = False) -> None:
        """Create the output directory for JSON payloads.

        Args:
            output_dir: Directory where the latest JSON payloads are written.
            write_history: When true, append each combined state/graph payload
                to ``state_graph_history.jsonl`` for offline tests.
        """
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._write_history = write_history
        self._history_path = self._output_dir / "state_graph_history.jsonl"

    def publish(self, snapshot: RobotStateSnapshot, graph: RobotGraph) -> None:
        """Write the latest module state, graph, and combined payload."""
        state_payload = snapshot_to_dict(snapshot)
        graph_payload = graph_to_dict(graph)
        combined_payload = {
            "timestamp": snapshot.timestamp,
            "state": state_payload,
            "graph": graph_payload,
        }
        self._write_json("module_states.json", state_payload)
        self._write_json("robot_graph.json", graph_payload)
        self._write_json("state_graph.json", combined_payload)
        if self._write_history:
            self._append_history(combined_payload)

    def _write_json(self, filename: str, payload: dict[str, object]) -> None:
        """Write one JSON payload using a temporary file and atomic replace."""
        path = self._output_dir / filename
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        temporary_path.write_text(to_json(payload), encoding="utf-8")
        temporary_path.replace(path)

    def publish_task_metrics(self, payload: dict[str, object]) -> None:
        """Write the latest task-level metrics payload."""
        self._write_json("task_metrics.json", payload)

    def _append_history(self, payload: dict[str, object]) -> None:
        """Append one combined state/graph payload as a JSON Lines record."""
        with self._history_path.open("a", encoding="utf-8") as history_file:
            history_file.write(to_json(payload) + "\n")
