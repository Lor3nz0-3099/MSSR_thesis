from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SOURCE = (
    ROOT
    / "mssr_ws/src/mssr_expert/mssr_expert/nodes/"
      "smores_parallel_self_assembly_node.py"
)


def test_terminal_dataset_is_finalized_before_terminal_state_is_published():
    text = SOURCE.read_text()

    decision_at = text.index("decision = self._executor.step(")
    terminal_at = text.index("if decision.done:", decision_at)

    publish_at = text.index(
        "self._publish_decision(",
        decision_at,
    )

    terminal_flush_at = text.index(
        "self._flush_pending_transition(",
        terminal_at,
    )

    assert terminal_flush_at < publish_at, (
        "Terminal self-assembly state is published before the terminal "
        "dataset transition is flushed; teleop may terminate the expert "
        "before done=True/success=True reaches the JSONL stream."
    )
