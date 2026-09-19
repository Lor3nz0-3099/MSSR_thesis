"""Terminal self-reconfiguration data must be durable before ROS announces done."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
NODE = (
    ROOT
    / "mssr_ws/src/mssr_expert/mssr_expert/nodes/"
      "smores_self_reconfiguration_node.py"
)


def test_terminal_transition_is_flushed_before_terminal_state_is_published():
    text = NODE.read_text(encoding="utf-8")

    start = text.index("    def _step(self) -> None:")
    end = text.index("    def _prepare_decision(", start)
    body = text[start:end]

    pending_at = body.index(
        "self._pending_transition = _PendingTransition("
    )

    # We need a second flush after constructing the current transition.
    # The existing flush at the top of _step only persists the previous step.
    flush_at = body.index(
        "self._flush_pending_transition(current_graph)",
        pending_at,
    )

    publish_at = body.index(
        "self._publish(decision, task_graph)",
        pending_at,
    )

    # For a terminal decision, disk persistence must happen before the
    # done=True expert-state packet can cause T5 to terminate the process.
    assert pending_at < flush_at < publish_at
