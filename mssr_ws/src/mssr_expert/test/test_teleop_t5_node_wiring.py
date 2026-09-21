"""Static T5 integration contracts for the ROS teleoperation shell."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
NODE = (
    ROOT
    / "mssr_ws/src/mssr_expert/mssr_expert/nodes/smores_teleop_node.py"
)


def node_source() -> str:
    return NODE.read_text(encoding="utf-8")


def test_node_declares_all_runtime_controllers_it_really_has():
    text = node_source()

    assert (
        'controller_morphologies={'
        '"rc_car8", "snake8", "mobile_manipulator8"}'
        in text
    )
    assert "self._rc = RcCarRuntime(" in text
    assert "self._snake = SnakeRuntime(" in text
    assert "self._mm8 = MobileManipulatorRuntime(" in text


def test_node_feeds_robot_graph_to_rc_runtime_and_generic_topology_detector():
    text = node_source()

    # RC-Car still needs the full physical graph for its own control state.
    assert "self._rc.observe_graph(payload" in text

    # Morphology identity must no longer be inferred from RcCarRuntime.
    assert "TeleopTopologyDetector" in text
    assert "SmoresSelfReconfigurationPlanner" in text
    assert "attributed_graph_from_dict" in text
    assert "self._topology_detector.detect(" in text


def test_generic_topology_has_a_freshness_lease():
    text = node_source()

    # A stopped /robot_graph stream must remove topology authority rather
    # than leaving the last morphology active forever.
    assert "self._topology_received_at" in text
    assert "topology_observation_timeout_s" in text


def test_tick_does_not_use_rc_runtime_as_global_morphology_identity():
    text = node_source()

    assert (
        "self.session.state.observe_topology(self._rc.topology(now))"
        not in text
    )

    assert "self.session.state.observe_topology(" in text



def test_node_wires_real_structural_macro_launcher():
    text = node_source()

    assert "StructuralMacroLauncher" in text
    assert "self._structural_macro = StructuralMacroLauncher()" in text

    # Coordinator invokes the launcher before its safety decision.
    assert "structural_request_handler=self._start_structural_macro" in text

    assert "def _start_structural_macro(" in text
    assert "self._structural_macro.start(" in text


def test_node_observes_real_self_reconfiguration_terminal_state():
    text = node_source()

    assert '"/mssr/expert/self_reconfiguration/state"' in text
    assert "def _on_self_reconfiguration_state(" in text

    assert "self._structural_macro.observe_expert_state(" in text
    assert "state=self.session.state" in text


def test_estop_interrupts_structural_process_and_cancels_owned_primitives():
    text = node_source()

    assert "self._structural_macro.interrupt()" in text
    assert "cancel_goal_ids" in text

    # Native primitive resources must be explicitly released.
    assert "for goal_id in cancel_goal_ids" in text
    assert "self._cancel.publish(" in text


def test_structural_launcher_receives_unique_execution_and_dataset_identity():
    text = node_source()

    assert "execution_id=" in text
    assert "episode_id=" in text
    assert "dataset_path=" in text

    # Never reuse the fixed legacy/default IDs for interactive macros.
    assert "time.time_ns()" in text



def test_recording_manifest_links_successfully_started_structural_stream():
    text = node_source()

    assert "self._recording.register_structural_stream(" in text
    assert "stream_id=execution_id" in text
    assert 'phase="self_reconfiguration"' in text
    assert "path=dataset_path" in text
    assert 'producer="deterministic_expert"' in text

    # Registration must happen only after the expert process was accepted.
    launch_at = text.index("self._structural_macro.start(")
    register_at = text.index(
        "self._recording.register_structural_stream("
    )
    assert register_at > launch_at


def test_recorded_structural_dataset_uses_same_teleop_episode():
    text = node_source()

    assert "manager.episode_dir" in text
    assert '/ "structural"' in text
    assert "episode_id = recording_episode_id" in text

    # Standalone structural runs remain separate when no T4 recording exists.
    assert "self._structural_dataset_root" in text
    assert 'episode_id = f"teleop-structural-{stamp}"' in text



def test_node_configures_structural_process_exit_grace():
    text = node_source()

    assert '"structural_exit_grace_s"' in text
    assert "self._structural_exit_grace_s" in text


def test_node_polls_structural_process_watchdog_every_tick():
    text = node_source()

    assert "self._structural_macro.check_process(" in text
    assert "state=self.session.state" in text
    assert "now=now" in text
    assert "exit_grace_s=self._structural_exit_grace_s" in text


def test_watchdog_failure_cancels_remaining_owned_primitive_goals():
    text = node_source()

    assert "watchdog_cancel_goal_ids" in text
    assert "if watchdog_cancel_goal_ids is not None:" in text
    assert "for goal_id in watchdog_cancel_goal_ids:" in text
    assert '{"goal_id": goal_id}' in text
