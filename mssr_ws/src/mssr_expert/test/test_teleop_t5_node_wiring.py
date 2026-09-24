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

    # Every deterministic macro belongs to the same episode manifest, but
    # its phase must preserve the real macro identity rather than pretending
    # that assembly / gap / stairs are self-reconfiguration.
    assert "phase=kind" in text
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


def test_t8_node_forwards_structural_kind_and_initial_assembly_target_graph():
    """T8: X+D-pad assembly intent must reach the real structural launcher."""

    text = node_source()

    # Coordinator now invokes the handler with:
    #   (target_morphology, structural_macro_kind)
    assert "def _start_structural_macro(" in text
    assert "self, target_morphology: str, kind: str" in text

    # The node must retain authoritative target-graph paths for
    # all three teleoperated morphologies.
    assert "self._structural_target_graphs" in text
    assert '"rc_car8"' in text
    assert '"snake8"' in text
    assert '"mobile_manipulator8"' in text

    # The actual launcher receives the semantic operation kind.
    assert "kind=kind" in text

    # Initial self-assembly needs the concrete target graph; the
    # reconfiguration path may continue using target_morphology.
    assert "target_graph_path=" in text
    assert "self._structural_target_graphs" in text


def test_t8_node_observes_real_self_assembly_terminal_state():
    """T8: completed initial assembly must release STRUCTURAL_MACRO."""

    text = node_source()

    assert '"/mssr/expert/self_assembly/state"' in text
    assert "def _on_self_assembly_state(" in text

    callback_at = text.index(
        "def _on_self_assembly_state("
    )
    callback_tail = text[callback_at:]

    assert "self._structural_macro.observe_expert_state(" in callback_tail
    assert "state=self.session.state" in callback_tail
    assert "payload=payload" in callback_tail



def test_t8_macro_authority_is_not_downgraded_for_inactive_controllers():
    """A newly detected target must not capture posture during its macro."""

    text = node_source()
    flat = " ".join(text.split())

    assert (
        'inactive_decision = ( decision '
        'if decision.authority == "STRUCTURAL_MACRO" '
        'else SafetyDecision("NONE", False, True, False) )'
        in flat
    )

    assert (
        'if controller == "rc_car8" '
        'else inactive_decision'
        in flat
    )
    assert (
        'if controller == "snake8" '
        'else inactive_decision'
        in flat
    )
    assert (
        'if controller == "mobile_manipulator8" '
        'else inactive_decision'
        in flat
    )



def test_initial_assembly_ready_comes_only_from_fresh_live_graph():
    text = node_source()

    # The same parsed live graph used for morphology detection must
    # explicitly prove that all modules are disconnected.
    assert "is_authoritative_loose_graph" in text
    assert (
        "self._initial_assembly_loose = "
        "is_authoritative_loose_graph(current_graph)"
        in text
    )

    # Loose authority must have the same freshness lease as topology.
    assert "def _initial_assembly_ready(" in text
    assert "self._topology_received_at" in text
    assert "self._topology_observation_timeout_s" in text

    # Every tick updates the ROS-independent session state.
    assert (
        "self.session.state.observe_initial_assembly_ready("
        in text
    )
    assert (
        "self._initial_assembly_ready(now)"
        in text
    )



def test_successful_reconfiguration_resets_source_teleop_lifecycle():
    text = node_source()

    # The source morphology must be captured before launching the expert.
    assert (
        "source_morphology = self.session.state.detected_morphology"
        in text
    )

    # There must be one explicit lifecycle-reset boundary.
    assert "def _reset_runtime_for_morphology_exit(" in text

    assert (
        '"rc_car8": self._rc'
        in text
    )
    assert (
        '"snake8": self._snake'
        in text
    )
    assert (
        '"mobile_manipulator8": self._mm8'
        in text
    )

    assert "reset_for_morphology_exit()" in text

    # Initial self-assembly has no source controller to reset.
    # Only a successfully launched self-reconfiguration performs the reset.
    assert (
        'if kind == "self_reconfiguration":'
        in text
    )
    assert (
        "self._reset_runtime_for_morphology_exit(source_morphology)"
        in text
    )



def test_node_shutdown_finalizes_recording_before_destroying_ros_node():
    """Ctrl+C must finalize the recording before ROS teardown."""
    from pathlib import Path

    source_path = (
        Path(__file__).resolve().parents[1]
        / "mssr_expert"
        / "nodes"
        / "smores_teleop_node.py"
    )
    source = source_path.read_text(
        encoding="utf-8"
    )

    main_start = source.index("def main(")
    main_source = source[main_start:]

    finally_start = main_source.index("    finally:")
    finally_source = main_source[finally_start:]

    shutdown_call = "node._recording.shutdown("
    destroy_call = "node.destroy_node()"

    assert shutdown_call in finally_source
    assert destroy_call in finally_source

    assert (
        finally_source.index(shutdown_call)
        < finally_source.index(destroy_call)
    )

    assert "ended_at=time.time()" in finally_source


def test_successful_reconfiguration_into_rc_arms_post_fold_entry_capture():
    text = node_source()

    start = text.index(
        "def _on_self_reconfiguration_state("
    )

    end = text.find(
        "\n    def ",
        start + 1,
    )

    callback = (
        text[start:]
        if end == -1
        else text[start:end]
    )

    # The structural launcher clears its own target/kind when it consumes
    # the terminal packet, so the node must use the authoritative terminal
    # payload itself.
    assert 'payload.get("target_morphology")' in callback

    # Only a successfully consumed self-reconfiguration into RC-Car8 may
    # establish the post-fold capture boundary.
    assert 'payload.get("success") is True' in callback
    assert 'payload.get("schema_version")' in callback
    assert '"mssr.self_reconfiguration_state.v1"' in callback
    assert '"rc_car8"' in callback

    assert "self._rc.prepare_for_morphology_entry()" in callback

    # The entry boundary must be downstream of authoritative terminal
    # consumption, never before the expert has actually completed.
    consume_at = callback.index(
        "self._structural_macro.observe_expert_state("
    )

    prepare_at = callback.index(
        "self._rc.prepare_for_morphology_entry()"
    )

    assert prepare_at > consume_at


def test_t8_node_routes_morphology_behavior_status_to_macro_lifecycle():
    text = node_source()

    assert '"/mssr/morphology/status"' in text
    assert "def _on_morphology_behavior_status(" in text

    start = text.index(
        "def _on_morphology_behavior_status("
    )
    end = text.find(
        "\n    def ",
        start + 1,
    )

    callback = (
        text[start:]
        if end == -1
        else text[start:end]
    )

    assert "self._structural_macro.observe_expert_state(" in callback


def test_t8_macro_dataset_identity_uses_actual_macro_kind():
    text = node_source()

    start = text.index(
        "def _start_structural_macro("
    )
    end = text.find(
        "\n    def ",
        start + 1,
    )

    block = (
        text[start:]
        if end == -1
        else text[start:end]
    )

    assert 'execution_id = f"teleop-{kind}-' in block
    assert "phase=kind" in block

    assert 'execution_id = f"teleop-reconfiguration-' not in block
    assert 'phase="self_reconfiguration"' not in block


def test_t8_estop_stops_resident_snake_behavior_and_cancels_native_goals():
    text = node_source()

    # Same native primitive cancellation path already used by
    # assembly/reconfiguration.
    assert "cancel_goal_ids = self._structural_macro.interrupt()" in text
    assert "for goal_id in cancel_goal_ids:" in text
    assert "self._cancel.publish(" in text

    # Gap/stairs live in the persistent morphology behavior node, so killing
    # only the spawned command client is insufficient. TELEOP must also send
    # an explicit morphology 'stop' command when E-stop interrupts them.
    assert '"/mssr/morphology/command"' in text
    assert "self._structural_macro.kind" in text

    assert '"snake_gap"' in text
    assert '"snake_stairs"' in text
    assert '"behavior": "stop"' in text
    assert '"morphology": "snake8"' in text

    # Capture macro identity before interrupt(), because interrupt clears it.
    kind_at = text.index(
        "self._structural_macro.kind"
    )
    interrupt_at = text.index(
        "self._structural_macro.interrupt()"
    )

    assert kind_at < interrupt_at
