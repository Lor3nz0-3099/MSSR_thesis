"""Authority and topology contracts for the actuator-free T1 shell."""
import importlib
import importlib.util

import pytest


def new_state():
    assert importlib.util.find_spec("mssr_expert.teleop.state") is not None, "missing T1 state machine"
    return importlib.import_module("mssr_expert.teleop.state").TeleopState()


def ready(state, morphology=None, connected=True):
    state.start_ready()
    state.set_connected(connected)
    state.observe_topology(morphology)
    return state


def test_startup_has_no_actuator_authority_even_with_known_topology():
    state = new_state()
    state.set_connected(True)
    state.observe_topology("rc_car8")
    assert state.phase == "STARTUP"
    assert state.active_controller is None
    assert state.authority == "NONE"
    assert not state.begin_macro()
    state.start_ready()
    assert state.phase == "READY"
    assert state.active_controller == "rc_car8"
    assert state.authority == "TELEOP"


def test_request_never_activates_controller_without_physical_observation():
    state = new_state()
    ready(state)
    assert state.request_morphology("snake8")
    assert state.requested_morphology == "snake8"
    assert state.detected_morphology is None
    assert state.active_controller is None
    assert state.authority == "NONE"
    state.observe_topology("snake8")
    assert state.active_controller == "snake8"


def test_failed_macro_follows_detected_morphology_not_requested_morphology():
    state = new_state()
    ready(state, "rc_car8")
    state.request_morphology("snake8")
    assert state.active_controller == "rc_car8"
    assert state.begin_macro()
    assert state.phase == "STRUCTURAL_MACRO"
    assert state.authority == "STRUCTURAL_MACRO"
    state.finish_macro(success=False)
    assert state.phase == "READY"
    assert state.requested_morphology == "snake8"
    assert state.active_controller == "rc_car8"
    assert state.last_macro_success is False


def test_topology_change_mid_macro_does_not_release_macro_authority():
    state = new_state()
    ready(state, "rc_car8")
    state.request_morphology("snake8")
    state.begin_macro()
    state.observe_topology("snake8")
    assert state.authority == "STRUCTURAL_MACRO"
    state.finish_macro(success=True)
    assert state.authority == "TELEOP"
    assert state.active_controller == "snake8"


def test_unsupported_observed_topology_disables_controller():
    state = new_state()
    ready(state, "snake8")
    state.observe_topology("unsupported_shape")
    assert state.detected_morphology is None
    assert state.active_controller is None
    assert state.authority == "NONE"


@pytest.mark.parametrize("name", ["bridge8", "scorpion8", "snake", "", None])
def test_only_three_active_morphologies_may_be_requested(name):
    state = new_state()
    ready(state, "rc_car8")
    state.request_morphology("snake8")
    with pytest.raises(ValueError):
        state.request_morphology(name)
    assert state.requested_morphology == "snake8"


def test_pause_has_priority_and_only_explicit_resume_releases_it():
    state = new_state()
    ready(state, "rc_car8")
    assert state.pause()
    assert not state.pause()
    state.observe_topology("snake8")
    state.set_connected(False)
    state.set_connected(True)
    assert state.phase == "ESTOP_PAUSED"
    assert state.authority == "ESTOP"
    assert not state.request_morphology("mobile_manipulator8")
    assert not state.begin_macro()
    assert state.resume()
    assert not state.resume()
    assert state.phase == "READY"


def test_pause_and_resume_preserve_running_macro():
    state = new_state()
    ready(state, "rc_car8")
    state.request_morphology("snake8")
    state.begin_macro()
    state.pause()
    assert state.macro_active
    assert state.authority == "ESTOP"
    state.resume()
    assert state.phase == "STRUCTURAL_MACRO"
    assert state.authority == "STRUCTURAL_MACRO"


def test_macro_terminal_while_paused_never_resumes_teleop():
    state = new_state()
    ready(state, "snake8")
    state.request_morphology("mobile_manipulator8")
    state.begin_macro()
    state.pause()
    state.finish_macro(success=False)
    assert state.phase == "ESTOP_PAUSED"
    assert state.authority == "ESTOP"


def test_disconnect_keeps_macro_and_recording_running():
    state = new_state()
    ready(state, "rc_car8")
    state.toggle_recording()
    state.request_morphology("snake8")
    state.begin_macro()
    state.set_connected(False)
    assert state.phase == "STRUCTURAL_MACRO"
    assert state.authority == "STRUCTURAL_MACRO"
    assert state.recording
    assert not state.recording_stop_pending
    state.finish_macro(success=True)
    assert state.recording
    assert state.authority == "NONE"


def test_repeated_stop_requests_defer_once_until_true_macro_terminal():
    state = new_state()
    ready(state)
    state.toggle_recording()
    state.request_morphology("rc_car8")
    state.begin_macro()
    state.toggle_recording()
    state.toggle_recording()
    assert state.recording and state.recording_stop_pending
    state.observe_topology("rc_car8")
    assert state.recording
    state.finish_macro(success=True)
    assert not state.recording and not state.recording_stop_pending


def test_start_recording_during_macro_is_allowed():
    state = new_state()
    ready(state)
    state.request_morphology("snake8")
    state.begin_macro()
    state.toggle_recording()
    assert state.recording
    assert not state.recording_stop_pending


def test_recording_toggle_outside_macro_is_immediate():
    state = new_state()
    ready(state)
    state.toggle_recording()
    assert state.recording
    state.toggle_recording()
    assert not state.recording


def test_macro_busy_rejects_a_second_request_and_keeps_original_intent():
    state = new_state()
    ready(state, "rc_car8")
    state.request_morphology("snake8")
    state.begin_macro()
    assert not state.request_morphology("mobile_manipulator8")
    assert not state.begin_macro()
    assert state.requested_morphology == "snake8"


def test_ready_unknown_topology_or_disconnect_never_grants_teleop():
    state = new_state()
    ready(state, "mobile_manipulator8")
    assert state.authority == "TELEOP"
    state.set_connected(False)
    assert state.phase == "READY" and state.authority == "NONE"
    state.set_connected(True)
    state.observe_topology(None)
    assert state.authority == "NONE"


def test_status_exposes_orthogonal_fields_as_a_snapshot():
    state = new_state()
    ready(state, "snake8")
    state.request_morphology("rc_car8")
    state.toggle_recording()
    payload = state.status()
    assert payload["requested_morphology"] == "rc_car8"
    assert payload["detected_morphology"] == "snake8"
    assert payload["active_controller"] == "snake8"
    assert payload["controller_connected"] is True
    assert payload["recording"] is True
    payload["recording"] = False
    assert state.recording


def test_finish_without_running_macro_cannot_mutate_recording_or_state():
    state = new_state()
    ready(state)
    state.toggle_recording()
    assert not state.finish_macro(success=False)
    assert state.last_macro_success is None
    assert state.recording
