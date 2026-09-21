"""T7 MobileManipulator8 manual per-module teleoperation contract."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from mssr_expert.behaviors.morphology_library import MorphologyLibrary
from mssr_expert.graph.serialization import (
    attributed_graph_from_dict,
    load_attributed_graph,
)
from smores_ep.primitives.file_channel import ActionFileChannel
from mssr_expert.teleop.rc_car import load_geometry
from mssr_expert.teleop.safety import SafetyDecision
from mssr_expert.teleop.mobile_manipulator import (
    MobileManipulatorRuntime,
)


CONFIG = Path(__file__).parents[1] / "config"

ENABLED = SafetyDecision(
    "TELEOP",
    True,
    False,
    True,
)

STOPPED = SafetyDecision(
    "ESTOP",
    False,
    True,
    False,
)


def graph(*, stamp=1.0, missing=False):
    payload = json.loads(
        (CONFIG / "smores_mobile_manipulator8.json").read_text()
    )

    ids = {
        node["node_id"]: "physical_" + node["node_id"]
        for node in payload["nodes"]
    }

    for index, node in enumerate(payload["nodes"]):
        node["node_id"] = ids[node["node_id"]]

        node["attributes"].update(
            node_type="physical_module",
            is_target_node=False,
            pose={
                "position": [index * 0.08, 0.0, 0.05],
                "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
            actuators={
                "pan": {
                    "position_rad": 0.0,
                    "lower_limit_rad": -1.2,
                    "upper_limit_rad": 1.2,
                },
                "tilt": {
                    "position_rad": 0.0,
                    "lower_limit_rad": -1.2,
                    "upper_limit_rad": 1.2,
                },
            },
        )

    for edge in payload["edges"]:
        edge["module_a_id"] = ids[edge["module_a_id"]]
        edge["module_b_id"] = ids[edge["module_b_id"]]

        edge["attributes"].update(
            is_target_edge=False,
            is_attached=True,
            relation_type="current_connection",
        )

    if missing:
        payload["edges"].pop()

    payload["stamp"] = stamp

    return attributed_graph_from_dict(payload)


def runtime():
    config = yaml.safe_load(
        (CONFIG / "smores_teleop.yaml").read_text()
    ).get("mobile_manipulator", {})

    return MobileManipulatorRuntime(
        MorphologyLibrary.load(
            CONFIG / "smores_morphology_behaviors.json"
        ),
        load_attributed_graph(
            CONFIG / "smores_mobile_manipulator8.json"
        ),
        geometry=load_geometry(),
        **config,
    )


def sample(**values):
    return SimpleNamespace(
        **{
            **dict(
                left_x=0.0,
                left_y=0.0,
                right_x=0.0,
                right_y=0.0,
                r2=0.0,
                l2=0.0,
                command_events=(),
            ),
            **values,
        }
    )


def role_to_module(output):
    return {
        role: module
        for module, role
        in output.actions.module_roles.items()
    }


def command_id(output):
    assert output.envelope is not None
    return json.loads(output.envelope)[
        "expert"
    ]["debug"]["command_id"]


def test_mm8_initial_selection_is_live_end_effector():
    mm8 = runtime()

    assert mm8.observe_graph(
        graph().to_dict(),
        now=1.0,
    )

    out = mm8.step(
        sample(),
        safety=ENABLED,
        now=1.0,
    )

    assert out.actions.intent["control_mode"] == "manual"
    assert out.actions.intent["selected_role"] == "end_effector"

    selected = out.actions.intent["selected_module_id"]

    assert out.actions.module_roles[selected] == "end_effector"


def test_r2_uses_exact_mm8_translation_pair():
    mm8 = runtime()
    mm8.observe_graph(graph().to_dict(), now=1.0)

    out = mm8.step(
        sample(r2=0.75),
        safety=ENABLED,
        now=1.0,
    )

    roles = role_to_module(out)

    expected = {
        roles["front_support"],
        roles["arm_lift"],
    }

    assert set(out.actions.module_actions) == expected

    assert all(
        command["vx"] > 0.0
        for command in out.actions.module_actions.values()
    )


def test_l2_reverses_exact_same_translation_pair():
    mm8 = runtime()
    mm8.observe_graph(graph().to_dict(), now=1.0)

    out = mm8.step(
        sample(l2=0.6),
        safety=ENABLED,
        now=1.0,
    )

    roles = role_to_module(out)

    expected = {
        roles["front_support"],
        roles["arm_lift"],
    }

    assert set(out.actions.module_actions) == expected

    assert all(
        command["vx"] < 0.0
        for command in out.actions.module_actions.values()
    )



def test_right_stick_pan_targets_only_selected_module():
    mm8 = runtime()
    mm8._mode = "manipulation_ready"

    mm8.observe_graph(graph().to_dict(), now=1.0)

    mm8.step(
        sample(),
        safety=ENABLED,
        now=1.0,
    )

    out = mm8.step(
        sample(right_x=1.0),
        safety=ENABLED,
        now=1.1,
    )

    selected = out.actions.intent["selected_module_id"]

    assert out.actions.intent["mode"] == "manipulation_ready"
    assert out.posture.goal is not None
    assert out.posture.goal.primitive == "rotate_pan_by"
    assert out.posture.goal.module_ids == (selected,)

    assert "pan_target_rad" in out.actions.module_actions[selected]

    assert sum(
        "pan_target_rad" in command
        for command in out.actions.module_actions.values()
    ) == 1

    # Manual manipulation never drives the base.
    assert all(
        abs(command.get("vx", 0.0)) < 1.0e-12
        for command in out.actions.module_actions.values()
    )



def test_r1_moves_selection_from_end_effector_to_arm_link():
    mm8 = runtime()
    mm8._mode = "manipulation_ready"

    mm8.observe_graph(graph().to_dict(), now=1.0)

    first = mm8.step(
        sample(),
        safety=ENABLED,
        now=1.0,
    )

    assert first.actions.intent["selected_role"] == "end_effector"

    switched = mm8.step(
        sample(command_events=("next_module",)),
        safety=ENABLED,
        now=1.1,
    )

    assert switched.actions.intent["selected_role"] == "arm_link"



def test_selected_module_tilt_is_manual_not_cartesian_ik():
    mm8 = runtime()
    mm8._mode = "manipulation_ready"

    mm8.observe_graph(graph().to_dict(), now=1.0)

    mm8.step(
        sample(),
        safety=ENABLED,
        now=1.0,
    )

    out = mm8.step(
        sample(right_y=1.0),
        safety=ENABLED,
        now=1.1,
    )

    selected = out.actions.intent["selected_module_id"]

    assert out.actions.intent["mode"] == "manipulation_ready"
    assert out.posture.goal is not None
    assert out.posture.goal.primitive == "set_tilt"
    assert out.posture.goal.module_ids == (selected,)

    assert "tilt_target_rad" in out.actions.module_actions[selected]

    assert "end_effector_target" not in out.actions.intent
    assert "cartesian_target" not in out.actions.intent

    assert all(
        abs(command.get("vx", 0.0)) < 1.0e-12
        for command in out.actions.module_actions.values()
    )


def test_invalid_mm8_topology_fails_closed():
    mm8 = runtime()

    assert not mm8.observe_graph(
        graph(missing=True).to_dict(),
        now=1.0,
    )

    assert mm8.topology(1.0) is None

    out = mm8.step(
        sample(r2=1.0),
        safety=ENABLED,
        now=1.0,
    )

    assert out.envelope is None
    assert out.actions.module_actions == {}


def test_estop_resume_renews_command_identity_after_native_quarantine(
    tmp_path,
):
    mm8 = runtime()

    assert mm8.observe_graph(
        graph(stamp=1.0).to_dict(),
        now=1.0,
    )

    first = mm8.step(
        sample(r2=0.5),
        safety=ENABLED,
        now=1.0,
    )

    first_id = command_id(first)

    action_file = tmp_path / "actions.json"

    action_file.write_text(
        first.envelope,
        encoding="utf-8",
    )

    native = ActionFileChannel(
        action_file,
        timeout_s=1.0,
        ignore_existing=False,
    )

    before_stop = native.commands(1.0)

    assert before_stop

    native.invalidate_modules(
        tuple(before_stop)
    )

    stopped = mm8.step(
        sample(),
        safety=STOPPED,
        now=1.1,
    )

    assert stopped.envelope is None

    assert mm8.observe_graph(
        graph(stamp=2.0).to_dict(),
        now=1.2,
    )

    resumed = mm8.step(
        sample(r2=0.5),
        safety=ENABLED,
        now=1.2,
    )

    resumed_id = command_id(resumed)

    assert resumed_id != first_id

    temporary = action_file.with_suffix(".json.tmp")

    temporary.write_text(
        resumed.envelope,
        encoding="utf-8",
    )

    temporary.replace(action_file)

    after_resume = native.commands(1.2)

    assert set(after_resume) == set(before_stop)

    assert all(
        abs(command.linear_x_m_s) > 1.0e-6
        for command in after_resume.values()
    )

    continued = mm8.step(
        sample(r2=0.5),
        safety=ENABLED,
        now=1.21,
    )

    assert command_id(continued) == resumed_id


def test_trigger_release_emits_explicit_zero_to_both_mm8_locomotors():
    """Dead-man release must overwrite the native non-zero wheel cache."""
    mm8 = runtime()

    mm8.observe_graph(
        graph().to_dict(),
        now=1.0,
    )

    moving = mm8.step(
        sample(r2=0.8),
        safety=ENABLED,
        now=1.0,
    )

    assert moving.envelope is not None
    assert all(
        command["vx"] > 0.0
        for command in moving.actions.module_actions.values()
    )

    released = mm8.step(
        sample(),
        safety=ENABLED,
        now=1.1,
    )

    roles = role_to_module(released)

    expected = {
        roles["front_support"],
        roles["arm_lift"],
    }

    # Releasing propulsion is an explicit zero command, not silence.
    assert set(released.actions.module_actions) == expected

    assert all(
        command["vx"] == pytest.approx(0.0)
        and command["vy"] == pytest.approx(0.0)
        and command["yaw_rate"] == pytest.approx(0.0)
        for command in released.actions.module_actions.values()
    )

    assert released.envelope is not None

    payload = json.loads(released.envelope)

    assert set(payload["locomotion"]) == expected

    assert all(
        command["vx"] == pytest.approx(0.0)
        for command in payload["locomotion"].values()
    )



def test_manual_module_switch_retires_old_goal_before_new_module_moves():
    mm8 = runtime()
    mm8._mode = "manipulation_ready"

    mm8.observe_graph(graph().to_dict(), now=1.0)

    mm8.step(
        sample(),
        safety=ENABLED,
        now=1.0,
    )

    pan = mm8.step(
        sample(right_x=1.0),
        safety=ENABLED,
        now=1.1,
    )

    old_goal = pan.posture.goal

    assert old_goal is not None
    assert old_goal.primitive == "rotate_pan_by"

    switched = mm8.step(
        sample(command_events=("next_module",)),
        safety=ENABLED,
        now=1.12,
    )

    assert switched.actions.intent["selected_role"] == "arm_link"
    assert old_goal.goal_id in switched.posture.cancel_goal_ids

    blocked = mm8.step(
        sample(right_y=1.0),
        safety=ENABLED,
        now=1.14,
    )

    assert blocked.posture.goal is None

    mm8.observe_status(
        {
            "goal_id": old_goal.goal_id,
            "primitive": "rotate_pan_by",
            "state": "canceled",
            "module_ids": list(old_goal.module_ids),
        }
    )

    mm8.step(
        sample(right_y=1.0),
        safety=ENABLED,
        now=1.16,
    )

    tilt = mm8.step(
        sample(right_y=1.0),
        safety=ENABLED,
        now=1.26,
    )

    assert tilt.posture.goal is not None
    assert tilt.posture.goal.primitive == "set_tilt"

    selected = tilt.actions.intent["selected_module_id"]

    assert tilt.posture.goal.module_ids == (selected,)




def test_module_switch_preserves_other_manipulation_ready_holds():
    mm8 = runtime()

    assert mm8.observe_graph(
        graph(stamp=1.0).to_dict(),
        now=1.0,
    )

    known_goals = {}

    first = mm8.step(
        sample(command_events=("home",)),
        safety=ENABLED,
        now=1.0,
    )

    manip, _ = _finish_mm8_mode_transition(
        mm8,
        first,
        expected_mode="manipulation_ready",
        now=1.0,
        known_goals=known_goals,
    )

    assert manip.actions.intent["selected_role"] == "end_effector"

    held_before = dict(
        mm8.posture.effective_targets()
    )

    assert len(held_before) == 8

    # First PAN request must retire the end-effector's retained
    # manipulation_ready TILT hold.
    retire_old = mm8.step(
        sample(right_x=1.0),
        safety=ENABLED,
        now=1.20,
    )

    assert retire_old.posture.cancel_goal_id is not None

    old_id = retire_old.posture.cancel_goal_id
    old = known_goals[old_id]

    mm8.observe_status(
        {
            "goal_id": old.goal_id,
            "primitive": old.primitive,
            "state": "canceled",
            "module_ids": list(old.module_ids),
        }
    )

    # Now PAN becomes the active manual goal on end_effector.
    pan = mm8.step(
        sample(right_x=1.0),
        safety=ENABLED,
        now=1.22,
    )

    assert pan.posture.goal is not None
    assert pan.posture.goal.primitive == "rotate_pan_by"

    outgoing_goal_id = pan.posture.goal.goal_id

    switched = mm8.step(
        sample(command_events=("next_module",)),
        safety=ENABLED,
        now=1.24,
    )

    assert switched.actions.intent["selected_role"] == "arm_link"

    canceled = set(switched.posture.cancel_goal_ids)

    if switched.posture.cancel_goal_id is not None:
        canceled.add(switched.posture.cancel_goal_id)

    # Switching selection retires only the outgoing module's
    # active manual primitive. Structural posture holds remain.
    assert canceled == {outgoing_goal_id}

    held_after = dict(
        mm8.posture.effective_targets()
    )

    assert set(held_after) >= (
        set(held_before)
        - {pan.posture.goal.module_ids[0]}
    )


def test_retiring_retained_goal_is_not_republished_before_cancel_ack():
    from mssr_expert.teleop.mobile_manipulator import (
        MobileManipulatorActions,
        MobileManipulatorPostureTransport,
    )

    transport = MobileManipulatorPostureTransport(
        retry_s=0.25,
        target_deadband_rad=0.04,
        retarget_rad=0.20,
    )

    initial = MobileManipulatorActions(
        joint_targets=(
            ("module_a", "tilt", 0.0, 0.0),
        ),
        allow_joint_updates=True,
        manual_override=True,
    )

    emitted = transport.step(
        initial,
        now=0.0,
    )

    assert emitted.goal is not None
    old_goal = emitted.goal

    transport.observe(
        {
            "goal_id": old_goal.goal_id,
            "primitive": old_goal.primitive,
            "state": "succeeded",
            "module_ids": list(old_goal.module_ids),
        }
    )

    # Difference 0.10 rad:
    # - above target_deadband 0.04 -> retained goal must be retired
    # - below retarget threshold 0.20 -> exposes the current retry race
    replacement = MobileManipulatorActions(
        joint_targets=(
            ("module_a", "tilt", 0.10, 0.0),
        ),
        allow_joint_updates=True,
        manual_override=True,
    )

    retiring = transport.step(
        replacement,
        now=1.0,
    )

    assert retiring.cancel_goal_id == old_goal.goal_id
    assert retiring.goal is None

    # Native cancellation is asynchronous.  Another 50 Hz tick can occur
    # before CANCELED_BY_CLIENT is observed.
    before_cancel_ack = transport.step(
        replacement,
        now=1.02,
    )

    # A goal already requested for retirement must never be republished.
    assert before_cancel_ack.goal is None

    pending_cancel_ids = set(
        before_cancel_ack.cancel_goal_ids
    )

    if before_cancel_ack.cancel_goal_id is not None:
        pending_cancel_ids.add(
            before_cancel_ack.cancel_goal_id
        )

    assert old_goal.goal_id in pending_cancel_ids

def test_two_right_stick_axes_are_rejected_without_joint_command():
    mm8 = runtime()
    mm8._mode = "manipulation_ready"

    mm8.observe_graph(graph().to_dict(), now=1.0)

    mm8.step(
        sample(),
        safety=ENABLED,
        now=1.0,
    )

    out = mm8.step(
        sample(
            right_x=0.8,
            right_y=0.8,
        ),
        safety=ENABLED,
        now=1.1,
    )

    assert out.posture.goal is None
    assert out.actions.joint_targets == ()

    assert (
        out.actions.intent["manual_rejection"]
        == "one_joint_axis_at_a_time"
    )

    assert all(
        abs(command.get("vx", 0.0)) < 1.0e-12
        for command in out.actions.module_actions.values()
    )



def test_estop_cancels_active_mm8_shape_goal_and_emits_no_motion():
    mm8 = runtime()
    mm8._mode = "manipulation_ready"

    mm8.observe_graph(
        graph().to_dict(),
        now=1.0,
    )

    mm8.step(
        sample(),
        safety=ENABLED,
        now=1.0,
    )

    moving = mm8.step(
        sample(right_y=1.0),
        safety=ENABLED,
        now=1.1,
    )

    active_goal = moving.posture.goal

    assert active_goal is not None

    stopped = mm8.step(
        sample(),
        safety=STOPPED,
        now=1.2,
    )

    assert stopped.envelope is None

    assert all(
        "vx" not in command
        or abs(command.get("vx", 0.0)) < 1.0e-12
        for command in stopped.actions.module_actions.values()
    )

    assert active_goal.goal_id in (
        stopped.posture.cancel_goal_ids
        + (
            (stopped.posture.cancel_goal_id,)
            if stopped.posture.cancel_goal_id is not None
            else ()
        )
    )


def test_stale_mm8_observation_fails_closed():
    mm8 = runtime()

    assert mm8.observe_graph(
        graph().to_dict(),
        now=1.0,
    )

    assert mm8.step(
        sample(r2=1.0),
        safety=ENABLED,
        now=1.1,
    ).envelope is not None

    stale = mm8.step(
        sample(r2=1.0),
        safety=ENABLED,
        now=1.6,
    )

    assert stale.envelope is None
    assert stale.actions.module_actions == {}
    assert mm8.topology(1.6) is None



def test_mm8_resume_recaptures_measured_joint_before_manual_motion():
    """A new TELEOP epoch integrates from measured manipulation posture."""
    mm8 = runtime()
    mm8._mode = "manipulation_ready"

    assert mm8.observe_graph(
        graph(stamp=1.0).to_dict(),
        now=1.0,
    )

    mm8.step(
        sample(),
        safety=ENABLED,
        now=1.0,
    )

    before_stop = mm8.step(
        sample(right_y=1.0),
        safety=ENABLED,
        now=1.1,
    )

    active_goal = before_stop.posture.goal

    assert active_goal is not None
    assert active_goal.primitive == "set_tilt"

    before_id = json.loads(
        before_stop.envelope
    )["expert"]["debug"]["command_id"]

    stopped = mm8.step(
        sample(),
        safety=STOPPED,
        now=1.2,
    )

    assert active_goal.goal_id in (
        stopped.posture.cancel_goal_ids
        + (
            (stopped.posture.cancel_goal_id,)
            if stopped.posture.cancel_goal_id is not None
            else ()
        )
    )

    mm8.observe_status(
        {
            "goal_id": active_goal.goal_id,
            "primitive": "set_tilt",
            "state": "canceled",
            "module_ids": list(active_goal.module_ids),
        }
    )

    payload = graph(stamp=2.0).to_dict()
    selected = active_goal.module_ids[0]

    for node in payload["nodes"]:
        if node["node_id"] == selected:
            node["attributes"]["actuators"]["tilt"][
                "position_rad"
            ] = 0.03
            break
    else:
        raise AssertionError("selected MM8 module missing")

    assert mm8.observe_graph(
        payload,
        now=1.3,
    )

    resumed = mm8.step(
        sample(),
        safety=ENABLED,
        now=1.3,
    )

    resumed_id = json.loads(
        resumed.envelope
    )["expert"]["debug"]["command_id"]

    assert resumed_id != before_id

    moved = mm8.step(
        sample(right_y=1.0),
        safety=ENABLED,
        now=1.4,
    )

    assert moved.posture.goal is not None
    assert moved.posture.goal.primitive == "set_tilt"

    assert moved.posture.goal.parameters[
        "angle_rad"
    ] == pytest.approx(0.08)


def test_mm8_starts_in_drive_ready_scorpion_mode():
    mm8 = runtime()

    assert mm8.observe_graph(
        graph().to_dict(),
        now=1.0,
    )

    out = mm8.step(
        sample(),
        safety=ENABLED,
        now=1.0,
    )

    assert out.actions.intent["mode"] == "drive_ready"
    assert out.actions.intent["mode_transition_pending"] is False


def test_drive_ready_allows_locomotion_but_rejects_manual_shape():
    mm8 = runtime()
    mm8.observe_graph(graph().to_dict(), now=1.0)

    out = mm8.step(
        sample(
            r2=0.5,
            right_x=1.0,
        ),
        safety=ENABLED,
        now=1.1,
    )

    roles = role_to_module(out)

    assert set(out.actions.module_actions) == {
        roles["front_support"],
        roles["arm_lift"],
    }

    assert all(
        command["vx"] > 0.0
        for command in out.actions.module_actions.values()
    )

    assert out.posture.goal is None
    assert out.actions.joint_targets == ()

    assert (
        out.actions.intent["manual_rejection"]
        == "manual_shape_requires_manipulation_ready"
    )


def test_circle_from_drive_ready_starts_prepare_manipulation_and_stops_drive():
    mm8 = runtime()
    mm8.observe_graph(graph().to_dict(), now=1.0)

    mm8.step(
        sample(r2=0.5),
        safety=ENABLED,
        now=1.0,
    )

    transition = mm8.step(
        sample(
            r2=0.5,
            command_events=("home",),
        ),
        safety=ENABLED,
        now=1.1,
    )

    assert transition.actions.intent["mode"] == "to_manipulation_ready"
    assert transition.actions.intent["mode_transition_pending"] is True
    assert (
        transition.actions.intent["mode_transition_behavior"]
        == "prepare_manipulation"
    )

    # Locomotion must be zeroed before posture transition starts.
    assert all(
        abs(command.get("vx", 0.0)) < 1.0e-12
        for command in transition.actions.module_actions.values()
    )

    assert transition.posture.goal is not None


def test_manipulation_ready_blocks_locomotion_and_enables_module_shape():
    mm8 = runtime()

    # Test-only state injection keeps this contract independent from the
    # asynchronous primitive sequencing of the transition itself.
    mm8._mode = "manipulation_ready"

    mm8.observe_graph(graph().to_dict(), now=1.0)

    mm8.step(
        sample(),
        safety=ENABLED,
        now=1.0,
    )

    out = mm8.step(
        sample(
            r2=1.0,
            right_x=1.0,
        ),
        safety=ENABLED,
        now=1.1,
    )

    assert out.actions.intent["mode"] == "manipulation_ready"

    assert out.actions.intent["selected_role"] == "end_effector"

    # Trigger cannot propel the robot in manipulation posture.
    assert all(
        abs(command.get("vx", 0.0)) < 1.0e-12
        for command in out.actions.module_actions.values()
    )

    assert (
        out.actions.intent["locomotion_rejection"]
        == "locomotion_requires_drive_ready"
    )

    assert out.posture.goal is not None
    assert out.posture.goal.primitive == "rotate_pan_by"


def test_circle_from_manipulation_ready_starts_restore_drive():
    mm8 = runtime()
    mm8._mode = "manipulation_ready"

    mm8.observe_graph(graph().to_dict(), now=1.0)

    transition = mm8.step(
        sample(command_events=("home",)),
        safety=ENABLED,
        now=1.1,
    )

    assert transition.actions.intent["mode"] == "to_drive_ready"
    assert transition.actions.intent["mode_transition_pending"] is True

    assert (
        transition.actions.intent["mode_transition_behavior"]
        == "restore_drive"
    )

    assert transition.posture.goal is not None


def test_estop_during_mode_transition_does_not_autoresume_transition():
    mm8 = runtime()
    mm8.observe_graph(graph(stamp=1.0).to_dict(), now=1.0)

    transition = mm8.step(
        sample(command_events=("home",)),
        safety=ENABLED,
        now=1.0,
    )

    assert transition.posture.goal is not None

    stopped = mm8.step(
        sample(),
        safety=STOPPED,
        now=1.1,
    )

    assert stopped.envelope is None

    assert stopped.actions.intent["mode"] == "transition_interrupted"
    assert stopped.actions.intent["mode_transition_pending"] is False

    assert (
        transition.posture.goal.goal_id
        in stopped.posture.cancel_goal_ids
        or transition.posture.goal.goal_id
        == stopped.posture.cancel_goal_id
    )

    assert mm8.observe_graph(
        graph(stamp=2.0).to_dict(),
        now=1.2,
    )

    resumed = mm8.step(
        sample(),
        safety=ENABLED,
        now=1.2,
    )

    # Resume/fresh-neutral cannot restart the posture program.
    assert resumed.actions.intent["mode"] == "transition_interrupted"
    assert resumed.posture.goal is None


def test_manipulation_selector_cycles_only_arm_modules():
    mm8 = runtime()
    mm8._mode = "manipulation_ready"

    mm8.observe_graph(
        graph().to_dict(),
        now=1.0,
    )

    out = mm8.step(
        sample(),
        safety=ENABLED,
        now=1.0,
    )

    observed_roles = [
        out.actions.intent["selected_role"]
    ]

    for index in range(4):
        out = mm8.step(
            sample(
                command_events=("next_module",),
            ),
            safety=ENABLED,
            now=1.1 + index * 0.1,
        )

        observed_roles.append(
            out.actions.intent["selected_role"]
        )

    assert observed_roles == [
        "end_effector",
        "arm_link",
        "arm_lift",
        "arm_ground_drive",
        "end_effector",
    ]

    assert not {
        "chassis_center",
        "left_drive",
        "right_drive",
        "front_support",
    }.intersection(observed_roles)


def test_previous_module_wraps_within_arm_chain_only():
    mm8 = runtime()
    mm8._mode = "manipulation_ready"

    mm8.observe_graph(
        graph().to_dict(),
        now=1.0,
    )

    first = mm8.step(
        sample(),
        safety=ENABLED,
        now=1.0,
    )

    assert first.actions.intent["selected_role"] == "end_effector"

    previous = mm8.step(
        sample(
            command_events=("previous_module",),
        ),
        safety=ENABLED,
        now=1.1,
    )

    assert (
        previous.actions.intent["selected_role"]
        == "arm_ground_drive"
    )


def test_drive_ready_does_not_change_manual_module_selection():
    mm8 = runtime()

    mm8.observe_graph(
        graph().to_dict(),
        now=1.0,
    )

    first = mm8.step(
        sample(),
        safety=ENABLED,
        now=1.0,
    )

    assert first.actions.intent["selected_role"] == "end_effector"

    attempted = mm8.step(
        sample(
            command_events=("next_module",),
        ),
        safety=ENABLED,
        now=1.1,
    )

    assert attempted.actions.intent["selected_role"] == "end_effector"

    assert (
        attempted.actions.intent["manual_rejection"]
        == "module_selection_requires_manipulation_ready"
    )


def _ack_transition_delivery(mm8, delivery, known_goals):
    goal = delivery.goal

    if goal is not None:
        known_goals[goal.goal_id] = goal

        mm8.observe_status(
            {
                "goal_id": goal.goal_id,
                "primitive": goal.primitive,
                "state": "succeeded",
                "module_ids": list(goal.module_ids),
            }
        )

    cancel_ids = list(delivery.cancel_goal_ids)

    if delivery.cancel_goal_id is not None:
        cancel_ids.append(delivery.cancel_goal_id)

    for goal_id in dict.fromkeys(cancel_ids):
        old = known_goals.get(goal_id)

        if old is None:
            raise AssertionError(
                f"transition canceled unknown goal {goal_id}"
            )

        mm8.observe_status(
            {
                "goal_id": old.goal_id,
                "primitive": old.primitive,
                "state": "canceled",
                "module_ids": list(old.module_ids),
            }
        )


def _finish_mm8_mode_transition(
    mm8,
    first,
    *,
    expected_mode,
    now,
    known_goals,
):
    out = first
    emitted = []

    max_steps = max(
        64,
        8 * len(mm8._mode_transition_targets),
    )

    for index in range(max_steps):
        if (
            out.actions.intent.get("mode") == expected_mode
            and not out.actions.intent.get(
                "mode_transition_pending",
                False,
            )
        ):
            return out, emitted

        if out.posture.goal is not None:
            emitted.append(out.posture.goal.goal_id)

        _ack_transition_delivery(
            mm8,
            out.posture,
            known_goals,
        )

        tick_now = now + 0.01 * (index + 1)

        # Native Isaac continuously publishes fresh state graphs while
        # a posture transition is running. Keep the unit-test observation
        # fresh as well; otherwise long restore sequences can exceed the
        # runtime's 0.5 s stale-observation fence.
        next_stamp = (
            mm8.latest_graph.stamp + 1.0e-3
        )

        assert mm8.observe_graph(
            graph(stamp=next_stamp).to_dict(),
            now=tick_now,
        )

        out = mm8.step(
            sample(),
            safety=ENABLED,
            now=tick_now,
        )

    pytest.fail(
        f"MM8 transition did not reach {expected_mode}; "
        f"last mode={out.actions.intent.get('mode')!r}"
    )


def test_prepare_manipulation_reaches_stable_manipulation_ready():
    mm8 = runtime()

    assert mm8.observe_graph(
        graph(stamp=1.0).to_dict(),
        now=1.0,
    )

    known_goals = {}

    first = mm8.step(
        sample(command_events=("home",)),
        safety=ENABLED,
        now=1.0,
    )

    assert first.actions.intent["mode"] == "to_manipulation_ready"
    assert (
        first.actions.intent["mode_transition_behavior"]
        == "prepare_manipulation"
    )

    completed, emitted = _finish_mm8_mode_transition(
        mm8,
        first,
        expected_mode="manipulation_ready",
        now=1.0,
        known_goals=known_goals,
    )

    # manipulation_ready declares one TILT target for every MM8 module.
    assert len(set(emitted)) == 8

    assert completed.actions.intent["mode"] == "manipulation_ready"
    assert completed.actions.intent["mode_transition_pending"] is False
    assert completed.actions.intent["selected_role"] == "end_effector"

    # Stable manipulation mode must remain non-locomotive.
    attempted_drive = mm8.step(
        sample(r2=1.0),
        safety=ENABLED,
        now=1.20,
    )

    assert (
        attempted_drive.actions.intent["locomotion_rejection"]
        == "locomotion_requires_drive_ready"
    )

    assert all(
        abs(command.get("vx", 0.0)) < 1.0e-12
        for command in attempted_drive.actions.module_actions.values()
    )



def test_restore_drive_targets_measured_initial_assembly_posture():
    mm8 = runtime()

    # First valid MM8 observation = physical Scorpion produced by assembly.
    assert mm8.observe_graph(
        graph(stamp=1.0).to_dict(),
        now=1.0,
    )

    assembly_targets = {
        (dof.module_id, dof.name): dof.position_rad
        for dof in mm8._observation.inventory.dofs
        if dof.name in {"pan", "tilt"}
    }

    assert len(assembly_targets) == 16

    # Later observation deliberately represents a different/manipulated pose.
    manipulated = graph(stamp=2.0).to_dict()

    for node in manipulated["nodes"]:
        actuators = node["attributes"]["actuators"]
        actuators["pan"]["position_rad"] = 0.50
        actuators["tilt"]["position_rad"] = -0.50

    assert mm8.observe_graph(
        manipulated,
        now=2.0,
    )

    mm8._mode = "manipulation_ready"

    transition = mm8.step(
        sample(command_events=("home",)),
        safety=ENABLED,
        now=2.1,
    )

    assert transition.actions.intent["mode"] == "to_drive_ready"
    assert (
        transition.actions.intent["mode_transition_behavior"]
        == "restore_drive"
    )

    restore_targets = {
        (target.module_id, target.joint): target.angle_rad
        for target in mm8._mode_transition_targets
    }

    # restore_drive must restore the pose captured at assembly,
    # NOT the current manipulated pose and NOT retreat_drive_base.
    assert restore_targets == pytest.approx(
        assembly_targets,
        abs=1.0e-12,
    )

def test_restore_drive_reaches_stable_scorpion_and_reenables_locomotion():
    mm8 = runtime()

    assert mm8.observe_graph(
        graph(stamp=1.0).to_dict(),
        now=1.0,
    )

    known_goals = {}

    to_manip = mm8.step(
        sample(command_events=("home",)),
        safety=ENABLED,
        now=1.0,
    )

    manip, _ = _finish_mm8_mode_transition(
        mm8,
        to_manip,
        expected_mode="manipulation_ready",
        now=1.0,
        known_goals=known_goals,
    )

    assert manip.actions.intent["mode"] == "manipulation_ready"

    to_drive = mm8.step(
        sample(command_events=("home",)),
        safety=ENABLED,
        now=1.20,
    )

    assert to_drive.actions.intent["mode"] == "to_drive_ready"
    assert (
        to_drive.actions.intent["mode_transition_behavior"]
        == "restore_drive"
    )

    drive, restore_goals = _finish_mm8_mode_transition(
        mm8,
        to_drive,
        expected_mode="drive_ready",
        now=1.20,
        known_goals=known_goals,
    )

    # restore_drive replays the complete PAN/TILT posture captured from
    # the physical Scorpion immediately after assembly.
    assert len(set(restore_goals)) == len(
        mm8._assembly_drive_targets
    )
    assert len(set(restore_goals)) == 16

    assert drive.actions.intent["mode"] == "drive_ready"
    assert drive.actions.intent["mode_transition_pending"] is False

    moving = mm8.step(
        sample(r2=0.5),
        safety=ENABLED,
        now=mm8._received_at + 0.01,
    )

    roles = role_to_module(moving)

    expected = {
        roles["front_support"],
        roles["arm_lift"],
    }

    moving_modules = {
        module
        for module, command in moving.actions.module_actions.items()
        if abs(command.get("vx", 0.0)) > 1.0e-12
    }

    assert moving_modules == expected

    assert all(
        moving.actions.module_actions[module]["vx"] > 0.0
        for module in expected
    )
