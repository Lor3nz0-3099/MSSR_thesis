"""CPU evidence for conservative, owner-attributed action redundancy."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from drl.MARL.c1_structure import C1Frame, C1Module, C1PrimitiveEvent


OWNERS = tuple(f"m{i}" for i in range(8))


def frame(time_s, *, tilt=0., x=0., events=(), rename=False):
    return C1Frame(time_s, tuple(C1Module(
        f"other{i}" if rename else owner, (i * .3 + x, 0., .1),
        joints=(("tilt.position_rad", tilt if i == 0 else 0.),
                ("pan.position_rad", 0.)),
    ) for i, owner in enumerate(OWNERS)), (), primitive_events=tuple(events))


def goal(action="set_tilt", angle=0., owner=OWNERS[0], **extra):
    return dict(primitive=action, owner_module_id=owner, module_ids=(owner,),
                arguments=(SimpleNamespace(name="angle_rad", value=angle),), **extra)


def monitor(**options):
    from drl.MARL.c1_redundancy import RedundancyMonitor
    return RedundancyMonitor(OWNERS, SimpleNamespace(values={"redundancy": {
        "enabled": True, "continuous_window_s": 1., "cooldown_s": 2.,
        **options,
    }}))


def test_legacy_settings_disable_monitor():
    from drl.MARL.c1_redundancy import RedundancyMonitor
    observer = RedundancyMonitor(OWNERS, SimpleNamespace(values={}))
    observer.step(frame(0), {"a": goal()})
    counts, diagnostics = observer.step(frame(1), {"a": goal(), "b": goal()})
    assert sum(counts.values()) == 0
    assert diagnostics["enabled"] is False


@pytest.mark.parametrize("action", ["set_tilt", "set_pan"])
def test_repeated_reached_target_charged_once_per_new_goal(action):
    observer = monitor()
    assert sum(observer.step(frame(0), {"a": goal(action)})[0].values()) == 0
    goals = {"a": goal(action), "b": goal(action)}
    counts, diagnostics = observer.step(frame(.1), goals)
    assert counts == {OWNERS[0]: 1}
    assert diagnostics["events"][0]["reason"] == "repeated_reached_target"
    assert sum(observer.step(frame(.2), goals)[0].values()) == 0


def test_target_that_still_requires_motion_is_not_redundant():
    observer = monitor()
    observer.step(frame(0), {"a": goal(angle=1.)})
    assert not observer.step(frame(.1), {"b": goal(angle=1.)})[0]


@pytest.mark.parametrize("action", ["align_faces", "undock", "reposition", "recover", "dock"])
def test_existing_failure_and_recovery_causes_are_never_duplicated(action):
    observer = monitor(failed_non_align_repeat=True)
    for i in range(3):
        event = C1PrimitiveEvent(str(i), action, "failed", "FAILED", OWNERS[0])
        counts, _ = observer.step(frame(i, events=(event,)), {str(i): goal(action)})
        assert not counts


def test_optional_non_align_failed_repeat_requires_new_goal_same_measured_state():
    observer = monitor(failed_non_align_repeat=True)
    observer.step(frame(0, events=(C1PrimitiveEvent("a", "set_tilt", "failed", "TIMEOUT", OWNERS[0]),)),
                  {"a": goal(angle=1.)})
    counts, diagnostics = observer.step(frame(.1), {"b": goal(angle=1.)})
    assert counts == {OWNERS[0]: 1}
    assert diagnostics["events"][0]["reason"] == "failed_non_align_repeat"
    assert not observer.step(frame(.2), {"b": goal(angle=1.)})[0]


def test_failure_retry_after_physical_change_or_progress_is_not_redundant():
    for changed, progress in ((True, False), (False, True)):
        observer = monitor(failed_non_align_repeat=True)
        observer.step(frame(0, events=(C1PrimitiveEvent("a", "set_tilt", "failed", "TIMEOUT", OWNERS[0]),)),
                      {"a": goal(angle=1.)})
        assert not observer.step(frame(1, x=.1 if changed else 0.),
                                 {"b": goal(angle=1.)}, useful_progress=progress)[0]


def test_external_charge_key_suppresses_duplicate_cause():
    observer = monitor()
    observer.step(frame(0), {"a": goal()})
    # Goal IDs are accepted alongside emitted event keys for existing feedback.
    counts, diagnostics = observer.step(frame(.1), {"b": goal()}, already_charged_keys={"b"})
    assert not counts
    assert diagnostics["suppressed_existing_charge_count"] == 1


def test_plain_held_commands_are_diagnostic_only_even_over_many_ticks():
    observer = monitor()
    command = SimpleNamespace(linear_x_m_s=.2, angular_z_rad_s=0., pan_velocity_rad_s=0.)
    for i in range(90):
        counts, diagnostics = observer.step(frame(i / 30), {}, continuous_commands={OWNERS[0]: command})
        assert not counts
    assert diagnostics["availability"]["continuous_request_identity"] is False
    assert "continuous_request_identity" in diagnostics["noninstrumented"]


def test_identified_stalled_continuous_requests_require_window_and_cooldown():
    observer = monitor()
    def commands(index):
        return {OWNERS[0]: {"request_id": str(index), "command": {
            "linear_x_m_s": .2, "angular_z_rad_s": 0., "pan_velocity_rad_s": 0.}}}
    assert not observer.step(frame(0), {}, continuous_commands=commands(0))[0]
    assert not observer.step(frame(.5), {}, continuous_commands=commands(1))[0]
    counts, diagnostic = observer.step(frame(1.1), {}, continuous_commands=commands(2))
    assert counts == {OWNERS[0]: 1}
    assert diagnostic["events"][0]["reason"] == "sustained_ineffectual_requests"
    assert not observer.step(frame(1.2), {}, continuous_commands=commands(3))[0]


def test_measured_motion_or_useful_progress_prevents_continuous_penalty():
    observer = monitor()
    for i in range(12):
        command = {OWNERS[0]: {"request_id": str(i), "command": {"linear_x_m_s": .2}}}
        assert not observer.step(frame(i * .5, x=i * .03), {}, continuous_commands=command)[0]
    for i in range(12, 24):
        command = {OWNERS[0]: {"request_id": str(i), "command": {"linear_x_m_s": .2}}}
        assert not observer.step(frame(i * .5), {}, continuous_commands=command, useful_progress=True)[0]


def test_posture_cycle_counts_return_once_and_observing_idle_does_not_count():
    observer = monitor()
    observer.step(frame(0), {})
    observer.step(frame(1, tilt=.5), {"a": goal(angle=.5)})
    event = C1PrimitiveEvent("b", "set_tilt", "succeeded", "OK", OWNERS[0])
    counts, diagnostics = observer.step(frame(2, events=(event,)), {"b": goal(angle=0.)})
    assert counts == {OWNERS[0]: 1}
    assert diagnostics["events"][0]["reason"] == "equivalent_state_cycle"
    assert not observer.step(frame(2.1), {"b": goal(angle=0.)})[0]


def test_cycle_state_is_id_independent_and_quantized():
    observer = monitor()
    _, first = observer.step(frame(0), {})
    _, renamed = observer.step(frame(1, tilt=.001, rename=True), {})
    assert first["state_signature"] == renamed["state_signature"]


def test_malformed_metadata_and_missing_measurements_are_safe():
    observer = monitor(failed_non_align_repeat=True)
    bad = {"a": None, "b": {"primitive": "set_tilt", "owner_module_id": OWNERS[0],
                             "arguments": None}, "c": goal(angle=float("nan"))}
    counts, diagnostics = observer.step(frame(0), bad)
    assert not counts
    assert diagnostics["missing_target_metadata_count"] >= 1
    incomplete = replace(frame(1), modules=(), evidence_complete=False)
    assert not observer.step(incomplete, {"d": goal()})[0]


def test_malformed_owner_cannot_crash_or_receive_a_penalty():
    observer = monitor()
    for i in range(3):
        assert not observer.step(frame(i), {str(i): goal(owner=["m0"])})[0]


def test_explicit_recovery_posture_requests_are_exempt():
    observer = monitor()
    for i in range(3):
        assert not observer.step(frame(i), {str(i): goal(recovery=True)})[0]


def test_wrong_action_terminal_event_cannot_create_a_failed_retry_cause():
    observer = monitor(failed_non_align_repeat=True)
    event = C1PrimitiveEvent("a", "align_faces", "failed", "TIMEOUT", OWNERS[0])
    observer.step(frame(0, events=(event,)), {"a": goal(angle=1.)})
    assert not observer.step(frame(.1), {"b": goal(angle=1.)})[0]


def test_missing_preexecution_measurement_cannot_prove_reached_target_repeat():
    observer = monitor()
    observer.step(frame(0), {"a": goal()})
    observer.step(replace(frame(.1), modules=(), evidence_complete=False), {})
    assert not observer.step(frame(.2), {"b": goal()})[0]


def test_multi_module_posture_goal_is_not_inferred_from_owner_joint_alone():
    observer = monitor()
    metadata = goal()
    metadata["module_ids"] = (OWNERS[0], OWNERS[1])
    observer.step(frame(0), {"a": metadata})
    assert not observer.step(frame(1), {"b": metadata})[0]


def test_cycle_progress_clears_return_history_and_cycle_cooldown_is_bounded():
    observer = monitor()
    observer.step(frame(0), {})
    for i, tilt in enumerate((.5, 0., .5, 0.), 1):
        event = C1PrimitiveEvent(str(i), "set_tilt", "succeeded", "OK", OWNERS[0])
        counts, _ = observer.step(frame(i / 10, tilt=tilt, events=(event,)),
                                  {str(i): goal(angle=tilt)})
        assert counts == ({OWNERS[0]: 1} if i == 2 else {})
    observer.step(frame(1, tilt=.5), {"progress": goal(angle=.5)}, useful_progress=True)
    event = C1PrimitiveEvent("after", "set_tilt", "succeeded", "OK", OWNERS[0])
    assert not observer.step(frame(4, events=(event,)), {"after": goal()})[0]


def test_measurement_gap_cannot_create_a_cycle_penalty():
    observer = monitor()
    observer.step(frame(0), {})
    observer.step(frame(1, tilt=.5), {"a": goal(angle=.5)})
    observer.step(replace(frame(2), modules=(), evidence_complete=False), {})
    event = C1PrimitiveEvent("b", "set_tilt", "succeeded", "OK", OWNERS[0])
    assert not observer.step(frame(3, events=(event,)), {"b": goal()})[0]
