import pytest


def test_handoff_starts_with_no_authority():
    from drl.MARL.runtime_handoff import (
        DirectMARLHandoff,
        RuntimeAuthority,
        RuntimePhase,
    )

    state = DirectMARLHandoff()

    assert state.authority is RuntimeAuthority.NONE
    assert state.phase is RuntimePhase.IDLE
    assert state.target_morphology is None
    assert state.last_expert_success is None
    assert not state.failed


def test_begin_expert_claims_expert_authority():
    from drl.MARL.runtime_handoff import (
        DirectMARLHandoff,
        RuntimeAuthority,
        RuntimePhase,
    )

    state = DirectMARLHandoff()

    assert state.begin_expert("rc_car8")

    assert state.authority is RuntimeAuthority.EXPERT_ASSEMBLY
    assert state.phase is RuntimePhase.ASSEMBLING
    assert state.target_morphology == "rc_car8"
    assert state.last_expert_success is None


def test_matching_topology_before_expert_success_cannot_grant_marl():
    from drl.MARL.runtime_handoff import (
        DirectMARLHandoff,
        RuntimeAuthority,
        RuntimePhase,
    )

    state = DirectMARLHandoff()
    assert state.begin_expert("rc_car8")

    assert not state.observe_topology("rc_car8")

    assert state.authority is RuntimeAuthority.EXPERT_ASSEMBLY
    assert state.phase is RuntimePhase.ASSEMBLING


def test_expert_success_releases_expert_but_waits_for_topology():
    from drl.MARL.runtime_handoff import (
        DirectMARLHandoff,
        RuntimeAuthority,
        RuntimePhase,
    )

    state = DirectMARLHandoff()

    assert state.begin_expert("rc_car8")
    assert state.observe_expert_terminal(True)

    assert state.authority is RuntimeAuthority.NONE
    assert state.phase is RuntimePhase.VERIFYING_TOPOLOGY
    assert state.last_expert_success is True
    assert not state.failed


def test_verified_target_topology_grants_marl_only_after_success():
    from drl.MARL.runtime_handoff import (
        DirectMARLHandoff,
        RuntimeAuthority,
        RuntimePhase,
    )

    state = DirectMARLHandoff()

    assert state.begin_expert("rc_car8")
    assert state.observe_expert_terminal(True)
    assert state.observe_topology("rc_car8")

    assert state.authority is RuntimeAuthority.MARL
    assert state.phase is RuntimePhase.MARL_CONTROL


@pytest.mark.parametrize(
    "observed",
    [
        None,
        "snake8",
        "mobile_manipulator8",
        "unknown",
    ],
)
def test_wrong_or_missing_topology_does_not_grant_marl(observed):
    from drl.MARL.runtime_handoff import (
        DirectMARLHandoff,
        RuntimeAuthority,
        RuntimePhase,
    )

    state = DirectMARLHandoff()

    assert state.begin_expert("rc_car8")
    assert state.observe_expert_terminal(True)

    assert not state.observe_topology(observed)

    assert state.authority is RuntimeAuthority.NONE
    assert state.phase is RuntimePhase.VERIFYING_TOPOLOGY


def test_expert_failure_is_terminal_for_episode_and_never_grants_marl():
    from drl.MARL.runtime_handoff import (
        DirectMARLHandoff,
        RuntimeAuthority,
        RuntimePhase,
    )

    state = DirectMARLHandoff()

    assert state.begin_expert("rc_car8")
    assert state.observe_expert_terminal(False)

    assert state.authority is RuntimeAuthority.NONE
    assert state.phase is RuntimePhase.FAILED
    assert state.last_expert_success is False
    assert state.failed

    assert not state.observe_topology("rc_car8")

    assert state.authority is RuntimeAuthority.NONE
    assert state.phase is RuntimePhase.FAILED


def test_expert_terminal_is_consumed_exactly_once():
    from drl.MARL.runtime_handoff import (
        DirectMARLHandoff,
        RuntimeAuthority,
        RuntimePhase,
    )

    state = DirectMARLHandoff()

    assert state.begin_expert("rc_car8")

    assert state.observe_expert_terminal(True)
    assert not state.observe_expert_terminal(True)
    assert not state.observe_expert_terminal(False)

    assert state.authority is RuntimeAuthority.NONE
    assert state.phase is RuntimePhase.VERIFYING_TOPOLOGY
    assert state.last_expert_success is True


def test_second_expert_cannot_start_while_authority_is_owned():
    from drl.MARL.runtime_handoff import DirectMARLHandoff

    state = DirectMARLHandoff()

    assert state.begin_expert("rc_car8")
    assert not state.begin_expert("snake8")

    assert state.target_morphology == "rc_car8"


def test_reset_revokes_authority_and_clears_previous_episode():
    from drl.MARL.runtime_handoff import (
        DirectMARLHandoff,
        RuntimeAuthority,
        RuntimePhase,
    )

    state = DirectMARLHandoff()

    assert state.begin_expert("rc_car8")
    assert state.observe_expert_terminal(True)
    assert state.observe_topology("rc_car8")

    assert state.authority is RuntimeAuthority.MARL

    state.reset()

    assert state.authority is RuntimeAuthority.NONE
    assert state.phase is RuntimePhase.IDLE
    assert state.target_morphology is None
    assert state.last_expert_success is None
    assert not state.failed


def test_empty_target_is_rejected_without_claiming_authority():
    from drl.MARL.runtime_handoff import (
        DirectMARLHandoff,
        RuntimeAuthority,
    )

    state = DirectMARLHandoff()

    with pytest.raises(
        ValueError,
        match="target_morphology",
    ):
        state.begin_expert("")

    assert state.authority is RuntimeAuthority.NONE
