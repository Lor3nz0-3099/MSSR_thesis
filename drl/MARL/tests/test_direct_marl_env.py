from __future__ import annotations

import pytest


def _make_env(
    *,
    launch_result=True,
    expert_status=None,
    topology=None,
):
    from drl.MARL.direct_marl_env import (
        DirectMARLEnv,
    )
    from drl.MARL.runtime_handoff import (
        DirectMARLHandoff,
    )

    handoff = DirectMARLHandoff()

    events = []

    class Bridge:
        def __init__(self):
            self.calls = []

        def step(
            self,
            env_index,
        ):
            self.calls.append(
                env_index
            )

            return {
                "env_index": env_index,
            }

    bridge = Bridge()

    status = {
        "value": expert_status,
    }

    observed_topology = {
        "value": topology,
    }

    def expert_launcher(
        env_index,
        target_morphology,
    ):
        events.append(
            (
                "launch",
                env_index,
                target_morphology,
                handoff.authority,
            )
        )

        return launch_result

    def expert_status_reader(
        env_index,
    ):
        events.append(
            (
                "expert_status",
                env_index,
            )
        )

        return status[
            "value"
        ]

    def topology_observer(
        env_index,
    ):
        events.append(
            (
                "topology",
                env_index,
            )
        )

        return observed_topology[
            "value"
        ]

    env = DirectMARLEnv(
        handoffs=(
            handoff,
        ),
        expert_launcher=(
            expert_launcher
        ),
        expert_status_reader=(
            expert_status_reader
        ),
        topology_observer=(
            topology_observer
        ),
        actuation_bridge=(
            bridge
        ),
        target_morphology=(
            "rc_car8"
        ),
    )

    return (
        env,
        handoff,
        bridge,
        events,
        status,
        observed_topology,
    )


def test_reset_launches_expert_before_claiming_expert_authority():
    from drl.MARL.runtime_handoff import (
        RuntimeAuthority,
        RuntimePhase,
    )

    (
        env,
        handoff,
        _,
        events,
        _,
        _,
    ) = _make_env()

    env.reset()

    assert events[
        0
    ][
        :3
    ] == (
        "launch",
        0,
        "rc_car8",
    )

    # The launcher sees authority NONE.
    # Authority is claimed only after successful launch.
    assert (
        events[
            0
        ][
            3
        ]
        is RuntimeAuthority.NONE
    )

    assert (
        handoff.authority
        is RuntimeAuthority.EXPERT_ASSEMBLY
    )

    assert (
        handoff.phase
        is RuntimePhase.ASSEMBLING
    )


def test_failed_expert_launch_never_claims_authority():
    from drl.MARL.runtime_handoff import (
        RuntimeAuthority,
    )

    (
        env,
        handoff,
        bridge,
        _,
        _,
        _,
    ) = _make_env(
        launch_result=False,
    )

    with pytest.raises(
        RuntimeError,
        match="Expert assembly launch failed",
    ):
        env.reset()

    assert (
        handoff.authority
        is RuntimeAuthority.NONE
    )

    assert bridge.calls == []


def test_expert_success_enters_topology_verification_but_not_marl():
    from drl.MARL.runtime_handoff import (
        RuntimeAuthority,
        RuntimePhase,
    )

    (
        env,
        handoff,
        bridge,
        _,
        status,
        _,
    ) = _make_env()

    env.reset()

    status[
        "value"
    ] = True

    env.advance_handoff(
        0
    )

    assert (
        handoff.authority
        is RuntimeAuthority.NONE
    )

    assert (
        handoff.phase
        is RuntimePhase.VERIFYING_TOPOLOGY
    )

    assert handoff.marl_active is False
    assert bridge.calls == []


def test_wrong_or_missing_topology_cannot_promote_marl():
    from drl.MARL.runtime_handoff import (
        RuntimeAuthority,
        RuntimePhase,
    )

    (
        env,
        handoff,
        bridge,
        _,
        status,
        topology,
    ) = _make_env()

    env.reset()

    status[
        "value"
    ] = True

    env.advance_handoff(
        0
    )

    topology[
        "value"
    ] = None

    env.advance_handoff(
        0
    )

    assert (
        handoff.authority
        is RuntimeAuthority.NONE
    )

    assert (
        handoff.phase
        is RuntimePhase.VERIFYING_TOPOLOGY
    )

    topology[
        "value"
    ] = "snake8"

    env.advance_handoff(
        0
    )

    assert (
        handoff.authority
        is RuntimeAuthority.NONE
    )

    assert handoff.marl_active is False
    assert bridge.calls == []


def test_verified_rc_car8_promotes_marl_only_after_independent_observation():
    from drl.MARL.runtime_handoff import (
        RuntimeAuthority,
        RuntimePhase,
    )

    (
        env,
        handoff,
        bridge,
        _,
        status,
        topology,
    ) = _make_env()

    env.reset()

    status[
        "value"
    ] = True

    # Tick 1:
    # Expert reports success.
    env.advance_handoff(
        0
    )

    assert handoff.marl_active is False

    topology[
        "value"
    ] = "rc_car8"

    # Tick 2:
    # independent topology observer verifies rc_car8.
    env.advance_handoff(
        0
    )

    assert (
        handoff.authority
        is RuntimeAuthority.MARL
    )

    assert (
        handoff.phase
        is RuntimePhase.MARL_CONTROL
    )

    assert handoff.marl_active is True

    # Authority promotion itself must not emit an action.
    assert bridge.calls == []


def test_expert_failure_can_never_promote_marl():
    from drl.MARL.runtime_handoff import (
        RuntimeAuthority,
        RuntimePhase,
    )

    (
        env,
        handoff,
        bridge,
        _,
        status,
        topology,
    ) = _make_env()

    env.reset()

    status[
        "value"
    ] = False

    topology[
        "value"
    ] = "rc_car8"

    env.advance_handoff(
        0
    )

    assert (
        handoff.phase
        is RuntimePhase.FAILED
    )

    assert (
        handoff.authority
        is RuntimeAuthority.NONE
    )

    env.advance_handoff(
        0
    )

    assert handoff.marl_active is False
    assert bridge.calls == []


def test_native_marl_actuation_runs_only_when_marl_already_owned_authority():
    (
        env,
        handoff,
        bridge,
        _,
        status,
        topology,
    ) = _make_env()

    env.reset()

    # Expert assembly: no MARL actuation.
    assert (
        env.step_marl_actuation(
            0
        )
        is None
    )

    assert bridge.calls == []

    status[
        "value"
    ] = True

    env.advance_handoff(
        0
    )

    # Verifying topology: still no MARL actuation.
    assert (
        env.step_marl_actuation(
            0
        )
        is None
    )

    assert bridge.calls == []

    topology[
        "value"
    ] = "rc_car8"

    env.advance_handoff(
        0
    )

    assert handoff.marl_active is True

    result = (
        env.step_marl_actuation(
            0
        )
    )

    assert result == {
        "env_index": 0,
    }

    assert bridge.calls == [
        0
    ]


def test_reset_after_marl_control_revokes_marl_before_relaunching_expert():
    from drl.MARL.runtime_handoff import (
        RuntimeAuthority,
        RuntimePhase,
    )

    (
        env,
        handoff,
        bridge,
        events,
        status,
        topology,
    ) = _make_env()

    env.reset()

    status[
        "value"
    ] = True

    env.advance_handoff(
        0
    )

    topology[
        "value"
    ] = "rc_car8"

    env.advance_handoff(
        0
    )

    assert handoff.marl_active is True

    env.step_marl_actuation(
        0
    )

    assert bridge.calls == [
        0
    ]

    events.clear()

    env.reset()

    # Launcher for the new episode must again observe NONE:
    # old MARL authority was revoked before launch.
    assert (
        events[
            0
        ][
            3
        ]
        is RuntimeAuthority.NONE
    )

    assert (
        handoff.authority
        is RuntimeAuthority.EXPERT_ASSEMBLY
    )

    assert (
        handoff.phase
        is RuntimePhase.ASSEMBLING
    )


def test_vectorized_runtime_keeps_env_authorities_independent():
    from drl.MARL.direct_marl_env import (
        DirectMARLEnv,
    )
    from drl.MARL.runtime_handoff import (
        DirectMARLHandoff,
    )

    handoffs = (
        DirectMARLHandoff(),
        DirectMARLHandoff(),
    )

    statuses = [
        None,
        None,
    ]

    topologies = [
        None,
        None,
    ]

    bridge_calls = []

    class Bridge:
        def step(
            self,
            env_index,
        ):
            bridge_calls.append(
                env_index
            )

            return env_index

    env = DirectMARLEnv(
        handoffs=handoffs,
        expert_launcher=(
            lambda env_index, target:
                True
        ),
        expert_status_reader=(
            lambda env_index:
                statuses[
                    env_index
                ]
        ),
        topology_observer=(
            lambda env_index:
                topologies[
                    env_index
                ]
        ),
        actuation_bridge=Bridge(),
        target_morphology="rc_car8",
    )

    env.reset()

    statuses[
        0
    ] = True

    statuses[
        1
    ] = False

    env.advance_handoff(
        0
    )

    env.advance_handoff(
        1
    )

    topologies[
        0
    ] = "rc_car8"

    env.advance_handoff(
        0
    )

    assert handoffs[
        0
    ].marl_active is True

    assert handoffs[
        1
    ].marl_active is False

    results = (
        env.step_marl_actuation_all()
    )

    assert results == (
        0,
        None,
    )

    assert bridge_calls == [
        0
    ]


@pytest.mark.parametrize('terminal_flag',['terminated','truncated'])
def test_terminal_transition_revokes_authority_per_env_until_reset(terminal_flag):
    from types import SimpleNamespace
    from drl.MARL.direct_marl_env import DirectMARLEnv
    from drl.MARL.runtime_handoff import DirectMARLHandoff
    handoffs=(DirectMARLHandoff(),DirectMARLHandoff())
    calls=[]
    class Bridge:
        def step(self,index):calls.append(index)
    env=DirectMARLEnv(handoffs=handoffs,expert_launcher=lambda i,t:True,
        expert_status_reader=lambda i:True,topology_observer=lambda i:'rc_car8',actuation_bridge=Bridge(),
        state_reader=lambda i:{'x':i},physics_step=lambda:None,
        transition_evaluator=lambda **kw:SimpleNamespace(**{terminal_flag:kw['env_index']==0}))
    env.reset();env.advance_all_handoffs();env.advance_all_handoffs()
    env.step_marl_transitions()
    assert not handoffs[0].marl_active and handoffs[1].marl_active
    assert env.step_marl_actuation(0) is None
    calls.clear();result=env.step_marl_transitions()
    assert calls==[1] and result[0] is None
    env.reset((0,));env.advance_handoff(0);env.advance_handoff(0)
    assert handoffs[0].marl_active


def test_physics_timeout_revokes_batch_authority_without_fabricated_transition():
    from drl.MARL.direct_marl_env import DirectMARLEnv
    from drl.MARL.runtime_handoff import DirectMARLHandoff
    handoff=DirectMARLHandoff();calls=[]
    class Bridge:
        def step(self,index):calls.append(index)
    def physics():raise TimeoutError('missing native completion')
    env=DirectMARLEnv(handoffs=(handoff,),expert_launcher=lambda i,t:True,
        expert_status_reader=lambda i:True,topology_observer=lambda i:'rc_car8',actuation_bridge=Bridge(),
        state_reader=lambda i:{'x':i},physics_step=physics,transition_evaluator=lambda **kw:None)
    env.reset();env.advance_all_handoffs();env.advance_all_handoffs()
    with pytest.raises(TimeoutError):env.step_marl_transitions()
    assert not handoff.marl_active
    assert env.step_marl_actuation(0) is None and calls==[0]
    assert env.episode_steps==(0,)
