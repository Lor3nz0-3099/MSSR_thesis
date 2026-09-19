"""Structural macros must require fresh neutral input before TELEOP resumes."""

from mssr_expert.teleop.safety import SafetyGate


def update(
    gate,
    *,
    at,
    l2=0.0,
    r2=0.0,
    macro=False,
):
    return gate.update(
        connected=True,
        l2=l2,
        r2=r2,
        received_at=at,
        macro_active=macro,
        topology_supported=True,
    )


def test_macro_disarms_previous_teleop_authority_until_fresh_neutral():
    gate = SafetyGate()

    # Initial fresh neutral arms normal teleoperation.
    decision = update(gate, at=10.0)
    assert decision.authority == "TELEOP"
    assert decision.motion_enabled

    # A structural macro takes authority while the human trigger is held.
    decision = update(
        gate,
        at=10.1,
        r2=1.0,
        macro=True,
    )
    assert decision.authority == "STRUCTURAL_MACRO"
    assert not decision.motion_enabled

    # Macro terminal: the still-held trigger must NOT immediately regain motion.
    decision = update(
        gate,
        at=10.2,
        r2=1.0,
        macro=False,
    )
    assert decision.authority == "NONE"
    assert not decision.motion_enabled

    # Releasing the controls in a genuinely newer packet rearms TELEOP.
    decision = update(
        gate,
        at=10.3,
        macro=False,
    )
    assert decision.authority == "TELEOP"
    assert decision.motion_enabled


def test_cached_neutral_from_macro_cannot_satisfy_post_macro_fence():
    gate = SafetyGate()

    assert update(gate, at=20.0).authority == "TELEOP"

    # Even if the controller was neutral during the macro, that packet
    # belongs to structural authority and cannot count as post-macro neutral.
    assert update(
        gate,
        at=20.1,
        macro=True,
    ).authority == "STRUCTURAL_MACRO"

    decision = update(
        gate,
        at=20.1,
        macro=False,
    )

    assert decision.authority == "NONE"
    assert not decision.motion_enabled

    # Only a newer neutral packet may re-arm locomotion.
    decision = update(
        gate,
        at=20.2,
        macro=False,
    )

    assert decision.authority == "TELEOP"
    assert decision.motion_enabled
