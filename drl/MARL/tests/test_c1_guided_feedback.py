"""Regression tests for optional C1 target guidance and ALIGN outcomes."""

import math

from drl.MARL.c1_guided_feedback import (
    AlignOutcomeFeedback,
    TargetGraphProgress,
    load_target,
)
from drl.MARL.c1_structure import (
    C1Connection,
    C1Frame,
    C1Module,
    C1PrimitiveEvent,
)


OWNERS = tuple(f"smores_{i:02d}" for i in range(1, 9))
FACES = ("LEFT", "RIGHT", "TOP", "BOTTOM")


def frame(time_s, connections=(), events=(), move=False):
    modules = []

    for index, owner in enumerate(OWNERS):
        x = index * 0.30

        if move and owner == "smores_02":
            x += 0.08

        connectors = tuple(
            (face, (x + 0.01 * number, 0.0, 0.10))
            for number, face in enumerate(FACES)
        )

        modules.append(
            C1Module(
                module_id=owner,
                position=(x, 0.0, 0.10),
                joints=(
                    ("pan.position_rad", 0.0),
                    ("tilt.position_rad", 0.0),
                ),
                connectors=connectors,
            )
        )

    return C1Frame(
        time_s=float(time_s),
        modules=tuple(modules),
        connections=tuple(connections),
        primitive_events=tuple(events),
    )


def physical_target_edge(target, number=0):
    slots = [node["node_id"] for node in target["nodes"]]

    mapping = dict(zip(slots, OWNERS))
    item = target["edges"][number]

    return C1Connection(
        joint_id=f"joint_{number}",
        module_a=mapping[item["module_a_id"]],
        connector_a=item["attributes"]["face_a"],
        module_b=mapping[item["module_b_id"]],
        connector_b=item["attributes"]["face_b"],
        joint_valid=True,
    )


def test_target_graph_progress_paid_only_once():
    target = load_target("navigation")
    tracker = TargetGraphProgress(target, OWNERS)

    gain, fraction = tracker.observe(frame(0))

    assert gain == 0
    assert fraction == 0

    connection = physical_target_edge(target)

    gain, fraction = tracker.observe(
        frame(1, (connection,))
    )

    assert math.isclose(gain, 1 / 7)
    assert math.isclose(fraction, 1 / 7)

    gain, fraction = tracker.observe(
        frame(2, (connection,))
    )

    assert gain == 0
    assert math.isclose(fraction, 1 / 7)

    tracker.observe(frame(3))

    gain, fraction = tracker.observe(
        frame(4, (connection,))
    )

    assert gain == 0
    assert math.isclose(tracker.best_fraction, 1 / 7)


def test_complete_target_graph_reaches_one():
    target = load_target("navigation")
    tracker = TargetGraphProgress(target, OWNERS)

    connections = tuple(
        physical_target_edge(target, number)
        for number in range(7)
    )

    gain, fraction = tracker.observe(
        frame(1, connections)
    )

    assert math.isclose(gain, 1.0)
    assert math.isclose(fraction, 1.0)


def test_alternative_connection_has_no_target_bonus():
    target = load_target("navigation")
    tracker = TargetGraphProgress(target, OWNERS)

    # TOP-TOP non compare tra le relazioni RC-Car8 target.
    alternative = C1Connection(
        joint_id="alternative",
        module_a="smores_01",
        connector_a="TOP",
        module_b="smores_02",
        connector_b="TOP",
        joint_valid=True,
    )

    gain, fraction = tracker.observe(
        frame(1, (alternative,))
    )

    assert gain == 0
    assert fraction == 0


def metadata():
    return {
        "primitive": "align_faces",
        "owner_module_id": "smores_02",
        "connectors": (
            ("smores_02", "BOTTOM"),
            ("smores_03", "RIGHT"),
        ),
    }


def event(goal_id, state, code):
    return C1PrimitiveEvent(
        goal_id=goal_id,
        action="align_faces",
        state=state,
        code=code,
        owner="smores_02",
    )


def test_failed_align_counted_once():
    feedback = AlignOutcomeFeedback()

    feedback.step(
        frame(0),
        {"goal_1": metadata()},
    )

    failed, retries, stagnant = feedback.step(
        frame(
            1,
            events=(event("goal_1", "failed", "TIMEOUT"),),
        ),
        {"goal_1": metadata()},
    )

    assert failed["smores_02"] == 1
    assert retries["smores_02"] == 0

    # Evento terminale duplicato: nessun secondo addebito.
    failed, retries, stagnant = feedback.step(
        frame(
            2,
            events=(event("goal_1", "failed", "TIMEOUT"),),
        ),
        {"goal_1": metadata()},
    )

    assert sum(failed.values()) == 0
    assert feedback.counts["align_failed"] == 1


def test_unchanged_retry_is_penalized():
    feedback = AlignOutcomeFeedback()

    feedback.step(
        frame(0),
        {"goal_1": metadata()},
    )

    feedback.step(
        frame(
            1,
            events=(event("goal_1", "failed", "TIMEOUT"),),
        ),
        {"goal_1": metadata()},
    )

    failed, retries, stagnant = feedback.step(
        frame(2),
        {
            "goal_1": metadata(),
            "goal_2": metadata(),
        },
    )

    assert sum(failed.values()) == 0
    assert retries["smores_02"] == 1
    assert feedback.counts["unproductive_retry"] == 1


def test_geometry_change_allows_retry():
    feedback = AlignOutcomeFeedback()

    feedback.step(
        frame(0),
        {"goal_1": metadata()},
    )

    feedback.step(
        frame(
            1,
            events=(event("goal_1", "failed", "TIMEOUT"),),
        ),
        {"goal_1": metadata()},
    )

    failed, retries, stagnant = feedback.step(
        frame(2, move=True),
        {
            "goal_1": metadata(),
            "goal_2": metadata(),
        },
    )

    assert sum(failed.values()) == 0
    assert sum(retries.values()) == 0


def test_episode_cancellation_not_penalized():
    feedback = AlignOutcomeFeedback()

    feedback.step(
        frame(0),
        {"goal_1": metadata()},
    )

    failed, retries, stagnant = feedback.step(
        frame(
            1,
            events=(
                event(
                    "goal_1",
                    "canceled",
                    "CANCELED_BY_CLIENT",
                ),
            ),
        ),
        {"goal_1": metadata()},
    )

    assert sum(failed.values()) == 0
    assert sum(retries.values()) == 0
    assert feedback.counts["align_failed"] == 0
