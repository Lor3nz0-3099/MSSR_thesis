"""T5 topology detection must be independent from the active controller."""

from mssr_expert.teleop.topology import TeleopTopologyDetector


class FakeMatcher:
    def __init__(self, matches):
        self.matches = set(matches)

    def configuration_assignment(self, current_graph, target_graph):
        key = (current_graph, target_graph)
        return object() if key in self.matches else None


def test_unique_live_match_selects_active_morphology():
    current = object()

    rc = object()
    snake = object()
    mm = object()
    legacy = object()

    catalog = {
        "rc_car8": rc,
        "snake8": snake,
        "mobile_manipulator8": mm,

        # Present in the repository, but intentionally outside
        # the current teleoperation pipeline.
        "bridge8": legacy,
    }

    detector = TeleopTopologyDetector(
        catalog=catalog,
        matcher=FakeMatcher({
            (current, snake),
        }),
    )

    assert detector.detect(current) == "snake8"


def test_legacy_morphology_is_not_admitted_into_teleop():
    current = object()

    rc = object()
    snake = object()
    mm = object()
    legacy = object()

    detector = TeleopTopologyDetector(
        catalog={
            "rc_car8": rc,
            "snake8": snake,
            "mobile_manipulator8": mm,
            "bridge8": legacy,
        },
        matcher=FakeMatcher({
            (current, legacy),
        }),
    )

    assert detector.detect(current) is None


def test_no_live_match_returns_none():
    current = object()

    detector = TeleopTopologyDetector(
        catalog={
            "rc_car8": object(),
            "snake8": object(),
            "mobile_manipulator8": object(),
        },
        matcher=FakeMatcher(set()),
    )

    assert detector.detect(current) is None


def test_ambiguous_live_match_never_guesses():
    current = object()

    rc = object()
    snake = object()
    mm = object()

    detector = TeleopTopologyDetector(
        catalog={
            "rc_car8": rc,
            "snake8": snake,
            "mobile_manipulator8": mm,
        },
        matcher=FakeMatcher({
            (current, rc),
            (current, snake),
        }),
    )

    assert detector.detect(current) is None
