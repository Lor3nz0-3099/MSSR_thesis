from __future__ import annotations

import math

import pytest

from smores_ep.docking.model import (
    DockingCommand,
    DockingFace,
    DockingFacePose,
    DockingThresholds,
    evaluate_face_pair,
    select_best_face_pair,
)


def _pose(
    module: str,
    face: str,
    position: tuple[float, float, float],
    normal: tuple[float, float, float],
    tangent: tuple[float, float, float] = (0.0, 1.0, 0.0),
) -> DockingFacePose:
    return DockingFacePose(
        DockingFace(
            module,
            face,
            f"/World/{module}/{face}/frame",
            f"/World/{module}/{face}/body",
        ),
        position,
        normal,
        tangent,
    )


def test_command_parser_supports_legacy_and_explicit_faces() -> None:
    command = DockingCommand.parse("attach active passive")
    assert command.action == "attach"
    assert command.unordered_module_pair == frozenset(("active", "passive"))
    assert not command.explicit_faces

    explicit = DockingCommand.parse(
        "attach active top passive base-chassis"
    )
    assert explicit.first_module == "active"
    assert explicit.first_face == "TOP"
    assert explicit.second_module == "passive"
    assert explicit.second_face == "BOTTOM"
    assert explicit.face_keys == frozenset(
        (("active", "TOP"), ("passive", "BOTTOM"))
    )
    with pytest.raises(ValueError):
        DockingCommand.parse("attach active active")
    with pytest.raises(ValueError):
        DockingCommand.parse("attach active TOP passive")
    with pytest.raises(ValueError):
        DockingCommand.parse("attach active FRONT passive BOTTOM")


def test_base_chassis_alias_is_normalized_to_bottom() -> None:
    face = DockingFace("module", "base-chassis", "/frame", "/body")
    assert face.face_name == "BOTTOM"


def test_opposed_contacting_faces_are_eligible() -> None:
    first = _pose("one", "TOP", (0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    second = _pose(
        "two",
        "BOTTOM",
        (0.001, 0.001, 0.0),
        (-1.0, 0.0, 0.0),
        (0.0, -1.0, 0.0),
    )
    result = evaluate_face_pair(first, second)
    assert result.eligible
    assert result.normal_separation_m == pytest.approx(0.001)
    assert result.lateral_offset_m == pytest.approx(0.001)
    assert result.clocking_quarter_turns in {0, 1, 2, 3}


def test_nominal_frame_gap_does_not_reject_touching_cad_faces() -> None:
    """Regression for a measured BOTTOM-to-BOTTOM contact in Isaac."""

    first = _pose(
        "one",
        "BOTTOM",
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
    )
    second = _pose(
        "two",
        "BOTTOM",
        (0.00538, 0.00093, 0.0),
        (-1.0, 0.0, 0.0),
    )
    result = evaluate_face_pair(first, second)
    assert result.eligible
    assert result.normal_separation_m == pytest.approx(0.00538)
    assert result.lateral_offset_m == pytest.approx(0.00093)


def test_top_bottom_dock_requires_outer_planes_to_be_nearly_coincident() -> None:
    first = _pose("one", "TOP", (0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    visible_gap = _pose(
        "two",
        "BOTTOM",
        (0.0063, 0.0, 0.0),
        (-1.0, 0.0, 0.0),
    )
    compact_contact = _pose(
        "two",
        "BOTTOM",
        (0.0014, 0.0, 0.0),
        (-1.0, 0.0, 0.0),
    )

    assert not evaluate_face_pair(first, visible_gap).eligible
    assert evaluate_face_pair(first, compact_contact).eligible


def test_last_snake_module_accepts_collision_limited_contact() -> None:
    """Regression for the sixth Snake7 connection in wave 3."""

    first = _pose("last", "TOP", (0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    chain = _pose(
        "chain",
        "BOTTOM",
        (0.001388, 0.004236, 0.0),
        (-1.0, 0.0, 0.0),
    )

    result = evaluate_face_pair(first, chain)

    assert result.normal_separation_m == pytest.approx(0.001388)
    assert result.lateral_offset_m == pytest.approx(0.004236)
    assert result.eligible


def test_top_bottom_accepts_calibrated_collision_limited_contact() -> None:
    """A loaded physical contact at 2.406 mm must progress to DOCK."""

    target = _pose("smores_06", "TOP", (0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    observed_contact = _pose(
        "smores_02",
        "BOTTOM",
        (0.002406, 0.004994, 0.0),
        (-1.0, 0.0, 0.0),
    )
    outside_gate = _pose(
        "outside",
        "BOTTOM",
        (0.002501, 0.004994, 0.0),
        (-1.0, 0.0, 0.0),
    )

    accepted = evaluate_face_pair(target, observed_contact)
    assert accepted.normal_separation_m == pytest.approx(0.002406)
    assert accepted.lateral_offset_m == pytest.approx(0.004994)
    assert accepted.eligible
    assert not evaluate_face_pair(target, outside_gate).eligible


def test_snake_top_bottom_contact_accepts_measured_marker_offset() -> None:
    """Regression for Snake7 after the first parallel dock settles."""

    first = _pose("root", "BOTTOM", (0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    observed_contact = _pose(
        "mobile",
        "TOP",
        (0.000481, 0.009163, 0.0),
        (-1.0, 0.0, 0.0),
    )
    outside_gate = _pose(
        "outside",
        "TOP",
        (0.000481, 0.010001, 0.0),
        (-1.0, 0.0, 0.0),
    )

    accepted = evaluate_face_pair(first, observed_contact)
    assert accepted.normal_separation_m == pytest.approx(0.000481)
    assert accepted.lateral_offset_m == pytest.approx(0.009163)
    assert accepted.eligible
    assert not evaluate_face_pair(first, outside_gate).eligible


def test_loaded_bottom_to_lateral_disk_keeps_cad_contact_tolerance() -> None:
    first = _pose("one", "BOTTOM", (0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    second = _pose(
        "two",
        "LEFT",
        (0.00658, 0.00782, 0.0),
        (-1.0, 0.0, 0.0),
    )

    assert evaluate_face_pair(first, second).eligible


def test_gap_alignment_and_square_clocking_are_required() -> None:
    limits = DockingThresholds()
    first = _pose("one", "BOTTOM", (0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    separated = _pose(
        "two",
        "BOTTOM",
        (2.0 * limits.normal_contact_tolerance_m, 0.0, 0.0),
        (-1.0, 0.0, 0.0),
    )
    assert not evaluate_face_pair(first, separated, limits).eligible

    angle = math.radians(45.0)
    misclocked = _pose(
        "two",
        "BOTTOM",
        (0.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0),
        (0.0, math.cos(angle), math.sin(angle)),
    )
    assert not evaluate_face_pair(first, misclocked, limits).eligible




def test_bottom_to_lateral_rotating_disk_does_not_require_clocking() -> None:
    first = _pose(
        "wheel_module",
        "BOTTOM",
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
    )
    angle = math.radians(45.0)
    lateral_disk = _pose(
        "chassis",
        "LEFT",
        (0.0055, 0.001, 0.0),
        (-1.0, 0.0, 0.0),
        (0.0, math.cos(angle), math.sin(angle)),
    )

    result = evaluate_face_pair(first, lateral_disk)

    assert result.clocking_error_rad == pytest.approx(angle)
    assert result.eligible


def test_selection_ignores_occupied_faces_and_uses_best_contact() -> None:
    first = [
        _pose("one", "TOP", (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        _pose(
            "one",
            "LEFT",
            (0.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (1.0, 0.0, 0.0),
        ),
    ]
    second = [
        _pose(
            "two",
            "BOTTOM",
            (0.00005, 0.0, 0.0),
            (-1.0, 0.0, 0.0),
        ),
        _pose(
            "two",
            "RIGHT",
            (0.0, 1.0008, 0.0),
            (0.0, -1.0, 0.0),
            (1.0, 0.0, 0.0),
        ),
    ]
    selected = select_best_face_pair(first, second)
    assert selected is not None
    assert selected.first.face.face_name == "TOP"
    selected = select_best_face_pair(
        first,
        second,
        occupied_faces={("one", "TOP")},
    )
    assert selected is not None
    assert selected.first.face.face_name == "LEFT"
