"""Tests for the primitive goal/status protocol."""

from __future__ import annotations

import pytest

from mssr_expert.execution.primitive_protocol import (
    GOAL_SCHEMA,
    PrimitiveProtocolError,
    make_align_faces_goal,
    make_dock_goal,
    make_drive_to_pose_goal,
    parse_primitive_statuses,
)


def test_drive_goal_matches_backend_schema() -> None:
    goal = make_drive_to_pose_goal(
        goal_id="drive-001",
        module_id="m0",
        x_m=0.2,
        y_m=-0.1,
        yaw_rad=1.0,
        timeout_s=20.0,
    )

    assert goal.to_dict() == {
        "schema_version": GOAL_SCHEMA,
        "goal_id": "drive-001",
        "primitive": "drive_to_pose",
        "module_ids": ["m0"],
        "parameters": {
            "x_m": 0.2,
            "y_m": -0.1,
            "yaw_rad": 1.0,
        },
        "timeout_s": 20.0,
    }


def test_align_goal_preserves_explicit_faces_and_clocking() -> None:
    goal = make_align_faces_goal(
        goal_id="align-001",
        mobile_module_id="m1",
        mobile_face="bottom",
        parent_module_id="m0",
        parent_face="top",
        clocking_quarter_turns=1,
    )

    assert goal.parameters == {
        "face_a": "BOTTOM",
        "face_b": "TOP",
        "clocking_quarter_turns": 1,
    }

    assert goal.module_ids == ("m1", "m0")


def test_align_goal_can_request_one_bounded_contact_quality_retry() -> None:
    goal = make_align_faces_goal(
        goal_id="align-quality-001",
        mobile_module_id="m1",
        mobile_face="BOTTOM",
        parent_module_id="m0",
        parent_face="TOP",
        clocking_quarter_turns=0,
        contact_quality_planar_tolerance_m=0.0015,
        contact_quality_retry_count=1,
    )

    assert goal.parameters["contact_quality_planar_tolerance_m"] == 0.0015
    assert goal.parameters["contact_quality_retry_count"] == 1


def test_align_goal_preserves_collective_execution_phase() -> None:
    goal = make_align_faces_goal(
        goal_id="align-phase-001",
        mobile_module_id="m1",
        mobile_face="BOTTOM",
        parent_module_id="m0",
        parent_face="TOP",
        clocking_quarter_turns=0,
        execution_phase="approach",
    )

    assert goal.parameters["execution_phase"] == "approach"

    with pytest.raises(
        PrimitiveProtocolError,
        match="execution_phase",
    ):
        make_align_faces_goal(
            goal_id="align-phase-invalid",
            mobile_module_id="m1",
            mobile_face="BOTTOM",
            parent_module_id="m0",
            parent_face="TOP",
            clocking_quarter_turns=0,
            execution_phase="dock",
        )


def test_align_goal_validates_staging_path_fallback_level() -> None:
    goal = make_align_faces_goal(
        goal_id="align-fallback-001",
        mobile_module_id="m1",
        mobile_face="BOTTOM",
        parent_module_id="m0",
        parent_face="TOP",
        clocking_quarter_turns=0,
        staging_path_fallback_level=2,
    )

    assert goal.parameters["staging_path_fallback_level"] == 2

    for invalid_level in (-1, 3, 1.5, True):
        with pytest.raises(
            PrimitiveProtocolError,
            match="staging_path_fallback_level",
        ):
            make_align_faces_goal(
                goal_id=f"align-fallback-invalid-{invalid_level}",
                mobile_module_id="m1",
                mobile_face="BOTTOM",
                parent_module_id="m0",
                parent_face="TOP",
                clocking_quarter_turns=0,
                staging_path_fallback_level=invalid_level,  # type: ignore[arg-type]
            )


def test_dock_goal_uses_rigid_docking_primitive() -> None:
    goal = make_dock_goal(
        goal_id="dock-001",
        mobile_module_id="m1",
        mobile_face="BOTTOM",
        parent_module_id="m0",
        parent_face="TOP",
        clocking_quarter_turns=0,
    )

    assert goal.primitive == "dock"
    assert goal.parameters["face_a"] == "BOTTOM"
    assert goal.parameters["face_b"] == "TOP"


def test_contact_gate_override_is_validated_and_serialized() -> None:
    goal = make_dock_goal(
        goal_id="dock-contact-gate",
        mobile_module_id="m1",
        mobile_face="TOP",
        parent_module_id="m0",
        parent_face="BOTTOM",
        clocking_quarter_turns=0,
        top_bottom_contact_tolerance_m=0.0035,
    )

    assert goal.parameters["top_bottom_contact_tolerance_m"] == 0.0035

    with pytest.raises(PrimitiveProtocolError):
        make_dock_goal(
            goal_id="bad-contact-gate",
            mobile_module_id="m1",
            mobile_face="TOP",
            parent_module_id="m0",
            parent_face="BOTTOM",
            clocking_quarter_turns=0,
            top_bottom_contact_tolerance_m=0.0,
        )


def test_single_status_is_parsed() -> None:
    statuses = parse_primitive_statuses(
        {
            "schema_version": "mssr.primitive_status.v1",
            "goal_id": "drive-001",
            "primitive": "drive_to_pose",
            "state": "succeeded",
            "module_ids": ["m0"],
            "phase": "terminal",
            "progress": 1.0,
            "code": "POSE_REACHED",
        }
    )

    status = statuses["drive-001"]

    assert status.terminal
    assert status.succeeded
    assert not status.failed


def test_status_batch_is_parsed() -> None:
    statuses = parse_primitive_statuses(
        {
            "schema_version": "mssr.primitive_status_batch.v1",
            "stamp_s": 10.0,
            "statuses": [
                {
                    "goal_id": "align-001",
                    "primitive": "align_faces",
                    "state": "running",
                    "module_ids": ["m1", "m0"],
                    "progress": 0.5,
                },
                {
                    "goal_id": "align-002",
                    "primitive": "align_faces",
                    "state": "rejected",
                    "module_ids": ["m2", "m0"],
                    "code": "RESOURCE_BUSY",
                },
            ],
        }
    )

    assert set(statuses) == {
        "align-001",
        "align-002",
    }

    assert not statuses["align-001"].terminal
    assert statuses["align-002"].terminal
    assert statuses["align-002"].failed


def test_invalid_face_is_rejected() -> None:
    with pytest.raises(
        PrimitiveProtocolError,
        match="Invalid face",
    ):
        make_dock_goal(
            goal_id="dock-invalid",
            mobile_module_id="m1",
            mobile_face="FRONT",
            parent_module_id="m0",
            parent_face="TOP",
            clocking_quarter_turns=0,
        )


def test_invalid_status_state_is_rejected() -> None:
    with pytest.raises(
        PrimitiveProtocolError,
        match="invalid state",
    ):
        parse_primitive_statuses(
            {
                "goal_id": "goal-1",
                "primitive": "dock",
                "state": "unknown",
                "module_ids": ["m0", "m1"],
            }
        )
