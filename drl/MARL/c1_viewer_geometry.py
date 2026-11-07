"""Read-only diagnostics for the native final-contact docking geometry.

This module is not an action-feasibility oracle.  In particular, `READY` only
means the native *final attach* geometry is within its contact gate; it does
not authorize an earlier ALIGN/DOCK proposal or change its admission mask.
"""
from __future__ import annotations

import math
from typing import Any

from smores_ep.docking.model import DockingThresholds, evaluate_face_pair


def _unknown(reason: str) -> dict[str, Any]:
    return {
        "classification": "UNKNOWN",
        "reason": reason,
        "native_attach_eligible": None,
        "gap_m": None,
        "connector_distance_m": None,
        "lateral_error_m": None,
        "normal_error_rad": None,
        "clocking_error_rad": None,
        "clocking_constrained": None,
        "relative_pose": None,
        "admission_mask_changed": False,
    }


def geometry_record(first, second, thresholds: DockingThresholds | None = None):
    """Inspect two measured docking poses using the *native* evaluator."""
    limits = thresholds if thresholds is not None else DockingThresholds()
    evaluation = evaluate_face_pair(first, second, limits)
    displacement = tuple(
        float(b - a)
        for a, b in zip(first.position_world_m, second.position_world_m, strict=True)
    )
    distance = math.sqrt(sum(component * component for component in displacement))
    face_names = frozenset((first.face.face_name, second.face.face_name))
    contact_limit = (
        limits.top_bottom_contact_tolerance_m
        if face_names == frozenset(("TOP", "BOTTOM"))
        else limits.normal_contact_tolerance_m
    )
    return {
        "classification": (
            "READY" if evaluation.eligible else "NOT_READY_FOR_FINAL_ATTACH"
        ),
        "native_attach_eligible": bool(evaluation.eligible),
        "first_module_id": first.face.module_id,
        "second_module_id": second.face.module_id,
        "first_face": first.face.face_name,
        "second_face": second.face.face_name,
        "gap_m": evaluation.normal_separation_m,
        "connector_distance_m": distance,
        "lateral_error_m": evaluation.lateral_offset_m,
        "normal_error_rad": evaluation.normal_misalignment_rad,
        "clocking_error_rad": evaluation.clocking_error_rad,
        "clocking_residual_rad": evaluation.clocking_residual_rad,
        "clocking_quarter_turns": evaluation.clocking_quarter_turns,
        "clocking_constrained": evaluation.clocking_constrained,
        "native_contact_tolerance_m": contact_limit,
        "native_lateral_tolerance_m": limits.lateral_offset_tolerance_m,
        "native_normal_tolerance_rad": limits.normal_alignment_tolerance_rad,
        "native_clocking_tolerance_rad": limits.clocking_tolerance_rad,
        "relative_pose": {
            "delta_world_m": displacement,
            "first_position_world_m": tuple(first.position_world_m),
            "second_position_world_m": tuple(second.position_world_m),
            "first_outward_normal_world": tuple(first.outward_normal_world),
            "second_outward_normal_world": tuple(second.outward_normal_world),
            "first_tangent_world": tuple(first.tangent_world),
            "second_tangent_world": tuple(second.tangent_world),
        },
        "admission_mask_changed": False,
    }


def capture_dock_geometry(docking_manager, goal) -> dict[str, Any]:
    """Snapshot a requested pair, returning UNKNOWN when poses are unavailable.

    A goal is expected to have `module_ids=(source,target)` and
    `parameters={'face_a': ..., 'face_b': ...}`.  It never starts a docking
    operation or infers positions from a missing/invalid frame.
    """
    module_ids = getattr(goal, "module_ids", None)
    parameters = getattr(goal, "parameters", None)
    if not isinstance(module_ids, (tuple, list)) or len(module_ids) != 2:
        return _unknown("module_ids_unavailable")
    if not isinstance(parameters, dict):
        return _unknown("goal_parameters_unavailable")
    first_face = parameters.get("face_a")
    second_face = parameters.get("face_b")
    if not isinstance(first_face, str) or not isinstance(second_face, str):
        return _unknown("face_selection_unavailable")
    pose_reader = getattr(docking_manager, "face_poses_for", None)
    if not callable(pose_reader):
        return _unknown("native_face_pose_reader_unavailable")

    try:
        first = next(
            (pose for pose in pose_reader(module_ids[0])
             if pose.face.face_name == first_face.upper()), None,
        )
        second = next(
            (pose for pose in pose_reader(module_ids[1])
             if pose.face.face_name == second_face.upper()), None,
        )
    except (KeyError, ValueError, RuntimeError, AttributeError) as error:
        return _unknown(f"native_pose_unavailable:{type(error).__name__}")
    if first is None or second is None:
        return _unknown("requested_face_not_observed")
    return geometry_record(first, second, getattr(docking_manager, "thresholds", None))
