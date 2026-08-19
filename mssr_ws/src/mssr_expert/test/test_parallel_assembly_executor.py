"""Tests for the parallel assembly execution state machine."""

from __future__ import annotations

import pytest

from mssr_expert.execution.assembly_policy import (
    DEFAULT_ASSEMBLY_EXECUTION_POLICY,
)
from mssr_expert.execution.parallel_assembly_executor import (
    ParallelAssemblyExecutor,
    physical_fold_push_pairs,
    physical_posture_groups,
)
from mssr_expert.planning.smores_ep.assembly_sequence import (
    AssemblyAction,
    AssemblyWave,
    ParallelAssemblyPlan,
)
from mssr_expert.planning.smores_ep.unfolding import PlanarPose


def _action(
    mobile_id: str = "m1",
    parent_id: str = "m0",
    requires_helper: bool = False,
) -> AssemblyAction:
    return AssemblyAction(
        mobile_module_id=mobile_id,
        mobile_face=(
            "LEFT"
            if requires_helper
            else "BOTTOM"
        ),
        parent_module_id=parent_id,
        parent_face="TOP",
        mobile_target_vertex=f"target_{mobile_id}",
        parent_target_vertex=f"target_{parent_id}",
        depth=1,
        clocking_quarter_turns=0,
        requires_helper=requires_helper,
    )


def _plan(
    actions: tuple[AssemblyAction, ...],
) -> ParallelAssemblyPlan:
    return ParallelAssemblyPlan(
        root_target_vertex="target_m0",
        root_module_id="m0",
        waves=(
            AssemblyWave(
                wave_index=0,
                depth=1,
                phase="ROOT_TOP_BOTTOM",
                actions=actions,
            ),
        ),
    )


def _status(
    goal_id: str,
    primitive: str,
    state: str,
    code: str = "",
) -> dict:
    return {
        "schema_version": "mssr.primitive_status.v1",
        "goal_id": goal_id,
        "primitive": primitive,
        "state": state,
        "module_ids": ["m1", "m0"],
        "code": code,
    }


def _single_goal_for_phase(
    executor: ParallelAssemblyExecutor,
    phase: str,
):
    """Succeed one-action barriers until ``phase`` is dispatched."""

    decision = executor.step()
    for _ in range(12):
        goal = decision.primitive_goal
        assert goal is not None
        if decision.phase == phase:
            return goal
        decision = executor.step(
            _status(goal.goal_id, goal.primitive, "succeeded")
        )
    raise AssertionError(f"Executor did not reach phase {phase}")


def _succeed_parallel_phase(
    executor: ParallelAssemblyExecutor,
    first_decision,
    action_count: int,
):
    """Admit a complete parallel barrier and report its next decision."""

    goals = []
    decision = first_decision
    for _ in range(action_count):
        goal = decision.primitive_goal
        assert goal is not None
        goals.append(goal)
        decision = executor.step(_status(
            goal.goal_id,
            goal.primitive,
            "accepted",
        ))
    assert decision.primitive_goal is None
    decision = executor.step({
        "statuses": [
            _status(goal.goal_id, goal.primitive, "succeeded")
            for goal in goals
        ]
    })
    return goals, decision


def test_single_action_executes_paper_docking_barriers() -> None:
    executor = ParallelAssemblyExecutor(
        _plan((_action(),)),
        execution_id="test",
    )

    decision = executor.step()

    assert decision.primitive_goal is not None
    assert decision.primitive_goal.primitive == "align_faces"
    assert decision.phase == "REACH"
    assert decision.primitive_goal.goal_id == "test-w0-a0-reach"
    assert decision.primitive_goal.parameters["execution_phase"] == "reach"

    decision = executor.step(
        _status(
            "test-w0-a0-reach",
            "align_faces",
            "accepted",
        )
    )

    assert decision.primitive_goal is None
    assert decision.state == "WAITING_REACH_RESULTS"

    decision = executor.step(
        _status(
            "test-w0-a0-reach",
            "align_faces",
            "succeeded",
        )
    )

    assert decision.primitive_goal is not None
    assert decision.phase == "ALIGN"
    assert decision.primitive_goal.primitive == "align_faces"
    assert decision.primitive_goal.parameters["execution_phase"] == "align"

    decision = executor.step(
        _status(decision.primitive_goal.goal_id, "align_faces", "succeeded")
    )
    assert decision.phase == "APPROACH"
    assert decision.primitive_goal is not None
    assert decision.primitive_goal.parameters["execution_phase"] == "approach"

    decision = executor.step(
        _status(decision.primitive_goal.goal_id, "align_faces", "succeeded")
    )
    assert decision.phase == "DOCK"
    assert decision.primitive_goal is not None
    assert decision.primitive_goal.goal_id == "test-w0-a0-dock"

    decision = executor.step(
        _status(
            "test-w0-a0-dock",
            "dock",
            "succeeded",
        )
    )

    assert decision.done
    assert decision.success
    assert decision.state == "SUCCEEDED"
    assert decision.completed_action_count == 1


def test_default_executor_uses_the_shared_morphology_agnostic_policy() -> None:
    executor = ParallelAssemblyExecutor(
        _plan((_action(),)),
        execution_id="default-policy",
    )

    reach = executor.step().primitive_goal

    assert reach is not None
    assert executor.max_concurrent_alignments_per_wave == 0
    assert reach.timeout_s == DEFAULT_ASSEMBLY_EXECUTION_POLICY.align_timeout_s
    assert reach.parameters["execution_phase"] == "reach"
    assert reach.parameters["contact_quality_planar_tolerance_m"] == (
        DEFAULT_ASSEMBLY_EXECUTION_POLICY
        .contact_quality_planar_tolerance_m
    )
    assert reach.parameters["contact_quality_retry_count"] == 2
    assert reach.parameters["top_bottom_contact_tolerance_m"] == 0.004
    assert reach.parameters["contact_approach_feedback"] is True


def test_wave_barriers_keep_independent_modules_parallel() -> None:
    executor = ParallelAssemblyExecutor(
        _plan((_action("m1", "m0"), _action("m2", "m0"))),
        execution_id="parallel-barriers",
    )

    first_reach = executor.step().primitive_goal
    assert first_reach is not None
    assert first_reach.module_ids == ("m1", "m0")

    second_reach = executor.step(
        _status(first_reach.goal_id, "align_faces", "accepted")
    ).primitive_goal
    assert second_reach is not None
    assert second_reach.module_ids == ("m2", "m0")
    assert second_reach.parameters["execution_phase"] == "reach"

    first_done = executor.step(
        {
            "statuses": [
                _status(first_reach.goal_id, "align_faces", "succeeded"),
                _status(second_reach.goal_id, "align_faces", "running"),
            ]
        }
    )
    assert first_done.primitive_goal is None
    assert first_done.phase == "REACH"

    first_align = executor.step(
        {
            "statuses": [
                _status(first_reach.goal_id, "align_faces", "succeeded"),
                _status(second_reach.goal_id, "align_faces", "succeeded"),
            ]
        }
    ).primitive_goal
    assert first_align is not None
    assert first_align.module_ids == ("m1", "m0")
    assert first_align.parameters["execution_phase"] == "align"


def test_contact_quality_policy_is_added_only_to_alignment_goal() -> None:
    executor = ParallelAssemblyExecutor(
        _plan((_action(),)),
        execution_id="quality",
        contact_quality_planar_tolerance_m=0.0015,
        contact_quality_retry_count=1,
    )

    align = executor.step().primitive_goal

    assert align is not None
    assert align.parameters["contact_quality_planar_tolerance_m"] == 0.0015
    assert align.parameters["contact_quality_retry_count"] == 1


def test_morphology_contact_gate_is_shared_with_root_chassis() -> None:
    executor = ParallelAssemblyExecutor(
        _plan((_action(),)),
        execution_id="contact-gate",
        top_bottom_contact_tolerance_m=0.0035,
    )

    align = executor.step().primitive_goal
    assert align is not None
    assert align.parameters["top_bottom_contact_tolerance_m"] == 0.0035

    dock = _single_goal_for_phase(
        ParallelAssemblyExecutor(
            _plan((_action(),)),
            execution_id="contact-gate-dock",
            top_bottom_contact_tolerance_m=0.0035,
        ),
        "DOCK",
    )
    assert dock is not None
    assert dock.primitive == "dock"
    assert dock.parameters["top_bottom_contact_tolerance_m"] == 0.0035


def test_morphology_contact_gate_is_shared_on_loaded_serial_chain() -> None:
    action = _action("m2", "m1")
    action = AssemblyAction(
        **{
            **action.__dict__,
            "mobile_target_vertex": "v2",
            "parent_target_vertex": "v1",
        }
    )
    root_action = _action("m1", "m0")
    executor = ParallelAssemblyExecutor(
        ParallelAssemblyPlan(
            root_target_vertex="target_m0",
            root_module_id="m0",
            waves=(
                AssemblyWave(
                    wave_index=0,
                    depth=1,
                    phase="ROOT_TOP_BOTTOM",
                    actions=(root_action,),
                ),
                AssemblyWave(
                    wave_index=1,
                    depth=2,
                    phase="DEPTH_PARALLEL",
                    actions=(action,),
                ),
            ),
        ),
        execution_id="serial-contact-gate",
        top_bottom_contact_tolerance_m=0.0035,
    )

    root_dock = _single_goal_for_phase(executor, "DOCK")
    serial_reach = executor.step(
        _status(root_dock.goal_id, "dock", "succeeded")
    ).primitive_goal

    assert serial_reach is not None
    assert serial_reach.parameters[
        "top_bottom_contact_tolerance_m"
    ] == 0.0035


def test_parallel_reach_goals_are_dispatched_after_admission() -> None:
    executor = ParallelAssemblyExecutor(
        _plan(
            (
                _action("m1", "m0"),
                _action("m2", "m0"),
            )
        ),
        execution_id="parallel",
        max_concurrent_alignments_per_wave=0,
    )

    first = executor.step()

    assert first.primitive_goal is not None
    assert first.primitive_goal.goal_id == "parallel-w0-a0-reach"

    second = executor.step(
        _status(
            "parallel-w0-a0-reach",
            "align_faces",
            "accepted",
        )
    )

    assert second.primitive_goal is not None
    assert second.primitive_goal.goal_id == (
        "parallel-w0-a1-reach"
    )

    waiting = executor.step(
        _status(
            "parallel-w0-a1-reach",
            "align_faces",
            "accepted",
        )
    )

    assert waiting.primitive_goal is None
    assert set(waiting.active_goal_ids) == {
        "parallel-w0-a0-reach",
        "parallel-w0-a1-reach",
    }

    align_finished = executor.step(
        {
            "schema_version": (
                "mssr.primitive_status_batch.v1"
            ),
            "statuses": [
                _status(
                    "parallel-w0-a0-reach",
                    "align_faces",
                    "succeeded",
                ),
                _status(
                    "parallel-w0-a1-reach",
                    "align_faces",
                    "succeeded",
                ),
            ],
        }
    )

    assert align_finished.primitive_goal is not None
    assert align_finished.phase == "ALIGN"
    assert align_finished.primitive_goal.primitive == "align_faces"
    assert align_finished.primitive_goal.parameters[
        "execution_phase"
    ] == "align"


def test_rejected_primitive_fails_execution() -> None:
    executor = ParallelAssemblyExecutor(
        _plan((_action(),)),
        execution_id="failure",
    )

    executor.step()

    decision = executor.step(
        _status(
            "failure-w0-a0-reach",
            "align_faces",
            "rejected",
            code="RESOURCE_BUSY",
        )
    )

    assert decision.done
    assert not decision.success
    assert decision.state == "FAILED"
    assert "RESOURCE_BUSY" in decision.message


def test_reach_timeout_is_retried_before_execution_fails() -> None:
    executor = ParallelAssemblyExecutor(
        _plan((_action(),)),
        execution_id="retry",
        align_retry_count=1,
    )

    first = executor.step()
    assert first.primitive_goal is not None
    assert first.primitive_goal.timeout_s == 60.0

    retry = executor.step(
        _status(
            "retry-w0-a0-reach",
            "align_faces",
            "failed",
            code="TIMEOUT",
        )
    )

    assert not retry.done
    assert retry.primitive_goal is not None
    assert retry.primitive_goal.goal_id == "retry-w0-a0-reach-r1"

    failed = executor.step(
        _status(
            "retry-w0-a0-reach-r1",
            "align_faces",
            "failed",
            code="TIMEOUT",
        )
    )

    assert failed.done
    assert not failed.success
    assert "TIMEOUT" in failed.message


def test_approach_timeout_realigns_instead_of_pushing_again() -> None:
    executor = ParallelAssemblyExecutor(
        _plan((_action(),)),
        execution_id="approach-recovery",
        align_retry_count=1,
    )
    approach = _single_goal_for_phase(executor, "APPROACH")

    recovery = executor.step(
        _status(
            approach.goal_id,
            "align_faces",
            "failed",
            code="CONTACT_TIMEOUT",
        )
    )

    assert recovery.phase == "ALIGN"
    assert recovery.primitive_goal is not None
    assert recovery.primitive_goal.goal_id == (
        "approach-recovery-w0-a0-align-r1"
    )
    assert recovery.primitive_goal.parameters["execution_phase"] == "align"


def test_blocked_path_is_replanned_after_parallel_peer_settles() -> None:
    executor = ParallelAssemblyExecutor(
        _plan((_action("m1", "m0"), _action("m2", "m0"))),
        execution_id="path-replan",
        align_retry_count=1,
        max_concurrent_alignments_per_wave=0,
    )

    first = executor.step()
    assert first.primitive_goal is not None
    second = executor.step(
        _status(
            "path-replan-w0-a0-reach",
            "align_faces",
            "accepted",
        )
    )
    assert second.primitive_goal is not None
    assert second.primitive_goal.goal_id == "path-replan-w0-a1-reach"

    deferred = executor.step(
        {
            "statuses": [
                _status(
                    "path-replan-w0-a0-reach",
                    "align_faces",
                    "failed",
                    code="NO_COLLISION_FREE_STAGING_PATH",
                ),
                _status(
                    "path-replan-w0-a1-reach",
                    "align_faces",
                    "running",
                ),
            ],
        }
    )
    assert not deferred.done
    assert deferred.primitive_goal is None
    assert deferred.active_goal_ids == ("path-replan-w0-a1-reach",)

    replanned = executor.step(
        _status(
            "path-replan-w0-a1-reach",
            "align_faces",
            "succeeded",
        )
    )
    assert replanned.primitive_goal is not None
    assert replanned.primitive_goal.goal_id == (
        "path-replan-w0-a0-reach-r1"
    )
    assert replanned.primitive_goal.primitive == "align_faces"
    assert replanned.primitive_goal.parameters[
        "staging_path_fallback_level"
    ] == 1


def test_blocked_staging_path_retries_rotate_between_actions() -> None:
    executor = ParallelAssemblyExecutor(
        _plan((_action("m1", "m0"), _action("m2", "m0"))),
        execution_id="path-fairness",
        align_retry_count=2,
        max_concurrent_alignments_per_wave=0,
    )

    executor.step()
    executor.step(
        _status(
            "path-fairness-w0-a0-reach",
            "align_faces",
            "accepted",
        )
    )
    first_retry = executor.step(
        {
            "statuses": [
                _status(
                    "path-fairness-w0-a0-reach",
                    "align_faces",
                    "failed",
                    code="NO_COLLISION_FREE_STAGING_PATH",
                ),
                _status(
                    "path-fairness-w0-a1-reach",
                    "align_faces",
                    "failed",
                    code="NO_COLLISION_FREE_STAGING_PATH",
                ),
            ],
        }
    )
    assert first_retry.primitive_goal is not None
    assert first_retry.primitive_goal.goal_id == (
        "path-fairness-w0-a0-reach-r1"
    )
    assert first_retry.primitive_goal.parameters[
        "staging_path_fallback_level"
    ] == 1

    second_retry = executor.step(
        _status(
            "path-fairness-w0-a0-reach-r1",
            "align_faces",
            "failed",
            code="NO_COLLISION_FREE_STAGING_PATH",
        )
    )
    assert second_retry.primitive_goal is not None
    assert second_retry.primitive_goal.goal_id == (
        "path-fairness-w0-a1-reach-r1"
    )
    assert second_retry.primitive_goal.parameters[
        "staging_path_fallback_level"
    ] == 1


def test_staging_fallback_escalates_and_persists_into_alignment() -> None:
    executor = ParallelAssemblyExecutor(
        _plan((_action(),)),
        execution_id="path-fallback",
        align_retry_count=2,
    )

    initial = executor.step().primitive_goal
    assert initial is not None
    assert "staging_path_fallback_level" not in initial.parameters

    fallback_one = executor.step(
        _status(
            initial.goal_id,
            "align_faces",
            "failed",
            code="NO_COLLISION_FREE_STAGING_PATH",
        )
    ).primitive_goal
    assert fallback_one is not None
    assert fallback_one.parameters["staging_path_fallback_level"] == 1

    fallback_two = executor.step(
        _status(
            fallback_one.goal_id,
            "align_faces",
            "failed",
            code="NO_COLLISION_FREE_STAGING_PATH",
        )
    ).primitive_goal
    assert fallback_two is not None
    assert fallback_two.parameters["staging_path_fallback_level"] == 2

    alignment = executor.step(
        _status(
            fallback_two.goal_id,
            "align_faces",
            "succeeded",
        )
    ).primitive_goal
    assert alignment is not None
    assert alignment.parameters["execution_phase"] == "align"
    assert alignment.parameters["staging_path_fallback_level"] == 2


def test_align_retry_count_must_be_non_negative() -> None:
    try:
        ParallelAssemblyExecutor(
            _plan((_action(),)),
            align_retry_count=-1,
        )
    except ValueError as error:
        assert "align_retry_count" in str(error)
    else:
        raise AssertionError("negative retry count was accepted")


def test_rejected_dock_returns_through_alignment_and_approach() -> None:
    executor = ParallelAssemblyExecutor(
        _plan((_action(),)),
        execution_id="recover",
        dock_recovery_count=1,
    )

    dock = _single_goal_for_phase(executor, "DOCK")
    assert dock.goal_id == "recover-w0-a0-dock"

    alignment_retry = executor.step(
        _status(
            "recover-w0-a0-dock",
            "dock",
            "failed",
            code="DOCKING_REJECTED",
        )
    )
    assert not alignment_retry.done
    assert alignment_retry.primitive_goal is not None
    assert alignment_retry.primitive_goal.goal_id == "recover-w0-a0-align-r1"
    assert alignment_retry.primitive_goal.primitive == "align_faces"
    assert alignment_retry.primitive_goal.parameters[
        "execution_phase"
    ] == "align"

    approach_retry = executor.step(
        _status(
            "recover-w0-a0-align-r1",
            "align_faces",
            "succeeded",
        )
    )
    assert approach_retry.primitive_goal is not None
    assert approach_retry.primitive_goal.goal_id == (
        "recover-w0-a0-approach-r1"
    )

    redock = executor.step(
        _status(
            "recover-w0-a0-approach-r1",
            "align_faces",
            "succeeded",
        )
    )
    assert redock.primitive_goal is not None
    assert redock.primitive_goal.goal_id == "recover-w0-a0-dock-r1"

    finished = executor.step(
        _status("recover-w0-a0-dock-r1", "dock", "succeeded")
    )
    assert finished.done
    assert finished.success
    assert finished.completed_action_count == 1


def test_dock_recovery_preserves_parallel_connection_already_latched() -> None:
    executor = ParallelAssemblyExecutor(
        _plan((_action("m1", "m0"), _action("m2", "m0"))),
        execution_id="parallel-recover",
        dock_recovery_count=1,
        max_concurrent_alignments_per_wave=0,
    )

    _, first_align = _succeed_parallel_phase(
        executor,
        executor.step(),
        2,
    )
    _, first_approach = _succeed_parallel_phase(
        executor,
        first_align,
        2,
    )
    _, first_dock = _succeed_parallel_phase(
        executor,
        first_approach,
        2,
    )
    assert first_dock.primitive_goal is not None
    assert first_dock.primitive_goal.goal_id.endswith("a0-dock")

    second_dock = executor.step(
        _status(
            "parallel-recover-w0-a0-dock",
            "dock",
            "succeeded",
        )
    )
    assert second_dock.primitive_goal is not None
    assert second_dock.primitive_goal.goal_id.endswith("a1-dock")

    realign_second = executor.step(
        _status(
            "parallel-recover-w0-a1-dock",
            "dock",
            "failed",
            code="DOCKING_REJECTED",
        )
    )
    assert realign_second.primitive_goal is not None
    assert realign_second.primitive_goal.goal_id.endswith("a1-align-r1")
    assert realign_second.primitive_goal.primitive == "align_faces"

    approach_second = executor.step(
        _status(
            "parallel-recover-w0-a1-align-r1",
            "align_faces",
            "succeeded",
        )
    )
    assert approach_second.primitive_goal is not None
    assert approach_second.primitive_goal.goal_id.endswith("a1-approach-r1")

    redock_second = executor.step(
        _status(
            "parallel-recover-w0-a1-approach-r1",
            "align_faces",
            "succeeded",
        )
    )
    assert redock_second.primitive_goal is not None
    assert redock_second.primitive_goal.goal_id.endswith("a1-dock-r1")

    finished = executor.step(
        _status(
            "parallel-recover-w0-a1-dock-r1",
            "dock",
            "succeeded",
        )
    )
    assert finished.done
    assert finished.success
    assert finished.completed_action_count == 2


def test_dock_timeout_realigns_instead_of_failing_wave() -> None:
    executor = ParallelAssemblyExecutor(
        _plan((_action(),)),
        execution_id="dock-timeout",
        dock_recovery_count=1,
    )
    dock = _single_goal_for_phase(executor, "DOCK")
    assert dock.goal_id == "dock-timeout-w0-a0-dock"

    retry = executor.step(
        _status(
            "dock-timeout-w0-a0-dock",
            "dock",
            "failed",
            code="TIMEOUT",
        )
    )

    assert not retry.done
    assert retry.primitive_goal is not None
    assert retry.primitive_goal.primitive == "align_faces"
    assert retry.primitive_goal.goal_id == "dock-timeout-w0-a0-align-r1"


def test_helper_action_borrows_future_leaf_and_runs_full_sequence() -> None:
    plan = ParallelAssemblyPlan(
        root_target_vertex="target_m0",
        root_module_id="m0",
        waves=(
            AssemblyWave(
                wave_index=0,
                depth=1,
                phase="ROOT_LEFT_RIGHT",
                actions=(_action("m1", "m0", requires_helper=True),),
            ),
            AssemblyWave(
                wave_index=1,
                depth=2,
                phase="DEPTH_PARALLEL",
                actions=(_action("m2", "m1"),),
            ),
        ),
    )
    executor = ParallelAssemblyExecutor(
        plan,
        execution_id="helper",
        enable_borrowed_helper=True,
    )

    decision = executor.step()
    assert executor.helper_module_id == "m2"
    expected_primitives = (
        "align_faces",
        "dock",
        "set_tilt",
        "set_tilt",
        "assisted_align_faces",
        "dock",
        "undock",
        "set_tilt",
    )
    for primitive in expected_primitives:
        assert decision.primitive_goal is not None
        assert decision.primitive_goal.primitive == primitive
        decision = executor.step(
            _status(
                decision.primitive_goal.goal_id,
                primitive,
                "succeeded",
            )
        )

    assert decision.primitive_goal is None
    assert decision.completed_action_count == 1
    assert decision.wave_index == 1

    resumed = executor.step()
    assert resumed.primitive_goal is not None
    assert resumed.primitive_goal.primitive == "align_faces"


def test_helper_action_fails_immediately_when_helper_is_disabled() -> None:
    executor = ParallelAssemblyExecutor(
        _plan((_action(requires_helper=True),)),
    )

    decision = executor.step()
    assert decision.primitive_goal is None
    assert decision.done
    assert not decision.success
    assert decision.state == "FAILED"
    assert "procedure" in decision.message
    assert "disabled" in decision.message


def test_helper_action_fails_when_no_future_leaf_can_be_borrowed() -> None:
    executor = ParallelAssemblyExecutor(
        _plan((_action(requires_helper=True),)),
        enable_borrowed_helper=True,
    )

    decision = executor.step()
    assert decision.done
    assert not decision.success
    assert "neither a dedicated reserve" in decision.message


def test_helper_action_prefers_a_dedicated_reserve() -> None:
    plan = ParallelAssemblyPlan(
        root_target_vertex="target_m0",
        root_module_id="m0",
        waves=(
            AssemblyWave(
                wave_index=0,
                depth=1,
                phase="ROOT_LEFT_RIGHT",
                actions=(_action("m1", "m0", requires_helper=True),),
            ),
            AssemblyWave(
                wave_index=1,
                depth=2,
                phase="DEPTH_PARALLEL",
                actions=(_action("m2", "m1"),),
            ),
        ),
    )
    executor = ParallelAssemblyExecutor(
        plan,
        enable_borrowed_helper=True,
        additional_known_module_ids=("reserve_08",),
    )

    assert executor.helper_module_id == "reserve_08"
    first = executor.step()
    assert first.primitive_goal is not None
    assert first.primitive_goal.module_ids == ("reserve_08", "m1")


def test_planar_layout_precedes_docking_and_final_posture() -> None:
    executor = ParallelAssemblyExecutor(
        _plan((_action(),)),
        execution_id="three-phase",
        layout_pose_by_module={
            "m0": PlanarPose(0.0, 0.0, 0.0),
            "m1": PlanarPose(0.1, 0.0, 0.0),
        },
        post_assembly_tilt_by_module={"m1": 0.5},
        coordinate_posture_tilts=True,
    )

    first_layout = executor.step()
    assert first_layout.phase == "LAYOUT"
    assert first_layout.primitive_goal is not None
    assert first_layout.primitive_goal.primitive == "drive_to_pose"

    second_layout = executor.step(
        _status(
            first_layout.primitive_goal.goal_id,
            "drive_to_pose",
            "accepted",
        )
    )
    assert second_layout.primitive_goal is not None
    assert second_layout.primitive_goal.primitive == "drive_to_pose"

    waiting = executor.step(
        _status(
            second_layout.primitive_goal.goal_id,
            "drive_to_pose",
            "accepted",
        )
    )
    assert waiting.state == "WAITING_LAYOUT_RESULTS"

    layout_done = executor.step(
        {
            "statuses": [
                _status(
                    first_layout.primitive_goal.goal_id,
                    "drive_to_pose",
                    "succeeded",
                ),
                _status(
                    second_layout.primitive_goal.goal_id,
                    "drive_to_pose",
                    "succeeded",
                ),
            ]
        }
    )
    assert layout_done.state == "PLANAR_LAYOUT_REACHED"

    reach = executor.step()
    assert reach.phase == "REACH"
    assert reach.primitive_goal is not None
    align = executor.step(
        _status(reach.primitive_goal.goal_id, "align_faces", "succeeded")
    )
    assert align.phase == "ALIGN"
    approach = executor.step(
        _status(align.primitive_goal.goal_id, "align_faces", "succeeded")
    )
    assert approach.phase == "APPROACH"
    dock = executor.step(
        _status(
            approach.primitive_goal.goal_id,
            "align_faces",
            "succeeded",
        )
    )
    assert dock.phase == "DOCK"
    assert dock.primitive_goal is not None
    assert dock.primitive_goal.primitive == "dock"
    posture = executor.step(
        _status(dock.primitive_goal.goal_id, "dock", "succeeded")
    )
    assert posture.primitive_goal is not None
    assert posture.primitive_goal.primitive == "set_tilt"
    assert posture.primitive_goal.parameters["coordination_group"] == (
        "three-phase-posture"
    )
    assert posture.primitive_goal.parameters["coordination_size"] == 1
    finished = executor.step(
        _status(posture.primitive_goal.goal_id, "set_tilt", "succeeded")
    )
    assert finished.done
    assert finished.success


def test_future_wave_reaches_only_after_chassis_wave_is_complete() -> None:
    plan = ParallelAssemblyPlan(
        root_target_vertex="target_m0",
        root_module_id="m0",
        waves=(
            AssemblyWave(
                wave_index=0,
                depth=1,
                phase="ROOT_TOP_BOTTOM",
                actions=(
                    _action("m1", "m0"),
                    _action("m2", "m0"),
                ),
            ),
            AssemblyWave(
                wave_index=1,
                depth=2,
                phase="DEPTH_PARALLEL",
                actions=(_action("m3", "m1"),),
            ),
        ),
    )
    executor = ParallelAssemblyExecutor(
        plan,
        execution_id="overlap",
        max_concurrent_alignments_per_wave=0,
    )

    _, first_align = _succeed_parallel_phase(
        executor,
        executor.step(),
        2,
    )
    _, first_approach = _succeed_parallel_phase(
        executor,
        first_align,
        2,
    )
    _, first_dock = _succeed_parallel_phase(
        executor,
        first_approach,
        2,
    )
    assert first_dock.primitive_goal is not None
    second_dock = executor.step(
        _status(first_dock.primitive_goal.goal_id, "dock", "succeeded")
    )
    assert second_dock.primitive_goal is not None
    assert second_dock.primitive_goal.goal_id == "overlap-w0-a1-dock"

    wheel_reach = executor.step(
        _status(second_dock.primitive_goal.goal_id, "dock", "succeeded")
    )
    assert wheel_reach.primitive_goal is not None
    assert wheel_reach.primitive_goal.primitive == "align_faces"
    assert wheel_reach.primitive_goal.goal_id == "overlap-w1-a0-reach"
    assert wheel_reach.phase == "REACH"
    assert wheel_reach.wave_index == 1


def test_empty_plan_succeeds_immediately() -> None:
    plan = ParallelAssemblyPlan(
        root_target_vertex="root",
        root_module_id="m0",
        waves=(),
    )

    decision = ParallelAssemblyExecutor(
        plan
    ).step()

    assert decision.done
    assert decision.success
    assert decision.state == "SUCCEEDED"


def test_empty_topology_delta_can_still_apply_target_posture() -> None:
    plan = ParallelAssemblyPlan(
        root_target_vertex="root",
        root_module_id="m0",
        waves=(),
    )
    executor = ParallelAssemblyExecutor(
        plan,
        post_assembly_tilt_by_module={"m0": 0.5},
    )

    posture = executor.step()

    assert posture.primitive_goal is not None
    assert posture.primitive_goal.primitive == "set_tilt"
    finished = executor.step(
        _status(
            posture.primitive_goal.goal_id,
            "set_tilt",
            "succeeded",
        )
    )
    assert finished.done
    assert finished.success


def test_empty_topology_delta_can_lock_final_pan_posture() -> None:
    plan = ParallelAssemblyPlan(
        root_target_vertex="root",
        root_module_id="m0",
        waves=(),
    )
    executor = ParallelAssemblyExecutor(
        plan,
        post_assembly_pan_by_module={"m0": 0.0},
    )

    posture = executor.step()

    assert posture.primitive_goal is not None
    assert posture.primitive_goal.primitive == "set_pan"
    assert posture.primitive_goal.parameters["angle_rad"] == 0.0
    finished = executor.step(
        _status(
            posture.primitive_goal.goal_id,
            "set_pan",
            "succeeded",
        )
    )
    assert finished.done
    assert finished.success


def test_posture_goal_preserves_explicit_tilt_tolerance() -> None:
    plan = ParallelAssemblyPlan(
        root_target_vertex="root",
        root_module_id="m0",
        waves=(),
    )
    executor = ParallelAssemblyExecutor(
        plan,
        post_assembly_tilt_by_module={"m0": -0.75},
        posture_tilt_tolerance_rad=0.07,
    )

    posture = executor.step()

    assert posture.primitive_goal is not None
    assert posture.primitive_goal.parameters["tolerance_rad"] == 0.07


def test_posture_groups_execute_symmetric_pairs_in_separate_waves() -> None:
    plan = ParallelAssemblyPlan(
        root_target_vertex="root",
        root_module_id="m0",
        waves=(),
    )
    executor = ParallelAssemblyExecutor(
        plan,
        execution_id="paired-fold",
        post_assembly_tilt_by_module={
            "m0": -0.75,
            "m1": -0.75,
            "m2": -0.75,
            "m3": -0.75,
        },
        posture_tilt_max_servo_error_rad=0.35,
        coordinate_posture_tilts=True,
        posture_tilt_groups_by_module=(("m0", "m2"), ("m1", "m3")),
        additional_known_module_ids=("m1", "m2", "m3"),
    )

    first = executor.step()
    assert first.primitive_goal is not None
    assert first.primitive_goal.module_ids == ("m0",)
    assert first.primitive_goal.parameters["coordination_group"].endswith(
        "group-0"
    )
    assert first.primitive_goal.parameters[
        "max_servo_error_rad"
    ] == pytest.approx(0.35)
    second = executor.step(
        _status(first.primitive_goal.goal_id, "set_tilt", "accepted")
    )
    assert second.primitive_goal is not None
    assert second.primitive_goal.module_ids == ("m2",)

    next_group = executor.step(
        {
            "schema_version": "mssr.primitive_status_batch.v1",
            "statuses": [
                _status(
                    first.primitive_goal.goal_id,
                    "set_tilt",
                    "succeeded",
                ),
                _status(
                    second.primitive_goal.goal_id,
                    "set_tilt",
                    "succeeded",
                ),
            ],
        }
    )
    assert next_group.primitive_goal is not None
    assert next_group.primitive_goal.module_ids == ("m1",)
    assert next_group.primitive_goal.parameters["coordination_group"].endswith(
        "group-1"
    )


def test_outside_in_posture_keeps_inner_tilt_structurally_held() -> None:
    plan = ParallelAssemblyPlan(
        root_target_vertex="root",
        root_module_id="inner",
        waves=(),
    )
    executor = ParallelAssemblyExecutor(
        plan,
        post_assembly_tilt_by_module={"outer": 0.6, "inner": 0.8},
        posture_tilt_groups_by_module=(("outer",), ("inner",)),
        additional_known_module_ids=("outer",),
    )

    outer = executor.step().primitive_goal
    assert outer is not None
    assert outer.module_ids == ("outer",)
    assert "passive_tilt_module_ids" not in outer.parameters
    assert "inner" in outer.parameters["hold_after_group_module_ids"]

    inner = executor.step(
        _status(outer.goal_id, "set_tilt", "succeeded")
    ).primitive_goal
    assert inner is not None
    assert inner.module_ids == ("inner",)
    assert "passive_tilt_module_ids" not in inner.parameters


def test_holonomic_posture_dispatches_coupled_pusher_and_lifter() -> None:
    plan = ParallelAssemblyPlan(
        root_target_vertex="root",
        root_module_id="center",
        waves=(),
    )
    executor = ParallelAssemblyExecutor(
        plan,
        execution_id="holonomic-fold",
        post_assembly_tilt_by_module={"outer": 1.35},
        posture_tilt_groups_by_module=(("outer",),),
        posture_push_by_lifter_module={"outer": ("inner", -0.025)},
        additional_known_module_ids=("inner", "outer"),
    )

    goal = executor.step().primitive_goal

    assert goal is not None
    assert goal.module_ids == ("outer",)
    assert goal.parameters["pusher_module_id"] == "inner"
    assert goal.parameters["pusher_linear_m_s"] == pytest.approx(-0.025)
    assert set(goal.parameters["hold_after_group_module_ids"]) == {
        "center", "inner", "outer"
    }
    assert goal.parameters["stabilize_during_group_module_ids"] == [
        "center"
    ]
    assert "passive_tilt_module_ids" not in goal.parameters


def test_holonomic_fold_pair_follows_observed_02_to_09_assignment() -> None:
    pairs = physical_fold_push_pairs(
        (
            {
                "pusher_vertex": "v6",
                "lifter_vertex": "v2",
                "linear_m_s": 0.025,
            },
        ),
        {"v2": "smores_09", "v6": "smores_02"},
        {"smores_09"},
    )

    assert pairs == {"smores_09": ("smores_02", 0.025)}


def test_target_vertex_posture_groups_follow_physical_assignment() -> None:
    groups = physical_posture_groups(
        (("v3", "v5"), ("v4", "v6")),
        {"v3": "rear", "v4": "left", "v5": "front", "v6": "right"},
        {"rear", "left", "front", "right"},
    )

    assert groups == (("rear", "front"), ("left", "right"))
