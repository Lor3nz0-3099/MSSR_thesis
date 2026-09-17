"""Production-loop integration contract for structure-only E-STOP."""
import inspect
from types import SimpleNamespace

from smores_ep.scenarios import parallel_self_assembly
from smores_ep.primitives.model import (
    PrimitiveGoal,
    PrimitiveName,
    PrimitiveState,
)


def service():
    function = getattr(
        parallel_self_assembly,
        "_service_teleop_runtime",
        None,
    )
    assert function is not None
    return function


def test_runtime_service_polls_without_pausing_or_advancing_app():
    calls = []

    runtime = SimpleNamespace(
        poll=lambda: calls.append("poll")
        or {
            "structure_stopped": True,
        }
    )

    status = service()(runtime)

    assert status["structure_stopped"] is True
    assert calls == ["poll"]


def test_runtime_service_reports_clear_without_special_resume_step():
    calls = []

    runtime = SimpleNamespace(
        poll=lambda: calls.append("poll")
        or {
            "structure_stopped": False,
        }
    )

    status = service()(runtime)

    assert status["structure_stopped"] is False
    assert calls == ["poll"]


def test_production_runtime_wires_structure_stop_through_interrupt_handler():
    source = inspect.getsource(
        parallel_self_assembly.run_parallel_self_assembly_scenario
    )

    assert "structure_stop_callback=handle_structure_stop" in source
    assert (
        "structure_stop_callback=command_router.set_emergency_stop"
        not in source
    )
    assert "_apply_structure_stop(" in source
    assert "omni.timeline.get_timeline_interface()" not in source


def test_production_loop_does_not_skip_physics_for_structure_stop():
    source = inspect.getsource(
        parallel_self_assembly.run_parallel_self_assembly_scenario
    )

    # Runtime servicing remains before robot work, but structure stop is
    # enforced by the command-router actuator boundary. The loop itself
    # must therefore keep reaching _advance_simulation().
    assert "_service_teleop_runtime(runtime)" in source

    assert (
        "if not _service_teleop_runtime(runtime):"
        not in source
    )



def test_structure_stop_interrupts_active_primitives_and_invalidates_old_behavior():
    events = []
    cancel_calls = []

    router = SimpleNamespace(
        set_emergency_stop=lambda active: events.append(
            ("router", active)
        )
    )

    executor = SimpleNamespace(
        active_goals=(
            SimpleNamespace(goal_id="goal-a"),
            SimpleNamespace(goal_id="goal-b"),
        ),
        cancel=lambda goal_id, now_s: (
            cancel_calls.append((goal_id, now_s))
            or f"canceled:{goal_id}"
        ),
    )

    action_channel = SimpleNamespace(
        invalidate_modules=lambda module_ids: events.append(
            ("invalidate", module_ids)
        )
    )

    statuses = parallel_self_assembly._apply_structure_stop(
        True,
        router,
        executor,
        action_channel,
        ("module_a", "module_b"),
        12.5,
    )

    assert events == [
        ("router", True),
        ("invalidate", ("module_a", "module_b")),
    ]

    assert cancel_calls == [
        ("goal-a", 12.5),
        ("goal-b", 12.5),
    ]

    assert statuses == (
        "canceled:goal-a",
        "canceled:goal-b",
    )


def test_structure_stop_clear_only_releases_router_latch():
    events = []
    cancel_calls = []

    router = SimpleNamespace(
        set_emergency_stop=lambda active: events.append(
            ("router", active)
        )
    )

    executor = SimpleNamespace(
        active_goals=(SimpleNamespace(goal_id="goal-a"),),
        cancel=lambda goal_id, now_s: cancel_calls.append(
            (goal_id, now_s)
        ),
    )

    action_channel = SimpleNamespace(
        invalidate_modules=lambda module_ids: events.append(
            ("invalidate", module_ids)
        )
    )

    statuses = parallel_self_assembly._apply_structure_stop(
        False,
        router,
        executor,
        action_channel,
        ("module_a", "module_b"),
        13.0,
    )

    assert events == [("router", False)]
    assert cancel_calls == []
    assert statuses == ()



def _estop_test_goal():
    return PrimitiveGoal(
        goal_id="goal-during-estop",
        primitive=PrimitiveName.SET_TILT,
        module_ids=("module_a",),
        parameters={"angle_rad": 0.25},
    )


def test_new_primitive_is_rejected_while_structure_stop_active():
    submitted = []

    executor = SimpleNamespace(
        submit=lambda goal, now_s: submitted.append(
            (goal.goal_id, now_s)
        )
    )

    status = parallel_self_assembly._admit_primitive_goal(
        executor,
        _estop_test_goal(),
        20.0,
        structure_stopped=True,
    )

    assert submitted == []
    assert status.goal_id == "goal-during-estop"
    assert status.state is PrimitiveState.REJECTED
    assert status.code == "ESTOP_ACTIVE"
    assert status.stamp_s == 20.0


def test_new_primitive_is_submitted_normally_after_structure_stop_clear():
    sentinel = object()
    submitted = []

    def submit(goal, now_s):
        submitted.append((goal.goal_id, now_s))
        return sentinel

    executor = SimpleNamespace(submit=submit)

    result = parallel_self_assembly._admit_primitive_goal(
        executor,
        _estop_test_goal(),
        21.0,
        structure_stopped=False,
    )

    assert result is sentinel
    assert submitted == [
        ("goal-during-estop", 21.0),
    ]


def test_production_runtime_routes_stop_through_interrupt_and_status_publish():
    source = inspect.getsource(
        parallel_self_assembly.run_parallel_self_assembly_scenario
    )

    assert "structure_stop_callback=handle_structure_stop" in source
    assert (
        "structure_stop_callback=command_router.set_emergency_stop"
        not in source
    )

    assert "_apply_structure_stop(" in source
    assert "_publish_primitive_statuses(" in source
    assert "_admit_primitive_goal(" in source



def test_structure_stop_clears_held_primitive_commands():
    held = {
        "module_a": object(),
        "module_b": object(),
    }

    router = SimpleNamespace(
        set_emergency_stop=lambda active: None
    )

    executor = SimpleNamespace(
        active_goals=(),
        cancel=lambda goal_id, now_s: None,
    )

    action_channel = SimpleNamespace(
        invalidate_modules=lambda module_ids: None
    )

    parallel_self_assembly._apply_structure_stop(
        True,
        router,
        executor,
        action_channel,
        ("module_a", "module_b"),
        30.0,
        held_primitive_commands=held,
    )

    assert held == {}
