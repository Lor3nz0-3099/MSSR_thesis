from __future__ import annotations

import importlib
import importlib.util
import json
import math
from pathlib import Path

import pytest


def _api():
    spec = importlib.util.find_spec(
        "drl.MARL.isaac_control_step"
    )

    assert (
        spec is not None
    ), "TASK4A_RED_CONTROL_STEP_MISSING"

    module = importlib.import_module(
        "drl.MARL.isaac_control_step"
    )

    return (
        module.FileBackedIsaacControlStep,
        module.IsaacControlStepResult,
    )


def _write_state_graph(
    runtime_dir: Path,
    *,
    stamp: float,
    xs: tuple[float, ...],
) -> None:
    runtime_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    modules = []

    for index, x_value in enumerate(
        xs,
        start=1,
    ):
        module_id = f"smores_{index:02d}"

        modules.append(
            {
                "module_id": module_id,
                "position": [
                    float(x_value),
                    float(index) * 0.01,
                    0.04,
                ],
            }
        )

    state = {
        "schema_version": "mssr.module_states.v2",
        "stamp": float(stamp),
        "timestamp": float(stamp),
        "modules": modules,
    }

    graph = {
        "schema_version": "mssr.robot_graph.v2",
        "stamp": float(stamp),
        "nodes": [],
        "edges": [],
        "global_attributes": {
            "module_count": len(modules),
        },
    }

    payload = {
        "schema_version": "mssr.state_graph.v2",
        "stamp": float(stamp),
        "state": state,
        "graph": graph,
    }

    path = runtime_dir / "state_graph.json"
    temporary = path.with_suffix(".json.tmp")

    temporary.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    temporary.replace(path)


def test_task4a_state_reader_extracts_exact_module_xy_and_arms_stamp(
    tmp_path,
):
    (
        FileBackedIsaacControlStep,
        _,
    ) = _api()

    runtime = tmp_path / "env0"

    _write_state_graph(
        runtime,
        stamp=1.25,
        xs=tuple(float(index) for index in range(8)),
    )

    control = FileBackedIsaacControlStep(
        runtime_dirs=(runtime,),
        expected_module_count=8,
        timeout_s=1.0,
        poll_interval_s=0.01,
    )

    state = control.state_reader(0)

    assert len(state) == 8

    assert state["smores_01"] == pytest.approx(
        (0.0, 0.01)
    )

    assert state["smores_08"] == pytest.approx(
        (7.0, 0.08)
    )

    assert control.armed_env_indices == (0,)
    assert control.baseline_stamp(0) == pytest.approx(1.25)


def test_task4a_barrier_rejects_stale_frame_and_caches_exact_fresh_frame(
    tmp_path,
):
    (
        FileBackedIsaacControlStep,
        IsaacControlStepResult,
    ) = _api()

    runtime = tmp_path / "env0"

    _write_state_graph(
        runtime,
        stamp=10.0,
        xs=(0.0,) * 8,
    )

    clock = [0.0]
    sleep_calls = [0]

    def monotonic():
        return clock[0]

    def sleep(duration):
        sleep_calls[0] += 1
        clock[0] += duration

        if sleep_calls[0] == 1:
            _write_state_graph(
                runtime,
                stamp=10.0,
                xs=(99.0,) * 8,
            )

        elif sleep_calls[0] == 2:
            _write_state_graph(
                runtime,
                stamp=10.1,
                xs=(1.0,) * 8,
            )

    control = FileBackedIsaacControlStep(
        runtime_dirs=(runtime,),
        expected_module_count=8,
        timeout_s=1.0,
        poll_interval_s=0.01,
        monotonic=monotonic,
        sleep=sleep,
    )

    previous = control.state_reader(0)

    assert previous["smores_01"][0] == pytest.approx(0.0)

    result = control.physics_step()

    assert isinstance(
        result,
        IsaacControlStepResult,
    )

    assert result.previous_stamp_by_env == {
        0: pytest.approx(10.0),
    }

    assert result.next_stamp_by_env == {
        0: pytest.approx(10.1),
    }

    assert sleep_calls[0] == 2

    # Simulate Isaac publishing yet another frame before DirectMARLEnv
    # performs its post-step state_reader().
    _write_state_graph(
        runtime,
        stamp=10.2,
        xs=(2.0,) * 8,
    )

    # The exact frame which satisfied the barrier must be returned,
    # not the newer 10.2 frame.
    next_state = control.state_reader(0)

    assert next_state["smores_01"][0] == pytest.approx(1.0)
    assert control.armed_env_indices == ()

    # Markov chaining is exact: the accepted 10.1 post-state becomes
    # the next cycle's pre-state even though Isaac has already published
    # a newer 10.2 frame.
    next_cycle = control.state_reader(0)

    assert next_cycle["smores_01"][0] == pytest.approx(1.0)
    assert control.baseline_stamp(0) == pytest.approx(10.1)

    control.reset()

    # Reset starts a new episode/cycle and may observe the newest frame.
    fresh = control.state_reader(0)

    assert fresh["smores_01"][0] == pytest.approx(2.0)
    assert control.baseline_stamp(0) == pytest.approx(10.2)

    control.reset()


def test_task4a_vector_barrier_waits_for_every_armed_environment(
    tmp_path,
):
    (
        FileBackedIsaacControlStep,
        _,
    ) = _api()

    runtime0 = tmp_path / "env0"
    runtime1 = tmp_path / "env1"

    _write_state_graph(
        runtime0,
        stamp=2.0,
        xs=(0.0,) * 8,
    )

    _write_state_graph(
        runtime1,
        stamp=5.0,
        xs=(10.0,) * 8,
    )

    clock = [0.0]
    sleeps = [0]

    def monotonic():
        return clock[0]

    def sleep(duration):
        sleeps[0] += 1
        clock[0] += duration

        if sleeps[0] == 1:
            _write_state_graph(
                runtime0,
                stamp=2.1,
                xs=(1.0,) * 8,
            )

        elif sleeps[0] == 2:
            _write_state_graph(
                runtime1,
                stamp=5.1,
                xs=(11.0,) * 8,
            )

    control = FileBackedIsaacControlStep(
        runtime_dirs=(
            runtime0,
            runtime1,
        ),
        expected_module_count=8,
        timeout_s=1.0,
        poll_interval_s=0.01,
        monotonic=monotonic,
        sleep=sleep,
    )

    control.state_reader(0)
    control.state_reader(1)

    assert control.armed_env_indices == (0, 1)

    result = control.physics_step()

    assert result.previous_stamp_by_env == {
        0: pytest.approx(2.0),
        1: pytest.approx(5.0),
    }

    assert result.next_stamp_by_env == {
        0: pytest.approx(2.1),
        1: pytest.approx(5.1),
    }

    state0 = control.state_reader(0)
    state1 = control.state_reader(1)

    assert state0["smores_01"][0] == pytest.approx(1.0)
    assert state1["smores_01"][0] == pytest.approx(11.0)


def test_task4a_stale_publication_times_out_fail_closed(
    tmp_path,
):
    (
        FileBackedIsaacControlStep,
        _,
    ) = _api()

    runtime = tmp_path / "env0"

    _write_state_graph(
        runtime,
        stamp=7.0,
        xs=(0.0,) * 8,
    )

    clock = [0.0]

    def monotonic():
        return clock[0]

    def sleep(duration):
        clock[0] += duration

    control = FileBackedIsaacControlStep(
        runtime_dirs=(runtime,),
        expected_module_count=8,
        timeout_s=0.03,
        poll_interval_s=0.01,
        monotonic=monotonic,
        sleep=sleep,
    )

    control.state_reader(0)

    with pytest.raises(
        TimeoutError,
        match=r"env 0.*7\.0",
    ):
        control.physics_step()

    # Timeout must never fabricate a post-state.
    assert control.armed_env_indices == (0,)

    control.reset()

    assert control.armed_env_indices == ()


def test_task4a_rejects_invalid_or_incomplete_physical_state(
    tmp_path,
):
    (
        FileBackedIsaacControlStep,
        _,
    ) = _api()

    runtime = tmp_path / "env0"

    _write_state_graph(
        runtime,
        stamp=3.0,
        xs=(0.0,) * 7,
    )

    control = FileBackedIsaacControlStep(
        runtime_dirs=(runtime,),
        expected_module_count=8,
    )

    with pytest.raises(
        ValueError,
        match="expected 8 modules",
    ):
        control.state_reader(0)

    _write_state_graph(
        runtime,
        stamp=math.nan,
        xs=(0.0,) * 8,
    )

    with pytest.raises(
        ValueError,
        match="finite stamp",
    ):
        control.state_reader(0)
