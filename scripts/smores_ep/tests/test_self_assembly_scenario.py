from __future__ import annotations

import math

import pytest

from smores_ep.config.simulation import SelfAssemblySimulationConfig
from smores_ep.control.teleop import SmoresCommand
from smores_ep.isaac.physics_asset import PHYSICS_ROOT
from smores_ep.scenarios.parallel_self_assembly import (
    _publish_primitive_statuses,
    closest_module_to_centroid,
    radial_spawn_layout,
    self_assembly_module_roots,
    self_assembly_spawn_layout,
    sparse_behavior_commands,
    triangular_spawn_layout,
)
from smores_ep.self_assembly_cli import build_argument_parser
from smores_ep.primitives.model import (
    PrimitiveName,
    PrimitiveState,
    PrimitiveStatus,
)


class _CapturingStatusChannel:
    def __init__(self) -> None:
        self.statuses: tuple[PrimitiveStatus, ...] = ()

    def publish_many(
        self,
        statuses: tuple[PrimitiveStatus, ...],
        _stamp_s: float,
    ) -> None:
        self.statuses = tuple(statuses)


def test_self_assembly_config_requires_at_least_two_distinct_modules(
    tmp_path,
) -> None:
    with pytest.raises(ValueError):
        SelfAssemblySimulationConfig(
            physics_usd=tmp_path / "physics.usd",
            module_ids=("same", "same", "third"),
        )

    with pytest.raises(ValueError):
        SelfAssemblySimulationConfig(
            physics_usd=tmp_path / "physics.usd",
            module_ids=("first",),
        )


def test_three_module_registry_reuses_source_and_creates_two_clones() -> None:
    roots = self_assembly_module_roots(
        ("smores_01", "smores_02", "smores_03")
    )

    assert roots["smores_01"] == PHYSICS_ROOT
    assert len(set(roots.values())) == 3
    assert roots["smores_02"].startswith("/World/")
    assert roots["smores_03"].startswith("/World/")


def test_seven_module_registry_and_radial_layout_scale_from_one_source(
    tmp_path,
) -> None:
    module_ids = tuple(f"smores_{index:02d}" for index in range(1, 8))
    config = SelfAssemblySimulationConfig(
        physics_usd=tmp_path / "physics.usd",
        module_ids=module_ids,
    )
    roots = self_assembly_module_roots(module_ids)
    layout = radial_spawn_layout(config)

    assert roots[module_ids[0]] == PHYSICS_ROOT
    assert len(set(roots.values())) == 7
    assert set(layout) == set(module_ids)
    assert closest_module_to_centroid(layout) == module_ids[0]
    assert self_assembly_spawn_layout(config) == layout

    for module_id in module_ids[1:]:
        x_m, y_m, _, _ = layout[module_id]
        assert math.hypot(x_m, y_m) == pytest.approx(config.spawn_radius_m)


def test_sparse_behavior_commands_do_not_brake_omitted_modules() -> None:
    command = SmoresCommand(linear_x_m_s=0.04)

    baseline = sparse_behavior_commands(
        {"locomotor": command},
        ("locomotor", "passive_payload"),
    )

    assert baseline == {"locomotor": command}
    assert "passive_payload" not in baseline


def test_sparse_behavior_commands_reject_unknown_modules() -> None:
    with pytest.raises(ValueError, match="unknown modules"):
        sparse_behavior_commands(
            {"ghost": SmoresCommand()},
            ("known",),
        )


def test_triangular_layout_is_separated_and_center_module_is_root_candidate(
    tmp_path,
) -> None:
    config = SelfAssemblySimulationConfig(
        physics_usd=tmp_path / "physics.usd",
    )
    layout = triangular_spawn_layout(config)

    assert closest_module_to_centroid(layout) == "smores_02"
    assert set(layout) == set(config.module_ids)

    for first_index, first_id in enumerate(config.module_ids):
        for second_id in config.module_ids[first_index + 1 :]:
            first = layout[first_id]
            second = layout[second_id]
            distance = math.hypot(
                first[0] - second[0],
                first[1] - second[1],
            )
            assert distance > 0.15


def test_self_assembly_cli_defaults_match_target_demonstration() -> None:
    args = build_argument_parser().parse_args([])

    assert (
        args.left_module_id,
        args.center_module_id,
        args.right_module_id,
    ) == ("smores_01", "smores_02", "smores_03")
    assert args.steps == 0
    assert args.headless is False
    assert args.performance is False
    assert args.simple_visuals is False
    assert args.module_count == 3
    assert args.module_prefix == "smores_"
    assert args.obstacle_course is False
    assert args.stair_test_course is False
    assert args.button_test_course is False
    assert args.gap_test_course is False
    assert args.stair_seed is None
    assert args.stair_rise_m is None
    assert args.stair_depth_m is None
    assert args.stair_count is None
    assert args.simulation_speed_factor == pytest.approx(1.0)


def test_course_cli_options_are_mutually_exclusive() -> None:
    parser = build_argument_parser()

    args = parser.parse_args(["--stair-test-course"])
    assert args.stair_test_course is True
    assert args.obstacle_course is False
    assert args.button_test_course is False
    assert args.gap_test_course is False
    with pytest.raises(SystemExit):
        parser.parse_args(["--obstacle-course", "--stair-test-course"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--button-test-course", "--gap-test-course"])

    button = parser.parse_args(["--button-test-course"])
    gap = parser.parse_args(["--gap-test-course"])
    assert button.button_test_course is True
    assert gap.gap_test_course is True


def test_stair_cli_accepts_seed_and_explicit_geometry_overrides() -> None:
    args = build_argument_parser().parse_args(
        [
            "--stair-test-course",
            "--stair-seed",
            "17",
            "--stair-rise-m",
            "0.055",
            "--stair-depth-m",
            "0.310",
            "--stair-count",
            "4",
            "--simulation-speed-factor",
            "2.0",
        ]
    )

    assert args.stair_seed == 17
    assert args.stair_rise_m == pytest.approx(0.055)
    assert args.stair_depth_m == pytest.approx(0.310)
    assert args.stair_count == 4
    assert args.simulation_speed_factor == pytest.approx(2.0)


def test_self_assembly_config_rejects_two_courses(tmp_path) -> None:
    with pytest.raises(ValueError, match="exclusive"):
        SelfAssemblySimulationConfig(
            physics_usd=tmp_path / "physics.usd",
            manual_obstacle_course=True,
            stair_test_course=True,
        )
    with pytest.raises(ValueError, match="exclusive"):
        SelfAssemblySimulationConfig(
            physics_usd=tmp_path / "physics.usd",
            button_test_course=True,
            gap_test_course=True,
        )


def test_self_assembly_config_validates_all_runtime_frequencies(
    tmp_path,
) -> None:
    with pytest.raises(ValueError):
        SelfAssemblySimulationConfig(
            physics_usd=tmp_path / "physics.usd",
            physics_hz=60,
            render_hz=30,
            state_publish_hz=7,
        )


def test_performance_switch_is_available() -> None:
    args = build_argument_parser().parse_args(["--performance"])

    assert args.performance is True
    assert args.physics_hz is None
    assert args.render_hz is None
    assert args.state_publish_hz is None
    assert args.simple_visuals is False


def test_simple_visuals_are_an_explicit_opt_in() -> None:
    args = build_argument_parser().parse_args(
        ["--performance", "--simple-visuals"]
    )

    assert args.performance is True
    assert args.simple_visuals is True


def test_terminal_primitive_status_remains_in_parallel_file_snapshot() -> None:
    channel = _CapturingStatusChannel()
    terminal_cache: dict[str, PrimitiveStatus] = {}
    running = PrimitiveStatus(
        goal_id="still-running",
        primitive=PrimitiveName.ALIGN_FACES,
        state=PrimitiveState.RUNNING,
        stamp_s=1.0,
        module_ids=("a", "root"),
    )
    succeeded = PrimitiveStatus(
        goal_id="already-finished",
        primitive=PrimitiveName.ALIGN_FACES,
        state=PrimitiveState.SUCCEEDED,
        stamp_s=1.0,
        module_ids=("b", "root"),
    )

    serialized = _publish_primitive_statuses(
        channel,  # type: ignore[arg-type]
        (running, succeeded),
        terminal_cache,
        now_s=1.0,
        physics_step=1,
        state_publish_interval=6,
        previous_serialized="",
    )
    _publish_primitive_statuses(
        channel,  # type: ignore[arg-type]
        (running,),
        terminal_cache,
        now_s=1.1,
        physics_step=6,
        state_publish_interval=6,
        previous_serialized=serialized,
    )

    assert {status.goal_id for status in channel.statuses} == {
        "still-running",
        "already-finished",
    }
    assert terminal_cache["already-finished"].state is PrimitiveState.SUCCEEDED
