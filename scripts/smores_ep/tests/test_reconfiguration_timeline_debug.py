from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts/smores_ep/run_reconfiguration_timeline_debug.py"


def test_timeline_debug_script_exists_with_required_contract() -> None:
    assert SCRIPT.is_file(), "diagnostic timeline script has not been implemented yet"
    text = SCRIPT.read_text(encoding="utf-8")

    for required in (
        '"/mssr/primitives/goal"',
        '"/mssr/primitives/status"',
        '"/mssr/primitives/cancel"',
        '"/mssr/actions"',
        '"/mssr/module_states"',
        '"/mssr/state_graph"',
        '"/mssr/expert/self_reconfiguration/state"',
        '"structural_hold_module_ids"',
        '"stabilize_during_group_module_ids"',
        '"hold_after_group_module_ids"',
        '"passive_module_ids"',
        "RC_CURVE",
        "SCORPION_OBSERVE",
        "MM8_RETREAT",
        "MANIPULATION_READY",
        "timeline.txt",
    ):
        assert required in text


def test_curve_is_short_and_asymmetric() -> None:
    assert SCRIPT.is_file()
    text = SCRIPT.read_text(encoding="utf-8")
    assert "(0.25, 0.08" in text
    assert "(0.50, 0.25" in text
    assert "math.radians(15.0)" in text
    assert "math.radians(30.0)" in text

def test_runtime_omits_empty_behavior_dataset_launch_argument() -> None:
    assert SCRIPT.is_file()
    text = SCRIPT.read_text(encoding="utf-8")
    assert '"behavior_dataset_path:="' not in text

def test_curve_is_time_bounded_in_simulation_not_goal_bounded() -> None:
    assert SCRIPT.is_file()
    text = SCRIPT.read_text(encoding="utf-8")

    assert '"--curve-sim-s"' in text
    assert "drive_curve_for_sim_time(" in text
    assert 'curve_sim_s=args.curve_sim_s' in text

    # The diagnostic must not fail merely because Nav2 did not reach
    # the exact route goal.
    assert 'raise RuntimeError(f"curved Nav2 route failed rc={route_rc}")' not in text

