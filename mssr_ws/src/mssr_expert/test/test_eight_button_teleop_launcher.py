from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
LAUNCHER = ROOT / "scripts/teleop/run_eight_button_demos.sh"

EXPECTED = {
    "button-01": 6217,
    "button-02": 6307,
    "button-03": 6533,
    "button-04": 6284,
    "button-05": 6341,
    "button-06": 6443,
    "button-07": 6265,
    "button-08": 6316,
}


def test_eight_button_launcher_preserves_existing_teleop_runtime_contract():
    assert LAUNCHER.is_file(), f"launcher mancante: {LAUNCHER}"

    source = LAUNCHER.read_text()

    for episode, seed in EXPECTED.items():
        assert episode in source
        assert str(seed) in source

    assert "--button-test-course" in source
    assert "--button-seed" in source
    assert "--module-count 8" in source

    assert "scoped_cleanup" in source

    assert "mssr_file_bridge.py" in source
    assert "mssr_smores_morphology_behavior_node" in source
    assert "run_self_assembly.sh" in source
    assert "smores_teleop.launch.py" in source

    assert "start_joy:=true" in source
    assert "smores_dualsense.yaml" in source
    assert "smores_teleop.yaml" in source

    assert "--check" in source

    # Il launcher deve lasciare assembly e reconfiguration all'operatore:
    # niente expert button/IK automatico.
    assert "run_button_expert_to_ik.py" not in source
