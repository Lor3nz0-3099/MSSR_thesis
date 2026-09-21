"""T7 native acceptance checker contracts for MobileManipulator8."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from mssr_expert.graph.serialization import load_attributed_graph
from test_mobile_manipulator_teleop import graph


ROOT = Path(__file__).resolve().parents[4]
CONFIG = ROOT / "mssr_ws/src/mssr_expert/config"
CHECKER = ROOT / "scripts/teleop/check_mobile_manipulator.py"


def checker(monkeypatch):
    assert CHECKER.is_file(), (
        "T7 MM8 native acceptance checker is not implemented"
    )

    monkeypatch.syspath_prepend(
        str(ROOT / "scripts/teleop")
    )

    spec = importlib.util.spec_from_file_location(
        "mssr_mm8_runtime_check",
        CHECKER,
    )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def test_mm8_checker_script_exists():
    assert CHECKER.is_file(), (
        "T7 MM8 native acceptance checker is not implemented"
    )


def test_checker_uses_live_mm8_role_assignment(monkeypatch):
    smoke = checker(monkeypatch)

    target = load_attributed_graph(
        CONFIG / "smores_mobile_manipulator8.json"
    )

    metrics = smoke.physical_metrics(
        graph().to_dict(),
        target,
    )

    assert metrics["roles"]["end_effector"] == "physical_v7"
    assert metrics["roles"]["front_support"] == "physical_v2"
    assert metrics["roles"]["arm_lift"] == "physical_v5"

    with pytest.raises(
        ValueError,
        match="valid MobileManipulator8",
    ):
        smoke.physical_metrics(
            graph(missing=True).to_dict(),
            target,
        )


def test_translation_pair_invariant_requires_exact_front_support_and_arm_lift(
    monkeypatch,
):
    smoke = checker(monkeypatch)

    physical = {
        "roles": {
            "front_support": "m_front",
            "arm_lift": "m_lift",
        }
    }

    assert smoke.translation_pair_invariant(
        {
            "m_front": {
                "vx": 0.04,
                "vy": 0.0,
                "yaw_rate": 0.0,
            },
            "m_lift": {
                "vx": 0.04,
                "vy": 0.0,
                "yaw_rate": 0.0,
            },
        },
        physical,
    )

    assert not smoke.translation_pair_invariant(
        {
            "m_front": {
                "vx": 0.04,
                "vy": 0.0,
                "yaw_rate": 0.0,
            },
        },
        physical,
    )

    assert not smoke.translation_pair_invariant(
        {
            "m_front": {
                "vx": 0.04,
                "vy": 0.0,
                "yaw_rate": 0.0,
            },
            "m_lift": {
                "vx": 0.04,
                "vy": 0.0,
                "yaw_rate": 0.0,
            },
            "wrong": {
                "vx": 0.04,
                "vy": 0.0,
                "yaw_rate": 0.0,
            },
        },
        physical,
    )


def test_recording_parser_requires_mm8_manual_effective_actions(
    tmp_path,
    monkeypatch,
):
    smoke = checker(monkeypatch)

    human = tmp_path / "human_behavior.jsonl"

    rows = (
        {
            "task_type": "snake8_teleop",
            "observation": {
                "intent": {
                    "control_mode": "manual",
                    "selected_module_id": "other",
                }
            },
            "expert_action": {
                "locomotion": {
                    "other": {
                        "vx": 1.0,
                    }
                }
            },
        },
        {
            "task_type": "mobile_manipulator8_teleop",
            "observation": {
                "intent": {
                    "control_mode": "manual",
                    "selected_module_id": "physical_v7",
                }
            },
            "expert_action": {
                "locomotion": {
                    "physical_v2": {
                        "vx": 0.03,
                    },
                    "physical_v5": {
                        "vx": 0.03,
                    },
                    "physical_v7": {
                        "pan_target_rad": 0.2,
                    },
                }
            },
        },
        {
            "task_type": "mobile_manipulator8_teleop",
            "observation": {
                "intent": {
                    "control_mode": "manual",
                    "selected_module_id": "physical_v6",
                }
            },
            "expert_action": {
                "locomotion": {
                    "physical_v2": {
                        "vx": -0.03,
                    },
                    "physical_v5": {
                        "vx": -0.03,
                    },
                    "physical_v6": {
                        "tilt_target_rad": 0.1,
                    },
                }
            },
        },
    )

    human.write_text(
        "\n".join(
            json.dumps(row)
            for row in rows
        )
        + "\n",
        encoding="utf-8",
    )

    assert smoke.recorded_mm8_actions(human) == {
        "rows": 2,
        "wheel_rows": 2,
        "shape_rows": 2,
        "manual_pan_rows": 1,
        "manual_tilt_rows": 1,
        "manual_selected_modules": [
            "physical_v6",
            "physical_v7",
        ],
    }


def test_runtime_commands_assemble_mm8_directly(
    tmp_path,
    monkeypatch,
):
    smoke = checker(monkeypatch)

    commands = smoke.mm8_runtime_commands(
        tmp_path,
        CONFIG / "smores_dualsense.yaml",
        "mm8-probe",
        0,
        runtime_dir=tmp_path / "runtime",
    )

    assert (
        f"target_graph_path:="
        f"{CONFIG / 'smores_mobile_manipulator8.json'}"
        in commands["assembly"]
    )

    assert (
        "execution_id:=t7-mm8-probe"
        in commands["assembly"]
    )

    assert not any(
        "smores_snake8.json" in arg
        or "smores_rc_car8.json" in arg
        for arg in commands["assembly"]
    )

    assert any(
        "mssr_mm8_teleop_mm8-probe" in arg
        for arg in commands["teleop"]
    )


def test_direct_mm8_preflight_does_not_require_reconfiguration_button(
    tmp_path,
):
    mapping = yaml.safe_load(
        (CONFIG / "smores_dualsense.yaml").read_text()
    )

    mapping["commands"]["select_mm8"] = None

    path = tmp_path / "deferred.yaml"

    path.write_text(
        yaml.safe_dump(mapping),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(CHECKER),
            "--input-config",
            str(path),
            "--device-id",
            "-1",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=5,
    )

    # Once the checker exists, preflight must progress far enough to reject
    # the deliberately invalid device id, not the unused D-pad MM8 selector.
    assert result.returncode == 2
    assert "device ID" in result.stderr


def test_valid_cli_delegates_to_native_mm8_acceptance(
    monkeypatch,
):
    smoke = checker(monkeypatch)

    called = {}

    def fake_native(args, mapping):
        called["args"] = args
        called["mapping"] = mapping
        return 0

    monkeypatch.setattr(
        smoke,
        "run_native_acceptance",
        fake_native,
        raising=False,
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(CHECKER),
            "--input-config",
            str(CONFIG / "smores_dualsense.yaml"),
            "--assembly-timeout",
            "1.0",
            "--device-id",
            "0",
        ],
    )

    result = smoke.main()

    assert result == 0
    assert called["args"].device_id == 0
    assert called["args"].assembly_timeout == pytest.approx(1.0)
    assert called["mapping"].commands.get("record_toggle") is not None


def test_translation_pair_invariant_allows_shape_only_holds_but_rejects_extra_locomotion(
    monkeypatch,
):
    smoke = checker(monkeypatch)

    physical = {
        "roles": {
            "front_support": "m_front",
            "arm_lift": "m_lift",
        }
    }

    with_hold = {
        "m_front": {
            "vx": 0.04,
            "vy": 0.0,
            "yaw_rate": 0.0,
        },
        "m_lift": {
            "vx": 0.04,
            "vy": 0.0,
            "yaw_rate": 0.0,
        },
        "m_arm": {
            "tilt_target_rad": 0.3,
        },
    }

    assert smoke.translation_pair_invariant(
        with_hold,
        physical,
    )

    unauthorized_motion = {
        **with_hold,
        "m_arm": {
            "tilt_target_rad": 0.3,
            "vx": 0.01,
            "vy": 0.0,
            "yaw_rate": 0.0,
        },
    }

    assert not smoke.translation_pair_invariant(
        unauthorized_motion,
        physical,
    )


def test_mm8_mode_matcher_tracks_runtime_fsm(
    monkeypatch,
):
    smoke = checker(monkeypatch)

    drive = {
        "active_controller": "mobile_manipulator8",
        "mm8_intent": {
            "mode": "drive_ready",
            "mode_transition_pending": False,
            "mode_transition_behavior": None,
            "selected_role": "end_effector",
        },
    }

    assert smoke.mm8_mode_matches(
        drive,
        "drive_ready",
        pending=False,
    )

    to_manip = {
        "active_controller": "mobile_manipulator8",
        "mm8_intent": {
            "mode": "to_manipulation_ready",
            "mode_transition_pending": True,
            "mode_transition_behavior": "prepare_manipulation",
            "selected_role": "end_effector",
        },
    }

    assert smoke.mm8_mode_matches(
        to_manip,
        "to_manipulation_ready",
        pending=True,
        behavior="prepare_manipulation",
    )

    manip = {
        "active_controller": "mobile_manipulator8",
        "mm8_intent": {
            "mode": "manipulation_ready",
            "mode_transition_pending": False,
            "mode_transition_behavior": None,
            "selected_role": "arm_link",
        },
    }

    assert smoke.mm8_mode_matches(
        manip,
        "manipulation_ready",
        pending=False,
        selected_role="arm_link",
    )

    wrong_controller = {
        **drive,
        "active_controller": "snake8",
    }

    assert not smoke.mm8_mode_matches(
        wrong_controller,
        "drive_ready",
        pending=False,
    )

    assert not smoke.mm8_mode_matches(
        to_manip,
        "manipulation_ready",
        pending=False,
    )


def test_native_acceptance_contains_mm8_runtime_lifecycle(
    monkeypatch,
):
    import inspect

    smoke = checker(monkeypatch)

    source = inspect.getsource(
        smoke.run_native_acceptance
    )

    required = (
        "scoped_cleanup",
        "mm8_runtime_commands",
        '"isaac"',
        '"bridge"',
        '"assembly"',
        '"joy"',
        '"teleop"',
        "mm8_mode_matches",
        '"mm8_intent"',
        '"mm8_effective_actions"',
        '"drive_ready"',
        '"to_manipulation_ready"',
        '"prepare_manipulation"',
        '"manipulation_ready"',
        '"to_drive_ready"',
        '"restore_drive"',
        "physical_wheel_rates",
        "translation_pair_invariant",
        "recorded_mm8_actions",
        "finalize_runtime",
    )

    missing = [
        token
        for token in required
        if token not in source
    ]

    assert not missing, (
        "native MM8 acceptance lifecycle is incomplete: "
        + ", ".join(missing)
    )


def test_isaac_environment_isolated_from_ros_and_keeps_gui_session(
    monkeypatch,
):
    smoke = checker(monkeypatch)

    inherited = {
        "HOME": "/home/test",
        "PATH": "/usr/bin",
        "DISPLAY": ":0",
        "WAYLAND_DISPLAY": "wayland-0",
        "XDG_RUNTIME_DIR": "/run/user/1000",
        "XDG_SESSION_TYPE": "wayland",
        "PYTHONPATH": "/opt/ros/humble/python",
        "LD_LIBRARY_PATH": "/opt/ros/humble/lib:/usr/lib",
        "AMENT_PREFIX_PATH": "/opt/ros/humble",
        "CMAKE_PREFIX_PATH": "/opt/ros/humble",
        "COLCON_PREFIX_PATH": "/tmp/ws/install",
        "ROS_DISTRO": "humble",
        "ROS_VERSION": "2",
        "ROS_PYTHON_VERSION": "3",
        "RMW_IMPLEMENTATION": "rmw_cyclonedds_cpp",
        "ROS_DOMAIN_ID": "42",
        "GAZEBO_MODEL_PATH": "/tmp/models",
        "GAZEBO_PLUGIN_PATH": "/tmp/plugins",
    }

    clean = smoke.isaac_subprocess_environment(
        inherited
    )

    # Ordinary process/session environment survives.
    assert clean["HOME"] == "/home/test"
    assert clean["PATH"] == "/usr/bin"

    # GUI/session variables must survive for native Isaac GUI.
    assert clean["DISPLAY"] == ":0"
    assert clean["WAYLAND_DISPLAY"] == "wayland-0"
    assert clean["XDG_RUNTIME_DIR"] == "/run/user/1000"
    assert clean["XDG_SESSION_TYPE"] == "wayland"

    # ROS/Gazebo overlays must not contaminate Isaac's Python/runtime loader.
    removed = {
        "PYTHONPATH",
        "LD_LIBRARY_PATH",
        "AMENT_PREFIX_PATH",
        "CMAKE_PREFIX_PATH",
        "COLCON_PREFIX_PATH",
        "ROS_DISTRO",
        "ROS_VERSION",
        "ROS_PYTHON_VERSION",
        "RMW_IMPLEMENTATION",
        "ROS_DOMAIN_ID",
        "GAZEBO_MODEL_PATH",
        "GAZEBO_PLUGIN_PATH",
    }

    assert removed.isdisjoint(clean)


def test_runtime_process_environment_cleans_only_isaac(
    monkeypatch,
):
    smoke = checker(monkeypatch)

    inherited = {
        "HOME": "/home/test",
        "DISPLAY": ":0",
        "PYTHONPATH": "/opt/ros/humble/python",
        "LD_LIBRARY_PATH": "/opt/ros/humble/lib",
        "ROS_DOMAIN_ID": "42",
        "RMW_IMPLEMENTATION": "rmw_cyclonedds_cpp",
    }

    isaac = smoke.runtime_process_environment(
        "isaac",
        inherited,
    )

    assert isaac["HOME"] == "/home/test"
    assert isaac["DISPLAY"] == ":0"
    assert "PYTHONPATH" not in isaac
    assert "LD_LIBRARY_PATH" not in isaac
    assert "ROS_DOMAIN_ID" not in isaac
    assert "RMW_IMPLEMENTATION" not in isaac

    for name in (
        "bridge",
        "assembly",
        "joy",
        "teleop",
    ):
        child = smoke.runtime_process_environment(
            name,
            inherited,
        )

        assert child == inherited
        assert child is not inherited


def test_native_acceptance_applies_process_specific_environment(
    monkeypatch,
):
    import inspect

    smoke = checker(monkeypatch)

    source = inspect.getsource(
        smoke.run_native_acceptance
    )

    assert "runtime_process_environment(" in source
    assert "env=" in source


def test_native_mm8_acceptance_includes_physical_estop_resume_fence():
    source = CHECKER.read_text(encoding="utf-8")

    required_checkpoints = (
        '"pre_estop_drive"',
        '"estop"',
        '"stop_physics_camera"',
        '"resume_fence"',
        '"fresh_neutral"',
        '"post_resume_forward"',
    )

    for checkpoint in required_checkpoints:
        assert checkpoint in source

    # Physical/native evidence required by the acceptance sequence.
    assert '["safety"]["authority"] == "ESTOP"' in source
    assert '["runtime_structure_stopped"] is True' in source
    assert '["runtime_structure_stopped"] is False' in source
    assert '["safety"]["motion_enabled"]' in source
    assert '"camera_applied"' in source
