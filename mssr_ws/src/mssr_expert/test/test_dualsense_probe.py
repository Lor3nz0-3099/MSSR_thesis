"""Real report aggregation and narrow runtime-process classification."""
import importlib.util
from pathlib import Path
import sys
import subprocess
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[4]


def load_script(name):
    path = ROOT / "scripts/teleop" / f"{name}.py"
    assert path.is_file(), f"missing hardware verification script: {path}"
    spec = importlib.util.spec_from_file_location(f"teleop_test_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def config():
    from mssr_expert.teleop.input import load_input_config
    return load_input_config(ROOT / "mssr_ws/src/mssr_expert/config/smores_dualsense.yaml")


def send(report, axes=None, pressed=False, at=1.0, phase="neutral"):
    buttons = [0] * 21
    buttons[6] = int(pressed)
    return report.accept([0.0] * 6 if axes is None else axes, buttons, at, phase)


def complete_report():
    report = load_script("check_dualsense").ProbeReport(config())
    send(report)
    send(report, at=1.4)
    report.end_phase(1.5, "neutral")
    send(report, axes=[1, 1, 1, 1, -1, -1], at=1.6, phase="travel")
    send(report, axes=[-1, -1, -1, -1, -0.5, -0.5], at=1.7, phase="travel")
    send(report, at=1.8, phase="travel")
    send(report, at=1.9, phase="start")
    send(report, pressed=True, at=2.0, phase="start")
    send(report, at=2.1, phase="start")
    send(report, pressed=True, at=2.2, phase="start")
    report.poll(2.8, "disconnect")
    send(report, pressed=True, at=2.9, phase="reconnect")
    send(report, at=3.0, phase="final_neutral")
    send(report, at=3.4, phase="final_neutral")
    report.end_phase(3.5, "final_neutral")
    return report


def test_report_requires_actual_evidence_for_all_hardware_checks():
    report = complete_report()
    result = report.summary()
    assert result["checks"] == {
        "joy_observed": True, "initial_neutral": True, "stick_travel": True,
        "trigger_travel": True, "trigger_analog": True,
        "start_edges": True, "disconnect": True, "reconnect": True,
        "final_neutral": True, "no_invalid_packets": True,
    }
    assert result["input_checks_passed"]
    assert result["record_toggle_edges"] == 2
    assert result["raw_axes_range"][5] == [-1.0, 0.0]


def test_no_hardware_samples_can_never_pass_verification():
    result = load_script("check_dualsense").ProbeReport(config()).summary()
    assert not result["input_checks_passed"]
    assert not result["checks"]["joy_observed"]
    assert result["raw_axes_range"] == []


def test_missing_vertical_stick_motion_fails_travel_gate():
    report = load_script("check_dualsense").ProbeReport(config())
    send(report)
    send(report, axes=[-1, 0, -1, 0, -1, -1], at=1.1, phase="travel")
    send(report, axes=[1, 0, 1, 0, 0, 0], at=1.2, phase="travel")
    assert not report.summary()["checks"]["stick_travel"]


def test_rejected_packets_fail_report_without_refreshing_timeout():
    report = complete_report()
    assert not report.accept([float("nan")] * 6, [0] * 21, 3.6, "final_neutral")
    report.poll(4.0, "final_neutral")
    assert not report.summary()["checks"]["no_invalid_packets"]
    assert not report.summary()["input_checks_passed"]


def test_disconnect_and_reconnect_outside_their_phases_do_not_satisfy_gate():
    report = load_script("check_dualsense").ProbeReport(config())
    send(report)
    report.poll(2.0, "travel")
    send(report, at=2.1, phase="start")
    checks = report.summary()["checks"]
    assert not checks["disconnect"]
    assert not checks["reconnect"]


@pytest.mark.parametrize("command,wanted", [
    (["python3", "-m", "smores_ep.self_assembly_cli"], True),
    (["python3", str(ROOT / "ros2_bridge/mssr_file_bridge.py")], True),
    ([str(ROOT / "mssr_ws/install/mssr_expert/lib/mssr_expert/mssr_smores_teleop_node")], True),
    (["ros2", "launch", "mssr_expert", "smores_runtime.launch.py"], True),
    (["ros2", "run", "joy", "game_controller_node"], True),
    (["/opt/ros/humble/lib/nav2_controller/controller_server"], True),
    (["python3", "-m", "pytest", "mssr_ws/src/mssr_expert/test"], False),
    (["code", str(ROOT / "ros2_bridge/mssr_file_bridge.py")], False),
    (["python3", "-c", "print('mssr_smores_teleop_node')"], False),
    (["/bin/bash", "-c", "ros2 launch mssr_expert smores_runtime.launch.py"], False),
    (["python3", "/other/project/ros2_bridge/mssr_file_bridge.py"], False),
    (["ros2", "launch", "unrelated_package", "camera.launch.py"], False),
    (["python3", "-m", "smores_ep.tests.test_geometry"], False),
    (["/other/isaac/kit", "--enable", "unrelated.extension"], False),
])
def test_cleanup_classifies_argument_positions_without_substring_kills(command, wanted):
    assert load_script("runtime_cleanup").is_runtime_command(command, ROOT) is wanted


def test_reused_pid_is_never_signalled(monkeypatch):
    cleanup = load_script("runtime_cleanup")
    original = cleanup.ProcessIdentity(100, 10, ("python3", "-m", "smores_ep.self_assembly_cli"))
    replacement = cleanup.ProcessIdentity(100, 11, ("python3", "unrelated.py"))
    monkeypatch.setattr(cleanup, "read_process", lambda pid: replacement)
    signalled = []
    monkeypatch.setattr(cleanup.os, "kill", lambda pid, sig: signalled.append(pid))
    assert not cleanup.signal_if_same(original, cleanup.signal.SIGTERM)
    assert signalled == []


def test_sdl_device_listing_identifies_sony_dualsense_guid():
    probe = load_script("check_dualsense")
    text = "ID : GUID : GamePad : Mapped : Joystick Device Name\n" + (
        "0 : 030000004c050000e60c000000010000 : true : true : Wireless Controller\n"
        "1 : 030000005e0400008e02000000010000 : true : true : Other Controller\n"
    )
    devices = probe.parse_devices(text)
    assert devices[0]["dualsense"]
    assert not devices[1]["dualsense"]
    assert devices[0]["device_id"] == 0


@pytest.mark.parametrize("ignore_term", [False, True])
def test_scoped_cleanup_stops_only_selected_real_processes(tmp_path, monkeypatch, ignore_term):
    cleanup = load_script("runtime_cleanup")
    installed = tmp_path / "mssr_ws/install/mssr_expert/lib/mssr_expert/mssr_smores_test_node"
    installed.parent.mkdir(parents=True)
    installed.write_text("import signal, time\n" +
                         ("signal.signal(signal.SIGTERM, signal.SIG_IGN)\n" if ignore_term else "") +
                         "print('ready', flush=True)\ntime.sleep(30)\n")
    unrelated = tmp_path / "unrelated.py"
    unrelated.write_text("import time\nprint('ready', flush=True)\ntime.sleep(30)\n")
    selected = subprocess.Popen([sys.executable, str(installed)], cwd=tmp_path,
                                stdout=subprocess.PIPE, text=True)
    other = subprocess.Popen([sys.executable, str(unrelated)], cwd=tmp_path,
                             stdout=subprocess.PIPE, text=True)
    try:
        assert selected.stdout.readline().strip() == "ready"
        assert other.stdout.readline().strip() == "ready"
        def daemon_stop(command, **kwargs):
            assert command == ["ros2", "daemon", "stop"]
            return SimpleNamespace(returncode=0, stdout="Daemon stopped", stderr="")
        monkeypatch.setattr(cleanup.subprocess, "run", daemon_stop)
        report = cleanup.scoped_cleanup(tmp_path, grace_s=0.2)
        selected.wait(timeout=2)
        assert selected.returncode == (-9 if ignore_term else -15)
        assert other.poll() is None
        assert [item["pid"] for item in report["selected"]] == [selected.pid]
        assert report["sigkill"] == ([selected.pid] if ignore_term else [])
        assert report["remaining"] == []
    finally:
        for process in (selected, other):
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=1)


def test_actual_kinematic_module_is_included_in_runtime_cleanup():
    assert load_script("runtime_cleanup").is_runtime_command(
        ["python3", "-m", "smores_ep.cli"], ROOT)


@pytest.mark.parametrize("pressed,axes,ending", [
    (True, [0] * 6, 1.6), (False, [0, 0, -1, 0, 0, 0], 1.6),
    (False, [0] * 6, 2.0),
])
def test_final_neutral_requires_fresh_released_controls_at_phase_end(pressed, axes, ending):
    report = load_script("check_dualsense").ProbeReport(config())
    send(report, at=1.0, phase="final_neutral")
    send(report, axes=axes, pressed=pressed, at=1.5, phase="final_neutral")
    report.end_phase(ending, "final_neutral")
    assert not report.summary()["checks"]["final_neutral"]


def test_single_transient_zero_packet_does_not_pass_neutral_gate():
    report = load_script("check_dualsense").ProbeReport(config())
    send(report, at=1.0, phase="neutral")
    report.end_phase(1.1, "neutral")
    assert not report.summary()["checks"]["initial_neutral"]


def test_neutral_settling_period_restarts_after_a_stale_gap():
    report = load_script("check_dualsense").ProbeReport(config())
    send(report, at=1.0, phase="final_neutral")
    send(report, at=2.0, phase="final_neutral")
    report.end_phase(2.1, "final_neutral")
    assert not report.summary()["checks"]["final_neutral"]


def test_hardware_driver_uses_the_same_exclusive_topic_as_its_consumer():
    probe = load_script("check_dualsense")
    topic = "/mssr/teleop_probe/run_test/joy"
    command = probe.build_driver_command(2, topic)
    assert command[0] == "/opt/ros/humble/lib/joy/game_controller_node"
    assert "joy:=" + topic in command
    assert "device_id:=2" in command
    with pytest.raises(ValueError):
        probe.build_driver_command(2, "/joy")


def test_device_preflight_does_not_claim_an_input_hardware_pass(tmp_path, monkeypatch):
    probe = load_script("check_dualsense")
    monkeypatch.setitem(sys.modules, "runtime_cleanup", SimpleNamespace(
        scoped_cleanup=lambda root: {"remaining": [], "daemon_stopped": True}))
    listing = "0 : 030000004c050000e60c000000010000 : true : true : DualSense\n"
    monkeypatch.setattr(probe.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(
        stdout=listing, stderr="", returncode=0))
    def forbidden_runtime(**kwargs):
        raise AssertionError("device-only preflight must not initialize ROS runtime")
    monkeypatch.setitem(sys.modules, "rclpy", SimpleNamespace(init=forbidden_runtime))
    monkeypatch.setitem(sys.modules, "rclpy.qos", SimpleNamespace(qos_profile_sensor_data=None))
    monkeypatch.setitem(sys.modules, "sensor_msgs.msg", SimpleNamespace(Joy=None))
    metadata = {}
    result, exit_code = probe.run(SimpleNamespace(device_id=0, preflight_only=True,
        config=ROOT / "mssr_ws/src/mssr_expert/config/smores_dualsense.yaml"), tmp_path, metadata)
    assert exit_code == 2
    assert metadata["terminal_reason"] == "hardware_acceptance_pending"
    assert not result["input_checks_passed"]
    assert result["valid_packets"] == 0


def test_default_probe_domain_fits_dds_ports_and_avoids_linux_ephemeral_ports(tmp_path):
    probe = load_script("check_dualsense")
    environment = {}
    probe.configure_probe_environment(environment, tmp_path)
    assert 0 <= int(environment["ROS_DOMAIN_ID"]) <= 101
    assert environment["ROS_LOG_DIR"] == str(tmp_path / "ros_logs")


@pytest.mark.parametrize("domain", ["0", "42", "101", "215", "232"])
def test_valid_user_domain_is_preserved(tmp_path, domain):
    environment = {"ROS_DOMAIN_ID": domain}
    load_script("check_dualsense").configure_probe_environment(environment, tmp_path)
    assert environment["ROS_DOMAIN_ID"] == domain


@pytest.mark.parametrize("domain", ["239", "233", "-1", "", "abc", "4.2",
                                  "08", "0x2a", "４２", "4_2", " 42", "+42"])
def test_invalid_domain_fails_before_runtime_without_mutating_environment(tmp_path, domain):
    environment = {"ROS_DOMAIN_ID": domain}
    with pytest.raises(ValueError, match="ROS_DOMAIN_ID"):
        load_script("check_dualsense").configure_probe_environment(environment, tmp_path)
    assert environment == {"ROS_DOMAIN_ID": domain}


def test_previous_failed_probe_is_a_cleanup_candidate():
    assert load_script("runtime_cleanup").is_runtime_command(
        ["python3", str(ROOT / "scripts/teleop/check_dualsense.py")], ROOT)
