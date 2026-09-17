"""T2 safety latch and camera evolution without ROS or Isaac."""
import importlib
import importlib.util
import math
from types import SimpleNamespace

import pytest


def component(name):
    module = "mssr_expert.teleop." + name
    assert importlib.util.find_spec(module) is not None, f"missing T2 {name}"
    return importlib.import_module(module)


def gate():
    return component("safety").SafetyGate()


def update(safety, *, connected=True, l2=0.0, r2=0.0, at=10.0, macro=False, topology=True):
    return safety.update(connected=connected, l2=l2, r2=r2,
                         received_at=at, macro_active=macro, topology_supported=topology)


def test_startup_held_trigger_does_not_arm():
    safety = gate()
    decision = update(safety, r2=0.6)
    assert not decision.motion_enabled and decision.zero_wheels
    assert decision.preserve_targets and not decision.allow_joint_updates


def test_fresh_neutral_arms_then_analog_input_is_allowed():
    safety = gate()
    assert update(safety).motion_enabled
    assert update(safety, r2=0.5, at=10.1).motion_enabled


def test_pause_is_idempotent_and_resume_is_explicit():
    safety = gate()
    update(safety)
    assert safety.pause()
    assert not safety.pause()
    assert update(safety, at=11).authority == "ESTOP"
    assert safety.resume(resumed_at=11.0)
    assert not safety.resume(resumed_at=11.0)


def test_resume_cannot_rearm_from_cached_pre_resume_neutral():
    safety = gate()
    update(safety)
    safety.pause()
    update(safety, at=11)
    safety.resume(resumed_at=11.0)
    assert not update(safety, at=11).motion_enabled
    assert update(safety, at=11.1).motion_enabled


def test_resume_with_held_trigger_requires_new_neutral_then_repress():
    safety = gate()
    update(safety)
    safety.pause()
    safety.resume(resumed_at=10.0)
    assert not update(safety, r2=1.0, at=11).motion_enabled
    assert not update(safety, l2=0.1, at=11.1).motion_enabled
    assert update(safety, at=11.2).motion_enabled
    assert update(safety, r2=1.0, at=11.3).motion_enabled


def test_disconnect_stops_teleop_and_preserves_targets():
    safety = gate()
    update(safety)
    decision = update(safety, connected=False, r2=1.0)
    assert decision.authority == "NONE" and decision.zero_wheels
    assert decision.preserve_targets and not decision.allow_joint_updates
    assert not update(safety, r2=1.0, at=11).motion_enabled
    assert update(safety, at=11.1).motion_enabled


def test_disconnection_does_not_take_actuators_from_structural_macro():
    safety = gate()
    decision = update(safety, connected=False, macro=True)
    assert decision.authority == "STRUCTURAL_MACRO"
    assert not decision.zero_wheels and not decision.allow_joint_updates
    safety.pause()
    assert update(safety, connected=False, macro=True).authority == "ESTOP"


def test_unsupported_topology_cannot_get_motion_authority():
    safety = gate()
    decision = update(safety, topology=False)
    assert not decision.motion_enabled and decision.authority == "NONE"


@pytest.mark.parametrize("l2,r2,at", [(float("nan"), 0, 10), (0, float("inf"), 10),
                                      (-0.1, 0, 10), (0, 1.1, 10), (0, 0, None)])
def test_invalid_input_cannot_arm_or_keep_authority(l2, r2, at):
    safety = gate()
    update(safety)
    decision = update(safety, l2=l2, r2=r2, at=at)
    assert not decision.motion_enabled and decision.zero_wheels


def test_out_of_order_receipt_cannot_rearm_after_resume():
    safety = gate()
    update(safety)
    safety.pause()
    safety.resume(resumed_at=10.0)
    assert not update(safety, at=9).motion_enabled
    assert update(safety, at=11).motion_enabled


def test_neutral_received_before_actual_resume_cannot_arm_when_consumed_later():
    safety = gate()
    update(safety, at=10.0)
    safety.pause()
    # Joy arrived at 10.9 while paused; the next gate update occurs after resume.
    safety.resume(resumed_at=11.0)
    assert not update(safety, at=10.9).motion_enabled
    assert not update(safety, connected=False, at=10.9).motion_enabled
    assert not update(safety, at=11.0).motion_enabled
    assert update(safety, at=11.1).motion_enabled


def sample(x=0.0, y=0.0, connected=True, **other):
    return SimpleNamespace(left_x=x, left_y=y, connected=connected, **other)


def camera(**kwargs):
    return component("camera").CameraController(**kwargs)


def test_camera_uses_only_left_stick_and_holds_on_release():
    controller = camera()
    initial = controller.step(sample(right_x=1, r2=1), 0.1)
    moved = controller.step(sample(x=1, y=0.5), 0.1)
    assert moved.azimuth_rad != initial.azimuth_rad
    assert moved.elevation_rad > initial.elevation_rad
    assert controller.step(sample(), 0.1) == moved


def test_camera_holds_when_controller_disconnected():
    controller = camera()
    initial = controller.step(sample(), 0.1)
    assert controller.step(sample(x=1, y=1, connected=False), 0.1) == initial


def test_camera_clamps_elevation_and_wraps_azimuth():
    controller = camera()
    for _ in range(200):
        orbit = controller.step(sample(x=1, y=1), 0.1)
    assert -math.pi <= orbit.azimuth_rad < math.pi
    assert orbit.elevation_rad == pytest.approx(math.radians(85))
    for _ in range(200):
        orbit = controller.step(sample(y=-1), 0.1)
    assert orbit.elevation_rad == pytest.approx(math.radians(5))


def test_camera_step_is_bounded_after_long_scheduler_gap():
    controller = camera()
    initial = controller.step(sample(), 0.0)
    moved = controller.step(sample(x=1), 100)
    assert abs(moved.azimuth_rad - initial.azimuth_rad) <= math.radians(90) * 0.1 + 1e-9


def test_camera_geometry_tracks_supplied_robot_center():
    controller = camera(radius_m=2)
    orbit = controller.step(sample(), 0.0)
    eye, target = orbit.view((3.0, 4.0, 0.2))
    assert target == (3.0, 4.0, 0.2)
    assert math.dist(eye, target) == pytest.approx(2)
    assert eye[2] > target[2]


@pytest.mark.parametrize("dt", [-1, float("nan"), float("inf")])
def test_camera_rejects_invalid_wall_delta(dt):
    with pytest.raises(ValueError):
        camera().step(sample(x=1), dt)


@pytest.mark.parametrize("radius", [0, -1, float("nan"), float("inf")])
def test_camera_rejects_invalid_radius(radius):
    with pytest.raises(ValueError):
        camera(radius_m=radius)
