from __future__ import annotations

import json
import math
from types import SimpleNamespace

import pytest

from smores_ep.control.teleop import InternalMotionMode, SmoresCommand
from smores_ep.control.differential_drive import (
    PlanarPose,
    twist_to_wheel_rates,
)
from smores_ep.docking.model import DockingFace, DockingFacePose
from smores_ep.primitives.file_channel import ActionFileChannel, PrimitiveFileChannel
from smores_ep.primitives.model import (
    PrimitiveGoal,
    PrimitiveName,
    PrimitiveState,
    PrimitiveStatus,
)
from smores_ep.primitives.pose_control import (
    axial_pose_adjustment_reference,
    drive_to_pose_step,
)


def test_goal_parser_validates_shape_and_normalizes_faces() -> None:
    goal = PrimitiveGoal.from_dict(
        {
            "schema_version": "mssr.primitive_goal.v1",
            "goal_id": "dock-001",
            "primitive": "dock",
            "module_ids": ["module_a", "module_b"],
            "parameters": {
                "face_a": "top",
                "face_b": "base-chassis",
            },
            "timeout_s": 4.0,
        }
    )
    assert goal.primitive is PrimitiveName.DOCK
    assert goal.parameters["face_a"] == "TOP"
    assert goal.parameters["face_b"] == "BOTTOM"

    with pytest.raises(ValueError):
        PrimitiveGoal.from_dict(
            {
                "goal_id": "bad",
                "primitive": "dock",
                "module_ids": ["only_one"],
                "parameters": {"face_a": "TOP", "face_b": "BOTTOM"},
            }
        )


def test_align_goal_validates_collective_execution_phase() -> None:
    goal = PrimitiveGoal.from_dict(
        {
            "goal_id": "align-phase",
            "primitive": "align_faces",
            "module_ids": ["module_a", "module_b"],
            "parameters": {
                "face_a": "bottom",
                "face_b": "top",
                "execution_phase": "APPROACH",
            },
        }
    )
    assert goal.parameters["execution_phase"] == "approach"

    with pytest.raises(ValueError, match="execution_phase"):
        PrimitiveGoal.from_dict(
            {
                "goal_id": "bad-align-phase",
                "primitive": "align_faces",
                "module_ids": ["module_a", "module_b"],
                "parameters": {
                    "face_a": "BOTTOM",
                    "face_b": "TOP",
                    "execution_phase": "dock",
                },
            }
        )


def test_align_goal_validates_staging_path_fallback_level() -> None:
    goal = PrimitiveGoal.from_dict(
        {
            "goal_id": "align-fallback",
            "primitive": "align_faces",
            "module_ids": ["module_a", "module_b"],
            "parameters": {
                "face_a": "BOTTOM",
                "face_b": "TOP",
                "staging_path_fallback_level": 2,
            },
        }
    )
    assert goal.parameters["staging_path_fallback_level"] == 2

    for invalid_level in (-1, 3, 1.5, True):
        with pytest.raises(ValueError, match="staging_path_fallback_level"):
            PrimitiveGoal.from_dict(
                {
                    "goal_id": f"bad-align-fallback-{invalid_level}",
                    "primitive": "align_faces",
                    "module_ids": ["module_a", "module_b"],
                    "parameters": {
                        "face_a": "BOTTOM",
                        "face_b": "TOP",
                        "staging_path_fallback_level": invalid_level,
                    },
                }
            )


def test_relative_pan_goal_accepts_unbounded_finite_delta() -> None:
    goal = PrimitiveGoal.from_dict(
        {
            "goal_id": "pan-001",
            "primitive": "rotate_pan_by",
            "module_ids": ["module_a"],
            "parameters": {"delta_rad": 5.0 * math.pi},
        }
    )
    assert goal.parameters["delta_rad"] == pytest.approx(5.0 * math.pi)


def test_tilt_goal_validates_optional_coordination_group() -> None:
    goal = PrimitiveGoal.from_dict(
        {
            "goal_id": "tilt-group-001",
            "primitive": "set_tilt",
            "module_ids": ["module_a"],
            "parameters": {
                "angle_rad": -0.8,
                "coordination_group": "car-posture",
                "coordination_size": 4,
            },
        }
    )
    assert goal.parameters["coordination_group"] == "car-posture"
    assert goal.parameters["coordination_size"] == 4

    with pytest.raises(ValueError, match="coordination_size"):
        PrimitiveGoal.from_dict(
            {
                "goal_id": "tilt-group-bad",
                "primitive": "set_tilt",
                "module_ids": ["module_a"],
                "parameters": {
                    "angle_rad": -0.8,
                    "coordination_group": "car-posture",
                    "coordination_size": 0,
                },
            }
        )


def test_tilt_goal_validates_coupled_fold_pusher() -> None:
    goal = PrimitiveGoal.from_dict(
        {
            "goal_id": "holonomic-fold",
            "primitive": "set_tilt",
            "module_ids": ["lifter"],
            "parameters": {
                "angle_rad": 1.35,
                "pusher_module_id": "spoke",
                "pusher_linear_m_s": -0.025,
            },
        }
    )
    assert goal.parameters["pusher_module_id"] == "spoke"
    assert goal.parameters["pusher_linear_m_s"] == pytest.approx(-0.025)

    with pytest.raises(ValueError, match="supplied together"):
        PrimitiveGoal.from_dict(
            {
                "goal_id": "bad-fold",
                "primitive": "set_tilt",
                "module_ids": ["lifter"],
                "parameters": {
                    "angle_rad": 1.35,
                    "pusher_module_id": "spoke",
                },
            }
        )


def test_assisted_alignment_goal_requires_payload_target_and_helper() -> None:
    goal = PrimitiveGoal.from_dict(
        {
            "goal_id": "help-align-001",
            "primitive": "assisted_align_faces",
            "module_ids": ["mobile_a", "root", "mobile_b"],
            "parameters": {
                "face_a": "left",
                "face_b": "right",
                "clocking_quarter_turns": 0,
            },
        }
    )
    assert goal.primitive is PrimitiveName.ASSISTED_ALIGN_FACES
    assert goal.parameters["face_a"] == "LEFT"
    with pytest.raises(ValueError, match="requires 3"):
        PrimitiveGoal.from_dict(
            {
                "goal_id": "bad-help-align",
                "primitive": "assisted_align_faces",
                "module_ids": ["mobile_a", "root"],
                "parameters": {"face_a": "LEFT", "face_b": "RIGHT"},
            }
        )


def test_pose_controller_respects_non_holonomic_motion() -> None:
    rotate = drive_to_pose_step(
        PlanarPose(),
        PlanarPose(0.0, 1.0, 0.0),
    )
    assert rotate.phase == "orient_to_path"
    assert rotate.linear_x_m_s == 0.0
    assert rotate.angular_z_rad_s > 0.0

    translate = drive_to_pose_step(
        PlanarPose(),
        PlanarPose(1.0, 0.0, 0.0),
    )
    assert translate.phase == "translate"
    assert translate.linear_x_m_s > 0.0
    assert translate.angular_z_rad_s == pytest.approx(0.0)

    done = drive_to_pose_step(
        PlanarPose(1.0, 2.0, 0.5),
        PlanarPose(1.0, 2.0, 0.5),
    )
    assert done.done


def test_paper_pose_adjustment_reference_realizes_lateral_feedback() -> None:
    reference = axial_pose_adjustment_reference(
        lateral_error_m=0.004,
        drive_direction=-1.0,
    )

    assert not reference.saturated
    assert reference.lateral_velocity_m_s == pytest.approx(-2.0 * 0.004)
    assert reference.desired_relative_yaw_rad > 0.0

    saturated = axial_pose_adjustment_reference(0.020, -1.0)
    assert saturated.saturated
    assert abs(saturated.desired_relative_yaw_rad) == pytest.approx(
        math.radians(25.0)
    )


def test_file_channel_delivers_new_goal_once_and_writes_status(tmp_path) -> None:
    goal_path = tmp_path / "goal.json"
    cancel_path = tmp_path / "cancel.json"
    status_path = tmp_path / "status.json"
    channel = PrimitiveFileChannel(
        goal_path,
        cancel_path,
        status_path,
        ignore_existing=True,
    )
    payload = {
        "goal_id": "tilt-001",
        "primitive": "set_tilt",
        "module_ids": ["module_a"],
        "parameters": {"angle_rad": 0.5},
    }
    goal_path.write_text(json.dumps(payload), encoding="utf-8")
    goal = channel.poll_goal()
    assert goal is not None
    assert goal.primitive is PrimitiveName.SET_TILT
    assert channel.poll_goal() is None

    cancel_path.write_text(
        json.dumps({"goal_id": "tilt-001"}),
        encoding="utf-8",
    )
    assert channel.poll_cancel() == "tilt-001"
    assert channel.poll_cancel() is None

    status = PrimitiveStatus(
        goal_id="tilt-001",
        primitive=PrimitiveName.SET_TILT,
        state=PrimitiveState.RUNNING,
        stamp_s=1.25,
        module_ids=("module_a",),
        phase="tilt",
        progress=0.5,
    )
    channel.publish(status)
    decoded = json.loads(status_path.read_text(encoding="utf-8"))
    assert decoded["schema_version"] == "mssr.primitive_status.v1"
    assert decoded["state"] == "running"

    channel.publish_many([status], stamp_s=1.5)
    decoded = json.loads(status_path.read_text(encoding="utf-8"))
    assert decoded["schema_version"] == "mssr.primitive_status_batch.v1"
    assert decoded["statuses"][0]["goal_id"] == "tilt-001"


def test_file_channel_detects_same_goal_payload_rewritten_atomically(
    tmp_path,
) -> None:
    goal_path = tmp_path / "goal.json"
    cancel_path = tmp_path / "cancel.json"
    status_path = tmp_path / "status.json"
    payload = json.dumps(
        {
            "goal_id": "repeatable-goal",
            "primitive": "set_tilt",
            "module_ids": ["module_a"],
            "parameters": {"angle_rad": 0.5},
        }
    )
    goal_path.write_text(payload, encoding="utf-8")
    channel = PrimitiveFileChannel(
        goal_path,
        cancel_path,
        status_path,
        ignore_existing=True,
    )

    assert channel.poll_goal() is None

    replacement = tmp_path / "goal.replacement.json"
    replacement.write_text(payload, encoding="utf-8")
    replacement.replace(goal_path)

    repeated = channel.poll_goal()
    assert repeated is not None
    assert repeated.goal_id == "repeatable-goal"
    assert channel.poll_goal() is None


def test_action_file_channel_routes_smores_locomotion_and_times_out(
    tmp_path,
) -> None:
    action_path = tmp_path / "actions.json"
    channel = ActionFileChannel(
        action_path,
        timeout_s=0.5,
        ignore_existing=True,
    )
    action_path.write_text(
        json.dumps(
            {
                "schema_version": "mssr.actions.v2",
                "locomotion": {
                    "wheel_a": {
                        "vx": 0.06,
                        "vy": 0.0,
                        "yaw_rate": -0.2,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    commands = channel.commands(10.0)
    assert commands["wheel_a"].linear_x_m_s == pytest.approx(0.06)
    assert commands["wheel_a"].angular_z_rad_s == pytest.approx(-0.2)
    assert channel.commands(10.49)
    assert channel.commands(10.51) == {}


def test_action_file_channel_rejects_holonomic_lateral_velocity(
    tmp_path,
) -> None:
    action_path = tmp_path / "actions.json"
    channel = ActionFileChannel(action_path, ignore_existing=True)
    action_path.write_text(
        json.dumps(
            {
                "schema_version": "mssr.actions.v2",
                "locomotion": {"module": {"vx": 0.0, "vy": 0.1}},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="lateral"):
        channel.commands(1.0)


def test_action_file_channel_routes_continuous_pan_drive(tmp_path) -> None:
    action_path = tmp_path / "actions.json"
    channel = ActionFileChannel(action_path, ignore_existing=True)
    action_path.write_text(
        json.dumps(
            {
                "schema_version": "mssr.actions.v2",
                "locomotion": {
                    "wheel_module": {
                        "vx": 0.0,
                        "vy": 0.0,
                        "yaw_rate": 0.0,
                        "pan_rate_rad_s": -1.25,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    command = channel.commands(1.0)["wheel_module"]
    assert command.internal_motion is InternalMotionMode.PAN_VELOCITY
    assert command.pan_velocity_rad_s == pytest.approx(-1.25)


def test_action_file_wheel_drive_retains_internal_shape(
    tmp_path,
) -> None:
    action_path = tmp_path / "actions.json"
    channel = ActionFileChannel(action_path, ignore_existing=True)
    action_path.write_text(
        json.dumps(
            {
                "schema_version": "mssr.actions.v2",
                "locomotion": {
                    "longitudinal_support": {
                        "vx": 0.03,
                        "vy": 0.0,
                        "yaw_rate": 0.0,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    command = channel.commands(1.0)["longitudinal_support"]
    assert command.internal_motion is InternalMotionMode.HOLD
    assert command.linear_x_m_s == pytest.approx(0.03)


class _FakeState:
    pan_joint_rad = 0.0
    tilt_joint_rad = 0.0


class _FakeStateReader:
    def read(self) -> _FakeState:
        return _FakeState()


class _MutableStateReader:
    def __init__(self, tilt_rad: float = 0.0, pan_rad: float = 0.0) -> None:
        self.state = SimpleNamespace(
            pan_joint_rad=pan_rad,
            # Isaac's articulation state and the public tilt convention use
            # opposite signs.
            tilt_joint_rad=-tilt_rad,
        )

    def read(self):
        return self.state


class _FakeDynamicArticulation:
    def __init__(self) -> None:
        self.state = SimpleNamespace(pan_joint_rad=0.0, tilt_joint_rad=0.0)
        self.targets: dict[str, float] = {}

    def read(self):
        return self.state

    def set_targets(self, **targets: float) -> None:
        self.targets = dict(targets)


def test_dynamic_controller_runs_pan_velocity_then_holds_current_angle() -> None:
    from smores_ep.config.geometry import SmoresGeometry
    from smores_ep.isaac.dynamic_stage import DynamicDriveController

    articulation = _FakeDynamicArticulation()
    controller = DynamicDriveController(
        articulation,  # type: ignore[arg-type]
        SmoresGeometry(),
        max_wheel_speed_rad_s=2.4,
    )
    drive = SmoresCommand(
        internal_motion=InternalMotionMode.PAN_VELOCITY,
        pan_velocity_rad_s=1.5,
        tilt_target_rad=0.4,
    )

    controller.apply(drive)
    assert articulation.targets["pan_joint_velocity_rad_s"] == pytest.approx(
        1.5
    )
    assert articulation.targets["tilt_joint_position_rad"] == pytest.approx(
        -0.4
    )
    articulation.state.pan_joint_rad = 0.2
    controller.apply(drive)
    controller.apply(SmoresCommand())
    assert articulation.targets["pan_joint_velocity_rad_s"] == pytest.approx(
        0.0
    )


class _FakeDocking:
    module_ids = ("module_a", "module_b")
    connections = ()


class _FakeDockingThree:
    module_ids = ("root", "mobile_a", "mobile_b")


class _FakeDockingFour:
    module_ids = ("module_a", "module_b", "module_c", "module_d")


class _FakeContactDocking:
    module_ids = ("mobile", "target")

    def __init__(self) -> None:
        self.normal_gap_m = 0.006
        self.handle_count = 0
        self.handle_thresholds = None

    def face_poses_for(self, module_id: str):
        if module_id == "mobile":
            return (
                DockingFacePose(
                    DockingFace("mobile", "BOTTOM", "/m/f", "/m/b"),
                    (0.0, 0.0, 0.0),
                    (1.0, 0.0, 0.0),
                    (0.0, 1.0, 0.0),
                ),
            )
        return (
            DockingFacePose(
                DockingFace("target", "TOP", "/t/f", "/t/b"),
                (self.normal_gap_m, 0.0, 0.0),
                (-1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
            ),
        )

    def handle(self, _command, thresholds=None, snap_to_nominal=False):
        self.handle_count += 1
        self.handle_thresholds = thresholds
        return SimpleNamespace(accepted=True, message="attached")


def _goal(
    goal_id: str,
    primitive: PrimitiveName,
    module_ids: tuple[str, ...],
    parameters: dict[str, object],
) -> PrimitiveGoal:
    return PrimitiveGoal(goal_id, primitive, module_ids, parameters)


def test_dock_closes_contact_forward_before_creating_joint() -> None:
    from smores_ep.isaac.primitive_executor import IsaacPrimitiveExecutor

    docking = _FakeContactDocking()
    executor = IsaacPrimitiveExecutor(
        stage=object(),
        module_roots={"mobile": "/M", "target": "/T"},
        states={
            "mobile": _FakeStateReader(),
            "target": _FakeStateReader(),
        },
        docking=docking,  # type: ignore[arg-type]
    )
    executor._planar_pose = lambda _module_id: PlanarPose(  # type: ignore[method-assign]
        0.0, 0.0, 0.0
    )
    goal = _goal(
        "contact-then-dock",
        PrimitiveName.DOCK,
        ("mobile", "target"),
        {"face_a": "BOTTOM", "face_b": "TOP"},
    )
    assert executor.submit(goal, 0.0).state is PrimitiveState.ACCEPTED

    closing = executor.step(0.1)
    assert closing.statuses[0].code == "CLOSING_CONTACT"
    assert closing.commands["mobile"].linear_x_m_s > 0.0
    assert docking.handle_count == 0
    target = executor._face_alignment_target(
        "mobile", "BOTTOM", "target", "TOP"
    )
    assert target.x_m == pytest.approx(0.006)

    docking.normal_gap_m = 0.0005
    docked = executor.step(0.2)
    assert docked.statuses[0].code == "DOCKED"
    assert docking.handle_count == 1


def test_dock_uses_morphology_specific_top_bottom_contact_gate() -> None:
    from smores_ep.isaac.primitive_executor import IsaacPrimitiveExecutor

    docking = _FakeContactDocking()
    docking.normal_gap_m = 0.00302
    executor = IsaacPrimitiveExecutor(
        stage=object(),
        module_roots={"mobile": "/M", "target": "/T"},
        states={
            "mobile": _FakeStateReader(),
            "target": _FakeStateReader(),
        },
        docking=docking,  # type: ignore[arg-type]
    )
    goal = _goal(
        "manipulator-contact-gate",
        PrimitiveName.DOCK,
        ("mobile", "target"),
        {
            "face_a": "BOTTOM",
            "face_b": "TOP",
            "top_bottom_contact_tolerance_m": 0.0035,
        },
    )
    assert executor.submit(goal, 0.0).state is PrimitiveState.ACCEPTED

    docked = executor.step(0.1)

    assert docked.statuses[0].code == "DOCKED"
    assert docking.handle_count == 1
    assert docking.handle_thresholds is not None
    assert (
        docking.handle_thresholds.top_bottom_contact_tolerance_m
        == pytest.approx(0.0035)
    )


def test_contact_timeout_does_not_restart_alignment() -> None:
    from smores_ep.isaac.primitive_executor import IsaacPrimitiveExecutor

    executor = IsaacPrimitiveExecutor(
        stage=object(),
        module_roots={"module_a": "/A", "module_b": "/B"},
        states={
            "module_a": _FakeStateReader(),
            "module_b": _FakeStateReader(),
        },
        docking=_FakeDocking(),  # type: ignore[arg-type]
    )
    goal = _goal(
        "contact-timeout",
        PrimitiveName.ALIGN_FACES,
        ("module_a", "module_b"),
        {"face_a": "BOTTOM", "face_b": "TOP"},
    )
    executor.submit(goal, 0.0)
    executor._active[goal.goal_id].alignment_approach_started = True

    timed_out = executor.step(31.0)
    assert timed_out.statuses[0].code == "CONTACT_TIMEOUT"
    assert "will not be restarted" in timed_out.statuses[0].message


def test_collective_face_phases_do_not_cross_their_barriers() -> None:
    from smores_ep.isaac.primitive_executor import IsaacPrimitiveExecutor

    executor = IsaacPrimitiveExecutor(
        stage=object(),
        module_roots={"module_a": "/A", "module_b": "/B"},
        states={
            "module_a": _FakeStateReader(),
            "module_b": _FakeStateReader(),
        },
        docking=_FakeDocking(),  # type: ignore[arg-type]
    )
    mobile_face = DockingFacePose(
        DockingFace("module_a", "BOTTOM", "/A/BOTTOM", "/A/body"),
        (0.04, 0.0, 0.0),
        (-1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
    )
    target_face = DockingFacePose(
        DockingFace("module_b", "TOP", "/B/TOP", "/B/body"),
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
    )
    executor._face_pose = (  # type: ignore[method-assign]
        lambda module_id, _face: (
            mobile_face if module_id == "module_a" else target_face
        )
    )
    executor._planar_pose = (  # type: ignore[method-assign]
        lambda _module_id: PlanarPose(0.04, 0.0, 0.0)
    )
    executor._face_alignment_target = (  # type: ignore[method-assign]
        lambda *_args: PlanarPose(0.0, 0.0, 0.0)
    )
    executor._face_alignment_staging_target = (  # type: ignore[method-assign]
        lambda *_args, **_kwargs: PlanarPose(0.04, 0.0, 0.0)
    )

    def phase_goal(phase: str) -> PrimitiveGoal:
        return _goal(
            f"collective-{phase}",
            PrimitiveName.ALIGN_FACES,
            ("module_a", "module_b"),
            {
                "face_a": "BOTTOM",
                "face_b": "TOP",
                "execution_phase": phase,
                "contact_approach_feedback": True,
            },
        )

    reach = phase_goal("reach")
    executor.submit(reach, 0.0)
    reached = executor.step(0.1)
    assert reached.commands == {}
    assert reached.statuses[0].code == "STAGING_REACHED"

    align = phase_goal("align")
    executor.submit(align, 0.2)
    aligned = executor.step(0.3)
    assert aligned.commands == {}
    assert aligned.statuses[0].code == "FACES_ALIGNED"

    approach = phase_goal("approach")
    executor.submit(approach, 0.4)
    approaching = executor.step(0.5)
    assert approaching.statuses[0].code == "CLOSING_CONTACT"
    assert approaching.commands["module_a"].linear_x_m_s < 0.0


def test_completed_contact_approach_never_runs_with_zero_command() -> None:
    """An invalid final contact must terminate instead of deadlocking."""

    from smores_ep.isaac.primitive_executor import IsaacPrimitiveExecutor

    executor = IsaacPrimitiveExecutor(
        stage=object(),
        module_roots={"module_a": "/A", "module_b": "/B"},
        states={
            "module_a": _FakeStateReader(),
            "module_b": _FakeStateReader(),
        },
        docking=_FakeDocking(),  # type: ignore[arg-type]
    )
    mobile_face = DockingFacePose(
        DockingFace("module_a", "TOP", "/A/TOP", "/A/body"),
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
    )
    target_face = DockingFacePose(
        DockingFace("module_b", "BOTTOM", "/B/BOTTOM", "/B/body"),
        (0.0005, 0.011, 0.0),
        (-1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
    )
    executor._face_pose = (  # type: ignore[method-assign]
        lambda module_id, _face: (
            mobile_face if module_id == "module_a" else target_face
        )
    )
    executor._planar_pose = (  # type: ignore[method-assign]
        lambda _module_id: PlanarPose(0.0, 0.0, 0.0)
    )
    executor._face_alignment_target = (  # type: ignore[method-assign]
        lambda *_args: PlanarPose(0.0, 0.0, 0.0)
    )
    goal = _goal(
        "invalid-contact",
        PrimitiveName.ALIGN_FACES,
        ("module_a", "module_b"),
        {"face_a": "TOP", "face_b": "BOTTOM"},
    )
    executor.submit(goal, 0.0)
    executor._active[goal.goal_id].alignment_approach_started = True

    result = executor.step(0.1)

    assert result.commands == {}
    assert result.statuses[0].state is PrimitiveState.FAILED
    assert result.statuses[0].code == "CONTACT_POSE_INVALID"


def test_top_bottom_alignment_corrects_pan_clocking_before_contact() -> None:
    """A driven TOP disk must be reclocked before it can dock as a chain."""

    from smores_ep.isaac.primitive_executor import IsaacPrimitiveExecutor

    executor = IsaacPrimitiveExecutor(
        stage=object(),
        module_roots={"module_a": "/A", "module_b": "/B"},
        states={
            "module_a": _MutableStateReader(),
            "module_b": _MutableStateReader(),
        },
        docking=_FakeDocking(),  # type: ignore[arg-type]
    )
    angle = math.radians(30.0)
    mobile_face = DockingFacePose(
        DockingFace("module_a", "TOP", "/A/TOP", "/A/body"),
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
    )
    target_face = DockingFacePose(
        DockingFace("module_b", "BOTTOM", "/B/BOTTOM", "/B/body"),
        (0.001, 0.0, 0.0),
        (-1.0, 0.0, 0.0),
        (0.0, math.cos(angle), math.sin(angle)),
    )
    executor._face_pose = (  # type: ignore[method-assign]
        lambda module_id, _face: (
            mobile_face if module_id == "module_a" else target_face
        )
    )
    goal = _goal(
        "clock-top-before-dock",
        PrimitiveName.ALIGN_FACES,
        ("module_a", "module_b"),
        {"face_a": "TOP", "face_b": "BOTTOM"},
    )

    assert executor.submit(goal, 0.0).state is PrimitiveState.ACCEPTED
    result = executor.step(0.1)

    assert result.statuses[0].state is PrimitiveState.RUNNING
    assert result.statuses[0].code == "ALIGNING_CLOCKING"
    command = result.commands["module_a"]
    assert command.internal_motion is InternalMotionMode.PAN
    assert command.pan_target_rad == pytest.approx(angle)
    assert "internal_motion:module_a" in executor._resource_owners


def test_valid_but_decentered_contact_gets_one_bounded_parking_retry() -> None:
    """A quality retry backs away once, then falls back to the hard gate."""

    from smores_ep.isaac.primitive_executor import IsaacPrimitiveExecutor

    executor = IsaacPrimitiveExecutor(
        stage=object(),
        module_roots={"module_a": "/A", "module_b": "/B"},
        states={
            "module_a": _FakeStateReader(),
            "module_b": _FakeStateReader(),
        },
        docking=_FakeDocking(),  # type: ignore[arg-type]
    )
    mobile_face = DockingFacePose(
        DockingFace("module_a", "TOP", "/A/TOP", "/A/body"),
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
    )
    contact_face = DockingFacePose(
        DockingFace("module_b", "BOTTOM", "/B/BOTTOM", "/B/body"),
        (0.0005, 0.008, 0.0),
        (-1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
    )
    separated_face = DockingFacePose(
        contact_face.face,
        (0.020, 0.008, 0.0),
        contact_face.outward_normal_world,
        contact_face.tangent_world,
    )
    target_face = contact_face
    executor._face_pose = (  # type: ignore[method-assign]
        lambda module_id, _face: (
            mobile_face if module_id == "module_a" else target_face
        )
    )
    executor._planar_pose = (  # type: ignore[method-assign]
        lambda _module_id: PlanarPose(0.0, 0.0, 0.0)
    )
    executor._face_alignment_target = (  # type: ignore[method-assign]
        lambda *_args: PlanarPose(0.10, 0.0, 0.0)
    )
    executor._face_alignment_staging_target = (  # type: ignore[method-assign]
        lambda *_args, **_kwargs: PlanarPose(0.04, 0.0, 0.0)
    )
    goal = _goal(
        "quality-contact",
        PrimitiveName.ALIGN_FACES,
        ("module_a", "module_b"),
        {
            "face_a": "TOP",
            "face_b": "BOTTOM",
            "contact_quality_planar_tolerance_m": 0.0015,
            "contact_quality_retry_count": 1,
        },
    )
    executor.submit(goal, 0.0)

    retry = executor.step(0.1)
    assert retry.statuses[0].state is PrimitiveState.RUNNING
    assert retry.statuses[0].code == "REPOSITIONING_CONTACT"
    assert retry.commands["module_a"].linear_x_m_s != 0.0
    runtime = executor._active[goal.goal_id]
    assert runtime.alignment_recovery_count == 1

    # Simulate leaving contact and starting the new monotonic approach.
    target_face = separated_face
    runtime.alignment_approach_started = True
    approaching = executor.step(0.2)
    assert approaching.statuses[0].state is PrimitiveState.RUNNING
    assert not runtime.contact_quality_recovery_active

    # If the single improvement attempt cannot remove an unactuated CAD
    # offset, the original physical gate remains the final authority.
    target_face = contact_face
    accepted = executor.step(0.3)
    assert accepted.statuses[0].state is PrimitiveState.SUCCEEDED
    assert accepted.statuses[0].code == "FACES_IN_CONTACT"


def test_proximal_decentered_pair_backs_off_before_contact_gate() -> None:
    """A CAD-blocked near contact must recover instead of pushing forever."""

    from smores_ep.isaac.primitive_executor import IsaacPrimitiveExecutor

    executor = IsaacPrimitiveExecutor(
        stage=object(),
        module_roots={"module_a": "/A", "module_b": "/B"},
        states={
            "module_a": _FakeStateReader(),
            "module_b": _FakeStateReader(),
        },
        docking=_FakeDocking(),  # type: ignore[arg-type]
    )
    mobile_face = DockingFacePose(
        DockingFace("module_a", "TOP", "/A/TOP", "/A/body"),
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
    )
    # The normal gap is outside the strict 2.5 mm gate but inside the
    # proximal recovery window; the 3 mm planar error also exceeds the
    # morphology's requested 1.5 mm parking quality.
    target_face = DockingFacePose(
        DockingFace("module_b", "BOTTOM", "/B/BOTTOM", "/B/body"),
        (0.0038, 0.0030, 0.0),
        (-1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
    )
    executor._face_pose = (  # type: ignore[method-assign]
        lambda module_id, _face: (
            mobile_face if module_id == "module_a" else target_face
        )
    )
    executor._planar_pose = (  # type: ignore[method-assign]
        lambda _module_id: PlanarPose(0.0, 0.0, 0.0)
    )
    executor._face_alignment_target = (  # type: ignore[method-assign]
        lambda *_args: PlanarPose(0.10, 0.0, 0.0)
    )
    executor._face_alignment_staging_target = (  # type: ignore[method-assign]
        lambda *_args, **_kwargs: PlanarPose(0.04, 0.0, 0.0)
    )
    goal = _goal(
        "proximal-quality-recovery",
        PrimitiveName.ALIGN_FACES,
        ("module_a", "module_b"),
        {
            "face_a": "TOP",
            "face_b": "BOTTOM",
            "contact_quality_planar_tolerance_m": 0.0015,
            "contact_quality_retry_count": 2,
        },
    )
    executor.submit(goal, 0.0)
    runtime = executor._active[goal.goal_id]
    runtime.alignment_approach_started = True

    retry = executor.step(0.1)

    assert retry.statuses[0].state is PrimitiveState.RUNNING
    assert retry.statuses[0].code == "REPOSITIONING_CONTACT"
    assert retry.commands["module_a"].linear_x_m_s != 0.0
    assert runtime.alignment_recovery_count == 1
    assert runtime.contact_quality_recovery_active


def test_planar_parking_error_ignores_uncorrectable_marker_height() -> None:
    """CAD vertical settling must not cause a pointless back/forward retry."""

    from smores_ep.isaac.primitive_executor import IsaacPrimitiveExecutor

    first = DockingFacePose(
        DockingFace("a", "TOP", "/a/top", "/a/body"),
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
    )
    second = DockingFacePose(
        DockingFace("b", "BOTTOM", "/b/bottom", "/b/body"),
        (0.0005, 0.001, 0.008),
        (-1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
    )

    error = IsaacPrimitiveExecutor._planar_face_lateral_offset(first, second)

    assert error == pytest.approx(0.001)


def test_executor_allows_locomotion_and_internal_motion_concurrently() -> None:
    from smores_ep.isaac.primitive_executor import IsaacPrimitiveExecutor

    executor = IsaacPrimitiveExecutor(
        stage=object(),
        module_roots={"module_a": "/A", "module_b": "/B"},
        states={
            "module_a": _FakeStateReader(),
            "module_b": _FakeStateReader(),
        },
        docking=_FakeDocking(),  # type: ignore[arg-type]
    )
    drive = _goal(
        "drive-a",
        PrimitiveName.DRIVE_TO_POSE,
        ("module_a",),
        {"x_m": 1.0, "y_m": 0.0, "yaw_rad": 0.0},
    )
    tilt = _goal(
        "tilt-a",
        PrimitiveName.SET_TILT,
        ("module_a",),
        {"angle_rad": 0.4},
    )
    pan = _goal(
        "pan-a",
        PrimitiveName.SET_PAN,
        ("module_a",),
        {"angle_rad": 0.4},
    )
    drive_b = _goal(
        "drive-b",
        PrimitiveName.DRIVE_TO_POSE,
        ("module_b",),
        {"x_m": 1.0, "y_m": 0.0, "yaw_rad": 0.0},
    )

    assert executor.submit(drive, 0.0).state is PrimitiveState.ACCEPTED
    assert executor.submit(tilt, 0.0).state is PrimitiveState.ACCEPTED
    assert executor.submit(pan, 0.0).code == "RESOURCE_BUSY"
    assert executor.submit(drive_b, 0.0).state is PrimitiveState.ACCEPTED


def test_executor_treats_spin_as_the_locomotion_resource() -> None:
    from smores_ep.isaac.primitive_executor import IsaacPrimitiveExecutor

    executor = IsaacPrimitiveExecutor(
        stage=object(),
        module_roots={"module_a": "/A", "module_b": "/B"},
        states={
            "module_a": _FakeStateReader(),
            "module_b": _FakeStateReader(),
        },
        docking=_FakeDocking(),  # type: ignore[arg-type]
    )
    first_motion = _goal(
        "spin-a",
        PrimitiveName.DRIVE_TO_POSE,
        ("module_a",),
        {"x_m": 0.0, "y_m": 0.0, "yaw_rad": math.pi},
    )
    second_motion = _goal(
        "forward-a",
        PrimitiveName.DRIVE_TO_POSE,
        ("module_a",),
        {"x_m": 1.0, "y_m": 0.0, "yaw_rad": 0.0},
    )

    assert executor.submit(first_motion, 0.0).state is PrimitiveState.ACCEPTED
    rejected = executor.submit(second_motion, 0.0)
    assert rejected.state is PrimitiveState.REJECTED
    assert rejected.code == "RESOURCE_BUSY"
    assert "locomotion:module_a" in rejected.message


def test_executor_allows_shared_stationary_target_for_parallel_alignment() -> None:
    from smores_ep.isaac.primitive_executor import IsaacPrimitiveExecutor

    module_ids = ("root", "mobile_a", "mobile_b")
    executor = IsaacPrimitiveExecutor(
        stage=object(),
        module_roots={
            module_id: f"/{module_id}"
            for module_id in module_ids
        },
        states={
            module_id: _FakeStateReader()
            for module_id in module_ids
        },
        docking=_FakeDockingThree(),  # type: ignore[arg-type]
    )
    align_a = _goal(
        "align-a",
        PrimitiveName.ALIGN_FACES,
        ("mobile_a", "root"),
        {"face_a": "BOTTOM", "face_b": "LEFT"},
    )
    align_b = _goal(
        "align-b",
        PrimitiveName.ALIGN_FACES,
        ("mobile_b", "root"),
        {"face_a": "BOTTOM", "face_b": "RIGHT"},
    )
    drive_root = _goal(
        "drive-root",
        PrimitiveName.DRIVE_TO_POSE,
        ("root",),
        {"x_m": 1.0, "y_m": 0.0, "yaw_rad": 0.0},
    )

    assert executor.submit(align_a, 0.0).state is PrimitiveState.ACCEPTED
    assert executor.submit(align_b, 0.0).state is PrimitiveState.ACCEPTED
    rejected = executor.submit(drive_root, 0.0)
    assert rejected.state is PrimitiveState.REJECTED
    assert rejected.code == "RESOURCE_BUSY"

    executor.cancel("align-a", 0.1)
    assert executor.submit(drive_root, 0.1).state is PrimitiveState.REJECTED
    executor.cancel("align-b", 0.2)
    assert executor.submit(drive_root, 0.2).state is PrimitiveState.ACCEPTED


def test_assisted_alignment_reserves_helper_locomotion() -> None:
    from smores_ep.isaac.primitive_executor import IsaacPrimitiveExecutor

    module_ids = ("root", "mobile_a", "mobile_b")
    executor = IsaacPrimitiveExecutor(
        stage=object(),
        module_roots={module_id: f"/{module_id}" for module_id in module_ids},
        states={module_id: _FakeStateReader() for module_id in module_ids},
        docking=_FakeDockingThree(),  # type: ignore[arg-type]
    )
    assisted = _goal(
        "assisted",
        PrimitiveName.ASSISTED_ALIGN_FACES,
        ("mobile_a", "root", "mobile_b"),
        {"face_a": "LEFT", "face_b": "RIGHT"},
    )
    drive_helper = _goal(
        "drive-helper",
        PrimitiveName.DRIVE_TO_POSE,
        ("mobile_b",),
        {"x_m": 1.0, "y_m": 0.0, "yaw_rad": 0.0},
    )
    assert executor.submit(assisted, 0.0).state is PrimitiveState.ACCEPTED
    rejected = executor.submit(drive_helper, 0.0)
    assert rejected.code == "RESOURCE_BUSY"


def test_rigid_driver_target_preserves_payload_relative_transform() -> None:
    from smores_ep.isaac.primitive_executor import IsaacPrimitiveExecutor

    target = IsaacPrimitiveExecutor._rigid_driver_target(
        payload_current=PlanarPose(1.0, 2.0, 0.0),
        driver_current=PlanarPose(0.5, 2.0, math.pi),
        payload_target=PlanarPose(3.0, 4.0, math.pi / 2.0),
    )
    assert target.x_m == pytest.approx(3.0)
    assert target.y_m == pytest.approx(3.5)
    assert target.yaw_rad == pytest.approx(-math.pi / 2.0)


def test_face_approach_only_recovers_from_material_axis_error() -> None:
    from smores_ep.isaac.primitive_executor import IsaacPrimitiveExecutor

    executor = IsaacPrimitiveExecutor(
        stage=object(),
        module_roots={"module_a": "/A", "module_b": "/B"},
        states={
            "module_a": _FakeStateReader(),
            "module_b": _FakeStateReader(),
        },
        docking=_FakeDocking(),  # type: ignore[arg-type]
    )
    observed_stall = SimpleNamespace(
        normal_separation_m=0.0117,
        lateral_offset_m=0.0078,
        normal_misalignment_rad=math.radians(6.5),
    )
    material_axis_error = SimpleNamespace(
        normal_separation_m=0.0117,
        lateral_offset_m=0.012,
        normal_misalignment_rad=math.radians(9.5),
    )
    excessive_cad_pitch = SimpleNamespace(
        normal_separation_m=0.0117,
        lateral_offset_m=0.008,
        normal_misalignment_rad=math.radians(9.0),
    )

    assert (
        executor._face_staging_pose_controller.position_tolerance_m
        == 0.002
    )
    assert (
        executor._face_staging_pose_controller.max_linear_speed_m_s
        == pytest.approx(0.055)
    )
    assert executor._face_staging_pose_controller.yaw_tolerance_rad == (
        pytest.approx(math.radians(1.5))
    )
    assert not executor._has_residual_contact_error(observed_stall)
    assert executor._has_residual_contact_error(excessive_cad_pitch)
    assert executor._has_residual_contact_error(material_axis_error)
    rejected_alignment = SimpleNamespace(
        eligible=False,
    )
    accepted_alignment = SimpleNamespace(
        eligible=True,
    )
    assert not executor._face_alignment_complete(rejected_alignment)
    assert executor._face_alignment_complete(accepted_alignment)
    approach = executor._straight_face_approach_step(
        PlanarPose(0.0, 0.0, 0.0),
        PlanarPose(0.010, 0.0, 0.0),
    )
    assert approach.linear_x_m_s >= 0.018
    reverse_approach = executor._straight_face_approach_step(
        PlanarPose(0.010, 0.003, 0.0),
        PlanarPose(0.0, 0.0, 0.0),
        feedback_enabled=True,
    )
    assert reverse_approach.linear_x_m_s <= -0.018
    assert reverse_approach.angular_z_rad_s > 0.0
    staging_correction = executor._staging_pose_step(
        PlanarPose(0.0, 0.0, 0.0),
        PlanarPose(0.003, 0.0, 0.0),
    )
    assert staging_correction.phase == "translate"
    assert staging_correction.linear_x_m_s >= 0.018
    run_7_runtime = SimpleNamespace(
        alignment_staging_position_reached=False,
    )
    yaw_step, subphase, ready = executor._alignment_staging_step(
        run_7_runtime,
        PlanarPose(-0.1598828, 0.0031368, 1.2329512),
        PlanarPose(-0.1591638, 0.0009635, -0.0040742),
    )
    assert subphase == "staging_yaw"
    assert not ready
    assert yaw_step.phase == "final_yaw"
    assert yaw_step.angular_z_rad_s < 0.0


def test_axial_face_pose_adjustment_translates_while_steering() -> None:
    """Regression for the wheel and centerline axial-face alignments."""

    from smores_ep.isaac.primitive_executor import IsaacPrimitiveExecutor

    executor = IsaacPrimitiveExecutor(
        stage=object(),
        module_roots={"module_a": "/A", "module_b": "/B"},
        states={
            "module_a": _FakeStateReader(),
            "module_b": _FakeStateReader(),
        },
        docking=_FakeDocking(),  # type: ignore[arg-type]
    )
    runtime = SimpleNamespace(alignment_staging_drive_direction=None)
    current = PlanarPose(-0.082071, -0.161760, -1.408149)
    target = PlanarPose(-0.089310, -0.152337, -1.566682)

    step, subphase, ready = executor._axial_face_staging_step(
        runtime,
        current,
        target,
    )

    assert subphase == "axial_pose_adjustment"
    assert not ready
    assert runtime.alignment_staging_drive_direction == -1.0
    assert step.linear_x_m_s == pytest.approx(-0.035)
    assert step.angular_z_rad_s > 0.0
    # In the target frame, the signed lateral velocity must oppose the
    # positive 7.2 mm error instead of rotating at a fixed position.
    relative_yaw = current.yaw_rad - target.yaw_rad
    assert step.linear_x_m_s * math.sin(relative_yaw) < 0.0

    # Ideal differential-drive integration reaches the lateral/yaw gate
    # without changing direction or entering a position/yaw limit cycle.
    for _ in range(200):
        step, _, ready = executor._axial_face_staging_step(
            runtime,
            current,
            target,
        )
        if ready:
            break
        dt_s = 0.01
        current = PlanarPose(
            current.x_m
            + step.linear_x_m_s * math.cos(current.yaw_rad) * dt_s,
            current.y_m
            + step.linear_x_m_s * math.sin(current.yaw_rad) * dt_s,
            current.yaw_rad + step.angular_z_rad_s * dt_s,
        )
    assert ready

    # The same controller is used by TOP/BOTTOM centerline actions, not only
    # by the four BOTTOM-to-lateral wheel actions.
    centerline_runtime = SimpleNamespace(
        alignment_staging_drive_direction=None
    )
    centerline_step, subphase, ready = executor._axial_face_staging_step(
        centerline_runtime,
        PlanarPose(0.010, 0.006, math.radians(10.0)),
        PlanarPose(0.0, 0.0, 0.0),
    )
    assert subphase == "axial_pose_adjustment"
    assert not ready
    assert centerline_step.linear_x_m_s < 0.0
    assert centerline_step.angular_z_rad_s > 0.0


def test_axial_alignment_hysteresis_preserves_curve_and_pivot_modes() -> None:
    """Regression for the 2.479 mm threshold chatter seen in run 11."""

    from smores_ep.isaac.primitive_executor import IsaacPrimitiveExecutor

    executor = IsaacPrimitiveExecutor(
        stage=object(),
        module_roots={"module_a": "/A", "module_b": "/B"},
        states={
            "module_a": _FakeStateReader(),
            "module_b": _FakeStateReader(),
        },
        docking=_FakeDocking(),  # type: ignore[arg-type]
    )
    runtime = SimpleNamespace(
        alignment_staging_drive_direction=-1.0,
        axial_alignment_mode="curve",
        axial_pivot_sample_started_s=None,
        axial_pivot_sample_yaw_rad=None,
        axial_escape_until_s=None,
        axial_stall_recovery_count=0,
    )
    target = PlanarPose(0.0741692337, -0.1523235013, -1.5548740369)
    measured = PlanarPose(0.0764791289, -0.1416706135, -1.3762423023)

    step, subphase, ready = executor._axial_face_staging_step(
        runtime,
        measured,
        target,
        now_s=0.0,
    )
    assert subphase == "axial_pose_adjustment"
    assert not ready
    assert runtime.axial_alignment_mode == "curve"
    assert step.phase == "curve"
    assert step.linear_x_m_s < 0.0

    def pose_with_errors(lateral_m: float, yaw_rad: float) -> PlanarPose:
        longitudinal_m = -0.010
        cosine = math.cos(target.yaw_rad)
        sine = math.sin(target.yaw_rad)
        return PlanarPose(
            target.x_m
            + cosine * longitudinal_m
            - sine * lateral_m,
            target.y_m
            + sine * longitudinal_m
            + cosine * lateral_m,
            target.yaw_rad + yaw_rad,
        )

    pivot_step, _, ready = executor._axial_face_staging_step(
        runtime,
        pose_with_errors(0.0019, math.radians(10.0)),
        target,
        now_s=0.1,
    )
    assert not ready
    assert runtime.axial_alignment_mode == "pivot"
    assert pivot_step.phase == "pivot"
    assert pivot_step.linear_x_m_s == 0.0
    wheel_rates = twist_to_wheel_rates(
        pivot_step.linear_x_m_s,
        pivot_step.angular_z_rad_s,
        executor._geometry.wheel_radius_m,
        executor._geometry.track_width_m,
    )
    assert wheel_rates.left_rad_s * wheel_rates.right_rad_s < 0.0

    # The pivot remains latched through ordinary CAD/contact drift and is
    # released only beyond the separate 4 mm threshold.
    retained_step, _, _ = executor._axial_face_staging_step(
        runtime,
        pose_with_errors(0.0039, math.radians(9.0)),
        target,
        now_s=0.2,
    )
    assert runtime.axial_alignment_mode == "pivot"
    assert retained_step.phase == "pivot"

    released_step, _, _ = executor._axial_face_staging_step(
        runtime,
        pose_with_errors(0.0041, math.radians(9.0)),
        target,
        now_s=0.3,
    )
    assert runtime.axial_alignment_mode == "curve"
    assert released_step.phase == "curve"


def test_stalled_axial_pivot_uses_bounded_unstick_arc() -> None:
    """A non-moving pivot must escape locally without restarting staging."""

    from smores_ep.isaac.primitive_executor import IsaacPrimitiveExecutor

    executor = IsaacPrimitiveExecutor(
        stage=object(),
        module_roots={"module_a": "/A", "module_b": "/B"},
        states={
            "module_a": _FakeStateReader(),
            "module_b": _FakeStateReader(),
        },
        docking=_FakeDocking(),  # type: ignore[arg-type]
    )
    current = PlanarPose(-0.010, 0.0015, math.radians(10.0))
    target = PlanarPose()
    runtime = SimpleNamespace(
        alignment_staging_drive_direction=-1.0,
        axial_alignment_mode="pivot",
        axial_pivot_sample_started_s=0.0,
        axial_pivot_sample_yaw_rad=current.yaw_rad,
        axial_escape_until_s=None,
        axial_stall_recovery_count=0,
    )

    step, subphase, ready = executor._axial_face_staging_step(
        runtime,
        current,
        target,
        now_s=0.81,
    )
    assert not ready
    assert subphase == "axial_pose_adjustment"
    assert step.phase == "unstick_arc"
    assert runtime.axial_alignment_mode == "escape"
    assert runtime.axial_stall_recovery_count == 1
    assert step.linear_x_m_s < 0.0
    assert abs(step.angular_z_rad_s) == pytest.approx(0.30)

    # The curvature cap guarantees forward/reverse rolling on both wheels;
    # this is not lateral motion and it does not remove differential pivoting.
    wheel_rates = twist_to_wheel_rates(
        step.linear_x_m_s,
        step.angular_z_rad_s,
        executor._geometry.wheel_radius_m,
        executor._geometry.track_width_m,
    )
    assert wheel_rates.left_rad_s * wheel_rates.right_rad_s > 0.0

    continuing, subphase, _ = executor._axial_face_staging_step(
        runtime,
        current,
        target,
        now_s=1.0,
    )
    assert subphase == "axial_pose_adjustment"
    assert continuing.phase == "unstick_arc"
    assert runtime.alignment_staging_drive_direction == -1.0


def test_stalled_curve_steering_turn_uses_same_unstick_arc() -> None:
    """Replay the immobile smores_03 steering turn from manipulator run 1."""

    from smores_ep.isaac.primitive_executor import IsaacPrimitiveExecutor

    executor = IsaacPrimitiveExecutor(
        stage=object(),
        module_roots={"module_a": "/A", "module_b": "/B"},
        states={
            "module_a": _FakeStateReader(),
            "module_b": _FakeStateReader(),
        },
        docking=_FakeDocking(),  # type: ignore[arg-type]
    )
    current = PlanarPose(
        0.0405052603,
        0.1536411554,
        2.6936884904,
    )
    target = PlanarPose(
        0.0622954682,
        0.1474125108,
        1.6428931706,
    )
    runtime = SimpleNamespace(
        alignment_staging_drive_direction=-1.0,
        axial_alignment_mode="curve",
        axial_pivot_sample_started_s=0.0,
        axial_pivot_sample_yaw_rad=current.yaw_rad,
        axial_escape_until_s=None,
        axial_stall_recovery_count=0,
    )

    step, subphase, ready = executor._axial_face_staging_step(
        runtime,
        current,
        target,
        now_s=0.81,
    )

    assert not ready
    assert subphase == "axial_pose_adjustment"
    assert step.phase == "unstick_arc"
    assert runtime.axial_alignment_mode == "escape"
    assert runtime.axial_stall_recovery_count == 1
    assert step.linear_x_m_s < 0.0
    assert abs(step.angular_z_rad_s) == pytest.approx(0.30)


def test_snake_axial_alignment_leaves_pivot_when_yaw_is_aligned() -> None:
    """Replay the two stalled first-wave Snake7 poses from episode 1."""

    from smores_ep.isaac.primitive_executor import IsaacPrimitiveExecutor

    executor = IsaacPrimitiveExecutor(
        stage=object(),
        module_roots={"module_a": "/A", "module_b": "/B"},
        states={
            "module_a": _FakeStateReader(),
            "module_b": _FakeStateReader(),
        },
        docking=_FakeDocking(),  # type: ignore[arg-type]
    )
    samples = (
        (
            PlanarPose(0.1686247079, 0.0023470097, -0.0064073675),
            PlanarPose(0.1418222896, -0.0003650013, -0.0044456768),
        ),
        (
            PlanarPose(-0.1281456937, 0.0044847429, -0.0041544481),
            PlanarPose(-0.1535248082, 0.0009426956, -0.0040830234),
        ),
    )

    for current, target in samples:
        runtime = SimpleNamespace(
            alignment_staging_drive_direction=-1.0,
            axial_alignment_mode="pivot",
            axial_pivot_sample_started_s=0.0,
            axial_pivot_sample_yaw_rad=current.yaw_rad,
            axial_escape_until_s=None,
            axial_stall_recovery_count=0,
        )
        step, subphase, ready = executor._axial_face_staging_step(
            runtime,
            current,
            target,
            now_s=0.2,
        )

        assert subphase == "axial_pose_adjustment"
        assert not ready
        assert runtime.axial_alignment_mode == "curve"
        assert step.phase == "curve"
        assert step.linear_x_m_s < 0.0
        # The residual is lateral, so a pure pivot cannot solve it.
        assert abs(step.angular_z_rad_s) > 0.0

        simulated = current
        for index in range(600):
            step, _, ready = executor._axial_face_staging_step(
                runtime,
                simulated,
                target,
                now_s=0.21 + index * 0.01,
            )
            if ready:
                break
            dt_s = 0.01
            simulated = PlanarPose(
                simulated.x_m
                + step.linear_x_m_s * math.cos(simulated.yaw_rad) * dt_s,
                simulated.y_m
                + step.linear_x_m_s * math.sin(simulated.yaw_rad) * dt_s,
                simulated.yaw_rad + step.angular_z_rad_s * dt_s,
            )
        assert ready


def test_executor_composes_teleop_wheels_with_primitive_tilt() -> None:
    from smores_ep.isaac.primitive_executor import IsaacPrimitiveExecutor

    executor = IsaacPrimitiveExecutor(
        stage=object(),
        module_roots={"module_a": "/A", "module_b": "/B"},
        states={
            "module_a": _FakeStateReader(),
            "module_b": _FakeStateReader(),
        },
        docking=_FakeDocking(),  # type: ignore[arg-type]
    )
    tilt = _goal(
        "tilt-a",
        PrimitiveName.SET_TILT,
        ("module_a",),
        {"angle_rad": 0.4},
    )
    assert executor.submit(tilt, 0.0).state is PrimitiveState.ACCEPTED
    composed = executor.compose_with_baseline(
        {
            "module_a": SmoresCommand(
                linear_x_m_s=0.08,
                angular_z_rad_s=0.5,
            )
        },
        {
            "module_a": SmoresCommand(
                tilt_target_rad=0.4,
                internal_motion=InternalMotionMode.TILT,
            )
        },
    )

    assert composed["module_a"].linear_x_m_s == pytest.approx(0.08)
    assert composed["module_a"].angular_z_rad_s == pytest.approx(0.5)
    assert composed["module_a"].tilt_target_rad == pytest.approx(0.4)
    assert (
        composed["module_a"].internal_motion
        is InternalMotionMode.TILT
    )


def test_tilt_primitive_can_retain_declared_structural_joint() -> None:
    from smores_ep.isaac.primitive_executor import IsaacPrimitiveExecutor

    executor = IsaacPrimitiveExecutor(
        stage=object(),
        module_roots={"module_a": "/A", "module_b": "/B"},
        states={
            "module_a": _FakeStateReader(),
            "module_b": _FakeStateReader(),
        },
        docking=_FakeDocking(),  # type: ignore[arg-type]
    )
    goal = _goal(
        "outside-in-fold",
        PrimitiveName.SET_TILT,
        ("module_a",),
        {
            "angle_rad": 0.5,
            "structural_hold_module_ids": ["module_b"],
        },
    )
    assert executor.submit(goal, 0.0).state is PrimitiveState.ACCEPTED

    step = executor.step(0.1)

    assert step.commands["module_a"].internal_motion is InternalMotionMode.TILT
    composed = executor.compose_with_baseline({}, step.commands)
    assert (
        composed["module_b"].internal_motion
        is InternalMotionMode.STRUCTURAL_HOLD
    )


def test_operational_pan_rate_limits_steering_and_holds_entire_structure() -> None:
    from smores_ep.isaac.primitive_executor import IsaacPrimitiveExecutor

    state_a = _MutableStateReader(tilt_rad=-1.25, pan_rad=0.0)
    state_b = _MutableStateReader(tilt_rad=0.35, pan_rad=0.2)
    executor = IsaacPrimitiveExecutor(
        stage=object(),
        module_roots={"module_a": "/A", "module_b": "/B"},
        states={"module_a": state_a, "module_b": state_b},
        docking=_FakeDocking(),  # type: ignore[arg-type]
    )
    goal = _goal(
        "slow-operational-steer",
        PrimitiveName.SET_PAN,
        ("module_a",),
        {
            "angle_rad": 1.0,
            "max_servo_error_rad": 0.05,
            "structural_hold_module_ids": ["module_a", "module_b"],
        },
    )
    assert executor.submit(goal, 0.0).state is PrimitiveState.ACCEPTED

    first = executor.step(0.1)
    assert first.commands["module_a"].pan_target_rad == pytest.approx(0.05)
    assert first.commands["module_a"].tilt_target_rad == pytest.approx(-1.25)
    composed = executor.compose_with_baseline({}, first.commands)
    assert (
        composed["module_b"].internal_motion
        is InternalMotionMode.STRUCTURAL_HOLD
    )
    assert composed["module_b"].pan_target_rad == pytest.approx(0.2)
    assert composed["module_b"].tilt_target_rad == pytest.approx(0.35)

    state_a.state.pan_joint_rad = 0.05
    second = executor.step(0.2)
    assert second.commands["module_a"].pan_target_rad == pytest.approx(0.10)


def test_tilt_servo_error_limit_softens_a_large_fold_target() -> None:
    from smores_ep.isaac.primitive_executor import IsaacPrimitiveExecutor

    state = _MutableStateReader(tilt_rad=0.0, pan_rad=0.2)
    executor = IsaacPrimitiveExecutor(
        stage=object(),
        module_roots={"module_a": "/A", "module_b": "/B"},
        states={
            "module_a": state,
            "module_b": _MutableStateReader(),
        },
        docking=_FakeDocking(),  # type: ignore[arg-type]
    )
    goal = _goal(
        "soft-fold",
        PrimitiveName.SET_TILT,
        ("module_a",),
        {
            "angle_rad": -math.radians(45.0),
            "max_servo_error_rad": 0.35,
        },
    )
    assert executor.submit(goal, 0.0).state is PrimitiveState.ACCEPTED

    first = executor.step(0.1)

    assert first.commands["module_a"].tilt_target_rad == pytest.approx(-0.35)
    assert first.commands["module_a"].pan_target_rad == pytest.approx(0.2)
    assert first.statuses[0].feedback["commanded_target_rad"] == pytest.approx(
        -0.35
    )


def test_coordinated_tilt_holds_fast_member_for_slower_support() -> None:
    """A fast fold corner must not mechanically strand its group peers."""

    from smores_ep.isaac.primitive_executor import IsaacPrimitiveExecutor

    class _CoordinatedFoldDocking:
        module_ids = ("fast", "slow")
        connections = ()

    states = {
        "fast": _MutableStateReader(tilt_rad=-0.50),
        "slow": _MutableStateReader(tilt_rad=-0.10),
    }
    executor = IsaacPrimitiveExecutor(
        stage=object(),
        module_roots={key: f"/{key}" for key in states},
        states=states,
        docking=_CoordinatedFoldDocking(),  # type: ignore[arg-type]
    )
    parameters = {
        "angle_rad": -math.radians(45.0),
        "max_servo_error_rad": 0.35,
        "max_coordination_lead_rad": math.radians(5.0),
        "coordination_group": "synchronized-fold",
        "coordination_size": 2,
    }
    fast = _goal(
        "fold-fast", PrimitiveName.SET_TILT, ("fast",), parameters
    )
    slow = _goal(
        "fold-slow", PrimitiveName.SET_TILT, ("slow",), parameters
    )
    assert executor.submit(fast, 0.0).state is PrimitiveState.ACCEPTED
    assert executor.submit(slow, 0.1).state is PrimitiveState.ACCEPTED

    step = executor.step(0.2)

    assert step.commands["fast"].tilt_target_rad == pytest.approx(-0.50)
    assert step.commands["slow"].tilt_target_rad == pytest.approx(-0.45)
    status_by_module = {
        status.module_ids[0]: status for status in step.statuses
    }
    assert status_by_module["fast"].feedback[
        "coordination_lead_limited"
    ] is True
    assert status_by_module["slow"].feedback[
        "coordination_lead_limited"
    ] is False


def test_tilt_primitive_drives_paired_pusher_with_held_internals() -> None:
    from smores_ep.isaac.primitive_executor import IsaacPrimitiveExecutor

    class _FoldDocking:
        module_ids = ("lifter", "spoke")

    executor = IsaacPrimitiveExecutor(
        stage=object(),
        module_roots={"lifter": "/Lifter", "spoke": "/Spoke"},
        states={
            "lifter": _FakeStateReader(),
            "spoke": _FakeStateReader(),
        },
        docking=_FoldDocking(),  # type: ignore[arg-type]
    )
    goal = _goal(
        "coupled-fold",
        PrimitiveName.SET_TILT,
        ("lifter",),
        {
            "angle_rad": 1.35,
            "pusher_module_id": "spoke",
            "pusher_linear_m_s": -0.025,
        },
    )
    assert executor.submit(goal, 0.0).state is PrimitiveState.ACCEPTED

    step = executor.step(0.1)

    assert step.commands["lifter"].internal_motion is InternalMotionMode.TILT
    assert step.commands["spoke"].linear_x_m_s == pytest.approx(-0.025)
    assert (
        step.commands["spoke"].internal_motion
        is InternalMotionMode.STRUCTURAL_HOLD
    )


def test_reached_fold_member_keeps_pushing_until_group_completion() -> None:
    from smores_ep.isaac.primitive_executor import IsaacPrimitiveExecutor

    class _FoldDocking:
        module_ids = ("lifter_a", "pusher_a", "lifter_b", "pusher_b")

    target = 1.35
    states = {
        "lifter_a": _MutableStateReader(tilt_rad=target),
        "pusher_a": _MutableStateReader(),
        "lifter_b": _MutableStateReader(),
        "pusher_b": _MutableStateReader(),
    }
    executor = IsaacPrimitiveExecutor(
        stage=object(),
        module_roots={key: f"/{key}" for key in states},
        states=states,
        docking=_FoldDocking(),  # type: ignore[arg-type]
    )
    common = {
        "angle_rad": target,
        "coordination_group": "fold",
        "coordination_size": 2,
        "hold_after_group_module_ids": [
            "lifter_a", "pusher_a", "lifter_b", "pusher_b"
        ],
    }
    first = _goal(
        "fold-a",
        PrimitiveName.SET_TILT,
        ("lifter_a",),
        {
            **common,
            "pusher_module_id": "pusher_a",
            "pusher_linear_m_s": 0.025,
        },
    )
    second = _goal(
        "fold-b",
        PrimitiveName.SET_TILT,
        ("lifter_b",),
        {
            **common,
            "pusher_module_id": "pusher_b",
            "pusher_linear_m_s": 0.025,
        },
    )
    assert executor.submit(first, 0.0).state is PrimitiveState.ACCEPTED
    assert executor.submit(second, 0.1).state is PrimitiveState.ACCEPTED

    step = executor.step(0.2)

    assert step.commands["pusher_a"].linear_x_m_s == pytest.approx(0.025)
    assert step.commands["pusher_b"].linear_x_m_s == pytest.approx(0.025)
    assert {status.code for status in step.statuses} == {
        "WAITING_JOINT_GROUP_COMPLETION",
        "MOVING_JOINT",
    }

    states["lifter_b"].state.tilt_joint_rad = -target
    completed = executor.step(0.3)
    assert {status.code for status in completed.statuses} == {
        "JOINT_TARGET_REACHED"
    }
    retained = executor.compose_with_baseline({}, {})
    assert set(retained) == set(states)
    assert all(
        command.internal_motion is InternalMotionMode.STRUCTURAL_HOLD
        for command in retained.values()
    )


def test_reached_tilt_is_retained_and_composes_with_wheel_drive() -> None:
    from smores_ep.isaac.primitive_executor import IsaacPrimitiveExecutor

    state_a = _MutableStateReader()
    executor = IsaacPrimitiveExecutor(
        stage=object(),
        module_roots={"module_a": "/A", "module_b": "/B"},
        states={
            "module_a": state_a,
            "module_b": _MutableStateReader(),
        },
        docking=_FakeDocking(),  # type: ignore[arg-type]
    )
    goal = _goal(
        "raise-and-hold",
        PrimitiveName.SET_TILT,
        ("module_a",),
        {"angle_rad": 0.6},
    )
    assert executor.submit(goal, 0.0).state is PrimitiveState.ACCEPTED
    assert executor.step(0.1).statuses[0].code == "MOVING_JOINT"
    state_a.state.tilt_joint_rad = -0.6
    assert executor.step(0.2).statuses[0].code == "JOINT_TARGET_REACHED"

    composed = executor.compose_with_baseline(
        {"module_a": SmoresCommand(linear_x_m_s=-0.03)},
        {},
    )
    command = composed["module_a"]
    assert command.linear_x_m_s == pytest.approx(-0.03)
    assert command.internal_motion is InternalMotionMode.STRUCTURAL_HOLD
    assert command.tilt_target_rad == pytest.approx(0.6)


def test_operational_drive_holds_connected_modules_but_not_reserves() -> None:
    from smores_ep.isaac.primitive_executor import IsaacPrimitiveExecutor

    class _ConnectedDocking:
        module_ids = ("drive", "frame", "reserve")
        connections = (
            SimpleNamespace(
                first_face=SimpleNamespace(module_id="drive"),
                second_face=SimpleNamespace(module_id="frame"),
            ),
        )

    executor = IsaacPrimitiveExecutor(
        stage=object(),
        module_roots={
            "drive": "/Drive",
            "frame": "/Frame",
            "reserve": "/Reserve",
        },
        states={
            "drive": _MutableStateReader(tilt_rad=0.1),
            "frame": _MutableStateReader(tilt_rad=-0.7, pan_rad=0.3),
            "reserve": _MutableStateReader(tilt_rad=0.9),
        },
        docking=_ConnectedDocking(),  # type: ignore[arg-type]
    )

    composed = executor.compose_with_baseline(
        {"drive": SmoresCommand(linear_x_m_s=0.02)},
        {},
    )

    assert set(composed) == {"drive", "frame"}
    assert all(
        command.internal_motion is InternalMotionMode.STRUCTURAL_HOLD
        for command in composed.values()
    )
    assert composed["frame"].pan_target_rad == pytest.approx(0.3)
    assert composed["frame"].tilt_target_rad == pytest.approx(-0.7)


def test_fold_preserves_an_earlier_retained_target() -> None:
    from smores_ep.isaac.primitive_executor import IsaacPrimitiveExecutor

    state_b = _MutableStateReader()
    executor = IsaacPrimitiveExecutor(
        stage=object(),
        module_roots={"module_a": "/A", "module_b": "/B"},
        states={
            "module_a": _MutableStateReader(),
            "module_b": state_b,
        },
        docking=_FakeDocking(),  # type: ignore[arg-type]
    )
    retained = _goal(
        "old-support",
        PrimitiveName.SET_TILT,
        ("module_b",),
        {"angle_rad": 0.3},
    )
    assert executor.submit(retained, 0.0).state is PrimitiveState.ACCEPTED
    executor.step(0.1)
    state_b.state.tilt_joint_rad = -0.3
    executor.step(0.2)
    assert "module_b" in executor.compose_with_baseline({}, {})

    fold = _goal(
        "outer-fold",
        PrimitiveName.SET_TILT,
        ("module_a",),
        {
            "angle_rad": 0.5,
        },
    )
    assert executor.submit(fold, 0.3).state is PrimitiveState.ACCEPTED
    step = executor.step(0.4)
    composed = executor.compose_with_baseline({}, step.commands)
    assert (
        composed["module_b"].internal_motion
        is InternalMotionMode.TILT
    )
    executor.cancel(fold.goal_id, 0.5)
    assert "module_b" in executor.compose_with_baseline({}, {})


def test_tilt_uses_loaded_joint_completion_tolerance() -> None:
    from smores_ep.isaac.primitive_executor import IsaacPrimitiveExecutor

    states = {
        "module_a": _MutableStateReader(tilt_rad=-math.radians(52.5)),
        "module_b": _MutableStateReader(),
        "module_c": _MutableStateReader(),
        "module_d": _MutableStateReader(),
    }
    executor = IsaacPrimitiveExecutor(
        stage=object(),
        module_roots={key: f"/{key}" for key in states},
        states=states,  # type: ignore[arg-type]
        docking=_FakeDockingFour(),  # type: ignore[arg-type]
    )
    goal = _goal(
        "loaded-tilt",
        PrimitiveName.SET_TILT,
        ("module_a",),
        {"angle_rad": -math.radians(55.0)},
    )
    assert executor.submit(goal, 0.0).state is PrimitiveState.ACCEPTED

    result = executor.step(0.1)
    assert result.statuses[0].code == "JOINT_TARGET_REACHED"


def test_tilt_goal_can_override_completion_tolerance() -> None:
    from smores_ep.isaac.primitive_executor import IsaacPrimitiveExecutor

    states = {
        "module_a": _MutableStateReader(tilt_rad=-math.radians(51.5)),
        "module_b": _MutableStateReader(),
    }
    executor = IsaacPrimitiveExecutor(
        stage=object(),
        module_roots={key: f"/{key}" for key in states},
        states=states,  # type: ignore[arg-type]
        docking=_FakeDocking(),  # type: ignore[arg-type]
    )
    goal = _goal(
        "relaxed-posture-tilt",
        PrimitiveName.SET_TILT,
        ("module_a",),
        {
            "angle_rad": -math.radians(55.0),
            "tolerance_rad": math.radians(4.0),
        },
    )
    assert executor.submit(goal, 0.0).state is PrimitiveState.ACCEPTED

    result = executor.step(0.1)
    assert result.statuses[0].code == "JOINT_TARGET_REACHED"


def test_coordinated_tilt_waits_then_releases_final_targets_together() -> None:
    from smores_ep.isaac.primitive_executor import IsaacPrimitiveExecutor

    states = {
        "module_a": _MutableStateReader(),
        "module_b": _MutableStateReader(),
        "module_c": _MutableStateReader(),
        "module_d": _MutableStateReader(),
    }
    executor = IsaacPrimitiveExecutor(
        stage=object(),
        module_roots={key: f"/{key}" for key in states},
        states=states,  # type: ignore[arg-type]
        docking=_FakeDockingFour(),  # type: ignore[arg-type]
    )
    parameters = {
        "angle_rad": -math.radians(55.0),
        "coordination_group": "race-car-posture",
        "coordination_size": 2,
    }
    first = _goal(
        "posture-a", PrimitiveName.SET_TILT, ("module_a",), parameters
    )
    second = _goal(
        "posture-b", PrimitiveName.SET_TILT, ("module_b",), parameters
    )
    assert executor.submit(first, 0.0).state is PrimitiveState.ACCEPTED

    waiting = executor.step(0.1)
    assert waiting.statuses[0].code == "WAITING_JOINT_GROUP"
    assert waiting.commands["module_a"].tilt_target_rad == pytest.approx(0.0)

    assert executor.submit(second, 0.2).state is PrimitiveState.ACCEPTED
    moving = executor.step(0.3)
    assert {status.code for status in moving.statuses} == {"MOVING_JOINT"}
    for module_id in ("module_a", "module_b"):
        command_target = moving.commands[module_id].tilt_target_rad
        assert command_target == pytest.approx(-math.radians(55.0))


def test_coordinated_tilt_at_target_waits_for_the_complete_group() -> None:
    """An early in-tolerance member must not break the group barrier."""

    from smores_ep.isaac.primitive_executor import IsaacPrimitiveExecutor

    target = -math.radians(55.0)
    states = {
        "module_a": _MutableStateReader(tilt_rad=target),
        "module_b": _MutableStateReader(),
    }
    executor = IsaacPrimitiveExecutor(
        stage=object(),
        module_roots={key: f"/{key}" for key in states},
        states=states,  # type: ignore[arg-type]
        docking=_FakeDocking(),  # type: ignore[arg-type]
    )
    parameters = {
        "angle_rad": target,
        "coordination_group": "race-car-posture-at-target",
        "coordination_size": 2,
    }
    first = _goal(
        "posture-at-target",
        PrimitiveName.SET_TILT,
        ("module_a",),
        parameters,
    )
    second = _goal(
        "posture-moving",
        PrimitiveName.SET_TILT,
        ("module_b",),
        parameters,
    )

    assert executor.submit(first, 0.0).state is PrimitiveState.ACCEPTED
    waiting = executor.step(0.1)
    assert waiting.statuses[0].code == "WAITING_JOINT_GROUP"
    assert executor.active_goals == (first,)

    assert executor.submit(second, 0.2).state is PrimitiveState.ACCEPTED
    released = executor.step(0.3)
    assert {status.code for status in released.statuses} == {
        "WAITING_JOINT_GROUP_COMPLETION",
        "MOVING_JOINT",
    }
    assert released.commands["module_a"].tilt_target_rad == pytest.approx(
        target
    )
    assert released.commands["module_b"].tilt_target_rad == pytest.approx(
        target
    )

    states["module_b"].state.tilt_joint_rad = -target
    completed = executor.step(0.4)
    assert {status.code for status in completed.statuses} == {
        "JOINT_TARGET_REACHED"
    }
    assert not executor.active_goals
