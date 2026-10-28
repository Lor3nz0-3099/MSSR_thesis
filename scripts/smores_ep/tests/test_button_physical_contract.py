"""USD authoring and measured-state contracts; no simulator is launched."""

from __future__ import annotations

import math

import pytest

from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade

from smores_ep.isaac.obstacle_course import (
    ButtonTargetSpec,
    CourseBox,
    _install_button_prismatic_joint,
    _install_course_boxes,
    install_mobile_manipulator_button_test_course,
)
from smores_ep.isaac.phase1_manipulation_curriculum import (
    install_phase1_manipulation_curriculum_course,
)
from smores_ep.isaac.state_graph_publisher import SmoresStateGraphPublisher


C1_ROOT = "/World/Phase1ManipulationCurriculum"
C1_PLUNGER = f"{C1_ROOT}/Phase1ManipulationButtonPlunger"
C1_JOINT = f"{C1_ROOT}/ButtonPrismaticJoint"


def _stage():
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdShade.Material.Define(stage, "/World/materials/dynamic_ground")
    return stage


def _assert_joint_contract(stage, plunger_path, joint_path, center, normal):
    plunger = stage.GetPrimAtPath(plunger_path)
    assert plunger.HasAPI(UsdPhysics.RigidBodyAPI)
    assert plunger.HasAPI(UsdPhysics.CollisionAPI)
    assert UsdPhysics.RigidBodyAPI(plunger).GetRigidBodyEnabledAttr().Get()
    joint = UsdPhysics.PrismaticJoint(stage.GetPrimAtPath(joint_path))
    assert joint, "a movable plunger needs a real prismatic joint"
    assert joint.GetBody0Rel().GetTargets() == []
    assert [str(p) for p in joint.GetBody1Rel().GetTargets()] == [plunger_path]
    lo, hi = joint.GetLowerLimitAttr().Get(), joint.GetUpperLimitAttr().Get()
    assert hi - lo == pytest.approx(0.004)
    assert lo <= 0.0 <= hi

    # Both joint frames must describe the same world line at rest, even
    # though the cube has its own yaw and nonuniform dimensions.
    canonical = {"X": Gf.Vec3d(1, 0, 0), "Y": Gf.Vec3d(0, 1, 0)}[
        joint.GetAxisAttr().Get()
    ]
    frame0 = Gf.Rotation(Gf.Quatd(joint.GetLocalRot0Attr().Get()))
    frame1 = Gf.Rotation(Gf.Quatd(joint.GetLocalRot1Attr().Get()))
    body_world = UsdGeom.Xformable(plunger).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    axis0 = frame0.TransformDir(canonical)
    body_rotation = body_world.RemoveScaleShear().ExtractRotation()
    axis1 = body_rotation.TransformDir(frame1.TransformDir(canonical))
    assert tuple(axis1) == pytest.approx(tuple(axis0), abs=1e-6)
    sign = 1.0 if hi > 0.0 else -1.0
    assert tuple(sign * axis0) == pytest.approx((*normal, 0.0), abs=1e-6)
    assert tuple(joint.GetLocalPos0Attr().Get()) == pytest.approx(center, abs=2e-7)
    assert tuple(body_world.Transform(Gf.Vec3d(joint.GetLocalPos1Attr().Get()))) == (
        pytest.approx(center, abs=2e-7)
    )


@pytest.mark.parametrize("seed,split", [(20000, "train"), (21000, "validation"),
                                        (22000, "test"), (22521, "challenge")])
def test_c1_installs_a_contact_driven_four_mm_plunger(seed, split):
    stage = _stage()
    course = install_phase1_manipulation_curriculum_course(stage, seed=seed, split=split)
    _assert_joint_contract(stage, C1_PLUNGER, C1_JOINT,
                           course.button_center_xyz_m, course.press_direction_world_xy)
    button = course.to_observation()["button"]
    assert button["plunger_path"] == C1_PLUNGER
    assert button["plunger_stroke_m"] == pytest.approx(0.004)
    # Installation must preserve the already validated oriented geometry.
    for box in (course.button_wall, course.button_plunger):
        prim = stage.GetPrimAtPath(f"{C1_ROOT}/{box.name}")
        xform = UsdGeom.Xformable(prim)
        assert tuple(xform.GetOrderedXformOps()[0].Get()) == pytest.approx(box.center_xyz_m)
        assert tuple(xform.GetOrderedXformOps()[-1].Get()) == pytest.approx(box.size_xyz_m)


@pytest.mark.parametrize("yaw_deg", [0.0, 90.0, 180.0, 270.0, 13.0, -33.0, 137.0, 225.0])
def test_shared_joint_tracks_the_exact_normal_in_both_frames(yaw_deg):
    stage = _stage()
    normal = (math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg)))
    center = (1.1, -0.4, 0.17)
    # Independent body yaw exercises the frame conversion rather than only
    # the convenient case where joint and plunger have the same rotation.
    box = CourseBox("Plunger", center, (0.04, 0.08, 0.08), (1.0, 0.0, 0.0),
                    yaw_deg=yaw_deg + 17.0)
    _install_course_boxes(stage, "/World/Button", (box,))
    _install_button_prismatic_joint(
        stage, root_path="/World/Button", plunger_name="Plunger",
        center_xyz_m=center, press_direction_world_xy=normal,
        plunger_yaw_deg=box.yaw_deg, joint_name="Joint",
    )
    _assert_joint_contract(stage, "/World/Button/Plunger", "/World/Button/Joint", center, normal)


@pytest.mark.parametrize("normal", [(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)])
def test_legacy_cardinal_button_retains_physical_press_contract(normal):
    stage = _stage()
    course = install_mobile_manipulator_button_test_course(
        stage, ButtonTargetSpec(press_direction_xy=normal)
    )
    root = "/World/MobileManipulatorButtonTestCourse"
    _assert_joint_contract(stage, f"{root}/ButtonPlunger", f"{root}/ButtonPrismaticJoint",
                           course.button_center_xyz_m, normal)


@pytest.mark.parametrize("profile", ["c1", "legacy"])
def test_runtime_depression_projects_actual_usd_motion(profile, tmp_path):
    stage = _stage()
    if profile == "c1":
        course = install_phase1_manipulation_curriculum_course(stage, seed=20000, split="train")
        path = C1_PLUNGER
    else:
        course = install_mobile_manipulator_button_test_course(stage)
        path = "/World/MobileManipulatorButtonTestCourse/ButtonPlunger"
    publisher = SmoresStateGraphPublisher(stage, {}, {}, None, output_dir=tmp_path,
                                          course_observation=course.to_observation())
    assert publisher._current_course_observation()["button"]["depression_m"] == pytest.approx(0.0)
    nx, ny = course.press_direction_world_xy
    x, y, z = course.button_center_xyz_m
    # Tangential and vertical movement must not masquerade as depression.
    position = (x + 0.0037 * nx - 0.002 * ny, y + 0.0037 * ny + 0.002 * nx, z + 0.001)
    translate = UsdGeom.Xformable(stage.GetPrimAtPath(path)).GetOrderedXformOps()[0]
    translate.Set(Gf.Vec3d(*position))
    measured = publisher._current_course_observation()["button"]
    assert measured["current_center_xyz_m"] == pytest.approx(position)
    assert measured["depression_m"] == pytest.approx(0.0037, abs=1e-9)
    # Reverse motion must remain signed; no success latch or clipping.
    translate.Set(Gf.Vec3d(x - 0.001 * nx, y - 0.001 * ny, z))
    assert publisher._current_course_observation()["button"]["depression_m"] == pytest.approx(-0.001)
