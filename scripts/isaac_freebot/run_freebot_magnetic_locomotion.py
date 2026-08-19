from __future__ import annotations

import argparse
import os
import re
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

from isaacsim import SimulationApp


parser = argparse.ArgumentParser()
parser.add_argument(
    "--usd",
    default="assets/freebot/usd_physics/freebot_cad_full_longer_casters.usd",
    help="FreeBOT physics USD stage to open.",
)
parser.add_argument("--headless", action="store_true")
parser.add_argument("--steps", type=int, default=12000)
parser.add_argument("--wheel-axis", choices=["X", "Y", "Z"], default="Y", help="Revolute joint axis for both wheels.")
parser.add_argument("--left-wheel-velocity", type=float, default=240.0, help="Left wheel target velocity [deg/s].")
parser.add_argument("--right-wheel-velocity", type=float, default=240.0, help="Right wheel target velocity [deg/s].")
parser.add_argument("--wheel-damping", type=float, default=500.0, help="Wheel velocity drive damping.")
parser.add_argument("--wheel-max-force", type=float, default=1.2, help="Wheel velocity drive max force/torque.")
parser.add_argument(
    "--wheel-shell-preload",
    type=float,
    default=0.0,
    help="Internal preload force that pushes each drive wheel against the inner shell [N].",
)
parser.add_argument("--inner-radius", type=float, default=0.0605)
parser.add_argument("--log-interval", type=int, default=240, help="Telemetry print interval in simulation steps.")
parser.add_argument(
    "--magnetic-law",
    choices=["dipole", "exponential"],
    default="dipole",
    help="Magnetic force law. 'dipole' is the current physically motivated default; 'exponential' keeps the previous paper-fit model.",
)
parser.add_argument("--magnet-br", type=float, default=1.47, help="Magnet remanence Br [T]. FreeBOT paper reports 14700 gauss.")
parser.add_argument("--magnet-size-x", type=float, default=0.020, help="Magnet dimension along local X [m].")
parser.add_argument("--magnet-size-y", type=float, default=0.020, help="Magnet dimension along local Y [m].")
parser.add_argument("--magnet-size-z", type=float, default=0.010, help="Magnet dimension along local Z / magnetization thickness [m].")
parser.add_argument(
    "--dipole-min-distance",
    type=float,
    default=0.010,
    help="Minimum effective dipole-surface distance used to regularize the ideal dipole law [m].",
)
parser.add_argument(
    "--dipole-force-coeff",
    type=float,
    default=0.029841551829730376,
    help="Coefficient for ideal dipole-plane force. Default is 3/(32*pi), then force is saturated by Fmax.",
)
parser.add_argument("--magnet-gap", type=float, default=0.0015, help="Preferred magnet/shell air gap [m].")
parser.add_argument("--magnet-preload", type=float, default=1.8, help="Legacy parameter kept for old command lines; not used by the exponential law.")
parser.add_argument("--magnet-k", type=float, default=260.0, help="Legacy parameter kept for old command lines; not used by the exponential law.")
parser.add_argument("--magnet-d", type=float, default=0.45, help="Normal magnetic damping gain [N/(m/s)].")
parser.add_argument("--max-force", type=float, default=16.0, help="Internal magnet/shell force clamp [N].")
parser.add_argument(
    "--magnet-decay-distance",
    type=float,
    default=0.012,
    help="Exponential decay length for the internal magnet/shell force law [m].",
)
parser.add_argument(
    "--shell-reaction-scale",
    type=float,
    default=1.0,
    help="Scale for the equal/opposite force on the shell. Use 1.0 for physics, 0.0 only for debugging.",
)
parser.add_argument("--wall-test", action="store_true", help="Add a ferromagnetic vertical wall and log climbing telemetry.")
parser.add_argument(
    "--ramp-test",
    action="store_true",
    help="Add a ferromagnetic ramp whose slope can increase during the simulation.",
)
parser.add_argument("--shell-radius", type=float, default=0.0665, help="Outer shell radius used by the wall adhesion model [m].")
parser.add_argument("--wall-x", type=float, default=0.45, help="X position of the wall center [m].")
parser.add_argument("--wall-width", type=float, default=2.0, help="Wall size along Y [m].")
parser.add_argument("--wall-height", type=float, default=1.2, help="Wall size along Z [m].")
parser.add_argument("--wall-thickness", type=float, default=0.04, help="Wall thickness along X [m].")
parser.add_argument("--ramp-start-x", type=float, default=-0.15, help="Ramp lower edge X coordinate [m].")
parser.add_argument("--ramp-start-z", type=float, default=-0.010, help="Ramp top-surface lower edge Z coordinate [m].")
parser.add_argument("--ramp-length", type=float, default=1.2, help="Ramp length measured along its surface [m].")
parser.add_argument("--ramp-width", type=float, default=1.2, help="Ramp width along Y [m].")
parser.add_argument("--ramp-thickness", type=float, default=0.04, help="Ramp thickness [m].")
parser.add_argument("--ramp-angle-start", type=float, default=0.0, help="Initial ramp angle [deg].")
parser.add_argument("--ramp-angle-end", type=float, default=55.0, help="Maximum ramp angle [deg].")
parser.add_argument("--ramp-angle-step", type=float, default=10.0, help="Ramp angle increment [deg].")
parser.add_argument("--ramp-step-interval", type=float, default=3.0, help="Time between ramp angle increments [s].")
parser.add_argument("--ramp-stair", action="store_true", help="Create a long segmented ramp with increasing fixed slopes.")
parser.add_argument("--ramp-segment-length", type=float, default=0.45, help="Length of each segmented ramp face [m].")
parser.add_argument("--ramp-segment-gap", type=float, default=0.008, help="Small gap between ramp segment centers along the path [m].")
parser.add_argument("--ramp-steering-control", action="store_true", help="Apply a simple differential wheel correction to stay centered on the ramp.")
parser.add_argument("--ramp-center-y", type=float, default=0.060, help="Target Y coordinate for ramp steering [m].")
parser.add_argument("--ramp-steering-gain", type=float, default=900.0, help="Wheel velocity correction gain [deg/s per m of lateral error].")
parser.add_argument("--ramp-steering-max-delta", type=float, default=140.0, help="Maximum steering correction added/subtracted from wheel velocity [deg/s].")
parser.add_argument("--ramp-steering-sign", type=float, default=1.0, help="Use -1 if the steering correction makes lateral drift worse.")
parser.add_argument("--wall-capture-gap", type=float, default=0.018, help="Maximum shell-wall magnetic capture gap [m].")
parser.add_argument("--wall-preload", type=float, default=3.0, help="Legacy parameter kept for old command lines; not used by the exponential law.")
parser.add_argument("--wall-k", type=float, default=450.0, help="Legacy parameter kept for old command lines; not used by the exponential law.")
parser.add_argument("--wall-d", type=float, default=0.8, help="External wall magnetic normal damping [N/(m/s)].")
parser.add_argument(
    "--ferro-static-friction",
    type=float,
    default=2.0,
    help="Static friction coefficient for generated ferromagnetic wall/ramp surfaces.",
)
parser.add_argument(
    "--ferro-dynamic-friction",
    type=float,
    default=1.6,
    help="Dynamic friction coefficient for generated ferromagnetic wall/ramp surfaces.",
)
parser.add_argument(
    "--wall-max-force",
    type=float,
    default=22.6,
    help="Clamp for external module-module/ferromagnetic surface attraction [N]. FreeBOT paper reports 22.6 N.",
)
parser.add_argument(
    "--wall-decay-distance",
    type=float,
    default=0.012,
    help="Exponential decay length for the external ferromagnetic attraction force law [m].",
)
parser.add_argument(
    "--start-on-wall",
    action="store_true",
    help="Initialize the robot already attached to the vertical wall for pure wall-climbing tests.",
)
parser.add_argument("--start-wall-gap", type=float, default=0.002, help="Initial shell-wall clearance for --start-on-wall [m].")
parser.add_argument("--start-wall-height", type=float, default=0.35, help="Initial shell center height for --start-on-wall [m].")
parser.add_argument(
    "--start-internal-rotation-deg",
    type=float,
    default=-90.0,
    help="Initial internal mechanism rotation about world Y for --start-on-wall [deg].",
)
parser.add_argument(
    "--wall-shell-normal-scale",
    type=float,
    default=1.0,
    help=(
        "Scale for the external magnetic force applied to the magnetized shell point. "
        "Use 0.0 to reproduce the older internal-only external attraction model."
    ),
)
parser.add_argument(
    "--wall-force-rate",
    type=float,
    default=35.0,
    help="Maximum wall magnetic force variation [N/s]. This avoids an impulsive attachment.",
)
parser.add_argument(
    "--wall-alignment-power",
    type=float,
    default=2.0,
    help="Exponent that gates wall adhesion by magnet alignment toward the wall normal.",
)
parser.add_argument(
    "--external-model",
    choices=["field-patch", "point"],
    default="point",
    help="External ferromagnetic surface model for ramp/wall. field-patch samples the surface magnetic field; point keeps the older equivalent-force model.",
)
parser.add_argument("--field-patch-radius", type=float, default=0.080, help="Radius of the sampled magnetic patch on ramp/wall [m].")
parser.add_argument("--field-rings", type=int, default=4, help="Number of rings used by the ramp/wall field patch.")
parser.add_argument("--field-ring-samples", type=int, default=12, help="Number of angular samples per non-central field-patch ring.")
parser.add_argument("--field-min-distance", type=float, default=0.010, help="Minimum magnet-to-surface distance for field calculation [m].")
parser.add_argument("--field-pressure-scale", type=float, default=1.0, help="Scale applied to Maxwell pressure B^2/(2*mu0).")
parser.add_argument(
    "--climb-after-capture",
    action="store_true",
    help="Switch wheel drive to stronger climbing parameters after wall magnetic capture.",
)
parser.add_argument("--climb-capture-threshold", type=float, default=0.65, help="Wall capture value that starts climb mode.")
parser.add_argument("--climb-settle-time", type=float, default=0.4, help="Delay after capture before applying climb drive [s].")
parser.add_argument("--climb-wheel-velocity", type=float, default=720.0, help="Wheel target velocity used in climb mode [deg/s].")
parser.add_argument("--climb-wheel-max-force", type=float, default=4.0, help="Wheel drive max force/torque used in climb mode.")
parser.add_argument("--climb-wheel-damping", type=float, default=900.0, help="Wheel velocity drive damping used in climb mode.")
parser.add_argument(
    "--ros2-teleop",
    action="store_true",
    help="Drive the wheels from a ROS 2 geometry_msgs/Twist topic instead of fixed wheel velocities.",
)
parser.add_argument("--cmd-vel-topic", default="/cmd_vel", help="ROS 2 Twist topic used by --ros2-teleop.")
parser.add_argument(
    "--cmd-linear-scale",
    type=float,
    default=900.0,
    help="Wheel target velocity generated by linear.x=1.0 [deg/s].",
)
parser.add_argument(
    "--cmd-angular-scale",
    type=float,
    default=360.0,
    help="Differential wheel target velocity generated by angular.z=1.0 [deg/s].",
)
parser.add_argument("--cmd-timeout", type=float, default=1.0, help="Stop wheels if no ROS 2 command arrives for this many seconds.")
parser.add_argument("--cmd-linear-sign", type=float, default=1.0, help="Use -1 if teleop forward moves the robot backward.")
parser.add_argument("--cmd-angular-sign", type=float, default=1.0, help="Use -1 if teleop yaw turns in the opposite direction.")
args, _ = parser.parse_known_args()

simulation_app = SimulationApp({"headless": args.headless})

import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.stage as stage_utils
from isaacsim.core.experimental.prims import RigidPrim, XformPrim
from isaacsim.core.rendering_manager import RenderingManager
from isaacsim.core.simulation_manager import SimulationManager
from pxr import Gf, UsdGeom, UsdPhysics, UsdShade


SHELL = "/World/freebot/shell_link"
INTERNAL = "/World/freebot/internal_link"
MAGNET_FRAME = "/World/freebot/internal_link/magnet_frame"
LEFT_WHEEL = "/World/freebot/left_wheel_link"
RIGHT_WHEEL = "/World/freebot/right_wheel_link"
LEFT_WHEEL_JOINT = "/World/freebot/joints/left_wheel_joint"
RIGHT_WHEEL_JOINT = "/World/freebot/joints/right_wheel_joint"
WALL_PATH = "/World/ferromagnetic_wall"
RAMP_PATH = "/World/ferromagnetic_ramp"
RAMP_STAIR_ROOT = "/World/ferromagnetic_ramp_stair"

INITIAL_CENTERS = {
    SHELL: np.array([0.023103, 0.059745, 0.055943], dtype=np.float64),
    INTERNAL: np.array([0.023065, 0.060436, 0.026953], dtype=np.float64),
    LEFT_WHEEL: np.array([0.023042, 0.094171, 0.032054], dtype=np.float64),
    RIGHT_WHEEL: np.array([0.023043, 0.026872, 0.031846], dtype=np.float64),
    "/World/freebot/left_wheel_mount_link": np.array([0.023042, 0.094171, 0.032054], dtype=np.float64),
    "/World/freebot/right_wheel_mount_link": np.array([0.023043, 0.026872, 0.031846], dtype=np.float64),
    "/World/freebot/caster_1_ball_link": np.array([-0.031633, 0.060363, 0.054020], dtype=np.float64),
    "/World/freebot/caster_2_ball_link": np.array([0.077764, 0.060363, 0.054020], dtype=np.float64),
}


def to_np(array_like) -> np.ndarray:
    if hasattr(array_like, "numpy"):
        return array_like.numpy()
    return np.asarray(array_like)


def first_vec(array_like) -> np.ndarray:
    return np.asarray(to_np(array_like)[0], dtype=np.float64)


def point_velocity(linear_velocity, angular_velocity, body_center, world_point):
    return linear_velocity + np.cross(angular_velocity, world_point - body_center)


def clamp_vector(vector, max_norm):
    norm = np.linalg.norm(vector)
    if norm <= max_norm or norm < 1.0e-9:
        return vector
    return vector * (max_norm / norm)


def smoothstep(value):
    value = np.clip(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def rotation_matrix_y(angle_rad):
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return np.array(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=np.float64,
    )


def quatf_from_y_rotation(angle_rad):
    half = 0.5 * angle_rad
    return Gf.Quatf(float(np.cos(half)), Gf.Vec3f(0.0, float(np.sin(half)), 0.0))


def quat_np_from_y_rotation(angle_rad):
    half = 0.5 * angle_rad
    return np.array([np.cos(half), 0.0, np.sin(half), 0.0], dtype=np.float32)


def set_xform_pose(stage, path, position, orientation=None):
    prim = stage.GetPrimAtPath(path)
    if not prim:
        raise RuntimeError(f"Missing prim for initial pose: {path}")

    xformable = UsdGeom.Xformable(prim)
    translate_op = None
    orient_op = None
    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            translate_op = op
        elif op.GetOpType() == UsdGeom.XformOp.TypeOrient:
            orient_op = op

    if translate_op is None:
        translate_op = xformable.AddTranslateOp()
    translate_op.Set(Gf.Vec3d(float(position[0]), float(position[1]), float(position[2])))

    if orientation is not None:
        if orient_op is None:
            orient_op = xformable.AddOrientOp()
        orient_op.Set(orientation)


def initialize_on_wall(stage):
    poses, orientation = compute_on_wall_initial_poses()
    usd_orientation = quatf_from_y_rotation(np.deg2rad(args.start_internal_rotation_deg))

    set_xform_pose(stage, SHELL, poses[SHELL])
    for path, target in poses.items():
        if path == SHELL:
            continue
        set_xform_pose(stage, path, target, usd_orientation)

    shell_target = poses[SHELL]
    print(
        "Initial wall-climb USD pose: "
        f"shell=({shell_target[0]:+.3f}, {shell_target[1]:+.3f}, {shell_target[2]:+.3f}) m, "
        f"wall_gap={args.start_wall_gap} m, "
        f"internal_rotation_y={args.start_internal_rotation_deg} deg"
    )


def compute_on_wall_initial_poses():
    wall_surface_x = args.wall_x - 0.5 * args.wall_thickness
    shell_target = np.array(
        [
            wall_surface_x - args.shell_radius - args.start_wall_gap,
            INITIAL_CENTERS[SHELL][1],
            args.start_wall_height,
        ],
        dtype=np.float64,
    )
    internal_rotation = rotation_matrix_y(np.deg2rad(args.start_internal_rotation_deg))
    internal_orientation = quat_np_from_y_rotation(np.deg2rad(args.start_internal_rotation_deg))

    poses = {SHELL: shell_target}
    for path, center in INITIAL_CENTERS.items():
        if path == SHELL:
            continue
        relative = center - INITIAL_CENTERS[SHELL]
        poses[path] = shell_target + internal_rotation @ relative

    return poses, internal_orientation


def apply_on_wall_runtime_pose(rigid_bodies_by_path):
    poses, orientation = compute_on_wall_initial_poses()
    identity = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    for path, body in rigid_bodies_by_path.items():
        position = poses[path].reshape(1, 3).astype(np.float32)
        if path == SHELL:
            body.set_world_poses(positions=position)
        else:
            body.set_world_poses(
                positions=position,
                orientations=orientation.reshape(1, 4),
            )
        if hasattr(body, "set_velocities"):
            body.set_velocities(
                linear_velocities=np.zeros((1, 3), dtype=np.float32),
                angular_velocities=np.zeros((1, 3), dtype=np.float32),
            )

    shell_target = poses[SHELL]
    print(
        "Initial wall-climb runtime pose: "
        f"shell=({shell_target[0]:+.3f}, {shell_target[1]:+.3f}, {shell_target[2]:+.3f}) m, "
        f"wall_gap={args.start_wall_gap} m, "
        f"internal_rotation_y={args.start_internal_rotation_deg} deg"
    )


def create_wall(stage):
    wall = UsdGeom.Cube.Define(stage, WALL_PATH)
    wall.CreateSizeAttr(1.0)
    wall.AddTranslateOp().Set(
        Gf.Vec3d(
            args.wall_x,
            0.060,
            args.wall_height * 0.5 - args.wall_thickness,
        )
    )
    wall.AddScaleOp().Set(Gf.Vec3f(args.wall_thickness, args.wall_width, args.wall_height))
    wall.CreateDisplayColorAttr([Gf.Vec3f(0.35, 0.36, 0.38)])
    UsdPhysics.CollisionAPI.Apply(wall.GetPrim())

    material = UsdShade.Material.Define(stage, "/World/materials/ferromagnetic_wall")
    physics = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics.CreateStaticFrictionAttr(args.ferro_static_friction)
    physics.CreateDynamicFrictionAttr(args.ferro_dynamic_friction)
    physics.CreateRestitutionAttr(0.0)
    UsdShade.MaterialBindingAPI.Apply(wall.GetPrim()).Bind(material)

    return wall


def make_ferromagnetic_material(stage):
    material = UsdShade.Material.Define(stage, "/World/materials/ferromagnetic_surface")
    physics = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics.CreateStaticFrictionAttr(args.ferro_static_friction)
    physics.CreateDynamicFrictionAttr(args.ferro_dynamic_friction)
    physics.CreateRestitutionAttr(0.0)
    return material


def ramp_frame(angle_deg):
    angle = np.deg2rad(angle_deg)
    tangent = np.array([np.cos(angle), 0.0, np.sin(angle)], dtype=np.float64)
    normal = np.array([-np.sin(angle), 0.0, np.cos(angle)], dtype=np.float64)
    start = np.array([args.ramp_start_x, 0.060, args.ramp_start_z], dtype=np.float64)
    center = start + 0.5 * args.ramp_length * tangent - 0.5 * args.ramp_thickness * normal
    return start, tangent, normal, center


def set_ramp_pose(stage, angle_deg):
    prim = stage.GetPrimAtPath(RAMP_PATH)
    if not prim:
        raise RuntimeError(f"Missing ramp prim: {RAMP_PATH}")

    _, _, _, center = ramp_frame(angle_deg)
    xformable = UsdGeom.Xformable(prim)
    translate_op = None
    rotate_op = None
    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            translate_op = op
        elif op.GetOpType() == UsdGeom.XformOp.TypeRotateY:
            rotate_op = op

    if translate_op is None:
        translate_op = xformable.AddTranslateOp()
    if rotate_op is None:
        rotate_op = xformable.AddRotateYOp()

    translate_op.Set(Gf.Vec3d(float(center[0]), float(center[1]), float(center[2])))
    rotate_op.Set(float(-angle_deg))


def create_ramp(stage):
    ramp = UsdGeom.Cube.Define(stage, RAMP_PATH)
    ramp.CreateSizeAttr(1.0)
    ramp.AddTranslateOp()
    ramp.AddRotateYOp()
    ramp.AddScaleOp().Set(Gf.Vec3f(args.ramp_length, args.ramp_width, args.ramp_thickness))
    ramp.CreateDisplayColorAttr([Gf.Vec3f(0.32, 0.34, 0.36)])
    UsdPhysics.CollisionAPI.Apply(ramp.GetPrim())
    UsdShade.MaterialBindingAPI.Apply(ramp.GetPrim()).Bind(make_ferromagnetic_material(stage))
    set_ramp_pose(stage, args.ramp_angle_start)
    return ramp


def stair_angles():
    if args.ramp_angle_step <= 0.0:
        return [args.ramp_angle_end]
    count = int(np.floor((args.ramp_angle_end - args.ramp_angle_start) / args.ramp_angle_step)) + 1
    angles = [args.ramp_angle_start + i * args.ramp_angle_step for i in range(max(count, 1))]
    if angles[-1] < args.ramp_angle_end:
        angles.append(args.ramp_angle_end)
    return [min(angle, args.ramp_angle_end) for angle in angles]


def create_ramp_stair(stage):
    root = UsdGeom.Xform.Define(stage, RAMP_STAIR_ROOT)
    material = make_ferromagnetic_material(stage)

    cursor = np.array([args.ramp_start_x, 0.060, args.ramp_start_z], dtype=np.float64)
    for index, angle_deg in enumerate(stair_angles()):
        angle = np.deg2rad(angle_deg)
        tangent = np.array([np.cos(angle), 0.0, np.sin(angle)], dtype=np.float64)
        normal = np.array([-np.sin(angle), 0.0, np.cos(angle)], dtype=np.float64)
        center = cursor + 0.5 * args.ramp_segment_length * tangent - 0.5 * args.ramp_thickness * normal

        segment = UsdGeom.Cube.Define(stage, f"{RAMP_STAIR_ROOT}/segment_{index:02d}_{int(round(angle_deg)):03d}deg")
        segment.CreateSizeAttr(1.0)
        segment.AddTranslateOp().Set(Gf.Vec3d(float(center[0]), float(center[1]), float(center[2])))
        segment.AddRotateYOp().Set(float(-angle_deg))
        segment.AddScaleOp().Set(Gf.Vec3f(args.ramp_segment_length, args.ramp_width, args.ramp_thickness))
        color = 0.28 + 0.45 * (index / max(len(stair_angles()) - 1, 1))
        segment.CreateDisplayColorAttr([Gf.Vec3f(float(color), float(color), float(color + 0.05))])
        UsdPhysics.CollisionAPI.Apply(segment.GetPrim())
        UsdShade.MaterialBindingAPI.Apply(segment.GetPrim()).Bind(material)

        cursor = cursor + (args.ramp_segment_length + args.ramp_segment_gap) * tangent

    return root


def ramp_stair_surface(shell_pos):
    cursor = np.array([args.ramp_start_x, 0.060, args.ramp_start_z], dtype=np.float64)
    best = None
    best_abs_s = np.inf
    for angle_deg in stair_angles():
        angle = np.deg2rad(angle_deg)
        tangent = np.array([np.cos(angle), 0.0, np.sin(angle)], dtype=np.float64)
        normal = np.array([-np.sin(angle), 0.0, np.cos(angle)], dtype=np.float64)
        relative = shell_pos - cursor
        s = np.dot(relative, tangent)
        clamped_s = np.clip(s, 0.0, args.ramp_segment_length)
        surface_point = cursor + clamped_s * tangent
        if 0.0 <= s <= args.ramp_segment_length:
            return surface_point, normal, angle_deg

        abs_s = min(abs(s), abs(s - args.ramp_segment_length))
        if abs_s < best_abs_s:
            best_abs_s = abs_s
            best = (surface_point, normal, angle_deg)

        cursor = cursor + (args.ramp_segment_length + args.ramp_segment_gap) * tangent

    return best


def current_ramp_angle(step):
    elapsed = step / 240.0
    if args.ramp_step_interval <= 0.0:
        return args.ramp_angle_end
    increments = int(elapsed // args.ramp_step_interval)
    angle = args.ramp_angle_start + increments * args.ramp_angle_step
    return min(angle, args.ramp_angle_end)


def magnet_metrics(shell_body, magnet_frame):
    shell_pos = first_vec(shell_body.get_world_poses()[0])
    magnet_pos = first_vec(magnet_frame.get_world_poses()[0])
    radial = magnet_pos - shell_pos
    radial_norm = np.linalg.norm(radial)
    if radial_norm < 1.0e-9:
        return {
            "angle_from_bottom_deg": 0.0,
            "normalized_height": 0.0,
            "radial_distance": 0.0,
            "gap": args.inner_radius,
            "face_gap": args.inner_radius,
        }

    radial_dir = radial / radial_norm
    down = np.array([0.0, 0.0, -1.0])
    cos_angle = np.clip(np.dot(radial_dir, down), -1.0, 1.0)
    angle_from_bottom_deg = np.degrees(np.arccos(cos_angle))

    # 0.0 = bottom of the shell, 0.5 = side/equator, 1.0 = top.
    normalized_height = 0.5 * (radial_dir[2] + 1.0)
    return {
        "angle_from_bottom_deg": angle_from_bottom_deg,
        "normalized_height": normalized_height,
        "radial_distance": radial_norm,
        "gap": args.inner_radius - radial_norm,
        "face_gap": args.inner_radius - radial_norm - 0.5 * args.magnet_size_z,
    }


def set_wheel_drive(stage, left_velocity, right_velocity, damping, max_force):
    for joint_path, velocity in (
        (LEFT_WHEEL_JOINT, left_velocity),
        (RIGHT_WHEEL_JOINT, right_velocity),
    ):
        joint = stage.GetPrimAtPath(joint_path)
        if not joint:
            raise RuntimeError(f"Missing wheel joint: {joint_path}")
        joint.GetAttribute("physics:axis").Set(args.wheel_axis)
        joint.GetAttribute("drive:angular:physics:targetVelocity").Set(velocity)
        joint.GetAttribute("drive:angular:physics:damping").Set(damping)
        joint.GetAttribute("drive:angular:physics:maxForce").Set(max_force)


def configure_wheel_drive(stage):
    set_wheel_drive(
        stage,
        args.left_wheel_velocity,
        args.right_wheel_velocity,
        args.wheel_damping,
        args.wheel_max_force,
    )


class Ros2CliTeleop:
    def __init__(self, topic):
        self.linear_x = 0.0
        self.angular_z = 0.0
        self.last_message_time = 0.0
        self.received = False
        self._reported_first_message = False
        self._reported_exit = False
        self._lock = threading.Lock()
        command = (
            "source /opt/ros/humble/setup.bash && "
            f"ros2 topic echo {topic} geometry_msgs/msg/Twist --csv"
        )
        env = os.environ.copy()
        env.pop("PYTHONHOME", None)
        env.pop("PYTHONPATH", None)
        self._process = subprocess.Popen(
            [
                "bash",
                "-lc",
                command,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        self._thread = threading.Thread(target=self._read_stdout, daemon=True)
        self._thread.start()

    def _read_stdout(self):
        if self._process.stdout is None:
            return
        for line in self._process.stdout:
            stripped = line.strip()
            values = [float(value) for value in re.findall(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?", line)]
            if len(values) < 6:
                if stripped and not stripped.startswith(("linear", "---")):
                    print(f"ROS 2 teleop bridge: {stripped}")
                continue
            with self._lock:
                self.linear_x = values[0]
                self.angular_z = values[5]
                self.last_message_time = time.monotonic()
                self.received = True
                if not self._reported_first_message:
                    print(
                        "ROS 2 teleop received first Twist: "
                        f"linear.x={self.linear_x:+.3f}, angular.z={self.angular_z:+.3f}"
                    )
                    self._reported_first_message = True

    def command(self):
        return_code = self._process.poll()
        if return_code is not None:
            if not self._reported_exit:
                print(f"ROS 2 teleop bridge exited with code {return_code}; wheel command is zero.")
                self._reported_exit = True
            return 0.0, 0.0
        with self._lock:
            if not self.received or time.monotonic() - self.last_message_time > args.cmd_timeout:
                return 0.0, 0.0
            return self.linear_x, self.angular_z

    def close(self):
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                self._process.kill()


def create_ros2_teleop():
    try:
        return Ros2CliTeleop(args.cmd_vel_topic)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "ROS 2 teleop requested, but the 'ros2' command is not available. "
            "Run the script from a terminal where ROS 2 is sourced: "
            "source /opt/ros/humble/setup.bash"
        ) from exc


def ros2_wheel_command(teleop):
    linear_x, angular_z = teleop.command()

    if linear_x == 0.0 and angular_z == 0.0:
        return 0.0, 0.0, 0.0, 0.0

    forward = args.cmd_linear_sign * args.cmd_linear_scale * linear_x
    turn = args.cmd_angular_sign * args.cmd_angular_scale * angular_z
    left_velocity = forward - turn
    right_velocity = forward + turn
    return float(left_velocity), float(right_velocity), float(forward), float(turn)


def apply_ramp_steering(stage, shell_body, base_velocity, damping, max_force):
    if not (args.ramp_test and args.ramp_steering_control):
        return 0.0, base_velocity, base_velocity

    shell_pos = first_vec(shell_body.get_world_poses()[0])
    lateral_error = shell_pos[1] - args.ramp_center_y
    correction = args.ramp_steering_sign * args.ramp_steering_gain * lateral_error
    correction = float(np.clip(correction, -args.ramp_steering_max_delta, args.ramp_steering_max_delta))
    left_velocity = base_velocity - correction
    right_velocity = base_velocity + correction
    set_wheel_drive(stage, left_velocity, right_velocity, damping, max_force)
    return correction, left_velocity, right_velocity


def exponential_magnetic_force(gap, normal_velocity, max_force, decay_distance, damping, capture=1.0, alignment=1.0):
    effective_gap = max(float(gap), 0.0)
    decay_distance = max(float(decay_distance), 1.0e-6)
    capture = float(np.clip(capture, 0.0, 1.0))
    alignment = float(np.clip(alignment, 0.0, 1.0))
    static_force = max_force * np.exp(-effective_gap / decay_distance)
    damped_force = static_force - damping * normal_velocity
    return min(max(alignment * capture * damped_force, 0.0), max_force)


def magnet_dipole_moment():
    mu0 = 4.0 * np.pi * 1.0e-7
    volume = args.magnet_size_x * args.magnet_size_y * args.magnet_size_z
    return args.magnet_br * volume / mu0


def orthonormal_basis_from_axis(axis):
    axis = axis / max(np.linalg.norm(axis), 1.0e-9)
    helper = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(np.dot(axis, helper)) > 0.92:
        helper = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    tangent_1 = np.cross(helper, axis)
    tangent_1 = tangent_1 / max(np.linalg.norm(tangent_1), 1.0e-9)
    tangent_2 = np.cross(axis, tangent_1)
    return tangent_1, tangent_2


def dipole_field_at_point(magnet_pos, magnet_axis, sample_point):
    mu0 = 4.0 * np.pi * 1.0e-7
    r_vec = sample_point - magnet_pos
    r_norm = max(np.linalg.norm(r_vec), args.field_min_distance, 1.0e-6)
    r_hat = r_vec / max(np.linalg.norm(r_vec), 1.0e-9)
    moment_vec = magnet_dipole_moment() * magnet_axis
    return (mu0 / (4.0 * np.pi)) * (3.0 * r_hat * np.dot(moment_vec, r_hat) - moment_vec) / (r_norm**3)


def ferromagnetic_plane_field_patch_force(magnet_pos, magnet_axis, active_shell_point, surface_point, surface_normal):
    mu0 = 4.0 * np.pi * 1.0e-7
    surface_normal = surface_normal / max(np.linalg.norm(surface_normal), 1.0e-9)
    tangent_1, tangent_2 = orthonormal_basis_from_axis(surface_normal)
    signed_distance = np.dot(magnet_pos - surface_point, surface_normal)
    patch_center = magnet_pos - signed_distance * surface_normal
    rings = max(args.field_rings, 1)
    ring_samples = max(args.field_ring_samples, 3)
    sample_count = 1 + rings * ring_samples
    patch_area = np.pi * args.field_patch_radius * args.field_patch_radius
    sample_area = patch_area / sample_count

    total_force = np.zeros(3, dtype=np.float64)
    weighted_point = np.zeros(3, dtype=np.float64)
    weighted_alignment = 0.0
    weighted_capture = 0.0
    pressure_sum = 0.0
    peak_b = 0.0
    min_gap = np.inf

    for ring in range(rings + 1):
        if ring == 0:
            offsets = [np.zeros(3, dtype=np.float64)]
        else:
            radius = args.field_patch_radius * ring / rings
            offsets = []
            for sample in range(ring_samples):
                phi = 2.0 * np.pi * sample / ring_samples
                offsets.append(radius * (np.cos(phi) * tangent_1 + np.sin(phi) * tangent_2))

        for offset in offsets:
            sample_point = patch_center + offset
            force_dir = sample_point - active_shell_point
            distance = np.linalg.norm(force_dir)
            if distance < 1.0e-6:
                continue
            force_dir = force_dir / distance
            field = dipole_field_at_point(magnet_pos, magnet_axis, sample_point)
            field_norm = np.linalg.norm(field)
            pressure = args.field_pressure_scale * field_norm * field_norm / (2.0 * mu0)
            local_force = pressure * sample_area * force_dir
            total_force += local_force

            capture = smoothstep((args.wall_capture_gap - distance) / args.wall_capture_gap)
            alignment = max(np.dot(magnet_axis, force_dir), 0.0)
            weighted_point += pressure * sample_point
            weighted_alignment += pressure * alignment
            weighted_capture += pressure * capture
            pressure_sum += pressure
            peak_b = max(peak_b, field_norm)
            min_gap = min(min_gap, distance)

    if pressure_sum > 1.0e-12:
        weighted_point /= pressure_sum
        weighted_alignment /= pressure_sum
        weighted_capture /= pressure_sum
    else:
        weighted_point = patch_center

    total_force = clamp_vector(total_force, args.wall_max_force)
    return total_force, {
        "force": np.linalg.norm(total_force),
        "gap": min_gap,
        "alignment": weighted_alignment,
        "capture": weighted_capture,
        "shell_point": active_shell_point,
        "surface_point": weighted_point,
        "peak_b": peak_b,
        "samples": sample_count,
    }


def dipole_plane_magnetic_force(gap, normal_velocity, max_force, damping, capture=1.0, alignment=1.0):
    mu0 = 4.0 * np.pi * 1.0e-7
    effective_distance = max(float(gap), args.dipole_min_distance, 1.0e-6)
    capture = float(np.clip(capture, 0.0, 1.0))
    alignment = float(np.clip(alignment, 0.0, 1.0))
    moment = magnet_dipole_moment()
    static_force = args.dipole_force_coeff * mu0 * moment * moment / (effective_distance**4)
    damped_force = static_force - damping * normal_velocity
    return min(max(alignment * capture * damped_force, 0.0), max_force)


def magnetic_force_magnitude(
    gap,
    normal_velocity,
    max_force,
    damping,
    capture=1.0,
    alignment=1.0,
    decay_distance=None,
):
    if args.magnetic_law == "dipole":
        return dipole_plane_magnetic_force(
            gap=gap,
            normal_velocity=normal_velocity,
            max_force=max_force,
            damping=damping,
            capture=capture,
            alignment=alignment,
        )

    return exponential_magnetic_force(
        gap=gap,
        normal_velocity=normal_velocity,
        max_force=max_force,
        decay_distance=decay_distance if decay_distance is not None else args.wall_decay_distance,
        damping=damping,
        capture=capture,
        alignment=alignment,
    )


def magnetic_law_summary():
    moment = magnet_dipole_moment()
    if args.magnetic_law == "dipole":
        return (
            "law=dipole-plane idealized, "
            f"Br={args.magnet_br} T, "
            f"size=({args.magnet_size_x * 1000.0:.1f},"
            f"{args.magnet_size_y * 1000.0:.1f},"
            f"{args.magnet_size_z * 1000.0:.1f}) mm, "
            f"m={moment:.3f} A*m^2, "
            f"z_min={args.dipole_min_distance * 1000.0:.1f} mm, "
            f"C={args.dipole_force_coeff:.6f}"
        )
    return (
        "law=exponential paper-fit fallback, "
        f"internal_decay={args.magnet_decay_distance} m, "
        f"external_decay={args.wall_decay_distance} m"
    )


def magnetic_force(shell_body, internal_body, magnet_frame):
    shell_pos = first_vec(shell_body.get_world_poses()[0])
    internal_pos = first_vec(internal_body.get_world_poses()[0])
    magnet_pos = first_vec(magnet_frame.get_world_poses()[0])

    shell_lin, shell_ang = shell_body.get_velocities()
    internal_lin, internal_ang = internal_body.get_velocities()
    shell_lin = first_vec(shell_lin)
    shell_ang = first_vec(shell_ang)
    internal_lin = first_vec(internal_lin)
    internal_ang = first_vec(internal_ang)

    radial = magnet_pos - shell_pos
    radial_norm = np.linalg.norm(radial)
    if radial_norm < 1.0e-6:
        return np.zeros((1, 3)), magnet_pos.reshape(1, 3), magnet_pos.reshape(1, 3)

    radial_dir = radial / radial_norm
    closest_inner_shell_point = shell_pos + args.inner_radius * radial_dir

    magnet_vel = point_velocity(internal_lin, internal_ang, internal_pos, magnet_pos)
    shell_point_vel = point_velocity(shell_lin, shell_ang, shell_pos, closest_inner_shell_point)
    relative_normal_velocity = np.dot(magnet_vel - shell_point_vel, radial_dir)

    gap = args.inner_radius - radial_norm - 0.5 * args.magnet_size_z
    normal_force = magnetic_force_magnitude(
        gap=gap,
        normal_velocity=relative_normal_velocity,
        max_force=args.max_force,
        damping=args.magnet_d,
        decay_distance=args.magnet_decay_distance,
    )
    force = normal_force * radial_dir
    return force.reshape(1, 3), magnet_pos.reshape(1, 3), closest_inner_shell_point.reshape(1, 3)


def ferromagnetic_surface_force(shell_body, internal_body, magnet_frame, surface_point, surface_normal):
    shell_pos = first_vec(shell_body.get_world_poses()[0])
    internal_pos = first_vec(internal_body.get_world_poses()[0])
    internal_lin, internal_ang = internal_body.get_velocities()
    internal_lin = first_vec(internal_lin)
    internal_ang = first_vec(internal_ang)
    magnet_pos = first_vec(magnet_frame.get_world_poses()[0])

    radial = magnet_pos - shell_pos
    radial_norm = np.linalg.norm(radial)
    if radial_norm < 1.0e-6:
        return (
            np.zeros((1, 3)),
            magnet_pos.reshape(1, 3),
            {"force": 0.0, "gap": np.inf, "alignment": 0.0, "capture": 0.0},
        )

    radial_dir = radial / radial_norm
    surface_normal = surface_normal / max(np.linalg.norm(surface_normal), 1.0e-9)
    magnetized_shell_point = shell_pos + args.shell_radius * radial_dir
    if args.external_model == "field-patch":
        force, metrics = ferromagnetic_plane_field_patch_force(
            magnet_pos=magnet_pos,
            magnet_axis=radial_dir,
            active_shell_point=magnetized_shell_point,
            surface_point=surface_point,
            surface_normal=surface_normal,
        )
        force_norm = np.linalg.norm(force)
        if force_norm > 1.0e-9:
            magnet_vel = point_velocity(internal_lin, internal_ang, internal_pos, magnet_pos)
            force_dir = force / force_norm
            damping_force = args.wall_d * np.dot(magnet_vel, force_dir)
            force = force_dir * max(force_norm - damping_force, 0.0)
            force = clamp_vector(force, args.wall_max_force)
            metrics["force"] = np.linalg.norm(force)
        return force.reshape(1, 3), magnet_pos.reshape(1, 3), metrics

    attraction_dir = -surface_normal
    alignment = max(np.dot(radial_dir, attraction_dir), 0.0)
    if alignment <= 0.0:
        return (
            np.zeros((1, 3)),
            magnet_pos.reshape(1, 3),
            {"force": 0.0, "gap": np.inf, "alignment": 0.0, "capture": 0.0},
        )

    gap = np.dot(magnetized_shell_point - surface_point, surface_normal)
    capture = smoothstep((args.wall_capture_gap - gap) / args.wall_capture_gap)

    alignment_scale = alignment**args.wall_alignment_power
    magnet_vel = point_velocity(internal_lin, internal_ang, internal_pos, magnet_pos)
    closing_velocity = np.dot(magnet_vel, attraction_dir)
    normal_force = magnetic_force_magnitude(
        gap=gap,
        normal_velocity=closing_velocity,
        max_force=args.wall_max_force,
        damping=args.wall_d,
        capture=capture,
        alignment=alignment_scale,
        decay_distance=args.wall_decay_distance,
    )
    force = normal_force * attraction_dir
    return (
        force.reshape(1, 3),
        magnet_pos.reshape(1, 3),
        {
            "force": normal_force,
            "gap": gap,
            "alignment": alignment,
            "capture": capture,
            "shell_point": magnetized_shell_point,
        },
    )


def wall_magnetic_force(shell_body, internal_body, magnet_frame):
    surface_point = np.array([args.wall_x - 0.5 * args.wall_thickness, 0.060, 0.0], dtype=np.float64)
    surface_normal = np.array([-1.0, 0.0, 0.0], dtype=np.float64)
    return ferromagnetic_surface_force(shell_body, internal_body, magnet_frame, surface_point, surface_normal)


def ramp_magnetic_force(shell_body, internal_body, magnet_frame, angle_deg):
    surface_point, _, surface_normal, _ = ramp_frame(angle_deg)
    return ferromagnetic_surface_force(shell_body, internal_body, magnet_frame, surface_point, surface_normal)


def ramp_stair_magnetic_force(shell_body, internal_body, magnet_frame):
    shell_pos = first_vec(shell_body.get_world_poses()[0])
    surface_point, surface_normal, angle_deg = ramp_stair_surface(shell_pos)
    force, pos, metrics = ferromagnetic_surface_force(
        shell_body,
        internal_body,
        magnet_frame,
        surface_point,
        surface_normal,
    )
    metrics["ramp_angle"] = angle_deg
    return force, pos, metrics


def apply_wheel_shell_preload(shell_body, internal_body, wheel_bodies):
    if args.wheel_shell_preload <= 0.0:
        return 0.0

    shell_pos = first_vec(shell_body.get_world_poses()[0])
    total_preload = 0.0
    for wheel_body in wheel_bodies:
        wheel_pos = first_vec(wheel_body.get_world_poses()[0])
        radial = wheel_pos - shell_pos
        radial_norm = np.linalg.norm(radial)
        if radial_norm < 1.0e-6:
            continue

        radial_dir = radial / radial_norm
        force = args.wheel_shell_preload * radial_dir
        wheel_body.apply_forces_and_torques_at_pos(
            forces=force.reshape(1, 3),
            positions=wheel_pos.reshape(1, 3),
        )
        internal_body.apply_forces_and_torques_at_pos(
            forces=(-force).reshape(1, 3),
            positions=wheel_pos.reshape(1, 3),
        )
        total_preload += args.wheel_shell_preload

    return total_preload


def main():
    usd_path = Path(args.usd).resolve()
    if not usd_path.exists():
        raise FileNotFoundError(usd_path)

    success, stage = stage_utils.open_stage(str(usd_path))
    if not success:
        raise RuntimeError(f"Failed to open stage: {usd_path}")
    if args.wall_test and args.ramp_test:
        raise RuntimeError("Use either --wall-test or --ramp-test, not both.")
    if args.wall_test:
        create_wall(stage)
    if args.ramp_test:
        if args.ramp_stair:
            create_ramp_stair(stage)
        else:
            create_ramp(stage)
    if args.start_on_wall:
        if not args.wall_test:
            raise RuntimeError("--start-on-wall requires --wall-test")
        initialize_on_wall(stage)
    if args.ros2_teleop:
        set_wheel_drive(stage, 0.0, 0.0, args.wheel_damping, args.wheel_max_force)
    else:
        configure_wheel_drive(stage)
    for _ in range(5):
        simulation_app.update()

    shell_body = RigidPrim(paths=SHELL)
    internal_body = RigidPrim(paths=INTERNAL)
    magnet_frame = XformPrim(paths=MAGNET_FRAME)
    left_wheel_body = RigidPrim(paths=LEFT_WHEEL)
    right_wheel_body = RigidPrim(paths=RIGHT_WHEEL)
    left_mount_body = RigidPrim(paths="/World/freebot/left_wheel_mount_link")
    right_mount_body = RigidPrim(paths="/World/freebot/right_wheel_mount_link")
    caster_1_body = RigidPrim(paths="/World/freebot/caster_1_ball_link")
    caster_2_body = RigidPrim(paths="/World/freebot/caster_2_ball_link")
    wheel_bodies = (
        left_wheel_body,
        right_wheel_body,
    )

    if args.start_on_wall:
        apply_on_wall_runtime_pose(
            {
                SHELL: shell_body,
                INTERNAL: internal_body,
                LEFT_WHEEL: left_wheel_body,
                RIGHT_WHEEL: right_wheel_body,
                "/World/freebot/left_wheel_mount_link": left_mount_body,
                "/World/freebot/right_wheel_mount_link": right_mount_body,
                "/World/freebot/caster_1_ball_link": caster_1_body,
                "/World/freebot/caster_2_ball_link": caster_2_body,
            }
        )
        for _ in range(3):
            simulation_app.update()

    SimulationManager.set_physics_dt(1.0 / 240.0)
    app_utils.play()
    simulation_app.update()

    print("Running FreeBOT magnetic locomotion test")
    print(f"USD: {usd_path}")
    print(
        "Magnetic model: "
        f"{magnetic_law_summary()}"
    )
    print(
        "Internal magnet-shell: "
        f"Fmax={args.max_force} N, "
        f"shell_reaction_scale={args.shell_reaction_scale}, "
        f"damping={args.magnet_d} N/(m/s)"
    )
    print(
        "Ferromagnetic surface material: "
        f"static_friction={args.ferro_static_friction}, "
        f"dynamic_friction={args.ferro_dynamic_friction}, restitution=0.0"
    )
    print(
        "Wheel drive: "
        f"axis={args.wheel_axis}, left={args.left_wheel_velocity} deg/s, "
        f"right={args.right_wheel_velocity} deg/s, "
        f"damping={args.wheel_damping}, maxForce={args.wheel_max_force}"
    )
    if args.wheel_shell_preload > 0.0:
        print(f"Wheel-shell preload: {args.wheel_shell_preload} N per drive wheel")
    print("Magnet telemetry: angle_from_bottom 0 deg=bottom, 90 deg=side/equator, 180 deg=top")
    if args.wall_test:
        print(
            "Wall test: "
            f"wall_x={args.wall_x} m, capture_gap={args.wall_capture_gap} m, "
            f"Fmax={args.wall_max_force} N, "
            f"shell_normal_scale={args.wall_shell_normal_scale}, "
            f"damping={args.wall_d} N/(m/s), external_model={args.external_model}"
        )
    if args.ramp_test:
        if args.ramp_stair:
            print(
                "Ramp stair test: "
                f"angles={args.ramp_angle_start}..{args.ramp_angle_end} deg, "
                f"step={args.ramp_angle_step} deg, segment_length={args.ramp_segment_length} m, "
                f"capture_gap={args.wall_capture_gap} m, Fmax={args.wall_max_force} N, "
                f"shell_normal_scale={args.wall_shell_normal_scale}, damping={args.wall_d} N/(m/s), "
                f"external_model={args.external_model}"
            )
        else:
            print(
                "Ramp test: "
                f"angle={args.ramp_angle_start}..{args.ramp_angle_end} deg, "
                f"step={args.ramp_angle_step} deg every {args.ramp_step_interval} s, "
                f"capture_gap={args.wall_capture_gap} m, Fmax={args.wall_max_force} N, "
                f"shell_normal_scale={args.wall_shell_normal_scale}, damping={args.wall_d} N/(m/s), "
                f"external_model={args.external_model}"
            )
    if args.external_model == "field-patch" and (args.wall_test or args.ramp_test):
        print(
            "External field patch: "
            f"radius={args.field_patch_radius} m, rings={args.field_rings}, "
            f"ring_samples={args.field_ring_samples}, pressure_scale={args.field_pressure_scale}"
        )
    if args.climb_after_capture:
        print(
            "Climb mode: "
            f"capture_threshold={args.climb_capture_threshold}, settle={args.climb_settle_time} s, "
            f"wheel_velocity={args.climb_wheel_velocity} deg/s, "
            f"damping={args.climb_wheel_damping}, maxForce={args.climb_wheel_max_force}"
        )
    ros2_teleop = None
    teleop_forward = 0.0
    teleop_turn = 0.0
    if args.ros2_teleop:
        ros2_teleop = create_ros2_teleop()
        print(
            "ROS 2 teleop: "
            f"topic={args.cmd_vel_topic}, linear_scale={args.cmd_linear_scale} deg/s, "
            f"angular_scale={args.cmd_angular_scale} deg/s, timeout={args.cmd_timeout} s"
        )

    max_magnet_angle = 0.0
    max_magnet_height = 0.0
    previous_wall_force = np.zeros(3, dtype=np.float64)
    physics_dt = 1.0 / 240.0
    capture_step = None
    climb_mode = False
    active_ramp_angle = args.ramp_angle_start
    steering_correction = 0.0
    commanded_left_velocity = 0.0 if args.ros2_teleop else args.left_wheel_velocity
    commanded_right_velocity = 0.0 if args.ros2_teleop else args.right_wheel_velocity
    for step in range(args.steps):
        if args.ros2_teleop:
            commanded_left_velocity, commanded_right_velocity, teleop_forward, teleop_turn = ros2_wheel_command(ros2_teleop)
            set_wheel_drive(
                stage,
                commanded_left_velocity,
                commanded_right_velocity,
                args.wheel_damping,
                args.wheel_max_force,
            )

        if args.ramp_test and not args.ramp_stair:
            next_ramp_angle = current_ramp_angle(step)
            if next_ramp_angle != active_ramp_angle or step == 0:
                active_ramp_angle = next_ramp_angle
                set_ramp_pose(stage, active_ramp_angle)

        force, magnet_pos, shell_target = magnetic_force(shell_body, internal_body, magnet_frame)
        internal_body.apply_forces_and_torques_at_pos(forces=force, positions=magnet_pos)
        if args.shell_reaction_scale != 0.0:
            shell_body.apply_forces_and_torques_at_pos(
                forces=-args.shell_reaction_scale * force,
                positions=shell_target,
            )
        wheel_preload = apply_wheel_shell_preload(shell_body, internal_body, wheel_bodies)
        if args.wall_test or args.ramp_test:
            if args.ramp_test:
                if args.ramp_stair:
                    wall_force, wall_pos, wall_metrics = ramp_stair_magnetic_force(
                        shell_body,
                        internal_body,
                        magnet_frame,
                    )
                    active_ramp_angle = wall_metrics["ramp_angle"]
                else:
                    wall_force, wall_pos, wall_metrics = ramp_magnetic_force(
                        shell_body,
                        internal_body,
                        magnet_frame,
                        active_ramp_angle,
                    )
            else:
                wall_force, wall_pos, wall_metrics = wall_magnetic_force(shell_body, internal_body, magnet_frame)
            if (
                not args.ros2_teleop
                and args.climb_after_capture
                and capture_step is None
                and wall_metrics["capture"] >= args.climb_capture_threshold
            ):
                capture_step = step

            if (
                not args.ros2_teleop
                and args.climb_after_capture
                and capture_step is not None
                and not climb_mode
                and step - capture_step >= int(args.climb_settle_time / physics_dt)
            ):
                set_wheel_drive(
                    stage,
                    args.climb_wheel_velocity,
                    args.climb_wheel_velocity,
                    args.climb_wheel_damping,
                    args.climb_wheel_max_force,
                )
                climb_mode = True

            if args.ros2_teleop:
                steering_correction = teleop_turn
            elif climb_mode:
                steering_correction, commanded_left_velocity, commanded_right_velocity = apply_ramp_steering(
                    stage,
                    shell_body,
                    args.climb_wheel_velocity,
                    args.climb_wheel_damping,
                    args.climb_wheel_max_force,
                )
            else:
                steering_correction, commanded_left_velocity, commanded_right_velocity = apply_ramp_steering(
                    stage,
                    shell_body,
                    0.5 * (args.left_wheel_velocity + args.right_wheel_velocity),
                    args.wheel_damping,
                    args.wheel_max_force,
                )

            raw_wall_force = wall_force.reshape(3)
            force_delta = raw_wall_force - previous_wall_force
            max_delta = args.wall_force_rate * physics_dt
            previous_wall_force = previous_wall_force + clamp_vector(force_delta, max_delta)
            wall_metrics["raw_force"] = wall_metrics["force"]
            wall_metrics["force"] = np.linalg.norm(previous_wall_force)
            internal_body.apply_forces_and_torques_at_pos(
                forces=previous_wall_force.reshape(1, 3),
                positions=wall_pos,
            )
            if args.wall_shell_normal_scale != 0.0:
                shell_wall_force = args.wall_shell_normal_scale * previous_wall_force
                shell_force_position = wall_metrics.get("shell_point")
                if shell_force_position is None:
                    shell_force_position = first_vec(shell_body.get_world_poses()[0])
                shell_force_position = np.asarray(shell_force_position, dtype=np.float64)
                shell_body.apply_forces_and_torques_at_pos(
                    forces=shell_wall_force.reshape(1, 3),
                    positions=shell_force_position.reshape(1, 3),
                )
                wall_metrics["shell_force"] = np.linalg.norm(shell_wall_force)
            else:
                wall_metrics["shell_force"] = 0.0
        else:
            wall_metrics = {
                "force": 0.0,
                "raw_force": 0.0,
                "shell_force": 0.0,
                "gap": np.inf,
                "alignment": 0.0,
                "capture": 0.0,
            }

        SimulationManager.step()
        RenderingManager.render()
        simulation_app.update()

        metrics = magnet_metrics(shell_body, magnet_frame)
        max_magnet_angle = max(max_magnet_angle, metrics["angle_from_bottom_deg"])
        max_magnet_height = max(max_magnet_height, metrics["normalized_height"])

        if args.log_interval > 0 and step % args.log_interval == 0:
            shell_pos = first_vec(shell_body.get_world_poses()[0])
            shell_ang = first_vec(shell_body.get_velocities()[1])
            print(
                f"t={step / 240.0:5.2f}s  shell=({shell_pos[0]:+.3f}, {shell_pos[1]:+.3f}, {shell_pos[2]:+.3f})  "
                f"omega=({shell_ang[0]:+.2f}, {shell_ang[1]:+.2f}, {shell_ang[2]:+.2f}) rad/s  "
                f"|Fmag|={np.linalg.norm(force):.2f} N  "
                f"Fwheel_preload={wheel_preload:.2f} N  "
                f"mag_angle={metrics['angle_from_bottom_deg']:5.1f} deg  "
                f"mag_h={metrics['normalized_height']:.2f}  "
                f"mag_face_gap={metrics['face_gap'] * 1000.0:+.1f} mm"
                f"  Fwall={wall_metrics['force']:.2f} N"
                f"  Fwall_raw={wall_metrics['raw_force']:.2f} N"
                f"  Fshell_wall={wall_metrics['shell_force']:.2f} N"
                f"  wall_gap={wall_metrics['gap'] * 1000.0:+.1f} mm"
                f"  wall_align={wall_metrics['alignment']:.2f}"
                f"  wall_capture={wall_metrics['capture']:.2f}"
                f"  Bpeak={wall_metrics.get('peak_b', 0.0):.3f} T"
                f"  field_samples={wall_metrics.get('samples', 0)}"
                f"  ramp_angle={active_ramp_angle:.1f}"
                f"  steer={steering_correction:+.1f}"
                f"  wheel_cmd=({commanded_left_velocity:+.0f},{commanded_right_velocity:+.0f})"
                f"  cmd_vel=({teleop_forward:+.0f},{teleop_turn:+.0f})"
                f"  climb={int(climb_mode)}"
            )

    if ros2_teleop is not None:
        ros2_teleop.close()

    print(
        "Magnet reach summary: "
        f"max_angle_from_bottom={max_magnet_angle:.1f} deg, "
        f"max_normalized_height={max_magnet_height:.2f}"
    )

    app_utils.stop()
    simulation_app.close()


if __name__ == "__main__":
    main()
