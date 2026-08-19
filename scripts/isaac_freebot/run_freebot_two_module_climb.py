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
    default="assets/freebot/usd_physics/freebot_cad_full_nearer_wheels_compliant.usd",
    help="FreeBOT physics USD containing one active CAD module.",
)
parser.add_argument("--headless", action="store_true")
parser.add_argument("--steps", type=int, default=12000)
parser.add_argument("--log-interval", type=int, default=240)
parser.add_argument("--wheel-axis", choices=["X", "Y", "Z"], default="Y")
parser.add_argument("--left-wheel-velocity", type=float, default=0.0, help="Fixed left wheel velocity [deg/s] when ROS 2 teleop is disabled.")
parser.add_argument("--right-wheel-velocity", type=float, default=0.0, help="Fixed right wheel velocity [deg/s] when ROS 2 teleop is disabled.")
parser.add_argument("--wheel-damping", type=float, default=500.0)
parser.add_argument("--wheel-max-force", type=float, default=1.2)
parser.add_argument("--ros2-teleop", action="store_true")
parser.add_argument("--cmd-vel-topic", default="/cmd_vel")
parser.add_argument("--cmd-linear-scale", type=float, default=900.0, help="Wheel-speed scale [deg/s per m/s]. Reduce explicitly for slow docking trials.")
parser.add_argument("--cmd-angular-scale", type=float, default=360.0)
parser.add_argument("--cmd-timeout", type=float, default=1.0)
parser.add_argument("--cmd-linear-sign", type=float, default=1.0)
parser.add_argument("--cmd-angular-sign", type=float, default=1.0)
parser.add_argument("--shell-radius", type=float, default=0.0665)
parser.add_argument("--inner-radius", type=float, default=0.0605)
parser.add_argument("--active-start-x", type=float, default=-0.30)
parser.add_argument("--active-start-y", type=float, default=0.060)
parser.add_argument("--platform-height", type=float, default=0.088, help="About two thirds of a module diameter.")
parser.add_argument("--platform-length", type=float, default=0.62)
parser.add_argument("--platform-width", type=float, default=0.42)
parser.add_argument("--platform-center-x", type=float, default=0.56)
parser.add_argument(
    "--passive-edge-clearance",
    type=float,
    default=0.0,
    help="Clearance between passive shell and platform vertical edge [m]. Positive values move the shell away from the platform.",
)
parser.add_argument(
    "--passive-geometry",
    choices=["module", "sphere"],
    default="module",
    help="Passive ferromagnetic base geometry. module clones the CAD/physics FreeBOT; sphere keeps the older analytic test body.",
)
parser.add_argument(
    "--passive-module-full-collisions",
    action="store_true",
    help="Keep all cloned passive-module colliders enabled. By default only the passive shell collides externally.",
)
parser.add_argument("--passive-count", type=int, choices=[1, 2], default=1, help="Number of passive shells. Validate docking with one shell before using a two-shell saddle.")
parser.add_argument("--passive-y-spacing", type=float, default=0.150, help="Center-to-center Y spacing between the two passive shells [m].")
parser.add_argument("--ground-static-friction", type=float, default=1.2)
parser.add_argument("--ground-dynamic-friction", type=float, default=1.0)
parser.add_argument("--platform-static-friction", type=float, default=1.4)
parser.add_argument("--platform-dynamic-friction", type=float, default=1.1)
parser.add_argument("--sphere-static-friction", type=float, default=2.0)
parser.add_argument("--sphere-dynamic-friction", type=float, default=1.6)
parser.add_argument("--magnet-br", type=float, default=1.47)
parser.add_argument("--magnet-size-x", type=float, default=0.020)
parser.add_argument("--magnet-size-y", type=float, default=0.020)
parser.add_argument("--magnet-size-z", type=float, default=0.010)
parser.add_argument("--dipole-min-distance", type=float, default=0.010)
parser.add_argument("--dipole-force-coeff", type=float, default=0.029841551829730376)
parser.add_argument("--internal-max-force", type=float, default=16.0)
parser.add_argument("--internal-damping", type=float, default=0.45)
parser.add_argument("--internal-align-stiffness", type=float, default=0.45, help="Radial magnet alignment stiffness [N*m/rad].")
parser.add_argument("--internal-align-damping", type=float, default=0.06, help="Relative angular damping for radial magnet alignment [N*m/(rad/s)].")
parser.add_argument("--internal-align-max-torque", type=float, default=0.20, help="Maximum internal magnetic alignment torque [N*m].")
parser.add_argument("--external-max-force", type=float, default=22.6, help="Calibrated normal magnetic force at zero shell gap [N].")
parser.add_argument("--external-decay-distance", type=float, default=0.010, help="Characteristic distance of the cubic magnetic decay law [m].")
parser.add_argument("--external-damping", type=float, default=0.8)
parser.add_argument("--external-capture-gap", type=float, default=0.025, help="Ramp-like magnetic capture tolerance [m] around the moving shell patch.")
parser.add_argument(
    "--external-gap-source",
    choices=["shell-patch", "shell-shell", "magnet-surface"],
    default="shell-patch",
    help=(
        "Distance used for the external magnetic force magnitude. "
        "shell-patch matches the ramp model using the magnetized active-shell point; "
        "shell-shell uses the gap between the two spherical shells plus an effective offset; "
        "magnet-surface uses the raw magnet-to-passive-shell gap."
    ),
)
parser.add_argument(
    "--external-effective-offset",
    type=float,
    default=0.010,
    help=(
        "Effective magnetic path length [m] added to shell-shell gap. "
        "This represents magnet/shell thickness and prevents the ideal dipole-plane singularity at shell contact."
    ),
)
parser.add_argument(
    "--external-min-distance",
    type=float,
    default=0.010,
    help="Minimum effective distance [m] used by the external dipole-to-steel formula.",
)
parser.add_argument(
    "--external-force-gain",
    type=float,
    default=1.0,
    help="Dimensionless gain applied to the analytic dipole-to-high-permeability-steel force.",
)
parser.add_argument("--external-force-rate", type=float, default=80.0, help="Maximum magnetic-force rise rate [N/s].")
parser.add_argument("--external-force-release-rate", type=float, default=160.0, help="Maximum magnetic-force release rate [N/s].")
parser.add_argument("--external-alignment-power", type=float, default=2.0)
parser.add_argument("--external-min-alignment", type=float, default=0.0, help="Legacy orientation baseline; zero matches the ramp model.")
parser.add_argument("--passive-switch-hysteresis", type=float, default=0.005, help="Distance advantage [m] required before switching magnetic target between two passive shells.")
parser.add_argument("--dock-align-threshold", type=float, default=0.65, help="Cosine alignment required to latch magnetic docking.")
parser.add_argument("--dock-release-align-threshold", type=float, default=0.30, help="Cosine alignment below which a latched dock is released.")
parser.add_argument("--dock-patch-gap", type=float, default=0.030, help="Maximum active/passive patch distance [m] for docking engagement.")
parser.add_argument("--dock-release-patch-gap", type=float, default=0.045, help="Patch distance [m] above which a latched dock is released.")
parser.add_argument("--dock-min-preload", type=float, default=6.0, help="Immediate normal preload [N] established when docking latches.")
parser.add_argument(
    "--magnetic-adhesion",
    action=argparse.BooleanOptionalAction,
    default=False,
    help=(
        "Enable compliant tangential magnetic hold for the internal mechanism. "
        "Normal and tangential magnetic forces act at the magnet while the active shell remains free to roll."
    ),
)
parser.add_argument(
    "--adhesion-gap",
    type=float,
    default=0.006,
    help="Shell-shell gap [m] below which the magnet-to-passive-surface adhesion is active.",
)
parser.add_argument("--adhesion-patch-gap", type=float, default=0.020, help="Maximum distance [m] between the moving magnetized patch and passive surface for stick/slip traction.")
parser.add_argument(
    "--adhesion-friction",
    type=float,
    default=1.4,
    help="Tangential hold coefficient for the magnetic adhesion patch, used as Ft_max = mu * Fn.",
)
parser.add_argument("--adhesion-max-force", type=float, default=22.6, help="Reduced-order tangential hold ceiling [N].")
parser.add_argument(
    "--adhesion-tangent-damping",
    type=float,
    default=4.0,
    help="Tangential viscous damping [N/(m/s)] at the magnetic adhesion patch, Coulomb-limited by adhesion_friction * Fn.",
)
parser.add_argument(
    "--adhesion-tangent-stiffness",
    type=float,
    default=1200.0,
    help="Tangential contact stiffness [N/m] used by the compliant magnetic stick/slip patch.",
)
parser.add_argument("--adhesion-supported-mass", type=float, default=0.300, help="Internal mechanism mass [kg] used to initialize gravity-balancing static hold at docking.")
parser.add_argument("--dock-require-neutral", action=argparse.BooleanOptionalAction, default=True, help="After first latch, stop the wheels until ROS teleop returns to neutral once.")
parser.add_argument("--dock-torsion-stiffness", type=float, default=0.35, help="Torsional docking stiffness about the magnetic normal [N*m/rad].")
parser.add_argument("--dock-torsion-damping", type=float, default=0.05, help="Torsional docking damping about the magnetic normal [N*m/(rad/s)].")
parser.add_argument("--dock-torsion-max-torque", type=float, default=0.12, help="Maximum compliant torsional holding torque [N*m].")
parser.add_argument("--dock-torsion", action=argparse.BooleanOptionalAction, default=False, help="Enable the legacy reduced-order torsional hold (disabled in the physical-contact baseline).")
parser.add_argument("--dock-release-on-reverse", action=argparse.BooleanOptionalAction, default=False, help="Legacy behavioral latch release; it never disables the normal magnetic force.")
parser.add_argument("--dock-detach-command-threshold", type=float, default=50.0, help="Negative scaled teleop command magnitude that enters manual undock mode.")
parser.add_argument(
    "--external-model",
    choices=["field-patch", "point"],
    default="point",
    help="External module-module magnetic model. point is the stable baseline; field-patch samples the passive shell surface experimentally.",
)
parser.add_argument("--field-cap-angle-deg", type=float, default=60.0, help="Angular radius of the sampled passive-shell magnetic patch [deg].")
parser.add_argument("--field-rings", type=int, default=4, help="Number of rings used by the passive-shell field patch.")
parser.add_argument("--field-ring-samples", type=int, default=12, help="Number of angular samples per non-central field-patch ring.")
parser.add_argument("--field-min-distance", type=float, default=0.010, help="Minimum magnet-to-surface distance for field calculation [m].")
parser.add_argument("--field-pressure-scale", type=float, default=1.0, help="Scale applied to Maxwell pressure B^2/(2*mu0).")
args, _ = parser.parse_known_args()

simulation_app = SimulationApp({"headless": args.headless})

import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.stage as stage_utils
from isaacsim.core.experimental.prims import RigidPrim, XformPrim
from isaacsim.core.rendering_manager import RenderingManager
from isaacsim.core.simulation_manager import SimulationManager
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade


ACTIVE = "/World/freebot"
SHELL = f"{ACTIVE}/shell_link"
INTERNAL = f"{ACTIVE}/internal_link"
MAGNET_FRAME = f"{ACTIVE}/internal_link/magnet_frame"
LEFT_WHEEL = f"{ACTIVE}/left_wheel_link"
RIGHT_WHEEL = f"{ACTIVE}/right_wheel_link"
LEFT_WHEEL_JOINT = f"{ACTIVE}/joints/left_wheel_joint"
RIGHT_WHEEL_JOINT = f"{ACTIVE}/joints/right_wheel_joint"
PASSIVE_SPHERE = "/World/passive_freebot_shell"
PASSIVE_SPHERE_2 = "/World/passive_freebot_shell_2"
PASSIVE_MODULE_PREFIX = "/World/passive_freebot"
PLATFORM = "/World/nonferromagnetic_platform"
GROUND = "/World/two_module_ground"

INITIAL_CENTERS = {
    SHELL: np.array([0.023103, 0.059745, 0.055943], dtype=np.float64),
    INTERNAL: np.array([0.023065, 0.060436, 0.026953], dtype=np.float64),
    LEFT_WHEEL: np.array([0.023042, 0.094171, 0.032054], dtype=np.float64),
    RIGHT_WHEEL: np.array([0.023043, 0.026872, 0.031846], dtype=np.float64),
    f"{ACTIVE}/left_wheel_mount_link": np.array([0.023042, 0.094171, 0.032054], dtype=np.float64),
    f"{ACTIVE}/right_wheel_mount_link": np.array([0.023043, 0.026872, 0.031846], dtype=np.float64),
    f"{ACTIVE}/caster_1_ball_link": np.array([-0.031633, 0.060363, 0.054020], dtype=np.float64),
    f"{ACTIVE}/caster_2_ball_link": np.array([0.077764, 0.060363, 0.054020], dtype=np.float64),
}

# Persistent target selection is used only when a two-passive saddle is requested.
# Hysteresis prevents a discontinuous force direction at the symmetry plane.
selected_passive_index = None


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


def create_physics_material(stage, path, static_friction, dynamic_friction, restitution=0.0):
    material = UsdShade.Material.Define(stage, path)
    physics = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics.CreateStaticFrictionAttr(static_friction)
    physics.CreateDynamicFrictionAttr(dynamic_friction)
    physics.CreateRestitutionAttr(restitution)
    material.GetPrim().CreateAttribute("physxMaterial:frictionCombineMode", Sdf.ValueTypeNames.Token).Set("max")
    material.GetPrim().CreateAttribute("physxMaterial:restitutionCombineMode", Sdf.ValueTypeNames.Token).Set("min")
    return material


def set_xform_translation(stage, path, position):
    prim = stage.GetPrimAtPath(path)
    if not prim:
        return
    xformable = UsdGeom.Xformable(prim)
    translate_op = None
    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            translate_op = op
            break
    if translate_op is None:
        translate_op = xformable.AddTranslateOp()
    translate_op.Set(Gf.Vec3d(float(position[0]), float(position[1]), float(position[2])))


def passive_module_path(index):
    return PASSIVE_MODULE_PREFIX if index == 0 else f"{PASSIVE_MODULE_PREFIX}_{index + 1}"


def retarget_relationships_to_clone(stage, clone_root_path):
    active_prefix = Sdf.Path(ACTIVE)
    clone_prefix = Sdf.Path(clone_root_path)
    clone_root = stage.GetPrimAtPath(clone_root_path)
    for prim in Usd.PrimRange(clone_root):
        for relationship in prim.GetRelationships():
            targets = relationship.GetTargets()
            if not targets:
                continue
            retargeted = []
            changed = False
            for target in targets:
                if target.HasPrefix(active_prefix):
                    suffix = target.MakeRelativePath(active_prefix)
                    retargeted.append(clone_prefix.AppendPath(suffix))
                    changed = True
                else:
                    retargeted.append(target)
            if changed:
                relationship.SetTargets(retargeted)


def make_clone_rigid_bodies_kinematic(stage, clone_root_path):
    clone_root = stage.GetPrimAtPath(clone_root_path)
    for prim in Usd.PrimRange(clone_root):
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rigid_body = UsdPhysics.RigidBodyAPI.Apply(prim)
            rigid_body.CreateKinematicEnabledAttr(True)


def remove_passive_clone_joints(stage, clone_root_path):
    # Passive modules are fixed environment bodies in this test. Their internal
    # joints are unnecessary and can create static/kinematic joint warnings.
    joints_path = f"{clone_root_path}/joints"
    if stage.GetPrimAtPath(joints_path):
        stage.RemovePrim(joints_path)


def keep_only_passive_shell_colliders(stage, clone_root_path):
    clone_root = stage.GetPrimAtPath(clone_root_path)
    shell_prefix = Sdf.Path(f"{clone_root_path}/shell_link")
    for prim in Usd.PrimRange(clone_root):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        collision_api = UsdPhysics.CollisionAPI.Apply(prim)
        collision_api.CreateCollisionEnabledAttr(prim.GetPath().HasPrefix(shell_prefix))


def initialize_passive_module_clone(stage, index, shell_center, passive_material):
    clone_root_path = passive_module_path(index)
    if stage.GetPrimAtPath(clone_root_path):
        stage.RemovePrim(clone_root_path)

    root_layer = stage.GetRootLayer()
    Sdf.CopySpec(root_layer, Sdf.Path(ACTIVE), root_layer, Sdf.Path(clone_root_path))
    retarget_relationships_to_clone(stage, clone_root_path)
    remove_passive_clone_joints(stage, clone_root_path)

    delta = shell_center - INITIAL_CENTERS[SHELL]
    for active_path, initial_center in INITIAL_CENTERS.items():
        clone_path = active_path.replace(ACTIVE, clone_root_path, 1)
        set_xform_translation(stage, clone_path, initial_center + delta)

    make_clone_rigid_bodies_kinematic(stage, clone_root_path)
    if not args.passive_module_full_collisions:
        keep_only_passive_shell_colliders(stage, clone_root_path)

    shell_prim = stage.GetPrimAtPath(f"{clone_root_path}/shell_link")
    if shell_prim:
        # The source SDF meshes already carry direct material bindings. Bind the
        # passive material on every actual collision prim, rather than relying
        # on inheritance from shell_link, so the requested friction is exactly
        # what PhysX resolves at shell-shell contact.
        for prim in Usd.PrimRange(shell_prim):
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(passive_material)
    return clone_root_path


def active_shell_start():
    return np.array([args.active_start_x, args.active_start_y, args.shell_radius], dtype=np.float64)


def passive_shell_center():
    left_edge = args.platform_center_x - 0.5 * args.platform_length
    return np.array(
        [
            left_edge - args.shell_radius - args.passive_edge_clearance,
            args.active_start_y,
            args.shell_radius,
        ],
        dtype=np.float64,
    )


def passive_shell_centers():
    first_center = passive_shell_center()
    if args.passive_count == 1:
        return [first_center]
    half_spacing = 0.5 * args.passive_y_spacing
    return [
        first_center + np.array([0.0, -half_spacing, 0.0], dtype=np.float64),
        first_center + np.array([0.0, +half_spacing, 0.0], dtype=np.float64),
    ]


def initialize_active_module(stage):
    shell_target = active_shell_start()
    delta = shell_target - INITIAL_CENTERS[SHELL]
    for path, initial_center in INITIAL_CENTERS.items():
        set_xform_translation(stage, path, initial_center + delta)
    print(
        "Active module initial pose: "
        f"shell=({shell_target[0]:+.3f}, {shell_target[1]:+.3f}, {shell_target[2]:+.3f}) m"
    )


def create_environment(stage):
    ground_material = create_physics_material(
        stage,
        "/World/materials/two_module_ground",
        args.ground_static_friction,
        args.ground_dynamic_friction,
    )
    platform_material = create_physics_material(
        stage,
        "/World/materials/nonferromagnetic_platform",
        args.platform_static_friction,
        args.platform_dynamic_friction,
    )
    passive_material = create_physics_material(
        stage,
        "/World/materials/passive_ferromagnetic_shell",
        args.sphere_static_friction,
        args.sphere_dynamic_friction,
    )

    ground = UsdGeom.Cube.Define(stage, GROUND)
    ground.CreateSizeAttr(1.0)
    ground.AddTranslateOp().Set(Gf.Vec3d(0.35, args.active_start_y, -0.01))
    ground.AddScaleOp().Set(Gf.Vec3f(1.8, 0.9, 0.02))
    ground.CreateDisplayColorAttr([Gf.Vec3f(0.55, 0.55, 0.55)])
    UsdPhysics.CollisionAPI.Apply(ground.GetPrim())
    UsdShade.MaterialBindingAPI.Apply(ground.GetPrim()).Bind(ground_material)

    platform = UsdGeom.Cube.Define(stage, PLATFORM)
    platform.CreateSizeAttr(1.0)
    platform.AddTranslateOp().Set(
        Gf.Vec3d(
            args.platform_center_x,
            args.active_start_y,
            0.5 * args.platform_height,
        )
    )
    platform.AddScaleOp().Set(Gf.Vec3f(args.platform_length, args.platform_width, args.platform_height))
    platform.CreateDisplayColorAttr([Gf.Vec3f(0.42, 0.42, 0.44)])
    UsdPhysics.CollisionAPI.Apply(platform.GetPrim())
    UsdShade.MaterialBindingAPI.Apply(platform.GetPrim()).Bind(platform_material)

    centers = passive_shell_centers()
    passive_paths = []
    for index, center in enumerate(centers):
        if args.passive_geometry == "module":
            passive_paths.append(initialize_passive_module_clone(stage, index, center, passive_material))
        else:
            path = PASSIVE_SPHERE if index == 0 else PASSIVE_SPHERE_2
            passive = UsdGeom.Sphere.Define(stage, path)
            passive.CreateRadiusAttr(args.shell_radius)
            passive.AddTranslateOp().Set(Gf.Vec3d(float(center[0]), float(center[1]), float(center[2])))
            passive.CreateDisplayColorAttr([Gf.Vec3f(0.88, 0.88, 0.88)])
            UsdPhysics.CollisionAPI.Apply(passive.GetPrim())
            UsdShade.MaterialBindingAPI.Apply(passive.GetPrim()).Bind(passive_material)
            passive_paths.append(path)

    print(
        "Two-module environment: "
        f"passive_geometry={args.passive_geometry}, "
        f"passive_module_collisions={'full' if args.passive_module_full_collisions else 'shell-only'}, "
        f"platform_height={args.platform_height:.3f} m, "
        f"platform_left_edge={args.platform_center_x - 0.5 * args.platform_length:+.3f} m, "
        f"passive_count={args.passive_count}, "
        f"passive_centers={[(round(c[0], 3), round(c[1], 3), round(c[2], 3)) for c in centers]}, "
        f"passive_paths={passive_paths}"
    )


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
            ["bash", "-lc", command],
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
            values = [float(value) for value in re.findall(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?", line)]
            if len(values) < 6:
                stripped = line.strip()
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

    def clear_command(self):
        """Forget the approach command; a new ROS Twist is required after latch."""
        with self._lock:
            self.linear_x = 0.0
            self.angular_z = 0.0
            self.last_message_time = 0.0
            self.received = False

    def close(self):
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                self._process.kill()


def ros2_wheel_command(teleop):
    linear_x, angular_z = teleop.command()
    if linear_x == 0.0 and angular_z == 0.0:
        return 0.0, 0.0, 0.0, 0.0
    forward = args.cmd_linear_sign * args.cmd_linear_scale * linear_x
    turn = args.cmd_angular_sign * args.cmd_angular_scale * angular_z
    return float(forward - turn), float(forward + turn), float(forward), float(turn)


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


def passive_shell_field_patch_force(magnet_pos, magnet_axis, active_shell_point, passive_center):
    mu0 = 4.0 * np.pi * 1.0e-7
    to_magnet = magnet_pos - passive_center
    to_magnet_norm = np.linalg.norm(to_magnet)
    if to_magnet_norm < 1.0e-6:
        return np.zeros(3), {
            "force": 0.0,
            "gap": np.inf,
            "alignment": 0.0,
            "capture": 0.0,
            "peak_b": 0.0,
            "samples": 0,
        }

    cap_axis = to_magnet / to_magnet_norm
    tangent_1, tangent_2 = orthonormal_basis_from_axis(cap_axis)
    cap_angle = np.deg2rad(max(args.field_cap_angle_deg, 1.0))
    rings = max(args.field_rings, 1)
    ring_samples = max(args.field_ring_samples, 3)
    sample_count = 1 + rings * ring_samples
    cap_area = 2.0 * np.pi * args.shell_radius * args.shell_radius * (1.0 - np.cos(cap_angle))
    sample_area = cap_area / sample_count

    total_force = np.zeros(3, dtype=np.float64)
    peak_b = 0.0
    min_gap = np.inf
    weighted_alignment = 0.0
    weighted_capture = 0.0
    pressure_sum = 0.0

    for ring in range(rings + 1):
        if ring == 0:
            ring_dirs = [cap_axis]
        else:
            theta = cap_angle * ring / rings
            ring_dirs = []
            for sample in range(ring_samples):
                phi = 2.0 * np.pi * sample / ring_samples
                ring_dirs.append(
                    np.cos(theta) * cap_axis
                    + np.sin(theta) * (np.cos(phi) * tangent_1 + np.sin(phi) * tangent_2)
                )

        for surface_normal in ring_dirs:
            surface_normal = surface_normal / max(np.linalg.norm(surface_normal), 1.0e-9)
            sample_point = passive_center + args.shell_radius * surface_normal
            force_dir = sample_point - active_shell_point
            force_distance = np.linalg.norm(force_dir)
            if force_distance < 1.0e-6:
                continue
            force_dir = force_dir / force_distance

            field = dipole_field_at_point(magnet_pos, magnet_axis, sample_point)
            field_norm = np.linalg.norm(field)
            pressure = args.field_pressure_scale * field_norm * field_norm / (2.0 * mu0)
            local_force_norm = pressure * sample_area
            total_force += local_force_norm * force_dir

            gap = force_distance
            capture = smoothstep((args.external_capture_gap - gap) / args.external_capture_gap)
            alignment = max(np.dot(magnet_axis, force_dir), 0.0)
            weighted_alignment += pressure * alignment
            weighted_capture += pressure * capture
            pressure_sum += pressure
            min_gap = min(min_gap, gap)
            peak_b = max(peak_b, field_norm)

    if pressure_sum > 1.0e-12:
        weighted_alignment /= pressure_sum
        weighted_capture /= pressure_sum

    force_norm = np.linalg.norm(total_force)
    if args.external_max_force > 0.0 and force_norm > args.external_max_force:
        total_force *= args.external_max_force / force_norm
        force_norm = args.external_max_force

    return total_force, {
        "force": force_norm,
        "gap": min_gap,
        "alignment": weighted_alignment,
        "capture": weighted_capture,
        "active_shell_point": active_shell_point,
        "peak_b": peak_b,
        "samples": sample_count,
    }


def dipole_force(gap, normal_velocity, max_force, damping, capture=1.0, alignment=1.0):
    mu0 = 4.0 * np.pi * 1.0e-7
    effective_distance = max(float(gap), args.dipole_min_distance, 1.0e-6)
    moment = magnet_dipole_moment()
    static_force = args.dipole_force_coeff * mu0 * moment * moment / (effective_distance**4)
    damped_force = static_force - damping * normal_velocity
    return min(max(capture * alignment * damped_force, 0.0), max_force)


def dipole_to_high_mu_plane_force(effective_distance, normal_velocity, max_force, damping, capture=1.0, alignment=1.0):
    """Analytic force of a magnetic dipole normal to an ideal high-permeability plane.

    This uses the image-dipole approximation:

        F = 3 * mu0 * m^2 / (32 * pi * z^4)

    where m is estimated from Br and magnet volume, and z is the effective magnetic
    path length. The shell-shell model passes z = shell_gap + external_effective_offset.

    Numerical fix: cap the raw physical force before multiplying by capture/alignment.
    If the cap is applied after alignment, the 1/z^4 term saturates to max_force even
    when the magnet is badly aligned, producing a launch impulse.
    """
    mu0 = 4.0 * np.pi * 1.0e-7
    z = max(float(effective_distance), args.external_min_distance, 1.0e-6)
    moment = magnet_dipole_moment()
    raw_static_force = args.external_force_gain * (3.0 * mu0 * moment * moment) / (32.0 * np.pi * z**4)
    positive_static_force = max(raw_static_force, 0.0)
    capped_static_force = min(positive_static_force, max_force) if max_force > 0.0 else positive_static_force
    damped_force = max(capped_static_force - damping * normal_velocity, 0.0)
    return min(max(capture, 0.0), 1.0) * min(max(alignment, 0.0), 1.0) * damped_force


def active_internal_magnetic_force(shell_body, internal_body, magnet_frame):
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
    shell_point = shell_pos + args.inner_radius * radial_dir

    magnet_vel = point_velocity(internal_lin, internal_ang, internal_pos, magnet_pos)
    shell_point_vel = point_velocity(shell_lin, shell_ang, shell_pos, shell_point)
    normal_velocity = np.dot(magnet_vel - shell_point_vel, radial_dir)
    gap = args.inner_radius - radial_norm - 0.5 * args.magnet_size_z
    force_norm = dipole_force(gap, normal_velocity, args.internal_max_force, args.internal_damping)
    return (force_norm * radial_dir).reshape(1, 3), magnet_pos.reshape(1, 3), shell_point.reshape(1, 3)


def active_internal_alignment_torque(shell_body, internal_body, magnet_frame):
    """Return an equal-and-opposite torque that keeps the magnet face radial.

    The vector from the internal body's centre to the magnet represents the
    magnet-facing direction. Its target is the local shell radius through the
    magnet. The spring term derives from angular misalignment, while damping
    removes only relative angular motion perpendicular to that radius; rotation
    about the radial axis remains free.
    """
    shell_pos = first_vec(shell_body.get_world_poses()[0])
    internal_pos = first_vec(internal_body.get_world_poses()[0])
    magnet_pos = first_vec(magnet_frame.get_world_poses()[0])
    shell_ang = first_vec(shell_body.get_velocities()[1])
    internal_ang = first_vec(internal_body.get_velocities()[1])

    magnet_axis = magnet_pos - internal_pos
    radial = magnet_pos - shell_pos
    magnet_axis_norm = np.linalg.norm(magnet_axis)
    radial_norm = np.linalg.norm(radial)
    if magnet_axis_norm < 1.0e-6 or radial_norm < 1.0e-6:
        return np.zeros(3, dtype=np.float64), 0.0

    magnet_axis /= magnet_axis_norm
    radial_dir = radial / radial_norm
    error_cross = np.cross(magnet_axis, radial_dir)
    error_sine = np.linalg.norm(error_cross)
    error_cosine = float(np.clip(np.dot(magnet_axis, radial_dir), -1.0, 1.0))
    error_angle = float(np.arctan2(error_sine, error_cosine))
    if error_sine > 1.0e-9:
        error_axis = error_cross / error_sine
        spring_torque = args.internal_align_stiffness * error_angle * error_axis
    else:
        spring_torque = np.zeros(3, dtype=np.float64)

    relative_angular_velocity = internal_ang - shell_ang
    tangential_angular_velocity = relative_angular_velocity - np.dot(relative_angular_velocity, radial_dir) * radial_dir
    torque = spring_torque - args.internal_align_damping * tangential_angular_velocity
    return clamp_vector(torque, args.internal_align_max_torque), error_angle


def active_to_passive_magnetic_force(shell_body, internal_body, magnet_frame):
    global selected_passive_index
    shell_pos = first_vec(shell_body.get_world_poses()[0])
    internal_pos = first_vec(internal_body.get_world_poses()[0])
    magnet_pos = first_vec(magnet_frame.get_world_poses()[0])
    internal_lin, internal_ang = internal_body.get_velocities()
    internal_lin = first_vec(internal_lin)
    internal_ang = first_vec(internal_ang)

    radial = magnet_pos - shell_pos
    radial_norm = np.linalg.norm(radial)
    if radial_norm < 1.0e-6:
        return np.zeros(3), magnet_pos, {"force": 0.0, "gap": np.inf, "alignment": 0.0, "capture": 0.0}

    radial_dir = radial / radial_norm
    active_shell_point = shell_pos + args.shell_radius * radial_dir
    passive_centers = passive_shell_centers()
    if args.external_model == "field-patch":
        force = np.zeros(3, dtype=np.float64)
        metrics = {
            "force": 0.0,
            "gap": np.inf,
            "alignment": 0.0,
            "capture": 0.0,
            "active_shell_point": active_shell_point,
            "peak_b": 0.0,
            "samples": 0,
            "passive_index": -1,
        }
        weighted_force = 0.0
        for index, passive_center in enumerate(passive_centers):
            candidate_force, candidate_metrics = passive_shell_field_patch_force(
                magnet_pos=magnet_pos,
                magnet_axis=radial_dir,
                active_shell_point=active_shell_point,
                passive_center=passive_center,
            )
            candidate_norm = np.linalg.norm(candidate_force)
            force += candidate_force
            if candidate_norm > weighted_force:
                weighted_force = candidate_norm
                metrics.update(candidate_metrics)
                metrics["passive_index"] = index
            metrics["samples"] += candidate_metrics.get("samples", 0)
            metrics["peak_b"] = max(metrics["peak_b"], candidate_metrics.get("peak_b", 0.0))

        force_norm = np.linalg.norm(force)
        if force_norm > 1.0e-9:
            magnet_vel = point_velocity(internal_lin, internal_ang, internal_pos, magnet_pos)
            force_dir = force / force_norm
            damping_force = args.external_damping * np.dot(magnet_vel, force_dir)
            force = force_dir * max(force_norm - damping_force, 0.0)
            if args.external_max_force > 0.0:
                force = clamp_vector(force, args.external_max_force)
            metrics["force"] = np.linalg.norm(force)
        return force, magnet_pos, metrics

    passive_distances = [np.linalg.norm(center - shell_pos) for center in passive_centers]
    nearest_index = int(np.argmin(passive_distances))
    if selected_passive_index is None or selected_passive_index >= len(passive_centers):
        selected_passive_index = nearest_index
    elif nearest_index != selected_passive_index:
        current_distance = passive_distances[selected_passive_index]
        if passive_distances[nearest_index] + args.passive_switch_hysteresis < current_distance:
            selected_passive_index = nearest_index
    passive_index = selected_passive_index
    passive_center = passive_centers[passive_index]

    # Physical-contact baseline: the magnetic attraction is represented by a
    # purely normal load at the closest points of the two outer shells. PhysX
    # then produces tangential traction exclusively through Coulomb contact.
    center_line = passive_center - shell_pos
    center_distance = np.linalg.norm(center_line)
    if center_distance < 1.0e-6:
        return np.zeros(3), magnet_pos, {
            "force": 0.0,
            "gap": np.inf,
            "shell_gap": np.inf,
            "magnetic_distance": np.inf,
            "alignment": 0.0,
            "capture": 0.0,
        }

    attraction_dir = center_line / center_distance
    active_shell_point = shell_pos + args.shell_radius * attraction_dir
    passive_surface_point = passive_center - args.shell_radius * attraction_dir
    patch_gap = np.linalg.norm(passive_surface_point - active_shell_point)
    shell_gap = max(center_distance - 2.0 * args.shell_radius, 0.0)
    axis_alignment = max(np.dot(radial_dir, attraction_dir), 0.0)
    capture = smoothstep((args.external_capture_gap - shell_gap) / max(args.external_capture_gap, 1.0e-9))

    # Calibrated short-range law requested by the model specification. There is
    # no orientation multiplier and no capture gate in the force itself.
    decay_distance = max(args.external_decay_distance, 1.0e-9)
    force_norm = args.external_max_force / (1.0 + shell_gap / decay_distance) ** 3
    force = force_norm * attraction_dir
    return force, active_shell_point, {
        "force": force_norm,
        "gap": shell_gap,
        "shell_gap": shell_gap,
        "shell_patch_gap": shell_gap,
        "magnetic_distance": shell_gap,
        "alignment": axis_alignment,
        "axis_alignment": axis_alignment,
        "capture": capture,
        "active_shell_point": active_shell_point,
        "passive_surface_point": passive_surface_point,
        "magnet_surface_distance": np.linalg.norm(passive_surface_point - magnet_pos),
        "magnet_surface_gap": max(np.linalg.norm(passive_surface_point - magnet_pos) - 0.5 * args.magnet_size_z, 0.0),
        "patch_gap": patch_gap,
        "attraction_dir": attraction_dir,
        "passive_index": passive_index,
    }


def magnetic_adhesion_patch_force(internal_body, magnet_pos, external_force, external_metrics, contact_state, physics_dt):
    """Return compliant tangential hold acting on the internal magnet.

    The magnetic normal force remains applied to the internal magnet. This term
    represents the unresolved magnet/active-shell/passive-shell friction chain.
    It supports the internal mechanism while leaving the active shell free to
    roll through its actual PhysX contacts. It is bounded by Coulomb friction and
    stores only a compliant tangential displacement, so wheel torque can drive it
    into slip and move the magnetic patch during climbing.
    """
    if not args.magnetic_adhesion:
        contact_state["displacement"] = np.zeros(3, dtype=np.float64)
        contact_state["passive_index"] = None
        return np.zeros(3), np.zeros(3), 0.0, 0.0, False, "free"

    shell_gap = external_metrics.get("shell_gap", np.inf)
    patch_gap = external_metrics.get("patch_gap", np.inf)
    normal_force_norm = np.linalg.norm(external_force)
    dock_latched = bool(external_metrics.get("dock_latched", False))
    if not dock_latched or normal_force_norm < 1.0e-6:
        contact_state["displacement"] = np.zeros(3, dtype=np.float64)
        contact_state["passive_index"] = None
        return np.zeros(3), np.zeros(3), 0.0, 0.0, False, "free"

    attraction_value = external_metrics.get("attraction_dir")
    if attraction_value is None:
        contact_state["displacement"] = np.zeros(3, dtype=np.float64)
        contact_state["passive_index"] = None
        return np.zeros(3), np.zeros(3), 0.0, 0.0, False, "free"
    attraction_dir = np.asarray(attraction_value, dtype=np.float64)
    attraction_dir_norm = np.linalg.norm(attraction_dir)
    if attraction_dir_norm < 1.0e-9:
        contact_state["displacement"] = np.zeros(3, dtype=np.float64)
        contact_state["passive_index"] = None
        return np.zeros(3), np.zeros(3), 0.0, 0.0, False, "free"
    normal_dir = attraction_dir / attraction_dir_norm

    application_pos = np.asarray(magnet_pos, dtype=np.float64)
    internal_pos = first_vec(internal_body.get_world_poses()[0])
    internal_lin, internal_ang = internal_body.get_velocities()
    magnet_velocity = point_velocity(first_vec(internal_lin), first_vec(internal_ang), internal_pos, application_pos)
    tangential_velocity = magnet_velocity - np.dot(magnet_velocity, normal_dir) * normal_dir

    passive_index = int(external_metrics.get("passive_index", 0))
    if contact_state.get("passive_index") != passive_index:
        contact_state["passive_index"] = passive_index
        # Static friction in a contact solver reacts immediately. Our compliant
        # surrogate instead needs spring displacement, so initialize it with the
        # small deflection that balances the tangential component of gravity.
        gravity_force = np.array([0.0, 0.0, -args.adhesion_supported_mass * 9.81], dtype=np.float64)
        gravity_tangent = gravity_force - np.dot(gravity_force, normal_dir) * normal_dir
        available_tangential_force = min(
            args.adhesion_friction * normal_force_norm,
            args.adhesion_max_force,
        )
        desired_hold = clamp_vector(-gravity_tangent, available_tangential_force)
        if args.adhesion_tangent_stiffness > 1.0e-9:
            contact_state["displacement"] = -(
                desired_hold + args.adhesion_tangent_damping * tangential_velocity
            ) / args.adhesion_tangent_stiffness
        else:
            contact_state["displacement"] = np.zeros(3, dtype=np.float64)
    displacement = np.asarray(contact_state.get("displacement", np.zeros(3)), dtype=np.float64)
    displacement -= np.dot(displacement, normal_dir) * normal_dir
    displacement += tangential_velocity * physics_dt

    tangential_candidate = (
        -args.adhesion_tangent_stiffness * displacement
        -args.adhesion_tangent_damping * tangential_velocity
    )
    tangential_limit = min(
        args.adhesion_friction * normal_force_norm,
        args.adhesion_max_force,
    )
    tangential_force = clamp_vector(tangential_candidate, tangential_limit)
    slipping = np.linalg.norm(tangential_candidate) > tangential_limit + 1.0e-9
    if slipping and args.adhesion_tangent_stiffness > 1.0e-9:
        # Return-map the spring state to the Coulomb surface. This prevents
        # unbounded stored energy and launch impulses after prolonged sliding.
        displacement = -(
            tangential_force + args.adhesion_tangent_damping * tangential_velocity
        ) / args.adhesion_tangent_stiffness
        displacement -= np.dot(displacement, normal_dir) * normal_dir
    contact_state["displacement"] = displacement
    return (
        tangential_force,
        application_pos,
        np.linalg.norm(tangential_force),
        tangential_limit,
        True,
        "slip" if slipping else "stick",
    )


def main():
    usd_path = Path(args.usd).resolve()
    if not usd_path.exists():
        raise FileNotFoundError(usd_path)

    success, stage = stage_utils.open_stage(str(usd_path))
    if not success:
        raise RuntimeError(f"Failed to open stage: {usd_path}")

    initialize_active_module(stage)
    create_environment(stage)
    if args.ros2_teleop:
        set_wheel_drive(stage, 0.0, 0.0, args.wheel_damping, args.wheel_max_force)
    else:
        set_wheel_drive(stage, args.left_wheel_velocity, args.right_wheel_velocity, args.wheel_damping, args.wheel_max_force)

    for _ in range(5):
        simulation_app.update()

    shell_body = RigidPrim(paths=SHELL)
    internal_body = RigidPrim(paths=INTERNAL)
    magnet_frame = XformPrim(paths=MAGNET_FRAME)

    SimulationManager.set_physics_dt(1.0 / 240.0)
    app_utils.play()
    simulation_app.update()

    print("Running FreeBOT two-module climb test")
    print(f"USD: {usd_path}")
    print(
        "Magnetic model: "
        f"internal=dipole-equivalent, external={args.external_model}, "
        f"Br={args.magnet_br} T, external_Fmax={args.external_max_force} N"
    )
    print(
        "Internal radial alignment: "
        f"stiffness={args.internal_align_stiffness} N*m/rad, "
        f"damping={args.internal_align_damping} N*m/(rad/s), "
        f"max_torque={args.internal_align_max_torque} N*m"
    )
    print(
        "Physical shell contact: "
        f"law=F0/(1+gap/d0)^3, F0={args.external_max_force} N, "
        f"d0={args.external_decay_distance} m, force_target=active_shell_contact"
    )
    if args.external_model == "field-patch":
        print(
            "External field patch: "
            f"cap_angle={args.field_cap_angle_deg} deg, rings={args.field_rings}, "
            f"ring_samples={args.field_ring_samples}, pressure_scale={args.field_pressure_scale}"
        )
    else:
        print(
            "External point model: "
            f"gap_source={args.external_gap_source}, "
            f"effective_offset={args.external_effective_offset} m, "
            f"min_distance={args.external_min_distance} m, "
            f"force_gain={args.external_force_gain}, "
            f"capture_gap={args.external_capture_gap} m, "
            "alignment_gating=disabled"
        )
    print(
        "Materials: "
        f"passive_shell_static={args.sphere_static_friction}, passive_shell_dynamic={args.sphere_dynamic_friction}, "
        f"platform_static={args.platform_static_friction}, platform_dynamic={args.platform_dynamic_friction}"
    )
    print(
        "Mobile magnetic contact: "
        f"enabled={args.magnetic_adhesion}, gap={args.adhesion_gap} m, "
        f"mu={args.adhesion_friction}, Ft_max={args.adhesion_max_force} N, "
        f"stiffness={args.adhesion_tangent_stiffness} N/m, "
        f"damping={args.adhesion_tangent_damping} N/(m/s)"
    )
    print(
        "Dock latch: "
        f"engage_align={args.dock_align_threshold}, release_align={args.dock_release_align_threshold}, "
        f"engage_patch={args.dock_patch_gap} m, release_patch={args.dock_release_patch_gap} m, "
        f"preload={args.dock_min_preload} N, require_neutral={args.dock_require_neutral}, "
        f"release_on_reverse={args.dock_release_on_reverse}"
    )
    print(
        "Dock torsion: "
        f"enabled={args.dock_torsion}, stiffness={args.dock_torsion_stiffness} N*m/rad, "
        f"damping={args.dock_torsion_damping} N*m/(rad/s), "
        f"max_torque={args.dock_torsion_max_torque} N*m"
    )
    print(
        "Teleop scaling: "
        f"linear={args.cmd_linear_scale} deg/s per m/s, angular={args.cmd_angular_scale} deg/s per rad/s"
    )

    teleop = Ros2CliTeleop(args.cmd_vel_topic) if args.ros2_teleop else None
    commanded_left = args.left_wheel_velocity
    commanded_right = args.right_wheel_velocity
    teleop_forward = 0.0
    teleop_turn = 0.0
    previous_external_force = np.zeros(3, dtype=np.float64)
    filtered_external_force_norm = 0.0
    dock_latched = False
    dock_passive_index = None
    dock_waiting_for_neutral = False
    manual_undock_active = False
    dock_torsion_angle = 0.0
    dock_torsion_torque = np.zeros(3, dtype=np.float64)
    previous_adhesion_force = np.zeros(3, dtype=np.float64)
    contact_state = {
        "displacement": np.zeros(3, dtype=np.float64),
        "passive_index": None,
    }
    adhesion_status = "free"
    adhesion_tangent_force_norm = 0.0
    adhesion_tangent_limit = 0.0
    physics_dt = 1.0 / 240.0

    for step in range(args.steps):
        if teleop is not None:
            commanded_left, commanded_right, teleop_forward, teleop_turn = ros2_wheel_command(teleop)

        internal_force, magnet_pos, shell_point = active_internal_magnetic_force(shell_body, internal_body, magnet_frame)
        internal_body.apply_forces_and_torques_at_pos(forces=internal_force, positions=magnet_pos)
        shell_body.apply_forces_and_torques_at_pos(forces=-internal_force, positions=shell_point)
        internal_alignment_torque, internal_alignment_error = active_internal_alignment_torque(
            shell_body,
            internal_body,
            magnet_frame,
        )
        internal_pos_for_alignment = first_vec(internal_body.get_world_poses()[0])
        shell_pos_for_alignment = first_vec(shell_body.get_world_poses()[0])
        internal_body.apply_forces_and_torques_at_pos(
            forces=np.zeros((1, 3), dtype=np.float64),
            torques=internal_alignment_torque.reshape(1, 3),
            positions=internal_pos_for_alignment.reshape(1, 3),
        )
        shell_body.apply_forces_and_torques_at_pos(
            forces=np.zeros((1, 3), dtype=np.float64),
            torques=(-internal_alignment_torque).reshape(1, 3),
            positions=shell_pos_for_alignment.reshape(1, 3),
        )

        raw_external_force, external_pos, external_metrics = active_to_passive_magnetic_force(
            shell_body,
            internal_body,
            magnet_frame,
        )
        axis_alignment = float(external_metrics.get("axis_alignment", 0.0))
        patch_gap = float(external_metrics.get("patch_gap", np.inf))
        shell_gap = float(external_metrics.get("shell_gap", np.inf))
        candidate_passive_index = int(external_metrics.get("passive_index", -1))
        newly_latched = False
        detach_requested = (
            args.dock_release_on_reverse
            and teleop is not None
            and teleop_forward <= -abs(args.dock_detach_command_threshold)
        )
        if dock_latched and detach_requested:
            manual_undock_active = True
            dock_latched = False
            dock_passive_index = None
            dock_waiting_for_neutral = False
            dock_torsion_angle = 0.0

        if manual_undock_active:
            separated = shell_gap > 2.0 * args.adhesion_gap or patch_gap > args.dock_release_patch_gap
            if separated and not detach_requested:
                manual_undock_active = False
            else:
                # Reverse may release only the behavioral latch. The physical
                # normal attraction remains active and decays with separation.
                pass
        elif dock_latched:
            lost_geometry = (
                shell_gap > 2.0 * args.adhesion_gap
                or patch_gap > args.dock_release_patch_gap
                or axis_alignment < args.dock_release_align_threshold
                or candidate_passive_index != dock_passive_index
            )
            if lost_geometry:
                dock_latched = False
                dock_passive_index = None
                dock_waiting_for_neutral = False
                dock_torsion_angle = 0.0
        elif (
            shell_gap <= args.adhesion_gap
            and patch_gap <= args.dock_patch_gap
            and axis_alignment >= args.dock_align_threshold
            and external_metrics.get("capture", 0.0) > 0.0
        ):
            dock_latched = True
            dock_passive_index = candidate_passive_index
            newly_latched = True
        external_metrics["dock_latched"] = dock_latched
        external_metrics["manual_undock"] = manual_undock_active

        if teleop is not None:
            if newly_latched and args.dock_require_neutral:
                dock_waiting_for_neutral = True
                teleop.clear_command()
            if dock_waiting_for_neutral:
                teleop_is_neutral = abs(teleop_forward) < 1.0e-6 and abs(teleop_turn) < 1.0e-6
                commanded_left = 0.0
                commanded_right = 0.0
                if teleop_is_neutral and not newly_latched:
                    dock_waiting_for_neutral = False
            set_wheel_drive(stage, commanded_left, commanded_right, args.wheel_damping, args.wheel_max_force)

        # No temporal ramp or artificial preload: apply the instantaneous
        # magnetic normal load directly to the active shell contact point.
        previous_external_force = raw_external_force
        filtered_external_force_norm = np.linalg.norm(raw_external_force)
        shell_force_pos = np.asarray(
            external_metrics.get("active_shell_point", first_vec(shell_body.get_world_poses()[0])),
            dtype=np.float64,
        )
        shell_body.apply_forces_and_torques_at_pos(
            forces=previous_external_force.reshape(1, 3),
            positions=external_pos.reshape(1, 3),
        )

        raw_adhesion_force, adhesion_pos, adhesion_tangent_force_norm, adhesion_tangent_limit, adhesion_active, adhesion_status = magnetic_adhesion_patch_force(
            internal_body,
            external_pos,
            previous_external_force,
            external_metrics,
            contact_state,
            physics_dt,
        )
        if adhesion_active:
            adhesion_delta = raw_adhesion_force - previous_adhesion_force
            adhesion_is_building = np.linalg.norm(raw_adhesion_force) > np.linalg.norm(previous_adhesion_force)
            adhesion_rate = args.external_force_rate if adhesion_is_building else args.external_force_release_rate
            adhesion_max_delta = adhesion_rate * physics_dt
            previous_adhesion_force = previous_adhesion_force + clamp_vector(adhesion_delta, adhesion_max_delta)
            # The filtered force must never exceed the instantaneous Coulomb
            # bound. Without this projection, release lag can keep the module
            # artificially attached after the normal magnetic load has fallen.
            previous_adhesion_force = clamp_vector(previous_adhesion_force, adhesion_tangent_limit)
        else:
            previous_adhesion_force = np.zeros(3, dtype=np.float64)
        if adhesion_active and np.linalg.norm(previous_adhesion_force) > 1.0e-6:
            internal_body.apply_forces_and_torques_at_pos(
                forces=previous_adhesion_force.reshape(1, 3),
                positions=np.asarray(adhesion_pos, dtype=np.float64).reshape(1, 3),
            )

        dock_torsion_torque = np.zeros(3, dtype=np.float64)
        attraction_value = external_metrics.get("attraction_dir")
        if args.dock_torsion and dock_latched and attraction_value is not None:
            torsion_normal = np.asarray(attraction_value, dtype=np.float64)
            torsion_normal /= max(np.linalg.norm(torsion_normal), 1.0e-9)
            internal_pos_for_torsion = first_vec(internal_body.get_world_poses()[0])
            internal_ang_for_torsion = first_vec(internal_body.get_velocities()[1])
            torsion_rate = float(np.dot(internal_ang_for_torsion, torsion_normal))
            dock_torsion_angle += torsion_rate * physics_dt
            raw_torsion_torque = (
                -args.dock_torsion_stiffness * dock_torsion_angle
                -args.dock_torsion_damping * torsion_rate
            )
            limited_torsion_torque = float(
                np.clip(
                    raw_torsion_torque,
                    -args.dock_torsion_max_torque,
                    +args.dock_torsion_max_torque,
                )
            )
            if abs(raw_torsion_torque) > args.dock_torsion_max_torque and args.dock_torsion_stiffness > 1.0e-9:
                dock_torsion_angle = -(
                    limited_torsion_torque + args.dock_torsion_damping * torsion_rate
                ) / args.dock_torsion_stiffness
            dock_torsion_torque = limited_torsion_torque * torsion_normal
            internal_body.apply_forces_and_torques_at_pos(
                forces=np.zeros((1, 3), dtype=np.float64),
                torques=dock_torsion_torque.reshape(1, 3),
                positions=internal_pos_for_torsion.reshape(1, 3),
            )
        else:
            dock_torsion_angle = 0.0

        SimulationManager.step()
        RenderingManager.render()
        simulation_app.update()

        if args.log_interval > 0 and step % args.log_interval == 0:
            shell_pos = first_vec(shell_body.get_world_poses()[0])
            shell_ang = first_vec(shell_body.get_velocities()[1])
            internal_ang = first_vec(internal_body.get_velocities()[1])
            relative_ang = shell_ang - internal_ang
            passive_centers = passive_shell_centers()
            passive_index = int(external_metrics.get("passive_index", 0))
            passive_index = max(0, min(passive_index, len(passive_centers) - 1))
            passive_center = passive_centers[passive_index]
            rel_passive = shell_pos - passive_center
            climb_height = shell_pos[2] - args.shell_radius
            active_patch = np.asarray(external_metrics.get("active_shell_point", shell_pos), dtype=np.float64)
            patch_rel = active_patch - shell_pos
            internal_pos = first_vec(internal_body.get_world_poses()[0])
            external_torque = np.cross(
                np.asarray(external_pos) - shell_pos,
                previous_external_force,
            )
            adhesion_torque = np.cross(np.asarray(adhesion_pos) - internal_pos, previous_adhesion_force)
            print(
                f"t={step / 240.0:5.2f}s  "
                f"active_shell=({shell_pos[0]:+.3f},{shell_pos[1]:+.3f},{shell_pos[2]:+.3f})  "
                f"omega=({shell_ang[0]:+.2f},{shell_ang[1]:+.2f},{shell_ang[2]:+.2f}) rad/s  "
                f"omega_rel=({relative_ang[0]:+.2f},{relative_ang[1]:+.2f},{relative_ang[2]:+.2f}) rad/s  "
                f"climb_h={climb_height:+.3f} m  "
                f"rel_passive=({rel_passive[0]:+.3f},{rel_passive[1]:+.3f},{rel_passive[2]:+.3f})  "
                f"Fint={np.linalg.norm(internal_force):.2f} N  "
                f"Tint={np.linalg.norm(internal_alignment_torque):.3f} Nm  "
                f"int_align_err={np.degrees(internal_alignment_error):.1f} deg  "
                f"Fmodule={np.linalg.norm(previous_external_force):.2f} N  "
                f"Fmodule_raw={np.linalg.norm(raw_external_force):.2f} N  "
                f"shell_gap={external_metrics.get('shell_gap', external_metrics['gap']) * 1000.0:+.1f} mm  "
                f"shell_patch_gap={external_metrics.get('shell_patch_gap', np.inf) * 1000.0:+.1f} mm  "
                f"patch_gap={external_metrics.get('patch_gap', np.inf) * 1000.0:+.1f} mm  "
                f"mag_z={external_metrics.get('magnetic_distance', external_metrics['gap']) * 1000.0:+.1f} mm  "
                f"module_align={external_metrics['alignment']:.2f}  "
                f"module_capture={external_metrics['capture']:.2f}  "
                f"Fadh_tan={np.linalg.norm(previous_adhesion_force):.2f} N  "
                f"Ftadh={adhesion_tangent_force_norm:.2f}/{adhesion_tangent_limit:.2f} N  "
                f"contact={adhesion_status}  "
                f"dock={'1' if dock_latched else '0'}  "
                f"undock={'1' if manual_undock_active else '0'}  "
                f"drive_armed={'0' if dock_waiting_for_neutral else '1'}  "
                f"patch_rel=({patch_rel[0]:+.3f},{patch_rel[1]:+.3f},{patch_rel[2]:+.3f})  "
                f"Text=({external_torque[0]:+.3f},{external_torque[1]:+.3f},{external_torque[2]:+.3f}) Nm  "
                f"Tmaghold=({adhesion_torque[0]:+.3f},{adhesion_torque[1]:+.3f},{adhesion_torque[2]:+.3f}) Nm  "
                f"Ttorsion=({dock_torsion_torque[0]:+.3f},{dock_torsion_torque[1]:+.3f},{dock_torsion_torque[2]:+.3f}) Nm  "
                f"passive_idx={passive_index}  "
                f"Bpeak={external_metrics.get('peak_b', 0.0):.3f} T  "
                f"field_samples={external_metrics.get('samples', 0)}  "
                f"wheel_cmd=({commanded_left:+.0f},{commanded_right:+.0f})  "
                f"cmd_vel=({teleop_forward:+.0f},{teleop_turn:+.0f})"
            )

    if teleop is not None:
        teleop.close()
    app_utils.stop()
    simulation_app.close()


if __name__ == "__main__":
    main()
