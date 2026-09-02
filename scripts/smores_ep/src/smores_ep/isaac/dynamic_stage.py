from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from smores_ep.config.geometry import SmoresGeometry
from smores_ep.config.physics import SmoresActuatorConfig
from smores_ep.control.differential_drive import twist_to_wheel_rates
from smores_ep.control.pan_tilt import (
    ContinuousAngleTracker,
    continuous_position_servo_velocity,
    normalize_revolute_target,
)
from smores_ep.control.teleop import InternalMotionMode, SmoresCommand
from smores_ep.isaac.physics_asset import PHYSICS_ROOT


JOINTS_ROOT = f"{PHYSICS_ROOT}/joints"
BODY_PATH = f"{PHYSICS_ROOT}/body_link"
MECHANISM_VISUALS_PATH = f"{BODY_PATH}/mechanism_visuals"


@dataclass(frozen=True)
class DynamicJointState:
    left_wheel_rad: float
    right_wheel_rad: float
    left_wheel_rad_s: float
    right_wheel_rad_s: float
    tilt_joint_rad: float
    tilt_joint_rad_s: float
    pan_joint_rad: float
    pan_joint_rad_s: float


class DynamicDriveController:
    """Write ROS-level commands to the four physical joint drives."""

    def __init__(
        self,
        articulation: ArticulationStateReader,
        geometry: SmoresGeometry,
        max_wheel_speed_rad_s: float,
    ) -> None:
        self._articulation = articulation
        self._geometry = geometry
        self._max_wheel_speed_rad_s = max_wheel_speed_rad_s
        self._pan_angle = ContinuousAngleTracker()
        self._pan_servo_gain_s = 4.0
        initial = self._articulation.read()
        self._pan_position_rad = self._pan_angle.update(
            initial.pan_joint_rad
        )
        self._pan_target_rad = self._pan_position_rad
        self._tilt_target_rad = -initial.tilt_joint_rad

    def apply(self, command: SmoresCommand) -> tuple[float, float]:
        rates = twist_to_wheel_rates(
            command.linear_x_m_s,
            command.angular_z_rad_s,
            self._geometry.wheel_radius_m,
            self._geometry.track_width_m,
        )
        left = max(
            -self._max_wheel_speed_rad_s,
            min(self._max_wheel_speed_rad_s, rates.left_rad_s),
        )
        right = max(
            -self._max_wheel_speed_rad_s,
            min(self._max_wheel_speed_rad_s, rates.right_rad_s),
        )
        self._pan_position_rad = self._pan_angle.update(
            self._articulation.read().pan_joint_rad
        )
        if command.internal_motion is InternalMotionMode.PAN:
            self._pan_target_rad = command.pan_target_rad
            # A steering operation belongs to the operational phase: PAN is
            # allowed to move, while TILT remains at the structural target.
            self._tilt_target_rad = max(
                self._geometry.tilt_min_rad,
                min(
                    self._geometry.tilt_max_rad,
                    command.tilt_target_rad,
                ),
            )
        elif command.internal_motion is InternalMotionMode.TILT:
            self._tilt_target_rad = max(
                self._geometry.tilt_min_rad,
                min(
                    self._geometry.tilt_max_rad,
                    command.tilt_target_rad,
                ),
            )
        elif command.internal_motion is InternalMotionMode.STRUCTURAL_HOLD:
            # Structural HOLD is deliberately different from PASSIVE.  It
            # captures both coordinates reached by a backdriven folding
            # mechanism and keeps that configuration after the pushing wheels
            # stop.
            self._pan_target_rad = command.pan_target_rad
            self._tilt_target_rad = max(
                self._geometry.tilt_min_rad,
                min(
                    self._geometry.tilt_max_rad,
                    command.tilt_target_rad,
                ),
            )
        if command.internal_motion is InternalMotionMode.PAN_VELOCITY:
            self._tilt_target_rad = max(
                self._geometry.tilt_min_rad,
                min(
                    self._geometry.tilt_max_rad,
                    command.tilt_target_rad,
                ),
            )
            pan_velocity = max(
                -self._max_wheel_speed_rad_s,
                min(
                    self._max_wheel_speed_rad_s,
                    command.pan_velocity_rad_s,
                ),
            )
            # Retain the measured position so a later explicit hold starts
            # from the released joint angle instead of snapping it back.
            self._pan_target_rad = self._pan_position_rad
        else:
            pan_velocity = continuous_position_servo_velocity(
                self._pan_position_rad,
                self._pan_target_rad,
                self._pan_servo_gain_s,
                self._max_wheel_speed_rad_s,
            )
        self._articulation.set_targets(
            left_wheel_velocity_rad_s=left,
            right_wheel_velocity_rad_s=right,
            # Positive ROS tilt raises TOP, while the USD revolute axis has
            # the opposite sign.
            tilt_joint_position_rad=-self._tilt_target_rad,
            pan_joint_velocity_rad_s=pan_velocity,
            pan_logical_target_rad=self._pan_target_rad,
        )
        return left, right

    @property
    def pan_position_rad(self) -> float:
        return self._pan_position_rad

    @property
    def internal_targets_rad(self) -> tuple[float, float]:
        """Return retained logical PAN and TILT targets."""

        return self._pan_target_rad, self._tilt_target_rad


class ArticulationStateReader:
    """Read actual PhysX joint coordinates, not commanded targets."""

    def __init__(
        self,
        module_root: str = PHYSICS_ROOT,
        actuators: SmoresActuatorConfig | None = None,
        stage: Any | None = None,
    ) -> None:
        from isaacsim.core.experimental.prims import Articulation

        self.module_root = module_root
        self._stage = stage
        self._wheel_contact_mode: str | None = None
        self._pan_contact_mode: str | None = None
        drive = actuators or SmoresActuatorConfig()
        self._drive = drive
        self._articulation = Articulation(module_root)
        paths = self._articulation.dof_paths[0]
        self.dof_paths = tuple(paths)
        joints_root = f"{module_root}/joints"
        self._indices = {
            name: paths.index(f"{joints_root}/{name}_joint")
            for name in ("left_wheel", "right_wheel", "tilt", "pan")
        }
        self._position_targets = {"tilt": 0.0, "pan": 0.0}
        wheel_indices = [
            self._indices["left_wheel"],
            self._indices["right_wheel"],
        ]
        internal_indices = [
            self._indices["tilt"],
            self._indices["pan"],
        ]
        # Use PhysX's implicit PD drives. They are integrated with the
        # constraint solver and remain stable during ground impacts; a
        # hand-written explicit derivative injected maximum torque when an
        # impact briefly produced a large joint velocity.
        self._articulation.set_dof_gains(
            stiffnesses=[0.0, 0.0],
            dampings=[
                drive.wheel_damping_nm_s_per_rad,
                drive.wheel_damping_nm_s_per_rad,
            ],
            dof_indices=wheel_indices,
        )
        self._articulation.set_dof_max_efforts(
            [
                drive.wheel_max_effort_nm,
                drive.wheel_max_effort_nm,
            ],
            dof_indices=wheel_indices,
        )
        self._articulation.set_dof_max_velocities(
            [drive.wheel_max_speed_rad_s] * 2,
            dof_indices=wheel_indices,
        )
        self._articulation.set_dof_gains(
            # Both shape coordinates are load-bearing.  PAN remains
            # continuous, but its unwrapped target is still a valid PhysX
            # position target and must resist out-of-plane gravity loads.
            stiffnesses=[
                drive.tilt_stiffness_nm_per_rad,
                drive.hold_stiffness_nm_per_rad,
            ],
            dampings=[
                drive.tilt_damping_nm_s_per_rad,
                drive.pan_damping_nm_s_per_rad,
            ],
            dof_indices=internal_indices,
        )
        self._articulation.set_dof_max_efforts(
            [drive.tilt_max_effort_nm, drive.pan_max_effort_nm],
            dof_indices=internal_indices,
        )
        self._articulation.set_dof_max_velocities(
            [drive.internal_max_speed_rad_s] * 2,
            dof_indices=internal_indices,
        )

    def read(self) -> DynamicJointState:
        positions = self._articulation.get_dof_positions().numpy()[0]
        velocities = self._articulation.get_dof_velocities().numpy()[0]

        def position(name: str) -> float:
            return float(positions[self._indices[name]])

        return DynamicJointState(
            left_wheel_rad=position("left_wheel"),
            right_wheel_rad=position("right_wheel"),
            left_wheel_rad_s=float(
                velocities[self._indices["left_wheel"]]
            ),
            right_wheel_rad_s=float(
                velocities[self._indices["right_wheel"]]
            ),
            tilt_joint_rad=position("tilt"),
            tilt_joint_rad_s=float(velocities[self._indices["tilt"]]),
            pan_joint_rad=position("pan"),
            pan_joint_rad_s=float(velocities[self._indices["pan"]]),
        )

    def configure_structural_hold_mode(
        self,
        docked_face_names: set[str] | frozenset[str],
    ) -> None:
        """Hold the assembled structure while leaving towing wheels free.

        TILT and PAN retain the completed posture. LEFT and RIGHT are held
        only while that specific rotating face is connected.
        A BOTTOM connection belongs directly to body_link and therefore needs
        no wheel brake.
        """

        normalized_faces = {name.upper() for name in docked_face_names}
        unknown = normalized_faces - {"LEFT", "RIGHT", "TOP", "BOTTOM"}
        if unknown:
            raise ValueError(
                "Unknown structural docking faces: "
                + ", ".join(sorted(unknown))
            )
        # Retained modules are load-bearing parts of the completed morphology.
        # Keep normal tire contact instead of turning their docked wheels into
        # low-friction skids.
        self._set_wheel_contact_mode("wheel")
        # All four coordinates retain a deliberate structure.  PAN is
        # continuous, so DynamicDriveController supplies its unwrapped target
        # on every frame; the implicit position drive provides the stiffness
        # that the former velocity-only servo could not provide under load.
        position_names = ("left_wheel", "right_wheel", "tilt", "pan")
        position_indices = [
            self._indices[name] for name in position_names
        ]
        positions = self._articulation.get_dof_positions().numpy()[0]
        position_targets = [
            float(positions[index]) for index in position_indices
        ]
        drive = self._drive
        self._articulation.set_dof_gains(
            stiffnesses=[drive.hold_stiffness_nm_per_rad] * 4,
            dampings=[drive.hold_damping_nm_s_per_rad] * 4,
            dof_indices=position_indices,
        )
        self._articulation.set_dof_max_efforts(
            [
                drive.wheel_max_effort_nm,
                drive.wheel_max_effort_nm,
                drive.tilt_max_effort_nm,
                drive.pan_max_effort_nm,
            ],
            dof_indices=position_indices,
        )
        self._articulation.set_dof_position_targets(
            position_targets,
            dof_indices=position_indices,
        )
        self._articulation.set_dof_velocity_targets(
            [0.0] * 4,
            dof_indices=position_indices,
        )

        free_wheel_names = [
            name
            for name, face in (
                ("left_wheel", "LEFT"),
                ("right_wheel", "RIGHT"),
            )
            if face not in normalized_faces
        ]
        if free_wheel_names:
            free_indices = [
                self._indices[name] for name in free_wheel_names
            ]
            self._articulation.set_dof_gains(
                stiffnesses=[0.0] * len(free_indices),
                # Small bearing/gear drag, not an active zero-speed brake.
                dampings=[0.02] * len(free_indices),
                dof_indices=free_indices,
            )
            self._articulation.set_dof_velocity_targets(
                [0.0] * len(free_indices),
                dof_indices=free_indices,
            )

    def configure_fully_passive_mode(self) -> None:
        """Make every unselected joint backdrivable.

        Docking remains a rigid magnetic constraint between the selected face
        links, but no motor drive resists external wheel, PAN or TILT motion.
        A small uniform damping represents bearing drag without acting as a
        zero-speed brake or a retained position target.
        """

        self._set_wheel_contact_mode("wheel")
        names = ("left_wheel", "right_wheel", "tilt", "pan")
        indices = [self._indices[name] for name in names]
        self._articulation.set_dof_gains(
            stiffnesses=[0.0] * len(indices),
            dampings=[0.02] * len(indices),
            dof_indices=indices,
        )
        self._articulation.set_dof_velocity_targets(
            [0.0] * len(indices),
            dof_indices=indices,
        )

    def configure_wheel_drive_with_passive_internals(self) -> None:
        """Energize the two wheels while PAN and TILT remain backdrivable."""

        self._set_wheel_contact_mode("wheel")
        drive = self._drive
        wheel_indices = [
            self._indices["left_wheel"],
            self._indices["right_wheel"],
        ]
        internal_indices = [
            self._indices["tilt"],
            self._indices["pan"],
        ]
        self._articulation.set_dof_gains(
            stiffnesses=[0.0, 0.0],
            dampings=[
                drive.wheel_damping_nm_s_per_rad,
                drive.wheel_damping_nm_s_per_rad,
            ],
            dof_indices=wheel_indices,
        )
        self._articulation.set_dof_max_efforts(
            [drive.wheel_max_effort_nm, drive.wheel_max_effort_nm],
            dof_indices=wheel_indices,
        )
        self._articulation.set_dof_max_velocities(
            [drive.wheel_max_speed_rad_s] * 2,
            dof_indices=wheel_indices,
        )
        self._articulation.set_dof_gains(
            stiffnesses=[0.0, 0.0],
            dampings=[0.02, 0.02],
            dof_indices=internal_indices,
        )
        self._articulation.set_dof_velocity_targets(
            [0.0, 0.0],
            dof_indices=internal_indices,
        )

    def configure_controlled_docking_mode(
        self,
        docked_face_names: set[str] | frozenset[str],
    ) -> None:
        """Enable wheel and internal drives without connector-face braking."""

        self._set_wheel_contact_mode("wheel")
        normalized_faces = {name.upper() for name in docked_face_names}
        unknown = normalized_faces - {"LEFT", "RIGHT", "TOP", "BOTTOM"}
        if unknown:
            raise ValueError(
                "Unknown controlled docking faces: "
                + ", ".join(sorted(unknown))
            )
        drive = self._drive
        wheel_indices = [
            self._indices["left_wheel"],
            self._indices["right_wheel"],
        ]
        self._articulation.set_dof_gains(
            stiffnesses=[0.0, 0.0],
            dampings=[
                drive.wheel_damping_nm_s_per_rad,
                drive.wheel_damping_nm_s_per_rad,
            ],
            dof_indices=wheel_indices,
        )
        self._articulation.set_dof_max_efforts(
            [drive.wheel_max_effort_nm, drive.wheel_max_effort_nm],
            dof_indices=wheel_indices,
        )
        self._articulation.set_dof_max_velocities(
            [drive.wheel_max_speed_rad_s] * 2,
            dof_indices=wheel_indices,
        )
        self._articulation.set_dof_velocity_targets(
            [0.0, 0.0],
            dof_indices=wheel_indices,
        )
        internal_indices = [
            self._indices["tilt"],
            self._indices["pan"],
        ]
        self._articulation.set_dof_gains(
            stiffnesses=[
                drive.tilt_stiffness_nm_per_rad,
                drive.hold_stiffness_nm_per_rad,
            ],
            dampings=[
                drive.tilt_damping_nm_s_per_rad,
                drive.pan_damping_nm_s_per_rad,
            ],
            dof_indices=internal_indices,
        )
        self._articulation.set_dof_max_efforts(
            [drive.tilt_max_effort_nm, drive.pan_max_effort_nm],
            dof_indices=internal_indices,
        )
        self._articulation.set_dof_max_velocities(
            [drive.internal_max_speed_rad_s] * 2,
            dof_indices=internal_indices,
        )
        positions = self._articulation.get_dof_positions().numpy()[0]
        self._articulation.set_dof_position_targets(
            [float(positions[index]) for index in internal_indices],
            dof_indices=internal_indices,
        )

    def configure_internal_drive_with_braked_wheels(
        self,
        docked_face_names: set[str] | frozenset[str],
    ) -> None:
        """Energize PAN/TILT while the two ground wheels provide reaction.

        PAN/TILT motion is an internal shape change, not free-space motion of
        an isolated link.  Leaving both coaxial wheels backdrivable lets the
        equal-and-opposite actuator torque roll the module body instead of
        moving the TOP-side subtree.  That failure mode is especially severe
        in a TOP--BOTTOM serial morphology: every successful relative TILT
        target can leave another chassis standing on its side.

        Capture the current wheel coordinates and use the high-gain holding
        profile for the duration of the internal motion.  A command that also
        requests locomotion is routed through ``controlled`` instead, so this
        brake does not prevent the train/vehicle wheels from driving.
        """

        self.configure_controlled_docking_mode(docked_face_names)
        wheel_indices = [
            self._indices["left_wheel"],
            self._indices["right_wheel"],
        ]
        positions = self._articulation.get_dof_positions().numpy()[0]
        wheel_targets = [float(positions[index]) for index in wheel_indices]
        drive = self._drive
        self._articulation.set_dof_gains(
            stiffnesses=[drive.hold_stiffness_nm_per_rad] * 2,
            dampings=[drive.hold_damping_nm_s_per_rad] * 2,
            dof_indices=wheel_indices,
        )
        self._articulation.set_dof_max_efforts(
            [drive.wheel_max_effort_nm] * 2,
            dof_indices=wheel_indices,
        )
        self._articulation.set_dof_position_targets(
            wheel_targets,
            dof_indices=wheel_indices,
        )
        self._articulation.set_dof_velocity_targets(
            [0.0, 0.0],
            dof_indices=wheel_indices,
        )

    def configure_pan_velocity_drive_with_braked_wheels(
        self,
        docked_face_names: set[str] | frozenset[str],
    ) -> None:
        """Allow deliberate PAN-rate motion while every support stays rigid."""

        self.configure_internal_drive_with_braked_wheels(docked_face_names)
        pan_index = self._indices["pan"]
        drive = self._drive
        self._articulation.set_dof_gains(
            stiffnesses=[0.0],
            dampings=[drive.pan_damping_nm_s_per_rad],
            dof_indices=[pan_index],
        )
        self._articulation.set_dof_velocity_targets(
            [0.0],
            dof_indices=[pan_index],
        )

    def _set_wheel_contact_mode(self, material_name: str) -> None:
        """Switch operational supports between tire grip and passive skid."""

        if material_name not in {"wheel", "passive_skid"}:
            raise ValueError(f"Unknown wheel contact material {material_name}")
        if self._stage is None or self._wheel_contact_mode == material_name:
            return

        from pxr import UsdShade

        material_prim = self._stage.GetPrimAtPath(
            f"{self.module_root}/materials/{material_name}"
        )
        if not material_prim:
            raise RuntimeError(
                f"Missing {material_name} material for {self.module_root}"
            )
        material = UsdShade.Material(material_prim)
        for link_name in ("left_wheel_link", "right_wheel_link"):
            tire = self._stage.GetPrimAtPath(
                f"{self.module_root}/{link_name}/colliders/tire"
            )
            if not tire:
                raise RuntimeError(
                    f"Missing wheel collider below {self.module_root}"
                )
            UsdShade.MaterialBindingAPI.Apply(tire).Bind(
                material,
                UsdShade.Tokens.weakerThanDescendants,
                "physics",
            )
        self._wheel_contact_mode = material_name


    def set_pan_contact_mode(self, material_name: str) -> None:
        """Select normal PAN contact or RC-Car8 traction contact."""

        if material_name not in {"pan_face", "wheel"}:
            raise ValueError(
                f"Unknown PAN contact material {material_name}"
            )
        if self._stage is None or self._pan_contact_mode == material_name:
            return

        from pxr import PhysxSchema, UsdPhysics, UsdShade

        # "wheel" here means that the PAN is acting as an RC-Car8 tire.
        # Do NOT modify the ordinary wheel material: create a dedicated
        # high-traction material local to this module instead.
        if material_name == "wheel":
            actual_material_name = "rc_car_pan_traction"
            material_path = (
                f"{self.module_root}/materials/"
                f"{actual_material_name}"
            )

            material_prim = self._stage.GetPrimAtPath(material_path)

            if not material_prim:
                material = UsdShade.Material.Define(
                    self._stage,
                    material_path,
                )
                material_prim = material.GetPrim()

                physics = UsdPhysics.MaterialAPI.Apply(material_prim)
                physics.CreateStaticFrictionAttr(1.20)
                physics.CreateDynamicFrictionAttr(1.00)
                physics.CreateRestitutionAttr(0.0)

                physx = PhysxSchema.PhysxMaterialAPI.Apply(
                    material_prim
                )
                physx.CreateFrictionCombineModeAttr("multiply")
            else:
                material = UsdShade.Material(material_prim)

        else:
            material_path = (
                f"{self.module_root}/materials/pan_face"
            )
            material_prim = self._stage.GetPrimAtPath(material_path)

            if not material_prim:
                raise RuntimeError(
                    f"Missing pan_face material for {self.module_root}"
                )

            material = UsdShade.Material(material_prim)

        pan_face = self._stage.GetPrimAtPath(
            f"{self.module_root}/pan_link/colliders/face"
        )
        if not pan_face:
            raise RuntimeError(
                f"Missing PAN face collider below {self.module_root}"
            )

        UsdShade.MaterialBindingAPI.Apply(pan_face).Bind(
            material,
            UsdShade.Tokens.weakerThanDescendants,
            "physics",
        )

        self._pan_contact_mode = material_name

    def set_targets(
        self,
        left_wheel_velocity_rad_s: float,
        right_wheel_velocity_rad_s: float,
        tilt_joint_position_rad: float,
        pan_joint_velocity_rad_s: float,
        pan_logical_target_rad: float,
    ) -> None:
        self._articulation.set_dof_velocity_targets(
            [
                left_wheel_velocity_rad_s,
                right_wheel_velocity_rad_s,
                pan_joint_velocity_rad_s,
            ],
            dof_indices=[
                self._indices["left_wheel"],
                self._indices["right_wheel"],
                self._indices["pan"],
            ],
        )
        self._position_targets["tilt"] = tilt_joint_position_rad
        self._position_targets["pan"] = pan_logical_target_rad
        self._articulation.set_dof_position_targets(
            [
                tilt_joint_position_rad,
                normalize_revolute_target(pan_logical_target_rad),
            ],
            dof_indices=[self._indices["tilt"], self._indices["pan"]],
        )

    def target_positions(self) -> tuple[float, float]:
        return (
            self._position_targets["tilt"],
            self._position_targets["pan"],
        )

    def physx_drive_targets(self) -> tuple[float, float]:
        positions = self._articulation.get_dof_position_targets().numpy()[0]
        velocities = self._articulation.get_dof_velocity_targets().numpy()[0]
        return (
            float(positions[self._indices["tilt"]]),
            float(velocities[self._indices["pan"]]),
        )


class MechanismVisualController:
    """Animate non-rigid gear visuals from measured articulation state."""

    def __init__(
        self,
        stage: Any,
        ratio: float,
        module_root: str = PHYSICS_ROOT,
    ) -> None:
        self._ratio = ratio
        mechanism_visuals_path = (
            f"{module_root}/body_link/mechanism_visuals"
        )

        def rotation(name: str) -> Any:
            path = f"{mechanism_visuals_path}/{name}"
            prim = stage.GetPrimAtPath(path)
            if not prim:
                raise RuntimeError(f"Missing mechanism visual: {path}")
            attribute = prim.GetAttribute("xformOp:rotateY")
            if not attribute:
                raise RuntimeError(f"Missing rotateY operation: {path}")
            return attribute

        self._outer_left_pinion = rotation("outer_left_pinion")
        self._outer_right_pinion = rotation("outer_right_pinion")
        self._inner_left_gear = rotation("inner_left_gear")
        self._inner_right_gear = rotation("inner_right_gear")
        self._inner_left_pinion = rotation("inner_left_pinion")
        self._inner_right_pinion = rotation("inner_right_pinion")

    def update(self, state: DynamicJointState) -> None:
        degrees = 180.0 / math.pi
        self._outer_left_pinion.Set(
            -self._ratio * state.left_wheel_rad * degrees
        )
        self._outer_right_pinion.Set(
            -self._ratio * state.right_wheel_rad * degrees
        )

        # q_tilt has the opposite sign to the public tilt command.
        inner_left = -state.tilt_joint_rad + state.pan_joint_rad
        inner_right = -state.tilt_joint_rad - state.pan_joint_rad
        self._inner_left_gear.Set(inner_left * degrees)
        self._inner_right_gear.Set(inner_right * degrees)
        self._inner_left_pinion.Set(-self._ratio * inner_left * degrees)
        self._inner_right_pinion.Set(-self._ratio * inner_right * degrees)


def configure_dynamic_stage(
    stage: Any,
    spawn_height_m: float,
    initial_pitch_deg: float,
    module_root: str = PHYSICS_ROOT,
) -> None:
    """Add gravity, a collidable floor, lighting, and a free initial pose."""

    from pxr import (
        Gf,
        PhysxSchema,
        Sdf,
        Usd,
        UsdGeom,
        UsdLux,
        UsdPhysics,
        UsdShade,
    )

    scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr(9.81)
    scene.GetPrim().CreateAttribute(
        "physxScene:enableCCD",
        Sdf.ValueTypeNames.Bool,
    ).Set(True)

    material = UsdShade.Material.Define(
        stage,
        "/World/materials/dynamic_ground",
    )
    physics_material = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics_material.CreateStaticFrictionAttr(0.9)
    physics_material.CreateDynamicFrictionAttr(0.75)
    physics_material.CreateRestitutionAttr(0.0)
    physx_ground_material = PhysxSchema.PhysxMaterialAPI.Apply(
        material.GetPrim()
    )
    physx_ground_material.CreateFrictionCombineModeAttr("multiply")

    ground = UsdGeom.Cube.Define(stage, "/World/Ground")
    ground.CreateSizeAttr(1.0)
    ground.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -0.01))
    ground.AddScaleOp().Set(Gf.Vec3f(4.0, 4.0, 0.02))
    ground.CreateDisplayColorAttr([Gf.Vec3f(0.18, 0.20, 0.22)])
    UsdPhysics.CollisionAPI.Apply(ground.GetPrim())
    UsdShade.MaterialBindingAPI.Apply(ground.GetPrim()).Bind(
        material,
        UsdShade.Tokens.weakerThanDescendants,
        "physics",
    )

    module = UsdGeom.Xformable(stage.GetPrimAtPath(module_root))
    if not module:
        raise RuntimeError(f"Missing articulation root: {module_root}")
    module.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, spawn_height_m))
    module.AddRotateYOp().Set(initial_pitch_deg)

    articulation = stage.GetPrimAtPath(module_root)
    physx_articulation = PhysxSchema.PhysxArticulationAPI.Apply(articulation)
    physx_articulation.CreateEnabledSelfCollisionsAttr(False)
    physx_articulation.CreateSolverPositionIterationCountAttr(32)
    physx_articulation.CreateSolverVelocityIterationCountAttr(8)
    for prim in Usd.PrimRange(articulation):
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        prim.CreateAttribute(
            "physxRigidBody:solverPositionIterationCount",
            Sdf.ValueTypeNames.Int,
        ).Set(32)
        prim.CreateAttribute(
            "physxRigidBody:solverVelocityIterationCount",
            Sdf.ValueTypeNames.Int,
        ).Set(8)
        prim.CreateAttribute(
            "physxRigidBody:maxDepenetrationVelocity",
            Sdf.ValueTypeNames.Float,
        ).Set(1.0)
        prim.CreateAttribute(
            "physxRigidBody:linearDamping",
            Sdf.ValueTypeNames.Float,
        ).Set(0.02)
        prim.CreateAttribute(
            "physxRigidBody:angularDamping",
            Sdf.ValueTypeNames.Float,
        ).Set(0.02)

    sun = UsdLux.DistantLight.Define(stage, "/World/Environment/Sun")
    sun.CreateIntensityAttr(3000.0)
    sun.CreateAngleAttr(0.5)
    sun.AddRotateXYZOp().Set(Gf.Vec3f(315.0, 0.0, 35.0))
    dome = UsdLux.DomeLight.Define(stage, "/World/Environment/Sky")
    dome.CreateIntensityAttr(500.0)
