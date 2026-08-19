from __future__ import annotations

import math
from typing import Any


class IsaacGroundSupportAnchor:
    """Runtime anti-tip support for exaggerated cantilever payload tests."""

    def __init__(
        self,
        stage: Any,
        body_path: str,
        joint_path: str = "/World/SmoresEPDockingJoints/active_ground_support",
        yaw_max_effort_nm: float = 6.9,
        yaw_damping_nm_s_per_rad: float = 12.0,
    ) -> None:
        if (
            not math.isfinite(yaw_max_effort_nm)
            or yaw_max_effort_nm <= 0.0
            or not math.isfinite(yaw_damping_nm_s_per_rad)
            or yaw_damping_nm_s_per_rad <= 0.0
        ):
            raise ValueError("Yaw-assist effort and damping must be positive")
        self._stage = stage
        self._body_path = body_path
        self._joint_path = joint_path
        self._yaw_max_effort_nm = yaw_max_effort_nm
        self._yaw_damping_nm_s_per_rad = yaw_damping_nm_s_per_rad
        if not stage.GetPrimAtPath(body_path):
            raise RuntimeError(f"Ground-support body does not exist: {body_path}")

    @property
    def engaged(self) -> bool:
        return bool(self._stage.GetPrimAtPath(self._joint_path))

    def engage(self) -> bool:
        """Constrain roll and pitch while leaving translation and yaw free."""

        if self.engaged:
            return False

        from pxr import Gf, Sdf, UsdGeom, UsdPhysics

        body = self._stage.GetPrimAtPath(self._body_path)
        body_world = UsdGeom.XformCache().GetLocalToWorldTransform(body)
        world_transform = Gf.Transform(body_world)

        # A generic USD joint becomes a PhysX D6 joint. Roll and pitch are
        # locked only after the robot has settled on both wheels. All
        # translations and yaw remain free.
        joint = UsdPhysics.Joint.Define(
            self._stage,
            self._joint_path,
        )
        joint.CreateBody0Rel().SetTargets([Sdf.Path(self._body_path)])
        # An omitted body1 relationship means the static world body.
        joint.CreateCollisionEnabledAttr(False)
        joint.CreateExcludeFromArticulationAttr(True)
        joint.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        joint.CreateLocalPos1Attr(
            Gf.Vec3f(world_transform.GetTranslation())
        )
        joint.CreateLocalRot1Attr(
            Gf.Quatf(world_transform.GetRotation().GetQuat())
        )
        for axis in (
            UsdPhysics.Tokens.rotX,
            UsdPhysics.Tokens.rotY,
        ):
            limit = UsdPhysics.LimitAPI.Apply(joint.GetPrim(), axis)
            # USD Physics encodes a locked D6 axis with low > high.
            limit.CreateLowAttr(1.0)
            limit.CreateHighAttr(-1.0)

        yaw_drive = UsdPhysics.DriveAPI.Apply(
            joint.GetPrim(),
            UsdPhysics.Tokens.rotZ,
        )
        yaw_drive.CreateTypeAttr(UsdPhysics.Tokens.force)
        yaw_drive.CreateStiffnessAttr(0.0)
        yaw_drive.CreateDampingAttr(self._yaw_damping_nm_s_per_rad)
        yaw_drive.CreateMaxForceAttr(self._yaw_max_effort_nm)
        yaw_drive.CreateTargetVelocityAttr(0.0)
        return True

    def set_yaw_velocity_target(self, angular_z_rad_s: float) -> bool:
        """Set the D6 yaw drive without overwriting the body's pose."""

        if not math.isfinite(angular_z_rad_s):
            raise ValueError("Yaw velocity target must be finite")
        if not self.engaged:
            return False

        from pxr import UsdPhysics

        drive = UsdPhysics.DriveAPI(
            self._stage.GetPrimAtPath(self._joint_path),
            UsdPhysics.Tokens.rotZ,
        )
        # body0-to-world D6 angular velocity has the opposite sign to ROS yaw.
        drive.GetTargetVelocityAttr().Set(-math.degrees(angular_z_rad_s))
        return True

    def release(self) -> bool:
        if not self.engaged:
            return False
        return bool(self._stage.RemovePrim(self._joint_path))
