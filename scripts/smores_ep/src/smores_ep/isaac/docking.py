from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

from smores_ep.docking.model import (
    DockingCommand,
    DockingFace,
    DockingFacePose,
    DockingPairEvaluation,
    DockingThresholds,
    evaluate_face_pair,
    select_best_face_pair,
)


@dataclass(frozen=True)
class DockingConnection:
    joint_path: str
    first_face: DockingFace
    second_face: DockingFace

    @property
    def module_pair(self) -> frozenset[str]:
        return frozenset(
            (self.first_face.module_id, self.second_face.module_id)
        )


@dataclass(frozen=True)
class DockingResult:
    accepted: bool
    message: str
    connection: DockingConnection | None = None


class IsaacDockingManager:
    """Discover SMORES-EP faces and manage runtime fixed joints."""

    def __init__(
        self,
        stage: Any,
        module_roots: Mapping[str, str],
        thresholds: DockingThresholds | None = None,
        joints_root: str = "/World/SmoresEPDockingJoints",
    ) -> None:
        if len(module_roots) < 2:
            raise ValueError("Docking requires at least two registered modules")
        if len(set(module_roots.values())) != len(module_roots):
            raise ValueError("Every module ID must map to a distinct USD root")
        self._stage = stage
        self._module_roots = dict(module_roots)
        self._thresholds = thresholds or DockingThresholds()
        self._joints_root = joints_root
        self._faces = self._discover_faces()
        self._connections: dict[frozenset[str], DockingConnection] = {}
        self._joint_serial = 0

        from pxr import UsdGeom

        UsdGeom.Scope.Define(stage, joints_root)

    @property
    def module_ids(self) -> tuple[str, ...]:
        return tuple(self._module_roots)

    @property
    def thresholds(self) -> DockingThresholds:
        """Return the contact gate shared with primitive execution."""

        return self._thresholds

    @property
    def connections(self) -> tuple[DockingConnection, ...]:
        return tuple(self._connections.values())

    @property
    def occupied_faces(self) -> frozenset[tuple[str, str]]:
        return frozenset(
            face.key
            for connection in self._connections.values()
            for face in (connection.first_face, connection.second_face)
        )

    def faces_for(self, module_id: str) -> tuple[DockingFace, ...]:
        self._require_module(module_id)
        return self._faces[module_id]

    def face_poses_for(self, module_id: str) -> tuple[DockingFacePose, ...]:
        return tuple(
            self._read_face_pose(face)
            for face in self.faces_for(module_id)
        )

    def handle_text(self, text: str) -> DockingResult:
        try:
            command = DockingCommand.parse(text)
        except ValueError as error:
            return DockingResult(False, f"invalid command: {error}")
        return self.handle(command)

    def handle(
        self,
        command: DockingCommand,
        thresholds: DockingThresholds | None = None,
        snap_to_nominal: bool = False,
    ) -> DockingResult:
        try:
            self._require_module(command.first_module)
            self._require_module(command.second_module)
        except KeyError as error:
            return DockingResult(False, str(error))
        if command.action == "attach":
            return self._attach(
                command,
                thresholds or self._thresholds,
                snap_to_nominal=snap_to_nominal,
            )
        return self._detach(command)

    def _require_module(self, module_id: str) -> None:
        if module_id not in self._module_roots:
            available = ", ".join(self._module_roots)
            raise KeyError(
                f"unknown module '{module_id}'; available: {available}"
            )

    def _discover_faces(self) -> dict[str, tuple[DockingFace, ...]]:
        from pxr import Usd, UsdPhysics

        discovered: dict[str, tuple[DockingFace, ...]] = {}
        for module_id, root_path in self._module_roots.items():
            root = self._stage.GetPrimAtPath(root_path)
            if not root:
                raise RuntimeError(
                    f"Registered module root does not exist: {root_path}"
                )
            faces = []
            for prim in Usd.PrimRange(root):
                attribute = prim.GetAttribute("smores:face")
                if not attribute:
                    continue
                rigid_body = prim.GetParent()
                while rigid_body and not rigid_body.HasAPI(
                    UsdPhysics.RigidBodyAPI
                ):
                    rigid_body = rigid_body.GetParent()
                if not rigid_body:
                    raise RuntimeError(
                        f"Docking frame has no rigid-body ancestor: "
                        f"{prim.GetPath()}"
                    )
                faces.append(
                    DockingFace(
                        module_id=module_id,
                        face_name=str(attribute.Get()),
                        frame_path=str(prim.GetPath()),
                        rigid_body_path=str(rigid_body.GetPath()),
                    )
                )
            if {face.face_name for face in faces} != {
                "LEFT",
                "RIGHT",
                "TOP",
                "BOTTOM",
            }:
                found = ", ".join(sorted(face.face_name for face in faces))
                raise RuntimeError(
                    f"Module {module_id} must expose LEFT, RIGHT, TOP and "
                    f"BOTTOM docking faces; found: {found}"
                )
            discovered[module_id] = tuple(faces)
        return discovered

    def _read_face_pose(self, face: DockingFace) -> DockingFacePose:
        from pxr import Gf, Usd, UsdGeom

        prim = self._stage.GetPrimAtPath(face.frame_path)
        transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        position = transform.ExtractTranslation()
        normal = transform.TransformDir(Gf.Vec3d(1.0, 0.0, 0.0))
        tangent = transform.TransformDir(Gf.Vec3d(0.0, 1.0, 0.0))
        return DockingFacePose(
            face=face,
            position_world_m=tuple(float(value) for value in position),
            outward_normal_world=tuple(float(value) for value in normal),
            tangent_world=tuple(float(value) for value in tangent),
        )

    def _attach(
        self,
        command: DockingCommand,
        thresholds: DockingThresholds,
        *,
        snap_to_nominal: bool = False,
    ) -> DockingResult:
        module_pair = command.unordered_module_pair
        existing = self._connections.get(module_pair)
        if existing is not None:
            return DockingResult(
                False,
                "modules are already attached through "
                f"{existing.first_face.face_name}<->"
                f"{existing.second_face.face_name}",
                existing,
            )

        first_poses = self.face_poses_for(command.first_module)
        second_poses = self.face_poses_for(command.second_module)
        if command.explicit_faces:
            selected = self._evaluate_requested_pair(
                command,
                first_poses,
                second_poses,
                thresholds,
            )
        else:
            selected = select_best_face_pair(
                first_poses,
                second_poses,
                occupied_faces=self.occupied_faces,
                thresholds=thresholds,
            )
        if selected is None:
            if command.explicit_faces:
                if command.face_keys & self.occupied_faces:
                    detail = "one or both requested faces are already occupied"
                else:
                    requested = self._requested_pair(
                        command,
                        first_poses,
                        second_poses,
                        thresholds,
                    )
                    detail = self._evaluation_detail("requested", requested)
            else:
                closest = self._closest_free_pair(
                    first_poses,
                    second_poses,
                    thresholds,
                )
                detail = self._evaluation_detail("closest", closest)
            return DockingResult(
                False,
                f"attach rejected: faces are not in valid contact; {detail}",
            )

        connection = self._create_fixed_joint(
            selected,
            snap_to_nominal=snap_to_nominal,
        )
        self._connections[module_pair] = connection
        return DockingResult(
            True,
            f"attached {connection.first_face.module_id}:"
            f"{connection.first_face.face_name} to "
            f"{connection.second_face.module_id}:"
            f"{connection.second_face.face_name}; "
            f"{self._evaluation_detail('contact', selected)}",
            connection,
        )

    def _requested_pair(
        self,
        command: DockingCommand,
        first_poses: tuple[DockingFacePose, ...],
        second_poses: tuple[DockingFacePose, ...],
        thresholds: DockingThresholds,
    ) -> DockingPairEvaluation:
        if command.first_face is None or command.second_face is None:
            raise ValueError("Requested pair requires explicit face names")
        first = next(
            pose
            for pose in first_poses
            if pose.face.face_name == command.first_face
        )
        second = next(
            pose
            for pose in second_poses
            if pose.face.face_name == command.second_face
        )
        return evaluate_face_pair(first, second, thresholds)

    def _evaluate_requested_pair(
        self,
        command: DockingCommand,
        first_poses: tuple[DockingFacePose, ...],
        second_poses: tuple[DockingFacePose, ...],
        thresholds: DockingThresholds,
    ) -> DockingPairEvaluation | None:
        if command.face_keys & self.occupied_faces:
            return None
        evaluation = self._requested_pair(
            command,
            first_poses,
            second_poses,
            thresholds,
        )
        return evaluation if evaluation.eligible else None

    @staticmethod
    def _evaluation_detail(
        label: str,
        evaluation: DockingPairEvaluation | None,
    ) -> str:
        if evaluation is None:
            return "no free face pair is available"
        return (
            f"{label}={evaluation.first.face.face_name}<->"
            f"{evaluation.second.face.face_name} "
            f"normal_gap={1e3*evaluation.normal_separation_m:.2f}mm "
            f"lateral={1e3*evaluation.lateral_offset_m:.2f}mm "
            f"normal_error="
            f"{math.degrees(evaluation.normal_misalignment_rad):.1f}deg "
            f"clocking_error="
            f"{math.degrees(evaluation.clocking_error_rad):.1f}deg"
        )

    def _closest_free_pair(
        self,
        first_poses: tuple[DockingFacePose, ...],
        second_poses: tuple[DockingFacePose, ...],
        thresholds: DockingThresholds,
    ) -> DockingPairEvaluation | None:
        occupied = self.occupied_faces
        candidates = [
            evaluate_face_pair(first, second, thresholds)
            for first in first_poses
            for second in second_poses
            if first.face.key not in occupied
            and second.face.key not in occupied
        ]
        return min(
            candidates,
            key=lambda candidate: candidate.score,
            default=None,
        )

    def _create_fixed_joint(
        self,
        selected: DockingPairEvaluation,
        *,
        snap_to_nominal: bool = False,
    ) -> DockingConnection:
        from pxr import Gf, Sdf, UsdGeom, UsdPhysics

        self._joint_serial += 1
        first = selected.first.face
        second = selected.second.face
        joint_path = (
            f"{self._joints_root}/dock_{self._joint_serial:04d}_"
            f"{first.module_id}_{first.face_name.lower()}_"
            f"{second.module_id}_{second.face_name.lower()}"
        )
        joint = UsdPhysics.FixedJoint.Define(self._stage, joint_path)
        joint.CreateBody0Rel().SetTargets([Sdf.Path(first.rigid_body_path)])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(second.rigid_body_path)])
        joint.CreateCollisionEnabledAttr(False)
        joint.CreateExcludeFromArticulationAttr(True)

        cache = UsdGeom.XformCache()
        first_face_world = cache.GetLocalToWorldTransform(
            self._stage.GetPrimAtPath(first.frame_path)
        )
        second_face_world = cache.GetLocalToWorldTransform(
            self._stage.GetPrimAtPath(second.frame_path)
        )
        common_world = first_face_world
        for body_index, (body_path, face_world) in enumerate(
            (
                (first.rigid_body_path, first_face_world),
                (second.rigid_body_path, second_face_world),
            )
        ):
            body_world = cache.GetLocalToWorldTransform(
                self._stage.GetPrimAtPath(body_path)
            )
            local = Gf.Transform(common_world) * Gf.Transform(
                body_world.GetInverse()
            )
            position = Gf.Vec3f(local.GetTranslation())
            rotation = Gf.Quatf(local.GetRotation().GetQuat())
            if body_index == 0:
                joint.CreateLocalPos0Attr(position)
                joint.CreateLocalRot0Attr(rotation)
            else:
                joint.CreateLocalPos1Attr(position)
                joint.CreateLocalRot1Attr(rotation)

        if snap_to_nominal:
            # Both local positions now refer to the real EP-face centres. The
            # solver removes the accepted millimetric separation instead of
            # preserving it forever. The common current orientation remains
            # unchanged, avoiding a 180-degree impulse from the opposed
            # outward-normal convention.
            first_body_world = cache.GetLocalToWorldTransform(
                self._stage.GetPrimAtPath(first.rigid_body_path)
            )
            second_body_world = cache.GetLocalToWorldTransform(
                self._stage.GetPrimAtPath(second.rigid_body_path)
            )
            first_local = Gf.Transform(first_face_world) * Gf.Transform(
                first_body_world.GetInverse()
            )
            second_local = Gf.Transform(second_face_world) * Gf.Transform(
                second_body_world.GetInverse()
            )
            joint.CreateLocalPos0Attr(
                Gf.Vec3f(first_local.GetTranslation())
            )
            joint.CreateLocalPos1Attr(
                Gf.Vec3f(second_local.GetTranslation())
            )

        return DockingConnection(joint_path, first, second)

    def _detach(self, command: DockingCommand) -> DockingResult:
        module_pair = command.unordered_module_pair
        connection = self._connections.get(module_pair)
        if connection is None:
            return DockingResult(
                False,
                f"{command.first_module} and {command.second_module} "
                "are not attached",
            )
        connected_face_keys = frozenset(
            (connection.first_face.key, connection.second_face.key)
        )
        if command.explicit_faces and command.face_keys != connected_face_keys:
            return DockingResult(
                False,
                "requested faces are not the connected pair; connected="
                f"{connection.first_face.module_id}:"
                f"{connection.first_face.face_name}<->"
                f"{connection.second_face.module_id}:"
                f"{connection.second_face.face_name}",
                connection,
            )
        self._connections.pop(module_pair)
        if not self._stage.RemovePrim(connection.joint_path):
            self._connections[module_pair] = connection
            return DockingResult(
                False,
                f"could not remove fixed joint {connection.joint_path}",
                connection,
            )
        return DockingResult(
            True,
            f"detached {connection.first_face.module_id}:"
            f"{connection.first_face.face_name} from "
            f"{connection.second_face.module_id}:"
            f"{connection.second_face.face_name}",
            connection,
        )
