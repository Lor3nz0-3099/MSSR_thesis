from __future__ import annotations

from itertools import combinations
import json
import math
from pathlib import Path
from typing import Any, Mapping

from smores_ep.config.geometry import SmoresGeometry
from smores_ep.config.physics import SmoresActuatorConfig
from smores_ep.docking.model import DockingFacePose, evaluate_face_pair
from smores_ep.isaac.docking import IsaacDockingManager
from smores_ep.isaac.dynamic_stage import ArticulationStateReader
from smores_ep.primitives.pose_control import wrap_angle


class SmoresStateGraphPublisher:
    """Write canonical SMORES-EP observations for the shared ROS bridge."""

    def __init__(
        self,
        stage: Any,
        module_roots: Mapping[str, str],
        states: Mapping[str, ArticulationStateReader],
        docking: IsaacDockingManager,
        output_dir: str | Path = "logs/bridge",
        geometry: SmoresGeometry | None = None,
        actuator_profiles: Mapping[str, SmoresActuatorConfig] | None = None,
        roles: Mapping[str, Mapping[str, Any]] | None = None,
        include_contact_candidates: bool = True,
        course_observation: Mapping[str, Any] | None = None,
    ) -> None:
        if set(module_roots) != set(states):
            raise ValueError("State publisher requires one reader per module")
        self._stage = stage
        self._module_roots = dict(module_roots)
        self._states = dict(states)
        self._docking = docking
        self._output_dir = Path(output_dir)
        self._include_contact_candidates = include_contact_candidates
        self._geometry = geometry or SmoresGeometry()
        default_actuators = SmoresActuatorConfig()
        self._actuators = {
            module_id: (
                actuator_profiles.get(module_id, default_actuators)
                if actuator_profiles is not None
                else default_actuators
            )
            for module_id in module_roots
        }
        self._roles = {
            module_id: dict(payload)
            for module_id, payload in (roles or {}).items()
        }
        self._course_observation = dict(course_observation or {})
        self._last_sample: dict[
            str,
            tuple[float, tuple[float, float, float], float],
        ] = {}

    def set_role(
        self,
        module_id: str,
        current_role: str,
        *,
        target_role: str = "",
        functional_role: Mapping[str, Any] | None = None,
        confidence: float = 1.0,
        source: str = "deterministic_expert",
    ) -> None:
        if module_id not in self._module_roots:
            raise KeyError(f"Unknown module: {module_id}")
        self._roles[module_id] = {
            "current_role": current_role,
            "target_role": target_role,
            "role_confidence": float(confidence),
            "role_source": source,
            "functional_role": dict(functional_role or {}),
        }

    def publish(
        self,
        stamp_s: float,
        *,
        support_engaged_by_module: Mapping[str, bool] | None = None,
        experiment_profile: str = "nominal",
    ) -> None:
        face_poses = {
            module_id: self._docking.face_poses_for(module_id)
            for module_id in self._module_roots
        }
        contacts = (
            self._contact_candidates(face_poses)
            if self._include_contact_candidates
            else []
        )
        connected_by_face = self._connected_by_face()
        contact_by_face = {
            endpoint
            for contact in contacts
            for endpoint in (
                (str(contact["module_a_id"]), str(contact["connector_a_id"])),
                (str(contact["module_b_id"]), str(contact["connector_b_id"])),
            )
        }
        support_state = support_engaged_by_module or {}
        modules = [
            self._module_payload(
                module_id,
                stamp_s,
                face_poses[module_id],
                connected_by_face,
                contact_by_face,
                bool(support_state.get(module_id, False)),
                experiment_profile,
            )
            for module_id in sorted(self._module_roots)
        ]
        attachments = self._attachment_payloads(face_poses)
        state_payload: dict[str, Any] = {
            "schema_version": "mssr.module_states.v2",
                        "course": dict(self._course_observation),
            "stamp": stamp_s,
            "timestamp": stamp_s,
            "robot_family": "smores_ep",
            "observation_model": {
                "type": "centralized_simulator_ground_truth",
                "localization_equivalent": "vicon",
                "noise_enabled": False,
                "latency_s": 0.0,
            },
            "experiment_profile": experiment_profile,
            "modules": modules,
            "contacts": contacts,
            "attachments": attachments,
        }
        graph_edges = [
            *(
                {
                    **attachment,
                    "edge_type": "docking_connection",
                    "connection_state": "latched",
                }
                for attachment in attachments
            ),
            *(
                {
                    **contact,
                    "edge_type": "contact_candidate",
                    "connection_state": "aligned_contact",
                }
                for contact in contacts
            ),
        ]
        graph_payload: dict[str, Any] = {
            "schema_version": "mssr.robot_graph.v2",
            "stamp": stamp_s,
            "nodes": [
                {
                    "module_id": module["module_id"],
                    "attributes": module,
                }
                for module in modules
            ],
            "edges": graph_edges,
            "global_attributes": {
                "robot_families": ["smores_ep"],
                "module_count": len(modules),
                "module_geometry": {
                    "wheel_radius_m": self._geometry.wheel_radius_m,
                    "forward_collision_extent_m": (
                        self._geometry.top_face_x_m
                    ),
                    "pan_face_radius_m": self._geometry.pan_face_radius_m,
                },
                "latched_connection_count": len(attachments),
                "contact_candidate_count": len(contacts),
                "experiment_profile": experiment_profile,
                "course": dict(self._course_observation),
            },
        }
        combined = {
            "schema_version": "mssr.state_graph.v2",
            "stamp": stamp_s,
            "state": state_payload,
            "graph": graph_payload,
        }
        self._write("module_states.json", state_payload)
        self._write("robot_graph.json", graph_payload)
        self._write("state_graph.json", combined)

    def _module_payload(
        self,
        module_id: str,
        stamp_s: float,
        face_poses: tuple[DockingFacePose, ...],
        connected_by_face: Mapping[tuple[str, str], tuple[str, str]],
        contact_by_face: set[tuple[str, str]],
        support_engaged: bool,
        experiment_profile: str,
    ) -> dict[str, Any]:
        position, orientation, yaw = self._body_pose(module_id)
        linear_velocity, angular_velocity = self._velocity(
            module_id,
            stamp_s,
            position,
            yaw,
        )
        joint_state = self._states[module_id].read()
        actuators = self._actuators[module_id]
        role = self._roles.get(module_id, {})
        current_role = str(role.get("current_role", "unassigned"))
        connectors = []
        for face_pose in sorted(
            face_poses,
            key=lambda item: item.face.face_name,
        ):
            key = face_pose.face.key
            connected = connected_by_face.get(key)
            status = (
                "connected"
                if connected is not None
                else "in_contact"
                if key in contact_by_face
                else "available"
            )
            connectors.append(
                {
                    "connector_id": face_pose.face.face_name,
                    "face_name": face_pose.face.face_name,
                    "connector_type": "smores_ep_face",
                    "status": status,
                    "is_enabled": connected is not None,
                    "connected_module_id": (
                        connected[0] if connected is not None else None
                    ),
                    "connected_connector_id": (
                        connected[1] if connected is not None else None
                    ),
                    "position_world": list(face_pose.position_world_m),
                    "outward_normal_world": list(
                        face_pose.outward_normal_world
                    ),
                    "tangent_world": list(face_pose.tangent_world),
                }
            )
        return {
            "module_id": module_id,
            "robot_family": "smores_ep",
            "module_type": "smores_ep_v1",
            "prim_path": self._module_roots[module_id],
            "body_frame_id": f"{module_id}/base_link",
            "pose": {
                "frame_id": "world",
                "position": list(position),
                "orientation": list(orientation),
                "orientation_xyzw": list(orientation),
            },
            "position": list(position),
            "orientation": list(orientation),
            "twist": {
                "linear": list(linear_velocity),
                "angular": list(angular_velocity),
            },
            "linear_velocity": list(linear_velocity),
            "angular_velocity": list(angular_velocity),
            "mass": self._geometry.module_mass_kg,
            "current_role": current_role,
            "role": current_role,
            "target_role": str(role.get("target_role", "")),
            "role_confidence": float(role.get("role_confidence", 0.0)),
            "role_source": str(role.get("role_source", "unassigned")),
            "functional_role": dict(role.get("functional_role", {})),
            "control_available": True,
            "health": "nominal",
            "capabilities": [
                "differential_drive",
                "continuous_pan",
                "limited_tilt",
                "four_face_rigid_docking",
                "tow_connected_modules",
                "lift_connected_chain",
            ],
            "design_profile": "enhanced_smores_ep_compatible",
            "design_requirements": {
                "tow_attached_payload": True,
                "lift_chain_module_count_min": 5,
                "lift_chain_module_count_target": 7,
            },
            "sensor_capabilities": [
                "joint_position",
                "external_localization_simulated",
            ],
            "observation_confidence": 1.0,
            "actuators": {
                "left_wheel": {
                    "position_rad": joint_state.left_wheel_rad,
                    "velocity_rad_s": joint_state.left_wheel_rad_s,
                    "continuous": True,
                    "max_velocity_rad_s": actuators.wheel_max_speed_rad_s,
                    "max_effort_nm": actuators.wheel_max_effort_nm,
                },
                "right_wheel": {
                    "position_rad": joint_state.right_wheel_rad,
                    "velocity_rad_s": joint_state.right_wheel_rad_s,
                    "continuous": True,
                    "max_velocity_rad_s": actuators.wheel_max_speed_rad_s,
                    "max_effort_nm": actuators.wheel_max_effort_nm,
                },
                "pan": {
                    "position_rad": joint_state.pan_joint_rad,
                    "continuous": True,
                    "max_velocity_rad_s": actuators.internal_max_speed_rad_s,
                    "max_effort_nm": actuators.pan_max_effort_nm,
                },
                "tilt": {
                    "position_rad": -joint_state.tilt_joint_rad,
                    "continuous": False,
                    "lower_limit_rad": self._geometry.tilt_min_rad,
                    "upper_limit_rad": self._geometry.tilt_max_rad,
                    "max_velocity_rad_s": actuators.internal_max_speed_rad_s,
                    "max_effort_nm": actuators.tilt_max_effort_nm,
                },
            },
            "connectors": connectors,
            "simulation_fixtures": {
                "ground_support_anchor": support_engaged,
                "unbreakable_docking_joint": True,
                "experiment_profile": experiment_profile,
            },
        }

    def _body_pose(
        self,
        module_id: str,
    ) -> tuple[
        tuple[float, float, float],
        tuple[float, float, float, float],
        float,
    ]:
        from pxr import Gf, Usd, UsdGeom

        path = f"{self._module_roots[module_id]}/body_link"
        matrix = UsdGeom.Xformable(
            self._stage.GetPrimAtPath(path)
        ).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        translation = matrix.ExtractTranslation()
        quaternion = matrix.ExtractRotationQuat()
        imaginary = quaternion.GetImaginary()
        orientation = (
            float(imaginary[0]),
            float(imaginary[1]),
            float(imaginary[2]),
            float(quaternion.GetReal()),
        )
        forward = matrix.TransformDir(Gf.Vec3d(1.0, 0.0, 0.0))
        yaw = math.atan2(float(forward[1]), float(forward[0]))
        return (
            tuple(float(value) for value in translation),
            orientation,
            yaw,
        )

    def _velocity(
        self,
        module_id: str,
        stamp_s: float,
        position: tuple[float, float, float],
        yaw: float,
    ) -> tuple[
        tuple[float, float, float],
        tuple[float, float, float],
    ]:
        previous = self._last_sample.get(module_id)
        self._last_sample[module_id] = (stamp_s, position, yaw)
        if previous is None:
            return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
        previous_stamp, previous_position, previous_yaw = previous
        dt = stamp_s - previous_stamp
        if dt <= 1.0e-9:
            return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
        linear = tuple(
            (current - old) / dt
            for current, old in zip(position, previous_position)
        )
        angular = (0.0, 0.0, wrap_angle(yaw - previous_yaw) / dt)
        return linear, angular  # type: ignore[return-value]

    def _connected_by_face(
        self,
    ) -> dict[tuple[str, str], tuple[str, str]]:
        result: dict[tuple[str, str], tuple[str, str]] = {}
        for connection in self._docking.connections:
            first = connection.first_face.key
            second = connection.second_face.key
            result[first] = second
            result[second] = first
        return result

    def _attachment_payloads(
        self,
        face_poses: Mapping[str, tuple[DockingFacePose, ...]],
    ) -> list[dict[str, Any]]:
        payloads = []
        for connection in self._docking.connections:
            first = next(
                pose
                for pose in face_poses[connection.first_face.module_id]
                if pose.face.key == connection.first_face.key
            )
            second = next(
                pose
                for pose in face_poses[connection.second_face.module_id]
                if pose.face.key == connection.second_face.key
            )
            evaluation = evaluate_face_pair(first, second)
            payloads.append(
                {
                    "connection_id": connection.joint_path,
                    "module_a_id": connection.first_face.module_id,
                    "connector_a_id": connection.first_face.face_name,
                    "face_a": connection.first_face.face_name,
                    "module_b_id": connection.second_face.module_id,
                    "connector_b_id": connection.second_face.face_name,
                    "face_b": connection.second_face.face_name,
                    "is_contact": True,
                    "is_attached": True,
                    "is_magnet_enabled": True,
                    "status": "connected",
                    "connector_type": "smores_ep_face",
                    "attachment_mode": "discrete_face_rigid",
                    "joint_type": "rigid",
                    "allows_rotation": False,
                    "allowed_relative_dofs": [],
                    "is_load_bearing": True,
                    "is_temporary": False,
                    "clocking_quarter_turns": (
                        evaluation.clocking_quarter_turns
                    ),
                    "clocking_error_rad": evaluation.clocking_error_rad,
                    "normal_gap_m": evaluation.normal_separation_m,
                    "lateral_error_m": evaluation.lateral_offset_m,
                    "normal_error_rad": evaluation.normal_misalignment_rad,
                    "relative_transform": self._relative_body_transform(
                        connection.first_face.module_id,
                        connection.second_face.module_id,
                    ),
                    "edge_role": "physical_docking",
                }
            )
        return payloads

    def _contact_candidates(
        self,
        face_poses: Mapping[str, tuple[DockingFacePose, ...]],
    ) -> list[dict[str, Any]]:
        occupied = self._docking.occupied_faces
        connected_pairs = {
            connection.module_pair
            for connection in self._docking.connections
        }
        candidates: list[dict[str, Any]] = []
        for first_id, second_id in combinations(sorted(face_poses), 2):
            if frozenset((first_id, second_id)) in connected_pairs:
                continue
            for first in face_poses[first_id]:
                for second in face_poses[second_id]:
                    if first.face.key in occupied or second.face.key in occupied:
                        continue
                    evaluation = evaluate_face_pair(first, second)
                    if not evaluation.eligible:
                        continue
                    candidates.append(
                        {
                            "module_a_id": first_id,
                            "connector_a_id": first.face.face_name,
                            "face_a": first.face.face_name,
                            "module_b_id": second_id,
                            "connector_b_id": second.face.face_name,
                            "face_b": second.face.face_name,
                            "is_contact": True,
                            "is_attached": False,
                            "is_magnet_enabled": False,
                            "status": "in_contact",
                            "connector_type": "smores_ep_face",
                            "attachment_mode": "discrete_face_candidate",
                            "joint_type": None,
                            "normal_gap_m": evaluation.normal_separation_m,
                            "lateral_error_m": evaluation.lateral_offset_m,
                            "normal_error_rad": (
                                evaluation.normal_misalignment_rad
                            ),
                            "clocking_error_rad": (
                                evaluation.clocking_error_rad
                            ),
                            "clocking_quarter_turns": (
                                evaluation.clocking_quarter_turns
                            ),
                            "edge_role": "docking_candidate",
                        }
                    )
        return candidates

    def _relative_body_transform(
        self,
        first_module_id: str,
        second_module_id: str,
    ) -> dict[str, list[float]]:
        from pxr import Gf, Usd, UsdGeom

        def body_matrix(module_id: str) -> Any:
            path = f"{self._module_roots[module_id]}/body_link"
            return UsdGeom.Xformable(
                self._stage.GetPrimAtPath(path)
            ).ComputeLocalToWorldTransform(Usd.TimeCode.Default())

        relative = Gf.Transform(
            body_matrix(second_module_id)
            * body_matrix(first_module_id).GetInverse()
        )
        translation = relative.GetTranslation()
        quaternion = relative.GetRotation().GetQuat()
        imaginary = quaternion.GetImaginary()
        return {
            "translation": [float(value) for value in translation],
            "orientation_xyzw": [
                float(imaginary[0]),
                float(imaginary[1]),
                float(imaginary[2]),
                float(quaternion.GetReal()),
            ],
        }

    def _write(self, filename: str, payload: Mapping[str, Any]) -> None:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        path = self._output_dir / filename
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, sort_keys=True),
            encoding="utf-8",
        )
        temporary.replace(path)
