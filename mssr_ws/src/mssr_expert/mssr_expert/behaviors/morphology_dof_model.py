"""Derive per-module operational degrees of freedom from a morphology graph."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from mssr_expert.graph.attributed_robot_graph import AttributedRobotGraph


@dataclass(frozen=True)
class OperationalDof:
    """One generalized coordinate and its current morphology-level use."""

    module_id: str
    name: str
    joint_kind: str
    affected_face: str
    mode: str
    connected: bool
    position_rad: float | None = None
    lower_limit_rad: float | None = None
    upper_limit_rad: float | None = None
    max_effort_nm: float | None = None
    motor_mix: tuple[tuple[str, float], ...] = ()
    locomotion_capable: bool = False
    shape_capable: bool = False

    @property
    def load_bearing(self) -> bool:
        return self.mode == "load_bearing"

    @property
    def locomotion_candidate(self) -> bool:
        return self.mode == "locomotion_candidate"

    @property
    def shape_candidate(self) -> bool:
        return self.mode == "shape_candidate"

    @property
    def can_locomote(self) -> bool:
        """Whether this free coordinate may be used as a locomotor.

        PAN is intentionally included: in wheel-support morphologies such as
        RC Car8 the free TOP/PAN disk is the ground-contact wheel.  ``mode``
        describes the coordinate's current structural use; it is not an
        exhaustive list of all functions that the coordinate can perform.
        """

        return self.locomotion_capable

    @property
    def can_shape(self) -> bool:
        """Whether this coordinate is a morphology-shaping actuator."""

        return self.shape_capable


@dataclass(frozen=True)
class ModuleDofInventory:
    module_id: str
    target_role: str
    connected_faces: frozenset[str]
    body_is_directly_attached: bool
    ground_support_anchor: bool
    dofs: tuple[OperationalDof, ...]


@dataclass(frozen=True)
class MorphologyDofInventory:
    modules: tuple[ModuleDofInventory, ...]

    @property
    def dofs(self) -> tuple[OperationalDof, ...]:
        return tuple(dof for module in self.modules for dof in module.dofs)

    def by_mode(self, mode: str) -> tuple[OperationalDof, ...]:
        return tuple(dof for dof in self.dofs if dof.mode == mode)

    def locomotion_dofs(self) -> tuple[OperationalDof, ...]:
        return tuple(dof for dof in self.dofs if dof.can_locomote)

    def shape_dofs(self) -> tuple[OperationalDof, ...]:
        return tuple(dof for dof in self.dofs if dof.can_shape)

    @property
    def signature(self) -> tuple[tuple[str, str, str], ...]:
        return tuple(
            (dof.module_id, dof.name, dof.mode) for dof in self.dofs
        )


class SmoresMorphologyDofAnalyzer:
    """Classify SMORES-EP DoFs from occupied connector faces.

    LEFT and RIGHT are mounted on their respective wheel coordinates. TOP is
    downstream of both TILT and PAN. BOTTOM is fixed to ``body_link`` and
    therefore consumes no actuator coordinate. A rigidly docked face does not
    remove its upstream joint: that joint becomes a load-bearing shape DoF.
    """

    _DOF_FACE = {
        "left_wheel": "LEFT",
        "right_wheel": "RIGHT",
        "tilt": "TOP",
        "pan": "TOP",
    }
    _DOF_KIND = {
        "left_wheel": "wheel",
        "right_wheel": "wheel",
        "tilt": "shape",
        "pan": "shape",
    }
    _MOTOR_MIX = {
        "tilt": (("motorA", 1.0), ("motorB", 1.0)),
        "pan": (("motorA", 1.0), ("motorB", -1.0)),
    }

    def analyze(self, graph: AttributedRobotGraph) -> MorphologyDofInventory:
        connected_faces = {
            node.node_id: set() for node in graph.nodes
        }
        for edge in graph.edges:
            if not self._is_structural_connection(edge.attributes):
                continue
            face_a = self._face(edge.attributes, "a")
            face_b = self._face(edge.attributes, "b")
            if face_a:
                connected_faces.setdefault(edge.module_a_id, set()).add(face_a)
            if face_b:
                connected_faces.setdefault(edge.module_b_id, set()).add(face_b)

        modules: list[ModuleDofInventory] = []
        for node in sorted(graph.nodes, key=lambda item: item.node_id):
            attributes = node.attributes
            faces = frozenset(connected_faces.get(node.node_id, set()))
            actuators = attributes.get("actuators", {})
            if not isinstance(actuators, Mapping):
                actuators = {}
            dofs = tuple(
                self._operational_dof(
                    node.node_id,
                    name,
                    faces,
                    actuators.get(name, {}),
                )
                for name in (
                    "left_wheel",
                    "right_wheel",
                    "tilt",
                    "pan",
                )
            )
            fixtures = attributes.get("simulation_fixtures", {})
            if not isinstance(fixtures, Mapping):
                fixtures = {}
            modules.append(
                ModuleDofInventory(
                    module_id=node.node_id,
                    target_role=str(attributes.get("target_role", "")),
                    connected_faces=faces,
                    body_is_directly_attached="BOTTOM" in faces,
                    ground_support_anchor=bool(
                        fixtures.get("ground_support_anchor", False)
                    ),
                    dofs=dofs,
                )
            )
        return MorphologyDofInventory(tuple(modules))

    def _operational_dof(
        self,
        module_id: str,
        name: str,
        connected_faces: frozenset[str],
        raw_actuator: Any,
    ) -> OperationalDof:
        affected_face = self._DOF_FACE[name]
        connected = affected_face in connected_faces
        kind = self._DOF_KIND[name]
        mode = (
            "load_bearing"
            if connected
            else "locomotion_candidate"
            if kind == "wheel"
            else "shape_candidate"
        )
        actuator = raw_actuator if isinstance(raw_actuator, Mapping) else {}
        return OperationalDof(
            module_id=module_id,
            name=name,
            joint_kind=kind,
            affected_face=affected_face,
            mode=mode,
            connected=connected,
            position_rad=self._optional_float(actuator.get("position_rad")),
            lower_limit_rad=self._optional_float(
                actuator.get("lower_limit_rad")
            ),
            upper_limit_rad=self._optional_float(
                actuator.get("upper_limit_rad")
            ),
            max_effort_nm=self._optional_float(
                actuator.get("max_effort_nm")
            ),
            motor_mix=self._MOTOR_MIX.get(name, ()),
            locomotion_capable=(
                not connected
                and name in {"left_wheel", "right_wheel", "pan"}
            ),
            shape_capable=name in {"tilt", "pan"},
        )

    @staticmethod
    def _is_structural_connection(attributes: Mapping[str, Any]) -> bool:
        relation = str(
            attributes.get("relation_type")
            or attributes.get("edge_type")
            or ""
        )
        return bool(
            attributes.get("is_target_edge", False)
            or attributes.get("is_attached", False)
            or relation in {"target_connection", "current_connection"}
        )

    @staticmethod
    def _face(attributes: Mapping[str, Any], endpoint: str) -> str:
        return str(
            attributes.get(f"face_{endpoint}")
            or attributes.get(f"connector_{endpoint}_id")
            or ""
        ).upper()

    @staticmethod
    def _optional_float(value: Any) -> float | None:
        return None if value is None else float(value)
