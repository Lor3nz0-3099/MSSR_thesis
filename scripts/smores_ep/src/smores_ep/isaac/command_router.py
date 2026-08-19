from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from smores_ep.control.teleop import InternalMotionMode, SmoresCommand
from smores_ep.isaac.docking import IsaacDockingManager
from smores_ep.isaac.dynamic_stage import (
    ArticulationStateReader,
    DynamicDriveController,
)


@dataclass(frozen=True)
class RoutedModuleState:
    control_mode: str
    docked_faces: frozenset[str]


class IsaacMultiModuleCommandRouter:
    """Route sparse commands to identical dynamic module articulations.

    Missing isolated modules are backdrivable. Missing connected modules keep
    their shape with towable free wheels. A wheel command from the morphology
    layer drives LEFT/RIGHT while retaining PAN/TILT; only an explicit PASSIVE
    command releases the internal drives.
    """

    def __init__(
        self,
        states: Mapping[str, ArticulationStateReader],
        drives: Mapping[str, DynamicDriveController],
        docking: IsaacDockingManager,
    ) -> None:
        if set(states) != set(drives):
            raise ValueError("Every routed module needs state and drive objects")
        if set(states) != set(docking.module_ids):
            raise ValueError("Router and docking module registries must match")
        self._states = dict(states)
        self._drives = dict(drives)
        self._docking = docking
        self._configured: dict[str, RoutedModuleState] = {}

    def apply(
        self,
        commands: Mapping[str, SmoresCommand],
    ) -> dict[str, tuple[float, float]]:
        unknown = set(commands) - set(self._states)
        if unknown:
            raise ValueError(
                "Commands reference unknown module(s): "
                + ", ".join(sorted(unknown))
            )
        wheel_rates: dict[str, tuple[float, float]] = {}
        for module_id, state in self._states.items():
            docked_faces = self._connected_faces(module_id)
            command = commands.get(module_id)
            has_wheel_motion = command is not None and (
                abs(command.linear_x_m_s) > 1.0e-12
                or abs(command.angular_z_rad_s) > 1.0e-12
            )
            if command is None:
                # A docked-but-uncommanded module must not go limp: without
                # this it sags under gravity while waiting for its own
                # posture goal (or forever, if it never gets one, like an
                # RC-Car chassis link). Only a genuinely isolated module
                # (no docked face at all) stays fully backdrivable/towable.
                control_mode = (
                    "structural_hold" if docked_faces else "passive_all"
                )
            elif (
                command.internal_motion is InternalMotionMode.PASSIVE
                and has_wheel_motion
            ):
                control_mode = "wheel_drive_passive_internal"
            elif command.internal_motion is InternalMotionMode.PASSIVE:
                control_mode = "passive_all"
            elif (
                command.internal_motion
                is InternalMotionMode.STRUCTURAL_HOLD
                and has_wheel_motion
            ):
                # A deployed morphology may retain PAN/TILT while its free
                # LEFT/RIGHT wheels become locomotors.  Route this through
                # the powered wheel profile; the command still supplies the
                # retained structural joint targets.
                control_mode = "controlled"
            elif (
                command.internal_motion
                is InternalMotionMode.STRUCTURAL_HOLD
            ):
                control_mode = "structural_hold"
            elif (
                command.internal_motion
                is InternalMotionMode.PAN_VELOCITY
            ):
                # Continuous PAN is the only shape mode that deliberately
                # releases the PAN position spring.  The wheels and TILT
                # remain rigid so the requested rotation cannot twist the
                # whole connected morphology.
                control_mode = "pan_velocity_braked_support"
            elif has_wheel_motion:
                control_mode = "controlled"
            else:
                control_mode = "internal_drive_braked_wheels"
            mode = RoutedModuleState(
                control_mode=control_mode,
                docked_faces=docked_faces,
            )
            if self._configured.get(module_id) != mode:
                if mode.control_mode == "controlled":
                    state.configure_controlled_docking_mode(docked_faces)
                elif mode.control_mode == "internal_drive_braked_wheels":
                    state.configure_internal_drive_with_braked_wheels(
                        docked_faces
                    )
                elif mode.control_mode == "pan_velocity_braked_support":
                    state.configure_pan_velocity_drive_with_braked_wheels(
                        docked_faces
                    )
                elif mode.control_mode == "wheel_drive_passive_internal":
                    state.configure_wheel_drive_with_passive_internals()
                elif mode.control_mode == "structural_hold":
                    state.configure_structural_hold_mode(docked_faces)
                else:
                    state.configure_fully_passive_mode()
                self._configured[module_id] = mode
            if command is not None:
                wheel_rates[module_id] = self._drives[module_id].apply(command)
        return wheel_rates

    def _connected_faces(self, module_id: str) -> frozenset[str]:
        return frozenset(
            face.face_name
            for connection in self._docking.connections
            for face in (connection.first_face, connection.second_face)
            if face.module_id == module_id
        )
