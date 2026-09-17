"""Global teleoperation state, separate from ROS, actuator and dataset adapters.

observe_topology is an internal boundary for an already-verified matcher
result. Controller intent and successful macro reports cannot establish it.
"""
from __future__ import annotations


ACTIVE_MORPHOLOGIES = frozenset({"rc_car8", "snake8", "mobile_manipulator8"})


class TeleopState:
    def __init__(self) -> None:
        self._started = False
        self._macro_active = False
        self._estop_active = False
        self._requested_morphology: str | None = None
        self._detected_morphology: str | None = None
        self._controller_connected = False
        self._recording = False
        self._recording_stop_pending = False
        self._last_macro_success: bool | None = None

    @property
    def phase(self) -> str:
        if self.estop_active:
            return "ESTOP_PAUSED"
        if self.macro_active:
            return "STRUCTURAL_MACRO"
        return "READY" if self._started else "STARTUP"

    @property
    def authority(self) -> str:
        if self.estop_active:
            return "ESTOP"
        if self.macro_active:
            return "STRUCTURAL_MACRO"
        if self._started and self.controller_connected and self.active_controller is not None:
            return "TELEOP"
        return "NONE"

    @property
    def active_controller(self) -> str | None:
        return self._detected_morphology if self._started else None

    @property
    def requested_morphology(self) -> str | None:
        return self._requested_morphology

    @property
    def detected_morphology(self) -> str | None:
        return self._detected_morphology

    @property
    def controller_connected(self) -> bool:
        return self._controller_connected

    @property
    def macro_active(self) -> bool:
        return self._macro_active

    @property
    def estop_active(self) -> bool:
        return self._estop_active

    @property
    def recording(self) -> bool:
        return self._recording

    @property
    def recording_stop_pending(self) -> bool:
        return self._recording_stop_pending

    @property
    def last_macro_success(self) -> bool | None:
        return self._last_macro_success

    def start_ready(self) -> None:
        self._started = True

    def set_connected(self, connected: bool) -> None:
        self._controller_connected = bool(connected)

    def request_morphology(self, name: str) -> bool:
        if name not in ACTIVE_MORPHOLOGIES:
            raise ValueError(f"unsupported teleop morphology: {name!r}")
        if self.phase != "READY":
            return False
        self._requested_morphology = name
        return True

    def observe_topology(self, name: str | None) -> None:
        self._detected_morphology = name if name in ACTIVE_MORPHOLOGIES else None

    def begin_macro(self) -> bool:
        if self.phase != "READY" or self.requested_morphology is None:
            return False
        self._macro_active = True
        self._last_macro_success = None
        return True

    def finish_macro(self, success: bool) -> bool:
        if not self.macro_active:
            return False
        self._macro_active = False
        self._last_macro_success = bool(success)
        if self.recording_stop_pending:
            self._recording = False
            self._recording_stop_pending = False
        return True

    def pause(self) -> bool:
        if self.estop_active:
            return False
        self._estop_active = True
        return True

    def resume(self) -> bool:
        if not self.estop_active:
            return False
        self._estop_active = False
        return True

    def toggle_recording(self) -> None:
        """Recording state only; episode creation/writing belongs to T4."""
        if not self.recording:
            self._recording = True
        elif self.macro_active:
            self._recording_stop_pending = True
        else:
            self._recording = False

    def status(self) -> dict:
        return {
            "phase": self.phase, "authority": self.authority,
            "requested_morphology": self.requested_morphology,
            "detected_morphology": self.detected_morphology,
            "active_controller": self.active_controller,
            "controller_connected": self.controller_connected,
            "macro_active": self.macro_active,
            "recording": self.recording,
            "recording_stop_pending": self.recording_stop_pending,
            "estop_active": self.estop_active,
            "last_macro_success": self.last_macro_success,
        }
