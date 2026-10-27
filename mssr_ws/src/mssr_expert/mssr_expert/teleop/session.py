"""Wall-clock coordination of normalized input and global state (no actuators)."""
from __future__ import annotations

from mssr_expert.teleop.input import DualSenseInput, InputConfig, InputSnapshot, STICKS, TRIGGERS
from mssr_expert.teleop.state import TeleopState


_SELECTIONS = {"select_rc": "rc_car8", "select_snake": "snake8",
               "select_mm8": "mobile_manipulator8"}


def input_payload(sample: InputSnapshot) -> dict:
    return {**{name: getattr(sample, name) for name in (*STICKS, *TRIGGERS)},
            "buttons": sorted(sample.buttons),
            "command_events": list(sample.command_events),
            "button_events": [{"button": event.button,
                               "pressed_buttons": sorted(event.pressed_buttons),
                               "received_at": event.received_at} for event in sample.button_events],
            "last_message_at": sample.last_message_at, "connected": sample.connected}


class TeleopSession:
    def __init__(
        self,
        input_config: InputConfig,
        *,
        controller_morphologies=None,
    ) -> None:
        self.input = DualSenseInput(input_config)
        self.state = TeleopState(
            controller_morphologies=controller_morphologies,
        )
        self.state.start_ready()
        self._previous_connected = False
        self._valid_packets = 0
        self._invalid_packets = 0

        # X is both the Snake8 gap button and the initial-assembly modifier.
        # Resolve "X alone" only on release; no hold-duration threshold.
        self._snake_gap_candidate = False
        self._snake_gap_chorded = False

    def update_joy(self, axes, buttons, received_at: float) -> bool:
        accepted = self.input.update(axes, buttons, received_at)
        if accepted:
            self._valid_packets += 1
        else:
            self._invalid_packets += 1
        return accepted

    def tick(self, now: float) -> dict:
        sample = self.input.snapshot(now)
        self.state.set_connected(sample.connected)
        events = []
        rejected = []
        structural_macro_request = None
        structural_macro_kind = None
        if sample.connected != self._previous_connected:
            events.append("controller_connected" if sample.connected else "controller_disconnected")
            self._previous_connected = sample.connected
        commands = sample.command_events
        if "estop" in commands:
            self.state.pause()
        elif "resume" in commands:
            self.state.resume()
        for command in commands:
            if command == "record_toggle":
                self.state.toggle_recording()
        morphology_dpad_buttons = {
            "dpad_left",
            "dpad_up",
            "dpad_right",
        }

        if not sample.connected:
            self._snake_gap_candidate = False
            self._snake_gap_chorded = False

        if "snake_gap" in commands:
            dpad_held = bool(
                morphology_dpad_buttons
                & sample.buttons
            )

            self._snake_gap_candidate = (
                self.state.detected_morphology == "snake8"
                and self.state.active_controller == "snake8"
                and self.state.authority == "TELEOP"
                and not dpad_held
            )
            self._snake_gap_chorded = dpad_held

        if (
            self._snake_gap_candidate
            and morphology_dpad_buttons & sample.buttons
        ):
            # X became part of a morphology-selection chord.  Its later
            # release must never leak into a gap macro.
            self._snake_gap_chorded = True

        selections = {_SELECTIONS[command] for command in commands if command in _SELECTIONS}
        if len(selections) > 1:
            rejected.append("ambiguous_morphology_selection")
        elif selections:
            requested = next(iter(selections))

            initial_dpad_selection = (
                self.state.detected_morphology is None
                and bool(
                    {
                        "dpad_left",
                        "dpad_up",
                        "dpad_right",
                    }
                    & sample.buttons
                )
            )

            if (
                initial_dpad_selection
                and "cross" not in sample.buttons
            ):
                rejected.append(
                    "initial_assembly_requires_cross"
                )
            elif (
                initial_dpad_selection
                and not self.state.initial_assembly_ready
            ):
                rejected.append(
                    "initial_assembly_topology_unavailable"
                )
            elif not self.state.request_morphology(requested):
                rejected.append(
                    "morphology_selection_unavailable"
                )
            elif requested != self.state.detected_morphology:
                structural_macro_request = requested
                structural_macro_kind = (
                    "self_assembly"
                    if initial_dpad_selection
                    else "self_reconfiguration"
                )
        # Square is an unambiguous one-edge Snake8 behavior macro.
        if (
            "snake_stairs" in commands
            and not selections
            and structural_macro_request is None
            and self.state.detected_morphology == "snake8"
            and self.state.active_controller == "snake8"
            and self.state.authority == "TELEOP"
        ):
            structural_macro_request = "snake8"
            structural_macro_kind = "snake_stairs"

        # X is deliberately resolved on release.  This leaves one or more Joy
        # packets for a D-pad direction to turn the gesture into assembly /
        # reconfiguration without accidentally starting gap_crossing first.
        if (
            self._snake_gap_candidate
            and "cross" not in sample.buttons
        ):
            launch_gap = (
                not self._snake_gap_chorded
                and structural_macro_request is None
                and self.state.detected_morphology == "snake8"
                and self.state.active_controller == "snake8"
                and self.state.authority == "TELEOP"
            )

            self._snake_gap_candidate = False
            self._snake_gap_chorded = False

            if launch_gap:
                structural_macro_request = "snake8"
                structural_macro_kind = "snake_gap"

        return {"schema_version": "mssr.teleop_status.v1", "stamp_monotonic": now,
                **self.state.status(),
                "structural_macro_request": structural_macro_request,
                "structural_macro_kind": structural_macro_kind,
                "controller_input": input_payload(sample),
                "events": events, "rejected_commands": rejected,
                "valid_joy_packets": self._valid_packets,
                "invalid_joy_packets": self._invalid_packets}
