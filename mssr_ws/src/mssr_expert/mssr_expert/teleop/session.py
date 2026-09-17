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
    def __init__(self, input_config: InputConfig) -> None:
        self.input = DualSenseInput(input_config)
        self.state = TeleopState()
        self.state.start_ready()
        self._previous_connected = False
        self._valid_packets = 0
        self._invalid_packets = 0

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
        selections = {_SELECTIONS[command] for command in commands if command in _SELECTIONS}
        if len(selections) > 1:
            rejected.append("ambiguous_morphology_selection")
        elif selections:
            if not self.state.request_morphology(next(iter(selections))):
                rejected.append("morphology_selection_unavailable")
        return {"schema_version": "mssr.teleop_status.v1", "stamp_monotonic": now,
                **self.state.status(), "controller_input": input_payload(sample),
                "events": events, "rejected_commands": rejected,
                "valid_joy_packets": self._valid_packets,
                "invalid_joy_packets": self._invalid_packets}
