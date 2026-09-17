"""Normalize existing ROS Joy messages without implementing a device driver.

Receipt timestamps and snapshot times must share a monotonic wall clock.
Button events retain order and their modifier context until consumed; set
views are convenient for diagnostics, but toggles must use command_events.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from pathlib import Path
from types import MappingProxyType

import yaml


STICKS = ("left_x", "left_y", "right_x", "right_y")
TRIGGERS = ("l2", "r2")
COMMANDS = frozenset({
    "record_toggle", "select_rc", "select_snake", "select_mm8", "home",
    "estop", "resume", "estop_toggle", "override", "previous_module", "next_module",
    "manual_pan_positive", "manual_pan_negative", "manual_tilt_positive",
    "manual_tilt_negative",
})


def _number(value: object, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite number") from error
    if isinstance(value, bool) or not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _index(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("binding index must be a nonnegative integer")
    return value


def _mapping(value: object, name: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


@dataclass(frozen=True)
class AxisBinding:
    index: int
    sign: float


@dataclass(frozen=True)
class TriggerBinding:
    index: int
    released: float
    pressed: float


@dataclass(frozen=True)
class InputConfig:
    deadzone: float
    trigger_deadzone: float
    disconnect_timeout_s: float
    axes: Mapping[str, AxisBinding]
    triggers: Mapping[str, TriggerBinding]
    buttons: Mapping[str, int]
    commands: Mapping[str, str | None]

    @classmethod
    def from_mapping(cls, payload: Mapping) -> InputConfig:
        payload = _mapping(payload, "input configuration")
        try:
            deadzone = _number(payload["deadzone"], "deadzone")
            trigger_deadzone = _number(payload["trigger_deadzone"], "trigger_deadzone")
            timeout = _number(payload["disconnect_timeout_s"], "disconnect_timeout_s")
            if not 0 <= deadzone < 1 or not 0 <= trigger_deadzone < 1 or timeout <= 0:
                raise ValueError("deadzones must be in [0,1); timeout must be positive")
            raw_axes = _mapping(payload["axes"], "axes")
            raw_triggers = _mapping(payload["triggers"], "triggers")
            if set(raw_axes) != set(STICKS) or set(raw_triggers) != set(TRIGGERS):
                raise ValueError("exactly four named sticks and l2/r2 are required")
            axes = {}
            for name in STICKS:
                binding = _mapping(raw_axes[name], name)
                sign = _number(binding["sign"], "axis sign")
                if sign not in (-1.0, 1.0):
                    raise ValueError("axis sign must be -1 or 1")
                axes[name] = AxisBinding(_index(binding["index"]), sign)
            triggers = {}
            for name in TRIGGERS:
                binding = _mapping(raw_triggers[name], name)
                released = _number(binding["released"], "released endpoint")
                pressed = _number(binding["pressed"], "pressed endpoint")
                if not -1 <= released <= 1 or not -1 <= pressed <= 1 or released == pressed:
                    raise ValueError("trigger endpoints must be distinct and within [-1,1]")
                triggers[name] = TriggerBinding(_index(binding["index"]), released, pressed)
            indices = [binding.index for binding in (*axes.values(), *triggers.values())]
            if len(set(indices)) != len(indices):
                raise ValueError("stick and trigger axes must be distinct")
            buttons = dict(_mapping(payload["buttons"], "buttons"))
            if "start" not in buttons or any(not isinstance(name, str) or not name for name in buttons):
                raise ValueError("named buttons including start are required")
            for index in buttons.values():
                _index(index)
            if len(set(buttons.values())) != len(buttons):
                raise ValueError("physical button indices must be distinct")
            commands = dict(_mapping(payload["commands"], "commands"))
            if set(commands) - COMMANDS:
                raise ValueError("unknown semantic command")
            if commands.get("record_toggle") != "start":
                raise ValueError("record_toggle must use the approved start button")
            assigned = []
            for button in commands.values():
                if button is not None:
                    if not isinstance(button, str) or button not in buttons:
                        raise ValueError("command must refer to a named button or null")
                    assigned.append(button)
            if len(set(assigned)) != len(assigned):
                raise ValueError("semantic commands must use distinct physical buttons")
            return cls(deadzone, trigger_deadzone, timeout,
                       MappingProxyType(axes), MappingProxyType(triggers),
                       MappingProxyType(buttons), MappingProxyType(commands))
        except KeyError as error:
            raise ValueError(f"missing input configuration field: {error.args[0]}") from error


def load_input_config(path: str | Path) -> InputConfig:
    with Path(path).open(encoding="utf-8") as stream:
        try:
            payload = yaml.safe_load(stream)
        except yaml.YAMLError as error:
            raise ValueError(f"invalid input YAML: {path}") from error
    return InputConfig.from_mapping(payload)


@dataclass(frozen=True)
class ButtonEvent:
    button: str
    pressed_buttons: frozenset[str]
    received_at: float


@dataclass(frozen=True)
class InputSnapshot:
    left_x: float = 0.0
    left_y: float = 0.0
    right_x: float = 0.0
    right_y: float = 0.0
    l2: float = 0.0
    r2: float = 0.0
    buttons: frozenset[str] = frozenset()
    rising_edges: frozenset[str] = frozenset()
    command_edges: frozenset[str] = frozenset()
    command_events: tuple[str, ...] = ()
    button_events: tuple[ButtonEvent, ...] = ()
    last_message_at: float | None = None
    connected: bool = False


def _deadzone(value: float, deadzone: float) -> float:
    value = max(-1.0, min(1.0, value))
    if abs(value) <= deadzone:
        return 0.0
    return math.copysign((abs(value) - deadzone) / (1.0 - deadzone), value)


class DualSenseInput:
    """Single-threaded state holder called by the ROS executor/control loop."""

    def __init__(self, config: InputConfig) -> None:
        self.config = config
        self._last_message_at: float | None = None
        self._values = {name: 0.0 for name in (*STICKS, *TRIGGERS)}
        self._buttons: frozenset[str] = frozenset()
        self._events: list[ButtonEvent] = []

    def _connected(self, now: float) -> bool:
        return self._last_message_at is not None and (
            now < self._last_message_at + self.config.disconnect_timeout_s
        )

    def update(self, axes: Sequence[float], buttons: Sequence[int], received_at: float) -> bool:
        """Accept one complete valid packet; reject atomically without refreshing time."""
        try:
            received_at = _number(received_at, "receipt time")
            if self._last_message_at is not None and received_at < self._last_message_at:
                return False
            axis_values = [float(value) for value in axes]
            if not all(math.isfinite(value) for value in axis_values):
                return False
            if any(value not in (0, 1) for value in buttons):
                return False
            values = {}
            for name, binding in self.config.axes.items():
                values[name] = _deadzone(axis_values[binding.index] * binding.sign,
                                         self.config.deadzone)
            for name, binding in self.config.triggers.items():
                value = (axis_values[binding.index] - binding.released) / (
                    binding.pressed - binding.released
                )
                values[name] = max(0.0, _deadzone(value, self.config.trigger_deadzone))
            pressed = frozenset(name for name, index in self.config.buttons.items()
                                if buttons[index])
        except (TypeError, ValueError, OverflowError, IndexError):
            return False
        if self._connected(received_at):
            for name in self.config.buttons:
                if name in pressed and name not in self._buttons:
                    self._events.append(ButtonEvent(name, pressed, received_at))
        else:
            # First packet establishes state; held buttons do not become commands.
            self._events.clear()
        self._buttons = pressed
        self._values = values
        self._last_message_at = received_at
        return True

    def snapshot(self, now: float) -> InputSnapshot:
        """Consume edge events once. Held axes remain available as recording context.

        A caller MUST gate actuator commands on connected. A stale snapshot
        never contains command edges, even when a button was pressed before loss.
        """
        now = _number(now, "snapshot time")
        connected = self._connected(now)
        events = tuple(self._events) if connected else ()
        self._events.clear()
        reverse_commands = {button: command for command, button in self.config.commands.items()
                            if button is not None}
        commands = tuple(reverse_commands[event.button] for event in events
                         if event.button in reverse_commands)
        return InputSnapshot(**self._values, buttons=self._buttons,
                             rising_edges=frozenset(event.button for event in events),
                             command_edges=frozenset(commands), command_events=commands,
                             button_events=events, last_message_at=self._last_message_at,
                             connected=connected)
