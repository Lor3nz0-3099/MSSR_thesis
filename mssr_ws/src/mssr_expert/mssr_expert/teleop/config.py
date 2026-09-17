"""Validated configuration shared by the external shell and Joy launch."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from pathlib import Path

import yaml


def control_rate(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("rate must be a finite number in (0,1000]") from error
    if isinstance(value, bool) or not math.isfinite(number) or not 0 < number <= 1000:
        raise ValueError("rate must be a finite number in (0,1000]")
    return number


@dataclass(frozen=True)
class TeleopConfig:
    control_rate_hz: float
    dataset_rate_hz: float
    joy_topic: str
    device_id: int
    device_name: str
    autorepeat_rate: float

    def joy_parameters(self) -> dict:
        return {"device_id": self.device_id, "device_name": self.device_name,
                "deadzone": 0.0, "sticky_buttons": False,
                "autorepeat_rate": self.autorepeat_rate}


def load_teleop_config(path: str | Path) -> TeleopConfig:
    try:
        with Path(path).open(encoding="utf-8") as stream:
            payload = yaml.safe_load(stream)
    except yaml.YAMLError as error:
        raise ValueError(f"invalid teleop YAML: {path}") from error
    if not isinstance(payload, Mapping):
        raise ValueError("teleop configuration must be a mapping")
    try:
        joy = payload["joy"]
        if not isinstance(joy, Mapping):
            raise ValueError("joy configuration must be a mapping")
        topic = joy["topic"]
        if not isinstance(topic, str) or not topic.startswith("/") or topic == "/":
            raise ValueError("Joy topic must be an absolute ROS topic")
        if joy["deadzone"] != 0.0 or joy["sticky_buttons"] is not False:
            raise ValueError("Joy requires deadzone=0.0 and sticky_buttons=false")
        device_id, name = joy["device_id"], joy["device_name"]
        if isinstance(device_id, bool) or not isinstance(device_id, int) or device_id < 0:
            raise ValueError("device_id must be a nonnegative integer")
        if not isinstance(name, str):
            raise ValueError("device_name must be a string")
        return TeleopConfig(control_rate(payload["control_rate_hz"]),
                            control_rate(payload["dataset_rate_hz"]), topic,
                            device_id, name, control_rate(joy["autorepeat_rate"]))
    except KeyError as error:
        raise ValueError(f"missing teleop configuration field: {error.args[0]}") from error
