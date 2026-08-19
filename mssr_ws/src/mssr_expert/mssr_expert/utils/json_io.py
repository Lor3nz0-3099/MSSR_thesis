"""JSON helpers for ROS 2 String payloads and offline files."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from std_msgs.msg import String


def dumps_json(payload: dict[str, Any]) -> str:
    """Serialize a payload with stable key order for deterministic logs."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def loads_json(text: str) -> dict[str, Any]:
    """Deserialize JSON text into a dict, returning empty dict on invalid input."""
    try:
        payload = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def dict_to_string_msg(payload: dict[str, Any]) -> String:
    """Wrap a dict as a std_msgs/String JSON message."""
    message = String()
    message.data = dumps_json(payload)
    return message


def string_msg_to_dict(message: String) -> dict[str, Any]:
    """Read a std_msgs/String JSON message as a dict."""
    return loads_json(message.data)


def read_json_file(path: Path) -> dict[str, Any]:
    """Read a JSON file as a dict, returning empty dict if missing or invalid."""
    try:
        return loads_json(path.read_text(encoding="utf-8"))
    except OSError:
        return {}


def write_json_file(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON atomically enough for the file bridge workflow."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(dumps_json(payload), encoding="utf-8")
    temporary_path.replace(path)
