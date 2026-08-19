"""Validated synchronous client for one morphology behavior command."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from mssr_expert.execution.morphology_behavior_executor import (
    MorphologyCommand,
)
from mssr_expert.utils.json_io import dict_to_string_msg, string_msg_to_dict


def parse_parameters_json(text: str) -> dict[str, Any]:
    """Parse a behavior parameter object without shell-side JSON assembly."""

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid --parameters-json: {error}") from error
    if not isinstance(payload, Mapping):
        raise ValueError("--parameters-json must contain one JSON object")
    return dict(payload)


def build_command_payload(
    *,
    command_id: str,
    morphology: str,
    behavior: str,
    parameters: Mapping[str, Any],
) -> dict[str, Any]:
    """Build and validate the exact payload consumed by the behavior node."""

    payload = {
        "schema_version": "mssr.morphology_command.v1",
        "command_id": command_id,
        "morphology": morphology,
        "behavior": behavior,
        "parameters": dict(parameters),
    }
    # Keep client and server validation identical.
    MorphologyCommand.from_mapping(payload)
    return payload


@dataclass(frozen=True)
class TerminalBehaviorStatus:
    success: bool
    state: str
    phase: str
    message: str


class SmoresMorphologyCommandClient(Node):
    """Publish one command and wait for its matching terminal status."""

    def __init__(self, payload: Mapping[str, Any]) -> None:
        command_id = str(payload["command_id"])
        safe_suffix = "".join(
            character if character.isalnum() else "_"
            for character in command_id
        )[-40:]
        super().__init__(f"smores_morphology_command_client_{safe_suffix}")
        self._command_id = command_id
        self._payload = dict(payload)
        self._terminal: TerminalBehaviorStatus | None = None
        self._last_signature: tuple[str, str, bool, bool] | None = None
        self._publisher = self.create_publisher(
            String,
            "/mssr/morphology/command",
            10,
        )
        self.create_subscription(
            String,
            "/mssr/morphology/status",
            self._on_status,
            10,
        )

    @property
    def behavior_subscription_count(self) -> int:
        return self._publisher.get_subscription_count()

    @property
    def terminal(self) -> TerminalBehaviorStatus | None:
        return self._terminal

    def publish(self) -> None:
        self._publisher.publish(dict_to_string_msg(self._payload))

    def _on_status(self, message: String) -> None:
        payload = string_msg_to_dict(message)
        if str(payload.get("command_id", "")) != self._command_id:
            return
        state = str(payload.get("state", "UNKNOWN"))
        phase = str(payload.get("phase", "UNKNOWN"))
        done = bool(payload.get("done", False))
        success = bool(payload.get("success", False))
        signature = (state, phase, done, success)
        if signature != self._last_signature:
            progress = float(payload.get("progress", 0.0))
            detail = str(payload.get("message", ""))
            print(
                f"[{state}/{phase}] progress={progress:.0%} {detail}",
                flush=True,
            )
            self._last_signature = signature
        if done:
            self._terminal = TerminalBehaviorStatus(
                success=success,
                state=state,
                phase=phase,
                message=str(payload.get("message", "")),
            )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Publish one validated SMORES morphology command and wait for "
            "the matching terminal status."
        )
    )
    parser.add_argument("--morphology", required=True)
    parser.add_argument("--behavior", required=True)
    parser.add_argument("--command-id", required=True)
    parser.add_argument("--parameters-json", default="{}")
    parser.add_argument("--discovery-timeout-s", type=float, default=5.0)
    parser.add_argument("--timeout-s", type=float, default=180.0)
    return parser


def main(args: Sequence[str] | None = None) -> None:
    parsed = build_argument_parser().parse_args(args)
    if parsed.discovery_timeout_s <= 0.0 or parsed.timeout_s <= 0.0:
        raise SystemExit("Timeouts must be positive")
    try:
        payload = build_command_payload(
            command_id=parsed.command_id,
            morphology=parsed.morphology,
            behavior=parsed.behavior,
            parameters=parse_parameters_json(parsed.parameters_json),
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error

    # argparse already consumed every client option; keep them out of the ROS
    # argument parser so values such as JSON objects remain entirely opaque.
    rclpy.init(args=[])
    node = SmoresMorphologyCommandClient(payload)
    try:
        discovery_deadline = time.monotonic() + parsed.discovery_timeout_s
        while (
            node.behavior_subscription_count == 0
            and time.monotonic() < discovery_deadline
        ):
            rclpy.spin_once(node, timeout_sec=0.1)
        if node.behavior_subscription_count == 0:
            raise SystemExit(
                "No subscriber on /mssr/morphology/command; start the "
                "morphology behavior node first."
            )

        print(json.dumps(payload, sort_keys=True), flush=True)
        node.publish()
        deadline = time.monotonic() + parsed.timeout_s
        while node.terminal is None and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        terminal = node.terminal
        if terminal is None:
            raise SystemExit(
                f"Timed out waiting for command {parsed.command_id!r}."
            )
        if not terminal.success:
            raise SystemExit(
                f"Behavior failed: {terminal.state}/{terminal.phase}: "
                f"{terminal.message}"
            )
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
