"""Action sources that provide complete per-step simulation actions."""

from __future__ import annotations

from typing import Protocol

from robots.actions import SimulationActions
from robots.command_sources import MultiModuleCommandSource
from robots.json_codec import actions_from_json


class ActionSource(Protocol):
    """Interface for objects that provide locomotion and magnetic actions."""

    def update(self) -> None:
        """Refresh the latest action set."""

    def get_actions(self) -> SimulationActions:
        """Return the latest simulation actions."""

    def close(self) -> None:
        """Release resources owned by the action source."""


class LocomotionCommandActionAdapter:
    """Wrap a locomotion-only source as a complete action source."""

    def __init__(self, source: MultiModuleCommandSource) -> None:
        """Store the locomotion source to adapt."""
        self._source = source

    def update(self) -> None:
        """Refresh the wrapped command source."""
        self._source.update()

    def get_actions(self) -> SimulationActions:
        """Return locomotion commands with no magnetic actions."""
        return SimulationActions(locomotion=self._source.get_commands())

    def close(self) -> None:
        """Release resources owned by the wrapped source."""
        self._source.close()


class JsonFileActionSource:
    """Read complete simulation actions from a JSON file."""

    def __init__(self, path: str, ignore_existing: bool = True) -> None:
        """Store the JSON action file path."""
        self._path = path
        self._last_payload: str | None = self._read_payload() if ignore_existing else None
        self._actions = SimulationActions()
        self._reset_consumed = False

    def update(self) -> None:
        """Reload the action file when its content changes.

        Invalid payloads are ignored so an external ROS 2 node cannot stop the
        simulator by publishing one malformed action message.
        """
        payload = self._read_payload()
        if payload is None:
            self._actions = SimulationActions()
            return

        if payload == self._last_payload:
            return

        self._last_payload = payload
        try:
            self._actions = actions_from_json(payload)
            self._reset_consumed = False
        except (KeyError, TypeError, ValueError):
            return

    def get_actions(self) -> SimulationActions:
        """Return the latest parsed simulation actions."""
        if self._actions.reset_requested and not self._reset_consumed:
            self._reset_consumed = True
            return self._actions
        if self._actions.reset_requested:
            return SimulationActions(
                locomotion=self._actions.locomotion,
                magnetic=self._actions.magnetic,
                reset_requested=False,
            )
        return self._actions

    def close(self) -> None:
        """No-op because the file is opened only during update."""

    def _read_payload(self) -> str | None:
        try:
            with open(self._path, encoding="utf-8") as action_file:
                return action_file.read()
        except FileNotFoundError:
            return None
