"""Teleop runtime bridge for camera and structure-only emergency stop.

This adapter never pauses the Isaac timeline and never advances physics.
Dependencies are injected so the protocol is testable without Isaac.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import time


class TeleopRuntimeBridge:
    def __init__(
        self,
        request_file: Path,
        status_file: Path,
        *,
        structure_stop_callback=None,
        camera_callback=None,
        center_callback=lambda: (0.0, 0.0, 0.0),
    ) -> None:
        self.request_file = Path(request_file)
        self.status_file = Path(status_file)
        self._structure_stop = structure_stop_callback
        self._camera = camera_callback
        self._center = center_callback
        self._structure_stopped = False
        self._seen: dict[str, dict] = {}
        self._last_ack = None

    def _structure_stop_request(self, request):
        if request is None:
            return

        if not isinstance(request, dict):
            raise ValueError("structure_stop_request must be an object")

        request_id = request.get("id")
        active = request.get("active")

        if (
            not isinstance(request_id, str)
            or not request_id
            or len(request_id) > 128
        ):
            raise ValueError(
                "structure stop request requires a bounded nonempty id"
            )

        if not isinstance(active, bool):
            raise ValueError(
                "structure stop request active must be a bool"
            )

        if request_id in self._seen:
            ack = self._seen[request_id]

            if ack["active"] is not active:
                raise ValueError(
                    "structure stop request id cannot be reused "
                    "for another state"
                )

            self._last_ack = ack
            return

        if self._structure_stopped is not active:
            if self._structure_stop is None:
                self._last_ack = {
                    "id": request_id,
                    "active": active,
                    "applied": False,
                }
                return

            self._structure_stop(active)
            self._structure_stopped = active

        self._last_ack = {
            "id": request_id,
            "active": active,
            "applied": self._structure_stopped is active,
        }

        if self._last_ack["applied"]:
            self._seen[request_id] = self._last_ack

    def _apply_camera(self, camera):
        if camera is None:
            return False

        if not isinstance(camera, dict):
            raise ValueError("camera must be an object")

        values = [
            camera.get(name)
            for name in (
                "azimuth_rad",
                "elevation_rad",
                "radius_m",
            )
        ]

        if not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            for value in values
        ):
            raise ValueError("camera values must be finite")

        yaw, elevation, radius = values

        if (
            not 0 < elevation < math.pi / 2
            or not 0 < radius <= 100
        ):
            raise ValueError(
                "camera elevation/radius outside runtime bounds"
            )

        if self._camera is None:
            return False

        target = tuple(self._center())

        if (
            len(target) != 3
            or not all(math.isfinite(value) for value in target)
        ):
            raise ValueError(
                "camera center must contain three finite coordinates"
            )

        horizontal = radius * math.cos(elevation)
        eye = (
            target[0] + horizontal * math.cos(yaw),
            target[1] + horizontal * math.sin(yaw),
            target[2] + radius * math.sin(elevation),
        )

        self._camera(eye, target)
        return True

    def poll(self) -> dict:
        error = None
        camera_applied = False

        try:
            try:
                payload = json.loads(
                    self.request_file.read_text(encoding="utf-8")
                )
            except FileNotFoundError:
                payload = {
                    "schema_version": "mssr.teleop_runtime.v1"
                }

            if (
                not isinstance(payload, dict)
                or payload.get("schema_version")
                != "mssr.teleop_runtime.v1"
            ):
                raise ValueError(
                    "unsupported teleop runtime schema"
                )

            # Structural E-STOP has priority over camera validation.
            self._structure_stop_request(
                payload.get("structure_stop_request")
            )

            camera_applied = self._apply_camera(
                payload.get("camera")
            )

        except (OSError, ValueError, TypeError, KeyError) as exception:
            error = str(exception)

        status = {
            "schema_version": "mssr.teleop_runtime_status.v1",
            "stamp_monotonic": time.monotonic(),
            "structure_stopped": self._structure_stopped,
            "structure_stop_ack": self._last_ack,
            "camera_applied": camera_applied,
            "error": error,
        }

        self.status_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary = self.status_file.with_suffix(
            self.status_file.suffix + ".tmp"
        )

        temporary.write_text(
            json.dumps(status, allow_nan=False) + "\n",
            encoding="utf-8",
        )

        temporary.replace(self.status_file)
        return status
