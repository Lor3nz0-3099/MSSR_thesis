"""Separate runtime channel, polled on app updates even with paused physics.

The scenario owns application updates; this adapter never advances physics
and never writes robot actions. Dependencies are injected for offline tests.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import time


class TeleopRuntimeBridge:
    def __init__(self, timeline, request_file: Path, status_file: Path, *,
                 camera_callback=None, center_callback=lambda: (0.0, 0.0, 0.0)) -> None:
        self.timeline = timeline
        self.request_file = Path(request_file)
        self.status_file = Path(status_file)
        self._camera = camera_callback
        self._center = center_callback
        self._seen: dict[str, dict] = {}
        self._last_ack = None

    def _timeline_request(self, request):
        if request is None:
            return
        if not isinstance(request, dict):
            raise ValueError("timeline_request must be an object")
        request_id, operation = request.get("id"), request.get("operation")
        if not isinstance(request_id, str) or not request_id or len(request_id) > 128:
            raise ValueError("timeline request requires a bounded nonempty id")
        if operation not in {"pause", "resume"}:
            raise ValueError("timeline operation must be pause or resume")
        if request_id in self._seen:
            ack = self._seen[request_id]
            if ack["operation"] != operation:
                raise ValueError("timeline request id cannot be reused for another operation")
            self._last_ack = ack
            return
        if operation == "pause" and self.timeline.is_playing():
            self.timeline.pause()
        elif operation == "resume" and not self.timeline.is_playing():
            self.timeline.play()
        applied = self.timeline.is_playing() == (operation == "resume")
        self._last_ack = {"id": request_id, "operation": operation, "applied": applied}
        if applied:
            self._seen[request_id] = self._last_ack

    def _apply_camera(self, camera):
        if camera is None:
            return False
        if not isinstance(camera, dict):
            raise ValueError("camera must be an object")
        values = [camera.get(name) for name in ("azimuth_rad", "elevation_rad", "radius_m")]
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in values):
            raise ValueError("camera values must be finite")
        yaw, elevation, radius = values
        if not 0 < elevation < math.pi / 2 or not 0 < radius <= 100:
            raise ValueError("camera elevation/radius outside runtime bounds")
        if self._camera is None:
            return False
        target = tuple(self._center())
        if len(target) != 3 or not all(math.isfinite(v) for v in target):
            raise ValueError("camera center must contain three finite coordinates")
        horizontal = radius * math.cos(elevation)
        eye = (target[0] + horizontal * math.cos(yaw), target[1] + horizontal * math.sin(yaw),
               target[2] + radius * math.sin(elevation))
        self._camera(eye, target)
        return True

    def poll(self) -> dict:
        error = None
        camera_applied = False
        try:
            try:
                payload = json.loads(self.request_file.read_text(encoding="utf-8"))
            except FileNotFoundError:
                payload = {"schema_version": "mssr.teleop_runtime.v1"}
            if not isinstance(payload, dict) or payload.get("schema_version") != "mssr.teleop_runtime.v1":
                raise ValueError("unsupported teleop runtime schema")
            # Pause is applied before camera validation so malformed orbit data cannot suppress ESTOP.
            self._timeline_request(payload.get("timeline_request"))
            camera_applied = self._apply_camera(payload.get("camera"))
        except (OSError, ValueError, TypeError, KeyError) as exception:
            error = str(exception)
        status = {"schema_version": "mssr.teleop_runtime_status.v1", "stamp_monotonic": time.monotonic(),
                  "timeline_playing": bool(self.timeline.is_playing()), "timeline_ack": self._last_ack,
                  "camera_applied": camera_applied, "error": error}
        self.status_file.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.status_file.with_suffix(self.status_file.suffix + ".tmp")
        temporary.write_text(json.dumps(status, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(self.status_file)
        return status
