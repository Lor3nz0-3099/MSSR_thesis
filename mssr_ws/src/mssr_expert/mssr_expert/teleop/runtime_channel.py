"""Ordered idempotent timeline delivery over the separate runtime channel."""
from collections import deque
import math
from uuid import uuid4


class RuntimeChannel:
    def __init__(self, timeout_s=0.5):
        self._timeout = timeout_s
        self._pending = deque()
        self._stamp = None
        self.timeline_playing = None

    def request(self, operation: str) -> None:
        if operation not in {"pause", "resume"}:
            raise ValueError("timeline operation must be pause or resume")
        if operation == "pause":
            # ESTOP must be delivered immediately, superseding all resume intent.
            # Retain an already pending pause's id so retries remain idempotent.
            pending_pause = self._pending[0] if self._pending and self._pending[0]["operation"] == "pause" else None
            self._pending.clear()
            if pending_pause is not None:
                self._pending.append(pending_pause)
                return
        self._pending.append({"id": uuid4().hex, "operation": operation})

    def payload(self, camera):
        return {"schema_version": "mssr.teleop_runtime.v1", "timeline_request":
                dict(self._pending[0]) if self._pending else None, "camera": camera}

    def ready(self, now):
        return self._stamp is not None and 0 <= now - self._stamp <= self._timeout

    def observe(self, payload, now):
        if not isinstance(payload, dict) or payload.get("schema_version") != "mssr.teleop_runtime_status.v1":
            return None
        stamp = payload.get("stamp_monotonic")
        if (isinstance(stamp, bool) or not isinstance(stamp, (float, int)) or not math.isfinite(stamp)
                or not 0 <= now - stamp <= self._timeout or not isinstance(payload.get("timeline_playing"), bool)):
            return None
        self._stamp = stamp
        self.timeline_playing = payload["timeline_playing"]
        ack = payload.get("timeline_ack")
        if not self._pending or not isinstance(ack, dict):
            return None
        request = self._pending[0]
        if (ack.get("id") == request["id"] and ack.get("operation") == request["operation"]
                and ack.get("applied") is True and self.timeline_playing == (request["operation"] == "resume")):
            return self._pending.popleft()["operation"]
        return None
