"""Ordered idempotent structure-stop delivery over the runtime channel."""
from collections import deque
import math
from uuid import uuid4


class RuntimeChannel:
    def __init__(self, timeout_s=0.5):
        self._timeout = timeout_s
        self._pending = deque()
        self._stamp = None
        self.structure_stopped = None

    def request_stop(self, active: bool) -> None:
        if not isinstance(active, bool):
            raise ValueError("structure stop state must be a bool")

        # Emergency STOP always has priority over a pending clear.
        if active:
            pending_stop = (
                self._pending[0]
                if self._pending and self._pending[0]["active"] is True
                else None
            )
            self._pending.clear()

            # Repeated STOP retries the same request id.
            if pending_stop is not None:
                self._pending.append(pending_stop)
                return

        # Avoid queuing the same desired state repeatedly.
        if self._pending and self._pending[-1]["active"] is active:
            return

        self._pending.append(
            {
                "id": uuid4().hex,
                "active": active,
            }
        )

    def payload(self, camera):
        return {
            "schema_version": "mssr.teleop_runtime.v1",
            "structure_stop_request": (
                dict(self._pending[0]) if self._pending else None
            ),
            "camera": camera,
        }

    def ready(self, now):
        return (
            self._stamp is not None
            and 0 <= now - self._stamp <= self._timeout
        )

    def observe(self, payload, now):
        if (
            not isinstance(payload, dict)
            or payload.get("schema_version")
            != "mssr.teleop_runtime_status.v1"
        ):
            return None

        stamp = payload.get("stamp_monotonic")
        stopped = payload.get("structure_stopped")

        if (
            isinstance(stamp, bool)
            or not isinstance(stamp, (float, int))
            or not math.isfinite(stamp)
            or not 0 <= now - stamp <= self._timeout
            or not isinstance(stopped, bool)
        ):
            return None

        self._stamp = stamp
        self.structure_stopped = stopped

        ack = payload.get("structure_stop_ack")

        if not self._pending or not isinstance(ack, dict):
            return None

        request = self._pending[0]

        if (
            ack.get("id") == request["id"]
            and ack.get("active") is request["active"]
            and ack.get("applied") is True
            and self.structure_stopped is request["active"]
        ):
            completed = self._pending.popleft()
            return "stop" if completed["active"] else "clear"

        return None
