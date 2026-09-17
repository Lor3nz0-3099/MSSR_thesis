"""Actuator permission and fresh-neutral latch, independent of transport.

A safe-hold decision preserves posture targets and commands zero wheels.
Structural macro authority is kept on disconnect; its own commands must not
be replaced by teleop safe hold. The Isaac timeline adapter enforces ESTOP.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class SafetyDecision:
    authority: str
    motion_enabled: bool
    zero_wheels: bool
    allow_joint_updates: bool
    preserve_targets: bool = True


class SafetyGate:
    def __init__(self) -> None:
        self._paused = False
        self._armed = False
        self._last_receipt: float | None = None
        self._neutral_after: float | None = None

    def pause(self) -> bool:
        if self._paused:
            return False
        self._paused = True
        self._disarm()
        return True

    def resume(self, *, resumed_at: float) -> bool:
        """Fence arming against the monotonic time of actual resume acknowledgment."""
        if (isinstance(resumed_at, bool) or not isinstance(resumed_at, (int, float))
                or not math.isfinite(resumed_at)):
            raise ValueError("resume time must be finite monotonic seconds")
        if not self._paused:
            return False
        self._paused = False
        self._disarm()
        self._neutral_after = max(resumed_at, self._neutral_after) if self._neutral_after is not None else resumed_at
        return True

    def _disarm(self) -> None:
        self._armed = False
        # Disconnect or invalid input must never move a resume fence backward.
        if self._last_receipt is not None:
            self._neutral_after = (max(self._neutral_after, self._last_receipt)
                                   if self._neutral_after is not None else self._last_receipt)

    def update(self, *, connected: bool, l2: float, r2: float,
               received_at: float | None, macro_active: bool = False,
               topology_supported: bool = False) -> SafetyDecision:
        valid = (isinstance(received_at, (int, float)) and not isinstance(received_at, bool)
                 and math.isfinite(received_at)
                 and all(isinstance(value, (int, float)) and not isinstance(value, bool)
                         and math.isfinite(value) and 0 <= value <= 1 for value in (l2, r2)))
        valid = valid and (self._last_receipt is None or received_at >= self._last_receipt)
        if valid:
            self._last_receipt = received_at
        if not connected or not valid or self._paused:
            self._disarm()
        elif not self._armed and l2 == 0.0 and r2 == 0.0:
            # A cached neutral snapshot cannot satisfy a post-resume latch.
            if self._neutral_after is None or received_at > self._neutral_after:
                self._armed = True
        if self._paused:
            return SafetyDecision("ESTOP", False, True, False)
        if macro_active:
            return SafetyDecision("STRUCTURAL_MACRO", False, False, False)
        enabled = bool(connected and valid and self._armed and topology_supported)
        return SafetyDecision("TELEOP" if enabled else "NONE", enabled, not enabled, enabled)
