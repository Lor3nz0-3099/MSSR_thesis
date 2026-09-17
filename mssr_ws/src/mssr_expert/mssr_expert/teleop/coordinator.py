"""Coordinate runtime acknowledgments, input, state and safety without ROS."""
from dataclasses import asdict
from types import SimpleNamespace

from mssr_expert.teleop.camera import CameraController
from mssr_expert.teleop.runtime_channel import RuntimeChannel
from mssr_expert.teleop.safety import SafetyGate


class RuntimeCoordinator:
    def __init__(self, session, *, camera=None):
        self.session = session
        self.camera = camera if camera is not None else CameraController()
        self.runtime = RuntimeChannel()
        self.safety = SafetyGate()
        self._paused = False
        self._previous_tick = None

    def observe_runtime(self, payload, now):
        operation = self.runtime.observe(payload, now)
        if operation == "resume":
            self.safety.resume(resumed_at=now)
            self.session.state.resume()
            self._paused = False
        elif self.runtime.ready(now) and self.runtime.timeline_playing is False:
            self.safety.pause()
            self.session.state.pause()
            self._paused = True

    def tick(self, now):
        was_paused = self._paused
        status = self.session.tick(now)
        sample = status["controller_input"]
        commands = sample["command_events"]
        if "estop" in commands:
            self.safety.pause()
            self._paused = True
            self.runtime.request("pause")
        elif "resume" in commands and was_paused:
            self.runtime.request("resume")
        # Session handles intent; only the matching runtime acknowledgment clears ESTOP.
        if self._paused:
            self.session.state.pause()
        ready = self.runtime.ready(now)
        decision = self.safety.update(
            connected=sample["connected"] and ready and self.runtime.timeline_playing is True,
            l2=sample["l2"], r2=sample["r2"], received_at=sample["last_message_at"],
            macro_active=self.session.state.macro_active,
            topology_supported=self.session.state.active_controller is not None)
        dt = 0.0 if self._previous_tick is None else now - self._previous_tick
        orbit = self.camera.step(SimpleNamespace(**sample), dt)
        self._previous_tick = now
        status.update(self.session.state.status())
        status.update(runtime_bridge_ready=ready, safety=asdict(decision))
        return status, self.runtime.payload(asdict(orbit))
