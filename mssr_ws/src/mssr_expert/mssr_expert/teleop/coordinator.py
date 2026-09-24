"""Coordinate runtime acknowledgments, input, state and safety without ROS."""
from dataclasses import asdict
from types import SimpleNamespace

from mssr_expert.teleop.camera import CameraController
from mssr_expert.teleop.runtime_channel import RuntimeChannel
from mssr_expert.teleop.safety import SafetyGate


class RuntimeCoordinator:
    def __init__(
        self,
        session,
        *,
        camera=None,
        structural_request_handler=None,
    ):
        self.session = session
        self.camera = camera if camera is not None else CameraController()
        self._structural_request_handler = structural_request_handler
        self.runtime = RuntimeChannel()
        self.safety = SafetyGate()
        self._estop_latched = False
        # Desired state tracks successive presses even while native ACKs wait.
        self._stop_requested = False
        self._previous_tick = None

    def observe_runtime(self, payload, now):
        operation = self.runtime.observe(payload, now)

        if operation == "clear":
            # Clear is accepted only after the runtime confirms that the
            # structure-stop state has actually been removed.
            self.safety.resume(resumed_at=now)
            self.session.state.resume()
            self._estop_latched = False
            self._stop_requested = False

        elif operation == "stop":
            if not self._estop_latched:
                self._stop_requested = True
            self.safety.pause()
            self.session.state.pause()
            self._estop_latched = True

        elif (
            self.runtime.ready(now)
            and self.runtime.structure_stopped is True
        ):
            # A stop observed directly from the runtime is authoritative too.
            # Fresh neutral input alone must never clear it.
            if not self._estop_latched:
                self._stop_requested = True
            self.safety.pause()
            self.session.state.pause()
            self._estop_latched = True

    def tick(self, now):
        was_stopped = self._estop_latched

        status = self.session.tick(now)
        sample = status["controller_input"]
        commands = sample["command_events"]

        if "estop" in commands:
            # E-STOP takes effect in the teleop safety layer immediately.
            # Runtime delivery then makes the stop authoritative at Isaac.
            self._request_stop(True)

        elif "resume" in commands and was_stopped:
            # Explicit resume requests a clear, but the latch remains set
            # until the matching runtime acknowledgment arrives.
            self._request_stop(False)

        else:
            for command in commands:
                if command == "estop_toggle":
                    self._request_stop(not self._stop_requested)

        if self._estop_latched:
            self.session.state.pause()

        # A structural selection must claim authority before the safety
        # decision for this same control tick.  Otherwise one final human
        # actuator command could leak after the reconfiguration has started.
        structural_request = status.get("structural_macro_request")
        structural_kind = status.get("structural_macro_kind")

        if (
            structural_request is not None
            and not self._estop_latched
            and self._structural_request_handler is not None
        ):
            self._structural_request_handler(
                structural_request,
                structural_kind,
            )

        ready = self.runtime.ready(now)

        decision = self.safety.update(
            connected=(
                sample["connected"]
                and ready
                and self.runtime.structure_stopped is False
            ),
            l2=sample["l2"],
            r2=sample["r2"],
            received_at=sample["last_message_at"],
            macro_active=self.session.state.macro_active,
            topology_supported=(
                self.session.state.active_controller is not None
            ),
        )

        dt = (
            0.0
            if self._previous_tick is None
            else now - self._previous_tick
        )

        orbit = self.camera.step(
            SimpleNamespace(**sample),
            dt,
        )

        self._previous_tick = now

        status.update(self.session.state.status())
        status.update(
            runtime_bridge_ready=ready,
            runtime_structure_stopped=self.runtime.structure_stopped,
            safety=asdict(decision),
        )

        return status, self.runtime.payload(asdict(orbit))

    def _request_stop(self, active):
        self._stop_requested = active
        if active:
            self.safety.pause()
            self._estop_latched = True
        self.runtime.request_stop(active)
