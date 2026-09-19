"""Launch deterministic structural experts for teleoperation."""

from __future__ import annotations

from pathlib import Path
import os
import signal
import subprocess
from typing import Callable, Sequence

from mssr_expert.teleop.state import TeleopState


Command = tuple[str, ...]


def _default_spawn(command: Sequence[str]):
    return subprocess.Popen(
        list(command),
        start_new_session=True,
    )


def _default_terminate(process) -> None:
    """Terminate only the structural expert process group we created."""

    pid = getattr(process, "pid", None)

    if (
        isinstance(pid, int)
        and not isinstance(pid, bool)
        and pid > 0
    ):
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            # The complete group has already disappeared.
            pass
        return

    # Test doubles or non-POSIX-like process handles may not expose a PID.
    terminate = getattr(process, "terminate", None)

    if callable(terminate):
        terminate()


class StructuralMacroLauncher:
    """Own one externally launched structural expert process."""

    def __init__(
        self,
        *,
        spawn: Callable[[Sequence[str]], object] | None = None,
    ) -> None:
        self._spawn = spawn if spawn is not None else _default_spawn
        self._process = None
        self._target_morphology: str | None = None
        self._active_goal_ids: tuple[str, ...] = ()
        self._process_exit_observed_at: float | None = None

    @property
    def process(self):
        return self._process

    @property
    def target_morphology(self) -> str | None:
        return self._target_morphology

    @property
    def active(self) -> bool:
        process = self._process
        return (
            process is not None
            and process.poll() is None
        )

    def start(
        self,
        *,
        state: TeleopState,
        target_morphology: str,
        execution_id: str,
        episode_id: str,
        dataset_path: Path,
    ):
        """Spawn reconfiguration, then claim macro authority."""

        if self.active:
            raise RuntimeError("structural macro already active")

        target = str(target_morphology)

        if state.requested_morphology != target:
            raise RuntimeError(
                "target morphology does not match the pending "
                "teleop request"
            )

        command: Command = (
            "ros2",
            "run",
            "mssr_expert",
            "mssr_smores_self_reconfiguration_node",
            "--ros-args",
            "-p",
            "source_graph_path:=auto",
            "-p",
            f"target_morphology:={target}",
            "-p",
            f"execution_id:={execution_id}",
            "-p",
            f"episode_id:={episode_id}",
            "-p",
            f"dataset_path:={Path(dataset_path)}",
        )

        # Do not change authority before a process exists.
        process = self._spawn(command)

        if process.poll() is not None:
            raise RuntimeError(
                "structural macro process exited during launch"
            )

        if not state.begin_macro():
            # A state transition may have occurred while spawning.
            _default_terminate(process)
            raise RuntimeError(
                "teleop state rejected structural macro start"
            )

        self._process = process
        self._target_morphology = target
        self._active_goal_ids = ()
        self._process_exit_observed_at = None
        return process

    def observe_expert_state(
        self,
        *,
        state: TeleopState,
        payload,
    ) -> bool:
        """Consume the matching expert terminal state exactly once."""

        # A terminal ROS packet may be delivered just after the expert
        # process exits. Ownership, not poll()==None, determines whether
        # this launcher may still consume that final state.
        if self._process is None or self._target_morphology is None:
            return False

        if not isinstance(payload, dict):
            return False

        if (
            payload.get("schema_version")
            != "mssr.self_reconfiguration_state.v1"
        ):
            return False

        if payload.get("target_morphology") != self._target_morphology:
            return False

        active_goal_ids = payload.get("active_goal_ids", ())

        if (
            isinstance(active_goal_ids, (list, tuple))
            and all(
                isinstance(goal_id, str)
                and bool(goal_id)
                for goal_id in active_goal_ids
            )
        ):
            self._active_goal_ids = tuple(
                sorted(set(active_goal_ids))
            )

        if payload.get("done") is not True:
            return False

        success = payload.get("success")

        if not isinstance(success, bool):
            return False

        if not state.finish_macro(success):
            return False

        process = self._process

        _default_terminate(process)

        self._process = None
        self._target_morphology = None
        self._active_goal_ids = ()
        self._process_exit_observed_at = None
        return True

    def check_process(
        self,
        *,
        state: TeleopState,
        now: float,
        exit_grace_s: float,
    ) -> tuple[str, ...] | None:
        """Fail an exited expert only if no terminal ROS packet follows."""

        process = self._process

        if process is None or self._target_morphology is None:
            self._process_exit_observed_at = None
            return None

        # A live expert owns the macro normally.  If it appeared dead during
        # a previous poll but is live again, discard that stale observation.
        if process.poll() is None:
            self._process_exit_observed_at = None
            return None

        now = float(now)
        exit_grace_s = float(exit_grace_s)

        if exit_grace_s < 0.0:
            raise ValueError(
                "exit_grace_s must be nonnegative"
            )

        observed_at = self._process_exit_observed_at

        # Process death is not immediately authoritative: the final ROS state
        # packet may still be queued for delivery.
        if observed_at is None:
            self._process_exit_observed_at = now
            return None

        if now - observed_at <= exit_grace_s:
            return None

        cancel_goal_ids = tuple(
            sorted(self._active_goal_ids)
        )

        # Grace expired without a matching terminal packet.  This is a failed
        # macro, but the actually observed physical topology is left untouched.
        state.finish_macro(False)

        self._process = None
        self._target_morphology = None
        self._active_goal_ids = ()
        self._process_exit_observed_at = None

        return cancel_goal_ids

    def interrupt(self) -> tuple[str, ...]:
        """Stop an interrupted macro without inventing a terminal result.

        The returned goal IDs are still owned by the native primitive
        executor and must be canceled by the ROS transport.
        """

        cancel_goal_ids = tuple(
            sorted(self._active_goal_ids)
        )

        process = self._process

        if process is not None:
            _default_terminate(process)

        self._process = None
        self._target_morphology = None
        self._active_goal_ids = ()
        self._process_exit_observed_at = None

        return cancel_goal_ids
