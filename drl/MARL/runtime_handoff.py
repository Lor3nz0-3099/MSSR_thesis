"""Authority-safe deterministic Expert -> MARL runtime handoff.

The Expert's terminal success does not establish physical morphology.
MARL authority is granted only after a separate matching topology
observation.
"""

from __future__ import annotations

from enum import Enum


class RuntimeAuthority(str, Enum):
    NONE = "NONE"
    EXPERT_ASSEMBLY = "EXPERT_ASSEMBLY"
    MARL = "MARL"


class RuntimePhase(str, Enum):
    IDLE = "IDLE"
    ASSEMBLING = "ASSEMBLING"
    VERIFYING_TOPOLOGY = "VERIFYING_TOPOLOGY"
    MARL_CONTROL = "MARL_CONTROL"
    TERMINAL = "TERMINAL"
    FAILED = "FAILED"


class DirectMARLHandoff:
    """Own the authority state for one DirectMARL episode.

    This deliberately mirrors the teleoperation structural-macro rule:
    control ownership and observed physical topology are separate facts.
    """

    def __init__(self) -> None:
        self.reset()

    @property
    def authority(self) -> RuntimeAuthority:
        return self._authority

    @property
    def phase(self) -> RuntimePhase:
        return self._phase

    @property
    def target_morphology(self) -> str | None:
        return self._target_morphology

    @property
    def last_expert_success(self) -> bool | None:
        return self._last_expert_success

    @property
    def failed(self) -> bool:
        return self._phase is RuntimePhase.FAILED

    @property
    def marl_active(self) -> bool:
        return self._authority is RuntimeAuthority.MARL

    def reset(self) -> None:
        self._authority = RuntimeAuthority.NONE
        self._phase = RuntimePhase.IDLE
        self._target_morphology: str | None = None
        self._last_expert_success: bool | None = None

    def finish_episode(self) -> None:
        """Latch terminal authority closure until reset starts a new assembly."""
        if self.marl_active:
            self._authority = RuntimeAuthority.NONE
            self._phase = RuntimePhase.TERMINAL

    def begin_expert(
        self,
        target_morphology: str,
    ) -> bool:
        """Claim deterministic Expert authority for a new assembly.

        Runtime integration must call this only after the Expert has
        actually been launched successfully.
        """
        target = str(
            target_morphology
        ).strip()

        if not target:
            raise ValueError(
                "target_morphology must be a non-empty string"
            )

        if (
            self._phase is not RuntimePhase.IDLE
            or self._authority is not RuntimeAuthority.NONE
        ):
            return False

        self._target_morphology = target
        self._last_expert_success = None
        self._authority = RuntimeAuthority.EXPERT_ASSEMBLY
        self._phase = RuntimePhase.ASSEMBLING
        return True

    def observe_expert_terminal(
        self,
        success: bool,
    ) -> bool:
        """Consume one authoritative Expert terminal event."""
        if (
            self._phase is not RuntimePhase.ASSEMBLING
            or self._authority
            is not RuntimeAuthority.EXPERT_ASSEMBLY
        ):
            return False

        if not isinstance(
            success,
            bool,
        ):
            raise TypeError(
                "Expert terminal success must be bool"
            )

        self._last_expert_success = success

        # Expert authority ends at the terminal event regardless of result.
        self._authority = RuntimeAuthority.NONE

        if success:
            self._phase = RuntimePhase.VERIFYING_TOPOLOGY
        else:
            self._phase = RuntimePhase.FAILED

        return True

    def observe_topology(
        self,
        name: str | None,
    ) -> bool:
        """Promote to MARL only for the independently verified target."""
        if (
            self._phase
            is not RuntimePhase.VERIFYING_TOPOLOGY
            or self._authority is not RuntimeAuthority.NONE
            or self._last_expert_success is not True
        ):
            return False

        if (
            not isinstance(name, str)
            or name != self._target_morphology
        ):
            return False

        self._authority = RuntimeAuthority.MARL
        self._phase = RuntimePhase.MARL_CONTROL
        return True
