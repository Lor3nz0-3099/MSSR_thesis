"""Per-environment structural option clock for native C1 training.

The policy clock mirrors the validated C1 GUI event scheduler while preserving
30 Hz physics/reward rows for APPO temporal credit assignment.
"""
from __future__ import annotations

from dataclasses import dataclass

from drl.MARL.c1_gui_viewer import StructuralOptionScheduler, StructuralOptionState


@dataclass
class _EnvClock:
    scheduler: StructuralOptionScheduler
    generation: object = None
    state: StructuralOptionState | None = None
    last_decided: bool = False


class NativeStructuralOptionController:
    """Run one independent structural option scheduler per vector environment."""

    def __init__(self, env, *, mode: str):
        if mode not in ("event", "fixed"):
            raise ValueError("structural scheduler mode must be event or fixed")
        self.env = env
        self.mode = mode
        self._clocks = [
            _EnvClock(StructuralOptionScheduler(mode=mode))
            for _ in range(int(env.num_envs))
        ]
        self._metrics = dict(
            decision_rows=0,
            sleeping_rows=0,
            wave_seals=0,
            generation_resets=0,
            continuous_rows=0,
        )

    def _generation(self, env_index: int):
        generations = getattr(self.env, "generation", None)
        return generations[env_index] if generations is not None else None

    def _state(self, env_index: int) -> StructuralOptionState:
        dispatcher = getattr(self.env, "native_dispatcher", None)
        if dispatcher is None:
            return StructuralOptionState((), (), 0)
        executor = dispatcher.executor_provider(env_index)
        active = tuple(sorted(
            str(getattr(goal, "goal_id", repr(goal)))
            for goal in tuple(getattr(executor, "active_goals", ()))
        ))
        prepared = []
        snapshot = getattr(executor, "prepared_relation_snapshot", None)
        if callable(snapshot):
            for relation in snapshot():
                prepared.append((
                    relation.source_module,
                    relation.source_face,
                    relation.target_module,
                    relation.target_face,
                    bool(getattr(relation, "docked", False)),
                ))
        managers = getattr(self.env, "docking_managers", ())
        connections = (len(tuple(getattr(managers[env_index], "connections", ())))
                       if env_index < len(managers) else 0)
        return StructuralOptionState(active, tuple(sorted(prepared)), connections)

    def _reset_clock(self, env_index: int, generation):
        self._clocks[env_index] = _EnvClock(
            StructuralOptionScheduler(mode=self.mode), generation=generation
        )
        self._metrics["generation_resets"] += 1

    def decision_mask(self) -> tuple[bool, ...]:
        decisions = []
        for env_index, clock in enumerate(self._clocks):
            generation = self._generation(env_index)
            if clock.generation is None:
                clock.generation = generation
            elif generation != clock.generation:
                self._reset_clock(env_index, generation)
                clock = self._clocks[env_index]

            current = self._state(env_index)
            previous = clock.state
            if previous is not None:
                clock.scheduler.observe_step(previous, current)
                added = set(current.active_goal_ids) - set(previous.active_goal_ids)
                if clock.last_decided and current.active_goal_ids and not added:
                    dispatcher = getattr(self.env, "native_dispatcher", None)
                    if dispatcher is not None:
                        seal = getattr(dispatcher.executor_provider(env_index),
                                       "seal_active_align_wave", None)
                        if callable(seal):
                            seal()
                            self._metrics["wave_seals"] += 1

            decide = bool(clock.scheduler.should_decide(current))
            clock.state = current
            clock.last_decided = decide
            decisions.append(decide)
            self._metrics["decision_rows" if decide else "sleeping_rows"] += 1
            if getattr(clock.scheduler, "_continuous_control", False):
                self._metrics["continuous_rows"] += 1
        return tuple(decisions)

    def set_continuous_control(self, env_index: int, active: bool) -> None:
        self._clocks[env_index].scheduler.set_continuous_control(active)

    def metrics(self) -> dict:
        total = self._metrics["decision_rows"] + self._metrics["sleeping_rows"]
        return {
            **self._metrics,
            "mode": self.mode,
            "decision_fraction": (
                self._metrics["decision_rows"] / total if total else 0.0
            ),
        }


def decision_selects_resource(decision, context) -> bool:
    """Return True when this sampled ActionSet contains a resource candidate."""
    if decision is None or getattr(decision, "action", None) is None:
        return False
    routing = getattr(context, "routing_snapshot", None) or {}
    resource_count = len(routing.get("resource_candidates", ()))
    if resource_count <= 0:
        return False
    presence = getattr(decision.action, "presence", None)
    if presence is None:
        return False
    selected = presence[:resource_count]
    return bool(selected.any().item())
