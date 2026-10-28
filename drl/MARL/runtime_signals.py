"""Common signals from captured native outcomes, with measurement provenance.

A zero raw field is meaningful only when its availability says measured_absent.
No sensor, reward delta, action count or mechanical work is invented here.
Sparse event/count mappings describe the caller's complete owner measurement;
missing mapping keys mean measured zero only when the mapping itself is supplied.

Instantiate one stagnation detector per environment/episode; reset on each reset.
The module performs no simulator reads and no action dispatch.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
import math
from typing import Any, Mapping, Sequence

from drl.MARL.phase1_reward import ActionOutcomeSignals


class SignalStatus(str, Enum):
    MEASURED_PRESENT = "measured_present"
    MEASURED_ABSENT = "measured_absent"
    NOT_INSTRUMENTED = "not_instrumented"
    INVALID = "invalid"


@dataclass(frozen=True)
class RuntimeSignalFrame:
    signals: ActionOutcomeSignals
    availability: Mapping[str, Mapping[str, Any]]
    diagnostics: Mapping[str, Any]


def _positive_interval(value: float, label: str) -> float:
    try:
        number = float(value)
    except (ValueError, TypeError) as error:
        raise ValueError(f"{label} must be finite and positive") from error
    if isinstance(value, bool) or not math.isfinite(number) or number <= 0:
        raise ValueError(f"{label} must be finite and positive")
    return number


def _owners(values) -> tuple[str, ...]:
    owners = tuple(values)
    if (not owners or any(not isinstance(owner, str) or not owner for owner in owners)
            or len(set(owners)) != len(owners)):
        raise ValueError("owner_ids must contain unique nonempty strings")
    return tuple(sorted(owners))


def _known(values, owners) -> frozenset[str]:
    values = frozenset(values)
    if any(not isinstance(value, str) for value in values) or not values.issubset(owners):
        raise ValueError("signal references unknown owner")
    return values


def _metadata(status, units, source, *, reason=None):
    result = dict(status=SignalStatus(status).value, units=units,
                  source=source, attribution="owner_and_team")
    if reason is not None:
        result["reason"] = reason
    return result


def _measured(present):
    return SignalStatus.MEASURED_PRESENT if present else SignalStatus.MEASURED_ABSENT


class StagnationDetector:
    """Net centroid translation over an exact physical-time window.

    XY or XYZ metres are accepted. The metric is endpoint displacement, not
    path length, reward change, actuator activity or task success. Oscillations
    with no net centroid motion are stagnant. An initial partial window is
    unavailable. A linearly interpolated centroid handles irregular intervals.
    State continuity is required to prevent reset jumps from entering a window.
    """

    def __init__(self, *, window_seconds=1.0, displacement_threshold_m=0.005):
        self.window_seconds = _positive_interval(window_seconds, "window_seconds")
        threshold = float(displacement_threshold_m)
        if not math.isfinite(threshold) or threshold < 0:
            raise ValueError("displacement_threshold_m must be finite and nonnegative")
        self.displacement_threshold_m = threshold
        self.reset()

    def reset(self):
        self._elapsed = 0.0
        self._samples = deque()
        self._previous_positions = None
        self._displacement = None

    @property
    def diagnostics(self):
        return dict(metric="centroid_displacement_m", window_seconds=self.window_seconds,
                    threshold_m=self.displacement_threshold_m, units="m",
                    elapsed_seconds=self._elapsed, displacement_m=self._displacement,
                    full_window=self._displacement is not None)

    @staticmethod
    def _positions(positions):
        if not isinstance(positions, Mapping):
            raise ValueError("module positions must be a mapping")
        owners = _owners(positions)
        points = {}
        dimensions = None
        for owner in owners:
            value = positions[owner]
            if not isinstance(value, (tuple, list)) or len(value) not in (2, 3):
                raise ValueError("positions must be finite XY or XYZ vectors in metres")
            try:
                point = tuple(float(item) for item in value)
            except (TypeError, ValueError) as error:
                raise ValueError("positions must be finite XY or XYZ vectors in metres") from error
            if not all(math.isfinite(item) for item in point):
                raise ValueError("positions must be finite XY or XYZ vectors in metres")
            if dimensions is not None and len(point) != dimensions:
                raise ValueError("position vector dimensions must match")
            dimensions = len(point)
            points[owner] = point
        return points

    def observe(self, previous_module_positions_m, next_module_positions_m, *, delta_seconds):
        dt = _positive_interval(delta_seconds, "delta_seconds")
        before = self._positions(previous_module_positions_m)
        after = self._positions(next_module_positions_m)
        if tuple(before) != tuple(after) or len(next(iter(before.values()))) != len(next(iter(after.values()))):
            raise ValueError("previous and next position owners/dimensions must match")
        if self._previous_positions is not None and before != self._previous_positions:
            raise ValueError("stagnation state continuity violated; reset per episode")
        dimensions = len(next(iter(before.values())))

        def centroid(points):
            return tuple(math.fsum(point[index] / len(points) for point in points.values())
                         for index in range(dimensions))

        before_centroid, after_centroid = centroid(before), centroid(after)
        elapsed = self._elapsed + dt
        # All validation above precedes the first history mutation.
        if not self._samples:
            self._samples.append((self._elapsed, before_centroid))
        self._samples.append((elapsed, after_centroid))
        self._elapsed = elapsed
        self._previous_positions = after
        if elapsed + 1e-12 < self.window_seconds:
            self._displacement = None
            return None
        cutoff = max(0.0, elapsed - self.window_seconds)
        while len(self._samples) > 2 and self._samples[1][0] <= cutoff:
            self._samples.popleft()
        t0, p0 = self._samples[0]
        t1, p1 = self._samples[1]
        fraction = max(0.0, min(1.0, (cutoff - t0) / (t1 - t0)))
        window_start = tuple(a + fraction * (b-a) for a,b in zip(p0,p1))
        self._displacement = math.dist(window_start, after_centroid)
        return self._displacement <= self.displacement_threshold_m


def collect_runtime_signals(
    *, owner_ids: Sequence[str], delta_seconds: float,
    proposal_outcome=None, proposal_source="native_admission_outcome",
    measured_collision_owner_ids=None, collision_source="physics_contact_sensor",
    redundant_action_count_by_owner=None, redundant_action_source=None,
    control_effort_by_owner=None, control_effort_units=None, control_effort_source=None,
    stagnation_detector: StagnationDetector | None = None,
    previous_module_positions_m=None, next_module_positions_m=None,
) -> RuntimeSignalFrame:
    """Attribute measured events and export explicit per-term availability.

    Outcomes accept the existing native outcome object or equivalent mapping.
    Missing measurements stay unavailable. Invalid numeric measurements produce
    availability INVALID and are excluded from ActionOutcomeSignals. Unknown
    owners and infrastructure rejections fail closed rather than becoming cost.
    Effort means exactly the units/source supplied by the caller; command-rate
    effort must not be labelled torque work or mechanical energy.
    ``time`` remains one decision in raw reward; physical dt is recorded here.
    """
    owners = _owners(owner_ids)
    dt = _positive_interval(delta_seconds, "delta_seconds")
    missing = SignalStatus.NOT_INSTRUMENTED
    availability = {
        "collision": _metadata(missing, "event", collision_source),
        "conflict": _metadata(missing, "event", proposal_source),
        "invalid": _metadata(missing, "event", proposal_source),
        "redundant_action": _metadata(missing, "count", redundant_action_source or "not_connected"),
        "control_effort": _metadata(missing, control_effort_units or "unknown", control_effort_source or "not_connected"),
        "stagnation": _metadata(missing, "event", "captured_pre_post_module_positions"),
        "time": _metadata(SignalStatus.MEASURED_PRESENT, "decision", "completed_physics_interval"),
    }
    diagnostics = dict(delta_seconds=dt, owner_ids=list(owners))
    collision, conflict, invalid = frozenset(), frozenset(), frozenset()
    redundant, effort = {}, {}
    stagnating = False
    if proposal_outcome is not None:
        def field(name, default=None):
            return proposal_outcome.get(name, default) if isinstance(proposal_outcome, Mapping) else getattr(proposal_outcome, name, default)
        accepted, code = field("accepted"), field("code")
        if type(accepted) is not bool:
            raise ValueError("proposal outcome accepted must be a bool")
        attributed = _known(field("owner_ids", ()), owners)
        if accepted:
            if code not in ("ACCEPTED", "EMPTY"):
                raise RuntimeError(f"Unexpected accepted proposal outcome: {code}")
        else:
            if code not in ("RESOURCE_BUSY", "INVALID_GOAL"):
                raise RuntimeError(f"Unexpected infrastructure rejection: {code}")
            if not attributed:
                raise ValueError("rejected policy proposal requires owner attribution")
            if code == "RESOURCE_BUSY":
                conflict = attributed
            else:
                invalid = attributed
        availability["conflict"] = _metadata(_measured(bool(conflict)), "event", proposal_source)
        availability["invalid"] = _metadata(_measured(bool(invalid)), "event", proposal_source)
        diagnostics["proposal"] = dict(accepted=accepted, code=code, owner_ids=sorted(attributed))
    if measured_collision_owner_ids is not None:
        collision = _known(measured_collision_owner_ids, owners)
        availability["collision"] = _metadata(_measured(bool(collision)), "event", collision_source)
    for term, values, source, units in (
        ("redundant_action", redundant_action_count_by_owner, redundant_action_source, "count"),
        ("control_effort", control_effort_by_owner, control_effort_source, control_effort_units),
    ):
        if values is None:
            continue
        if not isinstance(source, str) or not source.strip() or not isinstance(units, str) or not units.strip():
            raise ValueError(f"{term} requires explicit source and units")
        if not isinstance(values, Mapping):
            availability[term] = _metadata(SignalStatus.INVALID, units, source, reason="measurement must be an owner mapping")
            continue
        _known(values, owners)
        try:
            if term == "redundant_action":
                if any(type(value) is not int or value < 0 for value in values.values()):
                    raise ValueError("counts must be nonnegative integers")
                measured = dict(values)
            else:
                measured = {owner: float(value) for owner, value in values.items()}
                if any(not math.isfinite(value) or value < 0 for value in measured.values()):
                    raise ValueError("effort must be finite and nonnegative")
        except (TypeError, ValueError, OverflowError):
            availability[term] = _metadata(SignalStatus.INVALID, units, source, reason="invalid numeric measurement")
            continue
        availability[term] = _metadata(_measured(any(measured.values())), units, source)
        if term == "redundant_action":
            redundant = measured
        else:
            effort = measured
    if stagnation_detector is not None:
        try:
            for points in (previous_module_positions_m, next_module_positions_m):
                if not isinstance(points, Mapping) or set(points) != set(owners):
                    raise ValueError("stagnation positions must match signal owners")
            measured_stagnation = stagnation_detector.observe(
                previous_module_positions_m, next_module_positions_m, delta_seconds=dt)
        except (TypeError, ValueError, OverflowError) as error:
            availability["stagnation"] = _metadata(SignalStatus.INVALID, "event",
                "captured_pre_post_module_positions", reason=str(error))
        else:
            if measured_stagnation is None:
                availability["stagnation"]["reason"] = "physical_window_warmup"
            else:
                stagnating = measured_stagnation
                availability["stagnation"] = _metadata(_measured(stagnating), "event", "captured_pre_post_module_positions")
        diagnostics["stagnation"] = stagnation_detector.diagnostics
    return RuntimeSignalFrame(
        signals=ActionOutcomeSignals(collision_owner_ids=collision, conflict_owner_ids=conflict,
            invalid_owner_ids=invalid, redundant_action_count_by_owner=redundant,
            control_effort_by_owner=effort, stagnating=stagnating),
        availability=availability, diagnostics=diagnostics,
    )
