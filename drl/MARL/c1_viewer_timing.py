"""Opt-in passive host profiler for C1 viewer diagnostics.

Measurements are host-wall clock, not CUDA kernel execution times.  Timed
wrappers call the original function exactly once, preserve the return object,
and are fully detached after profiling.  No RNG or sampler calls are added.
"""
from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from functools import wraps
import json
import math
from pathlib import Path
import statistics
import time


def _quantiles(values):
    values = sorted(float(value) for value in values)
    if not values:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    count = len(values)
    return {
        "count": count,
        "mean": sum(values) / count,
        "p50": statistics.median(values),
        "p95": values[max(0, math.ceil(0.95 * count) - 1)],
        "max": values[-1],
    }


class HostTimingProbe:
    """Collect inclusive/exclusive spans, inter-render intervals and support."""

    def __init__(self, folder, *, clock=None):
        self.folder = Path(folder)
        self.clock = clock if clock is not None else time.perf_counter
        self._stack = []
        self._stage_times = defaultdict(list)
        self._exclusive_times = defaultdict(list)
        self._render_last_start = {}
        self._render_intervals = defaultdict(list)
        self._wrappers = []
        self._decision_start = None
        self._decision_count = 0
        self._decision_durations = []
        self._support_rows = []

    @contextmanager
    def span(self, name):
        start = self.clock()
        frame = {"name": str(name), "start": start, "child": 0.0}
        self._stack.append(frame)
        try:
            yield
        finally:
            elapsed = max(0.0, self.clock() - frame["start"])
            popped = self._stack.pop()
            if popped is not frame:
                raise RuntimeError("Profiler spans ended out of order")
            self._stage_times[frame["name"]].append(elapsed)
            self._exclusive_times[frame["name"]].append(
                max(0.0, elapsed - frame["child"])
            )
            if self._stack:
                self._stack[-1]["child"] += elapsed

    def wrap(self, obj, method_name, stage_name):
        original = getattr(obj, method_name)
        had_own_attribute = (
            hasattr(obj, "__dict__") and method_name in vars(obj)
        )

        @wraps(original)
        def wrapper(*args, **kwargs):
            with self.span(stage_name):
                return original(*args, **kwargs)

        setattr(obj, method_name, wrapper)
        self._wrappers.append((obj, method_name, original, had_own_attribute))
        return wrapper

    def detach(self):
        for obj, method_name, original, had_own_attribute in reversed(self._wrappers):
            if had_own_attribute:
                setattr(obj, method_name, original)
            else:
                delattr(obj, method_name)
        self._wrappers.clear()

    def begin_decision(self, env_index, simulation_time_s):
        if self._decision_start is not None:
            raise RuntimeError("Previous profiler decision was not ended")
        self._decision_start = self.clock()
        self._decision_count += 1

    def end_decision(self):
        if self._decision_start is None:
            raise RuntimeError("Profiler decision was not started")
        self._decision_durations.append(max(0.0, self.clock() - self._decision_start))
        self._decision_start = None

    def render_event(self, name, start_s, end_s):
        """Record call-start intervals; not an estimate of GPU frame duration."""
        name = str(name)
        start_s = float(start_s)
        end_s = float(end_s)
        if end_s < start_s:
            raise ValueError("A render call cannot end before it starts")
        previous = self._render_last_start.get(name)
        if previous is not None:
            if start_s < previous:
                raise ValueError("Render events must be chronological")
            self._render_intervals[name].append(start_s - previous)
        self._render_last_start[name] = start_s

    def observe_policy(self, output, action, availability):
        """Log support counts from the already sampled action (no resample)."""
        records = tuple(output["records"])
        mask = tuple(bool(value) for value in availability.candidate_mask)
        if len(mask) != len(records):
            raise ValueError("Availability size differs from candidate records")
        selected = action.presence.detach().cpu().tolist()
        if len(selected) != len(records):
            raise ValueError("Sampled presence size differs from records")
        row = {
            "available": {"resource": 0, "structural": 0},
            "sampled": {"resource": 0, "structural": 0},
            "masked": {"resource": 0, "structural": 0},
        }
        for record, masked in zip(records, mask, strict=True):
            kind = record.candidate_kind
            if kind not in row["available"]:
                raise ValueError(f"Unexpected candidate kind: {kind}")
            if masked:
                row["masked"][kind] += 1
            else:
                row["available"][kind] += 1
            if bool(selected[record.candidate_index]):
                row["sampled"][kind] += 1
        self._support_rows.append(row)
        self.folder.mkdir(parents=True, exist_ok=True)
        with (self.folder / "policy_support.jsonl").open("a", encoding="utf8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    def finish(self):
        if self._stack or self._decision_start is not None:
            raise RuntimeError("Cannot finalize a profiler with active spans")
        summary = {
            "decisions": self._decision_count,
            "decision_durations_s": _quantiles(self._decision_durations),
            "stages": {
                stage: {
                    "inclusive_s": _quantiles(values),
                    "exclusive_s": _quantiles(self._exclusive_times[stage]),
                }
                for stage, values in sorted(self._stage_times.items())
            },
            "render_intervals": {
                stage: {**{"p95_s": _quantiles(values)["p95"]},
                        "interval_s": _quantiles(values)}
                for stage, values in sorted(self._render_intervals.items())
            },
            "policy_support_samples": len(self._support_rows),
            "notes": {
                "cuda_synchronize_per_block": False,
                "time_source": "host_wall_clock_only",
                "render_intervals": "consecutive render-call starts; not displayed frames",
            },
        }
        self.folder.mkdir(parents=True, exist_ok=True)
        (self.folder / "host_timing_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf8"
        )
        return summary
