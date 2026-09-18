"""Opt-in assembly profiling with periodic snapshots and shutdown saves."""
import argparse
import cProfile
import math
from pathlib import Path
import pstats
import runpy
import signal
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--snapshot-interval", type=float, default=5.0)
    parser.add_argument("native_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not math.isfinite(args.snapshot_interval) or args.snapshot_interval <= 0:
        parser.error("snapshot interval must be positive and finite")
    native_args = args.native_args
    if native_args[:1] == ["--"]:
        native_args = native_args[1:]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    profiler = cProfile.Profile()
    recording = False

    def save():
        # All profiler operations run on the measured main thread, with
        # recording disabled. Report generation cannot pollute native timings.
        with tempfile.TemporaryDirectory(dir=args.output.parent, prefix=".assembly-profile-") as folder:
            binary = Path(folder) / "stats.pstats"
            report = Path(folder) / "stats.txt"
            profiler.dump_stats(str(binary))
            with report.open("w") as stream:
                stats = pstats.Stats(str(binary), stream=stream).strip_dirs()
                stats.sort_stats("cumulative").print_stats(40)
                stats.sort_stats("tottime").print_stats(40)
            report.replace(Path(str(args.output) + ".txt"))
            binary.replace(args.output)

    def checkpoint(signum, frame):
        signal.setitimer(signal.ITIMER_REAL, 0)
        profiler.disable()
        try:
            save()
        finally:
            if recording:
                profiler.enable()
                signal.setitimer(signal.ITIMER_REAL, args.snapshot_interval)

    def interrupted(signum, frame):
        nonlocal recording
        # Save immediately: Kit shutdown can exceed cleanup's grace period.
        recording = False
        signal.setitimer(signal.ITIMER_REAL, 0)
        profiler.disable()
        save()
        raise SystemExit(128 + signum)

    from smores_ep.scenarios import parallel_self_assembly
    original = parallel_self_assembly.run_parallel_self_assembly_scenario

    def measured(*positional, **keywords):
        nonlocal recording
        # The CLI calls this after SimulationApp initialization. Exclude Kit
        # startup and install handlers after Kit has installed its own.
        previous_signals = {
            sig: signal.signal(sig, handler) for sig, handler in (
                (signal.SIGALRM, checkpoint),
                (signal.SIGINT, interrupted), (signal.SIGTERM, interrupted),
            )
        }
        previous_timer = signal.getitimer(signal.ITIMER_REAL)
        try:
            recording = True
            profiler.enable()
            signal.setitimer(signal.ITIMER_REAL, args.snapshot_interval)
            return original(*positional, **keywords)
        finally:
            recording = False
            signal.setitimer(signal.ITIMER_REAL, 0)
            profiler.disable()
            try:
                save()
            finally:
                for sig, handler in previous_signals.items():
                    signal.signal(sig, handler)
                signal.setitimer(signal.ITIMER_REAL, *previous_timer)

    previous_argv = sys.argv
    sys.argv = ["smores_ep.self_assembly_cli", *native_args]
    parallel_self_assembly.run_parallel_self_assembly_scenario = measured
    try:
        runpy.run_module("smores_ep.self_assembly_cli", run_name="__main__")
    finally:
        parallel_self_assembly.run_parallel_self_assembly_scenario = original
        sys.argv = previous_argv


if __name__ == "__main__":
    main()
