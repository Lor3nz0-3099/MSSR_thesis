"""Opt-in native assembly profiling, saved before probe SIGTERM shutdown."""
import argparse
import cProfile
from pathlib import Path
import pstats
import runpy
import signal
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("native_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    native_args = args.native_args
    if native_args[:1] == ["--"]:
        native_args = native_args[1:]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    profiler = cProfile.Profile()

    def save():
        profiler.disable()
        profiler.dump_stats(str(args.output))
        with Path(str(args.output) + ".txt").open("w") as stream:
            stats = pstats.Stats(profiler, stream=stream).strip_dirs()
            stats.sort_stats("cumulative").print_stats(40)
            stats.sort_stats("tottime").print_stats(40)

    def interrupted(signum, frame):
        # Save immediately: Kit shutdown can exceed cleanup's grace period.
        save()
        raise SystemExit(128 + signum)

    previous_signals = {sig: signal.signal(sig, interrupted)
                        for sig in (signal.SIGINT, signal.SIGTERM)}
    previous_argv = sys.argv
    sys.argv = ["smores_ep.self_assembly_cli", *native_args]
    try:
        profiler.enable()
        runpy.run_module("smores_ep.self_assembly_cli", run_name="__main__")
    finally:
        save()
        sys.argv = previous_argv
        for sig, handler in previous_signals.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    main()
