"""Opt-in assembly profiling with periodic snapshots and shutdown saves."""
import argparse
import cProfile
import marshal
import math
from pathlib import Path
import pstats
import runpy
import signal
import sys
import tempfile
import threading


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
    save_lock = threading.RLock()
    finished = threading.Event()

    def save():
        with save_lock:
            # dump_stats/create_stats disable profiling. Snapshot directly so
            # later native work remains recorded after each periodic save.
            profiler.snapshot_stats()
            with tempfile.TemporaryDirectory(dir=args.output.parent, prefix=".assembly-profile-") as folder:
                binary = Path(folder) / "stats.pstats"
                report = Path(folder) / "stats.txt"
                with binary.open("wb") as stream:
                    marshal.dump(profiler.stats, stream)
                with report.open("w") as stream:
                    stats = pstats.Stats(str(binary), stream=stream).strip_dirs()
                    stats.sort_stats("cumulative").print_stats(40)
                    stats.sort_stats("tottime").print_stats(40)
                # Publish complete files even if Kit exits without unwinding
                # Python handlers/finally; retain the last valid snapshot.
                report.replace(Path(str(args.output) + ".txt"))
                binary.replace(args.output)

    def periodic_save():
        while not finished.wait(args.snapshot_interval):
            save()

    def interrupted(signum, frame):
        # Save immediately: Kit shutdown can exceed cleanup's grace period.
        profiler.disable()
        finished.set()
        save()
        raise SystemExit(128 + signum)

    previous_signals = {sig: signal.signal(sig, interrupted)
                        for sig in (signal.SIGINT, signal.SIGTERM)}
    previous_argv = sys.argv
    sys.argv = ["smores_ep.self_assembly_cli", *native_args]
    try:
        profiler.enable()
        threading.Thread(target=periodic_save, name="assembly-profile-writer", daemon=True).start()
        runpy.run_module("smores_ep.self_assembly_cli", run_name="__main__")
    finally:
        profiler.disable()
        finished.set()
        save()
        sys.argv = previous_argv
        for sig, handler in previous_signals.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    main()
