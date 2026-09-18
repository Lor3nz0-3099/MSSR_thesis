"""Profiling survives the probe's normal shutdown without launching Isaac."""
import os
from pathlib import Path
import pstats
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("interrupt", [False, True])
@pytest.mark.parametrize("output_name", ["assembly.pstats", "assembly.txt"])
def test_profile_records_native_work_and_survives_sigterm(tmp_path, interrupt, output_name):
    package = tmp_path / "smores_ep"
    package.mkdir()
    (package / "__init__.py").write_text("")
    # Replace only the expensive Isaac CLI boundary, run the real profiler.
    (package / "self_assembly_cli.py").write_text(
        "import sys, time\n"
        "assert sys.argv[1:] == ['--module-count', '8']\n"
        "def assembly_work():\n"
        "    return sum(range(10000))\n"
        "assembly_work()\n"
        "print('ready', flush=True)\n" +
        ("while True: time.sleep(0.05)\n" if interrupt else "")
    )
    output = tmp_path / output_name
    process = subprocess.Popen(
        [sys.executable, str(ROOT / "scripts/teleop/profile_assembly.py"),
         "--output", str(output), "--", "--module-count", "8"],
        env={**os.environ, "PYTHONPATH": str(tmp_path)},
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        assert process.stdout.readline().strip() == "ready"
        if interrupt:
            process.terminate()
        stdout, stderr = process.communicate(timeout=5)
        assert process.returncode == (143 if interrupt else 0), stderr
        stats = pstats.Stats(str(output))
        assert any(key[2] == "assembly_work" for key in stats.stats)
        assert "Ordered by:" in Path(str(output) + ".txt").read_text()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
