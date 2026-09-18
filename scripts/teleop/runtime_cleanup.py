"""Command-line scoped MSSR cleanup with PID identity checks (Linux).

Only current-user runtime processes with checkout context are selected.
Argument positions matter: a shell/editor mentioning a command is not it.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import signal
import subprocess
import time


_MODULES = frozenset({
    "smores_ep.self_assembly_cli", "smores_ep.dynamic_cli", "smores_ep.cli",
    "smores_ep.docking_cli", "smores_ep.multi_lift_cli",
})
_NAV2 = {
    "nav2_controller": "controller_server", "nav2_planner": "planner_server",
    "nav2_bt_navigator": "bt_navigator", "nav2_behaviors": "behavior_server",
    "nav2_smoother": "smoother_server", "nav2_velocity_smoother": "velocity_smoother",
    "nav2_lifecycle_manager": "lifecycle_manager", "nav2_map_server": "map_server",
    "nav2_amcl": "amcl", "nav2_waypoint_follower": "waypoint_follower",
}


@dataclass(frozen=True)
class ProcessIdentity:
    pid: int
    started_at: int
    command: tuple[str, ...]


def is_runtime_command(command: list[str] | tuple[str, ...], root: Path) -> bool:
    if not command:
        return False
    args = list(command)
    executable = Path(args[0]).name
    if executable.startswith("python"):
        if len(args) >= 3 and args[1] == "-m":
            return args[2] in _MODULES
        if len(args) < 2 or args[1].startswith("-"):
            return False
        args = args[1:]
        executable = Path(args[0]).name
    if executable == "ros2":
        return len(args) >= 4 and (
            (args[1] == "launch" and args[2] == "mssr_expert" and
             args[3] in {"smores_runtime.launch.py", "smores_nav2.launch.py", "smores_teleop.launch.py"})
            or (args[1] == "run" and args[2] == "mssr_expert" and args[3].startswith("mssr_smores_"))
            or (args[1:4] == ["run", "joy", "game_controller_node"])
        )
    path = Path(args[0])
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if path in {root / "ros2_bridge/mssr_file_bridge.py",
                root / "scripts/teleop/check_dualsense.py",
                root / "scripts/teleop/check_teleop_shell.py",
                root / "scripts/teleop/check_isaac_runtime.py",
                root / "scripts/teleop/check_rc_car.py",
                root / "scripts/teleop/profile_assembly.py"}:
        return True
    if path.is_relative_to(root / "mssr_ws/install") and executable.startswith("mssr_smores_"):
        return True
    # Narrow executable paths, with checkout association checked by the scanner.
    if path == Path("/opt/ros/humble/lib/joy/game_controller_node"):
        return True
    return any(path == Path("/opt/ros/humble/lib") / package / name
               for package, name in _NAV2.items())


def read_process(pid: int) -> ProcessIdentity | None:
    try:
        proc = Path("/proc") / str(pid)
        stat = proc.joinpath("stat").read_text().rsplit(")", 1)[1].split()
        if stat[0] == "Z":
            return None
        args = tuple(part.decode(errors="surrogateescape") for part in
                     proc.joinpath("cmdline").read_bytes().split(b"\0") if part)
        return ProcessIdentity(pid, int(stat[19]), args)
    except (OSError, ValueError, IndexError):
        return None


def _ancestors() -> set[int]:
    result = {os.getpid()}
    pid = os.getpid()
    while pid > 1:
        try:
            stat = (Path("/proc") / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()
            pid = int(stat[1])
        except (OSError, ValueError, IndexError):
            break
        result.add(pid)
    return result


def find_runtime_processes(root: Path) -> list[ProcessIdentity]:
    root = root.resolve()
    excluded = _ancestors()
    selected = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) in excluded:
            continue
        try:
            if proc.stat().st_uid != os.getuid():
                continue
            # A Nav2 or joy process from another checkout must not be stopped.
            cwd = proc.joinpath("cwd").resolve(strict=True)
            if not cwd.is_relative_to(root):
                continue
        except OSError:
            continue
        identity = read_process(int(proc.name))
        if identity is not None and is_runtime_command(identity.command, root):
            selected.append(identity)
    return selected


def signal_if_same(identity: ProcessIdentity, sig: signal.Signals) -> bool:
    if read_process(identity.pid) != identity:
        return False
    try:
        os.kill(identity.pid, sig)
        return True
    except ProcessLookupError:
        return False


def _wait_survivors(identities: list[ProcessIdentity], timeout_s: float) -> list[ProcessIdentity]:
    deadline = time.monotonic() + timeout_s
    while True:
        survivors = [item for item in identities if read_process(item.pid) == item]
        if not survivors or time.monotonic() >= deadline:
            return survivors
        time.sleep(0.1)


def scoped_cleanup(root: Path, grace_s: float = 5.0) -> dict:
    selected = find_runtime_processes(root)
    for identity in selected:
        print(f"cleanup SIGTERM pid={identity.pid} argv={list(identity.command)!r}", flush=True)
        signal_if_same(identity, signal.SIGTERM)
    survivors = _wait_survivors(selected, grace_s)
    for identity in survivors:
        print(f"cleanup SIGKILL surviving pid={identity.pid}", flush=True)
        signal_if_same(identity, signal.SIGKILL)
    remaining = _wait_survivors(survivors, 2.0)
    daemon = subprocess.run(["ros2", "daemon", "stop"], capture_output=True,
                            text=True, timeout=15, check=False)
    print(daemon.stdout.strip(), flush=True)
    if daemon.returncode:
        raise RuntimeError(f"ROS daemon stop failed: {daemon.stderr.strip()}")
    remaining.extend(find_runtime_processes(root))
    if remaining:
        raise RuntimeError(f"scoped cleanup incomplete: {[item.pid for item in remaining]}")
    print(f"cleanup verified: selected={len(selected)}, survivors=0, daemon stopped", flush=True)
    return {"selected": [{"pid": item.pid, "argv": list(item.command)} for item in selected],
            "sigkill": [item.pid for item in survivors], "remaining": [], "daemon_stopped": True}
