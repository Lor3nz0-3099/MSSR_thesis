"""Verify structure-only E-STOP while Isaac physics keeps advancing.

System-Python orchestrator performs scoped cleanup and runs native Isaac.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]


def native(output: Path) -> int:
    sys.path.insert(0, str(ROOT / "scripts/smores_ep/src"))

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    result = {
        "passed": False,
        "source": "native_isaac_structure_stop_and_live_physics",
    }

    app_utils = None

    try:
        import isaacsim.core.experimental.utils.app as app_utils
        import omni.usd

        from isaacsim.core.simulation_manager import SimulationManager
        from pxr import Gf, UsdGeom, UsdPhysics
        from smores_ep.isaac.teleop_runtime import TeleopRuntimeBridge

        omni.usd.get_context().new_stage()
        stage = omni.usd.get_context().get_stage()

        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

        physics = UsdPhysics.Scene.Define(stage, "/World/Physics")
        physics.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
        physics.CreateGravityMagnitudeAttr(9.81)

        cube = UsdGeom.Cube.Define(stage, "/World/Cube")
        cube.CreateSizeAttr(0.1)
        cube.AddTranslateOp().Set(Gf.Vec3d(0, 0, 10))
        UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())

        stop_calls = []

        def apply_structure_stop(active: bool) -> None:
            stop_calls.append(active)

        bridge = TeleopRuntimeBridge(
            output / "request.json",
            output / "status.json",
            structure_stop_callback=apply_structure_stop,
        )

        def z() -> float:
            transform = UsdGeom.XformCache().GetLocalToWorldTransform(
                cube.GetPrim()
            )
            return float(transform.ExtractTranslation()[2])

        def updates(count: int) -> None:
            for _ in range(count):
                app.update()

        def request(active: bool, request_id: str) -> dict:
            bridge.request_file.write_text(
                json.dumps(
                    {
                        "schema_version": "mssr.teleop_runtime.v1",
                        "structure_stop_request": {
                            "id": request_id,
                            "active": active,
                        },
                    }
                )
            )
            return bridge.poll()

        SimulationManager.set_physics_dt(1.0 / 240.0)
        app_utils.play()
        app.update()

        updates(30)
        moving_time_0 = SimulationManager.get_simulation_time()
        moving_z_0 = z()

        updates(30)
        moving_time_1 = SimulationManager.get_simulation_time()
        moving_z_1 = z()

        assert moving_time_1 > moving_time_0
        assert moving_z_1 < moving_z_0 - 1e-4

        stopped = request(True, "stop1")

        assert stopped["structure_stopped"] is True
        assert stopped["structure_stop_ack"] == {
            "id": "stop1",
            "active": True,
            "applied": True,
        }
        assert stop_calls == [True]

        stopped_time_0 = SimulationManager.get_simulation_time()
        stopped_z_0 = z()

        for _ in range(30):
            repeated = bridge.poll()
            assert repeated["structure_stopped"] is True
            app.update()

        stopped_time_1 = SimulationManager.get_simulation_time()
        stopped_z_1 = z()

        assert stopped_time_1 > stopped_time_0, (
            "simulation time must continue during structure E-STOP"
        )
        assert stopped_z_1 < stopped_z_0 - 1e-4, (
            "free physics must continue during structure E-STOP"
        )
        assert stop_calls == [True]

        cleared = request(False, "clear1")

        assert cleared["structure_stopped"] is False
        assert cleared["structure_stop_ack"] == {
            "id": "clear1",
            "active": False,
            "applied": True,
        }
        assert stop_calls == [True, False]

        clear_time_0 = SimulationManager.get_simulation_time()
        updates(30)
        clear_time_1 = SimulationManager.get_simulation_time()

        assert clear_time_1 > clear_time_0

        result.update(
            passed=True,
            stop_ack=stopped["structure_stop_ack"],
            clear_ack=cleared["structure_stop_ack"],
            stop_calls=stop_calls,
            physics_continued_during_stop=True,
            simulation_time_continued_during_stop=True,
            stopped_time_before=stopped_time_0,
            stopped_time_after=stopped_time_1,
            stopped_z_before=stopped_z_0,
            stopped_z_after=stopped_z_1,
        )

    except Exception as error:
        result["error"] = repr(error)

    finally:
        try:
            if app_utils is not None:
                app_utils.stop()
        finally:
            (output / "report.json").write_text(
                json.dumps(result, indent=2) + "\n"
            )
            app.close()

    return 0 if result["passed"] else 1
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-output", type=Path)
    parser.add_argument("--isaac-python", default="/home/lorenzo/isaac/python.sh")
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=360.0,
        help="Maximum wall-clock seconds allowed for the native Isaac probe (30-900).",
    )
    args = parser.parse_args()
    if not 30.0 <= args.timeout_s <= 900.0:
        parser.error("--timeout-s must be between 30 and 900 seconds")
    if args.native_output is not None:
        return native(args.native_output)
    # The documented system-Python command sources Humble, not the workspace.
    # Load the checkout package required by the shared probe environment helper.
    sys.path.insert(0, str(ROOT / "mssr_ws/src/mssr_expert"))
    from check_dualsense import configure_probe_environment
    from runtime_cleanup import scoped_cleanup
    output = ROOT / "logs/teleop/runtime_checks" / uuid4().hex
    configure_probe_environment(os.environ, output)
    output.mkdir(parents=True, exist_ok=False)
    cleanup = scoped_cleanup(ROOT)
    process = None
    try:
        with (output / "isaac.log").open("x") as log:
            process = subprocess.Popen([args.isaac_python, str(Path(__file__).resolve()),
                                        "--native-output", str(output)], cwd=ROOT,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            process.wait(timeout=args.timeout_s)
    except Exception as error:
        (output / "report.json").write_text(json.dumps({"passed": False, "error": repr(error)}))
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=2)
        final_cleanup = scoped_cleanup(ROOT)
    if (output / "report.json").exists():
        result = json.loads((output / "report.json").read_text())
    else:
        result = {"passed": False, "error": "Isaac exited without acceptance report"}
    result.update(cleanup=cleanup, final_cleanup=final_cleanup,
                  git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip())
    if process is not None and process.returncode != 0:
        result["passed"] = False
        result["isaac_returncode"] = process.returncode
    (output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print("T2_RUNTIME_RESULT=" + json.dumps(result), flush=True)
    print(f"REPORT={output / 'report.json'}", flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
