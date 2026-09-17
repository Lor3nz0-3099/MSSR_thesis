"""Observe actual Isaac timeline/physics freeze and resume; no ROS action IO.

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
    result = {"passed": False, "source": "native_isaac_timeline_and_rigid_body"}
    try:
        import omni.timeline
        import omni.usd
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
        timeline = omni.timeline.get_timeline_interface()
        bridge = TeleopRuntimeBridge(timeline, output / "request.json", output / "status.json")

        def z():
            return float(UsdGeom.XformCache().GetLocalToWorldTransform(cube.GetPrim()).ExtractTranslation()[2])

        def updates(count):
            for _ in range(count):
                app.update()

        def request(operation, request_id):
            bridge.request_file.write_text(json.dumps({"schema_version": "mssr.teleop_runtime.v1",
                "timeline_request": {"id": request_id, "operation": operation}}))
            return bridge.poll()

        timeline.play()
        updates(30)
        before_z = z()
        updates(30)
        assert z() < before_z - 1e-4, "rigid body must move before pause"
        pause = request("pause", "pause1")
        assert not timeline.is_playing() and pause["timeline_ack"]["applied"]
        updates(1)
        paused_time, paused_z = timeline.get_current_time(), z()
        for _ in range(30):
            assert not bridge.poll()["timeline_playing"]
            app.update()
        assert abs(timeline.get_current_time() - paused_time) < 1e-9, "sim time advanced during pause"
        assert abs(z() - paused_z) < 1e-9, "physics moved during pause"
        resume = request("resume", "resume1")
        assert timeline.is_playing() and resume["timeline_ack"]["applied"]
        updates(30)
        assert timeline.get_current_time() > paused_time and z() < paused_z - 1e-4
        result.update(passed=True, pause_ack=pause, resume_ack=resume, paused_time=paused_time,
                      resumed_time=timeline.get_current_time(), paused_z=paused_z, resumed_z=z(),
                      paused_app_updates=30, physics_frozen=True, sim_time_frozen=True)
    except Exception as error:
        result["error"] = repr(error)
    finally:
        (output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
        app.close()
    return 0 if result["passed"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-output", type=Path)
    parser.add_argument("--isaac-python", default="/home/lorenzo/isaac/python.sh")
    args = parser.parse_args()
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
            process.wait(timeout=180)
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
