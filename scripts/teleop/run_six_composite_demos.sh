#!/usr/bin/env bash
# Human-only teleoperation launcher for the six remaining composite missions.
set -euo pipefail

usage() {
    echo 'Uso: bash scripts/teleop/run_six_composite_demos.sh teleop-t14|teleop-t15|teleop-t10|teleop-t16|teleop-t09|teleop-t07 [--check]' >&2
    exit 2
}

[ "$#" -ge 1 ] && [ "$#" -le 2 ] || usage
EPISODE="$1"
case "$EPISODE" in
    teleop-t14|teleop-t15|teleop-t10|teleop-t16|teleop-t09|teleop-t07) ;;
    *) usage ;;
esac
if [ "$#" -eq 2 ] && [ "$2" != '--check' ]; then usage; fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PREVIEW="$ROOT/logs/teleop/course_previews/t01-t16-v2"
CATALOG="$ROOT/mssr_ws/src/mssr_expert/config/smores_composite_seed_catalog.json"

python3 - "$EPISODE" "$PREVIEW" "$CATALOG" <<'PY'
import json
from pathlib import Path
import sys

episode, preview_text, catalog_text = sys.argv[1:]
preview = Path(preview_text)
mission_path = preview / f"{episode}.mission.json"
course_path = preview / f"{episode}.course.json"
audit_path = preview / "geometry_audit.json"
for path in (mission_path, course_path, audit_path, Path(catalog_text)):
    if not path.is_file() or not path.stat().st_size:
        raise SystemExit(f"File mancante o vuoto: {path}")
mission = json.loads(mission_path.read_text())
course = json.loads(course_path.read_text())
audit = json.loads(audit_path.read_text())
record = next((item for item in audit["episodes"] if item["episode_id"] == episode), None)
if mission.get("episode_id") != episode or course.get("mission", {}).get("episode_id") != episode:
    raise SystemExit(f"ID missione incoerente: {episode}")
if not record or not record.get("valid") or record.get("geometry_sha256") != course.get("geometry_sha256"):
    raise SystemExit(f"Audit geometrico mancante o non coerente: {episode}")
expected = [(task["type"], task["seed"]) for task in mission["tasks"]]
actual = [(task["type"], task["seed"]) for task in course["mission"]["tasks"] if "seed" in task]
if expected != actual:
    raise SystemExit(f"Seed e ordine incoerenti fra missione e geometria: {episode}")
print(f"{episode}: geometria verificata, {len(expected)} ostacoli, hash {record['geometry_sha256'][:12]}")
PY

if [ "${2:-}" = '--check' ]; then
    exit 0
fi

set +u
source /opt/ros/humble/setup.bash
source "$ROOT/mssr_ws/install/setup.bash"
set -u
export ROS_DOMAIN_ID=0
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export PYTHONPATH="$ROOT/scripts/teleop:$ROOT/mssr_ws/src/mssr_expert:$ROOT/scripts/smores_ep/src${PYTHONPATH:+:$PYTHONPATH}"

python3 - <<'PY'
from pathlib import Path
from runtime_cleanup import scoped_cleanup
scoped_cleanup(Path.cwd())
PY

mkdir -p "$ROOT/logs/teleop/runs"
LOG_DIR="$(mktemp -d "$ROOT/logs/teleop/runs/${EPISODE}-$(date +%Y%m%d-%H%M%S).XXXXXX")"
RUNTIME_DIR="$(mktemp -d /dev/shm/mssr-teleop.XXXXXX)"
cp "$PREVIEW/$EPISODE.mission.json" "$LOG_DIR/mission.json"
cp "$PREVIEW/$EPISODE.course.json" "$LOG_DIR/course.json"
cp "$CATALOG" "$LOG_DIR/seed_catalog.json"
printf 'episode=%s\nruntime=%s\n' "$EPISODE" "$RUNTIME_DIR" > "$LOG_DIR/session.txt"

ACTION="$RUNTIME_DIR/actions.json"
GOAL="$RUNTIME_DIR/primitive_goal.json"
CANCEL="$RUNTIME_DIR/primitive_cancel.json"
PRIMITIVE_STATUS="$RUNTIME_DIR/primitive_status.json"
cp configs/idle_actions.json "$ACTION"
printf '{}\n' > "$GOAL"
printf '{}\n' > "$CANCEL"

cleanup() {
    rc=$?
    trap - EXIT INT TERM
    set +e
    python3 - <<'PY'
from pathlib import Path
from runtime_cleanup import scoped_cleanup
try:
    scoped_cleanup(Path.cwd())
except Exception as exc:
    print(f"Cleanup warning: {exc}")
PY
    printf '\nLog sessione: %s\nRuntime: %s\n' "$LOG_DIR" "$RUNTIME_DIR"
    exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

python3 ros2_bridge/mssr_file_bridge.py \
    --state-graph-dir "$RUNTIME_DIR" \
    --action-file "$ACTION" \
    --primitive-goal-file "$GOAL" \
    --primitive-cancel-file "$CANCEL" \
    --primitive-status-file "$PRIMITIVE_STATUS" \
    > "$LOG_DIR/bridge.log" 2>&1 &
BRIDGE_PID=$!

ros2 run mssr_expert mssr_smores_morphology_behavior_node \
    > "$LOG_DIR/morphology_behavior.log" 2>&1 &
BEHAVIOR_PID=$!

bash scripts/smores_ep/run_self_assembly.sh \
    --module-count 8 \
    --composite-mission "$LOG_DIR/mission.json" \
    --composite-seed-catalog "$LOG_DIR/seed_catalog.json" \
    --performance \
    --simple-visuals \
    --physics-hz 240 \
    --state-publish-hz 30 \
    --actuator-effort-scale 4.0 \
    --tilt-effort-scale 8.0 \
    --wheel-friction-scale 1.50 \
    --action-file "$ACTION" \
    --primitive-goal-file "$GOAL" \
    --primitive-cancel-file "$CANCEL" \
    --primitive-status-file "$PRIMITIVE_STATUS" \
    > "$LOG_DIR/isaac.log" 2>&1 &
ISAAC_PID=$!

printf '\nStage: %s\nLog: %s\nRuntime: %s\n' "$EPISODE" "$LOG_DIR" "$RUNTIME_DIR"
echo 'Attendo il primo grafo da Isaac (massimo 180 secondi)...'
ready=false
for ((i=0; i<180; i++)); do
    for pid in "$BRIDGE_PID" "$BEHAVIOR_PID" "$ISAAC_PID"; do
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "Un processo è terminato: controlla i log in $LOG_DIR"
            exit 1
        fi
    done
    if [ -s "$RUNTIME_DIR/robot_graph.json" ]; then
        ready=true
        break
    fi
    sleep 1
done
if [ "$ready" != true ]; then
    echo "Isaac non ha pubblicato il grafo: controlla $LOG_DIR/isaac.log"
    exit 1
fi

printf '\nSTART: avvia/ferma la registrazione; lascia ON durante tutte le macro.\n'
printf 'X + D-pad SINISTRA: assembly RC; X + D-pad SU: assembly Snake.\n'
printf 'D-pad: SINISTRA=RC, SU=Snake, DESTRA=manipolatore.\n'
printf 'Snake: SQUARE=scale, X rilasciato=gap. TRIANGLE=E-stop.\n'
printf 'Al goal: START per chiudere la registrazione; poi Ctrl+C.\n\n'

ros2 launch mssr_expert smores_teleop.launch.py \
    start_joy:=true \
    use_sim_time:=false \
    input_config_path:="$ROOT/mssr_ws/src/mssr_expert/config/smores_dualsense.yaml" \
    teleop_config_path:="$ROOT/mssr_ws/src/mssr_expert/config/smores_teleop.yaml"
