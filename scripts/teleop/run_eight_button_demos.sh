#!/usr/bin/env bash
# Human-only teleoperation launcher for eight isolated button demonstrations.
set -euo pipefail

usage() {
    echo 'Uso: bash scripts/teleop/run_eight_button_demos.sh button-01|button-02|button-03|button-04|button-05|button-06|button-07|button-08 [--check]' >&2
    exit 2
}

[ "$#" -ge 1 ] && [ "$#" -le 2 ] || usage

EPISODE="$1"

case "$EPISODE" in
    button-01) SEED=6217 ;;
    button-02) SEED=6307 ;;
    button-03) SEED=6533 ;;
    button-04) SEED=6284 ;;
    button-05) SEED=6341 ;;
    button-06) SEED=6443 ;;
    button-07) SEED=6265 ;;
    button-08) SEED=6316 ;;
    *) usage ;;
esac

if [ "$#" -eq 2 ] && [ "$2" != "--check" ]; then
    usage
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

export PYTHONPATH="$ROOT/scripts/teleop:$ROOT/mssr_ws/src/mssr_expert:$ROOT/scripts/smores_ep/src${PYTHONPATH:+:$PYTHONPATH}"

python3 - "$EPISODE" "$SEED" <<'PY'
import sys
from smores_ep.isaac.obstacle_course import sample_button_target_spec

episode = sys.argv[1]
seed = int(sys.argv[2])
spec = sample_button_target_spec(seed)

dx, dy = spec.press_direction_xy
if abs(dx) > abs(dy):
    direction = "+X" if dx > 0 else "-X"
else:
    direction = "+Y" if dy > 0 else "-Y"

print(
    f"{episode}: seed={seed}, "
    f"button=({spec.x_m:+.3f}, {spec.y_m:+.3f}, {spec.z_m:+.3f}) m, "
    f"press={direction}"
)
PY

if [ "${2:-}" = "--check" ]; then
    exit 0
fi

set +u
source /opt/ros/humble/setup.bash
source "$ROOT/mssr_ws/install/setup.bash"
set -u

export ROS_DOMAIN_ID=0
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

python3 - <<'PY'
from pathlib import Path
from runtime_cleanup import scoped_cleanup

scoped_cleanup(Path.cwd())
PY

mkdir -p "$ROOT/logs/teleop/runs"
LOG_DIR="$(mktemp -d "$ROOT/logs/teleop/runs/${EPISODE}-$(date +%Y%m%d-%H%M%S).XXXXXX")"
RUNTIME_DIR="$(mktemp -d /dev/shm/mssr-teleop.XXXXXX)"

printf 'episode=%s\nbutton_seed=%s\nruntime=%s\n' \
    "$EPISODE" "$SEED" "$RUNTIME_DIR" > "$LOG_DIR/session.txt"

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
    --button-test-course \
    --button-seed "$SEED" \
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

printf '\nStage: %s | button seed: %s\nLog: %s\nRuntime: %s\n' \
    "$EPISODE" "$SEED" "$LOG_DIR" "$RUNTIME_DIR"

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

echo
echo 'START: avvia/ferma la registrazione; lasciala ON durante assembly e reconfiguration.'
echo 'X + D-pad: assembly iniziale della morfologia scelta.'
echo 'D-pad: SINISTRA=RC-Car8, SU=Snake8, DESTRA=MobileManipulator8.'
echo 'TRIANGLE: E-stop/ripresa.'
echo 'Dopo aver premuto il button: START per chiudere la registrazione, poi Ctrl+C.'
echo

ros2 launch mssr_expert smores_teleop.launch.py \
    start_joy:=true \
    use_sim_time:=false \
    input_config_path:="$ROOT/mssr_ws/src/mssr_expert/config/smores_dualsense.yaml" \
    teleop_config_path:="$ROOT/mssr_ws/src/mssr_expert/config/smores_teleop.yaml"
