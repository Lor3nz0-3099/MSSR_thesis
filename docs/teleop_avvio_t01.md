# Registrare le teleoperazioni — partenza da T01

T01 della nuova campagna: **scale 3102 → RC 5101 → pulsante 6103**.
Sequenza delle morfologie: **Snake8 → RC-Car8 → MobileManipulator8**.
Questa guida avvia Isaac GUI, bridge, esecutore delle macro e DualSense.
La navigazione e l'allineamento sono teleoperati. Le nuove composizioni devono
ancora essere validate fisicamente: annota anche eventuali problemi del percorso.

## 1. Preparazione

Collega il DualSense. Chiudi la precedente sessione dopo aver fermato la
registrazione con START. Il blocco di avvio esegue una pulizia dei processi MSSR
di questo checkout prima di aprire la nuova sessione.

Da un terminale, aggiorna il pacchetto ROS e genera le missioni:

```bash
cd ~/MSSR_thesis && bash <<'BASH'
set -eo pipefail
source /opt/ros/humble/setup.bash
cd mssr_ws
colcon build --packages-select mssr_expert --symlink-install
cd ..
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
MPLCONFIGDIR=/tmp/mssr-teleop-mpl \
python3 scripts/smores_ep/preview_teleop_composite_campaign.py
BASH
```

Controlla l'anteprima `logs/teleop/course_previews/t01-t13-v1/teleop-t01.png`.
La generazione aggiorna le anteprime e le missioni in quella cartella; ogni avvio
qui sotto ne conserva una copia nella propria cartella di sessione.

## 2. Avvio completo — copia tutto il blocco

Per partire dal primo stage lascia `EPISODE=teleop-t01`.

```bash
cd ~/MSSR_thesis && bash <<'BASH'
set -eo pipefail
EPISODE=teleop-t01

set +u
source /opt/ros/humble/setup.bash
source "$PWD/mssr_ws/install/setup.bash"
set -u
export ROS_DOMAIN_ID=0
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export PYTHONPATH="$PWD/scripts/teleop:$PWD/mssr_ws/src/mssr_expert:$PWD/scripts/smores_ep/src${PYTHONPATH:+:$PYTHONPATH}"

PREVIEW="$PWD/logs/teleop/course_previews/t01-t13-v1"
CATALOG="$PWD/mssr_ws/src/mssr_expert/config/smores_composite_seed_catalog.json"
test -s "$PREVIEW/$EPISODE.mission.json"
test -s "$PREVIEW/$EPISODE.course.json"
test -s "$CATALOG"

python3 - <<'PY'
from pathlib import Path
from runtime_cleanup import scoped_cleanup
scoped_cleanup(Path.cwd())
PY

mkdir -p "$PWD/logs/teleop/runs"
LOG_DIR="$(mktemp -d "$PWD/logs/teleop/runs/${EPISODE}-$(date +%Y%m%d-%H%M%S).XXXXXX")"
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

echo 'Attendi che la scena sia stabile; START avvia la registrazione.'
echo 'T01: X + SU assembla Snake8; SQUARE avvia la macro scale.'
echo 'A fine prova: START per chiudere la registrazione, poi Ctrl+C.'
ros2 launch mssr_expert smores_teleop.launch.py \
    start_joy:=true \
    use_sim_time:=false \
    input_config_path:="$PWD/mssr_ws/src/mssr_expert/config/smores_dualsense.yaml" \
    teleop_config_path:="$PWD/mssr_ws/src/mssr_expert/config/smores_teleop.yaml"
BASH
```

## 3. Registrazione e sequenza di T01

1. Attendi il caricamento della scena e il riconoscimento dei moduli.
2. Premi **START/OPTIONS una volta** per registrare. Controlla lo stato nel
   secondo terminale descritto sotto: `recording` deve essere `true`,
   `recording_backend_ready` deve essere `true`, deve esserci un
   `recording_episode_id` e non devono esserci errori di registrazione.
3. Tieni **X** e premi **D-pad SU** per l'assemblaggio iniziale Snake8;
   rilascia i tasti e attendi il completamento.
4. Avvicina e allinea Snake8 alla scala. Premi **SQUARE** per la macro scale;
   attendi che tutto il serpente raggiunga il pianerottolo.
5. Raggiungi la pedana di riconfigurazione. Premi **D-pad SINISTRA** per
   Snake8 → RC-Car8; attendi il completamento, poi percorri il tratto RC.
6. Sulla pedana successiva premi **D-pad DESTRA** per RC-Car8 →
   MobileManipulator8. Attendi il completamento e raggiungi/premi il pulsante.
7. Completa il percorso fino al goal. Premi **START** per fermare la registrazione
   e verifica `recording: false`. Solo dopo chiudi il launcher con **Ctrl+C**.

Mantieni la registrazione attiva durante le riconfigurazioni. **TRIANGLE** alterna
E-stop/ripresa; **CIRCLE** richiede HOME. Sugli stage con gap, **X premuto e
rilasciato da solo** avvia la macro gap in Snake8. Allinea manualmente il robot
prima di avviare le macro; per ostacoli orientati il riferimento viene selezionato
dallo stage.

### Secondo terminale: controllare lo stato

```bash
cd ~/MSSR_thesis
source /opt/ros/humble/setup.bash
source mssr_ws/install/setup.bash
export ROS_DOMAIN_ID=0
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
ros2 topic echo /mssr/teleop/status
```

Annota `recording_episode_id`: identifica esattamente la dimostrazione.

## 4. Dove sono i dati

- Dimostrazione: `logs/teleop/recordings/<recording_episode_id>/manifest.json`
  e i flussi indicati nel manifest.
- Log tecnici, missione e geometria della sessione: la cartella
  `logs/teleop/runs/teleop-t01-...` stampata all'avvio.
- I file temporanei in `/dev/shm` non sono l'archivio della dimostrazione.

Annota accanto all'ID della registrazione: **T01**, esito effettivo (successo,
prefisso valido o fallimento), punto raggiunto ed eventuali problemi. La chiusura
corretta della registrazione non certifica automaticamente il successo del task.
Le prove da includere in `datasets/expert_v1` vanno selezionate dopo tale verifica.

## 5. Stage successivi

Per una nuova prova chiudi prima la sessione precedente e rilancia il blocco del
punto 2 cambiando soltanto `EPISODE=teleop-t02`, poi `teleop-t03`, fino a
`teleop-t13`. Non serve ricompilare o rigenerare se non hai modificato il codice
oppure la campagna. Per T02 il primo ostacolo è un gap: parti ancora da Snake8,
ma usa **X da solo** al posto di SQUARE. Negli stage che iniziano con RC usa
**X + D-pad SINISTRA** per l'assemblaggio iniziale.

Sequenze e anteprime: [campagna T01–T13](teleop_composite_campaign13.md).
