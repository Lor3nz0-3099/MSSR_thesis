# Comandi per gli expert singoli e composite

Riferimento alla configurazione presente nel checkout il 16 settembre 2026.
I blocchi sono comandi Bash da copiare separatamente: non eseguire tutto il
documento come uno script. Avvia un solo episodio alla volta.

## 1. Preparazione

Build del pacchetto dopo modifiche ai sorgenti o alle configurazioni:

```bash
cd /home/lorenzo/MSSR_thesis
source /opt/ros/humble/setup.bash
colcon build --base-paths mssr_ws/src --build-base mssr_ws/build \
  --install-base mssr_ws/install --symlink-install --packages-select mssr_expert
```

In **ogni terminale** usato per i comandi successivi:

```bash
cd /home/lorenzo/MSSR_thesis
source /opt/ros/humble/setup.bash
source mssr_ws/install/setup.bash
export ROS_DOMAIN_ID=0
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
mkdir -p logs/datasets
```

È necessaria l'installazione locale di Isaac Sim usata da
`scripts/smores_ep/run_self_assembly.sh`, oltre a ROS 2 Humble e Nav2.
I runner automatici avviano i processi necessari; non avviare anche un runtime
manuale per lo stesso episodio. Per i batch scegli una cartella di output nuova.

## 2. Expert singoli: esecuzione automatica headless

Ogni comando include il runtime e l'assemblaggio iniziale della morfologia.
`--seeds` accetta un seed, una lista (`3102,3104,4689`) o intervalli inclusivi
(`3100:3104`).

### Scale — Snake8

```bash
python3 scripts/smores_ep/run_stair_headless_batch.py \
  --seeds 3104 --output-dir logs/expert_stairs_3104_001
```

Runner alternativo mantenuto nel repository:

```bash
python3 scripts/smores_ep/run_stair_path_ik_batch.py \
  --seeds 3104 --output-dir logs/expert_stairs_path_ik_3104_001
```

Attualmente entrambi invocano `crawl_stairs_spatial_concertina`; il nome del
secondo script è storico. Per registrare anche l'assemblaggio aggiungere
`--record-assembly-dataset`.

### Gap — Snake8

```bash
python3 scripts/smores_ep/run_gap_headless_batch.py \
  --seeds 4106 --output-dir logs/expert_gap_4106_001
```

### Navigazione planare — RC-Car8 / Nav2

```bash
python3 scripts/smores_ep/run_rc_car_nav2_headless_batch.py \
  --seeds 5100 --output-dir logs/expert_rc_nav2_5100_001
```

### Pulsante — RC-Car8 → MobileManipulator8 → RC-Car8

Il runner gestisce la sequenza di navigazione, riconfigurazione e manipolazione IK.

```bash
python3 scripts/smores_ep/run_button_expert_to_ik.py --seed 6101 --headless
```

Con interfaccia grafica:

```bash
python3 scripts/smores_ep/run_button_expert_to_ik.py --seed 6101
```

Per fermarsi prima dell'IK e salvare lo stato diagnostico:

```bash
python3 scripts/smores_ep/run_button_expert_to_ik.py --seed 6101 --stop-before-ik
```

Gli artefatti del pulsante sono in `logs/button_expert_to_ik/`.

### Campagne su più seed

```bash
python3 scripts/smores_ep/run_stair_headless_batch.py \
  --seeds 3102,3104,4689 --continue-on-failure \
  --output-dir logs/expert_stairs_batch_001
```

```bash
python3 scripts/smores_ep/run_gap_headless_batch.py \
  --seeds 4100,4104,4106 --continue-on-failure \
  --output-dir logs/expert_gap_batch_001
```

```bash
python3 scripts/smores_ep/run_rc_car_nav2_headless_batch.py \
  --seeds 5100,5101,5102 --continue-on-failure \
  --output-dir logs/expert_rc_batch_001
```

I runner batch di scale, gap e RC supportano `--plan-only` per generare i
manifest senza avviare Isaac. I risultati e i log vengono scritti nella
cartella specificata da `--output-dir`.

Campagna adattiva scale/gap:

```bash
python3 scripts/smores_ep/run_snake_obstacle_dataset_campaign.py \
  --episodes-per-level 5 --base-seed 1000 \
  --output-dir logs/expert_snake_campaign_001
```

## 3. Expert singoli: runtime GUI e comandi manuali

### Visualizzazione CAD e collider

Con `ros2 launch mssr_expert smores_runtime.launch.py`:

- `headless:=false` apre la GUI; `headless:=true` esegue senza finestra.
- `simple_visuals:=false` mostra il CAD completo.
- `simple_visuals:=true` mostra le geometrie semplificate dei collider.

GUI e CAD sono già i default del launch; nei comandi sotto sono espliciti.
Scrivere `simple_visuals:=false` senza spazi. I collider restano usati per
la fisica anche quando viene visualizzato il CAD.

Il launch diretto non esegue i runner batch Python: modificare
`run_stair_headless_batch.py` o `run_gap_headless_batch.py` non cambia il
risultato di un comando `ros2 launch`.

### Terminale 1 — scegliere uno scenario

Scale:

```bash
ros2 launch mssr_expert smores_runtime.launch.py \
  stair_test_course:=true stair_seed:=3104 performance:=true \
  headless:=false simple_visuals:=false \
  behavior_dataset_path:=$PWD/logs/datasets/manual_stairs_behavior.jsonl
```

Gap:

```bash
ros2 launch mssr_expert smores_runtime.launch.py \
  gap_test_course:=true gap_seed:=4106 performance:=true \
  headless:=false simple_visuals:=false \
  behavior_dataset_path:=$PWD/logs/datasets/manual_gap_behavior.jsonl
```

Navigazione RC:

```bash
ros2 launch mssr_expert smores_runtime.launch.py \
  rc_car_planar_test_course:=true rc_car_seed:=5100 performance:=true \
  headless:=false simple_visuals:=false
```

Solo piano libero, per assemblaggio e riconfigurazione:

```bash
ros2 launch mssr_expert smores_runtime.launch.py performance:=true \
  headless:=false simple_visuals:=false
```

Questo launch avvia Isaac, file bridge e nodo dei comportamenti. Lasciarlo
attivo durante assemblaggio e azioni successive. Non avvia automaticamente
l'assemblaggio e l'attraversamento: eseguire anche i comandi dei terminali
successivi.

### CAD nei runner automatici Python

I runner automatici elencati sotto passano invece `simple_visuals:=true`
al launch. Per usare il CAD sostituire, nello script effettivamente avviato:

```python
"simple_visuals:=true",
```

con:

```python
"simple_visuals:=false",
```

| Runner in `scripts/smores_ep/` | Come aprire la GUI |
|---|---|
| `run_composite_course.py` | Omettere `--headless` |
| `run_button_expert_to_ik.py` | Omettere `--headless` |
| `run_gap_headless_batch.py` | Aggiungere `--gui` |
| `run_stair_path_ik_batch.py` | Aggiungere `--gui` |
| `run_stair_headless_batch.py` | Cambiare anche `"headless:=true",` in `"headless:=false",` nello script |
| `run_rc_car_nav2_headless_batch.py` | Cambiare anche `"headless:=true",` in `"headless:=false",` nello script |

Questi runner non accettano `simple_visuals:=false` come argomento Python.
La modifica della visualizzazione non cambia la sequenza automatica degli
expert. Non serve ricompilare ROS per modifiche agli script avviati con
`python3 scripts/smores_ep/...`.

Esempio gap automatico con GUI, dopo aver impostato `simple_visuals:=false`
in `run_gap_headless_batch.py`:

```bash
python3 scripts/smores_ep/run_gap_headless_batch.py \
  --seeds 4106 --gui --output-dir logs/expert_gap_cad_4106_001
```

Esempio scale automatiche con GUI, dopo la stessa modifica in
`run_stair_path_ik_batch.py`:

```bash
python3 scripts/smores_ep/run_stair_path_ik_batch.py \
  --seeds 3104 --gui --output-dir logs/expert_stairs_cad_3104_001
```

### Terminale 2 — self-assembly

Scegliere il target: `snake8` per scale/gap, `rc_car8` per navigazione,
`mobile_manipulator8` per manipolatore, `bridge8` per la morfologia ponte.

```bash
MORPHOLOGY=snake8
ros2 run mssr_expert mssr_smores_self_assembly_node --ros-args \
  -p target_graph_path:=$PWD/mssr_ws/src/mssr_expert/config/smores_${MORPHOLOGY}.json \
  -p execution_id:=manual-assembly-001 \
  -p episode_id:=manual-assembly-001 \
  -p dataset_path:=$PWD/logs/datasets/manual_assembly.jsonl
```

Attendere `Parallel self-assembly completed.`, poi interrompere **solo questo
nodo** con Ctrl-C prima di avviare un altro expert nello stesso terminale.

### Terminale 2 — comportamento scale o gap dopo assemblaggio Snake8

Scale:

```bash
ros2 run mssr_expert mssr_smores_morphology_command_client \
  --morphology snake8 --command-id manual-stairs-001 \
  --behavior crawl_stairs_spatial_concertina --parameters-json '{}'
```

Gap:

```bash
ros2 run mssr_expert mssr_smores_morphology_command_client \
  --morphology snake8 --command-id manual-gap-001 \
  --behavior gap_crossing --parameters-json '{}'
```

Usare un nuovo `--command-id` per ogni azione. Il client attende lo stato
terminale del comando.

### Navigazione GUI dopo assemblaggio RC-Car8

Terminale 3, avviare Nav2 dopo il completamento dell'assemblaggio:

```bash
ros2 launch mssr_expert smores_nav2.launch.py
```

Terminale 2, quando Nav2 è attivo, eseguire il percorso dello stesso seed
usato per lo scenario:

```bash
python3 scripts/smores_ep/run_rc_car_nav2_route.py \
  --seed 5100 --episode-id manual-rc-001 \
  --result-json logs/manual_rc_result.json \
  --dataset-path logs/datasets/manual_rc_route.jsonl
```

### Self-reconfiguration tra morfologie

Richiede un robot già assemblato. Lasciare attivo il runtime, fermare l'expert
precedente e scegliere `snake8`, `bridge8`, `rc_car8` o `mobile_manipulator8`:

```bash
TARGET_MORPHOLOGY=rc_car8
ros2 run mssr_expert mssr_smores_self_reconfiguration_node --ros-args \
  -p source_graph_path:=auto \
  -p target_morphology:=$TARGET_MORPHOLOGY \
  -p execution_id:=manual-reconfiguration-001 \
  -p episode_id:=manual-reconfiguration-001 \
  -p dataset_path:=$PWD/logs/datasets/manual_reconfiguration.jsonl
```

Attendere `Self-reconfiguration completed.` prima di inviare comportamenti
alla nuova morfologia. Le possibilità fisiche dipendono dallo stato e dallo
spazio disponibile nello scenario.

## 4. Expert composite

Il runner avvia un runtime unico, assembla la prima morfologia, esegue le
riconfigurazioni e gli expert necessari, e raggiunge il goal finale.
Scale/gap usano Snake8, la navigazione usa RC-Car8, il pulsante comprende
RC-Car8 → MobileManipulator8 → RC-Car8.

### Un episodio in GUI

```bash
python3 scripts/smores_ep/run_composite_course.py --episode composite-c05
```

### Lo stesso episodio headless

```bash
python3 scripts/smores_ep/run_composite_course.py \
  --episode composite-c05 --headless
```

### Tutti gli episodi attualmente configurati

Nonostante il nome `smores_composite_campaign16.json`, il file contiene
attualmente i seguenti **11 episodi**. Ogni riga è un lancio GUI indipendente;
aggiungere `--headless` per eseguire senza interfaccia grafica.

| Episodio | Sequenza (seed) |
|---|---|
| `composite-0001` | RC 5100 → gap 4100 → scale 3104 → pulsante 6101 |
| `composite-c01` | RC 5100 → gap 4103 → scale 3104 |
| `composite-c02` | scale 3102 → RC 5100 → pulsante 6101 |
| `composite-c03` | RC 5100 → RC 5100 → pulsante 6101 |
| `composite-c04` | scale 4689 → RC 5100 → gap 4106 |
| `composite-c05` | gap 4106 → RC 5100 → pulsante 6101 |
| `composite-c06` | RC 5100 → gap 4104 → RC 5100 |
| `composite-c07` | scale 3102 → scale 4689 → scale 3102 |
| `composite-c08` | gap 4106 → gap 4104 → gap 4106 |
| `composite-c09` | RC 5100 → RC 5100 → RC 5100 |
| `composite-c10` | gap 4106 → scale 4689 → gap 4106 |

```bash
python3 scripts/smores_ep/run_composite_course.py --episode composite-0001
python3 scripts/smores_ep/run_composite_course.py --episode composite-c01
python3 scripts/smores_ep/run_composite_course.py --episode composite-c02
python3 scripts/smores_ep/run_composite_course.py --episode composite-c03
python3 scripts/smores_ep/run_composite_course.py --episode composite-c04
python3 scripts/smores_ep/run_composite_course.py --episode composite-c05
python3 scripts/smores_ep/run_composite_course.py --episode composite-c06
python3 scripts/smores_ep/run_composite_course.py --episode composite-c07
python3 scripts/smores_ep/run_composite_course.py --episode composite-c08
python3 scripts/smores_ep/run_composite_course.py --episode composite-c09
python3 scripts/smores_ep/run_composite_course.py --episode composite-c10
```

### Campagna breve completa, in sequenza headless

Si arresta al primo episodio che termina con codice di errore:

```bash
for number in {01..10}; do
  python3 scripts/smores_ep/run_composite_course.py \
    --episode "composite-c${number}" --headless || break
done
```

### Pianificazione, anteprima e diagnostica

Generare il piano senza Isaac:

```bash
python3 scripts/smores_ep/run_composite_course.py \
  --episode composite-c05 --plan-only
```

Aprire solo lo scenario, senza eseguire gli expert:

```bash
python3 scripts/smores_ep/run_composite_course.py \
  --episode composite-c05 --preview-only
```

Fermare l'esecuzione dopo uno stage (gli ID sono in `stage_plan.json`):

```bash
python3 scripts/smores_ep/run_composite_course.py \
  --episode composite-c05 --stop-after-stage 0
```

Scegliere una directory di output **non ancora esistente**:

```bash
python3 scripts/smores_ep/run_composite_course.py \
  --episode composite-c05 --runtime-dir logs/composite_course/c05_run_001
```

Il default salva sotto `logs/composite_course/`. `--execute` è già attivo per
default. In GUI il runtime resta aperto al termine per ispezione; chiuderlo
prima di avviare il prossimo episodio.

## 5. Infrastruttura expert generica / precedente

Questi launch richiedono un backend che pubblichi le osservazioni e riceva
le azioni del relativo framework; non avviano lo scenario Isaac SMORES.

Nodo expert generico con configurazione `expert.yaml`:

```bash
ros2 launch mssr_expert expert.launch.py
```

Curriculum manager, in un altro terminale:

```bash
ros2 launch mssr_expert curriculum.launch.py
```

Per selezionare esplicitamente ciascun expert generico (un comando alla volta):

```bash
ros2 run mssr_expert mssr_expert_node --ros-args \
  -p expert_name:=stage0_gap_crossing -p stage_id:=0 \
  -p task_type:=gap_crossing_temporary_bridge
```

```bash
ros2 run mssr_expert mssr_expert_node --ros-args \
  -p expert_name:=stage1_obstacle_traversal -p stage_id:=1 \
  -p task_type:=low_obstacle_climb_over
```

```bash
ros2 run mssr_expert mssr_expert_node --ros-args \
  -p expert_name:=stage2_stair_climb -p stage_id:=2 \
  -p task_type:=multi_step_dynamic_support
```

Nodo del precedente percorso fisso, solo per diagnostica su runtime e robot
già predisposti secondo il vecchio runbook:

```bash
ros2 run mssr_expert mssr_smores_obstacle_course_node
```

Per le missioni composite attuali usare `run_composite_course.py`.

## 6. Monitoraggio

In terminali separati, con la preparazione della sezione 1:

```bash
ros2 topic echo /mssr/morphology/status std_msgs/msg/String
```

```bash
ros2 topic echo /mssr/expert/self_reconfiguration/state std_msgs/msg/String
```

Le opzioni complete dei runner si consultano aggiungendo `--help`.
Fonte per gli ID delle missioni:
`mssr_ws/src/mssr_expert/config/smores_composite_campaign16.json`.
Ulteriori dettagli: [runbook SMORES](smores_obstacle_course_runbook.md).
