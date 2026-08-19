# MSSR Thesis Framework - Recap

## Obiettivo del progetto

Il progetto serve a costruire un framework di simulazione per **Modular Self-Reconfigurable Robots** in Isaac Sim 6.0.

L'idea non è solo far muovere una sfera, ma creare una base modulare per:

- robot composti da molti moduli;
- connessioni magnetiche tra moduli;
- grafo dinamico della configurazione;
- expert deterministici;
- imitation learning;
- multi-agent reinforcement learning;
- curriculum learning.

---

## Idea generale

Il sistema è diviso in blocchi.

```text
Isaac Sim
  simula fisica, moduli, contatti, joint, ostacoli

Python framework
  rappresenta stato, azioni, grafo, scenari, task

ROS2
  controller esterni, expert deterministici, raccolta dati

rl/
  imitation learning, GCN, MARL, policy training
```

Isaac non deve contenere tutta l'intelligenza.
Isaac deve essere il mondo fisico.

ROS2 e `rl/` useranno Isaac come ambiente.

---

## Struttura del progetto

```text
MSSR_Thesis/
├── assets/
├── configs/
│   ├── actions.json
│   ├── sample_actions_v1.json
│   ├── reset_action_v1.json
│   └── scenarios/
├── graphs/
├── logs/
├── rl/
├── robots/
├── ros2_bridge/
├── scripts/
├── worlds/
└── README.md
```

---

## 1. Robot modulare

Per ora ogni modulo è una **sfera rigida**.

La sfera rappresenta un modulo robotico provvisorio. In futuro potrà essere sostituita da un modello 3D più realistico.

File principale:

```text
robots/spherical_robot.py
```

Contiene:

- configurazione del modulo;
- creazione della sfera in Isaac;
- massa;
- collisione;
- rigid body;
- colore;
- stato iniziale.

Ogni modulo ha:

```text
module_id
prim_path
body_frame_id
world_frame_id
radius
mass
position
```

Esempio:

```text
sphere_0
sphere_1
sphere_2
...
```

---

## 2. Stato dei moduli

Ogni modulo ha uno stato pubblico.

File:

```text
robots/module_state.py
```

Lo stato contiene:

- posizione;
- orientamento;
- velocità lineare;
- velocità angolare;
- massa;
- raggio;
- ruolo;
- capacità;
- comandi ricevuti;
- connessioni magnetiche;
- contatti sulla superficie.

Questo è importante perché sarà la base per:

- ROS2;
- grafo;
- expert;
- imitation learning;
- reinforcement learning.

---

## 3. Controllo del modulo

File:

```text
robots/control.py
```

Per ora il controllo è semplice:

```text
vx
vy
yaw_rate
```

Cioè:

- velocità in avanti/indietro;
- velocità laterale;
- rotazione.

Il controller applica velocità al rigid body in Isaac.

Esiste anche un controller multi-modulo, così in futuro ogni modulo potrà ricevere un comando diverso.

---

## 4. Azioni

File:

```text
robots/actions.py
```

Un'azione di simulazione contiene due parti:

```text
locomotion
magnetic
```

### Locomotion

Dice come si muove ogni modulo.

Esempio:

```json
"sphere_0": {
  "vx": 0.15,
  "vy": 0.0,
  "yaw_rate": 0.0
}
```

### Magnetic

Dice se due moduli devono attaccarsi o staccarsi.

Esempio:

```json
{
  "module_a_id": "sphere_0",
  "module_b_id": "sphere_1",
  "command": "attach",
  "joint_type": "spherical"
}
```

---

## 5. Connessione magnetica

I moduli si attaccano come se avessero magneti elettropermanenti.

Concetto importante:

```text
contatto != attacco
```

Due moduli possono essere in contatto, ma non sono collegati finché il magnete non viene attivato.

File:

```text
robots/magnetic_attachment.py
```

Quando il magnete viene attivato, Isaac crea un joint fisico tra i moduli.

Tipi di joint supportati:

```text
rigid
spherical
hinge
```

### Rigid

I moduli diventano praticamente solidali.

### Spherical

I moduli restano attaccati ma possono ruotare liberamente.

### Hinge

I moduli restano attaccati ma possono ruotare attorno a un asse.

Questa scelta è importante perché nel grafo ogni edge avrà anche il tipo di joint.

---

## 6. Contatti sulla superficie

File:

```text
robots/surface_attachment.py
```

I moduli sono sferici, quindi per ora possono attaccarsi su tutta la superficie.

Il sistema calcola quando due sfere sono abbastanza vicine da essere considerate in contatto.

Poi un algoritmo esterno può decidere se attivare o no il magnete.

---

## 7. Grafo del robot

File:

```text
graphs/robot_graph.py
```

Il robot viene rappresentato come grafo.

```text
nodi = moduli
edge = contatti o connessioni
```

Ogni nodo contiene attributi del modulo:

- posizione;
- velocità;
- massa;
- raggio;
- comandi;
- capacità.

Ogni edge contiene attributi della connessione:

- stato;
- tipo di joint;
- magnete attivo o no;
- posizione relativa;
- punto di contatto.

Questo grafo sarà l'input naturale per:

- GCN;
- expert deterministici;
- imitation learning;
- MARL.

---

## 8. Scenari

Gli scenari descrivono il mondo iniziale.

File:

```text
worlds/scenario_config.py
```

Gli scenari non sono codice Isaac.
Sono configurazioni JSON.

Esempio:

```text
configs/scenarios/
```

Ora gli scenari possono contenere:

- nome;
- tipo di task;
- livello del curriculum;
- difficoltà;
- numero di moduli;
- spawn casuale;
- goal;
- ostacoli;
- timeout episodio.

---

## 9. Spawn casuale

I moduli non partono già attaccati.

Esempio:

```json
"random_spawn": {
  "module_count": 6,
  "x_range": [-3.0, 2.5],
  "y_range": [-2.5, 2.5],
  "z": 1.0,
  "min_distance": 1.35,
  "seed": 100
}
```

Significa:

- crea 6 moduli;
- mettili in posizioni casuali;
- non farli sovrapporre;
- usa un seed per rendere l'esperimento ripetibile.

---

## 10. Curriculum Learning

Il curriculum learning viene considerato già negli scenari.

Ogni scenario ha:

```text
task_type
stage
difficulty
```

Esempio:

```json
"curriculum": {
  "task_type": "assemble_reconfigure_stair_climb",
  "stage": 2,
  "difficulty": 0.75
}
```

Questo permette in futuro di allenare la policy partendo da task semplici e aumentando la difficoltà.

---

## 11. Scenari attuali

### Stage 0

```text
curriculum_00_self_assembly_line.json
```

Task:

```text
spawn casuale
assemblaggio
movimento verso goal
```

### Stage 1

```text
curriculum_01_low_step.json
```

Task:

```text
spawn casuale
assemblaggio
superamento scalino basso
raggiungimento goal
```

### Stage 2

```text
curriculum_02_three_step_stair.json
```

Task:

```text
spawn casuale
assemblaggio
riconfigurazione
salita scala a 3 gradini
raggiungimento goal
```

---

## 12. Ostacoli

File:

```text
worlds/scenario_obstacles.py
```

Per ora gli ostacoli sono box statici con collisione.

Questo basta per rappresentare:

- scalini;
- piattaforme;
- scale semplici.

Gli ostacoli vengono descritti nello scenario JSON.

Esempio:

```json
{
  "name": "low_step",
  "position": [4.0, 0.0, 0.175],
  "size": [1.2, 3.0, 0.35]
}
```

---

## 13. TaskEvaluator

File:

```text
worlds/task_evaluator.py
```

Il TaskEvaluator misura lo stato del task.

Non è ancora la reward RL, ma è la base per costruirla.

Calcola:

- numero di moduli;
- numero di componenti connesse;
- dimensione del gruppo più grande;
- rapporto di assemblaggio;
- centroide dei moduli;
- altezza massima;
- distanza dal goal;
- moduli caduti;
- successo;
- timeout;
- fine episodio.

Esempio concettuale:

```text
assembled_ratio = quanti moduli sono nel gruppo principale
distance_to_goal = quanto manca al goal
is_success = assemblato + goal raggiunto
```

Queste metriche serviranno per:

- expert deterministici;
- dataset IL;
- reward MARL;
- risultati sperimentali;
- ablation study.

---

## 14. Gravità e fisica

File:

```text
worlds/basic_world.py
```

La gravità è stata aumentata di default a:

```text
19.62 m/s²
```

per evitare che le sfere sembrino fluttuare troppo.

Se si vuole tornare alla gravità terrestre:

```bash
--gravity-magnitude 9.81
```

---

## 15. JSON bridge

Isaac Sim 6.0 usa Python 3.12.
ROS2 Humble usa Python 3.10.

Per questo non conviene importare `rclpy` dentro Isaac.

La soluzione è un bridge tramite file JSON.

File:

```text
ros2_bridge/mssr_file_bridge.py
```

Isaac scrive file JSON:

```text
module_states.json
robot_graph.json
state_graph.json
task_metrics.json
```

Il bridge ROS2 li legge e li pubblica su topic.

Topic pubblicati:

```text
/mssr/module_states
/mssr/robot_graph
/mssr/state_graph
/mssr/task_metrics
```

Topic ricevuto:

```text
/mssr/actions
```

ROS2 pubblica azioni su `/mssr/actions`.
Il bridge le scrive in:

```text
configs/actions.json
```

Isaac legge quel file e applica le azioni.

---

## 16. Formato azioni

Formato consigliato:

```json
{
  "schema_version": "mssr.actions.v1",
  "reset": false,
  "locomotion": {
    "sphere_0": {
      "vx": 0.15,
      "vy": 0.0,
      "yaw_rate": 0.0
    }
  },
  "magnetic": [
    {
      "module_a_id": "sphere_0",
      "module_b_id": "sphere_1",
      "command": "attach",
      "joint_type": "spherical"
    }
  ]
}
```

Può contenere anche:

```json
"reset": true
```

per resettare l'episodio.

---

## 17. Come gira tutto insieme

Schema generale:

```text
Isaac Sim
  crea scenario
  spawna moduli
  simula fisica
  aggiorna stato
  aggiorna grafo
  calcola task metrics
  scrive JSON

ROS2 bridge
  legge JSON
  pubblica topic ROS2
  riceve azioni
  scrive actions.json

Expert / policy
  legge stato/grafo/task
  decide azioni
  pubblica /mssr/actions
```

---

## 18. Dove andranno IL e MARL

Gli algoritmi intelligenti non vanno dentro Isaac.

### ROS2

Qui metteremo:

- expert deterministici;
- controller esterni;
- nodi di test;
- nodi per raccogliere dati.

### rl/

Qui metteremo:

- dataset loader;
- behaviour cloning;
- GCN policy;
- APPO / MARL;
- reward;
- training;
- evaluation;
- ablation study.

---

## 19. Stato attuale

Abbiamo già:

- modulo sferico;
- controllo base;
- stato modulo;
- connessioni magnetiche;
- joint fisici;
- grafo dinamico;
- scenari curriculum;
- spawn casuale;
- ostacoli;
- TaskEvaluator;
- JSON bridge con task metrics;
- formato azioni versionato.

---

## 20. Prossimo passo

Il prossimo passo naturale è passare a ROS2.

Prima cosa da fare:

```text
creare un primo nodo ROS2 expert minimale
```

Questo nodo dovrà:

1. leggere `/mssr/state_graph`;
2. leggere `/mssr/task_metrics`;
3. decidere azioni semplici;
4. pubblicare `/mssr/actions`.

All'inizio l'expert può essere molto semplice:

```text
se due moduli sono vicini -> attach
se il gruppo è assemblato -> muovi verso goal
se bloccato -> prova riconfigurazione semplice
```

Questo expert servirà poi per generare dataset di imitation learning.
