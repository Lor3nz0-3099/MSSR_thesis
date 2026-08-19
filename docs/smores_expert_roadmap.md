# Roadmap degli expert SMORES-EP

## Obiettivo

La pipeline finale deve produrre episodi completi e task-conditioned:

1. moduli separati;
2. self-assembly della morfologia iniziale;
3. locomozione o manipolazione;
4. self-reconfiguration quando cambiano i requisiti del task;
5. nuova locomozione o manipolazione;
6. successo, fallimento e recovery osservabili nel dataset.

Le reference suggeriscono di mantenere inizialmente lo stesso numero di moduli
fra le forme. Sette moduli sono una scelta migliore di sei: sono sufficienti
per tre morfologie funzionalmente diverse e permettono transizioni dirette fra
RC Car, Snake e Mobile Manipulator.

## Catalogo iniziale

| Morfologia | Moduli | Funzione | Target | Stato self-assembly |
|---|---:|---|---|---|
| Snake7 | 7 | scale, terreno sconnesso, gap | `smores_snake7.json` | pipeline progressiva corretta sul replay del primo run; da riverificare in Isaac |
| Mobile Manipulator8 | 8 | locomozione piana, braccio, high reach | `smores_mobile_manipulator8.json` | quattro moduli longitudinali formano il braccio: quello collegato alla root resta a terra e traziona, i tre successivi costituiscono la parte sollevata |
| RC Car7 | 7 | differential drive e spinta di carichi | `smores_rc_car7.json` | due wave planari progressive senza helper più folding finale |

Le altre reference nominano circa dodici forme aggiuntive, con alcune
sovrapposizioni funzionali o nomi diversi per concetti simili: Holonomic
Vehicle, Rolling Loop, DoubleDriver, Stair Climber, Swerve Lifter, Backhoe,
Car, Proboscis, Scorpion, Walker, Omni-Driver e Mobile Observer. Non sono un
prerequisito per la self-reconfiguration. Il primo benchmark coerente richiede
forme con lo stesso numero di moduli, quindi il catalogo a sette moduli è già
sufficiente.

I target sono in `mssr_ws/src/mssr_expert/config/` e descrivono la topologia
cinematica, le facce coinvolte, i ruoli e le capacità attese.

Fonti principali:

- `references/SMORES-EP.pdf`, Fig. 6: Mobile Manipulator a 7 moduli;
- `references/SMORES-EP.pdf`, Fig. 12: RC Car a 7 moduli;
- `references/Accomplishing High-Level Tasks with Modular Robots.pdf`,
  Tabella 2 e Fig. 11: Snake7 e comportamento di salita/discesa;
- `references/chao_smores_reconfiguration_2019.pdf`, Figg. 15-16 e Tabella
  III: Driver a 7 moduli, Snake a 7 moduli e azioni Driver-to-Snake.

### Nota sulla RC Car

La RC Car operativa usa una variante planare della topologia di Fig. 12. La
prima wave forma una catena centrale da tre tramite connessioni `TOP/BOTTOM`.
La seconda wave collega quattro moduli usando la loro faccia `BOTTOM` sulle
facce `LEFT/RIGHT` dei due estremi. Tutte le facce mobili sono quindi assiali e
non serve un helper. Al termine quattro primitive `set_tilt` trasformano la
figura appiattita nella postura di locomozione.

L'ordine resta quello progressivo root-to-leaves dell'Algoritmo 1 del paper;
non è invece una replica esatta della topologia RC Car di Fig. 12, che richiede
un modulo aiutante per i due docking laterali del telaio.

Il `TILT` porta i moduli ruota nella postura di contatto col terreno. La
propulsione viene dai motori `LEFT/RIGHT`. Il `PAN` può orientare un gruppo
ruota collegato tramite la faccia rotante e quindi può servire per sterzare,
ma non sostituisce i motori delle ruote.

## Stato attuale

Sono già disponibili:

- grafo attribuito corrente, target graph e task graph;
- validazione di target ad albero con facce e clocking;
- unfolding planare;
- scelta del root e assegnamento ottimo modulo-slot;
- generazione di wave parallele root-to-leaves;
- primitive `drive_to_pose`, `align_faces`, `dock`, `undock`, `pan`, `tilt`;
- pianificazione in wave parallele con barriere collettive
  `REACH -> ALIGN -> APPROACH -> DOCK`, retry per fase e recovery dei docking
  rifiutati senza togliere il parallelismo tra moduli indipendenti;
- policy di esecuzione unica e indipendente dalla morfologia, condivisa tra
  self-assembly e fasi di assembly della self-reconfiguration;
- logging JSONL `mssr.expert_transition.v3` con `graph_t`, target, assignment,
  azione e `graph_t_plus_1`;
- scena di self-assembly generalizzata da 3 a N moduli, con layout radiale per
  gli esperimenti più grandi.

## Cosa manca

### 1. Completare l'expert di self-assembly

Priorità immediata:

- calibrare nella GUI le due wave progressive della nuova RC Car7 senza
  helping module e i quattro tilt finali da ±45°;
- verificare la topologia finale tramite isomorfismo del grafo e facce, non
  soltanto contando le connessioni;
- aggiungere un motion planner multi-modulo collision-free. L'esecuzione
  attuale coordina le risorse, ma non pianifica globalmente traiettorie che si
  evitano a vicenda;
- validare in Isaac le scene a 7 moduli con Snake7, Mobile Manipulator7 e RC
  Car7.

### 2. Separare topologia, postura e comportamento

Raggiungere le sei connessioni corrette non rende ancora la forma operativa.
Serve una libreria di morfologie con, per ogni forma:

- target graph;
- postura nominale post-assembly (`PAN/TILT` per modulo);
- comportamento parametrico o gait;
- capacità e limiti: terreno, altezza scalino, payload, ingombro;
- criterio di successo e condizioni di arresto sicuro.

Servono inoltre behavior composti per `drive_connected_component`,
`set_cluster_posture`, `snake_climb`, `high_reach` e `push_payload`.
`drive_connected_component` non deve diventare una falsa primitiva atomica:
deve coordinare i motori dei moduli che costituiscono il cluster.

Prima versione implementata:

- libreria `smores_morphology_behaviors.json` con posture, limiti e gruppi di
  locomotori risolti dai ruoli assegnati durante la self-assembly;
- nodo `mssr_smores_morphology_behavior_node`;
- comandi `prepare`, `drive`, `stop`, `wave`, `straighten`, `raise_arm`,
  `reach_forward`, `lower_arm` e `stow` dove applicabili;
- conversione del twist del cluster in velocità differenziali per i gruppi
  sinistro e destro;
- invio serializzato e verificato delle primitive PAN/TILT;
- dead-man timeout nel runtime Isaac: se il nodo smette di aggiornare
  `/mssr/actions`, i motori tornano a zero dopo `0.5 s`.

Questa è una baseline operativa, non ancora la calibrazione definitiva dei
gait. Gli angoli di ruote e braccio sono conservativi e devono essere misurati
nella GUI; in particolare, una postura che raggiunge il target articolare ma
non produce un support polygon stabile va corretta nella libreria senza
modificare executor o dataset schema.

### 3. Expert di self-reconfiguration

L'expert generale di reconfiguration è implementato per l'intero catalogo
`Snake7`, `RC Car7` e `Mobile Manipulator7`. Il planner:

- riconosce automaticamente la morfologia sorgente dal grafo fisico e dal
  catalogo installato;
- trova la massima sottoconfigurazione comune connessa alla root target;
- a parità di topologia conservata seleziona l'assegnamento a costo di moto
  minimo rispetto alle pose fisiche correnti, evitando scambi incrociati tra
  moduli equivalenti tramite Kuhn-Munkres/Hungarian;
- conserva da due a quattro connessioni nelle sei transizioni dirette note;
- applica la postura neutra specifica della sorgente prima degli undock;
- ordina i moduli mobili dalle foglie esterne verso il componente conservato,
  mantenendo l'identità longitudinale destra/sinistra nell'assegnamento;
- esegue fasi progressive `undock -> reach -> align -> approach -> dock`
  organizzate in ondate; il planner conserva l'indipendenza topologica e
  l'executor ammette in parallelo tutti i moduli della wave, imponendo una
  barriera prima di ogni fase successiva; retry, recovery e gate di contatto
  sono gli stessi della self-assembly e non dipendono dal target;
- applica la postura operativa specifica del target, incluso il folding
  coordinato delle quattro ruote RC Car e il sollevamento del braccio Mobile
  Manipulator;
- conclude soltanto quando il grafo corrente coincide con tutte le connessioni
  e le facce del target sotto l'assegnamento pianificato.

Il nodo ROS è `mssr_smores_self_reconfiguration_node`; il target può essere
selezionato per `target_morphology` oppure fornendo un nuovo target graph. Nel
task graph e nel dataset sono esposti sorgente, target, assegnamento, numero di
connessioni conservate e avanzamento delle operazioni: sono le variabili che
un policy MARL potrà osservare o scegliere. Restano da aggiungere la
ripianificazione online in caso di deviazione, il moto sicuro di sottocluster
rigidi e la generazione neurale di target validi. Il matching topologico usa
programmazione dinamica sugli alberi con facce etichettate; una volta fissata
la sottoconfigurazione conservata, i moduli mobili sono assegnati ai ruoli
liberi con Kuhn-Munkres in tempo polinomiale.

La parallelizzazione è conservativa e configurabile dal nodo ROS tramite
`parallel_path_clearance_m` (default `0.12 m`) e `max_parallel_actions`
(default `0`, cioè nessun limite artificiale alla dimensione dell'ondata).
Portando `max_parallel_actions` a `1` si ripristina esattamente l'esecuzione
seriale.

| Transizione | Connessioni conservate | Undock/Dock | Wave nuovi dock |
|---|---:|---:|---|
| Snake7 -> RC Car7 | 2 | 4 / 4 | 2 + 2 |
| Snake7 -> Mobile Manipulator7 | 4 | 2 / 2 | 1 + 1 |
| RC Car7 -> Snake7 | 2 | 4 / 4 | 2 + 2 |
| RC Car7 -> Mobile Manipulator7 | 4 | 2 / 2 | 2 |
| Mobile Manipulator7 -> Snake7 | 4 | 2 / 2 | 1 + 1 |
| Mobile Manipulator7 -> RC Car7 | 4 | 2 / 2 | 2 |

### 4. Expert end-to-end e dataset IL

Gli expert curriculum `gap_crossing`, `obstacle_traversal` e `stair_climb`
presenti oggi usano ancora primitive del prototipo sferico (`roll_to`,
`dock_to_surface`, `attach_as_pivot`) e non sono expert task-level SMORES.

L'expert end-to-end deve invece scegliere una entry dalla libreria in base a
task e ambiente, quindi invocare assembly, behavior e reconfiguration. Una
prima missione riproducibile può essere:

1. assemblare RC Car7;
2. raggiungere e spingere/trasportare un oggetto su piano;
3. rilevare una scala e riconfigurarsi in Snake7;
4. superare la scala;
5. riconfigurarsi in Mobile Manipulator7;
6. raggiungere o depositare l'oggetto in altezza.

All'inizio l'ambiente può essere osservato tramite ground truth di Isaac,
equivalente al VICON usato nei paper. Per imitation learning devono essere
registrati anche:

- `episode_id`, seed e curriculum difficulty;
- task e stato dell'ambiente;
- morfologia corrente e target;
- decisione high-level (`assemble`, `move`, `reconfigure`, `interact`);
- azione primitiva o behavior composto e relativa maschera di fattibilità;
- esito, reason code, retry, recovery e durata;
- grafo prima e dopo l'azione;
- successo finale del task.

Il logger esiste, ma non esistono ancora loader, split, controlli qualità,
behavior cloning/DAgger o codice di training in `drl/`. Quella parte va
iniziata solo dopo avere expert deterministici stabili; altrimenti il dataset
impara timeout, attese e recovery accidentali invece delle decisioni corrette.

## Ordine di implementazione consigliato

1. Riverificare Snake7 dopo la correzione curve/pivot con CAD completo.
2. Eseguire Mobile Manipulator7 e calibrare la postura high-reach.
3. Validare dinamicamente tutte le sei transizioni del catalogo e calibrare gli
   staging dopo ogni undock.
4. Implementare locomozione coordinata e gait di ogni morfologia.
5. Aggiungere comando task-level e generazione controllata di nuovi target.
6. Aggiungere mission planner task-conditioned e scenari con ambiente.
7. Generare dataset con molte pose iniziali, seed e difficoltà.
8. Aggiungere validazione dataset e training imitation learning.

Ogni milestone deve essere accettata sia in modalità headless per dataset ad
alto throughput, sia con CAD completo in GUI per l'ispezione qualitativa.
