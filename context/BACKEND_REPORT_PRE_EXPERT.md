# Report backend MSSR prima degli algoritmi deterministici

Data di consolidamento: 24 luglio 2026.

## 1. Stato esecutivo

Il backend SMORES-EP è strutturalmente pronto per essere usato dagli algoritmi
deterministici: asset fisico, controllo, docking, primitive goal-oriented,
routing multi-modulo, stato osservabile e grafo dinamico condividono contratti
espliciti. Gli algoritmi del workspace `mssr_ws` non sono stati riscritti in
questa fase.

FreeBOT non ha ancora lo stesso livello di maturità. Il codice sferico non è
un artefatto da eliminare: costituisce il prototipo del backend FreeBOT.
Docking fisico robusto e crawling tra due FreeBOT restano prove di accettazione
obbligatorie prima di dichiarare quel backend pronto per gli stessi esperti.

| Backend | Stato | Uso ammesso |
|---|---|---|
| SMORES-EP enhanced | implementato e verificato in smoke test | sviluppo e test iniziale degli esperti deterministici |
| FreeBOT sferico | in sviluppo | conservare, migrare e completare docking/crawling |
| Core grafo/bridge | condiviso | nessuna dipendenza dalla geometria CAD specifica |

I manifesti macchina sono:

- `configs/backends/smores_ep_enhanced.json`;
- `configs/backends/freebot.json`;
- `configs/backend_manifest_schema_v1.json`.

## 2. Separazione dei livelli

La dipendenza prevista è:

1. descrizione del robot;
2. backend runtime Isaac;
3. comandi atomici;
4. primitive goal-oriented;
5. adattatore stato/grafo;
6. esperti deterministici specifici della famiglia;
7. modello IL/MARL condiviso.

Il core condiviso non deve importare il backend SMORES o FreeBOT. Sono gli
adattatori della famiglia a tradurre il grafo e le azioni canoniche verso i
rispettivi attuatori e connettori.

I file SMORES principali sono:

```text
scripts/smores_ep/src/smores_ep/
  config/geometry.py            geometria misurata dal CAD
  config/physics.py             masse, attriti, coppie e velocità
  control/teleop.py             comando atomico e ROS teleop
  docking/model.py              geometria e selezione delle facce
  isaac/docking.py              FixedJoint e topologia runtime
  isaac/dynamic_stage.py        articolazione e drive PhysX
  isaac/command_router.py       routing dinamico multi-modulo
  isaac/primitive_executor.py   lifecycle e arbitraggio delle risorse
  isaac/state_graph_publisher.py
  primitives/model.py           protocollo goal/status
  scenarios/two_module_docking.py
  scenarios/multi_module_lift.py
```

## 3. Modello fisico SMORES-EP

Il modello deriva dal CAD importato, non dai bounding box ruotati:

- diametro delle due ruote motrici: `62.12 mm` nel proxy trasformato,
  coerente con il diametro CAD nominale di `62 mm`;
- raggio usato dal controllo: `31.06 mm`;
- carreggiata: `70.410 mm`;
- massa totale: `0.454 kg`;
- cinque rigid link e quattro giunti revolute;
- `LEFT` e `RIGHT` sono le due ruote motrici;
- `TOP` è il disco PAN/TILT;
- `BOTTOM` è il telaio passivo `base-chassis`;
- TILT limitato a `[-π/2, +π/2]`;
- PAN continuo, controllato in coordinate logiche non avvolte;
- velocità massima terrestre: `0.088 m/s`, pari a `1.1 body lengths/s`;
- velocità ruota corrispondente: circa `2.83 rad/s`.

La cinematica differenziale usa simultaneamente ruota sinistra e destra.
Girare sul posto significa comandare velocità uguali e opposte; è quindi un
uso completo della risorsa ruote, come i tasti `J`/`L` della teleoperazione.
Una curva usa contemporaneamente traslazione e yaw, ma non rappresenta due
primitive indipendenti.

I due motori interni sono accoppiati:

- versi concordi degli ingranaggi interni producono TILT;
- versi opposti producono PAN;
- PAN e TILT non possono essere richiesti contemporaneamente;
- le ruote possono invece muoversi mentre è in corso PAN oppure TILT.

Le animazioni degli ingranaggi rispettano le diagonali esterna e interna del
CAD e il rapporto visuale `48:9`.

## 4. Risorse e concorrenza

L’esecutore non blocca più tutto il modulo per un singolo goal. Le risorse
esclusive sono:

| Risorsa | Ambito | Attuatori |
|---|---|---|
| `locomotion:<module>` | per modulo | ruota LEFT + ruota RIGHT |
| `internal_motion:<module>` | per modulo | due motori interni, modalità PAN oppure TILT |
| `connector:<module>:<face>` | per faccia | magnete della faccia indicata |

Conseguenze:

- drive + PAN sullo stesso modulo: ammesso;
- drive + TILT sullo stesso modulo: ammesso;
- spin + PAN/TILT: ammesso;
- PAN + TILT sullo stesso modulo: rifiutato con `RESOURCE_BUSY`;
- due drive sullo stesso modulo: rifiutati;
- goal su moduli distinti: ammessi;
- due operazioni sulla stessa faccia magnetica: rifiutate;
- `align_faces` riserva la locomozione del modulo mobile e del target per
  impedire che il target venga spostato durante l’allineamento.

Il router multi-modulo tratta tutte le istanze come robot identici. Un modulo
presente nella mappa dei comandi ha i drive abilitati; un modulo senza comando
resta dinamico e trainabile. Non esiste una classe fisica “passive module”.

Per un modulo momentaneamente non comandato:

- PAN e TILT sono mantenuti per sostenere la struttura;
- una ruota collegata come faccia `LEFT` o `RIGHT` viene mantenuta;
- le altre ruote sono libere con piccolo attrito di cuscinetto;
- un collegamento `BOTTOM` non applica un freno alle ruote.

Quando quel modulo riceve un goal, il router riattiva i suoi controllori. Ciò
permette in futuro il drive cooperativo di tutti i moduli connessi.

## 5. Primitive SMORES-EP

Le primitive esposte sono:

| Primitive | Moduli | Parametri principali |
|---|---:|---|
| `drive_to_pose` | 1 | `x_m`, `y_m`, `yaw_rad` |
| `align_faces` | 2 | `face_a`, `face_b`, clocking |
| `dock` | 2 | endpoint faccia espliciti |
| `undock` | 2 | endpoint faccia espliciti |
| `set_pan` | 1 | `angle_rad` |
| `rotate_pan_by` | 1 | `delta_rad` |
| `set_tilt` | 1 | `angle_rad` |
| `rotate_tilt_by` | 1 | `delta_rad` |

Ogni goal ha:

- `goal_id` univoco;
- timeout;
- ammissione o rifiuto con reason code;
- stato `ACCEPTED`, `RUNNING`, `SUCCEEDED`, `FAILED` o `CANCELED`;
- fase, progresso e feedback numerico;
- risultato terminale.

I target PAN relativi possono attraversare più giri. Il controller usa un
servo di velocità su una coordinata continua e non invia a PhysX target
revolute oltre `[-2π, 2π]`. I target TILT, assoluti o relativi, sono validati
rispetto ai limiti meccanici.

## 6. Interfaccia ROS 2

Isaac Sim non importa il `rclpy` di ROS Humble. Il processo esterno
`ros2_bridge/mssr_file_bridge.py` scambia file JSON atomici con Isaac ed
espone:

- `/mssr/primitives/goal`;
- `/mssr/primitives/cancel`;
- `/mssr/primitives/status`;
- `/mssr/module_states`;
- `/mssr/robot_graph`;
- `/mssr/state_graph`.

Il protocollo è action-like, non un server ROS Action nativo. Conserva però
goal ID, cancellazione, feedback e risultati. Lo stato singolo usa
`mssr.primitive_status.v1`; gli snapshot concorrenti usano
`mssr.primitive_status_batch.v1`.

La teleoperazione diretta resta disponibile tramite `/cmd_vel`, PAN, PAN
relativo e TILT. Le primitive possono sovrapporre solo la risorsa che
possiedono: un goal TILT può quindi convivere con la locomozione teleoperata.

Il trasporto file è intenzionalmente un latest-value bridge. Goal distinti
destinati a essere concorrenti devono essere inviati in sequenza, attendendo
almeno la ricezione/ammissione del precedente, per evitare che due scritture
ROS avvenute prima del poll di Isaac si sovrascrivano.

## 7. Docking

Ogni modulo ha quattro connettori distinti: `LEFT`, `RIGHT`, `TOP`, `BOTTOM`.
Il comando deterministico deve indicare entrambi gli endpoint. La modalità
legacy che cerca la coppia migliore rimane solo per teleoperazione e test.

Un attach è ammesso quando:

- entrambe le facce sono libere;
- la separazione normale rientra nella capture gate;
- l’offset laterale conserva una sovrapposizione prevalente dell’area;
- le normali sono opposte entro tolleranza;
- il clocking è compatibile con multipli di 90°.

Il gate attuale usa fino a `8 mm` tra i marker nominali, `20 mm` di offset
laterale e `12°` per normale/clocking. Queste tolleranze rappresentano una
faccia magnetica con area, non un punto matematico.

L’attach crea un `UsdPhysics.FixedJoint` runtime fra i rigid link delle due
facce. L’undock elimina quel joint. La topologia viene aggiornata
dinamicamente e una faccia non può appartenere a due collegamenti.

Il joint è al momento non rompibile: non modella la failure meccanica del
magnete sotto carico. Tale scelta è registrata come fixture del benchmark.

## 8. Requisiti funzionali di gruppo

`drive_connected_component` e `lift_chain` non sono primitive atomiche.
Sono comportamenti composti e requisiti di accettazione:

- un modulo deve poter trainare un altro modulo collegato;
- un gruppo può muoversi cooperativamente comandando le ruote dei singoli
  moduli;
- il design SMORES-compatible migliorato deve sollevare almeno 5 moduli e
  mira a 7 moduli in catena.

Questi non sono marcati come capacità “solo simulazione”. Descrivono il target
di un futuro modulo fisico migliorato.

Per riprodurre il payload in Isaac vengono registrate separatamente tre
fixture:

- profilo di coppia maggiorato;
- joint di docking non rompibile;
- supporto anti-ribaltamento opzionale.

Il supporto è un D6 joint fra il corpo del modulo di base e il mondo. Blocca
roll e pitch quando inizia il sollevamento, ma lascia liberi `X`, `Y`, `Z` e
yaw; il modulo continua quindi a traslare e sterzare tramite le ruote. Include
un drive yaw limitato per compensare il grande momento della catena. Non è
parte della percezione o della primitiva SMORES: è metadato della prova e
rappresenta il requisito strutturale/di appoggio che un design reale dovrà
soddisfare senza affidarsi a un vincolo verso il mondo.

## 9. Stato osservabile e grafo dinamico

Il publisher produce gli schemi:

- `mssr.module_states.v2`;
- `mssr.robot_graph.v2`;
- `mssr.state_graph.v2`.

Ogni nodo contiene:

- famiglia e tipo di modulo;
- pose e twist nel mondo;
- posizioni, velocità, limiti e disponibilità dei quattro attuatori;
- stato e frame delle quattro facce;
- ruolo corrente, ruolo target, confidenza e sorgente;
- `functional_role` strutturato;
- salute, disponibilità del controllo e fixture attive;
- requisiti del profilo enhanced.

Ogni arco latched contiene:

- entrambi gli ID modulo;
- entrambi gli ID faccia;
- stato della connessione;
- tipo di joint e DoF relativi;
- trasformazione relativa;
- clocking discreto;
- ruolo strutturale e proprietà load-bearing.

I contatti candidati sono archi transitori separati dai collegamenti latched.
Il grafo cambia con dock, undock e cambiamenti della popolazione.

Il livello expert aggiunge `mssr.task_graph.v1`, un multigrafo attribuito che
conserva contemporaneamente:

- nodi fisici correnti;
- nodi logici `target_slot`;
- archi distinti `contact`, `current_connection` e `target_connection`;
- archi one-to-one `assignment` fra modulo fisico e slot target;
- stato di esecuzione dell'expert.

Relazioni diverse fra la stessa coppia non vengono più fuse. Il tree
cinematico usato internamente dal planner SMORES è una proiezione validata del
sottografo target, non il formato del dataset. Il logger supporta lo schema di
transizione `mssr.expert_transition.v3` con `graph_t`, target, task graph,
assignment e `graph_t_plus_1`.

I ruoli non sono un singolo indice. Possono descrivere `support`, `base`,
`joint` con numero di DoF, `elbow`, `wrist`, `link`, `locomotor` o
`end_effector`. Gli esperti deterministici forniranno inizialmente le label;
IL/MARL potrà poi predire ruolo, morfologia e comportamento.

## 10. Sensori e percezione

La prima fase usa stato centralizzato perfetto del simulatore, equivalente a
VICON:

- pose e twist di tutti i moduli;
- posizioni e velocità dei giunti;
- facce libere, in contatto, allineate o collegate;
- topologia completa;
- stato del task e dell’ambiente quando disponibile.

Questo non implica che ogni SMORES disponga di camera o lidar. Il modello
hardware rivendica encoder/posizione dei giunti e localizzazione esterna. Non
vengono rivendicati visione onboard, lidar o percezione autonoma della scena.
Rumore, latenza, dropout e osservabilità parziale dovranno essere introdotti
come dimensioni del curriculum.

## 11. FreeBOT e codice a sfere

Il codice Python sferico deve essere riusato e isolato, non eliminato. Sono
riutilizzabili come core:

- bridge JSON/ROS;
- grafo attribuito e builder;
- registry e interfacce degli expert;
- curriculum;
- dataset logger;
- ordinamento deterministico.

Restano specifici FreeBOT:

- geometria sferica e raggio;
- locomozione sulla sfera;
- coordinate continue di contatto;
- magneti e pivot;
- `roll_to`, `dock_to_surface`, `attach_as_pivot`,
  `rotate_around_attached`, `climb_on`;
- gli expert già costruiti intorno a tali primitive.

Prima della migrazione definitiva FreeBOT deve superare:

1. docking fisico ripetibile fra due moduli;
2. undock senza compenetrazioni o pose correction;
3. crawling di un modulo rispetto all’altro;
4. pubblicazione nello stesso grafo canonico;
5. consumo dello stesso envelope di goal/status;
6. test di compatibilità dei dataset.

Solo dopo questi test si potranno eliminare `graphs/robot_graph.py`, le parti
sfera-first di `robots/module_state.py`, gli action schema duplicati e i
publisher JSON duplicati.

## 12. Verifica eseguita

- compilazione statica di tutti i file Python SMORES: passata;
- validazione sintattica dei tre manifesti JSON: passata;
- 41 test logici SMORES: passati;
- test del GraphBuilder condiviso in `mssr_ws`: passato;
- smoke test Isaac due moduli, 120 step: passato;
- smoke test Isaac sei moduli con quattro joint preconnessi, 120 step:
  passato;
- grafo risultante: 6 nodi, 4 attuatori e 4 connettori per nodo, endpoint
  `TOP↔BOTTOM` presenti sugli archi.

Isaac è stato eseguito in fallback CPU perché l’ambiente headless non esponeva
il driver NVIDIA. Le segnalazioni NVML/RTX non hanno impedito l’esecuzione
PhysX e non indicano un errore del backend.

## 13. Limiti noti e gate successivi

SMORES può ora essere collegato agli algoritmi deterministici senza modificare
la sua fisica. Restano verifiche sperimentali, non redesign del backend:

- campagne più lunghe con goal concorrenti durante docking e payload;
- calibrazione di rumore e latenza;
- eventuale modello di rottura del magnete;
- validazione della catena target da 7 moduli;
- sostituzione futura del transport action-like con un ROS Action server
  nativo, mantenendo invariato il contratto.

Il progetto complessivo non è invece pronto per affermare generalizzazione fra
famiglie finché FreeBOT docking/crawling e compatibilità canonica non sono
conclusi.
