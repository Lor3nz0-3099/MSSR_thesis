# BC-v1 — specifica architetturale v1.0.0

Data: 2026-10-06. Stato: design approvato in conversazione, formalizzazione da revisionare prima dell'implementazione. Branch di riferimento: `wip/t8-c05-human-teleop-20260921`. Nessun commit autorizzato prima della revisione di specifica e piano.

Revisione documentale proposta: 2026-10-07, ancora draft v1.0.0. GRAVITY_SETTLE esplicitamente enabled di default; iperparametri baseline e mini-tuning prima del training finale; audit manuale dei 22 ALIGN con gate prima di A1. Nessun codice, filtro o conteggio congelato modificato. Evidenze: [audit manuale completo](../../bc_v1_align22_manual_audit_20261007.md).

Questa specifica congela la prima baseline BC-GNN. Cambiamenti di whitelist, filtri, frame, split o semantica delle label richiedono una nuova versione e invalidano cache/normalizzatori incompatibili. Il piano associato è [2026-10-06-bc-v1-implementation-v1.0.0.md](../plans/2026-10-06-bc-v1-implementation-v1.0.0.md).

## 1. Obiettivo e confini

Apprendere una policy feed-forward `(G_t, o_t, c_t) -> a_t` da dimostrazioni esistenti. Il core deve essere riutilizzabile per robot modulari diversi; SMORES-EP è il primo adapter. FreeBOT deve essere rappresentabile, senza implementarne oggi il backend completo.

Vincoli normativi:

- Canonical v5 e dimostrazioni sono sorgenti immutabili.
- Nessuna GRU, LSTM, memoria temporale o Transformer.
- Nessuna azione WAIT e nessuna head dispatch/no-dispatch.
- Il core si importa senza Isaac Sim, ROS2, Isaac Lab o package SMORES; il contratto si importa anche senza NumPy/PyTorch.
- Nessun numero fisso di moduli/connector, nessuna dipendenza PAN/TILT nel core.
- Nessun control_state storico, target graph, target morphology, futuro, provenance o supervision mask negli input della rete.
- Runtime/executor, recorder, Canonical v5 e installazione Isaac Lab non vengono modificati da questa implementazione.
- Split per episodio; normalizzazione appresa solo sul train.
- Le risorse omesse sono non supervisionate: mai target zero sintetici.

La baseline è centralizzata e usa lo stato disponibile nel simulatore. Non viene presentata come percezione onboard o come MARL già operativo.

## 2. Snapshot del dataset e unità di conteggio

Sorgente: `datasets/expert_v1_canonical_v5/manifest.json`, schema `mssr.canonical_dataset_collection.v5`, generato il `2026-10-06T13:15:15.131378+00:00`.

SHA-256 del manifest verificato durante la preparazione:

```text
0a6a63dccebeb300b2fe30ead583980288eaafa604b4823ce7c1a290138d9b88
```

I SHA-256 dei singoli episodi sono quelli del manifest. Verificare manifest e file prima della costruzione della cache; non rigenerare il Canonical usando il codice corrente. Non utilizzare `datasets/expert_v1_canonical`, le directory temporanee o copie precedenti come sostituti silenziosi.

| Misura | Prima del filtro | BC-v1 |
|---|---:|---:|
| Episodi nella sorgente | 77 | 77, sorgente invariata |
| Episodi expert / teleop nella sorgente | 56 / 21 | 56 / 21 |
| Transition Canonical | 190767 | 190767, sorgente invariata |
| Episodi con supervisione BC | 70 | 70: 49 expert + 21 teleop |
| Timestep BC | 119543 | 119521 |
| Label wheels, per modulo/risorsa | 576942 | 576942 |
| Label pan_velocity | 186484 | 186484 |
| Label pan_target | 611 | 611 |
| Label tilt_target | 84736 | 84736 |
| Label ALIGN_FACES | 482 | 460 |
| Label DOCK | 456 | 456 |
| Label UNDOCK | 106 | 106 |
| Label GRAVITY_SETTLE | 24 | 24 |

Le label non sono timestep né componenti scalari: una label wheels contiene due valori e un timestep può contenere molte label. I conteggi sopra riguardano risorse con mask di supervisione vero, non tutte le azioni serializzate nel manifest Canonical.

Partire esattamente da `supervision.valid_for_behavior_cloning is True`; conservare le singole supervisioni autorizzate dai `module_resource_masks`. I sette expert button con BC=0 rimangono esclusi. Non espandere `source_repeat_coverage` in nuove righe, non usare il coverage come peso loss e non reinterpretare una label ripetuta come un nuovo dispatch strutturale.

Gli otto tilt_target indicati come busy nell'audit precedente restano inclusi: il control_state storico non è una fonte di feasibility e non viene utilizzato per filtrare o costruire input.

## 3. Filtro proposto delle 22 label ALIGN — revisione manuale prima di A1

Versione del filtro: `mssr.bc_v1.align_filter.v1`. La chiave è `(logical_episode_id, canonical_transition_index, initiator_module_id, action_family)`; la famiglia è `align_faces`. Il filtro futuro sarà in `configs/bc_v1/align_filter.v1.json` e includerà hash del manifest, hash degli episodi coinvolti e motivazione per voce.

Queste label sono il primo goal esplicito registrato per un lifecycle che compare già in una fase interna: `primitive_goal.parameters.execution_phase` è align, clocking o approach. La sola fase non basta a dimostrare che non siano nuove richieste. L'audit del 2026-10-07 verifica per tutte un riferimento precedente allo stesso execution/wave/action in `debug.active_goal_ids`, con evidenza REACH; propone 18 CERTAIN_CONTINUATION e 4 LIKELY_CONTINUATION (indici 651 in 1f8b…, 654 in 8b55…, 650 in b145…, 657 in beaa…). Nessun AMBIGUOUS o VALID_NEW_ALIGN identificato. I quattro LIKELY hanno un intervallo di 83–84 righe senza riferimenti alla singola azione e richiedono conferma manuale. La verifica usa i record sorgente esclusivamente come evidenza delle label, mai come feature.

**Gate obbligatorio:** prima di A1 l'utente deve revisionare l'[audit completo con tutte le sorgenti e cinque sequenze](../../bc_v1_align22_manual_audit_20261007.md). La tabella e i conteggi restano quelli congelati della proposta; nessun filtro è applicato o modificato in questa fase. Un'eventuale riclassificazione richiede revisione esplicita di lista e conteggi prima dello step C.

| Episodio | Indice Canonical, zero-based | Modulo iniziatore | Fase iniziale |
|---|---:|---|---|
| teleop-170338d489a64eb397a10cf9910392a1 | 135 | smores_08 | align |
| teleop-170338d489a64eb397a10cf9910392a1 | 1767 | smores_06 | clocking |
| teleop-1f8b406b56f74377834cd863ebccf696 | 651 | smores_02 | align |
| teleop-1f8b406b56f74377834cd863ebccf696 | 1391 | smores_04 | clocking |
| teleop-1f8b406b56f74377834cd863ebccf696 | 1753 | smores_06 | clocking |
| teleop-1f8b406b56f74377834cd863ebccf696 | 2005 | smores_05 | clocking |
| teleop-8b55584534d24c86888b350e09c74e7b | 654 | smores_02 | align |
| teleop-8b55584534d24c86888b350e09c74e7b | 1398 | smores_04 | clocking |
| teleop-b1456ab292aa4a319c9d4bcc17e7e57f | 650 | smores_02 | align |
| teleop-b1456ab292aa4a319c9d4bcc17e7e57f | 2005 | smores_05 | clocking |
| teleop-beaa63ebe5f347c6abc58baff9d1394a | 657 | smores_02 | align |
| teleop-beaa63ebe5f347c6abc58baff9d1394a | 1408 | smores_04 | clocking |
| teleop-beaa63ebe5f347c6abc58baff9d1394a | 1580 | smores_06 | align |
| teleop-beaa63ebe5f347c6abc58baff9d1394a | 1972 | smores_05 | align |
| teleop-c78e0455357147369ad16bc1bf39a0c2 | 662 | smores_07 | align |
| teleop-c78e0455357147369ad16bc1bf39a0c2 | 1563 | smores_06 | align |
| teleop-c78e0455357147369ad16bc1bf39a0c2 | 2008 | smores_05 | clocking |
| teleop-dd6bf8b9926d4a2aab6af7572646d026 | 655 | smores_07 | align |
| teleop-dd6bf8b9926d4a2aab6af7572646d026 | 1182 | smores_04 | align |
| teleop-dd6bf8b9926d4a2aab6af7572646d026 | 2005 | smores_05 | clocking |
| teleop-edbefc66cecf40c180864b452aff8dab | 345 | smores_08 | approach |
| teleop-edbefc66cecf40c180864b452aff8dab | 1574 | smores_06 | align |

Totale: 12 align, 9 clocking, 1 approach. Durante la preparazione è stato verificato che ognuna di queste righe ha una sola risorsa supervisionata. Il filtro rimuove soltanto la label; la riga esce dalla vista BC perché resta senza supervisioni. Da qui `119543 - 22 = 119521`, senza eliminare episodi. Un test sintetico deve provare che una label wheels presente sulla stessa riga viene invece conservata.

L'applicazione deve essere idempotente e non mutare la transition sorgente. Qualsiasi voce non trovata, duplicata, con action diversa o fingerprint incompatibile provoca errore; nessuna euristica estesa ad altri episodi. Le label di fase reach che rappresentano un nuovo intent restano valide.

## 4. Contratto generico e confini dei dati

Il contratto pubblico usa dataclass frozen, tuple e mapping validati; nessun import hardware o tensore obbligatorio. Le classi principali sono:

| Tipo | Responsabilità |
|---|---|
| `RobotSpec` | Versione, tipi di modulo, geometria, coordinate degli attuatori, connector, risorse, capability e catalogo azioni |
| `TaskContext` | Task/stage semantici correnti, parametri ammessi e relativi validity flag |
| `StateGraph` | ModuleState, ConnectorState, ResourceState e ConnectionState correnti |
| `PolicyInputs` | Solo `graph`, `observation`, `context` |
| `SupervisionBundle` | Label tipizzate, mask loss e motivi di esclusione |
| `SampleAudit` | Episodio, indice, hash, provenance, tag di stratificazione; mai input del modello |
| `PolicySample` | `inputs`, `supervision`, `audit` in contenitori separati |
| `RobotAdapter` | Protocollo per encode_state, encode_labels e decode_action |

`RobotSpec` descrive tipi e capability, non impone un inventario di otto istanze. La cardinalità delle istanze viene dallo stato corrente. Ogni connector e risorsa ha un riferimento al modulo proprietario; una connessione ha endpoint e proprietà fisiche. Gli ID risolvono riferimenti e trasporto; non sono embedding o numeri osservati dalla GNN.

Un connector ha kind `fixed_port` o `surface`, geometria locale e descrittore del dominio. Un sito sulla superficie può essere un candidato indicizzato generato dalla geometria corrente oppure essere specificato da argomenti continui. Nella prima baseline si implementano i fixed_port SMORES; una fixture sintetica verifica che il contratto accetti surface senza quattro facce o fixed joint. Non si implementano ora campionamento della superficie, primitive FreeBOT o dinamica magnetica.

Ogni resource dichiara coordinate osservabili, modalità comandabili, unità, limiti, dimensioni del payload e codec. Le coordinate di attuatori condivisi vengono raggruppate sulla stessa risorsa. Le modalità sono un catalogo configurato, non branch PAN/TILT del trainer.

Argomenti action consentiti: scalar/vector continuo, categoria, riferimento a module/connector e collezione di riferimenti quando richiesta dal catalogo. Una relazione connect/disconnect non implica rigidità. Parametri specifici non apprendibili, quali tolleranze dell'executor, restano nell'adapter/config.

## 5. SMORESAdapter e join delle evidenze

Configurazione nuova `configs/bc_v1/smores_ep.robot.json`, derivata dal manifest `configs/backends/smores_ep_enhanced.json` e dalle convenzioni fisiche esistenti, senza modificarli. Contiene EP-Face, geometria, ruote, PAN/TILT, esclusività della risorsa interna, limiti e clocking. Nessun import del runtime per leggere questi valori.

Mapping label:

- wheels -> risorsa locomotion con payload di due velocità in rad/s;
- pan_velocity / pan_target / tilt_target -> tre modalità della risorsa interna condivisa;
- ALIGN_FACES -> prepare_connection; DOCK -> connect; UNDOCK -> disconnect;
- GRAVITY_SETTLE -> skill supervisionata di recupero della postura neutra, con scope partecipanti dichiarato nell'adapter, **enabled=true di default**.

Il contratto generico rappresenta questa skill tramite ActionSpec configurata, riferimenti ai partecipanti e parametri tipizzati, anche senza connector. Il core non contiene un branch sul nome GRAVITY_SETTLE. L'adapter preserva i parametri effettivi Canonical (durata, passive_module_ids, clear_passive_policy_on_finish), dichiarando quali sono target apprendibili e quali vengono dal config. L'obiettivo è il ritorno alla configurazione neutra/zero tramite la primitive esistente: nel runtime attuale essa rilascia i drive interni e fa assestare sotto gravità; non si inventano nuovi target numerici zero o una garanzia di servo a zero.

Le 24 label rimangono trainabili, rappresentabili nei candidati, incluse nelle loss/metriche e nel pool rare. Nessuna soglia di frequenza può disabilitare automaticamente l'azione. `enabled` è configurabile mediante ablation esplicita, registrata e distinta dalla baseline; non è derivato dai supporti del dataset o dello split. Supporto zero in un minibatch/split dà loss zero/metrica null, senza disabilitare la skill negli altri split. Un train split privo di supporto per una skill enabled deve produrre una diagnostica e un gate di revisione prima del training, senza cambiare silenziosamente split o catalogo.

Nomi SMORES possono apparire nell'adapter, nel suo config e nei report; non nel core generico. I codec strutturali mantengono la semantica delle label Canonical e non espongono REACH/ALIGN/CLOCKING/APPROACH come azioni distinte.

Il joint action Canonical non conserva sempre il peer dentro la label strutturale del modulo iniziatore. Usare prima le partecipazioni `action_t.modules[*].participates_in` e gli argomenti Canonical, quando identificano univocamente i partecipanti. Se necessario, l'adapter può recuperare `module_ids` e argomenti della decisione dai record puntati da `provenance.source_records`, tramite sorgenti esistenti e verificabili. Questo join è riservato alla costruzione del target, separato dalla costruzione degli input. Vietato ricavare il peer dal target graph, dal futuro, dall'esito della primitive o dal control_state. Una sorgente raw mancante non rende incompleta una label già ricostruibile interamente dalla joint action.

Per label multiple, selezionare l'evidenza che concorda con modulo iniziatore, primitive e parametri Canonical; deduplicare equivalenti. Se restano evidenze non equivalenti o non è recuperabile un argomento obbligatorio, fermare l'audit prima del training. Non eliminare silenziosamente label per raggiungere i conteggi approvati.

## 6. Whitelist G_t, o_t, c_t

### G_t: fisica e topologia corrente

- Module: posizione, orientamento XYZW convertito in rotazione 6D, velocità lineare e angolare correnti, validity per quantità.
- Resource/coordinate: posizioni e velocità osservate, rappresentazione sin/cos per coordinate periodiche, capability, assi, unità e limiti statici dal RobotSpec.
- Connector: geometria corrente rispetto al corpo del modulo, tipo, stato fisico di occupazione/contatto e validity. Le connessioni correnti sono autorevoli per l'occupazione.
- Edge: connessione o contatto corrente, endpoint, trasformazione relativa, proprietà fisiche supportate dal contratto; clocking SMORES attraverso il suo codec.

Esclusi role/root ottenuti da template, target roles, assignment, morphology/template embedding, USD paths, ID embedding e tutti i campi executor/control_state storici. Non fare pass-through di dizionari raw. Le feature statiche hardware sono quelle dichiarate dal RobotSpec, non un embedding del nome del backend.

### o_t: osservazioni dinamiche del task

Geometria relativa degli oggetti fisici del task corrente: bordi gap, gradini, coni/ostacoli, bottone, eventuale goal corrente esplicitamente disponibile online e stato fisico di depressione del bottone. Oggetti con cardinalità variabile sono token, con indici di batch e validity; nessun numero fisso di coni/gradini.

Non usare una route nominale o un prossimo waypoint per inventare un goal osservato. Nessun dato futuro necessario alla trasformazione.

### c_t: task ontology indipendente dall'hardware

Task vocab ordinato: UNKNOWN, navigation, gap, stairs, button. Stage vocab ordinato: UNKNOWN, assembly, reconfiguration, task_execution. Entrambi hanno embedding apprendibile di dimensione 8. Alias raw sono normalizzati tramite `configs/bc_v1/task_context.json`, non tramite action_t, goal_id o fsm DISPATCHING_*.

Parametri ammessi: dimensioni del gap, geometria dei gradini, dimensioni/posizione world del bottone e degli ostacoli/coni, tolleranze del task documentate come disponibili online. Scalari intrinseci statici alimentano c_t; pose world degli oggetti alimentano o_t dopo trasformazione. Non duplicare una posa come feature statica e relativa.

Fonti: metadati raw in logs quando esistono; environment_geometry e metadati compact che provano i valori costanti; task/stage correnti nei record. Ogni recupero registra fonte e validità solo nell'audit. Non imputare da record futuri, non inferire task/stage dall'azione, non usare episode_id/task_id come embedding. Task_id serve soltanto al lookup del task corrente esplicitamente identificato.

I parametri world statici restano costanti per il loro stage; task/stage e vista del contesto cambiano quando cambia il task/stage corrente. Le coordinate relative cambiano con lo stato fisico. In assenza di un identificatore corrente usare UNKNOWN; solo parametri con scope corrente certificabile sono visibili. Nessun forward/backward fill tra stage per indovinare dati mancanti.

Per RC, posizioni coni fisici solo da `physical_track.cone_centers_xy_m`, `environment_geometry.layout.cone_centers_xy_m` o parametri del task corrente, con frame esplicito. Esclusi `navigation.waypoints_xyyaw`, nominal_route, centerline e altre traiettorie pianificate. Un test deve modificare i waypoint lasciando invariati gli input dei coni.

## 7. Frame fisico e feature extraction

Origine: centroide aritmetico di tutti i moduli correntemente osservati. Assi: frame di localizzazione/world o stage disponibile online, con convenzione di gravità esplicita e costante per quello stage. La scelta è dichiarata nel feature schema e non cambia per ordine dei nodi, nomi dei moduli o matching di morfologie.

Posizioni in input sono p_i-centroide espresse negli assi fisici scelti. Orientamenti, twist e oggetti osservati sono espressi coerentemente negli stessi assi. Connector geometry è body-local; edge pose è relativa tra i due corpi. Non sottrarre la velocità del centroide dai twist se lo schema non lo dichiara: v1 usa twist fisici espressi negli assi del frame.

Usare la stessa convenzione in stati stabili e transienti. Nessun fallback min-ID, root derivata da target template o PCA. La rete è equivarante alle permutazioni delle entità, non è automaticamente SE(3)-equivariante.

Ricalcolare geometria relativa da metadati world ammessi e pose correnti. Una geometria Canonical root-relative può essere invertita solo con frame e posa origine certificati; altrimenti è missing. Non riempire una geometria transiente mediante una root scelta dalla futura morfologia.

`features.json` congela ordine, unità, codec, dimensione e presenza di ogni campo. La parte cinematica module base ha 15 valori (p3, R6, v3, w3) e 15 validity: 30 colonne. Stato coordinate, connector e capability sono token separati; le loro larghezze derivano dal feature schema versionato, mai da N o dal numero di connector.

## 8. Batch e BC-GNN

Collate per concatenazione: M moduli, C connector, R risorse, A coordinate attuatore, E edge, K token observation, Q candidati strutturali nel batch.

| Tensore | Shape |
|---|---|
| module_x / module_batch | [M,d_m] / [M] |
| connector_x / connector_to_module | [C,d_c] / [C] |
| resource_x / resource_to_module | [R,d_r] / [R] |
| coordinate_x / coordinate_to_resource | [A,d_a] / [A] |
| edge_index / edge_x / edge_connector_index | [2,E] / [E,d_e] / [2,E] |
| observation_x / observation_batch | [K,d_o] / [K] |
| context_x | [B,d_ctx] |
| structural_candidates / candidate_group | [Q,...] / [Q] |

Gli indici sono int64, le feature float32 e i mask bool. Endpoint assente usa sentinel -1 e validità, mai indicizzazione accidentale dell'ultima entità. Padding di payload/categorie è al massimo del batch/catalogo, con validità esplicita. Empty connector/resource/edge/observation set è legale; un campione BC richiede almeno un modulo e una label.

Candidate group identifica la decisione strutturale del modulo iniziatore. I candidati contengono operazione, connector proprio, peer module/connector e argomenti previsti; scope module-only/multi-module è gestito dal codec dell'operazione. Non restringere candidati offline usando esito, label o control_state. Il controllo statico di tipi/riferimenti non è il feasibility mask runtime. Ogni target ammesso deve essere rappresentabile.

Encoder: task/stage embedding 8+8; context e observation encoder output 64 ciascuno; encoder coordinate/module/resource/connector H=128; edge encoder 64. Aggregare coordinate sulla risorsa e token sul modulo con operazioni invarianti all'ordine. Condizionare i nodi con context/observation prima del message passing e nuovamente il decoder. Tre blocchi message passing con residual e LayerNorm, H=128; pooling mean+max per grafo proiettato a 128. Decoder condivisi per modulo/risorsa e scorer dei candidati. Dropout configurabile dopo l'attivazione nascosta degli MLP encoder/message/decoder; baseline p=0, p=0.1 nel tuning. Nessun dropout su indici, target, validity o output finale; eval lo disattiva.

Implementazione minima PyTorch con index_add/scatter e MLP, senza dipendenza PyG obbligatoria. Numeri di moduli, connector, risorse o candidati non sono parametri strutturali della rete. Eventuali ordinali dei connector servono all'indicizzazione, non come identità fisica appresa.

## 9. Decision point, action e loss contract

| Famiglia | Supervisione offline | Valutazione durante rollout, rinviata |
|---|---|---|
| Continuous locomotion | Ogni riga con relativo mask vero | Tick di controllo della risorsa libera |
| Continuous internal | Ogni riga con relativo mask vero | Tick di controllo della risorsa libera |
| Internal position target | Righe target valide, anche setpoint consecutivi mantenuti | Nuovo dispatch discreto solo al decision point; hold interno all'executor |
| Structural skill | Solo label intent Canonical valide dopo filtro | Nuovo decision point, nessun nuovo dispatch della skill attiva |

I timestep consecutivi di un target mantenuto sono supervisione del setpoint, non prova di più inizi della primitive. L'assenza di una label non insegna quando fare dispatch. Decision point e feasibility vengono dal runtime/executor nel rollout; non si apprendono con WAIT/no-dispatch. Se non esistono candidati eseguibili il scheduler non invoca il dispatch. Il criterio operativo completo del scheduler è un gate successivo, non ricostruito dal control_state storico.

Loss per modalità/operazione configurata:

- regressione Huber, delta=1 nello spazio normalizzato, media delle dimensioni effettivamente supervisionate;
- CE della modalità interna, solo se la risorsa ha più modalità e una label valida;
- CE sugli score dei candidati strutturali per ogni iniziatore supervisionato: il candidato include operazione e riferimenti, senza duplicare la stessa supervisione in un secondo CE degli stessi endpoint;
- CE per parametri categorici separati, quali clocking; regressione Huber per parametri continui apprendibili.

Più candidati realmente equivalenti e simultaneamente validi nello stesso gruppo usano una distribuzione target normalizzata sui positivi; non dedurre equivalenze da successo futuro. Per una sola label si usa il target one-hot. Action fields non apprendibili dell'executor sono ricostruiti dall'adapter/config e non hanno loss.

Ogni termine è diviso per il suo numero di supervisioni effettive, non per B o per tutti i moduli. Pesi iniziali dei termini presenti = 1, configurabili e salvati. Nessuna divisione per zero; una head senza target produce contributo zero. Wheels e internal supervisionati insieme contribuiscono entrambi. Nessuna penalità su dimensioni mancanti, risorse non supervisionate, categorie di padding o peer non partecipanti.

Metriche: MAE/RMSE in unità fisiche e per modalità; accuracy modalità; accuracy operazione e selezione endpoint; exact-match action strutturale e clocking; conteggi supporto. Riportare micro e macro per task/provenance/morfologia come metadati di valutazione. Una head senza supporto ha metrica null, non zero. Success rate del task e rejection rate appartengono ai rollout, non alla validation BC offline.

## 10. Split, normalizzazione e sampler

Seed iniziale: 20261006. Manifest split per 70 episodi: 48 train, 11 validation, 11 test.

| Strato episodio | Totale | Train | Validation | Test |
|---|---:|---:|---:|---:|
| Expert gap | 18 | 12 | 3 | 3 |
| Expert stairs | 18 | 12 | 3 | 3 |
| Expert navigation | 13 | 9 | 2 | 2 |
| Teleop button | 8 | 6 | 1 | 1 |
| Teleop mission composite | 13 | 9 | 2 | 2 |
| Totale | 70 | 48 | 11 | 11 |

Quote e definizione degli strati sono nel config dell'esperimento, non branch task-specifici del core split. Ordinare gli episode_id dentro ogni strato e usare RNG locale seedato. Tenere insieme lineage/sorgenti della stessa dimostrazione. Verificare collisioni degli hash e lineage prima della selezione. Se i vincoli di grouping rendono impossibili le quote, fermare la preparazione: non fare split per riga né rilassare silenziosamente le quote. Tag morfologia/task/provenance sono metadati di stratificazione; non diventano input.

Normalizzatori per campo/mode/unità: scaler deterministico da limiti fisici dichiarati per payload bounded; media/deviazione standard sul solo train per gli altri scalari, ignorando missing e padding. Deviazione nulla -> scala 1; categoria, rotazione 6D, sin/cos e validity non vengono standardizzati. Non usare min/max globali del file. Campi senza supporto train mantengono offset 0/scala 1 e un flag diagnostico. Salvare normalizzatori e hash degli episodi train.

Sampler iniziale B=128: 96 draw dalla gerarchia cella (task corrente, morfologia corrente, provenance), poi episodio e timestep uniformi dentro la cella; 32 draw da famiglie rare configurate nell'adapter (pan_target, ALIGN, DOCK, UNDOCK, SETTLE), poi episodio e timestep. Le tre modalità interne non sono stringhe nel sampler: il config fornisce gli ID delle famiglie. Campioni estratti con replacement e RNG locale; non spezzare l'azione congiunta. Pool vuoto -> fallback sul ramo generale registrato nelle statistiche. Nessuna creazione di label, duplicazione della sorgente o weighting per source coverage.

Training usa questa distribuzione dichiarata; validation/test percorrono tutte le righe eleggibili una volta, senza sampler bilanciato e senza drop_last. Registrare frequenze di estrazione e supporto per famiglia. Un budget iniziale è ceil(n_train_rows/128) batch per epoch; la copertura dei draw non è presentata come una visita completa del dataset.

## 11. Cache, training e riproducibilità

Cache derivata sotto `artifacts/bc_v1/`, mai dentro la sorgente Canonical. Chiave: versioni spec/contract/features/filter, manifest e hash episodio, hash dei metadati e delle evidenze aggiuntive realmente lette, RobotSpec, config TaskContext, split e normalizzatori. Prima cache di campioni non normalizzati per episodio, già trasformati una volta nel frame fisico; normalizzazione applicata successivamente dal collate. Rifiutare una cache con fingerprint diverso, non aggiornarla in-place come se fosse la stessa versione.

Dataset e collate non importano l'adapter hardware. L'adapter lavora durante prepare; il trainer consuma cache e schema generico. La cache può essere PyTorch per episodio con manifest JSON; nessuna conversione HDF5 o installazione Lab.

Small-overfit: fixture e piccolo sottoinsieme train con tutte le head iniziali; nessuna lettura del test set. Obiettivo di verifica: loss finite, gradiente per ogni head supervisionata e riduzione della loss deterministica; non certifica generalizzazione.

**Baseline iniziale, non iperparametri definitivi:** AdamW lr=1e-3, weight_decay=1e-4, dropout=0, batch_size=128 e general/rare=96/32; massimo 100 epoch, patience=15 su validation loss, gradient clip=1.0, CPU/GPU configurabile, float32 senza AMP nella prima verifica. Best e last checkpoint distinti. Metriche per head per evitare che un miglioramento aggregato nasconda regressioni rare. I valori finali di lr/weight_decay/dropout vengono scelti dal mini-tuning seguente; i budget di tuning e training finale sono distinti.

### 11.1 Mini hyperparameter tuning dopo small-overfit e prima del training finale

Grid riproducibile di **12 trial**: learning_rate ∈ {3e-4, 1e-3, 3e-3}, weight_decay ∈ {0, 1e-4}, dropout ∈ {0, 0.1}. La baseline è inclusa. Tenere batch_size=128, general/rare=96/32, H=128, tre blocchi e loss weights=1 fissi; batch size e rapporto sampler non ampliano la ricerca iniziale. Nessun framework HPO/Isaac Lab necessario.

Usare soltanto i 48 episodi train e gli 11 validation, stessi split/normalizzatori e seed=20261006 per tutti i trial; ogni trial ricrea modello, optimizer e RNG/sampler da zero. Ordine della grid ordinato e trial_id deterministici. Non caricare né valutare i 11 episodi test durante il tuning. Nessuna scelta di config, budget o loss weights in base alle metriche test.

Budget iniziale comune: massimo 10 epoch, massimo min(200, ceil(n_train_rows/128)) update per epoch, clip=1, float32, validation completa e non bilanciata a fine epoch. Niente early stopping per trial nella mini-grid, così ogni trial ha lo stesso budget di update. Totale massimo 24000 update e 120 passaggi validation. Prima della grid misurare un epoch train+validation con la baseline sullo stesso budget: se la stima supera una giornata GPU disponibile, ridurre uniformemente il budget di tutti i trial **prima** di avviarli e salvare la revisione; nessuna promessa di durata senza misura hardware. Trial sequenziali su una sola GPU, nessun ampliamento automatico dello spazio di ricerca; il cap temporale serve alla scadenza della tesi del 15 dicembre. Un trial fallito conserva stato e motivo, non ottiene un punteggio fittizio. Una grid incompleta deve dichiararlo e richiede revisione prima di una selezione finale.

Score primario = loss validation aggregata per supporti globali e famiglie come §9, mai media non pesata delle loss dei batch. Score del trial = minimo score validation entro il budget; parità entro 1e-6 risolta mediante trial_id ordinato. Salvare le metriche di tutte le head, inclusa GRAVITY_SETTLE, con supporti; non modificare a posteriori lo score perché una configurazione è preferita. Il budget corto seleziona una buona baseline, non dimostra un optimum.

Salvare `search_manifest.json` (grid, budget, seed, split/normalization/data/code fingerprints), `trials/<trial_id>/{config.json,environment.json,metrics.jsonl,best.pt,summary.json}` e `selection.json`. Registrare ogni epoch, metriche per head/supporti, durata, status e errori; mantenere anche i trial non vincenti. A parità di fingerprint si può riprendere la ricerca saltando trial già completi; un trial interrotto riparte dal suo seed, senza richiedere ancora resume a metà trial. Determinismo verificato su CPU; su GPU registrare device/versioni/impostazioni e non promettere identità bitwise tra piattaforme diverse.

Dopo la selezione, il training finale riparte **da zero** sui soli train, usando la configurazione scelta, budget massimo 100 epoch e patience=15 su validation. Non unire train e validation e non proseguire dal checkpoint corto del trial vincente. Il test set è valutato soltanto mediante il comando finale esplicito, dopo selezione di configurazione e checkpoint.

Checkpoint: pesi, optimizer, scheduler se configurato, epoch/global_step, posizione nel sampler, RNG Python/NumPy/Torch/CUDA, config, feature/action schema, RobotSpec, split, filter hash, normalizzatori, versioni librerie e codice. Resume rifiuta contract/schema/data/split/normalizzazione incompatibili. Checkpoint a confini di batch; per resume deterministico v1 usa num_workers=0 e nessun prefetch non serializzato. Timestamp e durata non fanno parte dell'uguaglianza riproducibile.

## 12. Lavoro rinviato e criteri di accettazione

Prima dei rollout: snapshot atomico, decision scheduler, feasibility online causale prima dell'inferenza, controllo ownership/versione al dispatch, arbitraggio delle azioni concorrenti, executor lifecycle e test di skill che continua senza redispatch. Nessun cambiamento a questi componenti nella presente fase.

Future demo/DAgger: snapshot prima della decisione, decision_seq/event_seq indipendenti dal physical stamp, risultato di ammissione, lifecycle e post-state; reset/replay completo dello stato executor/topologia. Nessuna correzione del recorder attuale e nessuna nuova demo come requisito BC-v1.

Isaac Lab: possibile integrazione ibrida successiva, solo dopo prova isolata di compatibilità con Sim 6.0 RC. Il primo training non lo installa, importa o avvia. Recorder/HDF5, vectorized env, RL/MARL e tuning Lab non sono deliverable della prima baseline.

Prima di A1: revisione personale dell'audit dei 22 ALIGN e approvazione delle correzioni documentali. Prima dei primi update (small-overfit): import isolation, contratto e adapter, conteggi e filtro esatto approvato, whitelist causale, frame/permutazioni, batch variabili, candidati con target rappresentabili, GRAVITY_SETTLE enabled e con gradiente, loss concorrenti, split senza leakage, normalizzazione train-only e sampler integro. Prima del training finale: overfit minimo superato, mini-grid completata e selection/config/metriche salvate, test mai usato per tuning/selezione, checkpoint/resume verificati. Nessun test runtime o backend FreeBOT completo è necessario per questi gate.
