# BC-v1 — Source Frame Bridges v1.0.0

Data: 2026-10-12. Branch: `wip/t8-c05-human-teleop-20260921`.
Stato: specifica scritta per **review e approvazione esplicita dell'utente**; non autorizza implementazione o prepare. Task 1–3 della milestone causal-topology sono completi; Task 4 resta fermo.

## 1. Scopo, autorità e contesto congelato

Questa specifica formalizza come BC-v1 ottiene trasformazioni certificate dei frame sorgente di task/ambiente, prima che E1 esprima la geometria nel frame fisico comune della policy. Il core resta hardware-agnostic; le eccezioni storiche appartengono al provider della sorgente.

Si aggiunge alla [specifica coalesced supervision + causal topology](2026-10-11-bc-v1-coalesced-supervision-causal-topology-v1.0.0.md), senza modificarla. Il [piano causal-topology](../plans/2026-10-11-bc-v1-coalesced-supervision-causal-topology-implementation-v1.0.0.md) resta invariato; questo documento pone requisiti di integrazione per il futuro Task 4, senza riscriverlo. [Spec originale](2026-10-06-bc-v1-architecture-v1.0.0.md) e [piano originale](../plans/2026-10-06-bc-v1-implementation-v1.0.0.md) restano riferimenti storici, con la precedenza delle decisioni successive già dichiarata dalla spec coalesced.

Sorgente immutabile: `datasets/expert_v1_canonical_v6`. SHA256 del **manifest**, non hash cumulativo della directory:

```text
31765272b6ecd7506c54b25e214b75d7b1b28691803ddfc286dafcce8b4ec562
```

| Split F1 | Episodi utili | Sample BC-valid |
|---|---:|---:|
| Train | 49 | 78302 |
| Validation | 12 | 22762 |
| Test | 12 | 18536 |
| Totale | 73 | 119600 |

Eleggibilità, indici, coalescing, target e split MUST restare invariati. `PolicyInputs` / `mssr.bc_v1.policy_inputs.v1`, `FrameGeometryView` separata e whitelist D2 MUST restare invariati. Nessun stage/configuration_goal, WAIT o head no-dispatch viene introdotto.

## 2. Problema verificato

Non è stato trovato un double transform numerico. La geometria fisica valida è coerente. D1 recupera alcune quantità in frame nominati e D2 le conserva valide, ma [G1 corrente](../../../drl/IL/bc_v1/dataset.py) chiama `transform_inputs(inputs, PhysicalFrameConfig())`, con `transforms=()`.

E1 non trova il bridge e produce una quantità presente ma invalida; E2 mantiene una rappresentazione geometrica invalida e non può costruire la relazione fisica prevista. Il comportamento fail-closed evita una trasformazione inventata, ma senza diagnostica rende invisibile alla policy parte della geometria disponibile.

Il problema è **MISSING CERTIFIED SOURCE-FRAME BRIDGE WIRING**, non BAD NUMERICAL TRANSFORMATION né DOUBLE TRANSFORM. Risolverlo MUST NOT richiedere modifiche a Canonical o un'inversione delle geometrie root-relative legacy.

## 3. Evidenze e limite della certificazione

Il precedente audit di provenance ha seguito sette bucket reali BC-valid, con exact source path/row/index, attraverso RAW / **DATASET_COMPACT** → loader → preprocessing/CanonicalTimestepBuilder → Canonical v6 → SMORESAdapter → D1/D2 → E1/E2. Risultato: **5634 confronti indipendenti, zero failure, errore massimo 7.77e-16, tolleranza assoluta 1e-9**, quaternion confrontati modulo segno. Nessuna doppia trasformazione o leakage target/futuro/stage/FSM/label osservata nei percorsi rappresentativi.

Il successivo audit runtime ha eseguito il percorso corrente `encode_state → read_task_metadata → build_task_view → select_policy_inputs → transform_inputs → extract_features`. L'evidenza autorevole è `/tmp/mssr-bc-v1-seven-case-runtime-audit.txt`; supporto: `/tmp/mssr-bc-v1-representative-audit-packet.txt` e `/tmp/mssr-bc-v1-runtime-api-setup.txt`. Sono artifact storici di audit, MUST NOT diventare dipendenze del prepare/trainer.

| Caso | Episode id | Index Canonical, 0-based | Relazioni default → probe |
|---|---|---:|---:|
| Gap expert | `expert:gap:seed-004100` | 386 | 16 → invariato |
| Stairs expert | `expert:stairs:seed-003100` | 555 | 32 → invariato |
| RC expert legacy | `expert:rc_car:seed-005100` | 476 | 8 → 24 |
| Navigation teleop | `teleop-0e4e34fe3a5f4ed1b6a86642acf4fac8` | 9058 | 32 → invariato |
| Button manipulation | `teleop-170338d489a64eb397a10cf9910392a1` | 3023 | 32 → invariato |
| Button-mission initial assembly | `teleop-170338d489a64eb397a10cf9910392a1` | 0 | 32 → invariato |
| Reconfiguration / stairs | `teleop-0e4e34fe3a5f4ed1b6a86642acf4fac8` | 1725 | 32 → 32 |

Verdetto runtime esatto: **NO_NEW_UNEXPECTED_PIPELINE_LOSS_DETECTED**. Le invalidazioni note map/stage_local non sono state conteggiate come nuove anomalie; questo verdetto MUST NOT essere interpretato come bridge già integrati o readiness del training.

**THIS IS NOT AN EXHAUSTIVE 119600-SAMPLE CENSUS.** La prova copre i principali percorsi semantici attuali, non ogni possibile record malformato della sorgente.

## 4. Contratto semantico dei frame

| Semantica | Regola normativa |
|---|---|
| `world` | Frame fisico dichiarato della sorgente corrente; identità verso lo stesso world. |
| `stage_local` | Frame geometrico del task corrente; richiede origine e orientamento causali certificati. Non è un stage embedding. |
| `map` legacy RC expert | Identità ammessa esclusivamente sotto il contratto storico SMORES/Isaac/Nav2 certificato in §9. |
| Connector module-body | Geometria locale al modulo proprietario, già prodotta dall'adapter; nessun bridge globale. |
| Edge B-in-A | Posa di B nel corpo di A; già relativa, MUST essere preservata. |
| Intrinsic/resource-local | Dimensioni, misure intrinseche e coordinate di risorsa; nessuna traslazione globale. |
| Unknown/unsupported | Nessuna coercizione; quantità invalida con ragione ispezionabile. |

Il nome testuale di un frame MUST NOT identificare da solo il suo contratto fisico: `map` di due sorgenti/backend può avere semantiche diverse. Valori, unità, tipo geometrico, frame e scope corrente MUST rimanere espliciti. I tre scope `g_current`, `c_static`, `o_dynamic` restano separati.

## 5. Astrazione minima SourceFrameBridge

Un bridge provider MUST produrre un risultato deterministico e immutabile che identifichi:

- frame sorgente e frame fisico/world destinatario, nello scope della sorgente corrente;
- trasformazione rigida diretta: `p_world = R_world_from_source * p_source + t_world_from_source`, con rotazione propria e valori finiti;
- validità/certificazione oppure motivo di mancata risoluzione;
- contratto sorgente, revisione del provider/policy e riferimenti/digest delle evidenze realmente utilizzate;
- scope di applicabilità corrente e configurazione deterministica necessaria al fingerprint della cache.

Il risultato MUST distinguere una trasformazione certificata da un'ipotesi diagnostica. Identità provider, provenance, path e digest sono routing/audit; MUST NOT entrare nei learned input. Certificazione non significa soltanto che una matrice è numericamente valida: deve provare anche origine, assi, unità e applicabilità alla sorgente.

E1 dispone già di `SourceFrameTransform` e `PhysicalFrameConfig.transforms` in [features.py](../../../drl/IL/bc_v1/features.py). Questo documento specifica il confine semantico e la certificazione; non impone nuove classi, firme, file o framework. Il bridge risolve il source→world; MUST NOT trasformare le quantità al posto di E1 o introdurre nuove feature.

## 6. Ownership dei provider

**E1 generico** MUST consumare trasformazioni certificate senza import Isaac Sim, ROS2, package SMORES o regole per un particolare episodio/morfologia. MUST restare l'autorità matematica di conversione.

**SMORES source/provider layer** MUST certificare la convenzione storica RC e leggere il frame fisico del task corrente. Il codec delle label e l'adapter dello stato robot MUST NOT decidere il bridge in base all'azione supervisionata.

**FreeBOT futuro** fornirà il proprio provider/config di sorgente. MUST poter riusare E1 senza ereditare la convenzione SMORES `map=world`. Non si implementa un backend FreeBOT in questo lavoro.

## 7. Regola world

`world → world` MUST essere identità quando i due nomi denotano lo stesso frame fisico dichiarato. È semantica generica, non un'eccezione SMORES. Una sorgente etichettata world ma priva di certificazione coerente degli assi/unità MUST NOT essere reinterpretata arbitrariamente.

E1 applica poi il frame fisico della policy: origine nel centroide dei moduli correntemente validi, assi fisici dichiarati. MUST NOT scegliere root, modulo min-ID, PCA o frame derivato dal target. Module pose/orientation/twist e ambiente MUST usare lo stesso frame; i twist sono ruotati, senza sottrarre la velocità del centroide.

## 8. Regola stage_local e regressione numerica

Il bridge MUST derivare esclusivamente da `world.task_frame.origin_xyz_m` e `world.task_frame.yaw_rad` correnti, oppure dall'equivalente `stage_frame.origin_world_xyz_m/yaw_rad` della sorgente prima di D2. Il contratto attuale certifica un frame planare: traslazione world più rotazione yaw attorno all'asse verticale dichiarato. MUST NOT inferire roll/pitch mancanti per una sorgente che non dichiara questa convenzione.

Per il record reconfiguration /1725, i metadati correnti dichiarano:

```text
origin_world_xyz_m = [1.7699752221047802, -2.17605937116128, 0.0]
yaw_rad           = -1.5707963267948966
first_riser_x_m   = -0.7000000000000001
Rz(yaw) * [-0.7, 0, 0] + origin
                  = [1.7699752221047802, -1.4760593711612797, 0.0]
```

Il risultato coincide, entro 1e-9, con `stairs.first_riser_center_world_xyz_m` registrato indipendentemente. Il punto illustra il frame; la quantità x-only resta una geometria parziale/piano in E1, MUST NOT diventare un punto misurato completo.

I tre campi `stage_local.stairs.base_height_m`, `first_riser_x_m`, `top_heights_m` passano da D1/D2 validi a E1 invalidi senza bridge. L'origine/yaw corrente rende il bridge causalmente ricostruibile. Frame mancante, malformato, non finito, fuori scope o con fonti contraddittorie MUST fallire chiuso; nessuna identità di fallback o fill da task futuri. Il nome `stage_local` indica una relazione geometrica, MUST NOT autorizzare l'uso di stage/FSM come input o selettore del bridge.

## 9. Regola map legacy RC expert

Il provider MUST ammettere `map → world = identity` soltanto quando la provenance certifica il **contratto storico legacy SMORES/Isaac/Nav2 RC expert**. Il nome `map`, il nome della morfologia o l'episode id da soli MUST NOT attivare l'eccezione.

Evidenze del caso /476:

- `datasets/expert_v1_compact/episodes/rc_car/seed-005100/environment_geometry.json` dichiara `layout.frame_id=map`, `layout_frame_id=map` e bounds x≈[-1.3, 3.5], y=[-1.6, 1.6]; centro e dimensioni risultano [1.1, 0.0] e [4.8, 3.2].
- `behavior_manifest.json`, `observation_constants.course`, dichiara separatamente `frame_id=world`, piattaforma al centro [1.1, 0.0, -0.01], dimensioni [4.8, 3.2, 0.02].
- [smores_nav2.launch.py](../../../mssr_ws/src/mssr_expert/launch/smores_nav2.launch.py) dichiara odom in coordinate Isaac-world e pubblica map→odom identità. La ricostruzione storica approvata associa questa convenzione alla sorgente RC; la sola lettura del launch corrente MUST NOT certificare sorgenti arbitrarie.

La corrispondenza geometrica sostiene il certificato assieme alla convenzione runtime e alla provenance della sorgente; un rettangolo coincidente da solo non prova un TF. Il futuro provider MUST verificare l'appartenenza al contratto e le evidenze, senza una allowlist di indici Canonical o ricerca del candidato richiesto dalla label.

Le sette quantità interessate sono `map.cones.0..5.center_xy_m` e `map.environment.platform_bounds_xy_m`. Nel record /476: D2 intrinsic/map/world=3/7/3; E2 default ha 16 righe ambiente, 6 valide e 10 invalide, con 8 relazioni collision_box. Il probe certificato ripristina **16/16 righe valide e 24 relazioni: collision_box=8, cone=16**. Il bridge MUST preservare le geometrie XY parziali, non inventare una quota dei coni.

L'eccezione MUST NOT propagarsi a FreeBOT, nuovi dati SMORES, Nav2 arbitrario, map di sorgenti non certificate o navigation teleop già WORLD. Il controllo teleop /9058 usa intrinsic/world=15/140 e produce 16 cone relations senza questo bridge.

## 10. Fail-closed e ragioni diagnostiche

Un bridge sconosciuto, unsupported, ambiguo o non certificato MUST lasciare invalida la quantità spaziale. MUST NOT rinominarla world, sostituire una trasformazione identità o mantenere `valid=true` su dati di default.

Ogni invalidazione MUST esporre una ragione audit/diagnostica, con campo, scope, source frame e provider/policy applicabile. Ragioni e certificati MUST stare fuori dagli input della policy. Contratti incompatibili o due trasformazioni discordanti per lo stesso frame/scope MUST essere riportati esplicitamente, senza scelta arbitraria.

## 11. E1 come unica autorità di trasformazione

D1/D2 MUST preservare valore, unità, frame semantico e validità delle quantità ammesse. MUST NOT pre-trasformare map/stage_local in world per aggirare E1; la sorgente dichiarata MUST rimanere tracciabile.

E1 MUST essere l'unica autorità che converte world/map/stage_local e futuri frame globali certificati nel frame fisico comune. Il provider fornisce la trasformazione, non una seconda vista delle feature. `FrameGeometryView` rimane una derivazione deterministica immutabile degli stessi PolicyInputs, senza widening della whitelist o schema learnable nuovo.

E2 MUST soltanto codificare quella geometria e calcolare relazioni nella stessa base; conversione XYZW→R6 e delta/distanza nel frame comune non sono nuove trasformazioni di frame sorgente.

## 12. Eccezioni locali e intrinseche

- Connector: Canonical mantiene valori WORLD sotto nomi generici; SMORESAdapter li converte WORLD→corpo del modulo proprietario. E1/E2 MUST preservare posizione/normale/tangente body-local.
- Edge: `relative_pose` è B-in-A, ricalcolata dal preprocessing dalle pose world correnti. E1 MUST preservarla; E2 può cambiare la rappresentazione dell'orientamento.
- Coordinate: posizione/velocità di attuatore e limiti del catalogo restano resource-local; nessun bridge globale.
- Misure intrinseche: dimensioni e scalari/vettori dichiarati intrinseci restano tali; `button.depression_m` non è un punto spaziale.

Questi canali MUST NOT essere instradati nel provider globale. Non esiste un fallback generico `body_local`: task/environment body-local senza proprietario/riferimento sufficiente MUST fallire chiuso. MUST NOT inferire il proprietario da ID minimo, target o azione.

## 13. Interazione con la canonicalizzazione E2

Una quantità resa valida dal bridge può essere equivalente a una rappresentazione WORLD più ricca già presente. E2 MUST conservare la physical-equivalence canonicalization attuale, separandola da SOURCE INFORMATION LOSS.

Per i coni teleop: `certified_3d_over_2d` richiede equivalenza fisica; geometria dimensionale per oggetto osservato ha precedenza sul default condiviso. Un alias senza riga E2 diretta può essere rappresentato dall'oggetto equivalente, con tracciabilità della canonicalizzazione. MUST NOT inventare componenti mancanti o creare duplicati per due nomi della stessa geometria.

Per stairs /1725: il bridge rende valide tre quantità, emesse prima del dedup come sei righe geometriche (base plane, riser plane, quattro top planes). Le rappresentazioni equivalenti WORLD/local vengono canonicalizzate: **50 righe ambiente, 44 valide/6 invalide → 45 righe, tutte valide**, con 32 relazioni conservate. Il conteggio può diminuire quando migliora la validità: MUST essere spiegato dalla dedup fisica, non interpretato automaticamente come perdita.

## 14. Diagnostica delle invalidazioni lungo la pipeline

Il futuro prepare G1 MUST confrontare presenza, validità e rappresentazione D1→D2→E1→E2, classificando almeno:

| Ragione | Significato |
|---|---|
| `EXPECTED_SOURCE_MISSING` | La fonte corrente non contiene la misura. |
| `INTENTIONAL_POLICY_EXCLUSION` | Esclusione esplicita della whitelist. |
| `EXPECTED_TRANSIENT_OR_NOT_APPLICABLE` | Canale non applicabile allo stato corrente; zero connessioni reali non sono missingness. |
| `KNOWN_FRAME_BRIDGE_MISSING_MAP` | Quantità map presente/valida, certificato o wiring mancante. |
| `KNOWN_FRAME_BRIDGE_MISSING_STAGE_LOCAL` | Quantità local presente/valida, frame corrente o wiring mancante. |
| `EXPECTED_SCHEMA_VALIDITY` | Missingness intenzionale del catalogo/schema, distinta dalla misura. |
| `UNEXPECTED_D1_LOSS` | Dato ammesso disponibile nella fonte ma perso nel task view. |
| `UNEXPECTED_D2_LOSS` | Dato ammesso valido perso dalla selezione. |
| `UNEXPECTED_E1_INVALIDATION` | Quantità D2 valida resa invalida senza ragione approvata. |
| `UNEXPECTED_E2_FEATURE_INVALIDATION` | Geometria E1 valida non rappresentata, senza codec/canonicalizzazione approvati. |
| `UNEXPECTED_RELATION_LOSS` | Relazione fisica attesa assente dopo object extraction/dedup. |
| `FRAME_SEMANTICS_UNKNOWN` | Scope/origine/assi/unità non certificabili. |
| `OTHER_UNEXPLAINED` | Caso non spiegato; MUST bloccare l'accettazione. |

La canonicalizzazione E2 MUST avere una ragione esplicita di physical equivalence e un riferimento alla rappresentazione mantenuta; MUST NOT essere classificata come perdita perché manca una riga diretta. Una quantità spaziale D2 `valid=true` MUST NOT diventare E1 invalida senza spiegazione ispezionabile.

Unexpected loss/invalidation, relazione inattesa mancante o OTHER_UNEXPLAINED MUST produrre fail-fast prima della pubblicazione della cache completa. Una scansione diagnostica può raccogliere gli errori, ma MUST NOT promuoverli a cache certificata. Le due classi KNOWN descrivono il baseline senza bridge; MUST NOT giustificare l'omissione dei bridge richiesti dai contratti certificati. Missingness sorgente o schema approvata resta rappresentabile con validity esplicita.

## 15. Invarianti a livello di policy graph

Le verifiche MUST controllare oggetti/relazioni, non soltanto il bit di validità. I numeri seguenti sono regressioni dei record rappresentativi, MUST NOT diventare cardinalità globali hard-coded:

- RC /476: 8→24 relazioni, con le 8 collision_box invariate e 16 cone ripristinate; nessun cono fisico duplicato.
- Stairs /1725: sei righe diventano valide prima della canonicalizzazione, le rappresentazioni equivalenti sono deduplicate, restano 32 relazioni fisicamente corrette. La relazione riser può passare dalla rappresentazione anchor+normal alla plane equivalente.
- Controlli gap, stairs expert, navigation WORLD, button e assembly: geometria e relazioni valide MUST restare equivalenti; zero connessioni dell'assembly sono topologia osservata reale.

Un bridge MUST cambiare l'espressione spaziale e recuperare rappresentazioni disponibili, senza inventare/eliminare oggetti fisici o alterare la topologia corrente.

## 16. Invarianti sui valori

La validazione futura MUST rifiutare valori non finiti dichiarati validi, mismatch di frame coerziti, trasformazioni dimensionali unsupported, double transform e applicazioni del bridge a dati locali/intrinseci. SHOULD usare confronti geometrici indipendenti con tolleranza dichiarata (regressioni attuali: 1e-9).

Una quantità sorgente nonzero MUST NOT diventare zero/default ancora valido per un fallback o perdita di dati. Uno zero matematicamente corretto, per esempio un punto coincidente col centroide, è ammesso soltanto se coincide con il risultato indipendente della trasformazione. Missingness MUST restare distinta da una misura reale zero. Geometria parziale MUST conservare i vincoli osservati; XY senza quota non autorizza XYZ misurato.

## 17. Causalità e leakage

Il provider MUST NOT dipendere da target morphology, target graph, G_t+1, future topology/observation, action/candidate label, mask (incluse supervision e feasibility masks), reward, success/outcome dell'operazione da imitare, stage, FSM, control_state o scelte geometriche target-dependent. MUST NOT usare task success come certificazione della posizione.

Il frame fisico dichiarato nel task corrente è ammesso; il suo nome storico `stage_frame` non autorizza il consumo di stage/FSM. Identità backend/sorgente e provenance possono soltanto instradare/certificare il provider, MUST NOT diventare embedding o feature apprese. A input fisici correnti ed evidenze ammesse uguali, cambiare label, target, maschere o record futuri MUST lasciare invariati bridge/E1/E2.

## 18. Identità della cache G1

**Identità globale / compatibility della cache G1** MUST includere o dipendere da:

- identità e revisione del bridge provider/policy;
- configurazione deterministica dei bridge;
- identità del contratto sorgente certificato e policy delle evidenze;
- identità dataset/sorgente necessaria a riprodurre la risoluzione dei bridge;
- policy di certificazione e missingness.

**Contenuto / audit per sample** MUST conservare informazioni deterministiche sufficienti a verificare quale source frame è stato risolto, quale bridge/risultato certificato è stato usato, la provenance/ragione rilevante e la geometria fisica/validità E1 risultante.

Il fingerprint globale di configurazione MUST NOT richiedere l'enumerazione diretta di ogni trasformazione numerica per sample quando è derivata deterministicamente dal contenuto immutabile del sample/sorgente. L'invariante è: **stessa identità globale della cache + stesso sample sorgente immutabile ⇒ stesso bridge risolto e stesso sample fisico in cache**. Nessuna provenance del bridge diventa una learned policy feature.

Cambiamenti di semantica del bridge MUST invalidare le cache interessate e richiederne rebuild; MUST NOT aggiornare in-place una cache come se fosse identica. Una semplice versione nominale senza le dipendenze effettive non basta. Questo documento non assegna un fingerprint finale né impone un formato di serializzazione.

## 19. Ordine concettuale G1 e causal-topology

```text
Canonical/current source state
  → SMORESAdapter / typed graph state
  → causal topology reconciliation, dove richiesta
  → D1 task/environment view
  → D2 PolicyInputs
  → certified source-frame bridge resolution
  → E1 physical frame transform
  → E2 feature/object/relation extraction
  → representability validation
  → physical G1 cache completa e validata
  → F2 normalization, lifecycle separato
```

L'ordine sopra è normativo: il core causal-topology esistente consuma uno `StateGraph` tipizzato; la decodifica dello stato MUST precedere la reconciliation. La reconciliation MUST precedere D2, E1, E2, candidate extraction e representability validation. Il label codec conserva le autorità già approvate nel piano coalesced; il bridge non le modifica.

Le responsabilità MUST restare indipendenti: causal topology certifica **quali connessioni correnti** sono disponibili nel sample coalesced; source-frame bridging certifica **come esprimere spazialmente** la geometria corrente. Il bridge MUST NOT rimuovere/aggiungere edge, cambiare boundary coalesced, ricostruire pose o modificare target/membership. Restano invariati un sample per transition e le categorie temporali 6174/113096/309/21.

## 20. Separazione G1/F2

L'attuale `prepare_dataset` chiama ancora `fit_normalizers` durante prepare. L'ordine approvato per l'integrazione futura è: **prima completare/validare/pubblicare la cache fisica G1; poi fit/apply F2 nel suo lifecycle separato**, train-only sui 49 episodi / 78302 sample train.

Il bridge MUST far parte della derivazione fisica prima di F2. Nessun fit su G1 incompleto, nessuna normalizzazione delle trasformazioni per nascondere un errore di frame. Questo è un vincolo di integrazione del Task 4; non ridisegna l'engine F2 e non autorizza un fit.

## 21. Test richiesti per l'implementazione futura

Questi sono criteri di accettazione, non step di un implementation plan; nessun test viene implementato o eseguito da questa consegna.

| Test/regressione | Esito richiesto |
|---|---|
| World identity | Stesso frame dichiarato; risultato E1 indipendente corretto. |
| Stage_local /1725 | Origine/yaw e riser numerici di §8 entro 1e-9; vincoli parziali preservati. |
| Stage_frame mancante o malformato | Fail-closed, ragione esplicita, nessun fallback. |
| Legacy RC certificato | Identità source→world ammessa dalla prova del contratto. |
| Map arbitrario/non certificato | Nessuna auto-risoluzione dal nome `map` o dal robot ID. |
| Navigation teleop WORLD | Non richiede/consuma l'eccezione map legacy. |
| Unknown global frame | Quantità invalida e diagnosi FRAME_SEMANTICS_UNKNOWN. |
| Mutazione target/action/labels/futuro/masks/stage/FSM | Bridge e derivazioni invarianti a fisica/evidenze uguali. |
| No double transform | Confronti indipendenti; world→policy eseguito una volta. |
| Connector body-local | Preservato esattamente; il provider globale non lo riceve. |
| Edge B-in-A | Preservato esattamente, salvo codec orientamento E2. |
| Intrinsic/resource-local | Valori preservati; depression e dimensioni non traslate. |
| Body_local task privo di owner | Fail-closed, nessuna owner inference. |
| E2 cone dedup | 3D/2D equivalenti canonicalizzati, nessuna componente inventata. |
| E2 stairs WORLD/local dedup | Equivalenza fisica, sei righe ripristinate prima del dedup, risultato §13. |
| RC relation restoration | 8→24, collision_box=8 e cone=16, nessun oggetto duplicato. |
| Valid-D2→invalid-E1 inatteso | Ragione non approvata provoca fail-fast prima di pubblicare G1. |
| Valori/frame/dimensionalità | Nonfinite, fallback zero, coercizioni e transform unsupported rifiutati. |
| Fingerprint/certificazione | Cambiare provider/config/evidenza cambia identità; cache incompatibile rifiutata. |
| Representative-path regression | Automatizzare dove praticabile i sette percorsi e il tracing dedup/relazioni; mantenere esplicito il limite rappresentativo. |

## 22. Coordinate limits: EXPECTED_SCHEMA_VALIDITY

Il [RobotSpec SMORES corrente](../../../configs/bc_v1/smores_ep.robot.json) dichiara per modulo quattro coordinate: `left_wheel`, `right_wheel`, `pan`, `tilt`. Le prime tre sono `periodic=true` e omettono `limits`; TILT dichiara [-π/2, π/2]. `CoordinateSpec.limits` ha default `None` nel [contratto generico](../../../drl/IL/bc_v1/bc_v1_contract.py).

In [extract_features](../../../drl/IL/bc_v1/feature_schema.py), `coordinate.limits` viene dal catalogo `cs.limits`, mentre posizione/velocità vengono dalla misura corrente. Su otto moduli questa semantica spiega **8 limit rows valide, 24 invalide**, con **32/32 position e velocity valide** nei casi osservati: **EXPECTED_SCHEMA_VALIDITY**, non perdita di sensori/pipeline.

L'assenza del limite di posizione nel catalogo MUST NOT essere interpretata come misura mancante, limite infinito inventato o assenza di limiti operativi di velocità/azione. Il conteggio 8/24 non è una regola del core per N variabile. Nessun redesign del RobotSpec è richiesto da questa specifica.

## 23. Non-goals

Fuori scope: implementazione Task 4; Canonical v7 o mutation v6; rewrite RAW/compact; runtime TF redesign; FreeBOT implementation; nuovi sensori; modello/GNN; G2; training/tuning; fit F2; prepare completo/cache G1; general legacy cleanup; causal-topology redesign. Nessun piano di implementazione viene scritto o modificato. Nessun stage/commit/push/reset/restore/clean/switch o intervento sui dirty file preesistenti.

## 24. Cleanup legacy

Il codice legacy/pre-BC-v1 MUST essere rimosso soltanto dopo consumer tracing e verifica di non-regressione. Le geometrie spaziali root-relative legacy possono diventare candidate di cleanup quando nessun consumer reale le richiede; oggi restano immutabili nel Canonical e non sono reinterpretate come WORLD.

**`edge.relative_pose` è REQUIRED e MUST NOT essere considerata un cleanup candidate.** Root/structural roles non sono feature BC attive, ma la loro esclusione dalla policy non dimostra che i relativi producer siano codice morto per tutti i consumer storici.

## 25. Limite rappresentativo, self-review e gate successivo

L'architettura è sostenuta da exact representative-path checks dei principali percorsi semantici/dati BC-v1. Gli audit **non provano che tutti i 119600 sample** siano privi di dati malformati/inattesi. Gli invarianti fail-fast del futuro G1 sono il meccanismo che impedisce a nuove istanze di questo problema di entrare silenziosamente nel training.

Self-review della specifica: nessun placeholder; nessuna regola map=world generica; un'autorità E1 per la trasformazione; provider causali con provenance fuori input; semantiche connector/edge locali preservate; dedup fisica distinta da perdita; compatibility con coalescing/topology e whitelist invariata; cache fisica prima di F2; limiti supportati da RobotSpec/code; nessun ampliamento verso runtime, FreeBOT, modello o cleanup.

**Gate successivo: USER REVIEW AND EXPLICIT APPROVAL OF THE WRITTEN SPEC.** Questo documento non approva né avvia implementation plan o Task 4.
