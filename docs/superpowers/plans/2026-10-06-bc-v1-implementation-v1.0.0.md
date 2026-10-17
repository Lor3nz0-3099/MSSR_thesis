# BC-v1 Implementation Plan — v1.0.0

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Questo documento non autorizza a iniziare: attendere la revisione e l'approvazione dell'utente. Nessun commit è autorizzato prima della revisione; nessun dispatch a sub-agent è implicito.

**Goal:** Preparare e implementare per piccoli cicli TDD una BC-GNN feed-forward hardware-agnostic sui 70 episodi / 119521 timestep selezionati dal Canonical v5 immutabile.

**Architecture:** Contratti generici e SMORESAdapter precedono qualsiasi trainer. La preparazione produce una cache derivata con input G/o/c separati da label e audit; la rete e il trainer consumano soltanto il contratto generico. Decision scheduler, feasibility runtime, recorder e Isaac Lab restano fuori da questo piano.

**Tech Stack:** Python 3.10+, dataclass/typing/json/hashlib stdlib per il contratto; NumPy per feature e statistiche; PyTorch per collate, GNN e training; pytest per verifica. Nessuna dipendenza Isaac/ROS2/SMORES/Isaac Lab o PyG obbligatoria nel core. Nessuna installazione durante la presente fase documentale.

**Spec:** [BC-v1 architecture v1.0.0](../specs/2026-10-06-bc-v1-architecture-v1.0.0.md). Leggere entrambi i documenti prima di eseguire un task.

**Revisione documentale 2026-10-07, draft da approvare:** GRAVITY_SETTLE enabled di default e trainabile; fase T di mini-tuning inserita tra J e training finale K; audit manuale dei 22 ALIGN prima di A1. [Audit completo](../../bc_v1_align22_manual_audit_20261007.md): 18 CERTAIN_CONTINUATION e 4 LIKELY_CONTINUATION, giudizi proposti. Nessun filtro o conteggio congelato modificato, nessuno step TDD iniziato.

## Global Constraints

- Sorgente immutabile: `datasets/expert_v1_canonical_v5`, manifest SHA-256 `0a6a63dccebeb300b2fe30ead583980288eaafa604b4823ce7c1a290138d9b88`.
- Partire da `valid_for_behavior_cloning is True`; filtro delle 22 sole label della tabella spec §3; 70 episodi / 119521 timestep finali.
- La lista delle 22 esclusioni è ancora da revisionare personalmente: **non iniziare A1 prima dell'approvazione dell'audit** e non applicare/modificare ora il filtro. I conteggi 70/119521 restano congelati come previsione del filtro proposto; qualsiasi riclassificazione richiede revisione esplicita.
- GRAVITY_SETTLE resta enabled=true di default nel catalogo configurato, con tutte le 24 label supervisionate. Nessun cutoff di frequenza può disabilitarla; il core usa il codec generico, non il nome della skill.
- Il core si importa senza Isaac Sim, ROS2, Isaac Lab o package SMORES; il contratto anche senza NumPy/PyTorch.
- Nessun N=8, C=4, PAN/TILT, morphology enum o face-logits fissi nel core.
- Nessuna WAIT/no-dispatch head; nessuna GRU/LSTM/Transformer o memoria temporale.
- Control_state storico, futuro, target graph/morphology, provenance e supervision mask non entrano negli input.
- Nessuna modifica a runtime/executor, recorder, codice Canonical, demo, manifest backend esistenti o Isaac Lab.
- Split per episodio 48/11/11, seed 20261006; normalizzazione train-only; loss simultanee wheels/internal.
- Artifact scritti solo sotto `artifacts/bc_v1/` o directory di run esplicitamente selezionata, mai nelle sorgenti.
- Non creare ora codice, config o test previsti: in questa fase si aggiornano soltanto specifica/piano e si crea il documento audit.

## Review Focus

1. Peer strutturale non presente direttamente nella label Canonical: recuperare solo evidenza della decisione, verificata e non ambigua; mai usare target/futuro. Test B1/B2.
2. Filtro applicato a snapshot diversi o righe con altre label: errore sul drift, mantenere le altre supervisioni. Test C.
3. Task/stage missing, route nominale, frame di metadati incerto: UNKNOWN/missing e nessuna geometria inventata. Test D1/D2/E1.
4. Grafi simmetrici/disconnessi, zero connector/edge e permutazioni di ID: batch corretto e output equivarianti senza root artificiale. Test E1/G2/H2.
5. Cache o checkpoint con dati/split/schema diversi e resume a metà epoch: rifiuto incompatibilità e ripristino RNG/sampler. Test G1/K2.
6. Skill rara esclusa dal catalogo/loss a causa del solo supporto: default enabled, scope/target/gradiente/sampler/metriche GRAVITY_SETTLE. Test A1/A2/B1/B2/E2/H2/I1/I2/J.
7. Leakage dal test durante tuning o scelta non riproducibile: accesso train/validation esplicito, grid di 12 trial, budget/seeds/fingerprint/config e metriche salvati. Test T/K1/K2.

---

## File previsti: lista esatta

Tutti i seguenti file di implementazione sono nuovi. Nessun file applicativo esistente è da modificare nel perimetro iniziale. Gli unici file prodotti o aggiornati nella fase documentale sono i tre documenti elencati qui sotto.

### Documenti creati nella fase 1

```text
docs/superpowers/specs/2026-10-06-bc-v1-architecture-v1.0.0.md
docs/superpowers/plans/2026-10-06-bc-v1-implementation-v1.0.0.md
docs/bc_v1_align22_manual_audit_20261007.md
```

### Package e dipendenze da creare dopo approvazione

```text
drl/IL/bc_v1/__init__.py
drl/IL/bc_v1/bc_v1_contract.py
drl/IL/bc_v1/smores_adapter.py
drl/IL/bc_v1/filters.py
drl/IL/bc_v1/task_context.py
drl/IL/bc_v1/features.py
drl/IL/bc_v1/split.py
drl/IL/bc_v1/normalization.py
drl/IL/bc_v1/dataset.py
drl/IL/bc_v1/model.py
drl/IL/bc_v1/losses.py
drl/IL/bc_v1/metrics.py
drl/IL/bc_v1/training.py
drl/IL/bc_v1/tuning.py
drl/IL/bc_v1/__main__.py
drl/IL/bc_v1/requirements.txt
```

### Config versionati da creare dopo approvazione

```text
configs/bc_v1/robot_spec.schema.json
configs/bc_v1/smores_ep.robot.json
configs/bc_v1/counts.v1.json
configs/bc_v1/align_filter.v1.json
configs/bc_v1/task_context.json
configs/bc_v1/features.json
configs/bc_v1/training.json
configs/bc_v1/tuning.json
```

### Fixture e test da creare dopo approvazione

```text
drl/IL/tests/bc_v1/fixtures.py
drl/IL/tests/bc_v1/test_contract.py
drl/IL/tests/bc_v1/test_robot_spec.py
drl/IL/tests/bc_v1/test_smores_adapter.py
drl/IL/tests/bc_v1/test_dataset_counts.py
drl/IL/tests/bc_v1/test_filters.py
drl/IL/tests/bc_v1/test_task_context.py
drl/IL/tests/bc_v1/test_whitelist.py
drl/IL/tests/bc_v1/test_features.py
drl/IL/tests/bc_v1/test_split.py
drl/IL/tests/bc_v1/test_normalization.py
drl/IL/tests/bc_v1/test_dataset.py
drl/IL/tests/bc_v1/test_model.py
drl/IL/tests/bc_v1/test_losses.py
drl/IL/tests/bc_v1/test_metrics.py
drl/IL/tests/bc_v1/test_training.py
drl/IL/tests/bc_v1/test_tuning.py
drl/IL/tests/bc_v1/test_checkpoint.py
```

Non sono previsti `FreeBOTAdapter`, file runtime nuovi, conversioni HDF5 o migrazioni dei moduli Canonical. `fixtures.py` contiene robot sintetici e una superficie FreeBOT-like, non un backend.

Artifact generati, non file sorgente da creare ora: `artifacts/bc_v1/v1.0.0/{input_manifest.json,label_evidence.json,counts.json,split.json,normalization.json,cache/manifest.json,cache/episodes/<episode_id>.pt}`; run sotto `artifacts/bc_v1/runs/<run_id>/{config.json,environment.json,metrics.jsonl,best.pt,last.pt}`. I segmenti dinamici sono determinati da episodio/run, non nomi di file applicativi.

Artifact tuning: `artifacts/bc_v1/tuning/<search_id>/{search_manifest.json,selection.json,trials/<trial_id>/{config.json,environment.json,metrics.jsonl,best.pt,summary.json}}`. Nessun artifact di training/tuning viene creato adesso.

Ordine aggiornato: **review manuale → A1/A2 → B1/B2 → C → D1/D2 → E1/E2 → F1/F2 → G1/G2 → H1/H2 → I1/I2 → J → T → K1/K2 → training finale → valutazione test esplicita**. T introduce il loop breve train/validation necessario ai trial; K lo estende per training completo/resume. L'implementazione del trainer non obbliga ad avviare subito il training finale.

## Procedura TDD comune

Comandi eseguiti dalla root repository con `python3 -m pytest`. Ogni task segue RED osservato -> implementazione minima -> GREEN -> controllo del diff. Il RED deve fallire per il comportamento richiesto o il modulo non ancora esistente, non per un errore di sintassi o per un dataset sbagliato. La suite del task non è saltata per mancanza di dipendenze: predisporre l'ambiente previsto prima del relativo ciclo.

Alla fine di ogni task registrare comando e risultato; non segnare completato un task senza verifica. Nessun commit automatico è incluso. Se si scopre la necessità di modificare un file escluso o cambiare conteggi, fermare la relativa dipendenza e proporre una revisione della spec, continuando soltanto lavoro indipendente autorizzato.

Per i test reali `test_dataset_counts.py`, `--source` predefinito è il percorso locale congelato. La suite completa richiede quel dataset: una sorgente assente produce un errore esplicito, non un falso GREEN. I test unitari restano separati e utilizzano le fixture.

## A1 — Dataclass generiche e import isolation

**File:** creare `drl/IL/bc_v1/{__init__.py,bc_v1_contract.py}` e `drl/IL/tests/bc_v1/{fixtures.py,test_contract.py}`. __init__ non importa model/training o adapter.

**Interfacce prodotte:** `RobotSpec`, `ModuleTypeSpec`, `ConnectorSpec`, `ResourceSpec`, `ActionSpec`, `TaskContext`, `StateGraph`, `ObservationBundle`, `PolicyInputs`, `ResourceTarget`, `StructuralTarget`, `SupervisionBundle`, `SampleAudit`, `PolicySample`, `ActionChoice`, `LabelEvidence`; protocollo `RobotAdapter` con `encode_state(graph: Mapping) -> StateGraph`, `encode_labels(action: Mapping, supervision: Mapping, evidence: Sequence[LabelEvidence]) -> SupervisionBundle`, `decode_action(choice: ActionChoice) -> Mapping`.

`PolicySample` ha esattamente `inputs`, `supervision`, `audit`; `PolicyInputs` ha esattamente `graph`, `observation`, `context`. ResourceTarget identifica resource/mode/payload/mask; StructuralTarget identifica iniziatore/operazione/partecipanti/connector/parametri/mask. LabelEvidence è label-only e non può essere un campo di PolicyInputs.

- [ ] **RED test:** `test_contract_imports_without_hardware_or_tensor_packages` usa un subprocess con import blocker per isaacsim/isaaclab/rclpy/smores_ep/freebot_docking/torch/numpy; l'import del package e del contratto deve riuscire. `test_policy_sample_separates_inputs_targets_and_audit` verifica campi separati e dataclass frozen; `test_state_supports_variable_entity_counts` usa N=2/5/8/12, connector=0/1/4/6 e una surface.
- [ ] **RED test aggiuntivo:** ActionSpec generica rappresenta una skill senza connector, con scope multi-module e parametri tipizzati, default enabled=true; la fixture neutral-recovery non ha nomi PAN/TILT o import hardware. Nessuna API del contratto deduce enabled dal conteggio delle label.
- [ ] **RED verifica:** `python3 -m pytest drl/IL/tests/bc_v1/test_contract.py -q`; osservare la mancanza del contratto o la violazione di un'asserzione.
- [ ] **Implementazione minima:** soli tipi stdlib, tuple/frozen dataclass e protocollo; riferimenti espliciti, nessun builder, encoder o trainer. Le fixture non importano hardware.
- [ ] **GREEN verifica:** ripetere lo stesso comando; aggiungere `python3 -c 'import drl.IL.bc_v1; import drl.IL.bc_v1.bc_v1_contract'`.
- [ ] **Completamento:** tutti i test verdi; contratto senza cardinalità hardware e senza dipendenze esterne; nessun file fuori dall'elenco A1 modificato.

## A2 — RobotSpec e schema validabile

**File:** modificare `bc_v1_contract.py`, `fixtures.py`; creare `configs/bc_v1/robot_spec.schema.json` e `test_robot_spec.py` nelle directory già indicate.

**Interfacce:** `load_robot_spec(path: Path) -> RobotSpec`, `validate_robot_spec(spec: RobotSpec) -> None`, `robot_spec_fingerprint(spec: RobotSpec) -> str`. Schema `mssr.bc_v1.robot_spec.v1`; tipi connector fixed_port/surface; capability e nomi modalità sono configurati.

- [ ] **RED test:** `test_spec_rejects_dangling_resource_and_connector_refs`, `test_spec_rejects_duplicate_ids_nonfinite_limits_and_unknown_codec`, `test_generic_surface_and_variable_payload_spec_roundtrip`, `test_spec_fingerprint_is_order_canonical`. Assert su errori espliciti per unità/limiti/payload inconsistenti e nessun vincolo a quattro connector. Default enabled=true; un override false è esplicito, serializzato e cambia il fingerprint, mai derivato dal supporto.
- [ ] **RED verifica:** `python3 -m pytest drl/IL/tests/bc_v1/test_robot_spec.py -q`.
- [ ] **Implementazione minima:** validazione stdlib dei campi richiesti, codec continuous/categorical/module_ref/connector_ref/reference_set e riferimenti; schema JSON coerente con le stesse regole. Canonical JSON per hash; non costruire una registry di plugin dinamici.
- [ ] **GREEN verifica:** stesso comando, poi `python3 -m pytest drl/IL/tests/bc_v1/test_contract.py drl/IL/tests/bc_v1/test_robot_spec.py -q`.
- [ ] **Completamento:** le fixture SMORES-like e surface passano lo stesso schema; errori di spec sono intercettati prima della tensorizzazione.

## B1 — SMORESAdapter e target strutturali completi

**File:** creare `smores_adapter.py`, `configs/bc_v1/smores_ep.robot.json`, `test_smores_adapter.py`; modificare `fixtures.py`.

**Consuma:** contratto A1/A2. **Produce:** `SMORESAdapter(spec: RobotSpec)` che implementa RobotAdapter; `read_label_evidence(row: Mapping, repo_root: Path) -> tuple[LabelEvidence,...]`. Lettura JSON pura delle sorgenti puntate, con index per file per evitare scansioni ripetute.

- [ ] **RED test:** mapping wheels e tre modalità interne con unità esatte; wheels+internal nello stesso modulo; connector geometry locale; ALIGN/DOCK/UNDOCK/SETTLE. `test_structural_peer_is_recovered_from_decision_only` usa label Canonical senza peer e sorgente con module_ids. `test_missing_or_ambiguous_peer_evidence_fails` rifiuta evidenze discordanti. `test_adapter_does_not_mutate_source` confronta deep copy.
- [ ] **RED test aggiuntivo:** `test_gravity_settle_is_enabled_by_default_and_label_roundtrips` verifica scope partecipanti, durata e parametri passivi senza sintetizzare target pan/tilt zero. Supporto 24, 1 o assente nel batch non modifica il catalogo; una config di ablation distinta è l'unico modo di cambiare enabled.
- [ ] **RED verifica:** `python3 -m pytest drl/IL/tests/bc_v1/test_smores_adapter.py -q`.
- [ ] **Implementazione minima:** config hardware derivato dai manifest/convenzioni esistenti; encode fisico e label separati; recuperare prima i partecipanti dalla joint action Canonical, poi usare l'evidenza sorgente soltanto dove necessaria. Confrontare iniziatore/primitive/parametri; hash delle sorgenti aggiuntive lette; decode serializzabile, senza dispatch. Nessun import smores_ep per limiti o conversioni.
- [ ] **GREEN verifica:** stesso comando; import adapter con import blocker hardware in subprocess.
- [ ] **Completamento:** target strutturali rappresentabili, missing non inventati, nomi hardware confinati a adapter/config/report.

## B2 — Regressione sui conteggi prima del filtro

**File:** creare `test_dataset_counts.py`, `configs/bc_v1/counts.v1.json`; modificare `smores_adapter.py` per `audit_canonical_source(root: Path, adapter: RobotAdapter) -> Mapping`.

- [ ] **RED test:** `test_unfiltered_source_matches_frozen_counts` verifica hash, 77 episodi, 190767 transition, 119543 righe BC, 70 episodi BC; label wheels=576942, pan_velocity=186484, pan_target=611, tilt_target=84736, align=482, dock=456, undock=106, settle=24. `test_all_supervised_structural_arguments_are_resolvable` verifica peer/scope. `test_source_fingerprints_unchanged_after_audit` confronta hash prima/dopo.
- [ ] **RED test aggiuntivo:** tutte le 24 label GRAVITY_SETTLE producono target abilitati e rappresentabili, senza threshold di rarità; il conteggio delle label non viene usato per inferire decision point.
- [ ] **RED verifica:** `python3 -m pytest drl/IL/tests/bc_v1/test_dataset_counts.py -k unfiltered -q` e `python3 -m pytest drl/IL/tests/bc_v1/test_dataset_counts.py -k 'arguments or fingerprints' -q`.
- [ ] **Implementazione minima:** census streaming readonly con SHA-256 file, conteggi per mask e report per task/morfologia/provenance; expected globali dal documento, non calcolati dalle stesse funzioni sotto test. Il config conserva expected prima/dopo e fingerprint; il report derivato conserva anche le celle complete. Nessun materializer o overwrite.
- [ ] **GREEN verifica:** ripetere entrambi i comandi; verificare che i sette expert button contribuiscano zero.
- [ ] **Completamento:** conteggi esatti e nessuna label strutturale incompleta. Un disaccordo blocca C: non applicare un filtro diverso per nasconderlo.

## C — Filtro versionato delle 22 ALIGN

**Prerequisito documentale:** la revisione manuale richiesta prima di A1 è completata. Usare soltanto la lista esplicitamente approvata; non promuovere i quattro LIKELY a esclusioni certe in base alla sola fase. I numeri sotto restano quelli congelati finché l'utente non revisiona l'audit.

**File:** creare `filters.py`, `configs/bc_v1/align_filter.v1.json`, `test_filters.py`; modificare `test_dataset_counts.py`.

**Interfacce:** `load_align_filter(path: Path) -> Mapping`, `apply_label_filter(row: Mapping, filter_spec: Mapping) -> Mapping`, `is_bc_eligible(row: Mapping) -> bool`. La vista filtrata è una copia; l'originale non cambia.

- [ ] **RED test:** le 22 chiavi spec §3 sono presenti esattamente, con hash manifest/episodio e fase documentata; `test_filter_preserves_other_labels_and_is_idempotent`; `test_filter_rejects_missing_duplicate_or_drifted_entry`; `test_filtered_source_matches_frozen_counts` assert 119521 timestep, 70 episodi, ALIGN=460 e tutte le altre label invariate.
- [ ] **RED verifica:** `python3 -m pytest drl/IL/tests/bc_v1/test_filters.py -q`, poi `python3 -m pytest drl/IL/tests/bc_v1/test_dataset_counts.py -k filtered -q`.
- [ ] **Implementazione minima:** allowlist esplicita di esclusioni, mai regex/euristica globale; azzerare solo il mask della label esclusa nella vista, aggiornare l'eleggibilità della copia. Validare che tutte le voci corrispondano una volta. Registrare motivi nell'audit, senza feature.
- [ ] **GREEN verifica:** stessi comandi; controllare 12 align + 9 clocking + 1 approach e 22 righe senza altre label nella sorgente corrente.
- [ ] **Completamento:** filtro riproducibile e 70/119521 raggiunti senza mutazioni o eliminazione di episodi.

## D1 — TaskContext da metadati correnti

**File:** creare `task_context.py`, `configs/bc_v1/task_context.json`, `test_task_context.py`; modificare `fixtures.py`.

**Interfacce:** `read_task_metadata(row: Mapping, repo_root: Path) -> Mapping`, `build_task_view(row: Mapping, metadata: Mapping, config: Mapping) -> tuple[TaskContext,ObservationBundle]`. Il primo metodo restituisce solo fonti di task/ambiente autorizzate, non primitive o target graph.

- [ ] **RED test:** task UNKNOWN/navigation/gap/stairs/button e stage UNKNOWN/assembly/reconfiguration/task_execution; `test_missing_context_does_not_use_next_record_or_action`; parametri costanti nello stage e cambio esplicito di stage; geometria con fonte/frame certificati; conflitti tra fonti equivalenti provocano errore, dati irrecuperabili sono missing.
- [ ] **RED verifica:** `python3 -m pytest drl/IL/tests/bc_v1/test_task_context.py -q`.
- [ ] **Implementazione minima:** lettura metadati raw/compact esistenti e environment_geometry, alias semantici espliciti, lookup del solo task corrente. Conservare fonte/hash fuori da TaskContext osservabile. Nessun forward/backward fill e nessuna deduzione da DISPATCHING_*.
- [ ] **GREEN verifica:** stesso comando con fixture raw, compact e missing.
- [ ] **Completamento:** TaskContext indipendente da RobotSpec/backend, con validity; nessun accesso a record futuri per completarlo.

## D2 — Whitelist G/o/c e separazione dall'audit

**File:** modificare `task_context.py`, `test_smores_adapter.py`; creare `test_whitelist.py`.

**Interfaccia:** `select_policy_inputs(graph: StateGraph, observation: ObservationBundle, context: TaskContext) -> PolicyInputs`. `SampleAudit` e `SupervisionBundle` non sono argomenti del metodo.

- [ ] **RED test:** `test_forbidden_fields_cannot_change_policy_inputs` perturba control_state, root/role da template, morphology, G_t+1, target graph/morphology, azioni, provenance e mask; input identici. `test_nav2_waypoints_are_not_cones` modifica route/waypoint e verifica coni invariati. `test_current_physical_cone_change_changes_observation` conferma sensibilità alla geometria ammessa. Configura una fixture di input con valori fisici identici e label opposte.
- [ ] **RED verifica:** `python3 -m pytest drl/IL/tests/bc_v1/test_whitelist.py -q`.
- [ ] **Implementazione minima:** costruzione esplicita dei soli campi whitelist, senza serializzare dizionari raw; escludere root/role template anche negli stati stabili. Il join label B1 non è chiamato dal costruttore input.
- [ ] **GREEN verifica:** stesso comando più D1.
- [ ] **Completamento:** equivalenza degli input sotto perturbazioni proibite e nessun embedding ID/morfologia/backend.

## E1 — Frame causale e trasformazioni fisiche

**File:** creare `features.py`, `test_features.py`; modificare `fixtures.py`.

**Interfacce:** `physical_frame(graph: StateGraph, frame_config: Mapping) -> Mapping`, `transform_inputs(inputs: PolicyInputs, frame_config: Mapping) -> PolicyInputs`. Frame restituito con origine, rotazione e validity, senza root_module_id.

- [ ] **RED test:** centroide di N moduli; rinomina e riordino preservano frame/features; grafi simmetrici/disconnessi/transienti; q e -q equivalenti; trasformazione coerente di coni, bottone, twist ed edge; geometria root-relative senza posa origine certificata è missing; quaternion nullo/nonfinite rifiutato se dichiarato valido.
- [ ] **RED verifica:** `python3 -m pytest drl/IL/tests/bc_v1/test_features.py -k frame -q`.
- [ ] **Implementazione minima:** NumPy, centroide aritmetico e assi fisici configurati; R6 per orientamento; connector body-local e edge relativi. Nessun PCA/min-ID/template e nessuna velocità sintetica da differenza di timestep futuri.
- [ ] **GREEN verifica:** stesso comando; traslazione comune di robot e geometria conserva le feature relative.
- [ ] **Completamento:** stesso frame in stabili/transienti, transform coerente e input indipendenti da nomi arbitrari.

## E2 — Feature schema, resource coordinate e candidati

**File:** modificare `features.py`, `test_features.py`; creare `configs/bc_v1/features.json`.

**Interfacce:** `extract_features(inputs: PolicyInputs, spec: RobotSpec, schema: Mapping) -> Mapping[str, np.ndarray]`, `enumerate_candidates(graph: StateGraph, spec: RobotSpec) -> Mapping`. Sono candidate table e appartenenze generiche, senza mask runtime.

- [ ] **RED test:** module base [N,30]; larghezze token da schema; missing con validity, nessun numero hard-coded di coordinate/connector; `test_candidates_do_not_depend_on_supervised_action` confronta stessi input con label opposte; endpoint/scope compatibili, ogni label B1 rappresentabile; SETTLE senza connector, enabled nel config default e con scope rappresentabile senza leggere la label; ordinali/ID non compaiono nelle feature.
- [ ] **RED verifica:** `python3 -m pytest drl/IL/tests/bc_v1/test_features.py -q`.
- [ ] **Implementazione minima:** field registry ordinata configurata per quantità/codec, coordinate token con riferimenti alla resource; enumerazione per capacità e riferimenti correnti. Pruning geometrico runtime non implementato; non escludere ALIGN lontani in base alla distanza finale DOCK.
- [ ] **GREEN verifica:** stesso comando con robot N=2/5/8/12 e surface fixture; widths uguali a schema, cardinalità diverse.
- [ ] **Completamento:** schema versionato con dimensioni/unità/validity e candidati agnostici; nessuna dipendenza da supervisione per costruirli.

## F1 — Split stratificato per episodio

**File:** creare `split.py`, `test_split.py` e `configs/bc_v1/training.json`, inizialmente con quote/strati e parametri congelati dalla spec.

**Interfacce:** `make_episode_split(episodes: Sequence[Mapping], strata_config: Mapping, seed: int=20261006) -> Mapping[str,tuple[str,...]]`, `validate_split(split: Mapping, episodes: Sequence[Mapping], strata_config: Mapping) -> None`. Le quote sono quelle spec §10, passate dal config; non imporre nomi di task/morfologie nel codice split.

- [ ] **RED test:** assert 48/11/11 e quote 12/3/3 gap, 12/3/3 stairs, 9/2/2 nav, 6/1/1 teleop button, 9/2/2 mission; disjoint union=70; indipendenza dall'ordine d'ingresso; raw/compact della stessa lineage insieme; collisione hash/lineage incompatibile con quote -> errore.
- [ ] **RED verifica:** `python3 -m pytest drl/IL/tests/bc_v1/test_split.py -q`.
- [ ] **Implementazione minima:** grouping, ordinamento e RNG locale; tag strato dall'audit, non input di policy. Manifest split con hash dati/config. Non randomizzare righe.
- [ ] **GREEN verifica:** stesso comando; due esecuzioni producono stessi episodi per split.
- [ ] **Completamento:** split deterministico e verificato senza leakage; nessun accesso al test set per calcolare normalizzatori.

## F2 — Normalizzazione train-only

**File:** creare `normalization.py`, `test_normalization.py`.

**Interfacce:** `fit_normalizers(samples: Iterable[PolicySample], train_ids: set[str], spec: RobotSpec, schema: Mapping) -> Mapping`, `normalize_features(features: Mapping, stats: Mapping) -> Mapping`, `normalize_target(target: ResourceTarget, stats: Mapping) -> ResourceTarget`, `denormalize_values(values: np.ndarray, key: str, stats: Mapping) -> np.ndarray`.

- [ ] **RED test:** holdout con valori estremi non cambia media/scala; missing/padding ignorati; scala zero ->1; campo senza supporto train -> identità con diagnostica; modalità rad e rad/s normalizzate separatamente; limiti bounded usati solo se dichiarati; categorie/R6/sincos/validity invariati; roundtrip target SI.
- [ ] **RED verifica:** `python3 -m pytest drl/IL/tests/bc_v1/test_normalization.py -q`.
- [ ] **Implementazione minima:** accumulatore streaming sui soli sample train, scaler per field/mode/unità; hash membership train nei parametri. Nessun global min/max, clipping o imputazione che muti le label.
- [ ] **GREEN verifica:** stesso comando; stats identiche cambiando validation/test.
- [ ] **Completamento:** fit isolato e trasformazioni riproducibili; metadati normalizzazione serializzabili.

## G1 — Dataset, cache e CLI prepare

**File:** creare `dataset.py`, `__main__.py`, `requirements.txt`, `test_dataset.py`; modificare `test_dataset_counts.py`.

**Interfacce:** `prepare_dataset(source: Path, output: Path, adapter: RobotAdapter, config: Mapping) -> Mapping`, `BCDataset(cache_root: Path, split_name: str)` con __len__/__getitem__ -> PolicySample, `cache_fingerprint(config: Mapping, input_hashes: Mapping) -> str`. CLI prepare con source/output/config espliciti; config predefinito `configs/bc_v1/training.json`.

Dipendenze dichiarate nel requirements: `numpy>=1.24,<3`, `torch>=2.4,<3`, `pytest>=8,<9`; nessun simulator/framework RL. Registrare le versioni effettivamente selezionate nell'ambiente di training. PyTorch non è trovato nel Python locale ispezionato: predisporre un ambiente isolato prima di G1, dopo l'approvazione dell'implementazione, senza modificare l'ambiente Isaac o installare Lab.

- [ ] **RED test:** cache roundtrip preserva label congiunte, input e audit separati; `test_cache_refuses_stale_source_filter_schema_or_split`; prepare rifiuta output dentro Canonical e file collisioni; atomicità del manifest finale dopo un errore; BCDataset non importa hardware; CLI help non avvia simulatori.
- [ ] **RED verifica:** `python3 -m pytest drl/IL/tests/bc_v1/test_dataset.py -k 'cache or prepare or roundtrip' -q`.
- [ ] **Implementazione minima:** pipeline B->C->D->E->F, file .pt per episodio e manifest JSON, scrittura temporanea e pubblicazione solo completa, Dataset lazy readonly e schema generico. I PolicySample in cache sono non normalizzati e già nel frame fisico; la trasformazione E1 si esegue una volta durante prepare. Fingerprint anche dei metadati/evidenze lette. Caricare soltanto cache locali di questa pipeline; controllare fingerprint prima del training. Nessun adapter importato dal Dataset. Il CLI importa l'adapter soltanto nel comando prepare; i comandi training leggono il contratto generico dalla cache.
- [ ] **GREEN verifica:** stesso comando; poi `python3 -m drl.IL.bc_v1 prepare --source datasets/expert_v1_canonical_v5 --output artifacts/bc_v1/v1.0.0 --config configs/bc_v1/training.json`, e `python3 -m pytest drl/IL/tests/bc_v1/test_dataset_counts.py -q`.
- [ ] **Completamento:** cache validata 70/119521, split 48/11/11, normalizzazione train-only e sorgenti con hash invariati. Un errore lascia la precedente cache valida intatta.

## G2 — Collate variabile e sampler iniziale

**File:** modificare `dataset.py`, `test_dataset.py`, `configs/bc_v1/training.json` con sampler/budget e parametri della spec.

**Interfacce:** `collate_policy_samples(samples: Sequence[PolicySample], spec: RobotSpec, schema: Mapping, normalizers: Mapping) -> Mapping`; `JointActionBatchSampler(index: Sequence[Mapping], config: Mapping, seed: int)` iterable di batch di indici, con `state_dict() -> Mapping` e `load_state_dict(state: Mapping) -> None`. Il collate produce contenitori distinti inputs (tensori), supervision (tensori target/mask) e audit (metadati), passando solo inputs al modello. L'indice contiene tag e famiglie delle label, non li aggiunge ai tensori input. Usare functools.partial per legare spec/schema/normalizzatori al DataLoader.

- [ ] **RED test:** offset module/connector/resource/coordinate/candidate corretti tra grafi; sentinel -1 preservato; empty sets; no edge tra sample; distribuzione baseline 96+32 senza stringhe PAN/TILT nel sampler; un sample raro conserva anche wheels; famiglia configurata SETTLE presente nel ramo rare nonostante 24 label; rare pool vuoto fallback; stato sampler riproduce i draw successivi; validation/test visitano una volta ogni riga, senza drop_last. Zero supporto train per una skill enabled dà diagnostica/gate esplicito, mai auto-disable o modifica dello split.
- [ ] **RED verifica:** `python3 -m pytest drl/IL/tests/bc_v1/test_dataset.py -q`.
- [ ] **Implementazione minima:** extract_features E2 sugli input già trasformati, normalize_features/normalize_target F2, concatenazioni e index shift per batch; padding solo payload/categorie con validity. RNG proprio, gerarchia cella->episodio->riga, ramo rare da ID config, replacement e budget ceil(n_train_rows/128). DataLoader num_workers=0 iniziale.
- [ ] **GREEN verifica:** stesso comando; audit frequenze su seed fisso, niente asserzioni fragili su una singola estrazione casuale.
- [ ] **Completamento:** batching senza assunzioni hardware, sampler preserva supervisioni congiunte e consente resume.

## H1 — Encoder e message passing feed-forward

**File:** creare `model.py`, `test_model.py`.

**Interfacce:** `BCGNN(spec: RobotSpec, feature_schema: Mapping, model_config: Mapping)`; `encode(inputs: Mapping[str,Tensor]) -> Mapping[str,Tensor]`. inputs è solo batch['inputs'], senza supervision/audit. Nessun import adapter nel modello.

- [ ] **RED test:** embedding task/stage 8+8, context/observation64, node/resource/connector128, edge64, tre blocchi H128 residual/LayerNorm; nodi disconnessi e empty observation; dropout configurato p=0/p=0.1 negli MLP nascosti ed escluso da input/target/output finale, eval lo disattiva; `test_no_temporal_module_or_state` esclude RNN/Transformer e hidden state persistente; stesso input stesso output in eval.
- [ ] **RED verifica:** `python3 -m pytest drl/IL/tests/bc_v1/test_model.py -k encode -q`.
- [ ] **Implementazione minima:** MLP condivisi, pooling coordinate->resource->module, index_add message passing, pooling mean+max128 per grafo. Context/observation condizionano nodi e decoder. Gestire insiemi vuoti con aggregato zero e validità, non con token hardware inventato.
- [ ] **GREEN verifica:** stesso comando su CPU, con finite forward/backward.
- [ ] **Completamento:** encoder e GNN agnostici, zero stato temporale e nessun array dimensionato sul numero dei moduli.

## H2 — Head generiche e selezione connector/peer

**File:** modificare `model.py`, `test_model.py`.

**Interfacce:** `BCGNN.forward(inputs: Mapping[str,Tensor]) -> Mapping[str,Tensor]`: mode logits, continuous payload per mode, candidate scores e parametri tipizzati. Il chiamante passa solo batch['inputs']. Output contiene binding verso resource/candidate table, non nomi PAN/TILT.

- [ ] **RED test:** head parametrizzate dal catalogo, payload variable-width e masked padding; scelta peer oltre otto moduli/connector oltre quattro; per-module/resource outputs permutati coerentemente e pooling invariato; rinomina ID non cambia score rimappati; ALTERARE supervision/audit non cambia forward; SETTLE module-only/multi-module rappresentabile senza connector e con score trainabile nella config default.
- [ ] **RED verifica:** `python3 -m pytest drl/IL/tests/bc_v1/test_model.py -q`.
- [ ] **Implementazione minima:** decoder condivisi su embedding resource+module+global+context; scorer MLP di candidato con endpoint/operazione configurata; nessuna head inactive/WAIT/dispatch e nessun mask runtime costruito dalla label.
- [ ] **GREEN verifica:** stesso comando, tolleranza float32 dichiarata per permutazioni e batch isolation.
- [ ] **Completamento:** il modello accetta batch di cardinalità diverse e produce solo output del contratto.

## I1 — Loss masked e concorrenza

**File:** creare `losses.py`, `test_losses.py`.

**Interfaccia:** `compute_bc_loss(prediction: Mapping, batch: Mapping, loss_config: Mapping) -> tuple[Tensor,Mapping]`, con totale e termini/supporti. Non legge input audit o backend strings.

- [ ] **RED test:** wheels e internal nello stesso timestep danno entrambi gradiente; maschera falsa -> nessun gradiente sul relativo output; missing dimension e padding esclusi; gruppi candidati separati; loss Huber delta1 e CE equivalenti a esempi numerici piccoli; distribuzione multi-positive normalizzata; zero supporto -> zero finito senza divisioni; modalità interna non valida non ha valore loss; no impliciti target zero.
- [ ] **RED test aggiuntivo:** un target SETTLE abilitato dà loss e gradiente non zero allo scorer generico; assenza nel batch dà contributo zero senza rimuovere i parametri/catalogo; il supporto globale 24 non altera il masking supervisionato.
- [ ] **RED verifica:** `python3 -m pytest drl/IL/tests/bc_v1/test_losses.py -q`.
- [ ] **Implementazione minima:** loss per codec e supporto effettivo, media dimensioni del payload, pesi iniziali1; una sola CE congiunta per candidato strutturale, più eventuali parametri separati. Validare supporti/mask, non ricostruirli dal control_state.
- [ ] **GREEN verifica:** stesso comando e finite backward con più famiglie concorrenti.
- [ ] **Completamento:** loss corrette per sample congiunti, niente coupling artificiale tra heads e nessun contributo da label escluse.

## I2 — Metriche in unità fisiche e supporto

**File:** creare `metrics.py`, `test_metrics.py`.

**Interfacce:** `BCMetrics(spec: RobotSpec, normalizers: Mapping)` con `update(prediction: Mapping, batch: Mapping, audit: Sequence[SampleAudit]) -> None`, `compute() -> Mapping`. Audit usato solo per aggregazione/report dopo il forward.

- [ ] **RED test:** MAE/RMSE denormalizzati corretti; CE accuracy mode, operazione, endpoint, clocking ed exact-match; macro/micro con supporti diseguali; zero supporto -> null, includendo SETTLE nel report anche se un minibatch/split non ha supporto; stesso risultato con suddivisioni del batch diverse; filtro non aumenta supporti; nessuna metrica offline chiamata success rate del task.
- [ ] **RED verifica:** `python3 -m pytest drl/IL/tests/bc_v1/test_metrics.py -q`.
- [ ] **Implementazione minima:** accumulatori somma/conteggio per head e gruppi di metadati, con nomi leggibili forniti dal config/adattamento. Nessuna nuova predizione o modifica del sampler.
- [ ] **GREEN verifica:** stesso comando con output numerici attesi indipendenti dall'implementazione.
- [ ] **Completamento:** report per task/morfologia/provenance e head con unità e supporto espliciti.

## J — Small-overfit prima del trainer completo

**File:** creare `training.py`, `test_training.py`; modificare `__main__.py`, `training.json`.

**Interfaccia:** `run_small_overfit(samples: Sequence[PolicySample], spec: RobotSpec, config: Mapping) -> Mapping`; CLI `python3 -m drl.IL.bc_v1 overfit --cache artifacts/bc_v1/v1.0.0 --split train --output artifacts/bc_v1/runs/overfit-v1`.

- [ ] **RED test:** fixture deterministica con tutte le head iniziali, inclusa la skill generica mappata a GRAVITY_SETTLE, loss finale <=20% loss iniziale entro500 update, valori/gradienti finiti e parametri supervisionati aggiornati; `test_small_overfit_reads_train_only` vieta accesso a validation/test. Il test numerico usa fixture coerente e piccola, non soglie di generalizzazione sui dati reali.
- [ ] **RED verifica:** `python3 -m pytest drl/IL/tests/bc_v1/test_training.py -k overfit -q`.
- [ ] **Implementazione minima:** loop diretto CPU/float32, seed fisso, AdamW e batch fisso, report per head; selezione di un sottoinsieme train reale che copra le famiglie disponibili per smoke separato. Non implementare ancora early stopping/resume.
- [ ] **GREEN verifica:** stesso comando e CLI overfit sopra. Per i dati reali registrare curve/supporti; non applicare automaticamente la soglia della fixture se i target sono ambigui.
- [ ] **Completamento:** prova che la rete può apprendere ogni head e che nessuna loss è scollegata; smoke reale senza lettura del test set.

## T — Mini-tuning riproducibile train + validation (12 trial)

**Posizione:** dopo il GREEN small-overfit J, prima del training finale. Questo task introduce un loop breve di trial train/validation; K1/K2 lo riusano ed estendono, senza richiedere un trainer completo già implementato.

**File:** creare `drl/IL/bc_v1/tuning.py`, `configs/bc_v1/tuning.json`, `drl/IL/tests/bc_v1/test_tuning.py`; modificare `training.py`, `__main__.py`, `test_training.py`. Nessuna nuova dipendenza HPO.

**Interfacce:** `build_trial_configs(base_config: Mapping, search_config: Mapping) -> tuple[Mapping,...]`; `run_trial(config: Mapping, train_loader: DataLoader, validation_loader: DataLoader, output: Path) -> Mapping`; `run_tuning(config: Mapping, search_config: Mapping, output: Path) -> Mapping`; `select_trial(summaries: Sequence[Mapping]) -> Mapping`. Introdurre `evaluate` e il salvataggio del best per trial in `training.py`, poi riusarli in K; nessun argomento test_loader nel tuning. CLI `python3 -m drl.IL.bc_v1 tune --config configs/bc_v1/training.json --search-config configs/bc_v1/tuning.json --cache artifacts/bc_v1/v1.0.0 --output artifacts/bc_v1/tuning/grid-v1`.

- [ ] **RED test:** grid esattamente 12 config uniche con lr={3e-4,1e-3,3e-3}, weight_decay={0,1e-4}, dropout={0,0.1}, baseline inclusa; B128 e ratio96/32 fissi. Trial_id/ordine/config identici a seed e fingerprint uguali; ogni trial reinizializza modello/optimizer/RNG, non eredita pesi dal precedente. `test_tuning_never_opens_test_split` fa fallire qualunque richiesta di dataset/loader test. Normalizzatori soltanto train; validation completa, eval/no_grad e nessun sampler rare. `test_validation_score_is_global_support_weighted` confronta batch partitions. Selezione dal best validation per trial, tie entro1e-6 per trial_id, mai metriche train/test. Tutte le config/metriche per epoch/head/supporti, incluso SETTLE, e stati errore sono salvati; NaN/failure non diventano uno score vincente. Fingerprint diversi impediscono riuso, trial completi saltati al restart; grid incompleta senza selezione finale.
- [ ] **RED verifica:** `python3 -m pytest drl/IL/tests/bc_v1/test_tuning.py drl/IL/tests/bc_v1/test_training.py -q`.
- [ ] **Implementazione minima:** mini-grid stdlib con loop AdamW comune al trainer; 10 epoch/trial, min(200, budget epoch normale) update/epoch, niente early stopping, stesso seed20261006 e stessa sequenza sampler per ogni trial. Benchmark baseline di un epoch completo e validazione prima della grid: stimare durata, ridurre uniformemente il budget prima dell'avvio se eccede una giornata GPU, salvare decisione. Esecuzione sequenziale senza espansione automatica; massimo24000 update/120 validation complete. Manifest/config/JSONL/summary atomici, best.pt per trial con schema/config/fingerprint; full mid-trial resume rinviato a K2, un trial interrotto riparte da seed. Non usare test, future demo, Isaac Lab, Optuna o Ray.
- [ ] **GREEN verifica:** comando unitario sopra; `python3 -m drl.IL.bc_v1 tune --config configs/bc_v1/training.json --search-config configs/bc_v1/tuning.json --output artifacts/bc_v1/tuning/grid-smoke-v1 --smoke` usa fixture train/validation e tutte le 12 configurazioni con budget ridotto, salvando esplicitamente il budget smoke (non dichiararlo tuning reale). Dopo il GREEN e J reale, eseguire il benchmark e la mini-grid sui train/validation reali tramite il comando completo sopra, conservando tutte le run.
- [ ] **Completamento:** ricerca reale completa, `search_manifest.json`, tutti i trial e `selection.json` persistiti; configura il training finale con i valori selezionati, senza leggere test. Baseline resta documentata, valori selezionati registrati separatamente. Determinismo verificato su CPU, limiti/versioni/device GPU dichiarati.

## K1 — Trainer completo, validation e configurazione selezionata

**File:** modificare `training.py`, `__main__.py`, `training.json`, `test_training.py`.

**Interfacce:** `train(config: Mapping) -> Mapping`, `evaluate(model: BCGNN, loader: DataLoader, spec: RobotSpec, normalizers: Mapping) -> Mapping` riusata dal task T. CLI train/evaluate con config, cache e output espliciti; `train --selection <selection.json>` risolve e salva la config finale con fingerprint compatibili.

- [ ] **RED test:** train usa sampler configurato e validation passa tutti i sample una volta senza gradiente; max epoch100, patience15 e best scelto solo da validation; optimizer AdamW legge lr/weight_decay/dropout selezionati da T (1e-3/1e-4/p0 sono soltanto baseline), clip1; `test_final_training_starts_from_scratch_with_selected_config`; mismatch selection/data/split/normalizzatori rifiutato; `test_test_split_is_not_used_for_model_selection`; aggregazione loss support-weighted indipendente dalla partizione in batch; logs separati per head, inclusa SETTLE.
- [ ] **RED verifica:** `python3 -m pytest drl/IL/tests/bc_v1/test_training.py -q`.
- [ ] **Implementazione minima:** estendere il loop breve T a budget completo, early stop, metriche JSONL, config/environment/selection salvati, best/last separati. Ripartire da zero sui soli train; niente train+validation merge o warm-start dal vincitore T. Budget epoch dal sampler, non da RLE coverage. Valutazione test solo tramite comando esplicito dopo selezione.
- [ ] **GREEN verifica:** stesso comando con fixture, poi `python3 -m drl.IL.bc_v1 train --config configs/bc_v1/training.json --cache artifacts/bc_v1/v1.0.0 --output artifacts/bc_v1/runs/smoke-v1 --max-epochs 2` sul train/validation reali. L'override viene salvato nel config della run; nessuna ricerca iperparametri sul test.
- [ ] **Completamento:** training seleziona un checkpoint con validation e produce report riproducibili; nessuna dipendenza Lab per logging.

## K2 — Checkpoint, resume e gate finale

**File:** modificare `training.py`, `__main__.py`, `test_training.py`; creare `test_checkpoint.py`.

**Interfacce:** `save_checkpoint(path: Path, model: BCGNN, optimizer: Optimizer, state: Mapping, fingerprints: Mapping) -> None`, `load_checkpoint(path: Path, expected_fingerprints: Mapping, device: str) -> Mapping`. Scheduler state soltanto se effettivamente configurato.

- [ ] **RED test:** run di 6 batch e run3+resume3 producono stessi pesi/optimizer/draw/step su CPU deterministica; checkpoint a metà epoch conserva sampler e RNG; mismatch source/filter/schema/spec/split/normalizzatori rifiutato prima dell'update; errore di scrittura non corrompe checkpoint precedente; best non sovrascritto da last peggiore.
- [ ] **RED verifica:** `python3 -m pytest drl/IL/tests/bc_v1/test_checkpoint.py -q`.
- [ ] **Implementazione minima:** serializzazione atomica dei campi spec §11, verifica fingerprint, ripristino RNG dopo costruzione modello/optimizer e prima del prossimo draw. Num_workers0 iniziale evita prefetch non serializzato. Resume oltre un confine di batch non supportato nella v1.
- [ ] **GREEN verifica:** comando precedente, poi `python3 -m pytest drl/IL/tests/bc_v1 -q`; regressioni Canonical esistenti `PYTHONPATH=drl/IL python3 -m unittest discover -s drl/IL -p 'test_canonical_dataset*.py'`; `git diff --check`. Il controllo SHA conferma che Canonical/raw/compact sono invariati.
- [ ] **Completamento:** tutta la checklist spec §12 verde con evidenza; spec e piano aggiornati allo stato osservato. Commit/integrations solo con autorizzazione successiva, non come parte automatica del task.

Dopo il GREEN K2, avviare il training finale con la selection T verificata: `python3 -m drl.IL.bc_v1 train --config configs/bc_v1/training.json --selection artifacts/bc_v1/tuning/grid-v1/selection.json --cache artifacts/bc_v1/v1.0.0 --output artifacts/bc_v1/runs/final-v1`. Conservare config effettiva, seed, metriche e checkpoint; soltanto dopo la scelta del best checkpoint è ammessa la valutazione test esplicita. Non eseguire questi comandi durante la presente revisione.

## Prima azione dopo approvazione

Soltanto dopo l'approvazione dell'audit manuale dei 22 ALIGN e delle correzioni spec/piano, eseguire A1: creare `test_contract.py` e le fixture minime, osservare il RED con `python3 -m pytest drl/IL/tests/bc_v1/test_contract.py -q`, quindi creare __init__ leggero e contratto stdlib fino al GREEN, inclusa la skill generica enabled di default. Nessun adapter, filtro, tensore o trainer prima di questo gate. **A1 non è iniziato.**

## Self-review documentale

- [x] Copertura: sezioni spec 1–12 assegnate ai task A–K e T; rollout/recorder/Lab esplicitamente esclusi.
- [x] Contratto prima del trainer; interfacce consumate definite nei task precedenti.
- [x] Ogni task contiene RED, file, implementazione minima, comando GREEN e criterio di completamento.
- [x] Sette Review Focus assegnati a test concreti, inclusi SETTLE e tuning senza test leakage.
- [x] Lista file limitata a nuovi file BC-v1/config/test; nessuna modifica alle sorgenti protette.
- [x] Nessun ciclo TDD eseguito in questa fase. Questo self-review è documentale e non certifica test passati.
