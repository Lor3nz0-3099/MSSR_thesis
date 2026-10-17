# BC-v1 Source Frame Bridges Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrare bridge sorgente certificati e causal-topology nel prepare fisico G1, impedendo perdite geometriche silenziose e separando il successivo fit F2 train-only.

**Architecture:** Un provider certifica i frame sorgente; E1 resta l'unica autorità di trasformazione. La reconciliation causal-topology resta separata, dopo la decodifica typed e prima di D2/E1/E2. G1 pubblica una cache fisica completa e validata prima del lifecycle F2 separato.

**Tech Stack:** Python 3.10+, pytest, pipeline BC-v1 corrente, frozen dataclasses, JSON/hashlib, NumPy, Torch CPU lazy; riuso di `SourceFrameTransform`, `PhysicalFrameConfig`, `cache_fingerprint`, `file_hash`, codec temporal audit e normalizzazione esistenti. Nessuna nuova dipendenza hardware.

**Spec:** [Source Frame Bridges v1.0.0](../specs/2026-10-12-bc-v1-source-frame-bridges-v1.0.0.md), APPROVATA dall'utente. Vincoli di integrazione: [coalesced causal-topology spec](../specs/2026-10-11-bc-v1-coalesced-supervision-causal-topology-v1.0.0.md) e [piano causal-topology](2026-10-11-bc-v1-coalesced-supervision-causal-topology-implementation-v1.0.0.md). Questo piano supplementare dettaglia il restante Task 4+; non riscrive i documenti precedenti né ripete la milestone causal-topology Task 1–3 già implementata. I dieci task sotto hanno numerazione propria.

## Global Constraints

- Canonical v6 immutabile: manifest SHA `31765272b6ecd7506c54b25e214b75d7b1b28691803ddfc286dafcce8b4ec562`. Nessun Canonical v7, rewrite RAW/DATASET_COMPACT/v6 o cambio di label/indice.
- Un PolicySample per transition BC-valid; 73 episodi / 119600 sample. F1: train 49/78302, validation 12/22762, test 12/18536; sette expert button BC=0 esclusi. Nessuno split/drop dei 309 coalesced sequenziali o dei 21 ambigui.
- Nessun widening D2. Configuration_goal, stage, FSM/control_state, target graph/morphology, G_t+1/futuro, label/mask/reward/success e provenance non sono learned input. Nav2 route/waypoint nominali non sono input né coni fisici.
- E1 unica autorità named/global spatial→physical frame; D1/D2 conservano semantica sorgente. Connector module-body, edge B-in-A REQUIRED e intrinsic/resource-local restano locali; nessun bridge body_local generico, root/min-ID/PCA o nuovo schema learnable.
- Unknown/uncertified frame fail-closed con reason. Map→world identità soltanto per il contratto legacy RC certificato, mai dal solo nome map/backend/morfologia/episode id. Stage_local deriva soltanto dal frame fisico CURRENT, senza stage/FSM/label/target/futuro.
- Physical-equivalence dedup E2 invariata; nessuna componente mancante inventata o duplicazione di oggetti. Causal-topology e frame bridging restano responsabilità indipendenti.
- Case 1189 invariato: previous-bucket detach02 ammesso, current-bucket detach05 differito, 7→6 edge, UNDOCK05 e ALIGN02 entrambi rappresentabili. Exact disconnect-only, nessuna aggiunta DOCK o pose/joint repair.
- 1110 target strutturali, inclusa GRAVITY_SETTLE enabled, e 849907 target totali invariati; 20 stale ALIGN e 17 companion conservati. Temporal kinds 6174/113096/309/21 sono audit-only.
- Cache fisica completa/validata PRIMA di F2; fit solo train, 49/78302, engine e policy matematiche invariati. Nessun normalizer fingerprint futuro nell'identità fisica G1.
- Nessun legacy cleanup, FreeBOT implementation, G2/model/training, runtime/recorder/ROS/Isaac/Lab o WAIT/no-dispatch.
- Task 1–9 sono implementazione e preflight economico; **Task 10 richiede una nuova approvazione esplicita dell'utente** dopo review delle evidenze. L'approvazione di questo piano non approva il run costoso.
- Nessuno staging/commit/push/reset/restore/clean/switch. Prima di ogni task auditare status/diff; modificare solo i file elencati, non cancellare untracked estranei. Anche un successivo staging richiede autorizzazione esplicita e MUST essere limitato ai file scoped.

## Review Focus

| Rischio | Test concreto e task proprietario |
|---|---|
| Map arbitrario che eredita l'eccezione legacy | Task 3: `test_arbitrary_map_and_other_backend_never_inherit_legacy_identity`. |
| Stage_frame missing/malformed/nonfinite | Task 2: `test_stage_local_bad_current_frame_fails_closed`, parametrizzato per ogni caso. |
| Due prove discordanti per stesso frame/scope | Task 1: `test_conflicting_frame_scope_certificates_rejected`; Task 2 verifica le sorgenti correnti discordanti. |
| Alias prima invalido reso valido e deduplicato | Task 7: `test_stairs_1725_bridge_dedup_preserves_physical_objects_and_32_relations`. |
| D2 valido→E1 invalido senza reason approvata | Task 5: `test_valid_d2_unexpected_e1_invalidation_blocks_publication`. |

## Stato reale, call site e prerequisiti

Ispezione 2026-10-12 sul branch `wip/t8-c05-human-teleop-20260921`, HEAD `b50c543731e88ef09a18db7b68ad461056795749`. I riferimenti di riga indicano lo snapshot, i simboli sono l'autorità per l'implementazione.

| Implementazione attuale | Delta richiesto |
|---|---|
| `dataset.prepare_dataset`, righe 311–323: encode_state→D1→D2→E1 con `PhysicalFrameConfig()`→E2 | Inserire reconciliation dopo encode_state e bridge resolution dopo D2; entrambi prima di feature/candidate/representability. |
| `dataset.CurrentCandidateCache.get`, 89–99; `_representable`, 204–239 | Usare unicamente il grafo riconciliato per i candidati; label codec continua a ricevere published graph. |
| `dataset._code_hashes`, 60–64; identity, 274–279; manifest reader, 141–171 | Includere core/provider/diagnostics e audit identities; distinguere compatibilità globale e bridge per sample. |
| `dataset.prepare_dataset`, 357–390 e 410–431 | Eliminare fit, pubblicazione/rollback normalizer e sua dipendenza nell'identità G1; spostarli nel wrapper F2. |
| `__main__.main/run_prepare`, 7–104 | CLI oggi solo prepare, carica policy F2 e stampa normalizer. Introdurre composition dei provider, preflight mirato e comando fit separato. |
| `HistoricalSmoresDisconnectProvider.for_bucket(source_refs, published_graph)` | Esiste e richiede run/file/prefix esplicitamente certificati; non è ancora collegato a G1. Non passargli il row completo. |
| `temporal_supervision.read_temporal_audit` | Codec già presente; artifact completo verificato non presente nel path ufficiale né nei due path /tmp storici controllati. |
| E1 `SourceFrameTransform` / `PhysicalFrameConfig.transforms` | Primitive matematiche già corrette; nessuna duplicazione o eccezione SMORES dentro E1. |
| E2 `_deduplicate_environment`, `_module_environment_relations` | Canonicalizzazione e nearest-object/ties già implementati; aggiungere solo tracing non learnable dove necessario. |

`artifacts/bc_v1/cache/g1_v1_blocked/structural_audit.json` contiene **20 casi / 10 episodi**, non l'indice di tutti i 1110 target. La fixture `causal_topology.v1.json` certifica 1189. Il vecchio `test_dataset_counts.v6_audit` esegue il censimento completo: MUST NOT essere invocato nei task economici. Le fixture E2 esistenti usano altri sei indici; non sono automaticamente i sette record della spec.

Prima del gate reale servono il temporal inventory originale completo verificato e un catalogo source-stream→runtime-run/hash senza ambiguità. Recuperare una copia verificata degli audit; se indisponibile, STOP e richiedere autorizzazione per il read-only re-audit necessario, come prescritto dalla spec causal-topology §14. Nessuna riclassificazione euristica, fallback joint o associazione run per solo nome simile. Questi prerequisiti non bloccano i test sintetici Task 1–8, ma bloccano l'accettazione Task 9/10.

## File Structure / Ownership

Lista chiusa dei file di implementazione previsti; nessuno viene creato/modificato in questa consegna, eccetto questo piano.

| Azione | File | Unica responsabilità |
|---|---|---|
| Create | `drl/IL/bc_v1/source_frame_bridges.py` | Contratto generico dei bridge certificati e relativa identità provider. |
| Create | `drl/IL/bc_v1/smores_source_frames.py` | Certificazione/routing della sorgente SMORES e risoluzione stage_local/map. |
| Create | `drl/IL/bc_v1/pipeline_validation.py` | Trace, reason vocabulary e validazione fail-fast della derivazione fisica. |
| Create | `drl/IL/bc_v1/structural_preflight.py` | Preflight rappresentativo/strutturale che riusa la derivazione G1. |
| Create | `drl/IL/bc_v1/normalization_prepare.py` | Lifecycle fit/receipt F2 da cache fisica già certificata. |
| Create | `configs/bc_v1/source_frames.json` | Policy versionata e contratto legacy esplicitamente certificato. |
| Modify | `drl/IL/bc_v1/dataset.py` | Derivazione, persistenza e lettura della cache fisica G1. |
| Modify | `drl/IL/bc_v1/__main__.py` | Composition source/provider e dispatch dei comandi offline. |
| Modify | `drl/IL/bc_v1/smores_disconnect_evidence.py` | Join degli evidence descriptors e lettura efficiente dei prefix già certificati. |
| Modify | `drl/IL/bc_v1/temporal_supervision.py` | Import verificato dell'inventario temporale preesistente. |
| Modify | `drl/IL/bc_v1/task_context.py` | Observer D1 non trasformativo delle quantità correnti selezionate/escluse. |
| Modify | `drl/IL/bc_v1/policy_inputs.py` | Observer D2 della selezione, senza modifica della whitelist. |
| Modify | `drl/IL/bc_v1/feature_schema.py` | Observer E2 di emissione/canonicalizzazione, senza variazione numerica/schema. |
| Modify | `configs/bc_v1/prepare.json` | Configurazione serializzabile dei provider e path dei lifecycle separati. |
| Create | `drl/IL/tests/bc_v1/test_source_frame_bridges.py` | Contratto bridge generico. |
| Create | `drl/IL/tests/bc_v1/test_smores_source_frames.py` | Certificazioni SMORES e casi negativi. |
| Create | `drl/IL/tests/bc_v1/test_pipeline_validation.py` | Loss/invalidation fault injection e reason taxonomy. |
| Create | `drl/IL/tests/bc_v1/test_source_frame_regressions.py` | Sette regressioni reali geometriche/relazionali. |
| Create | `drl/IL/tests/bc_v1/test_causal_topology_counts.py` | Venti stale ALIGN, companion e preflight condiviso. |
| Create | `drl/IL/tests/bc_v1/test_normalization_prepare.py` | Fit/receipt separato e holdout tripwire. |
| Create | `drl/IL/tests/bc_v1/fixtures/source_frames.v1.json` | Snapshot minimi dei sette record e relative identity/source evidence. |
| Modify | `drl/IL/tests/bc_v1/fixtures/causal_topology.v1.json` | Estensione oracle dei 20, mantenendo la prova 1189. |
| Modify | `drl/IL/tests/bc_v1/test_dataset.py` | Wiring, physical cache e transaction regressions. |
| Modify | `drl/IL/tests/bc_v1/test_dataset_counts.py` | Gate ufficiali fisico/F2 separati senza silent skip. |
| Modify | `drl/IL/tests/bc_v1/test_smores_disconnect_evidence.py` | Source/run join e prefix read/cache regressions. |
| Modify | `drl/IL/tests/bc_v1/test_temporal_supervision.py` | Import/certificazione e one-to-one temporal audit. |

Adapter, generic robot contract, `features.py`, candidates, split, normalization engine/policy, RobotSpec, feature config, training config e documenti approvati MUST restare invariati. Gli observer aggiungono solo keyword opzionali `audit_sink=None`; il risultato D1/D2/E2 e le firme esistenti senza keyword conservano comportamento e tipi. Questa estensione limitata serve a rendere visibili loss e dedup senza un secondo algoritmo di feature extraction.

Artifact futuri: `artifacts/bc_v1/audits/{temporal_supervision_v1.json,causal_sources_v1.json}`, preflight `artifacts/bc_v1/preflight/source_frames_v1/`, cache `artifacts/bc_v1/cache/g1_v1/` con audit temporal/topology e frame per episodio, F2 `artifacts/bc_v1/normalization/f2_v1/{normalizers.json,fit_context.json}`. Gli artifact blocked restano intatti. Nessun artifact è prodotto oggi.

## Interfacce condivise e ordine normativo

Nomi proposti da implementare; i tipi esistenti citati sono già presenti. Tutti i nuovi record sono frozen, con tuple/scalari immutabili; non inserire Mapping mutabili dentro record dichiarati immutabili.

- `BridgeEvidence(path: str, sha256: str)` e `SourceContractCertificate(contract_id: str, evidence: tuple[BridgeEvidence,...])`: certificazione source/provider, fuori dalla policy.
- `FrameSourceContext(backend_id: str, source_identity: tuple[tuple[str,str],...], certificates: tuple[SourceContractCertificate,...])`: routing certificato della sorgente corrente; nessuna label/stage/FSM. Il source identity lega anche i digest delle fonti esterne correnti.
- `BridgeResolution(source_frame: str, target_frame: str, scope_id: str, transform: SourceFrameTransform | None, certified: bool, reason: str, contract_id: str | None, evidence: tuple[BridgeEvidence,...])`. `transform=None` significa identità implicitamente certificata solo per world→world; con `certified=False` significa unresolved. Scope è quello fisico corrente, non stage embedding.
- `SourceFrameBridgeBatch(resolutions: tuple[BridgeResolution,...], provider_identity: tuple[tuple[str,str],...])`; `SourceFrameProvider.resolve(inputs: PolicyInputs, source: FrameSourceContext) -> SourceFrameBridgeBatch`, `identity() -> Mapping`. `frame_config_for(batch: SourceFrameBridgeBatch) -> PhysicalFrameConfig`: solo bridge certificati, graph/axes world, world identity non duplicata in transforms.
- `PipelineDiagnostic(stage: str, source_scope: str, source_name: str, classification: str, reason: str)`; `PipelineTrace.record(stage: str, source_scope: str, source_name: str, present: bool, valid: bool, reason: str, represented_by: tuple[str,...] = ()) -> None`. Il trace riguarda solo il sample corrente.
- `PreparedPhysicalInputs(published_graph: StateGraph, inputs: CachedInputs, geometry: FrameGeometryView, features: FeatureView, bridges: SourceFrameBridgeBatch, diagnostics: tuple[PipelineDiagnostic,...], topology_audit: tuple[tuple[str,str],...])`: risultato transitorio completo attraverso E2, non nuovo PolicyInputs/CachedSample. `features MUST be inputs.features`: un'unica FeatureView fisica completata, condivisa per identità. Published graph serve soltanto al label codec separato; topology_audit conserva coppie nome/valore JSON canonico lossless, riusando la serializzazione esistente. Diagnostics è vuota prima dell'integrazione Task 5.
- `dataset.derive_physical_inputs(row: Mapping, adapter: RobotAdapter, config: Mapping, candidate_cache: CurrentCandidateCache) -> PreparedPhysicalInputs` MUST eseguire adapter typed graph → reconciliation → D1 → D2 → bridge resolution → E1 → E2, con una sola chiamata a `extract_features` per derivazione. Dopo E2, il helper usa `candidate_cache.get(reconciled_graph)` per costruire `CachedInputs(policy_inputs, features, candidates)`; restituisce la derivazione fisica completa. Representability consuma poi `derived.inputs.candidates` e il bundle del label codec separato; nessun consumer riesegue E2. `prepare_dataset(source: Path, output: Path, adapter, config: Mapping) -> Mapping` resta la firma pubblica.

Ordine: **Canonical/current source → adapter typed published graph → provider disconnect/reconciliation → D1 → D2 → source-frame resolve → E1 → E2 → candidate extraction/representability → cache fisica completa → F2 separato**. Metadata readers/provider ricevono soltanto i sottoinsiemi di routing/prova ammessi; label codec resta un ramo separato con published graph. Nessun repair label-driven.

`configs/bc_v1/prepare.json` MUST contenere solo configurazione serializzabile dei provider, path, policy IDs/revisioni e artifact locations; MUST NOT contenere callable Python o oggetti runtime. `__main__.py` legge questi asset e costruisce `topology_evidence_provider`, `source_frame_provider`, `frame_source_reader` e temporal audit/provider objects. Soltanto la composition in memoria li inietta nel runtime `config` passato a dataset/`derive_physical_inputs`, preservando la dependency injection per i test.

## Procedura, comandi e protezione del workspace

Comandi dalla root con `BC_PYTHON=/home/lorenzo/.local/share/mssr-bc-v1/venv/bin/python`, `BC_TEST_DIR=drl/IL/tests/bc_v1`, `BC_CORE_DIR=drl/IL/bc_v1`. Tutti i comandi nei task sono **futuri**, non risultati di questa consegna.

Ogni task: audit `git status --short --untracked-files=all` e `git diff --stat` → RED pertinente (API/asserzione mancante, non errore di ambiente/sintassi) → minimo GREEN → regressione mirata → diff/status scoped → STOP/reviewer. Nessun commit. Per nuovi file untracked, controllare anche il contenuto/whitespace perché il normale git diff non li include.

Preservare questi sette tracked dirty file: `mssr_ws/src/mssr_expert/mssr_expert/{execution/parallel_assembly_executor.py,nodes/smores_parallel_self_assembly_node.py}`, `mssr_ws/src/mssr_expert/test/test_parallel_assembly_executor.py`, `scripts/dataset_tools/build_expert_v1.py`, `scripts/smores_ep/{run_button_expert_to_ik.py,src/smores_ep/control/button_ik_math.py,tests/test_button_ik_source_contract.py}`. Preservare anche gli88 untracked preesistenti, modificando esclusivamente quelli scoped dal task.

Nessun multiprocessing diagnostico. Leggere un episodio/record rappresentativo per volta, liberare graph/metadata dopo il caso; memoizzare soltanto identity/offset/proof minimi con signature verification. La regressione MUST NOT accumulare tutte le feature di 119600 sample. I full scans della fixture B2 sono esclusi dai gruppi economici. Misurare RSS/tempo del preflight; budget e interruzioni sono espliciti, mai retry automatici o pass dopo truncation. Gli artifact /tmp sono evidenza per estrarre fixture verificate, non dipendenze permanenti.

### Task 1: Contratto generico dei bridge certificati

**Purpose:** Introdurre result/provider e unresolved diagnostics attorno alle primitive E1 già corrette, senza comportamento SMORES.

**Files:**

- Create: drl/IL/bc_v1/source_frame_bridges.py.
- Test: drl/IL/tests/bc_v1/test_source_frame_bridges.py.

**Interfaces:** Consumes: `SourceFrameTransform`, `PhysicalFrameConfig`, `PolicyInputs`. Produces: tipi bridge/context/provider sopra e `PipelineDiagnostic` (record reason generico, usato dal validator Task 5); `frame_config_for(batch: SourceFrameBridgeBatch) -> PhysicalFrameConfig`; `bridge_policy_fingerprint(policy: Mapping) -> str`.

- [ ] Scrivere `test_world_identity_reuses_e1_without_redundant_transform`, `test_unknown_frame_unresolved_with_reason`, `test_conflicting_frame_scope_certificates_rejected`, `test_bridge_contract_frozen_and_not_policy_input`, `test_bridge_policy_fingerprint_deterministic_sensitive`. Assert identity E1, unresolved explicito, conflitto per stesso frame/scope senza scelta arbitraria, nessun import hardware/Torch o campo in PolicyInputs.
- [ ] RED: `$BC_PYTHON -B -m pytest $BC_TEST_DIR/test_source_frame_bridges.py::test_world_identity_reuses_e1_without_redundant_transform -q`; FAIL sull'API/contratto mancante.
- [ ] Implementare i tipi, validazioni e `frame_config_for` in `source_frame_bridges.py`: riusare la rotazione propria/finite validation E1; non ammettere un arbitrary named frame nel codec E1 chiuso o introdurre map identity generica.
- [ ] GREEN: stesso node; poi `$BC_PYTHON -B -m pytest $BC_TEST_DIR/test_source_frame_bridges.py $BC_TEST_DIR/test_features.py -q`, tutti PASS senza dataset census.
- [ ] `git diff --check -- $BC_CORE_DIR/source_frame_bridges.py $BC_TEST_DIR/test_source_frame_bridges.py`; `git status --short --untracked-files=all`; STOP/review. Completamento: contratto deterministico generico, local/intrinsic fuori resolver.

### Task 2: Provider SMORES stage_local corrente

**Purpose:** Certificare la trasformazione planare del task corrente senza trasformare preventivamente D1/D2.

**Files:**

- Create: drl/IL/bc_v1/smores_source_frames.py.
- Test: drl/IL/tests/bc_v1/test_smores_source_frames.py.

**Interfaces:** Consumes: Task 1 e `inputs.context.static_geometry`. Produces: `SmoresSourceFrameProvider(repo_root: Path, policy: Mapping)` con `resolve/identity` del Protocol; `resolve_stage_local(inputs: PolicyInputs, source: FrameSourceContext) -> BridgeResolution` nel medesimo modulo.

- [ ] Scrivere `test_stage_local_1725_exact_current_frame`, `test_stage_local_bad_current_frame_fails_closed`, `test_stage_local_conflicting_current_evidence_rejected`, `test_stage_local_ignores_stage_fsm_labels_and_future`. Parametri negativi: origin/yaw assenti, width errata, nonfinite, valid=false e prove discordanti. Tripwire su campi vietati; nessun roll/pitch inventato.
- [ ] RED: `$BC_PYTHON -B -m pytest $BC_TEST_DIR/test_smores_source_frames.py::test_stage_local_1725_exact_current_frame -q`; FAIL perché resolve stage_local non esiste.
- [ ] Implementare `resolve_stage_local` dai soli world.task_frame origin/yaw validi, costruendo `SourceFrameTransform` e reason/proof. Expected: Rz(-π/2)·[-0.7,0,0]+[1.7699752221047802,-2.17605937116128,0] = [1.7699752221047802,-1.4760593711612797,0], atol=1e-9.
- [ ] GREEN: stesso node; regressione `$BC_PYTHON -B -m pytest $BC_TEST_DIR/test_smores_source_frames.py $BC_TEST_DIR/test_source_frame_bridges.py -q`.
- [ ] `git diff --check -- $BC_CORE_DIR/smores_source_frames.py $BC_TEST_DIR/test_smores_source_frames.py`; status completo; STOP/review. Completamento: missing/malformed fail-closed e bridge indipendente dai target.

### Task 3: Certificazione legacy RC map

**Purpose:** Attivare l'identità map→world soltanto sotto il source contract storico approvato.

**Files:**

- Modify: drl/IL/bc_v1/smores_source_frames.py — resolve/source certificate reader.
- Create: configs/bc_v1/source_frames.json.
- Test: drl/IL/tests/bc_v1/test_smores_source_frames.py.

**Interfaces:** Consumes: Task 1/2, exact source references, current D1 metadata e manifest immutabili. Produces: `read_frame_source_context(provenance: Mapping, metadata: Mapping, *, repo_root: Path, policy: Mapping) -> FrameSourceContext`; `resolve_legacy_rc_map(source: FrameSourceContext) -> BridgeResolution`; estensione `SmoresSourceFrameProvider.resolve`.

- [ ] Scrivere `test_certified_legacy_rc_map_identity`, `test_arbitrary_map_and_other_backend_never_inherit_legacy_identity`, `test_legacy_identity_requires_all_pinned_source_evidence`, `test_teleop_world_navigation_never_requests_legacy_map`. Mutare separatamente nome map, morphology, episode id, generic Nav2 e backend FreeBOT: nessuna attivazione. Mismatch dei digest/declarations MUST fallire.
- [ ] RED: `$BC_PYTHON -B -m pytest $BC_TEST_DIR/test_smores_source_frames.py::test_certified_legacy_rc_map_identity -q`; manca la certificazione legacy.
- [ ] Implementare il source reader e la policy: pin del Canonical manifest e del suo compact manifest `253be8ef240f23f80c6174239e83a9b23e76282bd3ffc204155253b7e4fb7837`; join delle exact source refs con l'entry RC del manifest compact, poi episode/phase manifest e environment_geometry con hash verificati. Leggere solo dichiarazioni robot/source/course/world/map e piattaforma; nessun target/stage/route/success. `seed-005100` prova environment digest `360be45bab514717098e04ee627e4b017353b9ca4c0179f6f8199ec4f947e562`, course WORLD centro [1.1,0,-0.01], size [4.8,3.2,0.02], bounds MAP [-1.3,3.5,-1.6,1.6].
- [ ] Registrare in `source_frames.json` il contratto storico approvato e il digest effettivo della dichiarazione launch map→odom/Isaac-world; la combinazione frozen-source membership + evidence chain è il certificato. Il solo launch corrente/rettangolo coincidente non basta; nessuna allowlist di indici. Nuove sorgenti non ereditano il certificato.
- [ ] GREEN: stesso node; regressione `$BC_PYTHON -B -m pytest $BC_TEST_DIR/test_smores_source_frames.py -q`, usando fixture minime e sidecar piccoli, non tutte le righe RAW.
- [ ] `git diff --check -- $BC_CORE_DIR/smores_source_frames.py configs/bc_v1/source_frames.json $BC_TEST_DIR/test_smores_source_frames.py`; status; STOP/review. Completamento: routing certificato fuori dai learned input.

### Task 4: Wiring della derivazione fisica con entrambi i provider

**Purpose:** Collegare causal-topology già implementata e frame resolution nell'ordine approvato; coincide con l'integrazione finora fermata, senza full prepare.

**Files:**

- Modify: drl/IL/bc_v1/dataset.py — prepare 311–340, derive_physical_inputs; drl/IL/bc_v1/__main__.py — run_prepare/runtime composition; drl/IL/bc_v1/smores_disconnect_evidence.py — descriptors/prefix reads; drl/IL/bc_v1/temporal_supervision.py — import; configs/bc_v1/prepare.json — serializable provider configuration/paths/IDs.
- Test: drl/IL/tests/bc_v1/test_dataset.py; drl/IL/tests/bc_v1/test_smores_disconnect_evidence.py; drl/IL/tests/bc_v1/test_temporal_supervision.py.

**Interfaces:** Consumes: `HistoricalSmoresDisconnectProvider.for_bucket(...) -> DisconnectEvidenceBatch`, `reconcile_topology(...)`, bridge provider e configurazione runtime in memoria. Produces: `derive_physical_inputs(...) -> PreparedPhysicalInputs` con E2 completata e `derived.features is derived.inputs.features`; candidate extraction usa il grafo riconciliato dopo E2, representability usa i candidati restituiti. La composition CLI costruisce e inietta solo nel runtime config `config['topology_evidence_provider'](provenance: Mapping, published_graph: StateGraph) -> DisconnectEvidenceBatch | None`, `config['source_frame_provider']`, `config['frame_source_reader'](provenance, metadata) -> FrameSourceContext`, `config['temporal_audit']`; il JSON contiene esclusivamente i loro path/configuration/IDs serializzabili. New `read_disconnect_source_refs(provenance: Mapping, *, repo_root: Path, source_catalog: Mapping) -> Mapping | None` nel provider; `import_certified_temporal_audit(path: Path, source_identity: Mapping, episode_metadata: Mapping) -> tuple[TemporalSupervisionAudit,...]` nel codec temporal.

- [ ] Scrivere `test_derive_orders_adapter_topology_d1_d2_bridges_e1_e2`, `test_composed_providers_ignore_labels_targets_and_future`, `test_composed_1189_keeps_both_targets_and_7_to_6_edges`, `test_one_sample_retains_joint_sequential_and_ambiguous_targets`, `test_repeated_bucket_reads_verify_once_without_future_payloads`. Spy ordine completo e una sola chiamata E2; `derived.features is derived.inputs.features`; interning occupied/free distinto; label codec published graph, candidati reconciled graph; temporal kind non altera feature/target. Verificare round-trip JSON degli asset e costruzione/iniezione dei callable solo nel runtime config.
- [ ] RED: `$BC_PYTHON -B -m pytest $BC_TEST_DIR/test_dataset.py::test_derive_orders_adapter_topology_d1_d2_bridges_e1_e2 -q`; manca il derivation hook/provider wiring.
- [ ] Implementare `derive_physical_inputs` attraverso l'unica chiamata E2 e restituire features/geometry/bridges già completati, con CachedInputs costruito dai candidati del grafo riconciliato. Passare al topology provider solo provenance/exact current source pointers. CLI costruisce source/run descriptors dal catalogo certificato `causal_sources_v1.json`, verificando stream hash e run identity, mai da action_t o candidate failure. Solo un bucket senza primitive dispatch corrente osservabile e senza obbligo di reconciliation secondo il source contract produce un no-op audit non-applicable. Se il dispatch esiste ma boundary/run/proof richiesti mancano o sono ambigui, MUST fallire: non usare il no-op come fallback. Restituire published_graph per il label codec, senza decodificarlo due volte.
- [ ] Integrare il codec temporal esistente con inventory/source receipt verificati; controllare enum/counts e successivamente ogni key contro le righe effettive del prepare. Nessun classifier nuovo. Memoizzare solo source hash/offset/event identity ammessi per episodio con stat signature checks; non rileggere/hashare il RAW gigante per ciascun bucket, non interpretare future payload, non conservare graph RAW nel provider. La policy `for_bucket`/cutoff resta invariata.
- [ ] GREEN: stesso node; regressione `$BC_PYTHON -B -m pytest $BC_TEST_DIR/test_dataset.py::test_composed_1189_keeps_both_targets_and_7_to_6_edges $BC_TEST_DIR/test_smores_disconnect_evidence.py $BC_TEST_DIR/test_temporal_supervision.py -q`. Solo derivazioni/fixture, nessun prepare reale.
- [ ] `git diff --check -- $BC_CORE_DIR/dataset.py $BC_CORE_DIR/__main__.py $BC_CORE_DIR/smores_disconnect_evidence.py $BC_CORE_DIR/temporal_supervision.py configs/bc_v1/prepare.json $BC_TEST_DIR/test_dataset.py $BC_TEST_DIR/test_smores_disconnect_evidence.py $BC_TEST_DIR/test_temporal_supervision.py`; status; STOP/review. Completamento: ordine e 1189 preservati senza fondere i provider.

### Task 5: Trace e fail-fast delle perdite silenziose

**Purpose:** Rendere obbligatoria una spiegazione verificabile per loss/invalidation prima della pubblicazione fisica.

**Files:**

- Create: drl/IL/bc_v1/pipeline_validation.py.
- Modify: drl/IL/bc_v1/task_context.py — build_task_view; drl/IL/bc_v1/policy_inputs.py — select_policy_inputs; drl/IL/bc_v1/feature_schema.py — extract_features/emission/dedup; drl/IL/bc_v1/dataset.py — derive validation.
- Test: drl/IL/tests/bc_v1/test_pipeline_validation.py; drl/IL/tests/bc_v1/test_dataset.py.

**Interfaces:** Consumes: current metadata allowlist, D1 context/observation, D2 inputs, bridges, E1 geometry e l'unica E2 FeatureView calcolata da derive_physical_inputs, spec/schema e PipelineDiagnostic del Task 1. Produces: `PipelineTrace`; `validate_physical_pipeline(inputs: PolicyInputs, geometry: FrameGeometryView, features: FeatureView, bridges: SourceFrameBridgeBatch, trace: PipelineTrace, spec: RobotSpec, schema: Mapping) -> tuple[PipelineDiagnostic,...]`, che solleva ValueError per unexpected. Derive finalizza `PreparedPhysicalInputs.diagnostics` usando questi risultati già calcolati; il validator MUST NOT chiamare extract_features o creare una seconda FeatureView. Observer opzionale `audit_sink=trace.record`: aggiungere `*, audit_sink: Callable[..., None] | None = None` a `build_task_view(row: Mapping, metadata: Mapping, config: Mapping) -> tuple[TaskContext, ObservationBundle]`, `select_policy_inputs(graph: StateGraph, observation: ObservationBundle, context: TaskContext) -> PolicyInputs`, `extract_features(inputs: PolicyInputs, geometry: FrameGeometryView, spec: RobotSpec, schema: Mapping) -> FeatureView`; nessun cambiamento delle positional arguments esistenti.

- [ ] Scrivere `test_valid_d2_unexpected_e1_invalidation_blocks_publication`, `test_observers_do_not_change_d1_d2_e2_outputs`, `test_dedup_is_traced_as_equivalence_not_loss`, `test_coordinate_catalog_limits_are_expected_schema_validity`, `test_pipeline_faults_have_exact_reason_and_never_publish`. Fault injection D1/D2 omission, E1 invalidation dopo bridge certificato, E2 valid source→invalid/zero-default, relation deletion, nonfinite/frame mismatch; legitimate centroid zero MUST pass.
- [ ] RED: `$BC_PYTHON -B -m pytest $BC_TEST_DIR/test_pipeline_validation.py::test_valid_d2_unexpected_e1_invalidation_blocks_publication -q`; perdita ancora accettata/API mancante.
- [ ] Implementare observers: D1 registra quantità ammesse disponibili prima/dopo selezione, senza iterare target/route subtrees; D2 registra inclusioni/esclusioni secondo la whitelist esistente; E2 registra righe emesse e equivalenze scartate/mantenute. Validator verifica coverage/validità/frame/codec contro E1, local/intrinsic preservation e relazioni attese da oggetti validi, task policy, budget/ties correnti; non conta una riga per alias. Non usare label per attese geometriche.
- [ ] Implementare esattamente i 13 reason: `EXPECTED_SOURCE_MISSING`, `INTENTIONAL_POLICY_EXCLUSION`, `EXPECTED_TRANSIENT_OR_NOT_APPLICABLE`, `KNOWN_FRAME_BRIDGE_MISSING_MAP`, `KNOWN_FRAME_BRIDGE_MISSING_STAGE_LOCAL`, `EXPECTED_SCHEMA_VALIDITY`, `UNEXPECTED_D1_LOSS`, `UNEXPECTED_D2_LOSS`, `UNEXPECTED_E1_INVALIDATION`, `UNEXPECTED_E2_FEATURE_INVALIDATION`, `UNEXPECTED_RELATION_LOSS`, `FRAME_SEMANTICS_UNKNOWN`, `OTHER_UNEXPLAINED`. KNOWN non permette di omettere un bridge richiesto certificato; optional missing/unknown richiede reason prevista dalla policy, non un catch-all. Unexpected/OTHER blocca staging publication.
- [ ] GREEN: stesso node; regressione `$BC_PYTHON -B -m pytest $BC_TEST_DIR/test_pipeline_validation.py $BC_TEST_DIR/test_dataset.py::test_late_failure_no_publication_or_normalizer -q`. Trace sample-wise, soli contatori per episodio/globali, nessun diagnostic feature census in RAM.
- [ ] `git diff --check -- $BC_CORE_DIR/pipeline_validation.py $BC_CORE_DIR/task_context.py $BC_CORE_DIR/policy_inputs.py $BC_CORE_DIR/feature_schema.py $BC_CORE_DIR/dataset.py $BC_TEST_DIR/test_pipeline_validation.py $BC_TEST_DIR/test_dataset.py`; status; STOP/review. Completamento: assenza di ragione approvata è errore, non drop.

### Task 6: Identità globale e audit bridge per sample

**Purpose:** Separare compatibilità globale dalla prova del bridge utilizzato senza mutare i tipi learnable/cache sample.

**Files:**

- Modify: drl/IL/bc_v1/dataset.py — _code_hashes, identity/read_cache_manifest, frame audit writers.
- Test: drl/IL/tests/bc_v1/test_dataset.py.

**Interfaces:** Consumes: provider.identity, source hashes e PreparedPhysicalInputs completo, incluse `derived.geometry` e `derived.features is derived.inputs.features`. Produces: `frame_cache_identity(provider: SourceFrameProvider, source_identity: Mapping) -> Mapping` e `frame_sample_audit(key: tuple[str,int], derived: PreparedPhysicalInputs) -> Mapping` in dataset; frame audit JSONL per episodio, referenziato/hashato nel manifest. Audit writer e cache writer consumano la FeatureView restituita, MUST NOT richiamare extract_features.

- [ ] Scrivere `test_global_bridge_identity_and_sample_audit_are_separate`, `test_same_identity_and_immutable_source_reproduce_same_physical_sample`, `test_bridge_policy_source_or_evidence_change_rejects_cache`, `test_frame_audit_tampering_or_missing_key_rejected`. Fingerprint sensibile a provider/revision/config/contract/evidence policy/source/certification/missingness; non a ordering o path assoluto della cache.
- [ ] RED: `$BC_PYTHON -B -m pytest $BC_TEST_DIR/test_dataset.py::test_global_bridge_identity_and_sample_audit_are_separate -q`; livelli ancora assenti/conflati.
- [ ] Implementare identity/writers/read checks riusando `_hash`, `cache_fingerprint` e file_hash; estendere code identity a provider/core/trace realmente usati. Audit sample: key immutabile, source frame, risultato R/t certificato o unresolved reason, evidence digest e digest della E1 geometry/validity verificabile dai PolicyInputs più bridge. FeatureView è il contenuto fisico persistito; nessuna modifica a CachedInputs/CachedSample/SampleAudit o campo appreso.
- [ ] Scrivere temporal sidecar unico e topology/frame audit fuori `.pt`, con membership/hash validation e contatori bounded. Non enumerare tutti i numerical transforms nel global config fingerprint: source/evidence identity deterministica + per-sample receipt basta. Cache schema v2 e rimozione della dipendenza normalizer vengono finalizzate Task 8.
- [ ] GREEN: stesso node; regressione `$BC_PYTHON -B -m pytest $BC_TEST_DIR/test_dataset.py::test_same_identity_and_immutable_source_reproduce_same_physical_sample $BC_TEST_DIR/test_dataset.py::test_frame_audit_tampering_or_missing_key_rejected $BC_TEST_DIR/test_dataset.py::test_cache_fingerprint_order_independent_and_content_sensitive -q`.
- [ ] `git diff --check -- $BC_CORE_DIR/dataset.py $BC_TEST_DIR/test_dataset.py`; status; STOP/review. Completamento: identità riproducibile, audit fuori policy.

### Task 7: Regressioni E2 di oggetti, dedup e relazioni

**Purpose:** Provare sugli esatti sette record della spec che recupero dei bridge e canonicalizzazione preservano gli oggetti fisici.

**Files:**

- Create: drl/IL/tests/bc_v1/test_source_frame_regressions.py; drl/IL/tests/bc_v1/fixtures/source_frames.v1.json.
- Test: drl/IL/tests/bc_v1/test_source_frame_regressions.py.

**Interfaces:** Consumes: il PreparedPhysicalInputs completo restituito da derive_physical_inputs, con features E2 già calcolate, provider e observer; fixture con current graph/metadata minimi e record/source digests, keyed by gli episode/index spec §3. Produces: regressioni rappresentative automatizzate, senza dependency su /tmp. Le asserzioni leggono `derived.features` senza richiamare extract_features; baseline diagnostica e percorso bridged sono derivazioni distinte, ciascuna con una sola estrazione E2 e lo stesso algoritmo.

- [ ] Scrivere `test_rc_476_restores_16_valid_environment_rows_and_24_relations`, `test_stairs_1725_bridge_dedup_preserves_physical_objects_and_32_relations`, `test_teleop_9058_world_cones_dedup_preserves_partial_geometry`, `test_seven_paths_preserve_module_local_intrinsic_semantics`, `test_bridge_then_e1_transforms_world_once_preserves_partial_geometry`. Confrontare pose/twist/environment con expected indipendenti nel frame centroide, mai con helper di trasformazione usati dalla pipeline. Copiare snapshot minimi dopo exact provenance match, non record simili; leggere prefix di un episodio alla volta e soltanto le source rows richieste. Fixture hash/source identity verificati; nessun dump RAW completo.
- [ ] RED pertinente tramite prova negativa: lo stesso node RC con policy priva del contratto certificato MUST fallire l'assert 24, mostrando baseline 8; ripristinare la sola policy test e verificare il risultato con provider. Comando: `$BC_PYTHON -B -m pytest $BC_TEST_DIR/test_source_frame_regressions.py::test_rc_476_restores_16_valid_environment_rows_and_24_relations -q`. Non alterare la policy production per provocare RED.
- [ ] Completare le fixture/oracle indipendenti: RC 16/16 validi, collision_box 8+cone 16; stairs sei righe valide prima del dedup, 50(44 valid/6 invalid)→45 tutte valide, relazioni 32. WORLD teleop cone 16; controlli gap 16, stairs 32, button 32, assembly 32 con zero current connections. Atol 1e-9 per geometria; body connector/edge/resource values preservati esattamente, E2 R6 come codec.
- [ ] GREEN: node RC; regressione `$BC_PYTHON -B -m pytest $BC_TEST_DIR/test_source_frame_regressions.py $BC_TEST_DIR/test_feature_schema.py -q`. Nessun generic hardcode dei conteggi; observed sources e missing components non vengono inventati. Se fallisce, STOP e correggere soltanto il task proprietario dopo review.
- [ ] `git diff --check -- $BC_TEST_DIR/test_source_frame_regressions.py $BC_TEST_DIR/fixtures/source_frames.v1.json`; status; STOP/review. Completamento: sette exact paths coperti, non full census 119600.

### Task 8: Lifecycle cache fisica G1 / F2 separati

**Purpose:** Pubblicare cache complete senza normalizer e rendere il fit un comando distinto, senza cambiare la matematica F2.

**Files:**

- Create: drl/IL/bc_v1/normalization_prepare.py.
- Modify: drl/IL/bc_v1/dataset.py — prepare 242–434/reader VERSION; drl/IL/bc_v1/__main__.py — main/run_prepare/fit dispatch; configs/bc_v1/prepare.json — lifecycle paths.
- Test: drl/IL/tests/bc_v1/test_normalization_prepare.py; drl/IL/tests/bc_v1/test_dataset.py; drl/IL/tests/bc_v1/test_dataset_counts.py.

**Interfaces:** Consumes: read_cache_manifest, BCDataset, NormalizationSample, fit_normalizers/write_normalizers/read_normalizers. Produces: physical cache `mssr.bc_v1.physical_cache.v2`; `fit_prepared_normalizers(cache_root: Path, output_dir: Path, *, split_manifest: Mapping, spec: RobotSpec, schema: Mapping, policy: Mapping, expected_identity: Mapping) -> Mapping`; `read_fit_context(path: Path, *, expected_identity: Mapping) -> Mapping`, nel wrapper. CLI `prepare` invariata nelle flags ma fisica soltanto; nuovo `fit-normalizers --cache --output --config --prepare-config`, dove `--output` MUST essere la directory F2 che contiene `normalizers.json` e `fit_context.json`.

- [ ] Scrivere `test_prepare_never_fits_or_writes_official_normalizer`, `test_physical_cache_complete_without_training_readiness`, `test_separate_fitter_opens_only_train_and_keeps_g1_manifest_immutable`, `test_f2_failure_and_collision_preserve_cache_and_existing_artifacts`. Fit/write tripwire nel prepare; holdout factory tripwire nel fitter; duplicate/missing train index e incompatible bridge receipt rifiutati.
- [ ] RED: `$BC_PYTHON -B -m pytest $BC_TEST_DIR/test_dataset.py::test_prepare_never_fits_or_writes_official_normalizer -q`; oggi prepare chiama fit. Secondo RED: `$BC_PYTHON -B -m pytest $BC_TEST_DIR/test_normalization_prepare.py::test_separate_fitter_opens_only_train_and_keeps_g1_manifest_immutable -q`, prima di creare il wrapper: API mancante.
- [ ] Rimuovere fit/policy-load/normalizer publication/rollback dal prepare; finalizzare manifest v2 con physical counts/audit e transazione no-clobber. Reader/BCDataset non richiedono normalizer; v1 rifiutato con errore esplicito, nessuna conversione silenziosa. CLI prepare stampa counts/cache/performance, non normalizer; callers che aspettavano fit implicito usano due comandi, i test legacy accoppiati diventano test delle due transaction.
- [ ] Implementare wrapper F2 con sole factory 49 train e coverage 78302 verificata contro F1/source indices; fixture usa il proprio split. Preparare `normalizers.json` e `fit_context.json` in una directory staging e pubblicare atomicamente l'intera directory come `output_dir`, senza modificare G1. Receipt lega cache fingerprint, bridge/topology/evidence/audit identity, F1/spec/E2/policy e stats hash/fingerprint. Applicazione usa le funzioni F2 esistenti nel lifecycle successivo, non muta i valori fisici G1; nessun G2.
- [ ] GREEN: entrambi i node; regressione `$BC_PYTHON -B -m pytest $BC_TEST_DIR/test_dataset.py $BC_TEST_DIR/test_normalization_prepare.py $BC_TEST_DIR/test_normalization.py -q`; solo tmp fixture fit, nessun fit ufficiale. `$BC_PYTHON -B -m drl.IL.bc_v1 prepare --help` e `$BC_PYTHON -B -m drl.IL.bc_v1 fit-normalizers --help` MUST terminare senza import hardware.
- [ ] `git diff --check -- $BC_CORE_DIR/normalization_prepare.py $BC_CORE_DIR/dataset.py $BC_CORE_DIR/__main__.py configs/bc_v1/prepare.json $BC_TEST_DIR/test_normalization_prepare.py $BC_TEST_DIR/test_dataset.py $BC_TEST_DIR/test_dataset_counts.py`; status; STOP/review. Completamento: due lifecycle certificabili e rollback indipendenti.

### Task 9: Milestone preflight rappresentativa e strutturale

**Purpose:** Produrre prove economiche e un report review prima di autorizzare qualsiasi scansione ampia/prepare costoso.

**Files:**

- Create: drl/IL/bc_v1/structural_preflight.py; drl/IL/tests/bc_v1/test_causal_topology_counts.py.
- Modify: drl/IL/bc_v1/__main__.py — preflight dispatch; configs/bc_v1/prepare.json — preflight sources; drl/IL/tests/bc_v1/fixtures/causal_topology.v1.json — oracle extension.
- Test: drl/IL/tests/bc_v1/test_causal_topology_counts.py.

**Interfaces:** Consumes: lo stesso derive_physical_inputs del prepare, label codec separato e `_representable`; certified temporal/run inventories. Produces: `audit_structural_domains(source: Path, adapter: RobotAdapter, config: Mapping, *, selected_keys: tuple[tuple[str,int],...] | None) -> Mapping`. CLI `preflight-structural --source --prepare-config --output [--known-cases-only]`, con summary scope esplicito; `selected_keys=None` è full structural scan, manual-gated.

- [ ] Scrivere `test_known_stale20_and_17_companions_use_shared_g1_derivation`, `test_mixed_1189_preflight_preserves_both_labels`, `test_preflight_requires_certified_temporal_and_run_catalogues`, `test_preflight_never_opens_unselected_episode_or_fits_f2`, `test_full_structural_preflight_counts_1110_without_target_driven_repair` su fixture piccola. Le 20 keys del blocked audit sono oracle/selettore diagnostico, mai trigger del repair di produzione.
- [ ] RED: `$BC_PYTHON -B -m pytest $BC_TEST_DIR/test_causal_topology_counts.py::test_known_stale20_and_17_companions_use_shared_g1_derivation -q`; manca il preflight condiviso. Missing inventory deve fallire, non saltare o fabbricare categorie.
- [ ] Implementare `audit_structural_domains`: per-episode streaming selected keys, source proofs/cache-index minimi; contare target missing/ambiguous/contradictory/unrepresentable, removals/no-op/deferred e diagnostic reasons separatamente. Riutilizzare candidate domains derivati e nessuna label come provider input. Source/run joins unici e hash certificati dal catalogo; se manca una prova reale, STOP prima dell'accettazione.
- [ ] GREEN sintetico: stesso node; regressione `$BC_PYTHON -B -m pytest $BC_TEST_DIR/test_causal_topology_counts.py $BC_TEST_DIR/test_source_frame_regressions.py $BC_TEST_DIR/test_dataset.py::test_same_identity_and_immutable_source_reproduce_same_physical_sample -q`. Targeted real oracle 20/20+17, healthy controls e 1189; sette geometric paths, stage regression, RC restore, fingerprints e zero unexpected. No broad B2 tests.
- [ ] Run limitato: `$BC_PYTHON -B -m drl.IL.bc_v1 preflight-structural --source datasets/expert_v1_canonical_v6 --prepare-config configs/bc_v1/prepare.json --known-cases-only --output artifacts/bc_v1/preflight/source_frames_v1`. Manifest/F1 SHA prima/dopo, RSS/tempo osservati; nessun full cache.
- [ ] Per tutti 1110: usare un indice completo già certificato soltanto se recuperato e verificabile economicamente. **Attualmente manca**; non costruirlo con un full dataset scan prima del gate. In quel caso il report separa PASS dei casi noti dal gate 1110 non ancora verificato, e Task 10 lo esegue per primo solo dopo nuova approvazione. Non dichiarare full structural PASS da un report dei 20 casi.
- [ ] `git diff --check -- $BC_CORE_DIR/structural_preflight.py $BC_CORE_DIR/__main__.py configs/bc_v1/prepare.json $BC_TEST_DIR/test_causal_topology_counts.py $BC_TEST_DIR/fixtures/causal_topology.v1.json`; status; **STOP / USER REVIEW** del report Tasks 1–9. Inventari mancanti/prove irrisolte bloccano l'accettazione; nessun expensive run automatico.

### Task 10: Full structural gate, G1 fisico e F2 ufficiale dopo approvazione

**Purpose:** Eseguire i gate reali costosi soltanto dopo approvazione esplicita delle evidenze Task 9; prima preflight, poi G1 completo, infine F2 ufficiale.

**Files:**

- Modify: drl/IL/tests/bc_v1/test_dataset_counts.py — official physical/F2 gates.
- Test: drl/IL/tests/bc_v1/test_dataset_counts.py; drl/IL/tests/bc_v1/test_normalization_prepare.py.

**Interfaces:** Consumes: CLI preflight-structural/prepare/fit-normalizers e verifier dei Task 1–9; source/F1/spec pins congelati. Produces: report certificato 1110, physical cache 73/119600 e normalizer 49/78302 con receipt compatibile.

- [ ] **MANUAL GATE:** registrare approvazione esplicita dell'utente dopo review Task 9, inclusa la scansione strutturale completa se manca l'indice economico. Un timeout o sola approvazione del piano non autorizza il run. Inventari/certificazioni MUST essere disponibili e verificati prima di procedere.
- [ ] Scrivere `test_official_physical_cache_required_and_exact_counts`, `test_official_normalizer_required_and_exact_train_receipt`; RED del primo: `$BC_PYTHON -B -m pytest $BC_TEST_DIR/test_dataset_counts.py::test_official_physical_cache_required_and_exact_counts -q`, FAIL su cache assente, senza skip o implicit prepare. RED del secondo prima del fit: `$BC_PYTHON -B -m pytest $BC_TEST_DIR/test_dataset_counts.py::test_official_normalizer_required_and_exact_train_receipt -q`, FAIL su artifact assente.
- [ ] Se 1110 non ancora certificati: `$BC_PYTHON -B -m drl.IL.bc_v1 preflight-structural --source datasets/expert_v1_canonical_v6 --prepare-config configs/bc_v1/prepare.json --output artifacts/bc_v1/preflight/source_frames_full_v1`. Streaming una riga/episodio alla volta, decodifica fisica soltanto dei bucket strutturali; gate 1110, oracle 20+17, zero failure/extra non certificati. STOP sul failure prima di prepare.
- [ ] Un solo full prepare: `$BC_PYTHON -B -m drl.IL.bc_v1 prepare --source datasets/expert_v1_canonical_v6 --output artifacts/bc_v1/cache/g1_v1 --config configs/bc_v1/training.json --prepare-config configs/bc_v1/prepare.json`. Nessuna patch/retry automatico; G1 non fit-ta F2. GREEN node physical; membership/index fingerprints, categorie 6174/113096/309/21, counts 73/119600 e split 49/78302,12/22762,12/18536, targets 849907/1110 invariati, audit coverage/reasons verificati.
- [ ] Solo dopo GREEN fisico: `$BC_PYTHON -B -m drl.IL.bc_v1 fit-normalizers --cache artifacts/bc_v1/cache/g1_v1 --output artifacts/bc_v1/normalization/f2_v1 --config configs/bc_v1/training.json --prepare-config configs/bc_v1/prepare.json`. Pubblicazione atomica di `artifacts/bc_v1/normalization/f2_v1/` con entrambi `normalizers.json` e `fit_context.json`. GREEN node official normalizer; receipt completo, sole 49 train/78302, holdout opened=0, G1 manifest immutato.
- [ ] Regressione mirata: `$BC_PYTHON -B -m pytest $BC_TEST_DIR/test_dataset_counts.py::test_official_physical_cache_required_and_exact_counts $BC_TEST_DIR/test_dataset_counts.py::test_official_normalizer_required_and_exact_train_receipt $BC_TEST_DIR/test_normalization_prepare.py -q`. Nessun secondo census completo meramente ripetitivo. Registrare tempo/RSS/bytes reali e SHA source/F1/spec/E2/policy prima/dopo.
- [ ] `git diff --check -- $BC_TEST_DIR/test_dataset_counts.py`; status; STOP/review finale. Nessun G2/training/recorder/commit/push. Se il run fallisce, lasciare intatti sorgenti/artifact preesistenti e riportare errore, senza promuovere cache/stats parziali.

## Spec coverage e self-review

| Source-frame spec § | Task che soddisfa il requisito |
|---|---|
| 1,2,3: scopo/problema/evidenza e limite rappresentativo | Global Constraints, stato reale, 4,7,9,10 |
| 4,5,6: semantica/contratto/ownership | 1,2,3 |
| 7: world identity/frame fisico | 1,7 |
| 8: stage_local/current-only/numeri | 2,7 |
| 9: legacy map/evidence/restoration | 3,7 |
| 10: fail-closed/reasons | 1,2,3,5 |
| 11,12: autorità E1/local exceptions | 1,4,5,7 |
| 13: physical-equivalence canonicalization | 5,7 |
| 14: diagnostics taxonomy/fail-fast | 5,6,9,10 |
| 15,16: relation/value invariants | 5,7,9 |
| 17: causality/leakage | 1,2,3,4,5 |
| 18: global identity vs sample audit | 6,8 |
| 19: ordine e topology compatibility | 4,9 |
| 20: G1 prima di F2 | 8,10 |
| 21: future acceptance tests | 1–9, con official gates10 |
| 22: catalog limits expected validity | 5,7 |
| 23,24: non-goals/cleanup | Global Constraints e file esclusi |
| 25: representative limitation/fail-fast futuro | 5,9,10 |

Causal-topology: Task 4 integra core/provider già presenti e temporal audit senza cambiare §9; Task 9 conserva oracle 20+17/1189 e rende espliciti i prerequisiti mancanti; Task 10 completa solo dopo gate i requisiti membership1110/119600 e train-only receipt. Il nuovo piano non pretende che il vecchio testo storico «core/provider non esistono» descriva il worktree corrente.

- [x] Tutte le 25 sezioni coperte; nessun gap lasciato implicito.
- [x] Firme/tipi nuovi prodotti prima dei consumer, Primitive E1/CachedSample/PolicyInputs riusati, observer opzionali fuori feature.
- [x] Cinque Review Focus collegati a test concreti; RED/GREEN/regression e review per ogni task, nessun commit step.
- [x] Ordine adapter→topology→D1→D2→bridge→E1→E2→representability→G1→F2 coerente.
- [x] Provider senza label/target/futuro/stage/FSM/mask/outcome; provenienza solo routing/certificazione.
- [x] Unknown/fail-closed e dedup tracciati; KNOWN non nasconde bridge certificati omessi.
- [x] Nessun full feature census o scan largo prima del manual gate; indice1110 e temporal inventory mancanti dichiarati, non inventati.
- [x] Engine/statistica F2 invariati, cache v2 completa separata da readiness e normalizer/receipt.
- [x] Piano supplementare, senza cleanup e senza modificare altri documenti; comandi prescritti, non eseguiti.

**Gate successivo:** USER REVIEW AND APPROVAL OF THE IMPLEMENTATION PLAN, più scelta dell'approccio di esecuzione. Nessun Task1/Task 4 o expensive run è iniziato durante questa pianificazione.
