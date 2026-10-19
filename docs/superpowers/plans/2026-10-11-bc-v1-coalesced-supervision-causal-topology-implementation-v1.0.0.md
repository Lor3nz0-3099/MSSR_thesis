# BC-v1 coalesced supervision + causal topology Implementation Plan v1.0.0

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Aggiornamento 2026-10-12: autorizzata soltanto la milestone Task 1–3; STOP prima di Task 4. Nessun commit/push o dispatch a subagent è implicito.

**Goal:** Preparare 73 episodi / 119600 PolicySample dal Canonical v6 immutabile, mantenendo la supervisione coalesced e applicando soltanto disconnect certificati; poi eseguire F2 sui soli 49 episodi / 78302 sample train.

**Architecture:** Core generico stdlib, provider storico SMORES separato e audit sidecar fuori da PolicyInputs. Reconciliation prima di E2; cache fisica pubblicata prima del fit F2. Il gate §9 è APPROVATO: disconnect di primitive precedenti al bucket ammessi prima del boundary; disconnect di primitive dispatchate nel bucket corrente differiti.

**Tech Stack:** Python 3.10+, dataclass/json/hashlib stdlib, NumPy, Torch CPU e pytest già disponibili nell'ambiente isolato. Nessuna nuova dipendenza hardware o installazione.

**Spec:** [Coalesced supervision + causal topology v1.0.0](../specs/2026-10-11-bc-v1-coalesced-supervision-causal-topology-v1.0.0.md). Leggere anche i report citati dalla spec. Questo piano continua G1: non ricomincia A–F e non implementa G2–K.

## Global Constraints

- Canonical v6 immutabile, manifest SHA `31765272b6ecd7506c54b25e214b75d7b1b28691803ddfc286dafcce8b4ec562`; nessuna v7 o modifica RAW/compact/label/index.
- 73 episodi / 119600 sample: train 49/78302, validation 12/22762, test 12/18536. Sette expert button BC=0 esclusi.
- 309 sequenziali conservate, un sample ciascuna; 21 ambigue conservate aggregate. Vere joint action intatte; mai un sample per label.
- Requisito finale: 1110 target strutturali rappresentabili, 20/20 stale ALIGN recuperati, 17 target compagni ancora validi; missing/ambiguous mapping/contradictory/unrepresentable = 0.
- F1 fingerprint `aee975a2f8b044ad3f57f331fc659d6cfa91d7c1e493b90b812634fb6364e81b`, file e membership invariati.
- RobotSpec `c6e1c78d3fe5f8ede3c1d7e8ab5957799d0952580a171437db7ffcddc092ae19`; E2 `43ab1b93c21b2645176e614ab1af7437f37163ec2b6047c0671d18a354b67d57`; F2 policy `366b9b0dfe838208462e0b99c634f0cf844c8ba15959f1bcf62b3a363261fd91` invariati.
- Solo exact confirmed disconnect; nessuna aggiunta DOCK, ricostruzione pose/twist/posture/joint/geometry, riparazione futura/G_t+1 o guidata dai target. Regole E2 invariate.
- PolicyInputs invariata: temporal flags, stage/FSM, barrier, completion/executor/provenance/goal IDs restano fuori dalla rete. Nessuna WAIT/no-dispatch o memoria temporale.
- F2 engine/policy invariati; fit ufficiale dopo G1 fisico completo, solo 49 train / 78302 sample, nessuna apertura validation/test dal fitter.
- Nessun runtime, recorder, Isaac Lab, G2, model, trainer, sampler o HPO. Nessun commit/push/reset/restore/clean/add; preservare le dirty changes estranee.
- **Gate spec §9 APPROVATO il 2026-10-12:** COALESCED MIXED-BUCKET DISCONNECT POLICY, senza dipendenza dai target. Task 4–9 non autorizzati nella presente milestone.

## Review Focus

1. 1189: l'ultimo prefix libera ALIGN ma elimina UNDOCK. Il test deve conservare entrambe le label, usando una boundary policy approvata e indipendente dai target. Task 3/5.
2. Re-attach/reset dopo un detach: l'evidenza vecchia non libera un nuovo peer. Invalidazione nel provider e conflitto nel core. Task 2/3.
3. Recovery con companion targets e wave concorrenti: nessuna perdita di label o split del batch. Task 4/5/6.
4. Cache interning e identity: domini occupato/libero distinti; cambio di eventi/code/sidecar invalida la cache senza cambiare E2/F1. Task 4/7.
5. Prepare/fit: cache fisica completa non è training-ready; un crash F2 lascia G1 intatta; il fitter non apre holdout. Task 4/9/10.

---

## Stato reale e lista esatta dei file previsti

Core e provider causali non esistono: la precedente continuazione è stata bloccata prima dell'implementazione. Dataset, CLI prepare, adapter, PolicyInputs, E1/E2, split e normalizzazione esistono già. `prepare_dataset` oggi esegue F2 prima di pubblicare G1: il piano deve separare esplicitamente queste fasi.

Nuovi file da creare dopo approvazione:

```text
drl/IL/bc_v1/temporal_supervision.py
drl/IL/bc_v1/causal_topology.py
drl/IL/bc_v1/smores_disconnect_evidence.py
drl/IL/bc_v1/structural_preflight.py
drl/IL/bc_v1/normalization_prepare.py
configs/bc_v1/causal_topology.json
drl/IL/tests/bc_v1/test_temporal_supervision.py
drl/IL/tests/bc_v1/test_causal_topology.py
drl/IL/tests/bc_v1/test_smores_disconnect_evidence.py
drl/IL/tests/bc_v1/test_causal_topology_counts.py
drl/IL/tests/bc_v1/test_normalization_prepare.py
drl/IL/tests/bc_v1/fixtures/causal_topology.v1.json
```

File esistenti da modificare dopo approvazione:

```text
drl/IL/bc_v1/dataset.py
drl/IL/bc_v1/__main__.py
configs/bc_v1/prepare.json
drl/IL/tests/bc_v1/test_dataset.py
drl/IL/tests/bc_v1/test_dataset_counts.py
```

Nessuna modifica prevista a bc_v1_contract.py, policy_inputs.py, task_context.py, smores_adapter.py, candidates.py, features.py, feature_schema.py, split.py, normalization.py, RobotSpec, features.json, normalization.json, training.json o F1 split. Il provider viene iniettato dalla composition CLI/G1, separatamente dal label adapter. Un ulteriore cambio di contratto richiede revisione del piano.

Artifact derivati futuri, non prodotti oggi:

```text
artifacts/bc_v1/audits/temporal_supervision_v1.json
artifacts/bc_v1/preflight/coalesced_topology_v1/{report.md,summary.json,integrity.json}
artifacts/bc_v1/cache/g1_v1/{manifest.json,episodes/<sha256(episode_id)>.pt,audit/temporal_supervision.json,audit/topology.jsonl}
artifacts/bc_v1/normalization/f2_v1/{normalizers.json,fit_context.json}
artifacts/bc_v1/reports/coalesced_topology_v1/{report.md,verification.json}
```

Il sidecar viene dall'audit certificato, senza modificare Canonical. Le fixture copiano soltanto current state ed eventi minimi. I precedenti artifact blocked restano intatti.

## Procedura e ambiente

Comandi dalla root. Impostare `BC_PYTHON=/home/lorenzo/.local/share/mssr-bc-v1/venv/bin/python`; usare questo ambiente già disponibile senza nuove installazioni.

Ogni task segue: test nuovo → RED pertinente osservato → implementazione minima → stesso comando GREEN → controllo integrità/diff. Failure di sintassi, ambiente o sorgente sbagliata non costituiscono RED accettabile. Nessuno skip vale come GREEN di un gate ufficiale. Eseguire il prepare costoso una sola volta dopo preflight; nessun retry automatico sul fallimento. Tutti i comandi seguenti sono futuri: non vengono eseguiti durante questa consegna.

## Task 1 — Audit metadata contract

**File:** creare temporal_supervision.py e test_temporal_supervision.py nelle directory indicate.

**Interfacce:** frozen `TemporalSupervisionAudit(episode_id: str, transition_index: int, kind: str)`; `validate_temporal_audit(entries: tuple[TemporalSupervisionAudit,...], expected_keys: frozenset[tuple[str,int]]) -> None`; `temporal_audit_fingerprint(entries, source_identity: Mapping) -> str`; `read_temporal_audit(path: Path, *, expected_source_sha256: str, expected_keys) -> tuple[TemporalSupervisionAudit,...]`. Writer JSON deterministico nello stesso modulo. Versione ed enum esatti spec §14.

- [ ] **RED:** `test_audit_contract_exact_enum_and_frozen`, `test_audit_is_not_policy_input`, `test_duplicate_missing_foreign_keys_fail`, `test_temporal_fingerprint_order_invariant_content_sensitive`. Quattro enum soltanto; nessun campo nuovo in PolicyInputs; riordino equivalente conserva fingerprint, cambio di kind/source lo cambia.
- [ ] **Verifica RED:** `$BC_PYTHON -B -m pytest drl/IL/tests/bc_v1/test_temporal_supervision.py -q`; API mancante o asserzione sul comportamento richiesto.
- [ ] **Implementazione minima:** tipi stdlib, validazione one-to-one, ordinamento per chiave e hash JSON canonico. Nessun classifier euristico, tensor o hardware import.
- [ ] **Verifica GREEN:** stesso comando, poi test_policy_inputs.py e test_contract.py.
- [ ] **Completamento:** contratto audit versionato e isolato; nessun sample o target modificato.

## Task 2 — Generic causal topology types/core

**File:** creare causal_topology.py, test_causal_topology.py e configs/bc_v1/causal_topology.json. Registrare la mixed-bucket boundary policy approvata in spec §9.

**Interfacce:** `CausalBoundary`, `ConfirmedDisconnect`, `ReconciledTopologyResult` con i campi della spec §10; `reconcile_topology(graph: StateGraph, events: tuple[ConfirmedDisconnect,...], boundary: CausalBoundary) -> ReconciledTopologyResult`; `causal_topology_fingerprint(policy: Mapping) -> str`, includendo versione/algoritmo/policy approvata.

- [ ] **RED:** exact pair removal e occupazione coerente; source immutabile; idempotence; duplicate event; already-absent no-op; unrelated edge unchanged; wrong-peer conflict; event futuro/cross-domain rejected; nessun argomento label/stage/FSM/future; pose/twist/resources/connector geometry invarianti; ordine endpoint inverso equivalente; cardinalità variabile. Import blocker ROS/Isaac/SMORES/Torch/NumPy.
- [ ] **Verifica RED:** `$BC_PYTHON -B -m pytest drl/IL/tests/bc_v1/test_causal_topology.py -q`.
- [ ] **Implementazione minima:** dataclass stdlib e copia funzionale di connections/occupancy. Validare coerenza prima della rimozione, errore senza mutazione parziale. Nessun ID hardware, stato globale, edge addition o cinematica.
- [ ] **Verifica GREEN:** stesso comando, includendo fingerprint e import isolation.
- [ ] **Completamento:** il core opera soltanto su eventi già ammessi e non richiede label; il GREEN unitario non approva il cutoff storico.

## Task 3 — Historical SMORES disconnect evidence provider

**File:** creare smores_disconnect_evidence.py, test_smores_disconnect_evidence.py e fixture causal_topology.v1.json. Configurare la policy approvata in §9.

**Interfacce:** provider-local `HistoricalDecisionBoundary(source_records: tuple[tuple[str,int],...], dispatch_ids: tuple[str,...])`; `DisconnectEvidenceBatch(events: tuple[ConfirmedDisconnect,...], boundary: CausalBoundary, audit: Mapping)`; `HistoricalSmoresDisconnectProvider(repo_root: Path, policy: Mapping)` e `provider.for_bucket(source_refs: Mapping, published_graph: StateGraph) -> DisconnectEvidenceBatch`. source_refs contiene run identity/provenance pointers, non action_t/label/mask. Il core non importa questi tipi.

- [ ] **RED:** SUCCEEDED exact detach con ordine precedente ammesso; ACCEPTED/command-only/failed non ammessi; boundary mancante/duplicata/ambigua o log di altro run rifiutati; completion futura non ammessa; timestamp mancante resta missing; redock invalida old detach senza edge addition; reset invalida evidence; invocazione senza target/stage/FSM/control_state.
- [ ] **RED del gate approvato:** `test_mixed_bucket_boundary_1189_matches_approved_policy` deve ammettere detach02 e differire detach05, con 7→6 edge e entrambe le label rappresentabili. Aggiungere un esempio sintetico con ID/connector/goal diversi; nessun default latest-prefix o whitelist.
- [ ] **Verifica RED:** `$BC_PYTHON -B -m pytest drl/IL/tests/bc_v1/test_smores_disconnect_evidence.py -q`.
- [ ] **Implementazione minima:** parser puro di goal identity e physical completion; prefix/event order verificati e source hash. Riutilizzare source indexing senza import runtime. Boundary policy dichiarata e hashata; nessun inferred detach dai target.
- [ ] **Verifica GREEN:** stesso comando; fingerprint sensibile a eventi/parser/cutoff e compatibilità col core Task 2.
- [ ] **Completamento:** ogni evento restituito ha prova causale. Il gate mixed-case è approvato; i suoi test devono passare prima del GREEN definitivo. Integrazione esclusa dalla milestone.

## Task 4 — G1 integration prima di E2 e prepare fisico separato

**File:** modificare dataset.py, __main__.py, prepare.json, test_dataset.py e test_dataset_counts.py. Nessun cambio a adapter/core contract/E2/F2 engine.

**Consuma:** Task 1–3; adapter, select_policy_inputs, transform_inputs, extract_features e CurrentCandidateCache esistenti. **Produce:** stessa firma `prepare_dataset(source, output, adapter, config) -> Mapping`, con injected `config['topology_evidence_provider']` callable e `config['temporal_audit']` validato; cache fisica `mssr.bc_v1.physical_cache.v2`, senza fit F2. Reader e BCDataset accettano questa cache completa senza normalizer. CachedInputs/CachedSample invariati.

- [ ] **RED:** spy verifica reconciliation prima di D2/E1/E2; candidate interning distinto per occupato/libero; un bucket produce un sample con tutte le label; label codec usa original graph; action/mask/future/FSM non cambiano derivation; pins invariati. `test_prepare_never_fits_or_writes_official_normalizer`, `test_physical_cache_complete_without_training_readiness`; identity include core/provider/event/sidecar hashes; missing/changed evidence provoca errore.
- [ ] **Verifica RED:** `$BC_PYTHON -B -m pytest drl/IL/tests/bc_v1/test_dataset.py -q`.
- [ ] **Implementazione minima:** CLI inietta provider; hook generico prima delle feature; audit sidecar scritto in staging; code hashes estesi ai nuovi moduli e provider. Separare fit/pubblicazione normalizer dal prepare fisico; eliminare dipendenza cache identity dal futuro normalizer. Tenere transaction, integrity e no-clobber, con schema cache v2.
- [ ] **Verifica GREEN:** stesso comando, CLI prepare --help e regressioni E1/E2/PolicyInputs su fixture; nessun prepare reale.
- [ ] **Completamento:** cache fixture fisica pubblicabile e rileggibile senza F2. Aggiornare i test che presumevano fit automatico per verificare le due fasi; conservare tutti i rollback gate.

## Task 5 — Regression dei 20 stale ALIGN e healthy controls

**File:** creare test_causal_topology_counts.py; completare fixtures/causal_topology.v1.json. Eventuali fix di produzione restano nel perimetro Task 2–4.

**Consuma:** published states reali, venti chiavi/source hash congelati nel report G1 e proof indipendenti dal producer. Quattro healthy controls documentati dal precedente spike; nessuna allowlist nella produzione.

- [ ] **RED:** i venti ALIGN mancano nei published domains; dopo la derivazione devono essere 20/20 rappresentabili. Assert 37 target dei venti bundle conservati: 20 recuperati + 17 già validi; payload immutati; tutti i canali non topologici identici; quattro controls invariati. 1189 richiede ALIGN02 e UNDOCK05 rappresentabili nello stesso sample sotto la policy approvata.
- [ ] **Verifica RED:** `$BC_PYTHON -B -m pytest drl/IL/tests/bc_v1/test_causal_topology_counts.py -k 'stale20 or healthy or mixed_1189' -q`.
- [ ] **Implementazione minima:** perfezionare soltanto regola approvata/provider/removal. Fault injection wrong-peer, future completion e missing proof deve fallire. Nessun E2 relaxation, union di domini, drop/split o future state nelle fixture del provider.
- [ ] **Verifica GREEN:** stesso comando, poi test_candidates.py e test_e2_integration.py.
- [ ] **Completamento:** 20/20 + 17 e controls PASS; E2/fingerprint invariati. Se 1189 non passa entro la policy approvata, fermare l'integrazione e riportare il conflitto.

## Task 6 — Metadata dei 309 e 21 senza split

**File:** modificare temporal_supervision.py, test_temporal_supervision.py e test_dataset.py per la sola integrazione audit. Produrre sidecar derivato temporal_supervision_v1.json.

**Interfaccia:** `import_certified_temporal_audit(audit_root: Path, source_identity: Mapping, expected_keys) -> tuple[TemporalSupervisionAudit,...]`, dall'inventario storico completo verificato. CLI prepare legge il path configurato; il trainer non importa report o /tmp.

- [ ] **RED:** counts 6174/113096/309/21 e per-split spec §3; full membership 119600 con indici invariati; 309 retained incluse le dieci con cardinalità fine incerta; 21 retained unresolved. TILT/wheels e ALIGN wave restano nello stesso sample. Unknown/missing key non diventa joint; cambiare kind non cambia input/label/candidati a fisica/evidenze uguali.
- [ ] **Verifica RED:** `$BC_PYTHON -B -m pytest drl/IL/tests/bc_v1/test_temporal_supervision.py drl/IL/tests/bc_v1/test_dataset.py -q`.
- [ ] **Implementazione minima:** estrazione delle classificazioni certificate, JSON deterministico e completo, hashes di audit e Canonical. Nessun filtro/sampler speciale. Se l'inventario originale manca, fallire e recuperare una copia verificata; non indovinare categorie.
- [ ] **Verifica GREEN:** stesso comando e real inventory join read-only nel test `test_frozen_temporal_inventory_counts`; nessun prepare G1.
- [ ] **Completamento:** artifact portable, one-to-one e audit-only. I venti stale ALIGN non diventano un classifier dei sequenziali.

## Task 7 — Full targeted structural preflight

**File:** creare structural_preflight.py; estendere test_causal_topology_counts.py; aggiungere subcommand preflight-structural in __main__.py.

**Interfaccia:** `audit_structural_domains(source: Path, adapter, config: Mapping) -> Mapping`. Riusa preparation path/hook/candidate interning del Task 4, non un secondo algoritmo di repair. CLI con source, prepare-config e output espliciti.

- [ ] **RED:** 1110 target strutturali esatti; missing/ambiguous mapping/contradictory/unrepresentable = 0; oracle 20/20 e companion gate. Nessuna label aggiunta/eliminata, source target count 849907 conservato; nessun future access. Extra repair non certificati o failure bloccano prepare e hanno diagnostica sample/evidence. Removals/no-op/extra per split misurati, non assunti uguali a 20.
- [ ] **Verifica RED:** `$BC_PYTHON -B -m pytest drl/IL/tests/bc_v1/test_causal_topology_counts.py -k structural_preflight -q`.
- [ ] **Implementazione minima:** scan Canonical streaming, decodifica stati soltanto per righe con target strutturali, provider e codec separati, E2 per current domain e validazione congiunta dei target. No cache tensor/fit. Hash source/evidence e F1 before/after.
- [ ] **Verifica GREEN:** stesso test; poi `$BC_PYTHON -B -m drl.IL.bc_v1 preflight-structural --source datasets/expert_v1_canonical_v6 --prepare-config configs/bc_v1/prepare.json --output artifacts/bc_v1/preflight/coalesced_topology_v1`.
- [ ] **Completamento:** report reale 1110 / zero errori e 20/20; SHA Canonical/F1 invariati. Eventuali extra apply richiedono certification review prima di Task 8. Ambiguità temporale non confusa con target mapping ambiguity.

## Task 8 — Real G1 physical prepare

**File:** estendere test_dataset_counts.py con gate ufficiale fisico che non può essere saltato. Nessun nuovo codice product se Task 4–7 passano.

**Interfacce:** CLI prepare e BCDataset esistenti con cache v2; Canonical e F1 congelati.

- [ ] **RED:** `test_official_physical_cache_required_and_exact_counts` fallisce su cache assente/incompleta. Assert 73/119600, train 49/78302, validation 12/22762, test 12/18536, sette esclusi, transition index fingerprints per episodio, audit 309+21 e target 1110. Official normalizer assente prima del fit; il legacy test che salta non è accettazione del gate.
- [ ] **Verifica RED:** `$BC_PYTHON -B -m pytest drl/IL/tests/bc_v1/test_dataset_counts.py -k official_physical -q`.
- [ ] **Implementazione minima:** un solo prepare completo dopo Task 7, senza overwrite o patch durante la run: `$BC_PYTHON -B -m drl.IL.bc_v1 prepare --source datasets/expert_v1_canonical_v6 --output artifacts/bc_v1/cache/g1_v1 --config configs/bc_v1/training.json --prepare-config configs/bc_v1/prepare.json`.
- [ ] **Verifica GREEN:** stesso test; controlli indipendenti manifest/hash/types/counts, lazy read nei tre split, valori ancora fisici. Registrare tempo/RAM/dimensione osservati, senza riprendere stime del failed prepare.
- [ ] **Completamento:** cache fisica completa pubblicata atomicamente e verificata, sorgenti/F1 immutati; nessun F2 fit da prepare. F2 non è ancora completo.

## Task 9 — Official F2 train-only fit

**File:** creare normalization_prepare.py e test_normalization_prepare.py; modificare __main__.py, prepare.json per i path e test_dataset_counts.py per il gate ufficiale F2. Non modificare normalization.py o la policy.

**Interfaccia:** `fit_prepared_normalizers(cache_root: Path, output: Path, *, split_manifest: Mapping, spec: RobotSpec, schema: Mapping, policy: Mapping, expected_identity: Mapping) -> Mapping`. Output è normalizers.json; pubblicare atomicamente la directory f2_v1 con stats e fit_context.json. CLI fit-normalizers con cache/output/config/prepare-config espliciti.

- [ ] **RED:** G1 incomplete/incompatible rifiutata; factory validation/test con tripwire mai aperte; esattamente 49 train / 78302 sample e complete_train_coverage=true; duplicate/missing index rifiutati. Receipt lega cache/core/provider/audit/F1/E2/spec/policy e normalizer fingerprint/hash; mismatch rifiutato. Holdout mutation non cambia stats; fit/write failure non corrompe G1 o artifact precedente; nessun partial normalizer pubblicato; manifest G1 immutato.
- [ ] **Verifica RED:** `$BC_PYTHON -B -m pytest drl/IL/tests/bc_v1/test_normalization_prepare.py drl/IL/tests/bc_v1/test_normalization.py -q`.
- [ ] **Implementazione minima:** factory solo per 49 train, existing NormalizationSample dai CachedSamples, existing fit_normalizers/writer/reader. Verifica coverage e pubblicazione atomica stats+receipt, senza cambiare engine/statistica. Dipendenza stats→cache nel receipt; nessun ciclo con cache identity e nessun train+validation merge.
- [ ] **Verifica GREEN:** stesso comando; poi `$BC_PYTHON -B -m drl.IL.bc_v1 fit-normalizers --cache artifacts/bc_v1/cache/g1_v1 --output artifacts/bc_v1/normalization/f2_v1/normalizers.json --config configs/bc_v1/training.json --prepare-config configs/bc_v1/prepare.json`; `$BC_PYTHON -B -m pytest drl/IL/tests/bc_v1/test_dataset_counts.py -k official_normalizer -q`.
- [ ] **Completamento:** stats/receipt verificati, opened IDs esattamente i 49 F1 train ordinati, sample count 78302, holdout opened=0; engine/policy invariati. Nessun fit prima del GREEN fisico Task 8.

## Task 10 — Final verification/report e stop

**File:** nessun nuovo codice applicativo; aggiornare questi documenti con risultati e produrre report/verification artifact indicati.

- [ ] **Gate finale RED:** cache/receipt assenti, mismatch, target failure, counts drift o real gate saltato impediscono complete. Verificare source SHA, F1 fingerprint/membership e catalog pins; 309+21 e flags fuori input; healthy controls e 1110/20 oracle; train coverage.
- [ ] **Verifica GREEN completa:** `$BC_PYTHON -B -m pytest drl/IL/tests/bc_v1 -q` senza MSSR_BC_V1_SKIP_REAL_CENSUS; `$BC_PYTHON -B -m pytest drl/IL -q`; `git diff --check`. Segnalare collection failure/dependency mancanti senza modificare global/runtime. `sha256sum datasets/expert_v1_canonical_v6/manifest.json artifacts/bc_v1/splits/f1_v1/split.json configs/bc_v1/smores_ep.robot.json configs/bc_v1/features.json configs/bc_v1/normalization.json` confrontato con pins e baseline.
- [ ] **Deliverable minimo:** report delle identity source/derivation/cache/stats, counts, categorie per split, evidence/removals/no-op/differiti, target failure, 49/78302 coverage, tempi/RAM/bytes osservati, comandi/results e integrità delle dirty changes. Dichiarare il limite coalesced, senza spacciarlo per pre-state atomico.
- [ ] **Completamento:** gate reali required passati senza skip, G1 fisico e F2 certificati, report salvato. Stop: nessun G2/train/rollout/recorder/commit/push.

## Gate matrix e checklist prima dei primi update

| Gate | Task |
|---|---|
| Audit enum/one-to-one/flags fuori PolicyInputs | 1,6 |
| Exact removal, immutability, idempotence, absent no-op, wrong peer | 2 |
| No future/label/stage/FSM requirement, event ordering, redock invalidation | 2,3,4 |
| Boundary 1189 approvata, entrambe le label rappresentabili | 3,5 |
| E2 candidate restored, free-endpoint/current-connection rule invariata | 4,5 |
| 20/20 stale ALIGN, 17 companion, healthy controls | 5 |
| 309+21 retained; vere wave intatte | 4,6,8 |
| 1110 structural, missing/ambiguous mapping/contradictory/unrepresentable 0 | 7,8 |
| 119600/73, F1 counts e Canonical SHA/F1 pins invariati | 7,8,10 |
| Cache identity include core/provider/evidence/temporal audit | 4,7,8 |
| Prepare non fit-ta F2; F2 solo 49/78302 train con complete coverage | 4,9 |
| Cache/stat receipt compatibili; transaction/no-clobber e integrità | 4,9,10 |

Prima dei primi update: gate §9 risolto; Task 1–10 verificati; input/candidate/target contract senza leakage; cache fisica completa + normalizer train + receipt. G2 batching/sampler e H/I rete/loss rimangono da implementare e testare nel piano successivo. Questa consegna non autorizza né certifica il primo training. GRAVITY_SETTLE enabled, cardinalità variabili e mini-tuning di 12 trial train+validation dopo overfit restano vincoli del futuro G2–K.

## Self-review del piano

- [x] Dieci task ordinati: RED, file/interfacce, minimo GREEN, comando e criterio di completamento.
- [x] File esistenti riusati e nuovi file effettivamente mancanti; nessuna ripetizione A–F o filtro ALIGN22.
- [x] Spec coverage e cinque Review Focus coperti dalla gate matrix.
- [x] Boundary 1189 APPROVATA il 2026-10-12 e registrata senza estendere il perimetro oltre Task 1–3.
- [x] Firme/tipi coerenti; PolicyInputs/CachedSample invariati; core compatibile online separato dal provider storico.
- [x] Prepare fisico prima di F2, cache v2 e receipt separato; engine/policy invariati e nessun identity cycle.
- [x] Comandi futuri, non risultati osservati. Nessun task TDD, prepare o fit iniziato durante questa documentazione.

**Milestone autorizzata il 2026-10-12:** Task 1–3 con RED/GREEN; gate §9 approvato. STOP dopo Task 3, prima di Task 4; nessun avvio G1/F2 o commit.
