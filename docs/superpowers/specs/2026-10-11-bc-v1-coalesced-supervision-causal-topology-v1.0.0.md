# BC-v1 — temporally coalesced supervision + causal topology v1.0.0

Data: 2026-10-11. Branch: `wip/t8-c05-human-teleop-20260921`.
Direzione approvata: **Strategy B, temporally coalesced baseline, con reconciliation disconnect-only certificata**.
Stato: specifica documentale; nessuna implementazione autorizzata da questo documento. Il gate §9 è APPROVATO dall'utente il 2026-10-12; questa milestone implementa soltanto Task 1–3, senza integrazione G1.

## 1. Autorità, continuità e precedenza

Questa specifica sostituisce, per il presente delta, la [spec causal topology del 10 ottobre](2026-10-10-bc-v1-causal-topology-v1.0.0.md). La precedente rimane disponibile come evidenza del blocco e non viene sovrascritta. La [spec architetturale originale](2026-10-06-bc-v1-architecture-v1.0.0.md) e il [piano originale](../plans/2026-10-06-bc-v1-implementation-v1.0.0.md) descrivono il resto della BC-GNN; i loro numeri v5, filtro ALIGN22 e stage embedding sono storici, non prescrizioni da reintrodurre.

Le decisioni successive già implementate hanno precedenza: Canonical v6, 73 episodi BC, F1 49/12/12, TaskContext con task e geometria senza stage/configuration_goal, PolicyInputs D2, FrameGeometryView E1 separata e schema E2 congelato. Questo delta non ricomincia A–F e non applica un nuovo filtro delle 22 label storiche. Piano associato: [coalesced-supervision implementation v1.0.0](../plans/2026-10-11-bc-v1-coalesced-supervision-causal-topology-implementation-v1.0.0.md).

## 2. Problema reale

Canonical v6 contiene physical-stamp buckets, non garantisce che ogni bucket rappresenti un unico decision point causale. Più dispatch possono leggere lo stesso stato cached attraverso completion e barrier intermedie. È necessario distinguere una vera joint action da una supervisione sequenziale associata alla stessa misura disponibile.

Separatamente, venti ALIGN usano uno stato topologico pubblicato che conserva un edge già staccato con successo prima dell'ALIGN. L'expert ha rispettato l'ordine UNDOCK → detach riuscito → ALIGN. Questo secondo difetto riguarda la topologia pubblicata, non venti nuovi errori dell'expert o falsi inizi.

## 3. Evidenze e risultati dei due audit

Fonti lette per questo delta:

- [G1 blocked report](../../../artifacts/bc_v1/cache/g1_v1_blocked/report.md), inclusa la lista dei venti target e dei diciassette target compagni validi;
- [causal-boundary preflight](../../../artifacts/bc_v1/cache/g1_v1_causal_topology_blocked/report.md);
- `/tmp/mssr-g1-align20-spike/report.md`;
- `/tmp/mssr-bc-v1-decision-group-audit/{report.md,summary.json}`;
- `/tmp/mssr-raw-predecision-state-audit/{report.md,summary.json}`.

Gli artifact in /tmp sono evidenza storica, non dipendenze permanenti del trainer. L'implementazione dovrà rendere persistente soltanto il sidecar minimo verificato e le relative identità, senza ricopiare i dump RAW. La precedente proposta di proiezione in nuovi sample nel decision-group audit è superata dalla Strategy B approvata dopo l'audit RAW.

| Categoria temporale | Sample | Train | Validation | Test |
|---|---:|---:|---:|---:|
| SINGLE_DECISION | 6174 | 4053 | 1163 | 958 |
| JOINT_SINGLE_POINT | 113096 | 74026 | 21550 | 17520 |
| MULTI_DECISION_SEQUENTIAL | 309 | 212 | 46 | 51 |
| AMBIGUOUS | 21 | 11 | 3 | 7 |
| Totale | 119600 | 78302 | 22762 | 18536 |

Le 309 rappresentano **0.258361%** del totale. Pattern minimi certificati: TILT → wheels 239; structural → structural 50; structural → resource 13; altri resource → resource 7. Dieci delle 309 hanno cardinalità fine 2–3 non risolta; sono comunque sequenziali, non ambigue fra simultaneità e sequenza. I ventuno casi AMBIGUOUS sono una categoria distinta.

L'audit RAW ha esaminato tutti i 309 boundary minimi, dieci possibili boundary aggiuntivi e ventuno intervalli ambigui. Per i 309: fresh 0, partial 0, no-fresh 309, freshness ambiguous 0. RAW_TO_COMPACT_LOSS 0; COMPACT_TO_CANONICAL_LOSS 0; CANONICAL_GROUPING_LOSS di fresh snapshot 0; NO_LOSS_RAW_HAS_NONE 309. Nessuno snapshot fresco recuperabile è stato perso dalla compaction o dal grouping. La perdita del confine decisionale non equivale alla perdita di una misura fisica.

## 4. Sorgente e conteggi immutabili

Sorgente: `datasets/expert_v1_canonical_v6`. SHA256 del manifest:

```text
31765272b6ecd7506c54b25e214b75d7b1b28691803ddfc286dafcce8b4ec562
```

La directory v6 mantiene internamente gli schema Canonical già dichiarati nel manifest; non rinominarli per farli coincidere con il nome della directory. Gli ottanta episodi sorgente includono sette expert button BC=0; gli episodi BC-useful sono 73. Eleggibilità esattamente `valid_for_behavior_cloning is True`, senza nuovo filtro, espansione RLE, duplicazione, split o cambio degli indici.

| Split F1 | Episodi BC | Physical buckets / PolicySample |
|---|---:|---:|
| Train | 49 | 78302 |
| Validation | 12 | 22762 |
| Test | 12 | 18536 |
| Totale | 73 | 119600 |

Target supervisionati invarianti: wheels 576942; pan_velocity 186484; pan_target 611; tilt_target 84760; ALIGN_FACES 503; DOCK 477; UNDOCK 106; GRAVITY_SETTLE 24. Totale target 849907; strutturali **1110**, inclusa GRAVITY_SETTLE enabled di default. Non confondere label, sample e componenti scalari.

## 5. Perché non creare Canonical v7

Un rebuild potrebbe cambiare il raggruppamento, ma non produrre nuove misure fisiche mancanti nei RAW. La baseline mantiene la sorgente, gli indici, gli episodi, lo split e tutti i target. Non crea Canonical v7 e non converte o rifà le dimostrazioni. La semantica più prudente è registrata nella vista derivata BC e nel suo audit.

## 6. Semantica temporally coalesced

Una transition BC-valid produce **esattamente un PolicySample**. Per le 309 sequenziali, il significato è *temporally coalesced sequential supervision*: tutte le supervisioni conservate sono associate all'unico current physical state effettivamente osservato e disponibile nel bucket storico.

Esempio: TILT → posture reached → wheels può essere sequenziale nel runtime; se non esiste una misura intermedia, il sample conserva `G_t → {TILT, wheels}`. Non afferma che le due operazioni siano state fisicamente simultanee. La completion non misura un angolo, una posa o una velocità. Nessuna label viene usata come misura.

Questa è un'approssimazione offline dichiarata. Non certifica un pre-state atomico rigoroso per ciascuna label né dimostrazioni complete rigorose di self-assembly. L'assenza di memoria temporale nella GNN resta una scelta della baseline; il coalescing non introduce hidden state.

## 7. Vere joint action e casi ambigui

Le 113096 vere joint action rimangono unite. ALIGN di più moduli nella stessa wave, coordinated UNDOCK/DOCK, wheels multi-module e risorse concorrenti non vengono trasformati in un sample per label. Tutti i target presenti contribuiscono alle rispettive loss già previste.

Le 309 sequenziali rimangono tutte, senza split, drop, duplicazione o target sintetici. Le 21 AMBIGUOUS rimangono aggregate esattamente come oggi, con audit `ambiguous`; non viene forzata una spiegazione simultanea o sequenziale. Nessun mask loss dipende da questa classificazione.

## 8. Venti stale ALIGN e requisito di recovery

Audit certificato: 20/20 RAW_STALE_CONFIRMED_DISCONNECT_ONLY. Per ciascuno: precedente UNDOCK riuscito, exact old pair documentata nella completion, nuova ALIGN successiva, vecchio edge ancora nel published G_t, nessun fresh snapshot pre-ALIGN. Non sono label expert errate.

I venti appartengono a dieci episodi: 11 train, 6 validation, 3 test. La loro classificazione temporale è single 3, joint 16, sequential 1. Quindi non coincidono con i 309; l'intersezione contiene il caso 1189.

Requisito finale, non risultato già ottenuto: 20/20 ALIGN rappresentabili e tutti i 1110 target strutturali conservati e rappresentabili; missing 0, ambiguous target mapping 0, contradictory 0, unrepresentable 0. I diciassette target compagni validi nei venti bundle devono restare validi. La classificazione temporale ambiguous dei ventuno sample non significa ambiguous target mapping.

## 9. Confine misto 1189: gate APPROVATO il 2026-10-12

`teleop-39e2f4b4cd684e9ea72734cc21baca94 / 1189` conserva UNDOCK05 e ALIGN02. RAW227 contiene UNDOCK02 nel bucket precedente; RAW228 contiene UNDOCK05; RAW231 contiene REACH02. Log T15: ACCEPTED UNDOCK02 L3289, ACCEPTED UNDOCK05 L3290, SUCCEEDED detach02 L3291, SUCCEEDED detach05 L3292, ACCEPTED REACH02 L3294. Tutti questi riferimenti sono audit, non feature.

| Vista topologica | Edge | ALIGN02 | UNDOCK05 |
|---|---:|---|---|
| Pubblicata / prefix prima di UNDOCK05 | 7 | Non rappresentabile | Rappresentabile |
| Solo exact detach02 applicato | 6 | Rappresentabile | Rappresentabile |
| Tutti i detach prima di REACH02 | 5 | Rappresentabile | Non rappresentabile |

Con E2 invariato, un disconnect richiede una current connection pair. Scegliere semplicemente l'ultimo prefix del bucket non risolve il bundle. Il coalescing approvato elimina il requisito di separare i sample, ma da solo non definisce quale sottoinsieme di completion possa modificare il grafo condiviso.

**Decisione esplicita approvata: COALESCED MIXED-BUCKET DISCONNECT POLICY.** Applicare soltanto completion fisiche di primitive dispatchate prima dell'attuale bucket e completate prima della decisione successiva rilevante nel bucket. Differire i disconnect di primitive il cui dispatch appartiene al bucket corrente, anche quando completati prima della decisione successiva nello stesso bucket. Membership e ordine provengono esclusivamente da source records/provenance, dispatch IDs ed eventi fisici; mai da label, candidate failure, target, stage/FSM/control_state o futuro.

In 1189: UNDOCK02 è nel bucket precedente, UNDOCK05 in quello corrente. Applicare detach02; differire detach05. La vista condivisa passa da 7 a 6 edge e mantiene rappresentabili UNDOCK05 e ALIGN02 nella stessa candidate domain. Non è uno snapshot atomico osservato né un completo prefix globale: è la coalesced topology view approvata, con geometria misurata invariata.

Il gate architetturale è risolto; restano obbligatori i test di causalità, membership indipendente dai target e representability. Vietati hardcoding di 1189 o dei20 casi, selezione tramite ALIGN target, union di candidate table, rilassamento del disconnect su endpoint assente, eliminazione di UNDOCK05 o split del sample. Task 4 e successivi restano esclusi dalla milestone Task 1–3.

## 10. Core causale generico: solo disconnect

Modulo previsto: `drl/IL/bc_v1/causal_topology.py`, stdlib-only. Consuma i tipi generici `ConnectorRef` e `StateGraph`, senza hardware, nomi SMORES, file path, task, stage, FSM o dipendenze di episodio.

Contratto proposto:

```text
CausalBoundary(order_domain: str, before_order: int)
ConfirmedDisconnect(endpoint_a: ConnectorRef, endpoint_b: ConnectorRef,
                    evidence_id: str, order_domain: str, completed_order: int)
ReconciledTopologyResult(graph: StateGraph,
                         applied: tuple[ConfirmedDisconnect, ...],
                         already_absent: tuple[ConfirmedDisconnect, ...])
reconcile_topology(graph: StateGraph,
                   events: tuple[ConfirmedDisconnect, ...],
                   boundary: CausalBoundary) -> ReconciledTopologyResult
```

Le posizioni d'ordine sono ordinali in un dominio dichiarato, non timestamp inventati. Un evento di dominio diverso o non strettamente precedente al boundary viene rifiutato. Il core riceve eventi già ammessi dal provider, non sceglie il cutoff storico del bucket e non legge label.

Rimuovere soltanto l'exact edge attestato e aggiornare coerentemente `connected_to/status` dei due connector. Edge già assente e duplicato coerente sono no-op; un peer diverso o representations fisiche discordanti producono conflitto esplicito, senza rimozione arbitraria. Non cambiare la geometria del connector quando lo si libera. Source graph immutabile, risultati idempotenti, unrelated edges invarianti.

Nessuna aggiunta di edge DOCK, anche se la completion è nota. Nessuna ricostruzione di pose, twist, tilt/pan, posture, risorse, connector geometry, ambiente o task context. Nessuna stima cinematica. Lo stesso evento può essere prodotto in futuro da callback live; oggi si documenta soltanto l'API compatibile.

## 11. Historical SMORES evidence provider

Modulo previsto: `drl/IL/bc_v1/smores_disconnect_evidence.py`, separato dal core e dal label codec. Legge fonti storiche esattamente collegate al run, dispatch identity e completion fisiche `SUCCEEDED ... UNDOCKED detached A from B`. ACCEPTED o UNDOCK command da soli non sono prova di detach. Endpoints provengono dall'evento riuscito, non dall'ALIGN supervisionato.

L'ordine del log certificato e le source references danno il cutoff; l'uguaglianza dei physical stamps non ordina dispatch e completion. Missing o ambiguità necessari al repair fanno fallire la derivazione, non provocano un fallback future/target. Per episodi senza repair necessario, l'assenza di log opzionali è registrata e non inventa eventi.

Una successiva attachment completion già nel prefix ammesso invalida il vecchio detach per quegli endpoint: il provider non aggiunge il nuovo edge e non rimuove un nuovo peer. Reset/run boundary invalida prove non appartenenti all'attuale segmento. Prefissi successivi, stage/FSM e debug barrier non sono autorità operativa per ricostruire lo stato. Sono ammessi per l'audit storico delle categorie temporali, mai come input della policy o sostituti della completion fisica.

Il provider deve essere indipendente da action_t, supervision masks e target evidence. Current goal IDs possono ancorare il confine negli eventi registrati; non suggeriscono quale azione apprendere. La scelta del confine coalesced segue la policy approvata in §9.

## 12. Causalità e isolamento dai target

Non usare G_t+1, future graph, future observation, target graph/morphology, target TILT come misura, target DOCK come stato o outcome futuro. Nessun backward/forward fill di nuove misure. Nessun repair guidato dall'assenza di un candidato richiesto dal target.

Test necessari: derivazione invocabile senza label; a stato/evidence/boundary uguali, cambiare action_t, mask, target, stage/FSM e record futuri non cambia il grafo derivato. Mutare una completion ammessa cambia solo il relativo old edge; mutare una completion futura non cambia alcun canale corrente. L'uso di evidence historical non estende la whitelist osservabile: entra soltanto il fatto fisico derivato della disconnessione ammessa.

## 13. Policy-input contract invariato

Policy `pi(G_t, o_t, c_t) -> a_t`, feed-forward e hardware-agnostic. `PolicyInputs` e `mssr.bc_v1.policy_inputs.v1` restano invariati: `graph`, `observation`, `context`.

- G: fisica corrente whitelistata; soltanto topology/occupancy può essere riconciliata secondo il contratto.
- o: osservazioni dinamiche e geometria fisica ammessa, presenti anche durante riconfigurazione/assembly quando disponibili.
- c: task corrente assegnato online e geometria statica ammessa. L'attuale `TaskContext` ha `task_id` e `static_geometry`, **senza stage**.

Esclusi sequential/ambiguous flag, stage, FSM, barrier, executor/control_state, completion state, provenance, goal IDs, configuration_goal, target_morphology, target graph e loss masks. UNKNOWN/missing conserva la semantica D1/D2 senza usare il futuro. Task ontology resta separata dal backend. Nav2 waypoint nominali non sono posizione dei coni fisici.

E1 continua a produrre FrameGeometryView separata con scope g_current/c_static/o_dynamic. Centroide dei moduli e assi fisici causali, mai min-ID root; module geometry coerente col frame, connector body-local ed edge-relative non trasformati due volte. E2 e la regola ALIGN **compatible endpoints + currently free endpoints** restano invariati. UNDOCK resta su current connection pair. Nessuna head WAIT/no-dispatch e nessuna memoria temporale introdotte.

## 14. Contratto minimo di metadata audit

Scelta: **sidecar derivato dentro la cache G1**, referenziato e hashato dal manifest. Nessuna modifica Canonical, PolicyInputs, SupervisionBundle o CachedInputs; nessun nuovo embedding. Non duplicare il sidecar in ogni `.pt` e non aggiungere la classificazione a feature values.

Identità: `mssr.bc_v1.temporal_supervision_audit.v1`. Record minimo keyed by `(episode_id, transition_index)` con `temporal_supervision_kind` enum:

```text
single_decision
joint_single_point
temporally_coalesced_sequential
ambiguous
```

Artifact previsto: `audit/temporal_supervision.json`, con manifest source SHA, classificazioni per sample, conteggi per split, audit version e fingerprint. Mappa uno-a-uno i 119600 sample; unknown/missing/duplicato o chiave fuori split è errore, non default joint. I dati certificati possono essere estratti dagli artifact audit completi con inventario verificato; non riclassificare i 21 o i 10 boundary fini per compilare il sidecar. Se gli artifact /tmp non esistono più, fermare l'import del sidecar e recuperarne una copia verificata o riauditare read-only con autorizzazione; non inventare classificazioni.

Separatamente `audit/topology.jsonl` conserva chiavi sample, hash topologia originale/derivata, eventi applicati/no-op/differiti e proof references del provider. Nessun path entra nel core. Le20 chiavi sono oracle test, mai trigger di produzione. Quantità di label recuperate, sample riconciliati e edge rimossi sono contatori distinti, da misurare nel preflight; non assumere20 rimozioni perché sono 20 ALIGN.

G2/model futuri consumano solo gli input e candidate table approvati. Gli audit sono accessibili al reporting dopo il forward, senza influence su loss/sampler baseline o scelta delle head. La candidate table è derivata dalla stessa unica vista topologica del sample, non dal suo flag temporale.

## 15. Fingerprint e identità

| Asset congelato | Semantic fingerprint |
|---|---|
| F1 | `aee975a2f8b044ad3f57f331fc659d6cfa91d7c1e493b90b812634fb6364e81b` |
| RobotSpec | `c6e1c78d3fe5f8ede3c1d7e8ab5957799d0952580a171437db7ffcddc092ae19` |
| E2 feature schema | `43ab1b93c21b2645176e614ab1af7437f37163ec2b6047c0671d18a354b67d57` |
| F2 normalization policy | `366b9b0dfe838208462e0b99c634f0cf844c8ba15959f1bcf62b3a363261fd91` |

F1 file SHA è `f6759e6af89d732078ac656133d40236e1fd3cec8b43c92708907ed90ac965c7`, distinto dal fingerprint semantico. Tutte le identità congelate devono restare uguali.

Nuova causal derivation identity: `mssr.bc_v1.causal_topology.v1`. Fingerprint deterministico di contratto/versione, algoritmo e boundary policy approvata; provider identity separata include versione/parser/code e hashes delle evidence effettivamente lette. Nessun hash finale inventato prima dell'implementazione e della policy approvata §9.

Il temporal sidecar ha fingerprint semantico su versione, source identity e righe ordinate per chiave. SHA byte-integrity distinto. G1 cache identity include causal derivation, provider/evidence identity, temporal sidecar identity, adapter e code hashes rilevanti. Ordinamento filesystem, path assoluti della cache, timestamp e durata non determinano il fingerprint semantico. Modifica di eventi ammessi, cutoff policy o metadata invalida compatibilità; mai riusare cache con la vecchia identità.

## 16. Integrazione G1 e separazione F2

Stato reale: core/provider non implementati; G1 presente ma bloccato; cache ufficiale `artifacts/bc_v1/cache/g1_v1` e normalizer ufficiale non pubblicati. Non riportare i numeri attesi come prepare riuscito.

Pipeline futura: adapter typed published state → provider evidence → generic reconciliation → D2 select_policy_inputs → E1 transform_inputs → E2 extract_features + enumerate_candidates → label codec separato → representability → uno CachedSample per bucket. La reconciliation precede E2; l'interning dei candidati usa il dominio **derivato**, senza collisioni fra grafo occupato/libero. Label recovery continua a usare il published graph dove richiesto dal codec, senza diventare input del reconciler.

Prima del prepare costoso: gate §9 approvato,20-case regression, healthy controls e full structural preflight. Eventuali extra failure o repair non certificati bloccano la pubblicazione; non drop automatici. La nuova sidecar map deve coincidere con l'inventario BC congelato.

Il codice G1 attuale fit-ta F2 dentro `prepare_dataset` prima della pubblicazione. Per rispettare il nuovo ordine **G1 fisico completo → F2 ufficiale**, il piano separa questi due passi: prepare pubblica atomicamente cache fisica completa/non normalizzata; il comando fit successivo legge quella cache immutabile e scrive normalizer e receipt separati. Non modifica il manifest G1 pubblicato.

`complete=true` della cache fisica significa completezza di campioni, domini e audit, non training-ready. La nuova cache identity non dipende dal futuro normalizer fingerprint. Il receipt F2 lega cache fingerprint, derivation/evidence/audit identity, pins F1/RobotSpec/E2/policy e normalizer fingerprint/hash. Il training richiede entrambe le fasi certificate. Mantiene rollback/no-clobber di ciascuna pubblicazione; un F2 fallito lascia integra G1 e non pubblica normalizer incompleti.

Il manifest e i payload della nuova cache usano `mssr.bc_v1.physical_cache.v2`: il nuovo significato di completezza e la separazione dell'identità F2 non vengono introdotti silenziosamente come formato v1. I tipi CachedSample/PolicyInputs e lo schema feature E2 restano identici. Questa versione è un artifact derivato G1, non una nuova versione Canonical.

## 17. Implicazioni F2

F2 engine e policy invariati. `fit_normalizers(...)`, NormalizationSample, writer/reader e formule esistenti vengono riusati senza nuova politica statistica. Wrapper generico separato costruisce factory **solo train** dalla cache fisica,49 episodi / 78302 sample, complete_train_coverage=true. Validation/test non vengono aperti dal fitter; tutte le 119600 righe sono invece legittimamente preparate da G1.

No sample weighting per source coverage o temporal kind. Un sample coalesced contribuisce una volta con le sue supervisioni conservate. Features topologiche derivano dalla nuova vista: non riusare statistics della vecchia vista solo perché il feature schema coincide. Il receipt esterno lega esplicitamente le statistiche alla vista G1, senza cambiare il fingerprint della policy F2 o il suo engine. Non eseguire fit in questo task documentale.

## 18. Future data collection

Nuove demo/DAgger dovranno registrare current physical snapshot identity, snapshot pre-decision esplicito, dispatch ordering, completion ordering, graph publication/receipt e decision_group_id o wave/batch identity. Ordinare snapshot, decision/action, lifecycle e post-state senza equiparare physical stamp e dispatch ID. Un batch realmente concorrente condivide il proprio pre-state; un nuovo group ha un nuovo riferimento causale.

Documentazione soltanto: nessun recorder/runtime/executor fix, raccolta nuove demo, installazione/integration Isaac Lab o conversione HDF5 oggi.

## 19. Ablation futura

Baseline tutti 119600 sample versus ablation che esclude i soli 309 temporally_coalesced_sequential, per misurare l'effetto dello 0.258%. Non eseguirla ora; i 21 ambigui restano separatamente dichiarati, senza nuova classificazione. F1 originale non viene riscritto. L'esperimento avrà propria eligibility/identity e statistiche train-only; non riutilizzerà una baseline cache/normalizer come se membership fosse identica. I counts della vista ablation sarebbero train 78090, validation 22716, test 18485, totale 119291; questi non sono i counts BC-v1 attuali.

## 20. Non-goals e gate finale

Esclusi: Canonical v7, modifiche RAW/compact/Canonical/label/index/F1, split dei 309, one-label-per-sample, filtro dei 21, aggiunta DOCK, posture/joint/pose repair, label-driven candidate admission, WAIT/no-dispatch, G2/collate/DataLoader/sampler, rete/loss/trainer/HPO, FreeBOT backend completo, ROS/Isaac/Lab, runtime/recorder e commit/push.

Gate prima di dichiarare G1/F2 pronti:20/20 ALIGN e 1110 target rappresentabili insieme, zero mapping failure ; 119600 sample/73 episodi; F1 pins e counts invariati; 309+21 retained e audit-only; input whitelist invariata; Canonical SHA invariato; F2 legge soltanto 49/78302 train; cache/normalizer/receipt pubblicati e verificati con nuove identity. Tutto questo è requisito futuro, non attestazione di GREEN oggi.

## 21. Self-review documentale

- [x] Le309 sono sequenziali coalesced, non simultanee; nessuno split o nuovo Canonical.
- [x] Nessun target come misura, G_t+1/futuro nel repair o DOCK edge addition.
- [x] Temporal flags, stage/FSM, lifecycle/provenance e goal IDs esclusi da PolicyInputs.
- [x] Core hardware-agnostic, current graph immutabile, evidence separata e exact-pair removal soltanto.
- [x] I20 stale ALIGN sono distinti dai 22 storici e dai309; categorie temporal ambiguity e target ambiguity distinte.
- [x] Counts e pins attuali prevalgono sui documenti storici v5; split e normalizzazione policy non rigenerati.
- [x] F2 già accoppiato a prepare nel codice: separazione esplicita nel piano, senza dichiararlo già separato.
- [x] Confine 1189 approvato il 2026-10-12: policy coalesced esplicita, entrambe le label conservate, nessuna affermazione di snapshot atomico.
- [x] Nessun task applicativo, prepare, fit o test TDD eseguito per produrre questa spec. Self-review documentale, non prova di correttezza del futuro fix.
