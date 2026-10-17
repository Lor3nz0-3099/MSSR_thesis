# MSSR BC-v1 F1 — split stratificato per episodio, v1

F1 COMPLETE. Primo holdout ufficiale Canonical v6, seed20261006:49/12/12. Nessun commit/push/F2/cross-validation.

## A. File creati

Tutti nuovi, nessun file preesistente modificato:

- `drl/IL/bc_v1/split.py`: core stdlib, grouping/holdout, validazione, fingerprint, lettura/scrittura frozen, diagnostica.
- `drl/IL/tests/bc_v1/test_split.py`:39 test.
- `configs/bc_v1/training.json`: soltanto sorgente e split F1, strati/quote dichiarativi e pin ufficiale del manifest.
- `artifacts/bc_v1/splits/f1_v1/episodes.json`:80 descrittori offline certificati per identità, provenienza raw/compact e conteggi B2.
- `artifacts/bc_v1/splits/f1_v1/split.json`: manifest ufficiale.
- `artifacts/bc_v1/splits/f1_v1/audit.json`: diagnostica post-split.
- `artifacts/bc_v1/splits/f1_v1/report.md`: questo report.

Artifact derivati fuori dalla sorgente, schema/version espliciti e pronti per la revisione/versionamento Git; nessun commit effettuato.

## B. Algoritmo

Si validano gli80 descrittori episodio (ID univoco, lineage autorevole, hash SHA256, count BC non negativo), gli alias e gli hash delle rappresentazioni sorgenti. Identità/hash condivisi tra lineage dichiarate diverse, hash incompatibili della stessa rappresentazione o duplicati producono errore. Si selezionano soltanto episodi con count BC>0:73 eleggibili,7 esclusi. Ogni eleggibile deve corrispondere a uno e un solo strato configurato. Si raggruppa per lineage; un gruppo non può attraversare strati incompatibili. Gruppi ordinati deterministicamente per identità e membri, strati ordinati per nome; un solo RNG locale20261006 mescola ogni catalogo di gruppi. Un allocatore esatto con memoizzazione assegna gruppi interi alle quote: il peso è il numero di episodi eleggibili, mai timestep o label. Se i pesi non permettono le quote, fallisce. Nessun retry di seed, nessun aggiustamento delle quote. La membership viene restituita come tuple di ID ordinate. Nessun RNG globale modificato, dipendenza dall’ordine di filesystem/input/dict o branch hardware/task nel core.

Il dataset reale ha73 gruppi eleggibili singleton: raw/compact sono rappresentazioni della stessa dimostrazione, non episodi aggiuntivi. Verificati raw/compact manifest, source-run/episode alias, dichiarazioni degli stream e identità del primo record Canonical di ciascun episodio. Nessun source-run/path/hash condiviso incoerente trovato.140 piccoli metadata file verificati per SHA256; per legacy teleop-f888… mancavano i digest dei tre stream conservati: calcolati read-only soltanto quelli. Non raggruppiamo dimostrazioni indipendenti per la sola somiglianza dell’ambiente o dei tre assembly.

## C. Conteggi esatti

| Split | Episodi | Timestep BC | Expert | Teleop |
|---|---:|---:|---:|---:|
| train | 49 | 78302 | 34 | 15 |
| validation | 12 | 22762 | 9 | 3 |
| test | 12 | 18536 | 9 | 3 |
| Totale |73|119600|52|21|

Union esattamente i73 BC utili, partizioni pairwise disjoint;7 expert button BC=0 esclusi. L’unità è sempre l’episodio/lineage, non la transition.

## D. Sei strati

| Strato | Totale | Train | Validation | Test |
|---|---:|---:|---:|---:|
| expert_assembly |3|1|1|1|
| expert_gap |18|12|3|3|
| expert_navigation |13|9|2|2|
| expert_stairs |18|12|3|3|
| teleop_button |8|6|1|1|
| teleop_mission |13|9|2|2|

Le definizioni sono config/data: expert_navigation usa il task storico `rc_car` soltanto nel config; teleop_profile deriva offline da course=BUTTON oppure missione dichiarata. Legacy f888… non ha course nel manifest: la missione multi-task è già certificata dal protocollo approvato in task_context.json. Questi tag, lineage, source/hash e split non diventano PolicyInputs. Nessun target morphology, stage o control_state aggiunto alla policy.

## E. Elenco esatto degli episode_id

### train (49)

```text
expert:assembly:seed-006307
expert:gap:seed-004100
expert:gap:seed-004102
expert:gap:seed-004103
expert:gap:seed-004105
expert:gap:seed-004106
expert:gap:seed-004108
expert:gap:seed-004111
expert:gap:seed-004303
expert:gap:seed-005998
expert:gap:seed-006911
expert:gap:seed-007564
expert:gap:seed-012398
expert:rc_car:seed-005100
expert:rc_car:seed-005101
expert:rc_car:seed-005102
expert:rc_car:seed-005103
expert:rc_car:seed-005104
expert:rc_car:seed-006001
expert:rc_car:seed-006017
expert:rc_car:seed-006053
expert:rc_car:seed-006056
expert:stairs:seed-003102
expert:stairs:seed-003104
expert:stairs:seed-003105
expert:stairs:seed-003107
expert:stairs:seed-003108
expert:stairs:seed-003109
expert:stairs:seed-003110
expert:stairs:seed-003292
expert:stairs:seed-004007
expert:stairs:seed-004047
expert:stairs:seed-006403
expert:stairs:seed-006922
teleop-0e4e34fe3a5f4ed1b6a86642acf4fac8
teleop-1f8b406b56f74377834cd863ebccf696
teleop-39e2f4b4cd684e9ea72734cc21baca94
teleop-5bb5a4712d924617aab5604b6deb20b4
teleop-5e0c4d97db80492bbb561b16ef132949
teleop-69b37b8db7e141339518d4eef585e4bd
teleop-8b55584534d24c86888b350e09c74e7b
teleop-952c36750c624fb19de8421019325d13
teleop-a7d7e08f5fd14a59bd3af6afb53ccde1
teleop-adcf33433f954669838220b9f617d952
teleop-beaa63ebe5f347c6abc58baff9d1394a
teleop-c78e0455357147369ad16bc1bf39a0c2
teleop-dd6bf8b9926d4a2aab6af7572646d026
teleop-edbefc66cecf40c180864b452aff8dab
teleop-f8888f12c46249fc80ef0af8c56b4dcf
```

### validation (12)

```text
expert:assembly:seed-006217
expert:gap:seed-004104
expert:gap:seed-004109
expert:gap:seed-005855
expert:rc_car:seed-005105
expert:rc_car:seed-005106
expert:stairs:seed-003100
expert:stairs:seed-003111
expert:stairs:seed-004689
teleop-170338d489a64eb397a10cf9910392a1
teleop-1bea8f28c1c04323a4c3ec7e2829d07c
teleop-fb69637be28d49ccbff572e721cfae24
```

### test (12)

```text
expert:assembly:seed-006284
expert:gap:seed-004107
expert:gap:seed-004110
expert:gap:seed-014544
expert:rc_car:seed-006000
expert:rc_car:seed-006004
expert:stairs:seed-003101
expert:stairs:seed-003106
expert:stairs:seed-004676
teleop-3aa4f69128774d5eba1f8d851c02cf53
teleop-b1456ab292aa4a319c9d4bcc17e7e57f
teleop-ce8293656bb444ac999f87c0520672fc
```

## F. Manifest e fingerprint

Riferimento ufficiale: `artifacts/bc_v1/splits/f1_v1/split.json`. Schema `mssr.bc_v1.episode_split.v1`, algoritmo `stratified_grouped_holdout.v1`. Contiene identità dataset/schema/hash, seed, config/quote, membership completa, strato/lineage/hash/sorgenti di ogni episodio, gruppo, esclusioni e conteggi globali/per-strato.

- Config strata fingerprint: `aed24e3953f9c4d25165f762e4efc3089967a1e38b6f88753201dc3afa7e5120`.
- Manifest fingerprint semantico: `aee975a2f8b044ad3f57f331fc659d6cfa91d7c1e493b90b812634fb6364e81b`.
- SHA256 dei byte del file: `f6759e6af89d732078ac656133d40236e1fd3cec8b43c92708907ed90ac965c7`.

Fingerprint semantico = SHA256 JSON canonico a chiavi ordinate, UTF-8, senza campo self-fingerprint; nessun timestamp casuale. Le collezioni catalogo degli strati e gli ID sono ordinati. Il writer crea esclusivamente e accetta un file esistente solo se i byte sono identici; rifiuta overwrite diverso e output dentro la sorgente. Il reader certifica dati/config/membership e controlla seed/fingerprint attesi indipendenti: le chiamate ufficiali passano entrambi i pin da training.json. Aggiungere demo o cambiare sorgente/config/metadata produce incompatibilità esplicita, non una nuova membership silenziosa. F2 e task futuri devono leggere questo file usando i pin, senza reinventare lo split.

## G. Canonical v6

SHA256 manifest verificato prima/dopo: `31765272b6ecd7506c54b25e214b75d7b1b28691803ddfc286dafcce8b4ec562`. Dataset80 episodi (59expert/21teleop),197240 transition Canonical,119600 BC-valid,73 BC utili. Il dataset si chiama v6, ma i suoi schema interni collection/transition rimangono v5: identità preservata fedelmente, nessuna rinomina/rebuild. Tutte80 firme inode/size/mtime_ns/ctime_ns episodio identiche prima/dopo. Hash payload degli episodi sono quelli della sorgente immutabile approvata; nessun full rehash/census Canonical in F1.

## H. Audit diagnostico

Conteggi sotto sono label per modulo/risorsa con mask di supervisione, importati dal censimento B2 congelato verificato per ogni episodio/hash/bytes/count; non command_instances né componenti scalari. Somme dei tre split coincidono con tutti i conteggi B2. Audit eseguito DOPO la membership e mai usato per modificarla.

| Famiglia | Train label (episodi con supporto) | Validation | Test |
|---|---:|---:|---:|
| wheel_velocity | 374122 (48) | 105940 (11) | 96880 (11) |
| pan_velocity | 121016 (18) | 38712 (4) | 26756 (4) |
| pan_target | 527 (13) | 47 (3) | 37 (2) |
| tilt_target | 58324 (49) | 15759 (12) | 10677 (12) |
| align_faces | 332 (48) | 86 (11) | 85 (11) |
| dock | 313 (48) | 83 (11) | 81 (11) |
| undock | 60 (9) | 24 (2) | 22 (2) |
| gravity_settle | 15 (8) | 6 (2) | 3 (2) |
| tilt_velocity | 0 (0) | 0 (0) | 0 (0) |
| pan_delta | 0 (0) | 0 (0) | 0 (0) |
| tilt_delta | 0 (0) | 0 (0) | 0 (0) |

Piccoli/grandi, timestep BC:

| Split | Minimo | Massimo |
|---|---|---|
|train|expert:assembly:seed-006307 (19)|teleop-5e0c4d97db80492bbb561b16ef132949 (9725)|
|validation|expert:assembly:seed-006217 (19)|teleop-fb69637be28d49ccbff572e721cfae24 (12350)|
|test|expert:assembly:seed-006284 (19)|teleop-ce8293656bb444ac999f87c0520672fc (10384)|

Finding diagnostici: dimensioni molto diseguali tra demo, come previsto; validation ha22762 righe contro18536 test malgrado entrambi12 episodi. I teleop compositi dominano le righe, mentre assembly ha19 righe BC per demo. GRAVITY_SETTLE mantiene supporto15/6/3 label e8/2/2 episodi: nessuno split senza supporto. PAN_TARGET ha527/47/37 label, quindi supporto limitato in holdout. tilt_velocity/pan_delta/tilt_delta hanno zero positivi in tutti i tre split, ma restano capability valide del RobotSpec; nessun pruning. Nessuna scelta di seed/quote/feature/hyperparameter in base a questi numeri o al test.

## I. RED osservato

Prima implementazione:37failed, errore assertion «Missing F1 grouped episode split», nessun collection error (`/tmp/mssr-f1-audit/red.txt`). Prima produzione scritta soltanto dopo questo RED.

Review finale:0Critical,1Important,0Minor. Il reader accettava un manifest autoconsistente generato con20261007 al posto di20261006. Reproduzione RED:3failed/36deselected (alternate seed accettato, parametri freeze assenti, pin ufficiale assente dal config). Corretto in un unico pass minimo: reader con expected_seed/expected_fingerprint, config pin del fingerprint ufficiale già esistente e passaggio dei pin nei read ufficiali. Membership/artifact originali identici, nessuna seconda review, nessun finding Important rinviato.

## J–K. GREEN e regressioni

Tutti i comandi finali exit0, output letti:

| Gate | Passed | Skipped | Failed |
|---|---:|---:|---:|
| test_split.py |39|0|0|
| test_contract.py +test_policy_inputs.py +test_task_context.py |246|0|0|
| drl/IL/tests/bc_v1 |989|6|0|
| drl/IL |1030|6|0|

Comandi `env PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -p no:cacheprovider <path> -q`; suite BC-v1/full IL aggiungono `MSSR_BC_V1_SKIP_REAL_CENSUS=1`. I6 skip sono soltanto i test B2 della fixture census reale esclusa. Non sommare gate sovrapposti. Prima del fix review:37pass dedicati e987pass/6skip BC-v1. Dopo il fix tutte le suite sopra rieseguite. Log finali: `/tmp/mssr-f1-audit/{green-final,regressions-final,bc-v1-final,full-il-final}.txt`. Import isolation esegue il core con NumPy/Torch/ROS/Isaac/backend/reader/policy input import bloccati. `git diff --check` exit0.

## L. Rulings e limiti

- Delta approvato prevale sul piano storico v5/70/48-11-11: si usa v6/73/49-12-12 e lo strato assembly1/1/1. Nessuna vecchia membership preservata.
- Lavoro in-place sul branch richiesto e nessun commit/worktree/install: mantenuta la situazione dirty preesistente e lo scope autorizzato.
- Source schema rimane v5 come dichiara il manifest v6; task rc_car/teleop profile sono metadata di stratificazione offline/config, mai logica di policy o branch core.
- Conteggi diagnostici dai B2 expected già congelati, coerenti con identità Canonical. F1 non è un nuovo censimento o audit del lifecycle/stato pre-decisione. Nuovo hash solo dove mancava per tre payload legacy.
- Lineage dichiarate devono essere certificate dal preparatore offline: il core non indovina la genealogia da numeri di seed/nomi robot/difficoltà/target. Collisioni esplicite falliscono. Il dataset attuale ha gruppi singleton; test sintetici esercitano gruppi multi-rappresentazione e quote pesate, incluse incompatibilità.
- Questo holdout è versionato e pinned; futura grouped5-fold richiederà un task separato e test finale sigillato. Metadata/grouping esposti restano riusabili. Nessun supporto CV implementato.
- Reviewer non valuta F2/training, un diverso seed per bilanciare le metriche, modifiche dirty estranee o un full rehash payload: confini approvati rispettati, non difetti rinviati. Nessun Minor rinviato.

## M. Git e scope finale

Branch `wip/t8-c05-human-teleop-20260921`, HEAD `b50c543731e88ef09a18db7b68ad461056795749` invariati.551 file preesistenti hanno SHA256 invariato; nessuna modifica/rimozione. I7 file nuovi sono elencati inA. `git diff` sui tracked è identico byte-per-byte alla baseline: contiene soltanto i7 dirty file preesistenti (parallel_assembly_executor.py, smores_parallel_self_assembly_node.py, test_parallel_assembly_executor.py, build_expert_v1.py, run_button_expert_to_ik.py, button_ik_math.py, test_button_ik_source_contract.py), preservati. I file F1 sono untracked insieme ai precedenti BC-v1: nessun staging. Git status aggiunge la directory artifacts/, senza cancellare o ripristinare alcun untracked. Stato/diff completi e safety evidence in `/tmp/mssr-f1-audit/`.

## N. Conferma finale

**F1 COMPLETE.** No commit; no push; no modifica Canonical v6/demo/label; no F2; no normalizzazione; no trainer/HPO/model; no cross-validation. Runtime/executor/recorder/Isaac non toccati. STOP dopo F1.
