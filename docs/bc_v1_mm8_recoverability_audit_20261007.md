# Audit limitato MM8: certificazione delle dimostrazioni complete — 2026-10-07

Esito **C: nessuno degli otto assembly è recuperabile come dimostrazione completa rigorosa con le evidenze attualmente disponibili**. Nessun dato, label, mask o filtro è stato modificato. A1 può procedere per autorizzazione esplicita successiva a questo audit.

## Criterio di certificazione

Una decisione è certa se il payload iniziale REACH (label ALIGN_FACES) o DOCK è registrato nello stesso tick del grafo acquisito prima di executor.step e della pubblicazione. Il grafo fisico Canonical è stato confrontato con la sorgente per pose, velocità e coordinate attuatore; la connessione della stessa operazione è assente. I 60 target diretti superano questi controlli. control_state storico, target graph e piano non diventano input.

Un active_goal_id certifica un dispatch avvenuto, non il suo grafo completo. Il codice storico al commit `4d408111e875526ef682ae7f72aa2eef27c1559d` conserva solo il primo pending per stamp; ricostruisce però current_graph a ogni tick da observation e graph_payload aggiornati con callback indipendenti. Lo stamp deriva prioritariamente dalla observation e non è un hash/versione atomica del grafo. Non è quindi dimostrabile che il grafo del primo pending sia identico a quello di un dispatch successivo perso allo stesso stamp. I log ACCEPTED sono privi di un collegamento a snapshot completo/decision_seq. Topologia invariata non certifica pose, coordinate e contenuto pre-decisione.

Classifica per numero totale di decisioni direttamente certificate, decrescente; parità risolta privilegiando la presenza di ID nel flusso strutturale, poi numero button. È una classifica di evidenza residua: nessuna riga soddisfa 7+7.

| Ordine | Button / episode id | ALIGN certi | DOCK certi | Recovery aggiuntivi con G certificabile | Restano incerti |
|---|---|---:|---:|---:|---|
| 1 | button_04 / `teleop-8b55584534d24c86888b350e09c74e7b` | 4/7 | 6/7 | 0 | 3 ALIGN + 1 DOCK; 0 DOCK senza ID nel flusso strutturale |
| 2 | button_07 / `teleop-dd6bf8b9926d4a2aab6af7572646d026` | 4/7 | 5/7 | 0 | 3 ALIGN + 2 DOCK; 2 DOCK senza ID nel flusso strutturale |
| 3 | button_05 / `teleop-b1456ab292aa4a319c9d4bcc17e7e57f` | 5/7 | 3/7 | 0 | 2 ALIGN + 4 DOCK; 1 DOCK senza ID nel flusso strutturale |
| 4 | button_01 / `teleop-c78e0455357147369ad16bc1bf39a0c2` | 4/7 | 3/7 | 0 | 3 ALIGN + 4 DOCK; 0 DOCK senza ID nel flusso strutturale |
| 5 | button_03 / `teleop-170338d489a64eb397a10cf9910392a1` | 4/7 | 3/7 | 0 | 3 ALIGN + 4 DOCK; 0 DOCK senza ID nel flusso strutturale |
| 6 | button_08 / `teleop-edbefc66cecf40c180864b452aff8dab` | 5/7 | 2/7 | 0 | 2 ALIGN + 5 DOCK; 1 DOCK senza ID nel flusso strutturale |
| 7 | button_02 / `teleop-1f8b406b56f74377834cd863ebccf696` | 3/7 | 3/7 | 0 | 4 ALIGN + 4 DOCK; 0 DOCK senza ID nel flusso strutturale |
| 8 | button_06 / `teleop-beaa63ebe5f347c6abc58baff9d1394a` | 3/7 | 3/7 | 0 | 4 ALIGN + 4 DOCK; 0 DOCK senza ID nel flusso strutturale |

Totali: 32 ALIGN + 28 DOCK certificate direttamente; 24 ALIGN + 28 DOCK non certificate temporalmente. Sono 48 casi con active ID e 4 DOCK senza active ID nel JSONL. **Novità rispetto all’audit precedente:** i quattro DOCK sono confermati nominalmente nei log Isaac, insieme a tutti gli altri: 112/112 ACCEPTED. Questo recupera evidenza dell’evento, senza recuperare G_t.

## Punti ancora incerti, episodio per episodio

`w/a` sono indici di wave/action, non ID dei moduli. Tutti i punti sotto richiederebbero un’ulteriore prova del grafo al dispatch; non sono automaticamente supervisione valida.

### button_04 — teleop-8b55584534d24c86888b350e09c74e7b

Sorgente: [teleop-self_assembly-1791277741283844569.jsonl](//home/lorenzo/MSSR_thesis/datasets/expert_v1/teleop/episodes/teleop-8b55584534d24c86888b350e09c74e7b/structural/teleop-self_assembly-1791277741283844569.jsonl). SHA-256 `d0df65cc8daf8409217157022144da4fa2774e5fcbdd27fb417c01e1d187f7cb`.

| Decisione | Primo active ID / precedente candidato | Evidenza runtime ACCEPTED |
|---|---|---|
| ALIGN w1/a0 | L468 / L467 | [L406](/home/lorenzo/MSSR_thesis/logs/teleop/runs/button-04-20261006-110837.d1G95S/isaac.log:406) |
| ALIGN w1/a1 | L469 / L468 | [L407](/home/lorenzo/MSSR_thesis/logs/teleop/runs/button-04-20261006-110837.d1G95S/isaac.log:407) |
| DOCK w1/a1 | L979 / L978 | [L442](/home/lorenzo/MSSR_thesis/logs/teleop/runs/button-04-20261006-110837.d1G95S/isaac.log:442) |
| ALIGN w2/a0 | L983 / L982 | [L445](/home/lorenzo/MSSR_thesis/logs/teleop/runs/button-04-20261006-110837.d1G95S/isaac.log:445) |

### button_07 — teleop-dd6bf8b9926d4a2aab6af7572646d026

Sorgente: [teleop-self_assembly-1791281675662209143.jsonl](//home/lorenzo/MSSR_thesis/datasets/expert_v1/teleop/episodes/teleop-dd6bf8b9926d4a2aab6af7572646d026/structural/teleop-self_assembly-1791281675662209143.jsonl). SHA-256 `15602722fe1c9f6a013ddc0219c248815cf6e4120f5963dbed9db9ee705520d3`.

| Decisione | Primo active ID / precedente candidato | Evidenza runtime ACCEPTED |
|---|---|---|
| DOCK w1/a0 | Assenti nel JSONL | [L442](/home/lorenzo/MSSR_thesis/logs/teleop/runs/button-07-20261006-121407.Iiv9al/isaac.log:442) |
| ALIGN w1/a1 | L470 / L469 | [L410](/home/lorenzo/MSSR_thesis/logs/teleop/runs/button-07-20261006-121407.Iiv9al/isaac.log:410) |
| ALIGN w2/a0 | L979 / L978 | [L448](/home/lorenzo/MSSR_thesis/logs/teleop/runs/button-07-20261006-121407.Iiv9al/isaac.log:448) |
| DOCK w2/a0 | Assenti nel JSONL | [L472](/home/lorenzo/MSSR_thesis/logs/teleop/runs/button-07-20261006-121407.Iiv9al/isaac.log:472) |
| ALIGN w4/a0 | L1824 / L1823 | [L498](/home/lorenzo/MSSR_thesis/logs/teleop/runs/button-07-20261006-121407.Iiv9al/isaac.log:498) |

### button_05 — teleop-b1456ab292aa4a319c9d4bcc17e7e57f

Sorgente: [teleop-self_assembly-1791281380803603286.jsonl](//home/lorenzo/MSSR_thesis/datasets/expert_v1/teleop/episodes/teleop-b1456ab292aa4a319c9d4bcc17e7e57f/structural/teleop-self_assembly-1791281380803603286.jsonl). SHA-256 `6c819d5cf27bdf8a2ebe9ff0a2d4578de7dc9cafc71629a2813538b8fe522441`.

| Decisione | Primo active ID / precedente candidato | Evidenza runtime ACCEPTED |
|---|---|---|
| DOCK w0/a0 | Assenti nel JSONL | [L398](/home/lorenzo/MSSR_thesis/logs/teleop/runs/button-05-20261006-120922.hGoie9/isaac.log:398) |
| ALIGN w1/a0 | L464 / L463 | [L404](/home/lorenzo/MSSR_thesis/logs/teleop/runs/button-05-20261006-120922.hGoie9/isaac.log:404) |
| DOCK w1/a1 | L976 / L975 | [L438](/home/lorenzo/MSSR_thesis/logs/teleop/runs/button-05-20261006-120922.hGoie9/isaac.log:438) |
| DOCK w3/a0 | L1820 / L1819 | [L491](/home/lorenzo/MSSR_thesis/logs/teleop/runs/button-05-20261006-120922.hGoie9/isaac.log:491) |
| ALIGN w4/a0 | L1825 / L1824 | [L494](/home/lorenzo/MSSR_thesis/logs/teleop/runs/button-05-20261006-120922.hGoie9/isaac.log:494) |
| DOCK w4/a0 | L2118 / L2117 | [L511](/home/lorenzo/MSSR_thesis/logs/teleop/runs/button-05-20261006-120922.hGoie9/isaac.log:511) |

### button_01 — teleop-c78e0455357147369ad16bc1bf39a0c2

Sorgente: [teleop-self_assembly-1791280535089537950.jsonl](//home/lorenzo/MSSR_thesis/datasets/expert_v1/teleop/episodes/teleop-c78e0455357147369ad16bc1bf39a0c2/structural/teleop-self_assembly-1791280535089537950.jsonl). SHA-256 `78ab5594e462e4957e8ef22b9ee8607a0059793a437d2fdc3b247863c0fa08d3`.

| Decisione | Primo active ID / precedente candidato | Evidenza runtime ACCEPTED |
|---|---|---|
| DOCK w0/a1 | L466 / L465 | [L399](/home/lorenzo/MSSR_thesis/logs/teleop/runs/button-01-20261006-115518.i1wrxF/isaac.log:399) |
| DOCK w1/a0 | L980 / L979 | [L435](/home/lorenzo/MSSR_thesis/logs/teleop/runs/button-01-20261006-115518.i1wrxF/isaac.log:435) |
| ALIGN w1/a1 | L475 / L474 | [L403](/home/lorenzo/MSSR_thesis/logs/teleop/runs/button-01-20261006-115518.i1wrxF/isaac.log:403) |
| DOCK w1/a1 | L981 / L980 | [L438](/home/lorenzo/MSSR_thesis/logs/teleop/runs/button-01-20261006-115518.i1wrxF/isaac.log:438) |
| ALIGN w3/a0 | L1472 / L1471 | [L468](/home/lorenzo/MSSR_thesis/logs/teleop/runs/button-01-20261006-115518.i1wrxF/isaac.log:468) |
| DOCK w3/a0 | L1827 / L1826 | [L488](/home/lorenzo/MSSR_thesis/logs/teleop/runs/button-01-20261006-115518.i1wrxF/isaac.log:488) |
| ALIGN w4/a0 | L1830 / L1829 | [L491](/home/lorenzo/MSSR_thesis/logs/teleop/runs/button-01-20261006-115518.i1wrxF/isaac.log:491) |

### button_03 — teleop-170338d489a64eb397a10cf9910392a1

Sorgente: [teleop-self_assembly-1791277307509730992.jsonl](//home/lorenzo/MSSR_thesis/datasets/expert_v1/teleop/episodes/teleop-170338d489a64eb397a10cf9910392a1/structural/teleop-self_assembly-1791277307509730992.jsonl). SHA-256 `21e6e70d069e0703c9b72c03e5873cb6a6b09ce311cf63ac7fbb4c649223831b`.

| Decisione | Primo active ID / precedente candidato | Evidenza runtime ACCEPTED |
|---|---|---|
| DOCK w0/a0 | L468 / L467 | [L400](/home/lorenzo/MSSR_thesis/logs/teleop/runs/button-03-20261006-110118.R0Vye3/isaac.log:400) |
| ALIGN w0/a1 | L3 / L2 | [L369](/home/lorenzo/MSSR_thesis/logs/teleop/runs/button-03-20261006-110118.R0Vye3/isaac.log:369) |
| DOCK w1/a0 | L983 / L982 | [L439](/home/lorenzo/MSSR_thesis/logs/teleop/runs/button-03-20261006-110118.R0Vye3/isaac.log:439) |
| DOCK w1/a1 | L984 / L983 | [L442](/home/lorenzo/MSSR_thesis/logs/teleop/runs/button-03-20261006-110118.R0Vye3/isaac.log:442) |
| ALIGN w3/a0 | L1482 / L1481 | [L473](/home/lorenzo/MSSR_thesis/logs/teleop/runs/button-03-20261006-110118.R0Vye3/isaac.log:473) |
| ALIGN w4/a0 | L1838 / L1837 | [L495](/home/lorenzo/MSSR_thesis/logs/teleop/runs/button-03-20261006-110118.R0Vye3/isaac.log:495) |
| DOCK w4/a0 | L2135 / L2134 | [L513](/home/lorenzo/MSSR_thesis/logs/teleop/runs/button-03-20261006-110118.R0Vye3/isaac.log:513) |

### button_08 — teleop-edbefc66cecf40c180864b452aff8dab

Sorgente: [teleop-self_assembly-1791278705931892690.jsonl](//home/lorenzo/MSSR_thesis/datasets/expert_v1/teleop/episodes/teleop-edbefc66cecf40c180864b452aff8dab/structural/teleop-self_assembly-1791278705931892690.jsonl). SHA-256 `9ffae09eb152acb1e89f04f3bc54d998ecc1077012c11346dc3167eaade2abc5`.

| Decisione | Primo active ID / precedente candidato | Evidenza runtime ACCEPTED |
|---|---|---|
| ALIGN w0/a1 | L3 / L2 | [L367](/home/lorenzo/MSSR_thesis/logs/teleop/runs/button-08-20261006-112447.r0C9Dc/isaac.log:367) |
| DOCK w0/a1 | L464 / L463 | [L401](/home/lorenzo/MSSR_thesis/logs/teleop/runs/button-08-20261006-112447.r0C9Dc/isaac.log:401) |
| DOCK w1/a1 | L978 / L977 | [L438](/home/lorenzo/MSSR_thesis/logs/teleop/runs/button-08-20261006-112447.r0C9Dc/isaac.log:438) |
| DOCK w2/a0 | L1480 / L1479 | [L467](/home/lorenzo/MSSR_thesis/logs/teleop/runs/button-08-20261006-112447.r0C9Dc/isaac.log:467) |
| ALIGN w3/a0 | L1484 / L1483 | [L471](/home/lorenzo/MSSR_thesis/logs/teleop/runs/button-08-20261006-112447.r0C9Dc/isaac.log:471) |
| DOCK w3/a0 | Assenti nel JSONL | [L490](/home/lorenzo/MSSR_thesis/logs/teleop/runs/button-08-20261006-112447.r0C9Dc/isaac.log:490) |
| DOCK w4/a0 | L2133 / L2132 | [L511](/home/lorenzo/MSSR_thesis/logs/teleop/runs/button-08-20261006-112447.r0C9Dc/isaac.log:511) |

### button_02 — teleop-1f8b406b56f74377834cd863ebccf696

Sorgente: [teleop-self_assembly-1791216463965830663.jsonl](//home/lorenzo/MSSR_thesis/datasets/expert_v1/teleop/episodes/teleop-1f8b406b56f74377834cd863ebccf696/structural/teleop-self_assembly-1791216463965830663.jsonl). SHA-256 `901df1c2dec4de1ef70fc0f17b65796a52e58c054ddf52118ca61907374eaa24`.

| Decisione | Primo active ID / precedente candidato | Evidenza runtime ACCEPTED |
|---|---|---|
| ALIGN w1/a0 | L466 / L465 | [L403](/home/lorenzo/MSSR_thesis/logs/teleop/runs/button-02-20261005-180726.WpQRHH/isaac.log:403) |
| DOCK w1/a0 | L974 / L973 | [L435](/home/lorenzo/MSSR_thesis/logs/teleop/runs/button-02-20261005-180726.WpQRHH/isaac.log:435) |
| DOCK w1/a1 | L975 / L974 | [L437](/home/lorenzo/MSSR_thesis/logs/teleop/runs/button-02-20261005-180726.WpQRHH/isaac.log:437) |
| ALIGN w2/a0 | L979 / L978 | [L442](/home/lorenzo/MSSR_thesis/logs/teleop/runs/button-02-20261005-180726.WpQRHH/isaac.log:442) |
| DOCK w2/a0 | L1471 / L1470 | [L466](/home/lorenzo/MSSR_thesis/logs/teleop/runs/button-02-20261005-180726.WpQRHH/isaac.log:466) |
| ALIGN w3/a0 | L1473 / L1472 | [L469](/home/lorenzo/MSSR_thesis/logs/teleop/runs/button-02-20261005-180726.WpQRHH/isaac.log:469) |
| DOCK w3/a0 | L1822 / L1821 | [L489](/home/lorenzo/MSSR_thesis/logs/teleop/runs/button-02-20261005-180726.WpQRHH/isaac.log:489) |
| ALIGN w4/a0 | L1823 / L1822 | [L492](/home/lorenzo/MSSR_thesis/logs/teleop/runs/button-02-20261005-180726.WpQRHH/isaac.log:492) |

### button_06 — teleop-beaa63ebe5f347c6abc58baff9d1394a

Sorgente: [teleop-self_assembly-1791278310185191810.jsonl](//home/lorenzo/MSSR_thesis/datasets/expert_v1/teleop/episodes/teleop-beaa63ebe5f347c6abc58baff9d1394a/structural/teleop-self_assembly-1791278310185191810.jsonl). SHA-256 `1a3f40c37a2a1e10daf2a5b87c3339d304f4c9f746124f6bb1446a2aa721565f`.

| Decisione | Primo active ID / precedente candidato | Evidenza runtime ACCEPTED |
|---|---|---|
| DOCK w0/a1 | L471 / L470 | [L401](/home/lorenzo/MSSR_thesis/logs/teleop/runs/button-06-20261006-111811.DCLcsu/isaac.log:401) |
| ALIGN w1/a0 | L472 / L471 | [L404](/home/lorenzo/MSSR_thesis/logs/teleop/runs/button-06-20261006-111811.DCLcsu/isaac.log:404) |
| DOCK w1/a0 | L979 / L978 | [L437](/home/lorenzo/MSSR_thesis/logs/teleop/runs/button-06-20261006-111811.DCLcsu/isaac.log:437) |
| DOCK w1/a1 | L980 / L979 | [L438](/home/lorenzo/MSSR_thesis/logs/teleop/runs/button-06-20261006-111811.DCLcsu/isaac.log:438) |
| ALIGN w2/a0 | L993 / L992 | [L443](/home/lorenzo/MSSR_thesis/logs/teleop/runs/button-06-20261006-111811.DCLcsu/isaac.log:443) |
| ALIGN w3/a0 | L1491 / L1490 | [L471](/home/lorenzo/MSSR_thesis/logs/teleop/runs/button-06-20261006-111811.DCLcsu/isaac.log:471) |
| ALIGN w4/a0 | L1854 / L1853 | [L494](/home/lorenzo/MSSR_thesis/logs/teleop/runs/button-06-20261006-111811.DCLcsu/isaac.log:494) |
| DOCK w4/a0 | L2154 / L2153 | [L512](/home/lorenzo/MSSR_thesis/logs/teleop/runs/button-06-20261006-111811.DCLcsu/isaac.log:512) |

## Impatto operativo

Nessun episodio viene promosso a complete MM8 demonstration. La supervisione strutturale direttamente valida può essere conservata in una futura vista derivata; le 22 label spostate non vengono interpretate come nuovi dispatch sul loro stato tardivo. Il dataset esistente resta immutabile e il filtro non viene applicato. I vecchi 119521 timestep sono una previsione precedente non validata da questo audit; nessun nuovo conteggio globale viene congelato qui.

Come deciso dall’utente, pianificare successivamente 2–3 registrazioni assembly-only dopo il fix del recorder. Non si avviano nuove registrazioni o modifiche al recorder in questa fase. A1 è indipendente dal recupero/filtro: solo contratto stdlib, senza adapter concreto, dataset, trainer o runtime.

Verifica READ-ONLY: SHA-256 delle otto sorgenti e degli otto Canonical confrontati con l’audit precedente; 60 grafo/goal pre-dispatch diretti verificati; tutte le 112 acceptance trovate nei rispettivi log. Nessuna interpolazione, copia del post-stato o modifica alle dimostrazioni.
