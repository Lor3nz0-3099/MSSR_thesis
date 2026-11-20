# BC-v5 — GUI diagnostics sul VALIDATION set

## Provenienza e perimetro

- Data delle acquisizioni: **20 novembre 2026**.
- Archivio originale: `raw/good_datas_GUI.zip` — SHA-256 `2a8f0da056b5ae24852a1ecca111034dfc044a7375f6e920f2206a5d8badacb2`.
- Checkpoint: **BC-v5 last.pt, epoca 20**. SHA-256: `38b98fd70ca104e248bd507a59d19973fcf40c64c07f46ed2471f36b8d6d2205`.
- Policy: solo actor; decoder a **logit threshold 1.0**. Episodi S0 con moduli separati.
- Materiale analizzato: `run_config.json`, `initial_state.json`, `target_topology.json`, `events.jsonl`, `bc_v5_policy_logits.jsonl`, `summary.json` quando presente.
- Finalità: diagnostica online closed-loop; **non** misura quantitativa offline e **non** TEST.

## Risultati degli otto episodi

| Task | Seed ambiente | Spawn | DOCK | Topologia (facce incluse) | Tilt completi | Ultima evidenza (s sim.) | Stato runner |
|---|---:|---|---:|---|---:|---:|---|
| gap | 17001 | radiale | 7/7 | Sì | 4 | 71.50 | missing_summary |
| gap | 17001 | scattered (20261120) | 7/7 | Sì | 21 | 64.40 | missing_summary |
| stairs | 13001 | radiale | 7/7 | Sì | 29 | 81.83 | missing_summary |
| stairs | 13001 | scattered (20261120) | 7/7 | Sì | 47 | 83.20 | missing_summary |
| navigation | 5100 | radiale | 7/7 | No | 3 | 66.60 | missing_summary |
| navigation | 8000 | scattered (20261120) | 3/7 | No | 0 | 52.80 | missing_summary |
| manipulation | 21002 | radiale | 4/7 | No | 0 | 120.00 | completed |
| manipulation | 21005 | scattered (20261120) | 7/7 | Sì | 16 | 92.20 | error |

**Sintesi:** 8 prove; 6/8 con sette DOCK; **5/8** con topologia target compatibile per facce; **2** run con `summary.json`, **6** senza riepilogo finale.

## Interpretazione rigorosa

1. **RC-Car8 navigation/5100, radiale:** sette DOCK e sette VALID CONNECTION, **ma grafo fisico non isomorfo alla topologia target neppure ignorando le facce**. Non classificarlo come successo di assembly.
2. **MM8 manipulation/21005, scattered/20261120:** sette DOCK compatibili con grafo target (incluse facce), sedici `SET_TILT COMPLETE`, nessuna azione strutturale selezionata nelle ultime sonde. Il folding è stato osservato dall’operatore, ma il report non contiene uno snapshot finale dei giunti per la verifica numerica delle tolleranze.
3. **MM8 manipulation/21005:** `summary.json` riporta `status=error` a 92,4 s simulati a causa di un `FileNotFoundError` per `bc_v5_policy_logits.jsonl` durante una scrittura. **Errore strumentale distinto dall’esito fisico dei DOCK.** Non correggere retroattivamente il summary.
4. **MM8 manipulation/21002, radiale:** `status=completed` indica raggiungimento orizzonte **120 s**, non assembly completato: presenti solo quattro DOCK.
5. **Altri sei run:** `summary.json` assente; secondo il resoconto dell’operatore sono stati interrotti volontariamente. L’ultima sonda registrata non è un timestamp di termine certificato. Non normalizzare a 120 s e non dedurre un tasso di successo dal solo `status`.
6. **Stop per assenza di decisioni:** `structural_eligible=0` significa che nell’ultima sonda non esistono candidati strutturali eleggibili; **non prova** che i logit fossero sotto soglia. Dove candidati esistono e `selected_aligns=selected_docks=0`, il comportamento è compatibile con l’assenza di nuove selezioni.

## Classificazione consigliata

Questa campagna è **GUI diagnostics sul VALIDATION set**, una verifica qualitativa e con conteggi osservabili della policy frozen in simulazione online. Il subset analizzato include un seed `5100` del generatore Expert diagnostico, distinto dal curriculum C1 VALIDATION ordinario; pertanto non riportare tutte le prove come benchmark C1 omogeneo. L’analisi quantitativa offline e il TEST sono due sezioni separate.

## Regole di archiviazione

- Conservare l’archivio originale senza modificarne i log.
- Conservare `audit/episode_audit.csv` e `audit/manifest.json`, calcolati dai file originali.
- Non introdurre manualmente `summary.json` dove manca, né sostituire `status=error` con `completed`.
- La directory `artifacts/bc_v1/` è ignorata da Git nella configurazione remota corrente: la copia è locale ma **non versionata** senza `git add -f` esplicito.
- Prima del TEST: congelare checkpoint, F2 (solo TRAIN), soglia 1.0 scelta sul VALIDATION; utilizzare soltanto split e holdout TEST previsti da F1/G2, senza adattamenti né ricerca di iperparametri sul TEST.

## Gate prima del TEST offline

Verificato nel repository remoto di riferimento: `F1-v8` include **102 episodi TEST riservati** (429 TRAIN, 102 VALIDATION, 102 TEST), ma il G2 Stage11 attualmente usato dall'evaluator BC-v5 contiene **zero episodi TEST** e l'evaluator `scripts/bc_v5/evaluate_checkpoint.py` rifiuta espressamente dati TEST in quel G2. Serve dunque una nuova elaborazione TEST **separata**: stessi F1-v8, Canonical v8, G1/F2 TRAIN-fitted, H1/H2 epoca 20, decoder a soglia 1.0 congelato; nessuna rifinitura o selezione sul TEST. La verifica `audit/test_gate_readiness.py` legge esclusivamente identità e metadati, non apre gli esempi TEST.
