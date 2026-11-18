# BC-v5 v8 — Valutazione quantitativa del checkpoint migliore

Questo codice è parte della **repository MSSR_thesis**, nella cartella eseguibile `scripts/bc_v5/`.
Non modifica `best.pt`, normalizzatori, Canonical v8, G1, G2 o i dati originali.
La prima versione ammette **solo VALIDATION**: il TEST resta sigillato.

## Uso

Dalla root `~/MSSR_thesis`:

```bash
bash scripts/bc_v5/run_validation.sh
```

Il launcher esegue quattro unit test, verifica identità e integrità e valuta il
checkpoint `best.pt` dell'esperimento `bc_v5_v8_primitive_first_20261117T143857Z_118314`.
Monitor: `~/bc_v5_best_validation_eval.log`.

In alternativa:

```bash
BC_PY="$HOME/.local/share/mssr-bc-v1/venv/bin/python"
PYTHONPATH="$PWD" "$BC_PY" -B scripts/bc_v5/evaluate_checkpoint.py --check
PYTHONPATH="$PWD" "$BC_PY" -B scripts/bc_v5/evaluate_checkpoint.py --validation
```

Per una verifica limitata e NON conclusiva, utilizzare `--validation --max-batches 2`.
In quel caso il report porta esplicitamente lo stato `PARTIAL`.

## Output

La directory (nuova, mai sovrascritta) è creata in:

`artifacts/bc_v1/evaluations/bc_v5_v8_best_epoch12_validation_<timestamp>/`

File:

- `metrics.json`: metriche globali e per primitiva/task/morfologia;
- `primitive_metrics.csv`: P, R, F1, TP, FP, FN, TN, supporto positivo;
- `task_metrics.csv`, `morphology_metrics.csv`: metriche aggregate per contesto;
- `regression_metrics.csv`: MAE/RMSE per primitiva, parametro, componente, unità;
- `identity.json`: fingerprint, SHA-256 checkpoint, soglia, split e protocollo.

**Exact-set**: un'osservazione è corretta solo se *tutte* le candidate con
supervisione valida coincidono con le etichette. Le candidate mascherate non
contano come errori. I 2 grafi VAL diventati senza positive in Stage11 rimangono
esclusi, esattamente come nel trainer: 23.749 campioni valutati.

**Regressione**: `oracle_positive_*` calcola MAE/RMSE sui target numerici
positivi supervisionati, anche se la testa di presenza non li seleziona.
`selected_true_positive_*` usa esclusivamente positive correttamente selezionate.
Per entrambe si distinguono `normalized` e `physical`. Le unità fisiche originali
provengono dalle binding F2, con verifica dell'inversione rispetto ai target expert
fisici conservati nel G2. La metrica `nrmse_range` è RMSE diviso il range dei
target osservati, `null` se il range è zero.

La VALIDATION non è ricampionata e nessuna statistica F2 viene ricalcolata.
Lo script ricalcola la VALIDATION loss con la stessa loss del trainer e la
confronta con quella storicizzata nel checkpoint. Un mismatch interrompe la
pubblicazione del report.

**Nota:** i raggruppamenti per morfologia/task derivano dagli audit associati
alle singole osservazioni; l'assenza di un valore viene indicata come `unknown`.
L'evaluator non ricava la riuscita fisica di una missione da queste metriche:
serve una verifica separata in Isaac Sim.

## TEST

Non usare il G2 corrente come se contenesse il TEST. Per il TEST occorre una
pipeline di caricamento/trasformazione dedicata, con F2 identica al TRAIN,
modello e soglia congelati, ed esplicita autorizzazione all'apertura dell'holdout.
Questo script non supporta `--test` proprio per prevenire errori metodologici.
