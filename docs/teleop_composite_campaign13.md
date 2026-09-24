# Campagna teleoperata T01–T13

Configurazione: `mssr_ws/src/mssr_expert/config/smores_teleop_composite_campaign13.json`.
Profilo: `teleop_connected_v1`. Gli stage C01–C10 mantengono la geometria precedente.

Questi percorsi sono generati e controllati geometricamente; non sono ancora
certificati da esecuzioni fisiche complete in Isaac. La validazione originaria
dei seed riguarda gli ostacoli sorgenti, non automaticamente ogni nuova composizione.

## Geometria e spazio per Snake8

- Nessuna rampa: ogni supporto è orizzontale; le sole variazioni di quota sono i gradini.
- Lunghezza di progetto di Snake8: **0,80 m**.
- Dopo un gap: sponda piana lunga **almeno 1,20 m** e larga 1,20 m, misurata lungo
  l'attraversamento. Questa lunghezza non include il raccordo successivo.
- Dopo le scale: pianerottolo lungo almeno **1,20 m** (attualmente 1,32 m dai seed).
- Pedane di manovra: **2,40 × 2,40 m**. Raccordi: **1,40 m** di larghezza.
- Il raccordo arriva alla pedana e il nuovo ostacolo parte dal suo bordo: girare
  sulla pedana, dopo aver fatto salire tutto Snake sulla sponda di arrivo.
- Alzate, pedate, numero di gradini, larghezza del gap e geometria del pulsante
  provengono dal seed. Le piattaforme di appoggio possono essere più grandi.
- Il livello raggiunto dopo una scala è mantenuto nei tratti successivi.

Il controllo usa intersezioni di poligoni per rilevare gap coperti e sovrapposizioni
fra tratti non adiacenti. Verifica le dimensioni dei gradini e delle sponde, e
campiona l'appoggio lungo ostacoli, raccordi e arrivo ogni 5 cm su una fascia
di 60 cm. Non sostituisce una verifica dinamica di sterzata, aderenza, collisione
con i coni, pressione del pulsante o ingombro durante la riconfigurazione.

## Stage

Le direzioni sono assolute nel mondo, rispetto a +X. Le curve RC modificano la
direzione di uscita; le pedane raccordano tale direzione con l'ostacolo seguente.

| ID | Sequenza (seed) | Orientamenti degli ingressi |
|---|---|---|
| T01 | scale 3102 → RC 5101 → pulsante 6103 | 0°, 0°, −90° |
| T02 | gap 4106 → RC 6053 → pulsante 6107 | 0°, 0°, −90° |
| T03 | RC 5102 → gap 4104 → RC 6001 | 0°, 0°, 0° |
| T04 | scale 4689 → RC 5106 → gap 4103 | 0°, 0°, +90° |
| T05 | RC 6000 → RC 6056 → pulsante 6251 | 0°, 0°, +90° |
| T06 | gap 4102 → RC 6004 → pulsante 8568 | 0°, 0°, +90° |
| T07 | scale 3104 → RC 5103 → pulsante 8936 | 0°, 0°, −90° |
| T08 | RC 5104 → gap 4105 → RC 5105 | 0°, 0°, 0° |
| T09 | gap 4107 → scale 3101 → pulsante 9974 | 0°, +90°, +90° |
| T10 | RC 6017 → scale 3105 → RC 5100 | 0°, −90°, −90° |
| T11 | gap 4106 → scale 3102 → gap 4104 | 0°, +90°, +90° |
| T12 | RC 5102 → RC 6053 → RC 6000 | 0°, 0°, −90° |
| T13 | gap 4103 → RC 6001 → pulsante 6101 | 0°, 0°, +90° |

Copertura: 13 seed RC e 7 pulsanti distinti, tutti nel catalogo validato.
Le curve RC vanno controllate tramite `rc_car_planar_obstacle_layout`: il vecchio
campo `route_kind` non descrive necessariamente la centerline fisica attuale.

## Generare viste e missioni

Dalla radice del repository:

```bash
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
MPLCONFIGDIR=/tmp/mssr-teleop-mpl \
python3 scripts/smores_ep/preview_teleop_composite_campaign.py
```

`PYTHONNOUSERSITE=1` evita il conflitto locale fra NumPy 2 installato nell'utente
e Matplotlib compilato per NumPy 1 presente nel sistema.

Output: `logs/teleop/course_previews/t01-t13-v1/`:

- `overview.png` e `overview.pdf`: viste dall'alto di tutta la campagna;
- `teleop-tXX.png`: dettaglio di ogni stage;
- `teleop-tXX.mission.json`: missione da passare a Isaac;
- `teleop-tXX.course.json`: geometria mondiale effettiva;
- `geometry_audit.json`: esito, ambito dei controlli e hash dei percorsi.

Se cambi configurazione o generatore, rigenera le missioni e conserva il precedente
manifest insieme alle registrazioni già acquisite. Il grafo registrato include
ID del percorso, versione, seed, trasformazioni locali e `geometry_sha256`.
L'hash identifica l'osservazione geometrica statica prima di aggiungere il campo
hash stesso; non è un hash degli stati dinamici del pulsante.

## Usare il launcher fisico con DualSense

Il vecchio comando `--obstacle-course` seleziona `manual_obstacle_course()`:
contiene una rampa di circa 15° e soli 40 cm tra gap e scala. In quella modalità
`--gap-seed` e `--stair-seed` non modificano la geometria del percorso manuale.

Nel launcher con bridge, nodo morphology e DualSense, sostituire **i tre argomenti**:

```bash
--obstacle-course \
--gap-seed 4106 \
--stair-seed 3000 \
```

con, per esempio, T11:

```bash
--composite-mission "$PWD/logs/teleop/course_previews/t01-t13-v1/teleop-t11.mission.json" \
--composite-seed-catalog "$PWD/mssr_ws/src/mssr_expert/config/smores_composite_seed_catalog.json" \
```

Gli altri canali, il bridge, il nodo morphology e il launcher DualSense restano
quelli della sessione fisica. Questa sostituzione carica soltanto il mondo:
non avvia un esecutore autonomo dei task. Non eseguire contemporaneamente un
secondo launcher del simulatore sullo stesso runtime.

**Macro orientate:** `snake_gap` e `snake_stairs` usano ora il riferimento dello
stage, compresi +Y e −Y. Il nucleo dei gait +X resta invariato: si trasformano
solo le posizioni per la pianificazione e l'asse dei traguardi dell'esecutore.
I grafi registrati rimangono nel mondo. Avvicinare e allineare manualmente il
serpente prima di premere X (gap) o SQUARE (scale). La selezione automatica
richiede tutti i moduli nella corsia di approccio (±0,35 m lateralmente e da
2 m prima a 0,20 m dopo il primo bordo); i controlli del gait verificano poi
l'allineamento e la geometria. Se la selezione è ambigua la macro viene rifiutata.
Lo stage selezionato e il suo riferimento restano fissati durante la macro.
I test verificano l'equivalenza dei comandi ruotati e il feedback su ±Y;
la riuscita fisica in Isaac su questi orientamenti deve ancora essere provata.

Per controllare il piano semantico senza avviare Isaac:

```bash
python3 scripts/smores_ep/run_composite_course.py \
  --campaign mssr_ws/src/mssr_expert/config/smores_teleop_composite_campaign13.json \
  --episode teleop-t11 --plan-only
```

L'esecuzione automatica `--execute` è rifiutata per questo profilo. Il runner può
aprire il mondo con `--preview-only`, ma non avvia il controller DualSense.

## Registrazione

Annotare l'ID Txx e mantenere la registrazione attiva durante le riconfigurazioni.
Conservare gli esiti distinti: missione completa, prefisso valido, interruzione.
Non attribuire successo completo a un episodio terminato prima del goal.
Le prime esecuzioni sono anche la verifica fisica delle nuove composizioni;
un percorso geometricamente valido non è ancora una dimostrazione riuscita.
