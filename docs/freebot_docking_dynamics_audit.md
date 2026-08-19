# Audit tecnico della dinamica FreeBOT

Data dell'audit: 17 luglio 2026.

Ambito: pacchetto `scripts/freebot_docking`, asset
`assets/freebot/usd_physics/freebot_cad_full_nearer_wheels_rigid.usd` e
confronto con `scripts/isaac_freebot/run_freebot_two_module_climb.py`.
Nel repository non esiste un file letteralmente chiamato
`run_freebot_two_module_climb(1).py`; il confronto usa la sola copia presente,
`run_freebot_two_module_climb.py`. Le opzioni fenomenologiche disabilitate di
default sono indicate come tali.

> **Stato post-audit (17 luglio 2026).** Le conclusioni sotto descrivono la
> baseline ispezionata. Sono state ora applicate correzioni mirate nel pacchetto
> `freebot_docking`: ruote a clearance analitica zero, caster a 2 mm con contact
> offset inferiore al gap, joint anchor aggiornati insieme ai corpi, COM/inerzie
> finite, precarico interno diretto lungo la normale radiale, nessuna
> extrapolazione angolare oltre 90°, rimozione della tolleranza angolare mista,
> attrito shell-shell PhysX senza patch custom ritardata, comando motore senza
> doppia rampa e gap iniziale predefinito di 20 mm. Dopo il primo run dinamico
> sono stati provati prima un drive di posizione sulle ruote ferme e poi un
> modello di attrito/back-drive del motoriduttore. Entrambi sono stati rimossi
> integralmente dopo i test dinamici: il controllo ruote è tornato al solo
> velocity drive con inviluppo coppia-velocità della baseline corretta. Anche
> il successivo confronto che applicava la forza esterna direttamente alla shell
> è stato annullato perché non modificava il comportamento osservato. La forza
> esterna torna quindi ad agire sul carrier nel punto della faccia magnetica e
> raggiunge la shell attiva soltanto tramite i contatti interni; la reazione
> opposta resta sulla passiva. Il precarico interno da 9,5 N continua ad agire
> fra carrier e shell attiva. La suite pura aggiornata passa con `134 passed`;
> il nuovo run dinamico va eseguito su una sessione con driver NVIDIA disponibile.

> **Aggiornamento fisico (18 luglio 2026).** La baseline resta priva di latch,
> hold del carrier e attrito shell-shell custom. La risultante esterna Fig. 4--5
> parte ora dalla faccia mobile del magnete e la sua semiretta individua la patch
> di applicazione sulla shell passiva: forza e reazione sono collineari e quindi
> conservano forza e momento globali senza coppie aggiunte. Il controllo automatico
> del piano di salita è disabilitato per default. A comando ruote nullo rimane una
> capacità frenante attiva esplicita di 0,12 N m per ruota, parametrica e limitata
> sotto la coppia di stallo. Il precarico interno resta costante a 9,5 N. Suite
> pura aggiornata: `137 passed`.

> **Aggiornamento pneumatici (18 luglio 2026).** I supporti ruota restano
> rigidamente collegati al carrier. Il raggio CAD caricato resta 16,000287 mm,
> mentre il proxy collisionale rappresenta il pneumatico scarico con 0,4 mm di
> inviluppo radiale aggiuntivo. Il materiale ruota usa il contatto cedevole
> implicito e force-based di PhysX (`k=12000 N/m`, `c=15 N s/m`): l'overlap del
> proxy è quindi deformazione elastica del battistrada, non compenetrazione di
> due corpi rigidi. Le caster conservano 2 mm nella posa nominale ma possono
> diventare appoggi reali quando il carrier si inclina. Il log separa ora forza
> normale sinistra/destra, gap del proxy e compressione di ciascun pneumatico.
> Il precarico interno resta 9,5 N. Suite pura aggiornata: `141 passed`.

> **Taratura appoggi (18 luglio 2026).** Dopo la prova dinamica, l'inviluppo
> scarico del pneumatico passa a 0,9 mm oltre il raggio CAD caricato; il contatto
> viene ammorbidito e smorzato a `k=8000 N/m`, `c=40 N s/m`. Le due caster sono
> ora allineate a clearance nominale zero per provare un appoggio interno a
> quattro punti. Restano sfere passive rigide a basso attrito: nessun vincolo le
> incolla alla shell. Il precarico interno resta costante a 9,5 N. Suite pura:
> `142 passed`.

> **Ablazione del peso (18 luglio 2026).** La CLI espone ora `--mass-scale`
> esclusivamente come prova diagnostica. Il fattore scala coerentemente masse
> e inerzie di shell, carrier, ruote e caster, senza modificare geometria,
> gravità, forze magnetiche o coppia motore. Il default fisico resta `1.0`; la
> prova proposta usa `0.25` (76,975 g, peso circa 0,755 N). Suite pura:
> `146 passed`.

## Metodo e grado delle evidenze

- **D — dimostrato staticamente**: segue direttamente dal codice o dagli
  attributi/vertici USD.
- **C — calcolato**: conseguenza numerica riproducibile di dati D.
- **R — da misurare a runtime**: dipende dal solver PhysX (contatti, impulsi,
  inerzie auto-calcolate o stato dopo il transitorio).

I vertici delle due mesh di collisione della shell sono stati trasformati nel
frame world con USD, includendo istanze e trasformazioni ereditate. Non sono
state usate le sole AABB. Un tentativo di run Isaac breve è arrivato
all'inizializzazione ma non al primo step: l'ambiente non disponeva del driver
NVIDIA (`NVML_ERROR_DRIVER_NOT_LOADED`, nessun device Vulkan/CUDA). I test puri
Python danno `129 passed`; non sostituiscono le misure PhysX richieste.

## A. Sintesi esecutiva

1. **Critico — lo scenario nasce fuori dalla zona magnetica.** Il gap iniziale
   è 40 mm, mentre la curva realmente chiamata vale esattamente zero da 30 mm
   in poi. Il riferimento fenomenologico vale invece 0,1808 N a 40 mm e non ha
   cutoff finito.
2. **Critico — le ruote fisiche inizialmente non toccano la shell interna.** I
   proxy cilindrici runtime da raggio 16,000287 mm hanno clearance conservative
   di 2,985 e 2,743 mm. I caster sono invece progettati per **non toccare** la
   shell, con 2 mm di luce ciascuno. Tale luce risulta 1,924/1,999 mm usando il
   body origin previsto dal CAD, ma diventa 0,0217/3,182 mm rispetto al centro
   sferico fitted usato dalla simulazione: è un'incoerenza di datum, non un
   supporto voluto sui caster. Il riferimento usa inoltre mount ruota cedevoli
   fino a ±28°, assenti nell'asset fisico rigid.
3. **Critico — la forza esterna è applicata al carrier, non alla shell attiva.**
   Il trasferimento carrier→shell è lasciato esclusivamente ai contatti interni;
   con le ruote separate e i caster intenzionalmente distaccati, la forza sposta
   prima il carrier senza una catena di trasferimento garantita. Un contatto sul
   caster sarebbe accidentale e indicherebbe il datum errato. La versione
   fenomenologica bypassa questo tratto applicando la forza direttamente a
   `shell_body`.
4. **Critico — manca la coppia di allineamento radiale del carrier.** La forza
   interna da 9,5 N segue l'asse del magnete solidale al carrier; non segue la
   normale locale e non crea la molla angolare presente nel riferimento
   (`0,45 N m/rad`, limite `0,20 N m`). L'inclinazione può quindi scaricare le
   ruote e rendere errato l'angolo di Figura 5.
5. **Critico — esiste una finestra senza attrito shell-shell.** Con il contatto
   custom attivo, l'attrito PhysX delle due shell è zero. Il custom usa il
   carico normale letto dopo lo step precedente, inizializzato a zero: al primo
   frame di contatto entrambi i limiti tangenziali sono zero.
6. **Critico — il nuovo comando motore limita due volte l'avviamento.** A
   240 Hz la rampa porta il primo target a soli 3 deg/s e il modello imposta
   `maxForce = 0,00572 N m`, 210 volte sotto 1,2 N m del riferimento. Inoltre il
   limite diventa zero quando velocità reale e target coincidono, per cui una
   coppia resistente richiede un grande errore permanente di velocità.
7. **Importante — posa e Figura 5 chiudono una seconda porta alla cattura.** La
   posa iniziale ha asse magnetico −Z e normale fra moduli +X, quindi θ=90°.
   Anche a contatto il modello darebbe solo circa 2,082 N
   (`A_parallel=2,08 N`, `A_perp=0,10 N`), non 22,6 N. Con μ=1,1 servono almeno
   2,746 N di normale per sostenere il peso di 3,0205 N, anche assumendo
   trasferimento perfetto.
8. **Risolto — segno della componente trasversa di Figura 5.** Il pacchetto
   usa ora `- A_perp t`, coerentemente con le frecce della Figura 5 e con
   `run_freebot_emergent_docking.py`. Nella posa iniziale la componente punta
   verso la linea di contatto e non più verso il basso. Un test di regressione
   verifica anche che un magnete sollevato generi momento positivo di salita.
9. **Importante — il datum di contatto PhysX non è ancora verificato.** Le mesh
   confermano i raggi 63,3472/61,3472 mm, ma un'implementazione precedente
   documenta circa 2,5 mm fra contatto del collider e gap sferico. Il pacchetto
   corrente non sottrae tale offset dalla Figura 4. Se il dato runtime è ancora
   valido, al contatto usa 10,97 N invece di 22,6 N.
10. **Importante — COM e inerzie non sono determinati nel modello sorgente.**
    Il runtime sovrascrive le masse ma non COM né tensori d'inerzia. Nel USD
    `centerOfMass=(-inf,-inf,-inf)` e inerzia diagonale zero; `internal_link` è
    visual-only e senza collider. I valori effettivi devono essere letti da
    PhysX e non sono riproducibili dal solo file.

La perdita principale è quindi doppia:

```text
motore -> ruota --X--> shell attiva
                         ^
magnete esterno -> carrier --contatti ruota incerti--> shell attiva
```

La prima X impedisce di portare il magnete verso il punto di docking; la
seconda impedisce alla forza magnetica esterna di diventare normale
shell-shell. Il riferimento funzionante chiude entrambe le lacune con ruote SDF
CAD sovradimensionate/già a contatto, forza esterna direttamente sulla shell e,
quando abilitati, allineamento, latch e adesione tangenziale fenomenologica.

## B. Problemi critici

### B1. Forza esterna nulla nella posa iniziale

- **File/funzione/riga:**
  `physics/external_magnet.py`, `TabulatedAlignedForceCurve.attraction_force_n`,
  righe 58–72; `freebot_figure4_force_curve`, righe 76–91;
  `cli.py`, righe 42 e 80–86.
- **Comportamento attuale:** `np.interp(..., right=0.0)` restituisce zero oltre
  l'ultimo campione. Lo scenario costruisce esattamente 40 mm di gap.
- **Comportamento atteso:** la cattura iniziale deve essere ottenuta dal moto
  reale delle ruote fino a meno di 30 mm; non esiste alcuna attrazione che possa
  supplire a quel moto.
- **Evidenza numerica:**

  | gap | Figura 4 attiva | legge fenomenologica `22.6/(1+g/0.01)^3` |
  |---:|---:|---:|
  | 0 mm | 22,600 N | 22,600 N |
  | 10 mm | 1,350 N | 2,825 N |
  | 30 mm | 0 N | 0,3531 N |
  | 40 mm | **0 N** | **0,1808 N** |

  Il default reale del fenomenologico parte a circa 350,5 mm e conserva ancora
  0,000482 N; il confronto a 40 mm sopra isola la sola legge.
- **Test che lo dimostra:** loggare `gap`, `distance_scale` e `Fext` nei primi
  dieci step; devono essere rispettivamente circa 40 mm, 0 e 0.
- **Classificazione:** critico.

### B2. Catena motore→ruota→shell interrotta dalle clearance

- **File/funzione/riga:** `config/geometry.py`, righe 56–64;
  `isaac/stage_builder.py`, righe 212–250 e 253–283;
  `isaac/module_handles.py`, righe 243–332.
- **Comportamento attuale:** gli SDF delle gomme CAD vengono rimossi e
  sostituiti da cilindri esatti di raggio 16,000287 mm; i caster diventano sfere
  da 4,650 mm. Le posizioni dei link non vengono avvicinate alla superficie.
- **Comportamento atteso:** entrambe le ruote devono ricevere una normale
  positiva dalla shell, altrimenti la coppia del joint produce solo spin libero.
- **Evidenza numerica, posa USD iniziale:**

  | elemento | centro world [mm] | clearance analitica dalla superficie interna |
  |---|---|---:|
  | ruota sinistra | (23,042; 95,224; 32,054) | **+2,985 mm** |
  | ruota destra | (23,043; 25,819; 31,846) | **+2,743 mm** |
  | caster 1 | (−31,633; 60,363; 54,020) | **+0,0217 mm** |
  | caster 2 | (77,764; 60,363; 54,020) | **+3,182 mm** |

  Valore positivo significa separazione e la colonna usa il centro geometrico
  fitted della mesh shell. Il progetto dei caster richiede invece 2 mm di luce:
  usando il body origin previsto dal CAD si ottengono 1,924 e 1,999 mm, entro
  circa 0,08 mm dal nominale. La discordanza nasce perché il centro fitted della
  mesh shell è traslato rispetto al body origin di
  `(1,554; 0,877; 4,668) mm`. Quindi i valori 0,0217/3,182 mm non dimostrano un
  appoggio voluto: dimostrano che collider shell e carrier non stanno usando lo
  stesso datum. I vecchi SDF visuali misurano circa
  45,25 e 43,75 mm di diametro esterno: il riferimento non sostituisce questi
  collider e beneficia quindi di una geometria molto più grande del diametro
  nominale 32 mm. Il suo asset `compliant` aggiunge per ciascuna ruota un mount
  con rotX/rotZ ±28°, `k=0,10 N m/rad`, `c=0,035 N m/(rad/s)` e limite
  0,40 N m; può quindi adattare ulteriormente la gomma alla superficie. L'asset
  fisico `rigid` collega invece direttamente ruota e carrier.
- **Test che lo dimostra:** Test A e B; registrare per ogni contatto normale,
  velocità tangenziale e slip. Nominalmente ci si attende `N_caster1≈N_caster2≈0`;
  qualsiasi impulso persistente su un caster è una regressione geometrica.
- **Classificazione:** critico.

### B3. La forza magnetica esterna non arriva direttamente alla shell attiva

- **File/funzione/riga:** `physics/external_magnet.py`, righe 436–468;
  `isaac/simulator.py`, righe 305–328.
- **Comportamento attuale:** la forza è trasformata in wrench sul
  `active.internal_body` al centro della faccia magnetica. La reazione va alla
  shell passiva. Non esiste una wrench esterna sulla shell attiva.
- **Comportamento atteso:** il carrier deve spingere la shell attiva tramite le
  normali interne, generando una normale shell-shell. Questa è la fisica
  richiesta, ma oggi non è garantita dai contatti.
- **Evidenza numerica:** prima del contatto interno lungo +X, la shell attiva
  riceve 0 N della forza esterna. Le ruote hanno 2,74–2,99 mm di clearance; i
  caster devono conservarne 2 mm e non partecipare al trasferimento. La forza
  può quindi traslare/inclinare il carrier prima di produrre `N_shell-shell`.
  Il parametro decisivo è
  `eta_N=N_shell-shell/Fmag,n`, non ancora loggato.
- **Confronto:** il fenomenologico applica `raw_external_force` direttamente a
  `shell_body` (`run_freebot_two_module_climb.py`, righe 1194–1205), cioè pone
  implicitamente `eta_N≈1` e bypassa il carrier.
- **Test che lo dimostra:** Test C contro D, con la stessa posa e stessa forza;
  misurare `eta_N` e ritardo alla comparsa della normale.
- **Classificazione:** critico.

### B4. Precarico assiale senza allineamento radiale

- **File/funzione/riga:** `scenarios/two_module_docking.py`,
  `compute_internal_magnetic_preload_interaction`, righe 221–263;
  `physics/geometry.py`, righe 257–328.
- **Comportamento attuale:** il codice calcola punto sulla shell, normale e
  `alignment_cosine`, ma usa soltanto `face_center`; applica
  `F=9.5*magnet_axis`. La normale calcolata non orienta la forza e non genera
  una coppia di allineamento separata.
- **Comportamento atteso:** il valore 9,5 N resta invariato, ma il carrier deve
  avere una dinamica che mantenga faccia e raggio coerenti senza scaricare le
  ruote.
- **Evidenza numerica iniziale:** rispetto al vero centro della shell, l'asse
  −Z è disallineato di circa 1,57° alla faccia; il precarico contiene circa
  9,496 N radiali e **0,260 N tangenziali**. Il suo momento sul centro della
  shell è circa **0,01460 N m**; usando l'origine del carrier come stima del COM,
  il momento sul carrier è solo **0,00153 N m**. Questi momenti dipendono dal
  braccio, non sono una molla restaurativa.
- **Confronto:** `active_internal_alignment_torque()` usa
  `k=0.45 N m/rad`, `c=0.06 N m/(rad/s)`, limite 0,20 N m. Produce 0,0785 N m a
  10° e 0,157 N m a 20°, con reazione opposta sulla shell.
- **Test che lo dimostra:** Test B, loggando errore asse-raggio, carichi delle
  quattro appoggiature e velocità angolare relativa scomposta.
- **Classificazione:** critico.

### B5. Finestra di attrito nullo shell-shell

- **File/funzione/riga:** `isaac/simulator.py`, righe 87–102, 223–225,
  346–377; `scenarios/two_module_docking.py`, righe 82–198.
- **Comportamento attuale:** con `custom_shell_friction=True`, μ PhysX della
  shell è 0/0. Il custom si attiva solo se `gap<=2,5 mm` e `load>0`; `load` è la
  normale PhysX dello step precedente ed è inizializzato a zero.
- **Comportamento atteso:** al primo impatto il solver fisico dovrebbe offrire
  immediatamente attrito statico fino a μN.
- **Evidenza numerica:** primo frame di contatto:
  `resolved_shell_normal_n=0`, `static_limit=1.1*0=0`,
  `dynamic_limit=1.0*0=0`; dunque `Ft=0` anche se il contatto geometrico è
  attivo. A 240 Hz la finestra minima è un frame, 4,167 ms, e può allungarsi se
  la lettura contatto resta zero/ritardata.
- **Test che lo dimostra:** Test E con log per ogni frame di gap, N PhysX,
  regime custom e forze tangenziali dei due canali.
- **Classificazione:** critico.

### B6. Doppia limitazione del motore e velocità sotto carico

- **File/funzione/riga:** `control/wheel_drive.py`, righe 17–25, 185–233,
  257–292; `isaac/simulator.py`, righe 277–296.
- **Comportamento attuale:** il target cresce di 720 deg/s²; a 240 Hz sono
  3 deg/s per frame. `maxForce` è poi proporzionale a
  `abs(target-actual)/360`, anziché essere il solo envelope alla tensione
  richiesta.
- **Comportamento atteso:** il modello deve distinguere rampa del comando,
  tensione equivalente, velocità reale e coppia resistente, senza azzerare la
  capacità di carico quando la ruota raggiunge il target.
- **Evidenza numerica:** vedere la sezione D. Il primo limite è 0,00572 N m,
  contro 1,2 N m costante. Per sostenere 0,10 N m occorre un errore permanente
  di `0.10/0.6864655*360 = 52,44 deg/s`.
- **Test che lo dimostra:** Test A con target a gradino e a rampa; log ogni frame
  di target richiesto, target slewed, velocità, errore, maxForce e coppia joint.
- **Classificazione:** critico.

### B7. Asse di avanzamento non coincide con la tangente di salita

- **File/funzione/riga:** `control/wheel_drive.py`, righe 275–291;
  `isaac/module_handles.py`, righe 149–183; joint USD
  `/World/freebot/joints/{left,right}_wheel_joint`.
- **Comportamento attuale:** entrambi i joint hanno asse locale Y, quaternioni
  locali identità e target con lo stesso segno. Con posa modulo identità,
  l'asse world comune è +Y.
- **Evidenza numerica:** alla connessione iniziale fra moduli
  `n_shell=+X`, `t_up=+Z`; per target positivo,
  `t_drive=a_wheel×n_shell=Y×X=−Z`, errore **180°**. Usando invece le normali
  radiali delle ruote nella posa bassa, le direzioni sono circa
  `(-0.9984,0,+0.056)` e l'errore rispetto a +Z è 86,76°/86,79°: il carrier
  viene prima mosso lungo X, come previsto per raggiungere il fianco, ma il
  segno deve cambiare coerentemente quando si entra nel piano di salita.
- **Altri assi USD:** asse longitudinale carrier e linea caster 1→2 sono +X;
  asse magnetico iniziale −Z. L'asse di rotazione ideale per salire nel piano
  XZ è ±Y, non +X.
- **Test che lo dimostra:** Test F e impulso di un solo segno motore; misurare
  la velocità del carrier lungo `t_up` e le componenti di ω shell.
- **Classificazione:** critico per il segno del comando di salita; la geometria
  degli assi è dimostrata.

### B8. Datum analitico e datum PhysX potenzialmente separati

- **File/funzione/riga:** `config/geometry.py`, righe 10–16;
  `physics/geometry.py`, righe 70–109;
  `run_freebot_emergent_docking.py`, righe 207–212 e 1537–1539.
- **Comportamento attuale:** il pacchetto usa direttamente
  `dcentri-Ra-Rp`. I raggi e il centro coincidono con i vertici; non è però
  applicato l'offset di contatto da 2,5 mm misurato/documentato nello script
  precedente.
- **Evidenza numerica condizionale:** se PhysX risolve già il contatto a gap
  analitico +2,5 mm, la Figura 4 fornisce **10,97 N** invece di 22,6 N, −51,5%.
  A +1 mm fornisce 16,51 N, −27,0%. Questa conclusione sul valore di gap al
  contatto è R; i valori di forza sono C.
- **Test che lo dimostra:** avvicinamento quasi statico con motori spenti;
  registrare il primo frame con `N>0` e il corrispondente signed gap.
- **Classificazione:** importante, diventa critico se l'offset runtime è
  confermato.

### B9. Segno trasverso di Figura 5

- **File/funzione/riga:** `physics/external_magnet.py`, righe 409–430;
  confronto con `run_freebot_emergent_docking.py`, righe 1720–1728.
- **Comportamento corretto:** `t` è la proiezione dell'asse magnetico e la
  forza usa `-Aperp*t`, cioè il richiamo verso la linea di contatto indicato
  dalle frecce della Figura 5.
- **Evidenza numerica:** nella posa asse −Z/normale +X, `t=−Z`; il pacchetto
  aggiunge 0,10 N verso il basso a contatto. Il segno opposto aggiungerebbe
  0,10 N verso l'alto. A 20° il termine raggiunge 1,40 N e il segno diventa
  dinamicamente rilevante.
- **Test che lo dimostra:** il test di regressione a +20° verifica sia la
  direzione trasversa sia il segno positivo del momento attorno all'asse di
  salita.
- **Classificazione:** risolto il 18 luglio 2026 confrontando codice, frame
  della Figura 5 ed equilibrio del paper.

### B10. Inerzie e COM non chiusi

- **File/funzione/riga:** `isaac/stage_builder.py`, righe 155–179;
  `create_freebot_cad_full_v2.py`, righe 208–215 e 353–367.
- **Comportamento attuale:** sono scritte solo le masse. Il CAD sorgente ha COM
  sentinella `(-inf,-inf,-inf)`, inerzia diagonale `(0,0,0)` e assi principali
  nulli. Il carrier è visual-only, senza collider da cui ricavare un tensore.
- **Comportamento atteso:** COM e tensori devono corrispondere al prototipo e
  restare riproducibili, perché il meccanismo del paper dipende da `m*r*sinθ`.
- **Evidenza numerica:** la sola inerzia sferica ideale della shell fisica,
  calcolata con 90 g e i raggi reali, è circa `2,3333e-4 kg m²`; con i 60 g del
  riferimento è `1,5555e-4 kg m²`. L'inerzia effettiva PhysX e quella del
  carrier sono R.
- **Test che lo dimostra:** subito dopo `play()`, loggare `get_coms()` e
  `get_inertias()` di tutti i sei corpi e confrontare con un pendolo/torque test.
- **Classificazione:** importante.

## C. Confronto con la versione fenomenologica

| Voce | Implementazione fisica | `run_freebot_two_module_climb.py` | Effetto | Natura del termine nel riferimento |
|---|---|---|---|---|
| Raggio esterno | 63,3472 mm, fit vertici | 66,5 mm sul body origin | critico per datum/ground | ridotto |
| Raggio interno | 61,3472 mm | 60,5 mm | importante; cambia gap magnete | ridotto |
| Centro geometrico | body origin + (1,55433; 0,87740; 4,66804) mm | body origin della shell | critico | artificiale rispetto al CAD |
| Spessore | 2,0000 mm | 6,0000 mm impliciti | importante | ridotto |
| Massa shell | 0,090 kg | 0,060 kg USD | importante | fisico ma diverso |
| Carrier completo | 0,2179 kg | 0,300 kg | importante | fisico ma diverso |
| Massa modulo | 0,3079 kg | 0,360 kg | importante | fisico ma diverso |
| Inerzie/COM | auto/indeterminati; carrier senza collider | auto USD, anch'essi non espliciti | importante | incompleto in entrambi |
| Gap faccia magnete interno | 5,17 mm sul vero centro | circa 9,01 mm sul centro/raggio ridotti | importante | il riferimento ricrea il gap nominale con geometria errata |
| Forza interna | fissa 9,5 N lungo asse magnete | dipolare, max 16 N, con damping | importante | ridotto/fenomenologico |
| Allineamento radiale | assente | molla+coppia damping, max 0,20 N m | critico | fenomenologico che compensa dinamica mancante |
| Forza esterna | Figura 4 × Figura 5 | cubica `F0/(1+g/d0)^3` | critico in cattura | ridotto continuo |
| Cutoff esterno | 0 da 30 mm | nessun cutoff finito | critico | artificiale ma diagnostico |
| Forza a gap 40 mm | 0 N | 0,1808 N | critico | fenomenologico |
| Orientazione | riduce entrambe le componenti; 2,082 N a 90°/contatto | la normale non è scalata dall'allineamento | critico | ridotto |
| Punto/corpo forza esterna | faccia magnete su carrier | punto shell attiva su `shell_body` | critico | bypass artificiale del trasferimento |
| Reazione esterna | shell passiva, saltata se kinematic | nessuna reazione applicata alla passiva | secondario per fixture, importante per energia | artificiale |
| Attrito shell-shell | PhysX 0 quando custom; custom μs/μd 1,1/1,0 | PhysX passiva 2,0/1,6 | importante | fisico ma sovratarato |
| Adesione tangenziale | molla custom sulla shell, carico N ritardato | opzionale sul carrier, μ=1,4, k=1200, c=4 | critico se abilitata | fenomenologico |
| Precarico tangenziale | zero | inizializza la molla per compensare 0,300·g | critico contro la caduta iniziale | artificiale |
| Caster-shell | μ 0,03/0,02, proxy 4,65 mm, luce nominale 2 mm | collider SDF CAD | importante | nessun contatto previsto nel fisico; geometria diversa nel riferimento |
| Wheel-shell | μ 2,2/1,9, proxy r=16 mm separati | SDF CAD circa r=22 mm, già vincolanti | critico | il CAD errato compensa il contatto mancante |
| Mount ruota | rigido, joint wheel direttamente sul carrier | cedevole rotX/rotZ ±28°, k=0,10, c=0,035, max 0,40 N m | critico | compliance ridotta che mantiene il contatto |
| Damping wheel joint | 500 | 500 | irrilevante come differenza | ridotto |
| MaxForce | envelope 0…0,6865 N m dipendente dall'errore | 1,2 N m costante | critico | il riferimento è sovra-attuato |
| No-load speed | 360 deg/s hard clamp | nessun clamp; scala 900 | importante | fisico solo nel nuovo |
| Slew | 720 deg/s² = 3 deg/s/frame | assente | importante | controllo ridotto |
| Scala lineare | 720 deg/s per m/s | 900 deg/s per m/s | importante: −20% sotto clamp | controllo |
| Latch | assente | engage: gap≤6 mm, patch≤30 mm, cos≥0,65; release gap>12 mm/patch>45 mm/cos<0,30 | importante | logico/fenomenologico |
| Torsione dock | assente | opzionale, 0,35 N m/rad, max 0,12 N m | importante se abilitata | fenomenologico |
| Condizione rilascio fisica | coda Figura 4 e perdita contatto | latch con isteresi; la normale resta attiva | secondario | logico |

### Che cosa rende funzionante il riferimento

- **Fisico:** joint revolute, coppie azione/reazione interne, contatti PhysX,
  gravità, attrito Coulomb, movimento del carrier e conseguente spostamento del
  COM.
- **Ridotto:** forza magnetica cubica, magnete dipolare interno, shell sferica
  analitica, friction coefficients elevati.
- **Fenomenologico:** coppia di allineamento radiale, adesione tangenziale con
  stato stick/slip, latch e torsione opzionale.
- **Artificiale:** applicazione della forza esterna direttamente alla shell,
  precarico della molla tangenziale che annulla subito la gravità, ruote CAD
  sovradimensionate/compliant e maxForce 1,2 N m indipendente dalla velocità.

L'adesione opzionale introduce esattamente:

1. molla tangenziale `k=1200 N/m`;
2. damping `4 N s/m`;
3. inizializzazione con una deflessione che bilancia `0,300g=2,943 N`;
4. limite `min(1,4*Fn,22,6 N)`;
5. return mapping in slip;
6. reset su perdita latch/forza/target.

Solo stick/slip, damping e limite Coulomb dovrebbero emergere dal contatto
fisico. Il precarico gravitazionale istantaneo e il latch sono aiuti numerici,
non effetti da copiare nel modello fisico.

## D. Cause delle ruote lente

### D1. Parametri e comando frame per frame

Con `dt=1/240=0,0041667 s`:

```text
delta_target_max = 720 deg/s² * dt = 3 deg/s per frame
tau_stall = 7 kgf cm = 7 * 0.0980665 = 0.6864655 N m
tau_limit = tau_stall * min(1, |target-actual|/360)
```

Per un comando lineare unitario, il target richiesto `720 deg/s` viene clampato
a 360. La sequenza slewed è 3, 6, 9, …, 360 deg/s e richiede 0,5 s. Per un
Twist 0,1 m/s il nuovo target è 72 deg/s, il riferimento è 90 deg/s e non ha
rampa.

| stato | target/actual usati dal codice | `maxForce` nuovo | riferimento |
|---|---:|---:|---:|
| primo frame | 3 / 0 deg/s | **0,0057205 N m** | **1,2 N m** |
| target 72, ruota ferma | 72 / 0 | 0,137293 N m | 1,2 N m |
| target 72, actual 3 | 72 / 3 | 0,131572 N m | 1,2 N m |
| target=actual=72 | 72 / 72 | **0 N m** | fino a 1,2 N m se c'è errore infinitesimo |
| target 360, actual 180 | 360 / 180 | 0,343233 N m | 1,2 N m |
| target 360, actual 350 | 360 / 350 | 0,0190685 N m | 1,2 N m |
| target 360, actual 359 | 360 / 359 | 0,00190685 N m | 1,2 N m |

La legge fisica a piena tensione, considerata soltanto come funzione della
velocità reale, darebbe:

| velocità | coppia piena tensione |
|---:|---:|
| 0 | 0,68647 N m |
| 3 deg/s | 0,68074 N m |
| 72 deg/s | 0,54917 N m |
| 180 deg/s | 0,34323 N m |
| 350 deg/s | 0,01907 N m |
| 359 deg/s | 0,001907 N m |

Il primo frame non riceve 0,6807 N m perché la rampa viene interpretata anche
come tensione di appena 3/360. L'ordine di grandezza perso rispetto al
riferimento è 210× al primo frame; a target 72 e ruota ferma è 8,74×. Il valore
di stallo nuovo non è di per sé troppo piccolo: è la combinazione rampa +
envelope sull'errore + contatto resistente a mantenere bassa la velocità.

### D2. Damping, stiffness e saturazione

- stiffness = 0 in entrambi;
- damping richiesto = 500 in entrambi;
- la drive USD è di tipo `force` e il target è in deg/s;
- il damping genera la richiesta di coppia, ma `maxForce` la tronca al valore
  sopra;
- il sorgente USD aveva damping 1000 e maxForce 5, entrambi sovrascritti dal
  runtime;
- i joint wheel hanno damping rigid-body 0,02; `internal_link` ha damping
  lineare 0,08 e angolare 0,35;
- il runtime riduce a 4 le velocity iterations di tutti i body, rispetto a 12
  per carrier/ruote nel USD.

La coppia resistente reale, accelerazione e saturazione frame-per-frame sono R.
Il log esistente mostra target, actual e maxForce, ma non la coppia joint né
`tau*omega`; Test A deve aggiungere queste due letture diagnostiche senza
cambiare la legge.

## E. Cause del mancato docking

| Categoria | Diagnosi | Stato |
|---|---|---|
| forza assente | a 40 mm Figura 4 = 0 N | D/C, critico |
| forza insufficiente | a θ=90° e contatto solo 2,082 N; μN < peso | C, importante/critico |
| geometria errata | raggi/centro nuovi corretti; proxy ruote non chiudono la geometria di contatto; quota ground lascia 3,153 mm di caduta iniziale | D/C, critico |
| trasferimento normale assente | forza sul carrier; nessuna forza sulla shell finché i contatti interni non reagiscono | D; efficienza R |
| attrito assente | un frame minimo con PhysX shell μ=0 e custom limit=0 | D/C, critico |
| allineamento errato | carrier senza molla radiale; θ iniziale 90°; segno A_perp incoerente | D/C, critico |

Il raggio `0,0665 m` non è il raggio del collider sferico. Nel pacchetto fisico
è solo la quota Z iniziale. Con centro a 66,5 mm e raggio collider 63,3472 mm,
la shell parte 3,1528 mm sopra il piano. Il fenomenologico tratta invece 66,5
mm come raggio e il body origin come centro, anticipando analiticamente il
contatto di `2*(66,5-63,3472)=6,3056 mm` rispetto alle superfici sferiche vere.

## F. Cause del mancato crawling

| Categoria | Diagnosi | Evidenza |
|---|---|---|
| mancato spostamento carrier | ruote senza normale e coppia iniziale ridotta; i caster non devono portare carico | D/C; moto effettivo R |
| mancata variazione COM | COM/inertia non definiti e carrier non portato sul fianco | D; traiettoria COM R |
| coppia motrice insufficiente | primo maxForce 0,00572 N m; sotto carico serve errore permanente | C |
| asse di rotazione errato | asse longitudinale +X; salita richiede rotazione shell circa ±Y; comando positivo al contatto punta −Z | D/C |
| datum caster incoerente | luce voluta 2/2 mm; rispetto al centro fitted risulta 0,0217/3,182 mm | C; contatti caster devono essere nulli |
| slip shell-shell | finestra iniziale Ft=0 e limite basato su N precedente | D/C |
| trasferimento carrier-shell inefficiente | forza esterna applicata solo al carrier; `eta_N` non misurato | D/R |

Il meccanismo del paper non richiede che la forza magnetica centrale generi da
sola il momento di salita. Richiede che le ruote spostino il carrier lungo la
superficie interna, alzando il COM; il momento gravitazionale è
`tau=m*r*g*sin(theta)`. Nel modello corrente la catena si ferma prima dello
spostamento del carrier. Una forza magnetica allineata e centrale ha momento
zero; è corretto. Aspettarsi da essa il momento di crawling confonderebbe il
vincolo di adesione con l'attuazione.

## G. Piano di correzione

Le correzioni elencate in questa sezione erano proposte al momento dell'audit;
lo stato post-audit applicato è riepilogato nella nota iniziale.

### 1. Geometria

- **Modifica minima:** chiudere la posa radiale delle sole ruote rispetto a
  `R_in=61,3472 mm`. Separatamente, usare un unico datum per shell e intero
  meccanismo interno, conservando la geometria relativa del carrier e **2 mm di
  luce su entrambi i caster**. Le due opzioni coerenti sono: ricentrare visuale
  e collider della shell sul body origin, oppure traslare l'intero meccanismo
  interno (non i soli caster) dell'offset centro-fitted meno body-origin,
  `(1,554; 0,877; 4,668) mm`, aggiornando insieme joint e punti magnetici. Se i
  caster sono puramente visuali, disabilitarne i collider; se sono fine-corsa di
  sicurezza, mantenerli collisionali ma non a contatto nella posa nominale.
  Definire infine la quota ground dal collider effettivo e verificare il datum
  shell-shell.
- **Risultato atteso:** due normali ruota-shell positive e normali caster nulle;
  gap analitico coerente con il primo N PhysX.
- **Regressione:** Test B statico, clearance ≤ rest/contact tolerance per
  entrambe le ruote; nessuna penetrazione iniziale; sweep shell-shell.
- **Rischio:** sovravincolo del carrier o penetrazione SDF se si correggono
  insieme posizioni e contact offset; un contact offset troppo grande può
  attivare artificialmente i caster pur con 2 mm di gap geometrico.

### 2. Motori

- **Modifica minima:** dopo Test A, separare esplicitamente tensione/comando,
  envelope torque-speed e slew; validare le unità PhysX di `maxForce`.
- **Risultato atteso:** coppia di avviamento coerente con la tensione richiesta
  e velocità sotto carico prevedibile.
- **Regressione:** curve tau(omega) a 20%, 50%, 100% comando e test libero a
  3/72/180/350 deg/s.
- **Rischio:** impulsi se si rimuove la rampa senza risolvere prima i contatti.

### 3. Contatti interni

- **Modifica minima:** validare proxy, materiali, offset e inerzie; non aumentare
  μ per compensare una ruota separata.
- **Risultato atteso:** carico trasferito dalle due ruote, caster scarichi e
  slip ruota-shell misurabile.
- **Regressione:** Test B con normali delle ruote, `N_caster≈0` e bilancio
  potenza.
- **Rischio:** chatter SDF e carico eccessivo dei joint.

### 4. Trasferimento carrier-shell

- **Modifica minima:** mantenere la forza esterna sul carrier e misurare
  `eta_N`; aggiungere soltanto la dinamica fisica necessaria a mantenere il
  carrier radialmente appoggiato, senza forza diretta artificiale sulla shell.
- **Risultato atteso:** `N_shell-shell` segue `Fmag,n` con ritardo piccolo e
  rapporto stabile.
- **Regressione:** confronto Test C/D, target `eta_N` definito da tolleranza
  sperimentale.
- **Rischio:** introdurre implicitamente un vincolo rigido carrier-shell.

### 5. Contatto shell-shell

- **Modifica minima:** eliminare la finestra di doppio zero dopo aver misurato
  il timing del report contatti; un solo modello tangenziale deve essere
  responsabile in ogni frame.
- **Risultato atteso:** stick disponibile dal primo impatto, senza doppio
  conteggio.
- **Regressione:** Test E PhysX/custom/entrambi, energia tangenziale mai
  positiva salvo restituzione elastica limitata.
- **Rischio:** doppia frizione o impulso se il passaggio di stato non è
  continuo.

### 6. Modello magnetico

- **Modifica minima:** conservare separazione Figura 4/Figura 5 e 9,5 N;
  correggere solo datum, segno/frame e coppia di allineamento sulla base dei
  test. Non scegliere ora una nuova legge di distanza.
- **Risultato atteso:** forza continua nel frame corretto e carrier radialmente
  stabile.
- **Regressione:** sweep gap/θ, azione-reazione, lavoro su ciclo chiuso e Test D.
- **Rischio:** doppio momento magnetico o uso di dati fuori dal dominio
  pubblicato.

### 7. Controllore

- **Modifica minima:** definire convenzione di segno nel piano di salita e
  abilitare il heading control solo dopo che assi e contatti sono validati.
- **Risultato atteso:** `t_drive·t_up>0` e componente ω lungo ±Y dominante.
- **Regressione:** Test F in quattro pose cardinali e comando avanti/indietro.
- **Rischio:** un controllore può mascherare una geometria ancora errata.

## Appendice 1 — Catena dinamica completa

| Collegamento | Ingresso e corpo/punto | Reazione attesa | Valore | Perdita/incoerenza |
|---|---|---|---|---|
| motore→joint | target velocity su joint revolute Y; drive damping 500 | coppia opposta carrier/ruota | 0,00572 N m primo frame | limite dipende dall'errore e dalla rampa |
| joint→ruota | `body0=internal`, `body1=wheel`, local rotations identità | accelerazione ruota e reazione carrier | R | asse corretto staticamente, segno salita non coerente |
| ruota→shell interna | cilindro r=16,000287 mm, μ=2,2/1,9 | `Ft<=μN`, momento `Ft*r` | clearance 2,985/2,743 mm | **catena interrotta inizialmente** |
| caster→shell | sfere r=4,65 mm, μ=0,03/0,02 | nessuna reazione nominale | luce voluta 2/2 mm; 0,0217/3,182 mm col centro fitted corrente | datum incoerente; contatto eventuale accidentale |
| carrier→precarico | +9,5 N lungo asse sul carrier, faccia magnete | −9,5 N sulla shell attiva allo stesso punto | 9,496 N radiali +0,260 N tangenziali | nessuna coppia radiale restaurativa |
| carrier→shell attiva | contatti ruote (caster esclusi nominalmente) | normale interna e trazione | `eta_N` R | **collegamento non garantito** |
| magnete esterno→carrier | Figura 4×5 alla faccia | carrier spinge shell attiva | 0 N iniziale; 2,082 N a contatto/90° | prima muove/inclina il carrier |
| shell attiva→shell passiva | collider SDF shell-shell | N PhysX e attrito | N R | custom usa N precedente; μ PhysX=0 |
| reazione passiva | wrench opposta allo stesso punto world | conservazione globale | saltata se passiva kinematic | fixture assorbe lavoro/momento non loggato |

Il precarico interno, prima che i contatti si equilibrino, tira il carrier verso
il basso e la shell verso l'alto. Sulla sola massa shell da 0,09 kg, 9,5 N meno
il peso 0,883 N corrispondono a 95,7 m/s²; sul meccanismo da 0,2179 kg, la
risultante libera sarebbe circa 33,8 m/s² verso il basso. Con la geometria di
progetto i caster non devono arrestare questo moto: la reazione deve nascere
dalle ruote. Se PhysX registra invece un impulso sul caster 1, il transitorio è
alterato dal datum incoerente. Le accelerazioni effettive sono R perché i joint
e il contatto intervengono subito.

## Appendice 2 — Geometria USD e trasformazioni

### Shell

- `metersPerUnit=1`, up axis Z.
- trasformazione `cad_reference`: translate
  `(−0,023103,−0,059745,−0,055943)` m, poi scala uniforme 0,01.
- 4356 vertici complessivi sulle superfici concentriche 61,3472 e 63,3472 mm.
- centro fit/world sorgente:
  `(0,02465733,0,06062240,0,06061104)` m.
- offset dal body origin:
  `(0,00155433,0,00087740,0,00466804)` m.
- spessore: 2,0000 mm.
- estremi radiali dei vertici per emisfero differiscono da R di meno di circa
  0,07 mm per tessellazione; non giustificano 66,5 mm.
- i collision prim hanno SDF resolution 160, margin 1 mm, narrow band 3 mm;
  nessun contact/rest offset esplicito sulle shell.

### Running gear e joint

- assi joint wheel: Y; `localRot0=localRot1=(1,0,0,0)`.
- left joint local position carrier:
  `(−0,000023,+0,033735,+0,005101)` m.
- right joint:
  `(−0,000022,−0,033564,+0,004893)` m.
- proxy runtime wheel: cilindro asse Y, raggio 16,000287 mm, larghezza
  6,002602 mm, offset assiale ±1,05277 mm, tilt X 0,16702°.
- i proxy nuovi non ri-autorizzano i contact/rest offset 1,5/0,2 mm dei vecchi
  SDF; usano i default PhysX (R).
- caster joint: sferico, centri lungo +X distanti 109,397 mm.

### Effetto dei quattro raggi citati

| valore | significato effettivo | errore rispetto a R_out/R_in reale |
|---:|---|---:|
| 0,0665 m | raggio fenomenologico/quota iniziale | +3,1528 mm su R_out |
| 0,0633472 m | superficie esterna vertici | 0 |
| 0,0613472 m | superficie interna vertici | 0 |
| 0,0605 m | raggio interno fenomenologico | −0,8472 mm |

Con 66,5 mm come raggio analitico, due shell risultano a gap zero 6,3056 mm
prima del contatto geometrico delle superfici da 63,3472 mm. Con i raggi nuovi,
il gap geometrico è corretto; resta da misurare l'anticipo dovuto alla
generazione contatti PhysX.

## Appendice 3 — Scansione delle leggi magnetiche

### Funzione realmente attiva nel pacchetto

La CLI chiama solo:

```text
run_isaac_simulation
  -> compute_external_magnetic_interaction
     -> freebot_figure4_force_curve
     -> freebot_figure5_angular_force_curve
```

Proprietà della Figura 4 attiva:

- monotona non crescente: sì;
- interpolazione lineare senza overshoot: sì;
- continua in valore fino al campione 30 mm: sì;
- derivata continua: no, cambia bruscamente a ogni campione e diventa zero dopo
  30 mm;
- cutoff: 30 mm, con valore già zero;
- scenario iniziale: fuori cutoff;
- distanza: gap shell-shell con raggi fitted;
- orientazione: applicata separatamente mediante Figura 5;
- corpo attivo: carrier;
- punto: faccia attiva del magnete.

Implementazioni presenti ma non chiamate dalla CLI:

- `fit_anchored_exponential_force_curve()`: fit esponenziale, definito ma non
  usato; λ=3,92038 mm, RMSE 0,6778 N.
- `run_freebot_emergent_docking.py`: modello esponenziale continuo o paper;
  script separato.
- `run_freebot_magnetic_locomotion.py`: esponenziale/dipolare/patch per pareti
  e rampe; script separato.
- `run_freebot_two_module_climb.py`: legge cubica realmente usata dal
  riferimento nel ramo point; le funzioni dipolari e high-μ non sono chiamate
  da quel ramo default.

### Figura 5 e posa iniziale

La tolleranza 5° modifica solo l'argomento parallelo: a θ=90° usa 85° per
`A_parallel` e 90° per `A_perp`. A gap zero:

```text
A_parallel = 2.08 N
A_perp     = 0.10 N
|F|        = 2.0824 N
```

Il flag `in_angular_range` non disabilita la forza. Oltre 90° il codice
estrapola esponenzialmente le code fino a 180°. Questa è un'assunzione non
pubblicata, non un cutoff.

## Appendice 4 — Punti di applicazione e momenti

| forza | corpo | punto | momento/nota |
|---|---|---|---|
| precarico interno | carrier | faccia magnete | circa 0,00153 N m rispetto all'origine carrier; COM reale R |
| reazione precarico | shell attiva | stesso punto | 0,01460 N m sul centro geometrico iniziale |
| magnetica esterna | carrier | faccia magnete | a θ=90°/contatto, usando origine carrier, circa 0,0468 N m soprattutto su Y |
| reazione esterna | shell passiva | prima intersezione fra la semiretta della risultante e la shell passiva | la linea fra faccia magnetica e patch è collineare alla forza, quindi il momento globale resta nullo |
| attrito custom | entrambe le shell | media dei due punti nominali | momento `r×Ft` sul centro |
| forza fenomenologica | shell attiva | `active_shell_point` nel ramo point | normale centrale: momento zero |
| adesione fenomenologica | carrier | magnete | genera braccio e momento; reazione non applicata perché ambiente passivo |
| allineamento fenomenologico | carrier/shell | coppia pura opposta | molla radiale esplicita |

Nel riferimento viene calcolata la variabile `shell_force_pos` alle righe
1198–1201, ma la chiamata usa `external_pos`. Nel ramo point i due valori
coincidono; nel ramo field-patch `external_pos` può essere il magnete, quindi il
punto calcolato non viene usato. Non è il ramo di default.

Applicare la reazione passiva nello stesso punto world della forza attiva evita
un momento netto inventato, ma sposta la reazione lontano dalla superficie
passiva per la componente trasversa. Applicarla invece alla patch senza una
coppia magnetica complementare romperebbe la conservazione del momento. Il
modello deve scegliere esplicitamente quale riduzione della distribuzione di
campo rappresenta; non basta cambiare il punto.

## Appendice 5 — Contatti e carico

Valori configurati:

| coppia | μ statico/dinamico | collider | offset espliciti runtime |
|---|---:|---|---|
| wheel-shell | 2,20/1,90 (`max`) | cylinder/SDF shell | nessuno sui proxy nuovi |
| caster-shell | 0,03/0,02 (`max`) | sphere/SDF shell, nominalmente inattivo | nessuno |
| shell-shell PhysX con custom | 0/0 | SDF/SDF | nessuno |
| shell-shell custom | 1,10/1,00 | punto analitico | tolerance 2,5 mm |
| shell-ground | 1,50/1,25 (`max`) | SDF/cube | default |

Il modello custom usa:

```text
active = signed_gap <= tolerance and normal_load > 0
x_trial = projection(x_old) + dt * v_t
F_trial = k*x_trial + c*v_t
stick se |F_trial| <= mu_s*N
slip altrimenti, |F| = mu_d*N, con return mapping
```

Il carico normale non può essere ricavato dalla sola clearance dopo che il
carrier si è mosso. Va calcolato come:

```text
wheel_share  = (N_left + N_right) / N_internal_total
caster_share = (N_c1 + N_c2) / N_internal_total
```

con normali proiettate sulla normale locale, non con il solo modulo del vettore
di contatto. Il criterio nominale è `caster_share≈0`, non semplicemente una
quota minoritaria.

## Appendice 6 — Bilancio energetico

### Energia disponibile

Alla coppia piena tensione la potenza DC meccanica ideale di una ruota è
`P=tau*omega`; il massimo della retta torque-speed è a 180 deg/s:

```text
P_wheel,max = 0.34323 * pi = 1.078 N m/s = 1.078 W
P_two,max   = 2.156 W
```

L'energia per alzare l'intero modulo di un raggio reale è:

```text
DeltaEg = 0.3079 * 9.81 * 0.0633472 = 0.19134 J
```

Il limite energetico ideale sarebbe circa 0,089 s, prima di perdite. Il
riferimento con cap 1,2 N m a 180 deg/s potrebbe numericamente erogare fino a
7,54 W complessivi, oltre il motore tabellato.

### Termini da loggare

- `P_motor_i = tau_joint_i * omega_joint_i`;
- `P_contact_pair = Ft · (v_first-v_second)`, che deve essere dissipativo in
  slip e può restituire solo l'energia elastica accumulata in stick;
- `P_mag_pair = F · (v_active_point-v_passive_point)`;
- variazione `m*g*z_COM` e energia cinetica dei sei corpi;
- dissipazione da damping dei joint e dei rigid body.

La molla di adesione fenomenologica crea istantaneamente la deflessione che
sostiene 2,943 N. L'energia elastica iniziale corrispondente è circa
`F²/(2k)=0,00361 J`; è piccola ma viene creata senza un transitorio fisico e
impedisce la caduta iniziale. La forza magnetica tabulata e il precarico
costante non hanno nel runtime un potenziale esplicitamente contabilizzato;
il loro lavoro va quindi registrato per individuare cicli che iniettano energia.

## Appendice 7 — Test diagnostici isolati

### Test A — motori senza contatti complessi

- carrier fissato o inerzia nota, ruote libere;
- step a 3, 72, 180 e 360 deg/s e rampa standard;
- log per frame: requested, slewed, actual, error, damping torque request,
  maxForce, joint torque, alpha, `tau*omega`;
- criterio: curva libera e curve sotto coppia frenante nota.

### Test B — carrier nella propria shell

- una shell, precarico 9,5 N, nessun esterno;
- log delle quattro normali, clearance, slip, asse magnete/raggio, COM modulo e
  omega relativa;
- azionare prima una ruota, poi entrambe con entrambi i segni;
- criterio: entrambe le ruote cariche, `N_caster1≈N_caster2≈0`, spostamento COM
  coerente con il paper.

### Test C — shell-shell statico

- shell già a contatto; forza nota direttamente sulla shell attiva; solo PhysX;
- misurare N, Ft, gap al primo contatto e capacità di sostenere 3,0205 N;
- questo test definisce il datum PhysX e il limite superiore `eta_N=1`.

### Test D — trasferimento carrier-shell

- stessa posa del C, forza identica sulla faccia del carrier;
- `eta_N=N_shell-shell/Fmag,n`, ritardo, rotazione carrier e corpo interno che
  prende il carico;
- criterio: eta stabile e nessun contatto caster; un impulso caster segnala
  errore di datum/contact offset.

### Test E — attrito custom

- tre run: solo PhysX, solo custom, entrambi;
- log ogni substep attorno al primo contatto;
- criterio: nessun frame con contatto/N positivi e somma delle trazioni zero;
  nessun doppio μN.

### Test F — assi di rotazione

- salvare assi world da trasformi dei joint, non dal parser;
- decomporre:
  `omega_wheel=omega·a_wheel`,
  `omega_longitudinale=omega·a_longitudinale`,
  `omega_normal=omega·n_contact`;
- calcolare a ogni frame `t_drive`, `t_up` e loro errore angolare;
- criterio di salita: componente shell attorno a ±Y dominante e incremento
  monotono di quota/COM senza orbita attorno a +X.

## Risposta finale

Gli elementi fenomenologici che compensano la fisica mancante sono: forza
esterna continua e applicata direttamente alla shell, ruote CAD effettivamente
più grandi, su mount cedevoli e già vincolate alla shell, coppia di allineamento
radiale, eventuale molla tangenziale inizializzata contro gravità,
latch/torsione e un drive a 1,2 N m senza envelope né slew.

La capacità di docking si perde prima nel tratto **motori→ruote→shell attiva**:
le ruote proxy sono separate e il motore parte con 0,00572 N m; pertanto il
modulo non entra dalla posa a 40 mm nella zona della Figura 4. Se viene portato
manualmente nella zona, si perde di nuovo nel tratto
**carrier→shell attiva→shell passiva**: la forza esterna agisce sul carrier,
che non è radialmente allineato e non ha un contatto ruota affidabile per
creare la normale shell-shell; al primo contatto la trazione shell-shell è
inoltre zero per un frame. Senza spostamento affidabile del carrier non cambia
il COM, non compare il momento `m*r*g*sin(theta)` del paper. Un'eventuale orbita
attorno all'asse longitudinale va attribuita alla dinamica/agli assi e non ai
caster, salvo che i log mostrino un loro contatto accidentale.
