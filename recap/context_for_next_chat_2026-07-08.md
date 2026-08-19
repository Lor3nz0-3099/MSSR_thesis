# Contesto per prossima chat - FreeBOT Isaac Sim

## Obiettivo generale

Tesi magistrale su progettazione e simulazione di un robot modulare self-reconfigurable ispirato a FreeBOT. Il CAD e' stato progettato in Autodesk Inventor e importato in Isaac Sim 6.0 come USD fisico.

L'obiettivo non e' solo visualizzare il CAD, ma ottenere un modello fisicamente simulabile con PhysX per locomozione, magnete interno, riconfigurazione e test multi-modulo.

## Modello USD attuale

Modello fisico principale:

```text
assets/freebot/usd_physics/freebot_cad_full_longer_casters.usd
```

Il modello precedente `freebot_cad_full_shorter_casters.usd` resta disponibile come backup. Il nuovo modello punta al visual:

```text
assets/freebot/usd_visual/FreeBOT_version4_visual.usd
```

con root:

```text
/World/FreeBOT_version4_casters_radius_5_935/FreeBOT_simplified_clean
```

Nel CAD nuovo il telaio delle caster e' stato allungato. I centri delle ball caster misurati dal visual sono:

```text
caster 1: (-0.031633, 0.060363, 0.054020) m
caster 2: (+0.077764, 0.060363, 0.054020) m
```

Il modello con caster corte migliorava la locomozione base e la rampa, ma non stabilizzava abbastanza il meccanismo nella prova di salita su un altro modulo. La versione con caster allungate serve a verificare se le caster riescono a funzionare come appoggi temporanei/stabilizzatori nelle pose inclinate.

## Script principali

Banco prova locomozione, rampa e muro:

```text
scripts/isaac_freebot/run_freebot_magnetic_locomotion.py
```

Banco prova due moduli:

```text
scripts/isaac_freebot/run_freebot_two_module_climb.py
```

Il secondo script e' stato creato apposta per evitare di allungare ancora lo script rampa/muro.

## Modello magnetico attuale

La legge magnetica di default e':

```text
--magnetic-law dipole
```

Formula implementata:

```text
m = Br * V / mu0
z_eff = max(z, z_min)
F_raw(z_eff) = C * mu0 * m^2 / z_eff^4
F(z, v) = clamp(capture * alignment * (F_raw(z_eff) - D * v_normal), 0, Fmax)
```

Parametri adottati:

- `Br = 1.47 T`, dal paper FreeBOT, pari a `14700 gauss`;
- magnete `20 x 20 x 10 mm`, coerente con CAD e Table I del paper;
- forza massima esterna `22.6 N`, dal paper FreeBOT;
- `z_min = 10 mm`, per regolarizzare la singolarita' del dipolo;
- superfici ferromagnetiche trattate come ferro ideale / alta permeabilita' locale.

La legge esponenziale precedente resta disponibile:

```text
--magnetic-law exponential
```

Risultato: il modello dipolare si comporta in modo simile all'esponenziale durante la salita su rampa, ma e' piu' stabile quando il modulo resta fermo in verticale. Rimane pero' un leggero scivolamento verso il basso, probabilmente legato ad attrito e dondolio.

## Comandi principali usati

Rampa segmentata teleoperata:

```bash
/home/lorenzo/isaac/python.sh /home/lorenzo/MSSR_thesis/scripts/isaac_freebot/run_freebot_magnetic_locomotion.py --ramp-test --ramp-stair --ramp-angle-start 0 --ramp-angle-end 80 --ramp-angle-step 10 --ros2-teleop --magnetic-law dipole --ferro-static-friction 2.0 --ferro-dynamic-friction 1.6
```

Teleop ROS 2:

```bash
source /opt/ros/humble/setup.bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

Due moduli:

```bash
/home/lorenzo/isaac/python.sh /home/lorenzo/MSSR_thesis/scripts/isaac_freebot/run_freebot_two_module_climb.py --ros2-teleop
```

## Risultati principali

- La sfera e' teleoperabile via ROS 2.
- Su rampa ferromagnetica segmentata arriva fino a circa 80 gradi.
- In verticale resta piu' stabile con la legge dipolare, ma scivola ancora lentamente.
- L'aumento leggero dell'attrito aiuta ma non elimina del tutto lo scivolamento.
- Il dondolio appare soprattutto quando la superficie e' ferromagnetica, probabilmente per l'attrazione magnetica che genera momenti e oscillazioni.

## Test due moduli

Scenario corretto:

- modulo attivo: CAD completo, teleoperato;
- modulo passivo: shell sferica ferromagnetica statica;
- passiva appoggiata a terra, tangente al bordo verticale di una piattaforma;
- piattaforma non ferromagnetica, alta circa due terzi del diametro modulo.

Correzione gia' fatta: la passiva non deve stare sopra la piattaforma, ma a terra contro il bordo. Nel codice il centro e':

```text
x = platform_left_edge - shell_radius - passive_edge_clearance
z = shell_radius
```

Osservazione del test: il meccanismo interno si orienta verso la passiva, ma non riesce a mantenere la posa verticale e torna giu'. Questo indica che il problema e' probabilmente meccanico/CAD, non solo magnetico.

## Aggiornamento modello magnetico due-moduli

Il modello puntuale precedente per l'attrazione tra modulo attivo e shell passiva era stato temporaneamente affiancato da un modello a campo/patch, per provare a rappresentare meglio la zona magnetizzata sulla shell passiva. Il paper FreeBOT mostra infatti che il magnete genera un campo che attraversa la shell e magnetizza una zona esterna della shell, quindi il massimo del campo dovrebbe spostarsi con il magnete.

Per questo in `run_freebot_two_module_climb.py` e' stato introdotto:

```text
--external-model field-patch
```

Il modello:

1. considera il magnete interno come dipolo equivalente;
2. campiona una calotta della shell passiva rivolta verso il magnete;
3. calcola il campo dipolare in ogni punto campionato:

```text
B(r) = mu0 / (4 pi) * (3 r_hat (m . r_hat) - m) / |r|^3
```

4. converte il campo locale in pressione magnetica:

```text
p = B^2 / (2 mu0)
```

5. somma i contributi locali sulla calotta;
6. applica a PhysX una forza risultante equivalente, saturata a `external_max_force`.

Questo non e' ancora un FEM magnetostatico completo, ma e' piu' vicino all'immagine del paper rispetto alla forza centrale: la zona di massima attrazione si sposta sulla shell passiva seguendo la posizione del magnete.

Parametri utili:

```text
--field-cap-angle-deg
--field-rings
--field-ring-samples
--field-min-distance
--field-pressure-scale
```

Risultato dei test: anche con field-patch e con due sfere passive a sella/canale, il modulo attivo arriva a orientarsi quasi in verticale ma non ottiene un docking stabile. Tende a scendere/scivolare e non resta agganciato alla shell passiva. Il problema quindi non sembra essere solo lo scenario, ma il modello di aggancio magnetico e/o il trasferimento del momento dal magnete interno alla shell attiva.

Conclusione attuale: il default dello script due-moduli e' tornato al modello puntuale, piu' stabile e piu' semplice da interpretare:

```text
--external-model point
```

Il field-patch resta disponibile solo come opzione sperimentale:

```text
--external-model field-patch
```

## Field-patch su rampa/muro

Lo stesso concetto e' stato aggiunto anche a:

```text
scripts/isaac_freebot/run_freebot_magnetic_locomotion.py
```

per i test con rampa e muro ferromagnetici. Dopo i test, il modello con campo e patch e' risultato troppo instabile per la rampa. Quindi per rampa/muro il default e' tornato al modello puntuale:

```text
--external-model point
```

Il field-patch resta disponibile solo come opzione sperimentale:

```text
--external-model field-patch
```

La patch viene campionata sul piano della rampa o del muro attorno alla proiezione del magnete.

Nota importante: questo non e' un materiale magnetico nativo di PhysX. Isaac/PhysX non consente di assegnare direttamente un campo magnetico a un collider come proprieta' fisica standard. Il campo viene quindi calcolato a runtime nello script e convertito in forze/coppie equivalenti applicate ai corpi rigidi.

## Conclusione CAD sulle caster

Le caster corte migliorano la locomozione base, ma nel test due moduli non stabilizzano abbastanza il meccanismo quando deve stare inclinato/verticale contro la sfera passiva.

Ipotesi progettuale attuale:

- allungare leggermente i bracci caster;
- lasciare un piccolo gap tra caster e superficie interna della shell in posa neutra;
- farle entrare in contatto solo quando il telaio si inclina o dondola;
- evitare che diventino il contatto principale in locomozione piana.

Regola funzionale:

```text
ruote = trazione e contatto principale
magnete = precarico e attrazione
caster = stabilizzatori/fine corsa nelle pose critiche
```

Gap CAD consigliato iniziale:

```text
0.5 - 1.0 mm tra caster e shell interna in posa neutra
```

Se il CAD/Inventor ha tolleranze poco affidabili o la simulazione PhysX e' rumorosa, partire da:

```text
1.0 mm
```

Poi ridurre a:

```text
0.5 mm
```

se le caster entrano troppo tardi e non stabilizzano abbastanza.

Evitare gap zero: renderebbe le caster sempre in contatto e rischierebbe di riprodurre il problema precedente, cioe' caster dominanti sulla direzione di moto.

## Prossimi passi consigliati

1. Modificare CAD allungando leggermente i bracci caster.
2. Lasciare gap caster-shell interna di circa `0.5-1.0 mm` in posa neutra.
3. Riesportare STEP.
4. Reimportare in Isaac Sim, salvare nuovo USD visual.
5. Riapplicare modello fisico con gli script esistenti.
6. Ritestare prima locomozione/rampa, poi due-moduli.

## Test due moduli con sella a due sfere passive

Lo scenario `scripts/isaac_freebot/run_freebot_two_module_climb.py` e' stato modificato senza riscriverlo da capo: ora puo' creare una o due basi passive. Dopo i test con sfere analitiche, il default e' stato cambiato per usare moduli CAD/physics veri e propri come passivi bloccati:

```text
--passive-geometry module
```

La vecchia sfera analitica resta disponibile come fallback diagnostico:

```text
--passive-geometry sphere
```

Il default resta con due passive:

```text
--passive-count 2
--passive-y-spacing 0.150
```

I due moduli passivi sono clonati dal modulo CAD/physics attivo, ritargettando le relazioni interne dei joint verso il clone e impostando tutti i rigid body del clone come cinematici. In questo modo restano bloccati nello scenario ma conservano shell CAD reale e geometria del FreeBOT. Dopo il primo test, il default e' stato corretto per usare sui passivi CAD solo i collider della shell esterna:

```text
passive_module_collisions=shell-only
```

Le collisioni complete del modulo passivo restano disponibili solo con:

```text
--passive-module-full-collisions
```

Motivo: nel docking esterno devono contare shell ferromagnetica e contatto shell-shell; se l'attivo urta parti interne/wheels/caster del modulo passivo attraverso aperture o compenetrazioni numeriche, PhysX puo' generare impulsi non fisici e catapultare il modulo.

Esito aggiornato: con due moduli CAD passivi e `external_Fmax = 22.6 N` dal paper, il docking e' avvenuto almeno qualitativamente: il modulo attivo e' riuscito a salire/agganciarsi, ma poi si e' catapultato dall'altra parte. Questo e' un progresso rispetto alla sfera fake: ora il problema non e' piu' "non si aggancia mai", ma stabilizzare l'energia post-aggancio e rimuovere collisioni/impulsi spurii.

Correzione successiva: i cloni passivi si portavano dietro anche i joint interni, generando warning PhysX del tipo `cannot create a joint between static bodies`. Poiche' nel test i passivi sono ambiente bloccato e non articulation da comandare, lo script ora rimuove `/joints` dai cloni passivi. Questo dovrebbe eliminare una possibile sorgente di warning/instabilita' numerica.

Correzione concettuale successiva: nel modello puntuale esterno il `gap` non deve essere calcolato tra punto della shell attiva e shell passiva. Deve essere la distanza fisica tra il magnete interno e la superficie ferromagnetica della shell passiva. Lo script ora calcola:

```text
passive_surface_point = punto piu' vicino sulla shell passiva rispetto al magnete
surface_distance = |passive_surface_point - magnet_pos|
gap = surface_distance - proiezione della mezza altezza del magnete lungo la direzione di attrazione
F = C * mu0 * m^2 / gap^4
```

Il risultato viene saturato a:

```text
Fmax = 22.6 N
```

come riportato nel paper FreeBOT. Nel ramo `--external-model point`, l'allineamento non scala piu' la forza: prima si riparte dalla legge base distanza-forza, poi eventuali fattori di orientamento/capture potranno essere reintrodotti solo se servono per stabilita' numerica.

Aggiornamento successivo: per il docking e' stato aggiunto un modello opzionale di contatto adesivo magnetico:

```text
--magnetic-adhesion
```

Quando la distanza shell-shell e' sotto `--adhesion-gap`, il punto esterno della shell attiva corrispondente alla direzione del magnete diventa una patch magnetizzata mobile. Su quella patch vengono applicati:

```text
F_normale = forza magnetica verso la shell passiva
F_tangenziale <= mu_adhesion * F_normale
```

La componente tangenziale non e' un vincolo rigido: e' una tenuta Coulomb-limitata, cioe' un modello macroscopico di attrito/adesione generato dal carico normale magnetico. Serve a verificare se il punto magnetizzato puo' funzionare da perno locale sulla shell passiva durante la salita.

Correzione al modello adesivo: la patch magnetizzata deve muoversi con il magnete. Lo script ora calcola sempre `active_shell_point` dalla posizione corrente del magnete rispetto al centro della shell attiva. Inoltre, quando l'adesione non e' attiva, la forza adesiva precedente viene azzerata subito invece di decadere lentamente: in questo modo non resta una forza residua applicata a un punto vecchio o nullo. Il log stampa anche:

```text
adh
patch_rel
```

dove `patch_rel` e' la posizione della patch rispetto al centro della shell attiva. Se il magnete si muove dentro la shell, `patch_rel` deve cambiare.

Revisione successiva: l'adesione non deve essere applicata come forza normale+tangenziale sulla shell attiva. Il contributo fisico parte dal magnete verso la superficie passiva. Quindi lo script ora applica:

```text
Fmodule = forza normale magnete -> superficie passiva, applicata al magnete/internal_link
Fadh_tan = sola tenuta tangenziale, applicata sempre al magnete/internal_link
```

La forza adesiva tangenziale usa la velocita' del magnete rispetto alla normale verso la superficie passiva ed e' limitata da:

```text
Fadh_tan <= adhesion_friction * Fmodule
```

In questo modo non viene duplicata la forza normale sulla shell attiva e il punto di adesione segue il magnete per costruzione.

Il log ora stampa anche:

```text
passive_idx
```

che indica quale delle due sfere passive sta contribuendo maggiormente all'attrazione magnetica esterna.
