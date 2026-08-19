# Specifica per docking e arrampicata magnetica FreeBOT

## Obiettivo

Implementare in Isaac Sim 6.0/PhysX un docking fisicamente plausibile tra un
modulo FreeBOT attivo e una shell passiva ferromagnetica.

Il modulo attivo e composto da:

- una shell sferica dinamica;
- un meccanismo interno dinamico, libero rispetto alla shell;
- ruote e caster che trasferiscono moto e carico alla superficie interna;
- un magnete solidale al meccanismo interno.

## Comportamento richiesto

1. Il magnete mantiene il meccanismo interno aderente alla superficie interna
   della shell attiva.
2. Avvicinandosi a una shell passiva, il magnete viene attratto dalla sua
   superficie ferromagnetica e porta il meccanismo in posizione verticale.
3. L'attrazione esterna e calcolata tra la posizione corrente del magnete e il
   punto piu vicino della superficie passiva.
4. Il punto magnetizzato sulla shell attiva e la proiezione radiale corrente del
   magnete: deve quindi muoversi insieme al meccanismo.
5. In prossimita del contatto, la patch shell-shell deve fornire una tenuta
   tangenziale compliant con regime stick/slip e limite di Coulomb. Non deve
   essere creato un fixed joint o un vincolo rigido istantaneo.
6. Quando le ruote muovono il meccanismo, la shell attiva deve poter rotolare
   attorno alla patch di contatto e salire sulla shell passiva. La patch deve
   aggiornarsi continuamente seguendo magnete, shell e geometria passiva.
7. La forza normale magnetica e la tenuta tangenziale equivalente agiscono sul
   magnete/meccanismo interno. La seconda rappresenta la catena di attrito non
   risolta magnete-shell attiva-shell passiva e sostiene il meccanismo verticale.
   La shell attiva non deve ricevere un ancoraggio tangenziale artificiale: deve
   restare libera di rotolare attraverso i contatti PhysX reali.

## Requisiti numerici

- Usare la distanza magnete-superficie, corretta con la semidimensione del
  magnete lungo la direzione di attrazione.
- Saturare la forza esterna a 22.6 N e limitarne la velocita di variazione.
- Non annullare l'attrazione quando il dipolo non e ancora normale alla
  superficie: usare una dipendenza pari dall'orientamento con baseline non
  nulla e lasciare che la forza eccentrica generi la coppia di allineamento.
- Usare una salita della forza limitata e un rilascio molto piu rapido, azzerando
  subito vettori residui fuori dalla regione di cattura.
- Calcolare la velocita della patch mediante velocita lineare e angolare della
  shell attiva.
- Usare una molla-smorzatore tangenziale proiettata sul piano locale, limitata da
  `F_t <= mu * F_n`, azzerando lo stato quando il contatto viene perso.
- Distinguere la soglia di contatto tra shell dalla tolleranza geometrica della
  patch magnetizzata, per tenere conto di collider CAD/SDF e raggio nominale.
- Applicare realmente il materiale ferromagnetico ai collider SDF dei moduli
  passivi clonati.
- Evitare commutazioni rapide tra due passive mediante isteresi; validare prima
  lo scenario con una sola passiva.

## Diagnostica richiesta

Registrare almeno:

- forza magnetica interna ed esterna;
- distanza shell-shell e magnete-superficie;
- allineamento e capture;
- stato `free`, `stick` o `slip` della patch;
- forza tangenziale e limite di Coulomb;
- punto mobile della patch rispetto al centro shell;
- coppia magnetica esterna sul meccanismo;
- coppia di tenuta tangenziale applicata al magnete/meccanismo;
- velocita angolare relativa tra shell e meccanismo interno;
- indice della passiva selezionata e comando ruote.

## Criterio di successo

Il modulo deve potersi agganciare senza impulsi di lancio, restare verticale a
ruote ferme senza scivolamento macroscopico e, comandando le ruote, rotolare
progressivamente sulla passiva senza fixed joint o teleport.

La velocita di locomozione generale non deve essere ridotta dal modello di
docking. Per prove di avvicinamento lento si deve usare un parametro esplicito
(`--cmd-linear-scale`) senza cambiare il default della locomozione.

## Configurazione di confronto con la rampa

Il test a due moduli usa temporaneamente come default una configurazione
compatibile con il test rampa, senza modificare quest'ultimo:

- distanza magnetica dalla patch della shell attiva alla superficie passiva;
- capture gap sferico di 25 mm;
- distanza dipolare minima di 10 mm;
- allineamento positivo elevato al quadrato;
- salita/rilascio del solo modulo a 80/160 N/s, ricostruendo la direzione dalla
  normale sferica corrente e azzerando la forza fuori capture;
- forza equivalente applicata sia al magnete sia alla patch della shell;
- modello tangenziale stick/slip sul meccanismo attivo per default, necessario
  per impedire la fuga tangenziale che non esiste nel caso piano della rampa.

Questa configurazione serve a effettuare un confronto controllato. La forza
aggiunta alla shell e un'approssimazione equivalente ereditata dal test rampa e
non va interpretata come una seconda interazione magnetica indipendente.

La tenuta tangenziale usa una tolleranza patch di 20 mm e shell di 6 mm. Dopo il
filtraggio viene nuovamente proiettata sul limite di Coulomb istantaneo, così il
ritardo di rilascio non può mantenere una forza superiore a `mu * F_normal`.

Il docking usa inoltre un latch geometrico compliant: entra quando il coseno di
allineamento supera 0.72 e la distanza tra patch scende sotto 25 mm; esce sotto
0.30 o oltre 45 mm. All'ingaggio stabilisce subito 6 N di precarico normale e
inizializza la piccola deformazione tangenziale necessaria a bilanciare il peso
del meccanismo da 0.300 kg. Il latch non blocca pose o velocità e tutte le forze
restano limitate dal modello magnetico e dalla legge di Coulomb.

Per evitare che il comando di avvicinamento continui a spingere durante il
transitorio di aggancio, al primo latch il drive viene portato a zero e resta
disarmato finché il comando ROS non torna neutro una volta. La successiva
pressione viene interpretata come comando deliberato di salita. Nella baseline
mock la tenuta tangenziale è limitata da
`min(mu * F_normal, 22.6 N)`. Il cap è una scelta comportamentale esplicita, non
una misura tangenziale derivata dal paper.

Poiché il dipolo puntuale è assialsimmetrico, da solo non genera coppia attorno
alla normale magnetica. Il magnete CAD rettangolare viene quindi rappresentato
anche da una molla-smorzatore torsionale compliant attorno a tale normale,
limitata a 0.12 N m. Questa componente conserva la direzione di marcia acquisita
al docking ma può saturare e scorrere; non limita la rotazione necessaria alla
salita negli altri due assi.

Durante il distacco teleoperato un comando inverso apre il latch e disattiva
temporaneamente soltanto adesione tangenziale e torsione. La forza dipolare
normale resta attiva e continua a essere ricalcolata dalla posizione del magnete
e dalla patch corrispondente. L'undock termina dopo separazione geometrica e
rilascio del comando inverso. La forza tangenziale decade automaticamente con
la normale perché il suo limite è `mu * F_normal`.

## Default comportamentale corrente

Il tentativo con collider fisico del magnete e soli attriti PhysX non ha prodotto
un docking o una salita affidabili. Il default è quindi tornato al modello
reduced-order che riproduce il comportamento desiderato:

- magnete e telaio interno visual-only;
- forza esterna limitata a 22.6 N;
- forza esterna equivalente distribuita su magnete e patch mobile della shell;
- adesione tangenziale compliant attiva, limitata da `mu*Fn` e da 22.6 N;
- torsione compliant attiva;
- latch, pausa al docking e switch di undock attivi.

Questa configurazione è un mock fisicamente informato: geometria, distanze,
forza normale e limiti Coulomb hanno base fisica, ma distribuzione delle forze,
adesione, torsione e undock sono surrogate comportamentali. Devono essere
presentati come tali nella tesi.
