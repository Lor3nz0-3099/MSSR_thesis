# Audit della prova T01 del 25 settembre 2026

## Prompt operativo

> Individua la prova T01 avviata il 25 settembre 2026 verso le 11:40 usando
> orari, missione e manifest. Verifica tutte le righe dei flussi umani e
> strutturali: JSON, ordine temporale, grafi prima/dopo l'azione, otto moduli,
> connessioni, attributi necessari all'IL, geometria identificata dall'hash e
> validità delle etichette. Ricostruisci la sequenza scale → RC → manipolatore
> → tentativo di ritorno a RC e distingui i tentativi riusciti da quelli
> falliti. Escludi le prove precedenti alle scale e il ritorno finale MM8→RC
> incompleto. Tratta la pressione del pulsante come successo confermato
> dall'operatore e conserva il prefisso T01 nel dataset, distinguendo
> provenance umana e stato registrato. Non modificare gli originali.

## Identificazione e provenienza

- Launcher: `logs/teleop/runs/teleop-t01-20260925-114404.OsrpQe/`.
- Registrazione: `logs/teleop/recordings/teleop-69b37b8db7e141339518d4eef585e4bd/`, dalle **11:51:11 alle 12:32:05 CEST**. La registrazione delle 10:46–11:19 è un'altra prova.
- Missione nel grafo: `teleop-t01`, layout versione **5**, hash della geometria
  `d624316f48f47e73908a6e23441e8ba9566849e6cef57a28091506e215d8e709`;
  coincide con la copia `course.json` nella cartella del launcher.
- Manifest: `status=completed`, `eligible_for_import=true`, `task_success=null`.
  `completed` indica che il file è stato chiuso correttamente, non il successo
  dell'intera missione; il successo del pulsante è stato confermato successivamente dall'operatore.

## Integrità dei grafi

Ho letto **tutte le 31.319 righe JSONL**: 4.344 umane e 26.975 strutturali.
Nessuna riga JSON illeggibile, senza `graph_t`/`graph_t_plus_1`, con timestamp
invertito, senza attributi globali, con meno di otto nodi o senza le sette
connessioni. Ogni nodo ha identificativo, posizione, attuatori e connettori.
Le transizioni hanno anche azione esperta e osservazione. Nel flusso umano
tutte le azioni sono marcate valide per behavior cloning.

I 4.344 grafi umani sono `mssr.robot_graph.v2`: contengono pose, orientamenti,
velocità, stato dei giunti e connettori, oltre all'intera geometria T01 e allo
stesso hash. Anche i 5.604 campioni della macro scale usano quel grafo e hash.
I 21.371 campioni delle riconfigurazioni usano invece
`mssr.attributed_graph.v3`: hanno gli otto moduli, la topologia e lo stato
necessari all'esperto strutturale, ma **non** incorporano il campo `course`.
Per addestrare un modello agnostico rispetto ai backend, l'adapter dovrà
normalizzare i due schemi e associare la geometria della missione tramite
manifest/hash, senza supporre che sia ripetuta in ogni riga strutturale.

Il manifest configura 10 Hz ma misura **1,77 campioni umani/s di tempo reale**.
Quando la teleoperazione umana è attiva, 4.327 intervalli consecutivi hanno
passo di circa 0,033 s di tempo simulato. Cinque intervalli lunghi coincidono
con le macro strutturali, registrate nei loro flussi separati; il maggiore,
208,658→310,025 s, è coperto dal gait delle scale. Quindi il dato di 1,77 Hz
non dimostra da solo perdita di frame durante la guida. La simulazione e la
registrazione non avanzano alla stessa velocità del tempo reale.

## Timeline in tempo simulato

| Intervallo (s) | Evento | Esito dai dati |
|---|---|---|
| 122,692–123,792 | primi 34 campioni umani Snake8 | presenti prima del gait delle scale |
| 123,958–160,225 | Snake8→RC, 3.351 righe | successo, topologia verificata |
| 160,225–168,892 | RC, 261 campioni umani | presente |
| 169,025–207,858 | RC→Snake8, 3.107 righe | successo, topologia verificata |
| 208,792–310,058 | gait scale, 5.604 righe | successo, `progress=1.0` |
| 310,025–313,358 | Snake8, 101 campioni umani | presente dopo le scale |
| 313,525–349,425 | Snake8→RC, 4.775 righe | successo, topologia verificata |
| 349,425–445,625 | RC, 2.884 campioni umani | passa oltre i sette coni; a 428,725 s il centro è circa (4,075; −1,687) m |
| 443,758–443,792 | primo RC→manipolatore, 7 righe | **fallito**: `RESOURCE_BUSY` su `smores_03` |
| 445,725–468,625 | secondo RC→manipolatore, 2.381 righe | successo, topologia verificata |
| 468,658–503,558 | manipolatore, 1.039 campioni umani | pressione del pulsante confermata dall'operatore |
| 503,758–566,358 | manipolatore→RC, 7.750 righe | **incompleto**: termina in `WAITING_POSTURE_RESULTS`, senza stato terminale riuscito |

Il breve tentativo RC→MM8 delle 443,758 s fallisce per `RESOURCE_BUSY`;
quello delle 445,725 s è la ripartenza riuscita. Per l'IL strutturale il primo
flusso non è un esempio riuscito, ma non invalida la dimostrazione T01 che
prosegue fino al pulsante. Il flusso umano termina a 503,558 s e il suo ultimo
`graph_t_plus_1` è a 503,592 s: entrambi precedono il tentativo finale
MM8→RC a 503,758 s. Nei dati di questa sessione non compare una ripartenza
riuscita di quel ritorno a RC. La copia curata esclude l'intero flusso finale.

## I quattro tilt del tentativo finale

Il gruppo `smores_03`, `smores_05`, `smores_06`, `smores_08` riceve a 525,3 s
il target **−0,7854 rad**. Tutti e quattro i goal scadono dopo 30 s e vengono
ritentati. Nell'ultimo grafo, a 566,358 s, `smores_03`, `05` e `06` misurano
circa −0,780, −0,782 e −0,782 rad: sono vicini al target; `smores_08` misura
**−0,086 rad**, distante circa 0,70 rad. Questo conferma il blocco osservato
su un modulo e spiega perché il gruppo non completa la postura. I log non
dimostrano da soli la causa meccanica del blocco.

## Diagnosi del ritorno manipolatore→RC

Il log Isaac mostra i due distacchi e i due docking necessari, poi
`rigid connections: 7/7` e `TARGET TOPOLOGY REACHED`. La macro ha
completato **13 delle 17 operazioni**; le quattro rimanenti sono i tilt
coordinati dei moduli `smores_03`, `05`, `06`, `08` a **−0,7854 rad**,
con tolleranza **0,12 rad**. I goal sono ammessi a circa **525,3 s**.
A 526,0 s i primi tre misurano circa −0,78 rad, mentre `smores_08`
misura circa −0,085 rad e resta a quel valore fino all'ultimo grafo
(566,358 s), con errore di circa **0,70 rad**. Tutti e quattro i goal
scadono a circa 555,3 s perché il gruppo termina solo quando ogni
membro raggiunge il target; vengono poi ritentati. La registrazione finisce
durante il primo retry in `WAITING_POSTURE_RESULTS`: **non contiene un
verdetto terminale di fallimento o successo** per questa macro.

| | Assembly iniziale verso RC | Ritorno MM8→RC |
|---|---|---|
| Stato iniziale | Moduli separati sulla zona di partenza | Manipolatore già connesso, dopo uso di PAN/TILT e pulsante |
| Costruzione | Onde progressive fino a sette connessioni | Cinque connessioni conservate; distacco e nuovo docking di due moduli |
| Gruppo tilt finale | `03, 04, 06, 08` | `03, 05, 06, 08` (diversa assegnazione fisica degli slot) |
| Risultato osservato | Quattro goal di postura `SUCCEEDED` | Tre tilt vicini al target; `08` resta vicino a −0,085 rad |

L'assembly iniziale e Snake→RC più tardi nella sessione mostrano che
`smores_08` **può** raggiungere una postura RC in altre configurazioni.
Nell'ultimo ritorno è ancora connesso a `smores_04:RIGHT`, ma è in una
posa diversa dopo l'uso del manipolatore. Il target −0,785 rad rientra nei
limiti del giunto; non c'è un errore di ammissione o un messaggio PhysX
specifico del tilt finale. Nei **3.976 campioni** fra 526,0 e 555,3 s il tilt di `smores_08`
resta fra **−0,0854 e −0,0849 rad**, mentre la velocità assoluta riportata
ha mediana **3,28 rad/s**. Questa coppia di misure non descrive un semplice
moto continuo verso il target; può indicare oscillazione rapida non risolta
dai campioni o un'incoerenza della telemetria. In ogni caso la posizione
misurata non raggiunge il target: questo conferma il
disaccordo fra comando e posizione, ma **non permette di distinguere** con
certezza collisione/contrasto meccanico, saturazione del servo o incoerenza
della telemetria della velocità. Il flusso non contiene forze di contatto
sufficienti per una diagnosi più specifica. Alcuni errori PhysX sul range
di un target revolute compaiono **prima** di questo tentativo, durante
l'attività precedente; il log non li associa al tilt `08` finale.

La logica del controller spiega l'effetto di gruppo: i tre membri arrivati
a target restano energizzati e attendono `08`, perciò scadono tutti con
`TIMEOUT`. Dopo l'esaurimento dei retry, il codice per RC può accettare
una postura finale fuori tolleranza come **best effort** se la topologia è
esattamente quella desiderata. Questo ramo non è stato raggiunto nei dati
salvati; se usato, deve essere annotato separatamente da una RC con tutti
i quattro tilt effettivamente a target.

Per isolare la causa fisica alla prossima prova servono almeno: target e
posizione del tilt di ogni modulo nel tempo, effort/coppia applicata,
contatti per `smores_08`, e stato del giunto prima/dopo docking. Una prova
controllata a parità di posizione sul pianerottolo, partendo una volta da
Snake8 e una volta da MM8, permetterebbe di distinguere l'effetto della
morfologia sorgente da quello del terreno.

## Perché il numero totale si riduce

La registrazione originale T01 contiene 31.319 righe; la copia curata ne
contiene 16.784. La differenza di 14.535 righe è composta esattamente da:

| Segmento escluso | Righe |
|---|---:|
| Campioni umani delle prove iniziali | 320 |
| Due riconfigurazioni delle prove iniziali | 3.351 + 3.107 = 6.458 |
| Tentativo RC→MM8 rifiutato per `RESOURCE_BUSY` | 7 |
| Tentativo finale MM8→RC incompleto | 7.750 |

Il flusso **umano T01 scende soltanto da 4.344 a 4.024 campioni**. Non è
stato decimato o ricampionato: tutte le righe conservate sono identiche
all'originale. Il gait delle scale (5.604 righe) e le due riconfigurazioni
riuscite dopo le scale (4.775 e 2.381 righe) sono mantenuti integralmente.

## Decisione sui dati per IL, dopo la conferma dell'operatore

L'operatore conferma che il pulsante è stato premuto e considera T01 riuscita
fino a quel punto. Il successo del pulsante è dunque **un'etichetta umana**:
il manifest originale resta `task_success=null` e i log non forniscono qui
una misura indipendente di attivazione. Il dataset curato registra
`task_success=true` con `success_basis=user_confirmed_button_press_2026-09-25`
e `full_mission_success=false`, perché il ritorno opzionale a RC e il goal
terminale non sono stati verificati in questa copia.

La copia in `datasets/expert_v1/teleop` contiene **16.784 transizioni** (4.024 umane e 12.760 delle macro riuscite), delle quali 16.781 marcate valide per behavior cloning. Parte dal gait delle scale a
**208,792 s**. I primi campioni umani conservati riprendono a 310,025 s dopo
la macro; include la riconfigurazione Snake→RC riuscita, la guida oltre i coni,
il retry RC→MM8 riuscito e il segmento MM8 fino a 503,558 s. Esclude tutti
i flussi delle prove precedenti alle scale, le sette righe del primo
RC→MM8 rifiutato e il flusso MM8→RC incompleto. Ogni riga conservata resta
byte per byte uguale all'originale. Il manifest curato contiene intervalli,
hash, origine dei file e ragione della selezione. Gli originali restano intatti.
