# Audit teleoperazioni composite — 28 settembre 2026

## Criterio concordato

La dimostrazione è completa per la campagna se supera gli ostacoli previsti. **La piattaforma finale non è richiesta**: l'operatore ha interrotto volontariamente le registrazioni dopo il corso. `status=completed` significa solo che il recorder è stato chiuso; nei raw `task_success=null` e i campioni umani hanno `done=false`, quindi la riuscita va attribuita sulla base di macro, stato fisico e conferma dell'operatore.

## Risultato

**Dieci episodi compositi candidati utilizzabili:** i quattro già curati T02, T03, T05 e T06, più uno per ciascuno dei nuovi stage T14, T15, T10, T16, T09 e T07. I tre episodi già curati C05, T01 e T04 restano prefissi utili, con scope diverso, fuori da questo conteggio. Al 28 settembre le sei nuove prove erano solo in `logs/teleop/recordings/`. Aggiornamento del 29 settembre: sono state curate e compattate in `datasets/`; le copie dei log sono state poi eliminate dopo verifica SHA-256.

| Stage | Raw ID da conservare | Ostacoli e seed nell'ordine | Evidenza osservata | Righe umane |
|---|---|---|---|---:|
| T14 | `teleop-952c36750c624fb19de8421019325d13` | button 6103 → RC 6017 → stairs 3105 | depressione pulsante ~4 mm, RC→MM8→RC e RC→Snake terminali, macro scale terminale, tutti i moduli sul piano alto | 3.888 |
| T15 | `teleop-39e2f4b4cd684e9ea72734cc21baca94` | button 6101 → RC 5104 → gap 4105 | depressione ~4 mm; retry RC→Snake terminale; macro gap terminale; tutti i moduli oltre bordo lontano (x > 6,172 m) | 4.396 |
| T10 | `teleop-0e4e34fe3a5f4ed1b6a86642acf4fac8` | RC 6017 → stairs 3105 (−Y) → RC 5100 | RC→Snake, macro scale e Snake→RC terminali; ultimi moduli oltre l'uscita della seconda tratta RC | 4.009 |
| T16 | `teleop-5e0c4d97db80492bbb561b16ef132949` | RC 5100 → stairs 3101 (+Y) → gap 4103 (+Y) | RC→Snake e macro scale/gap terminali; tutti i moduli oltre il bordo lontano del gap (y > 5,503 m) | 4.521 |
| T09 | `teleop-ce8293656bb444ac999f87c0520672fc` | gap 4107 → stairs 3101 (+Y) → button 9974 | macro gap/scale e quattro reconfiguration terminali; pulsante depresso ~4 mm | 5.049 |
| T07 | `teleop-fb69637be28d49ccbff572e721cfae24` | stairs 6403 → RC 5103 → button 8936 (−Y) | macro scale e quattro reconfiguration terminali; pulsante depresso 3,89 mm, RC finale ripristinata | 6.588 |

La soglia usata per la pressione fisica del pulsante nei curatori precedenti è 3,5 mm. `goal` rimane nella missione, ma non è usato come requisito di questa campagna.

## Tentativi e segmenti da non promuovere come successi

- **T14:** la registrazione sopra è la più recente del 27 settembre (run `teleop-t14-20260927-163145.sNCjwM`). Le due prove delle 15:29 e 15:42 fermano MM8→RC a 13/17 durante il tilt di `smores_03`; la prova delle 16:06 arriva solo a metà percorso; quella delle 16:22 si ferma dopo il pulsante e una riconfigurazione incompleta. Non contare le versioni precedenti come nuovi episodi.
- **T15:** il primo RC→Snake (`teleop-self_reconfiguration-1790522607927123138`) fallisce a 304,425 s simulati per `RESOURCE_BUSY internal_motion:smores_02`, occupata da un goal RC ancora attivo. Il retry avviato a 306,958 s riesce. La breve macro fallita non va etichettata come successo; il successivo completamento del task è verificato.
- **T16:** il primo tentativo (`teleop-cd1f43700bdb47428e089d6e427bd36b`) finisce durante l'allineamento RC→Snake; usare il secondo ID della tabella.
- **T07:** il tentativo delle 13:19 (`teleop-7e71b72ac8954b7583d370957f810a9d`) ha solo 355 righe umane e si interrompe durante la macro scale; usare quello delle 13:31.
- **T07 del 26 settembre** (`teleop-cc2da0b28bcd42ae95837bcc8e62497c`) non ha pressione del pulsante e termina con MM8→RC incompleta; non conta.
- Le registrazioni T14 intermedie contengono tentativi di riconfigurazione riusciti, ma non tutti gli ostacoli previsti. Possono offrire segmenti isolati se in futuro serve; non valgono come episodi compositi completi.

## Qualità e limiti dei sei nuovi episodi

- **28.451 righe umane:** tutte e sei le serie hanno il numero di righe dichiarato nel manifest, timestamp simulati strettamente crescenti e senza duplicati, `action_valid=true`, `graph_t` e `graph_t_plus_1` presenti in ogni riga. Ogni manifest termina `completed` con `unwritten_records=0`.
- Le sei serie contengono **25 stream strutturali terminali riusciti**. Quattro stream iniziali di self-assembly (T10, T16, T09, T07) restano senza terminale registrato perché la registrazione/interazione subentra durante l'assembly; il primo campione umano mostra comunque 7 connessioni e la morfologia attesa. Non attribuire successo a questi quattro stream. Il quinto stream non riuscito è il primo tentativo RC→Snake di T15.
- La frequenza misurata sul tempo di parete è circa **1,95–3,65 righe/s** contro 10 Hz configurati, coerente con il rallentamento di Isaac. La mediana degli intervalli tra timestamp simulati è ~0,033 s; i grandi salti fra righe umane coincidono con le macro strutturali registrate in stream separati. Per IL serviranno timestamp e maschere di disponibilità, senza presumere un passo temporale uniforme.
- I campioni umani raw usano `stage_name=human_behavior` e `stage_id` come indice progressivo del campione, non come identificatore dell'ostacolo; la missione e la geometria sono nel grafo, ma **l'ostacolo corrente non è ancora un'etichetta esplicita affidabile in questi raw**. La fase successiva dovrà ricostruirlo e documentare intervalli ambigui; non trattare il nome del file come label di ogni riga.
- I dati delle macro hanno `producer=deterministic_expert`, quelli di guida `producer=human_expert`. Nella preparazione per IL i due tipi di supervisione vanno mantenuti distinguibili. Per T15 il breve stream `RESOURCE_BUSY` va escluso o marcato come retry fallito.
- Le missioni con flat navigation non registrano un flag automatico che certifichi l'assenza di contatti con ogni cono. Il completamento di questi tratti poggia sulla progressione fisica oltre il corso e sulla conferma dell'operatore; una revisione visiva resta utile se serve certificare specificamente il non-contatto.

## Copertura dei percorsi

Nei dieci episodi candidati compaiono dieci combinazioni di seed; l'ordine gap→RC→button è ripetuto in T02 e T06, mentre gli altri ordini aggiungono casi nuovi. Le due dimostrazioni con pulsante all'inizio sono T14 e T15. T15 e T16 terminano con il gap; T07 e T09 mettono il pulsante dopo le scale. La campagna copre tutte le morfologie richieste, ma è ancora composta solo da SMORES-EP; la normalizzazione verso FreeBOT appartiene alla fase dell'adapter successiva.

## Fonti e metodo

Manifest, JSONL strutturali e file `human_behavior.jsonl` in `logs/teleop/recordings/`; missioni in `logs/teleop/runs/*/course.json`; manifest curati in `datasets/expert_v1/teleop/episodes/`. Per i sei ID della tabella sono state controllate tutte le righe umane per timestamp, `action_valid` e presenza dei grafi; sono stati letti i terminali di tutti gli stream strutturali e gli stati iniziali/finali dei grafi. Nessun file sorgente è stato modificato da questo audit.
