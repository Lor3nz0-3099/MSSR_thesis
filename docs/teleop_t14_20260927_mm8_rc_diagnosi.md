# Diagnosi MM8 → RC — T14, 27 settembre 2026

## Esito verificato

In entrambe le ultime due registrazioni il modulo che non completa la piega è **smores_03**, collegato con BOTTOM alla LEFT di smores_02. Il target viene ricevuto ed è −45°. Gli otto azzeramenti preparatori e i due nuovi agganci riescono. Il blocco compare nella postura finale, con tutte le sette connessioni presenti.

| Registrazione | Inizio piega (tempo simulato) | Tilt 03 finale | Tilt 04 / 06 / 08 finali | Attesa registrata dalla prima richiesta di piega |
|---|---:|---:|---|---:|
| T14, avvio 15:29:15 | 165,425 s | −7,65° | −44,80° / −44,80° / −44,70° | 16,47 s |
| T14, avvio 15:42:46 | 213,492 s | −7,86° | −44,82° / −44,84° / −44,64° | 8,00 s |
| T06 riuscita, confronto | 349,992 s | −41,25° | −41,92° / −41,41° / −40,65° | successo dopo 0,20 s |

Tolleranza configurata: 0,12 rad, circa 6,88°. T06 termina correttamente entro questa tolleranza, mentre i motori continuano a mantenere il target.

Nelle T14 il modulo 03 si muove inizialmente fino a −21,32° / −20,00°, poi torna verso −8°. Il feedback finale è `MOVING_JOINT`, target comandato −0,785398 rad, senza limitazione della velocità del servo o dell'anticipo rispetto agli altri moduli. Gli altri tre feedback sono `WAITING_JOINT_GROUP_COMPLETION`: hanno raggiunto il target ma attendono 03.

Le macro registrate terminano ancora in `WAITING_POSTURE_RESULTS`, 13/17 operazioni, `done=false`. Non esiste un fallimento terminale registrato: la raccolta termina prima del timeout di 30 secondi simulati della postura. Questo dettaglio distingue il blocco fisico osservato dall'esito formale della macro.

## Cosa indica il confronto

1. Il comando tilt arriva ed è corretto. Anche il tilt di 03 aveva funzionato nell'assembly iniziale della stessa sessione e nell'azzeramento preparatorio del ritorno da MM8.
2. Durante il blocco di 03, PAN di 02 deriva da −0,29° a −156,07° nella prima prova e da +13,04° a −64,57° nella seconda. La deriva marcata segue l'inizio della piega; da sola non dimostra di esserne la causa iniziale.
3. I giunti di 03 riportano velocità istantanee elevate pur restando vicino alla stessa postura campionata: per esempio, nella prima prova tilt +6,93 rad/s e PAN −8,29 rad/s nell'ultimo campione. È un'indicazione di mancata convergenza fisica/servo, non un motore che non riceve istruzioni. I campioni non permettono di ricostruire l'oscillazione a frequenza fisica.
4. In T06 il modulo 03 è sul lato RIGHT di 02; nelle due T14 è sul lato LEFT. Cambiano quindi assegnazione, postura e condizioni meccaniche iniziali. Il confronto non è un esperimento controllato che isoli una singola causa.
5. Il braccio viene già riportato verso la postura preparatoria dalla macro: tutti gli otto goal preparatori sono `JOINT_TARGET_REACHED`. I dati non supportano l'ipotesi che queste due prove si blocchino perché manca un HOME manuale.

**Diagnosi supportata:** mancata convergenza della piega sotto vincoli/carico nel gruppo 03–02. **Causa fisica specifica ancora da isolare:** contatto/interferenza oppure conflitto dei riferimenti e delle reazioni dei servo. Non sono registrate coppie applicate e forze di contatto; `observation.contacts` è vuoto e non costituisce una prova di assenza di collisioni. Non attribuire con certezza il problema alla potenza della macchina, al PAN, o a una specifica collisione.

## Punti del controllo da verificare nella correzione

- `scripts/smores_ep/src/smores_ep/isaac/dynamic_stage.py`, `configure_structural_hold_mode`: PAN dei moduli mantenuti strutturalmente ha stiffness zero e solo damping. Questo rende plausibile la deriva sotto reazione della piega; serve verificare anche chi mantiene il target e con quale modalità viene realmente instradato ogni modulo.
- `mssr_ws/src/mssr_expert/mssr_expert/execution/parallel_assembly_executor.py`, costruzione di `_posture_structural_hold_module_ids`: ricava i moduli dal sotto-piano di assembly. Nei goal osservati del ritorno MM8→RC `hold_after_group_module_ids` contiene solo `smores_01`; mancano 02, 05 e 07. Inoltre questa richiesta agisce **dopo** il completamento del gruppo, quindi non stabilizza il telaio durante la piega e non basta da sola a spiegare il blocco iniziale.
- La piega richiede simultaneamente −45° ai quattro moduli, senza `max_coordination_lead_rad` nei goal osservati. Il codice supporta già una limitazione del disallineamento; il suo effetto su questa RC deve essere provato mantenendo gli stessi stati iniziali.
- `self_reconfiguration_executor.py` contiene un'accettazione best-effort del timeout della postura RC quando la topologia è corretta. Non viene raggiunta nelle finestre registrate. Tale accettazione non dimostra che i tilt siano a posto e non risolve il difetto fisico.

Prima di validare una correzione: prova breve della sola transizione, con logging di target effettivi, modalità del router, posizioni/velocità e sforzi dei giunti 02/03, più contatti fisici. Confrontare la configurazione attuale con una stabilizzazione esplicita del telaio durante la piega, cambiando una condizione alla volta. L'assembly iniziale deve restare nel controllo di regressione.

## Fonti locali

- `logs/teleop/recordings/teleop-a71485a6bf76465687a3cd9f2f55a12e/structural/teleop-self_reconfiguration-1790516404142051588.jsonl` — 3055 righe lette.
- `logs/teleop/recordings/teleop-aeb5be648f8f4ae48cb12cb5fe89ff42/structural/teleop-self_reconfiguration-1790517167286425988.jsonl` — 2034 righe lette.
- `datasets/expert_v1/teleop/episodes/teleop-a7d7e08f5fd14a59bd3af6afb53ccde1/structural/teleop-self_reconfiguration-1790433329997872459.jsonl` — 1527 righe lette.
- `logs/teleop/runs/teleop-t14-20260927-152915.xgG4JZ/isaac.log` e `session.txt`.
- `logs/teleop/runs/teleop-t14-20260927-154246.JQyi6d/isaac.log`, `course.json` e `session.txt`.
- Feedback finali conservati nei runtime `/dev/shm/mssr-teleop.tygJRi/primitive_status.json` e `/dev/shm/mssr-teleop.Rfpv34/primitive_status.json` (temporanei).

Analisi offline: nessuna modifica a controller, registrazioni, dataset o adapter; nessun rollout Isaac avviato.
