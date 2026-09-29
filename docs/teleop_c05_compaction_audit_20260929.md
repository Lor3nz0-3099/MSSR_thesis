# Audit della compattazione pilota C05 — 29 settembre 2026

Aggiornamento: il batch dei 12 episodi restanti è stato completato successivamente;
lo stato finale è in `docs/teleop_compaction_audit_20260929.md`. Le note
seguenti descrivono il checkpoint pilota precedente al batch.

Su richiesta dell'operatore la compattazione si ferma a C05
(`teleop-f8888f12c46249fc80ef0af8c56b4dcf`). Il raw curato in
`datasets/expert_v1/teleop` resta autorevole. C05 è un prefisso certificato,
non una missione composite completa. Non è stato creato un manifest della
raccolta compatta perché gli altri 12 episodi non sono stati elaborati.

| Stream | Raw | Record semantici | Raggruppati | Raw byte | JSONL semantico byte | Archivio byte |
|---|---:|---:|---:|---:|---:|---:|
| Guida umana | 5.099 | 5.099 | 0 | 1.372.291.080 | 444.259.424 | 34.468.914 |
| Macro snake→RC | 3.140 | 2.209 | 931 | 855.460.991 | 375.709.324 | 8.576.490 |
| Macro RC→MM | 1.506 | 1.368 | 138 | 410.190.527 | 230.155.495 | 2.493.716 |
| **Totale** | **9.745** | **8.676** | **1.069** | **2.637.942.598** | **1.050.124.243** | **45.539.120** |

La pulizia semantica riduce i byte JSONL del **60,19%** prima di zstd.
I 1.069 raggruppamenti sono il **23,01%** dei 4.646 campioni macro.
La riduzione totale su disco con zstd è del **98,27%**; questo secondo
numero include la compressione del contenitore e non misura la sola pulizia.

La geometria del corso si trova nella base degli attributi globali del
grafo nei metadati dello stream umano. Le variazioni dinamiche restano
nei record. Alias esatti e campi uguali alla base sono ricostruibili dal
codec. Le macro raggruppano soltanto transizioni consecutive semanticamente
identiche; i valori temporali di ogni campione restano nel gruppo.

Una verifica indipendente ha decodificato i tre archivi e confrontato ogni
record con il rispettivo JSONL raw: **9.745/9.745 uguali**. Le macro
contengono 927 e 136 gruppi ripetuti, con lunghezza massima 3.
Lo stream umano contiene 3.756 coppie consecutive con azione identica
e 591 righe con comando di guida nullo; tutti i suoi 5.099 campioni
sono conservati. Azioni uguali non dimostrano da sole una pausa, ma
mostrano perché un RLE indiscriminato cancellerebbe campioni umani utili.

Il compattatore ha verificato SHA-256 e conteggi dei raw e degli archivi;
la suite mirata dei dataset tool ha dato 19 test superati. Prima dello stop
erano stati verificati per SHA-256 i raw curati da C05 fino a T10
nell'ordine del manifest; l'audit degli ultimi tre episodi è stato
interrotto per rispettare lo stop a C05. I 60 JSONL raw di tutti i
13 episodi risultano presenti nel dataset, ma questo controllo di
presenza non sostituisce l'audit completo dei loro hash.

## Controllo del contratto nelle macro

Le sole macro curate di C05 sono due stream di riconfigurazione:
Snake8→RC-Car8 (3.140 righe ricostruite) e RC-Car8→MobileManipulator8
(1.506 righe ricostruite). In entrambi, ogni riga contiene
`graph_t`, `observation`, `expert_action` e `graph_t_plus_1`.
Tutte le 4.644 coppie adiacenti interne agli stream soddisfano
`graph_t_plus_1[i] == graph_t[i+1]` esattamente. Ogni macro termina
con `done=true`, `success=true` e una singola riga
`action_valid=false`; quella riga serve come terminale, non come
target per behavior cloning.

Gli esempi estratti dal compatto sono:
`datasets/expert_v1_compact/teleop/examples/c05_macro_snake_to_rc_5_12.jsonl`
e
`datasets/expert_v1_compact/teleop/examples/c05_macro_rc_to_mm_2_9.jsonl`.
Ogni file contiene otto transizioni consecutive complete
`G_t, o_t, a_t, G_t_plus_1`, verificate campo per campo contro i raw.
Le righe 10–11 del primo e 7–8 del secondo provengono ciascuna da un
solo record semantico con ripetizione; `source_compact` ne conserva
indice, offset e cardinalità. La label `rc-1` o `button-1` in
`o_t.obstacle_context` è derivata dalla destinazione della
riconfigurazione con confidenza alta; non è una label raw per riga.
