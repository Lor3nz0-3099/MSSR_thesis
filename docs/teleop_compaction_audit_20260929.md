# Audit della compattazione teleop — 29 settembre 2026

La raccolta `datasets/expert_v1_compact/teleop` contiene tutti i 13 episodi
raw curati in `datasets/expert_v1/teleop`: 10 corsi con ostacoli
completati e tre prefissi C05, T01 e T04. I raw restano la fonte autorevole.
Sono presenti 60 archivi, uno per stream, e un manifest per episodio
più il manifest della raccolta. Non restano directory di staging.

## Conteggi

| Tipo | Stream | Campioni raw | Record semantici | Raggruppati |
|---|---:|---:|---:|---:|
| Guida umana | 13 | 59.635 | 59.635 | 0 |
| Macro deterministiche | 47 | 160.065 | 128.379 | 31.686 |
| **Totale** | **60** | **219.700** | **188.014** | **31.686** |

| Episodio | Campioni raw | Record semantici | Raggruppati |
|---|---:|---:|---:|
| C05 | 9.745 | 8.676 | 1.069 |
| T01 | 16.784 | 13.138 | 3.646 |
| T02 | 17.368 | 14.886 | 2.482 |
| T03 | 11.905 | 10.319 | 1.586 |
| T04 | 14.427 | 12.228 | 2.199 |
| T05 | 8.410 | 8.107 | 303 |
| T06 | 15.439 | 13.411 | 2.028 |
| T14 | 17.242 | 16.136 | 1.106 |
| T15 | 13.178 | 11.529 | 1.649 |
| T10 | 18.740 | 15.742 | 2.998 |
| T16 | 16.890 | 16.138 | 752 |
| T09 | 27.855 | 22.636 | 5.219 |
| T07 | 31.717 | 25.068 | 6.649 |

## Dimensioni e significato

- JSONL raw: 69.231.427.390 byte.
- JSONL semantico prima di zstd: 23.005.831.546 byte,
  riduzione del 66,77%. Questa è la misura della pulizia
  di alias, basi statiche e ripetizioni delle macro.
- Archivi zstd: 872.559.281 byte, riduzione complessiva
  del 98,74% rispetto ai raw. Questa misura include
  anche la compressione del contenitore.

La geometria del corso è nella base degli attributi globali dei grafi
nei metadati degli stream che la registrano; i campioni contengono le
variazioni. Nessuna riga umana è stata raggruppata: ripetizioni di
azione e stato possono descrivere pause reali. Solo transizioni macro
consecutive semanticamente identiche condividono un record compatto;
i valori temporali sono conservati per ricostruire ogni campione.
Non sono stati applicati ricampionamento o interpolazione.

## Verifiche

- Il compattatore ha confrontato ogni record codificato e ogni
  ripetizione ricostruita con il raw, verificando SHA-256, byte e
  conteggi di ciascuno dei 60 stream sorgente.
- Ogni archivio è stato decodificato dal compattatore per verificare
  l'hash del JSONL semantico scritto. L'audit finale
  `audit_expert_v1_20260928.py --mode compact` ha verificato SHA-256
  e dimensioni di tutti i 60 archivi e i totali del catalogo.
- La suite mirata dei dataset tool ha superato 19 test.
- Il manifest finale conferma 59.635 campioni umani conservati uno
  per uno, 31.686 righe macro raggruppate e zero directory di staging.

L'ostacolo corrente non è ancora una label esplicita affidabile per
ogni riga: l'adapter successivo potrà derivarlo dalla geometria con
evidenza e confidenza, senza modificare i raw o questi archivi.


## Pulizia dei log sorgente

Dopo la verifica dei 176 raw canonici, il 29 settembre sono state eliminate le copie nei log e le prove fallite documentate. I dataset raw e compatti restano integri. Dettagli e conteggi sono in [pulizia dei log](teleop_log_cleanup_20260929.md). Le 41 macro strutturali isolate riuscite rimangono in `logs/teleop/structural/`, fuori dal manifest dei 13 episodi.
