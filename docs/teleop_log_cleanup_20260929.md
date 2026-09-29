# Pulizia dei log teleop e delle prove singole — 29 settembre 2026

Sono state eliminate 187 directory di sorgenti già copiati nel dataset e tentativi falliti o incompleti, liberando 119.645.921.280 byte (119,6 GB; 111,4 GiB) di spazio effettivo. La dimensione apparente dei percorsi rimossi era maggiore perché alcuni file erano hard link dei raw curati.

Prima della cancellazione sono stati verificati gli SHA-256 dei 176 raw canonici e dei sorgenti ancora presenti. T01 è un caso particolare: il file umano curato coincide byte per byte con un suffisso allineato a una riga JSONL del sorgente, corrispondente all'intervallo conservato. Dopo la pulizia risultano presenti tutti i 56 episodi singoli e i 13 teleop, sia raw sia compatti; nessuno dei 187 percorsi pianificati esiste ancora.

Le directory eliminate comprendono 56 sorgenti canonici delle prove singole, 34 fallimenti classificati dall'audit di importazione, 42 tentativi recenti incompleti del pulsante, un tentativo in sospeso, 7 sorgenti teleop canonici, 7 registrazioni teleop scartate o superate, 33 controlli teleop con `passed=false` e 7 run teleop falliti documentati. L'elenco dettagliato è nel file locale `logs/cleanup_20260929.json`, escluso da Git insieme agli altri log.

Le 41 tracce in `logs/teleop/structural/` non compaiono nel manifest dei 13 episodi: sono macro deterministiche isolate che terminano con successo. Sono state conservate per una successiva valutazione come dati di riconfigurazione, assemblaggio o locomozione autonomi.
