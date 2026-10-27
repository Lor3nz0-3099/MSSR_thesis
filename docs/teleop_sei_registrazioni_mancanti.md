# Sei registrazioni composite da fare

**Stato verificato il 27 settembre 2026.** Nel dataset curato ci sono quattro
missioni complete: **T02, T03, T05, T06**. C05 è un prefisso riuscito di
**gap 4106 → RC 5100 → pulsante 6101**, interrotto prima della riconfigurazione
finale in RC: **non ha scale** e non conta come missione completa. I prefissi
che contengono scale sono T01 e T04. L'ultima T07 registrata non è completa.
L'obiettivo di circa dieci missioni complete richiede quindi **sei nuovi
successi**, non semplicemente sei avvii.

## Preparazione una volta sola

Da una shell nuova, con il DualSense collegato:

```bash
cd ~/MSSR_thesis
source /opt/ros/humble/setup.bash
cd mssr_ws
colcon build --packages-select mssr_expert --symlink-install
cd ..
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
MPLCONFIGDIR=/tmp/mssr-teleop-mpl \
python3 scripts/smores_ep/preview_teleop_composite_campaign.py \
  --output-dir logs/teleop/course_previews/t01-t16-v2
```

Controlla le sei missioni senza aprire Isaac:

```bash
cd ~/MSSR_thesis
for episode in teleop-t14 teleop-t15 teleop-t10 teleop-t16 teleop-t09 teleop-t07; do
  bash scripts/teleop/run_six_composite_demos.sh "$episode" --check
done
```

Il controllo confronta ID, ordine dei seed e hash geometrico con l'audit.
Il launcher copia missione e geometria nei log della nuova sessione e non
avvia alcun esecutore autonomo del corso: guida e allineamento sono umani.

## Ordine consigliato

**Esegui una sola riga per volta.** Ogni comando resta aperto finché termini
la sessione con Ctrl+C. Dopo una prova, ferma prima la registrazione con START,
poi chiudi il launcher; soltanto allora passa alla riga successiva.

| Priorità | Stage e comando | Sequenza e motivo |
|---|---|---|
| 1 | `cd ~/MSSR_thesis && bash scripts/teleop/run_six_composite_demos.sh teleop-t14` | **Pulsante 6103 → RC 6017 → scale 3105.** Primo pulsante iniziale e prima missione completa che può finire dopo le scale. |
| 2 | `cd ~/MSSR_thesis && bash scripts/teleop/run_six_composite_demos.sh teleop-t15` | **Pulsante 6101 → RC 5104 → gap 4105.** Verifica il gap finale, assente dalle prove complete. |
| 3 | `cd ~/MSSR_thesis && bash scripts/teleop/run_six_composite_demos.sh teleop-t10` | **RC 6017 → scale 3105 (−Y) → RC 5100.** Ottiene un altro ordine con scale e senza riconfigurazione del pulsante. |
| 4 | `cd ~/MSSR_thesis && bash scripts/teleop/run_six_composite_demos.sh teleop-t16` | **RC 5100 → scale 3101 (+Y) → gap 4103 (+Y).** Secondo gap finale, con traiettoria ruotata. |
| 5 | `cd ~/MSSR_thesis && bash scripts/teleop/run_six_composite_demos.sh teleop-t09` | **Gap 4107 → scale 3101 (+Y) → pulsante 9974 (+Y).** Copre due ostacoli consecutivi in Snake8, poi pulsante. |
| 6 | `cd ~/MSSR_thesis && bash scripts/teleop/run_six_composite_demos.sh teleop-t07` | **Scale 6403 → RC 5103 → pulsante 8936 (−Y).** Completa l'ordine già visto nel prefisso T01; priorità minore perché la precedente T07 non ha premuto il pulsante e ha fallito la riconfigurazione finale MM8→RC. |

Le missioni T14–T16 e gli altri tre stage passano l'audit geometrico statico;
questo non dimostra che la teleoperazione fisica in Isaac riuscirà. Se T15
non riesce al gap finale, passa a T10 e verifica il problema prima di consumare
altri tentativi sullo stesso stage. Se T07 fallisce ancora MM8→RC, conserva
solo il prefisso certificabile e analizza quella macro prima di ritentare.

## Comandi DualSense durante ogni prova

Premi **START una volta prima dell'assembly** e lascialo attivo per guida,
macro, riconfigurazioni e goal. Premi **START di nuovo soltanto alla fine**.
Per l'assembly iniziale: **X + D-pad SINISTRA** per RC-Car8 (T14, T15, T10,
T16), **X + D-pad SU** per Snake8 (T09, T07). Poi D-pad **SINISTRA = RC**,
**SU = Snake**, **DESTRA = MobileManipulator**. In Snake8, **SQUARE** avvia le
scale e **X premuto/rilasciato da solo** avvia il gap; allinea manualmente
l'intero serpente prima della macro. In T14/T15 il pulsante è subito davanti:
passa da RC a manipolatore con D-pad DESTRA, premi fisicamente il pulsante e
torna in RC con D-pad SINISTRA prima del percorso successivo.

In T16, dopo le scale resta in Snake8 per il gap a +Y. In T09, dopo gap e scale
passa a RC e poi a manipolatore per il pulsante. **TRIANGLE** alterna E-stop e
ripresa; **CIRCLE** richiede HOME. Completa la riconfigurazione finale
richiesta dal corso e raggiungi il goal prima di spegnere START.

## Controllare la registrazione

In un secondo terminale:

```bash
cd ~/MSSR_thesis
source /opt/ros/humble/setup.bash
source mssr_ws/install/setup.bash
export ROS_DOMAIN_ID=0
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
ros2 topic echo /mssr/teleop/status
```

Dopo START verifica `recording: true`, `recording_backend_ready: true` e annota
`recording_episode_id`. Dopo il secondo START verifica `recording: false`.
Il manifest si trova in `logs/teleop/recordings/<recording_episode_id>/manifest.json`;
i log tecnici e la missione effettivamente usata sono in
`logs/teleop/runs/teleop-tXX-.../`. Una registrazione chiusa correttamente non
certifica da sola il successo: controlla macro terminali, ostacoli, pulsante
quando presente, goal e morfologia finale prima di contarla tra le dieci.
